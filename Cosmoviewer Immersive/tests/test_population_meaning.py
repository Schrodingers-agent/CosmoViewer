"""
What a selection means, wherever it is made, shown, saved, sent or exported.

Every way of filling a selection set records what the set is (selection_sets.py:
kind, definition, the snapshot it was defined at, whether it is complete and,
if not, why), so a set taken from the display sample is never mistaken for a
complete one, a region for a halo, or a set defined at one snapshot for one
evaluated at another:

- the selection criteria (radius, property, FoF group, catalogue group, typed
  IDs; alone or together), Load IDs, picking and "List top";
- editing a set by hand (add, subset) gives plain IDs that say what they came from;
- the selection panel shows it, Send to Explorer and Save IDs write it, sessions
  (version 2) keep it with each snapshot's FoF settings, and trajectory exports
  carry it in their metadata.

Synthetic snapshots have distinct Times (the loader sorts by time). Catalogue
groups use _dev_tools/inspector_check/dm_cut, the first 400,000 particles of
DM-L50-N128 snapshot_002 with its full fof_tab_002: groups 1 to 29 lie inside
the cut, group 30 only partly. The real-data case runs when
COSMOVIEWER_G4_OUTPUT points at DM-L50-N128/output (read only).
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import selection_sets  # noqa: E402

BOX, N, BUDGET = 20.0, 6000, 1000          # a display sample of about 1,000 of 6,000 particles
FOF = {'b_linking': 0.2, 'min_size': 20, 'r_link': None, 'link_types': [1]}
DM_CUT = ROOT.parent / "_dev_tools" / "inspector_check" / "dm_cut"
G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
ALL_TYPES = [0, 1, 2, 3, 4, 5]
_FINISHED = []                              # closed controllers, kept until the session ends


def _write(folder, name="snapshot_000.hdf5", time=1.0, n=N, seed=3, cosmological=True):
    """Eight clumps of dark matter in a periodic box, with a Potential per particle. The same
    particles (and IDs) in every file; returns the clump centres."""
    rng = np.random.default_rng(seed)
    centres = rng.random((8, 3)) * BOX
    pos = (centres[rng.integers(0, 8, n)] + rng.normal(0, 0.3, (n, 3))) % BOX
    with h5py.File(Path(folder) / name, "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": time, "Redshift": 1.0 / time - 1.0 if cosmological else 0.0,
                        "NumFilesPerSnapshot": 1, "MassTable": np.array([0.0, 1.0, 0, 0, 0, 0]),
                        "NumPart_ThisFile": np.array([0, n, 0, 0, 0, 0]),
                        "NumPart_Total": np.array([0, n, 0, 0, 0, 0])})
        p = f.create_group("Parameters")
        p.attrs.update({"UnitLength_in_cm": 3.085678e24 if cosmological else 3.085678e21,
                        "UnitMass_in_g": 1.989e43, "UnitVelocity_in_cm_per_s": 1e5, "HubbleParam": 0.7,
                        "Omega0": 0.3, "ComovingIntegrationOn": 1 if cosmological else 0})
        g = f.create_group("PartType1")
        g["Coordinates"] = pos.astype(np.float32)
        g["Velocities"] = np.zeros((n, 3), np.float32)
        g["ParticleIDs"] = rng.permutation(n).astype(np.uint64) + 1
        g["Potential"] = rng.normal(0.0, 1.0, n).astype(np.float32)
        g["Potential"].attrs.update({"a_scaling": -1.0, "h_scaling": 0.0})
    return centres


def _read(path, field):
    with h5py.File(path, "r") as f:
        return f[f"PartType1/{field}"][:], f["PartType1/ParticleIDs"][:]


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _controller(monkeypatch, tmp_path):
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c.max_particles = BUDGET
    return c


@pytest.fixture
def ctl(app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    warnings = []
    for name in ("information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: (warnings.append(str(a[2]) if len(a) > 2 else ""),
                                         QMessageBox.StandardButton.Ok)[1])
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    import lod.budget
    import lod.store
    monkeypatch.setattr(lod.store, "CACHE_DIR", tmp_path / "lod_cache")
    monkeypatch.setattr(lod.budget, "MIN_BASE_SAMPLE", 500)     # so a 6,000-particle run is sampled
    monkeypatch.setattr(selection_sets, "FALLBACK_ROOT", tmp_path / "home" / "sets")
    monkeypatch.setattr(selection_sets, "INSTANCES_DIR", tmp_path / "home" / "explorer_instances")
    c = _controller(monkeypatch, tmp_path)
    c.warnings = warnings
    yield c
    _close(c)


def _close(c):
    job = getattr(c, '_top_job', None)
    if job is not None:
        job.wait()
    c._shutdown_fof()
    c._reset_tracking()
    c._shutdown_lod()
    c.view.close()
    _FINISHED.append(c)


def _wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _load(app, c, folder):
    c._load_directory(Path(folder))
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 300)
    app.processEvents()


def _show(app, c, index):
    c.view.slide_time.setValue(index)            # as the user does, so the frame updates
    app.processEvents()
    assert c.current_snap_idx == index


def _run_fof(app, c):
    snap = c.model.get_snapshot(c.current_snap_idx)
    c._open_fof_dialog()
    c._run_fof_algorithm(dict(FOF))
    assert _wait(app, lambda: 'FoFGroupIndex' in snap and not c._fof_worker.isRunning())
    return snap


def _active(c):
    return c.model.tracker.get_set(c.view.selection_panel.active_set_idx)


def _apply(c, radius=None, full=True, fof=None, prop=None, cat=None, ids=None, mode='new'):
    """Tick exactly these criteria, as a user would, and apply them to the active set."""
    sp = c.view.selection_panel
    for box in (sp.chk_radius, sp.chk_fof, sp.chk_prop, sp.chk_cat, sp.chk_ids):
        box.setChecked(False)
    sp.chk_full.setChecked(full)
    {'new': sp.radio_new, 'add': sp.radio_add, 'subset': sp.radio_subset}[mode].setChecked(True)
    if radius is not None:
        sp.chk_radius.setChecked(True)
        sp.spin_radius.setValue(radius)
    if fof is not None:
        sp.chk_fof.setChecked(True)
        sp.spin_fof_id.setValue(fof)
    if prop is not None:
        name, lo, hi = prop
        sp.chk_prop.setChecked(True)
        sp.combo_prop.setCurrentText(name)
        assert sp.combo_prop.currentText() == name
        sp.edit_prop_min.setText("" if lo is None else repr(lo))
        sp.edit_prop_max.setText("" if hi is None else repr(hi))
    if cat is not None:
        group, which = cat
        sp.chk_cat.setChecked(True)
        sp.combo_cat.setCurrentText(which)
        sp.spin_cat.setValue(group)
    if ids is not None:
        sp.chk_ids.setChecked(True)
        sp.edit_ids.setText(ids)
    c._apply_selection_criteria()
    return _active(c)


def _in_sphere(path, centre, radius):
    pos, pid = _read(path, "Coordinates")
    d = pos.astype(np.float64) - centre
    d -= BOX * np.round(d / BOX)
    return set(int(i) for i in pid[np.einsum('ij,ij->i', d, d) <= radius * radius])


# ----------------------------------------------------------------------------- selection_sets.py wording

def test_why_a_set_is_incomplete_is_said_in_its_own_words():
    group = {'finder': 'catalogue', 'group': 30, 'subhalo': False, 'catalogue_file': 'fof_tab_002.hdf5'}
    at = {'file': 'snapshot_002.hdf5', 'index': 0}
    ids = np.arange(37, dtype=np.uint64)
    missing = selection_sets.SelectionSet(ids, "g30", kind="group", definition=group, defined_at=at,
                                          complete=False, n_expected=52, incomplete_reason="missing_in_file")
    assert missing.describe() == ("catalogue group 30 of fof_tab_002.hdf5 at snapshot_002.hdf5 · "
                                  "members missing from the file: 37 of 52")
    assert missing.completeness_warning() == ("The snapshot files hold only 37 of this group's 52 members; "
                                              "results describe these particles only, not the whole group.")
    unknown = selection_sets.SelectionSet(ids, "g30", kind="group", definition=group, defined_at=at,
                                          complete=False, n_expected=52)
    assert unknown.describe().endswith("· incomplete: 37 of 52")
    assert unknown.completeness_warning() == ("This set is incomplete: it holds 37 of its 52 particles; "
                                              "results describe these particles only, not the whole group.")
    sample = selection_sets.SelectionSet(ids, "s", kind="region", definition={'radius': 1.0}, complete=False,
                                         incomplete_reason="display_sample")
    assert "display sample only" in sample.describe()
    assert sample.completeness_warning().startswith("This set was taken from the display sample")


def test_incomplete_reason_is_written_read_and_checked(tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    snap.write_bytes(b"")
    p = selection_sets.write_set(str(snap), [1, 2], "g", kind="group", definition={'group': 1}, complete=False,
                                 n_expected=5, incomplete_reason="missing_in_file")
    assert selection_sets.read_set(p).incomplete_reason == "missing_in_file"
    with h5py.File(p, "a") as f:                     # a version-2 file from before the reason existed
        del f.attrs["incomplete_reason"]
    assert selection_sets.read_set(p).incomplete_reason is None
    with pytest.raises(ValueError):                  # not a known reason
        selection_sets.write_set(str(snap), [1], "x", complete=False, incomplete_reason="lost")
    with pytest.raises(ValueError):                  # a complete set has no reason to be incomplete
        selection_sets.write_set(str(snap), [1], "x", complete=True, incomplete_reason="display_sample")
    from model.simulation_model import ParticleSelectionSet
    s = ParticleSelectionSet(0)
    s.set_population([1, 2], 'group', {'group': 1}, complete=False, n_expected=5,
                     incomplete_reason="missing_in_file")
    assert s.as_selection_set().incomplete_reason == "missing_in_file"
    with pytest.raises(ValueError):
        s.set_population([3], 'region', {}, complete=None, incomplete_reason="display_sample")
    assert s.selected_ids == {1, 2}                  # a refused population leaves the set as it was
    s.add_ids([3])                                   # a hand edit makes plain IDs, but an incomplete
    assert (s.kind, s.complete, s.incomplete_reason) == ('ids', False, 'missing_in_file')   # set stays so


def test_open_bounds_top_criteria_and_one_particle_read_plainly():
    q = {'criteria': [{'kind': 'property', 'field': 'Density', 'lo': 1.0, 'hi': None, 'units': 'K'},
                      {'kind': 'property', 'field': 'Masses', 'lo': None, 'hi': float('inf'), 'units': ''},
                      {'kind': 'property', 'field': 'Potential', 'lo': -2.0, 'hi': 3.0,
                       'units': 'code units, × a^-1 for physical'}]}
    text = selection_sets.describe('query', q, {}, True, 3)
    assert text == ("Density >= 1 K and any value of Masses and -2 <= Potential <= 3 "
                    "[code units, × a^-1 for physical] · complete · 3 particles")
    top = {'criteria': [{'kind': 'top', 'field': 'Density', 'n': 200, 'rank': 2, 'scope': 'full'}]}
    assert selection_sets.describe('query', top, {'file': 's.hdf5'}, True, 1, 1) == \
        "2nd highest Density (from a list of 200) at s.hdf5 · complete · 1 particle"
    drawn = {'criteria': [{'kind': 'top', 'field': 'T', 'n': 5, 'rank': 1, 'scope': 'sample'}]}
    assert selection_sets.describe('query', drawn, {}, False, 1, None, 'display_sample') == \
        "highest T of the drawn particles (from a list of 5) · display sample only · 1 particle"


def test_code_length_labels():
    from controller.tracking_controller import code_length_label
    cosmo = {'header': {'Time': 0.5, 'Redshift': 1.0},
             'unit_source': {'UnitLength_in_cm': 3.085678e24, 'ComovingIntegrationOn': 1}}
    assert code_length_label(cosmo) == "Mpc/h (comoving)"
    assert code_length_label({'header': {'Time': 0.5, 'Redshift': 1.0},           # EAGLE: no such parameter
                              'unit_source': {'UnitLength_in_cm': 3.085678e24}}) == "Mpc/h (comoving)"
    assert code_length_label({'header': {'Time': 2.0},
                              'unit_source': {'UnitLength_in_cm': 3.085678e21, 'ComovingIntegrationOn': 0}}) == "kpc"
    assert code_length_label({'header': {}, 'unit_source': {'UnitLength_in_cm': 1.5428e21,
                                                            'ComovingIntegrationOn': 0}}) == "× 0.5 kpc"
    assert code_length_label({'header': {'Time': 1.0}}) == "code units"      # never guessed


# ----------------------------------------------------------------------------- criteria -> definitions

def test_radius_is_a_region_complete_only_when_every_particle_was_searched(app, ctl, tmp_path):
    centres = _write(tmp_path)
    _load(app, ctl, tmp_path)
    assert len(ctl.model.get_snapshot(0)['c']) < N           # premise: the display sample is partial
    ctl.view.set_camera_centre(centres[0])
    centre = ctl._view_centre()
    s = _apply(ctl, radius=1.0, full=True)
    want = _in_sphere(tmp_path / "snapshot_000.hdf5", centre, 1.0)
    assert s.selected_ids == want and len(want) > 100
    assert (s.kind, s.complete, s.incomplete_reason, s.n_expected) == ('region', True, None, len(want))
    d = dict(s.definition)
    np.testing.assert_allclose(d.pop('centre'), centre)
    from model.tracking import sphere_wraps
    assert d == {'shape': 'sphere', 'radius': 1.0, 'units': "Mpc/h (comoving)", 'periodic': True,
                 'wraps': sphere_wraps(centre, 1.0, BOX), 'types': ALL_TYPES, 'scope': 'full'}
    assert s.defined_at == {'file': 'snapshot_000.hdf5', 'index': 0, 'time': 1.0, 'redshift': 0.0}
    assert s.describe().startswith("sphere r = 1 Mpc/h (comoving) about (")
    assert s.describe().endswith(f"at snapshot_000.hdf5 · complete · {len(want):,} particles")
    assert s.completeness_warning() is None

    # the same sphere searched in the display sample only: incomplete, and it says why
    s = _apply(ctl, radius=1.0, full=False)
    assert 0 < len(s.selected_ids) < len(want) and s.selected_ids <= want
    assert (s.kind, s.complete, s.incomplete_reason, s.n_expected) == ('region', False, 'display_sample', None)
    assert s.definition['scope'] == 'sample' and "display sample only" in s.describe()
    assert s.completeness_warning().startswith("This set was taken from the display sample")


def test_a_sample_that_holds_every_particle_searches_the_whole_snapshot(app, ctl, tmp_path):
    centres = _write(tmp_path, n=800)
    _load(app, ctl, tmp_path)
    assert len(ctl.model.get_snapshot(0)['c']) == 800
    ctl.view.set_camera_centre(centres[0])
    s = _apply(ctl, radius=1.0, full=False)
    assert s.selected_ids == _in_sphere(tmp_path / "snapshot_000.hdf5", ctl._view_centre(), 1.0)
    assert (s.complete, s.definition['scope']) == (True, 'full')


def test_a_property_range_is_a_query_with_its_units(app, ctl, tmp_path):
    _write(tmp_path)
    _load(app, ctl, tmp_path)
    info = ctl._file_properties(ctl.model.get_snapshot(0))['Potential']
    s = _apply(ctl, prop=('Potential', 0.5, None), full=True)
    pot, pid = _read(tmp_path / "snapshot_000.hdf5", "Potential")
    want = set(int(i) for i in pid[pot >= 0.5])
    assert s.selected_ids == want
    assert (s.kind, s.complete, s.n_expected) == ('query', True, len(want))
    assert s.definition == {'criteria': [{'kind': 'property', 'field': 'Potential', 'lo': 0.5, 'hi': None,
                                          'units': info.unit_label, 'types': ALL_TYPES, 'scope': 'full'}]}
    assert s.describe().startswith(f"Potential >= 0.5 [{info.unit_label}] at snapshot_000.hdf5 · complete")

    s = _apply(ctl, prop=('Potential', 0.5, None), full=False)
    assert s.selected_ids < want
    assert (s.complete, s.incomplete_reason) == (False, 'display_sample')
    assert s.definition['criteria'][0]['scope'] == 'sample'


def test_fof_groups_alone_with_a_sphere_and_as_a_sample_property(app, ctl, tmp_path):
    _write(tmp_path)
    _load(app, ctl, tmp_path)
    snap = _run_fof(app, ctl)
    s = _apply(ctl, fof=1)
    size = snap['fof_sizes'][1]
    assert len(s.selected_ids) == size
    assert (s.kind, s.complete, s.n_expected, s.definition) == ('group', True, size, ctl._fof_definition(snap, 1))
    assert s.describe().startswith("FoF group 1 (b = 0.2, N_min = 20, DM) at snapshot_000.hdf5 · complete")

    # with a sphere about one of its members: a query of both, complete only if the sphere was
    pos, pid = _read(snap['filepath'], "Coordinates")
    ctl.view.set_camera_centre(pos[np.isin(pid, list(s.selected_ids))][0].astype(np.float64))
    s = _apply(ctl, fof=1, radius=0.3, full=True)
    assert s.kind == 'query' and [c['kind'] for c in s.definition['criteria']] == ['region', 'group']
    assert s.definition['criteria'][1]['finder'] == 'fof' and len(s.selected_ids) > 0
    assert (s.complete, s.n_expected) == (True, len(s.selected_ids))
    s = _apply(ctl, fof=1, radius=0.3, full=False)
    assert (s.kind, s.complete, s.incomplete_reason) == ('query', False, 'display_sample')

    # the FoF groups attached to the display sample are a property of the sample only:
    # searched there even with "Search full snapshot" ticked, and the set says so
    s = _apply(ctl, prop=('FoF group (computed here)', 1, 1), full=True)
    assert 0 < len(s.selected_ids) < size
    assert (s.complete, s.incomplete_reason) == (False, 'display_sample')
    assert s.definition['criteria'][0]['scope'] == 'sample'


def test_typed_ids_and_editing_by_hand(app, ctl, tmp_path):
    centres = _write(tmp_path)
    _load(app, ctl, tmp_path)
    s = _apply(ctl, ids="1-10")
    assert (s.kind, s.definition, s.complete) == ('ids', {'source': 'typed'}, True)
    assert s.describe() == "particle IDs · complete · 10 particles"

    ctl.view.set_camera_centre(centres[0])
    region = _apply(ctl, radius=1.0, full=True).describe()
    s = _apply(ctl, ids="1-3", mode='add')
    assert (s.kind, s.definition, s.complete) == ('ids', {'edited_from': region, 'operation': 'add'}, None)
    s = _apply(ctl, ids="1-2", mode='subset')
    assert s.definition['operation'] == 'subset' and s.selected_ids == {1, 2}
    s.clear()
    s = _apply(ctl, radius=1.0, full=True, mode='add')       # adding to an empty set is a new set
    assert (s.kind, s.complete) == ('region', True)


@pytest.fixture(scope="module")
def dm_cut(tmp_path_factory):
    if not (DM_CUT / "snapshot_002.hdf5").exists():
        pytest.fail("tracked test fixture _dev_tools/inspector_check/dm_cut is missing: restore it with git checkout -- _dev_tools/inspector_check/dm_cut")
    d = tmp_path_factory.mktemp("dm_cut")
    for name in ("snapshot_002.hdf5", "fof_tab_002.hdf5"):
        (d / name).symlink_to(DM_CUT / name)          # read only: nothing here sends or saves a set
    return d


def test_catalogue_groups_whole_and_with_members_missing_from_the_file(app, ctl, dm_cut):
    with h5py.File(DM_CUT / "fof_tab_002.hdf5", "r") as f:
        lengths = f["Group/GroupLen"][:].astype(int)
        start = f["Group/GroupOffsetType"][:, 1].astype(int)
    with h5py.File(DM_CUT / "snapshot_002.hdf5", "r") as f:
        rows = f["PartType1/ParticleIDs"].shape[0]
    ctl.max_particles = 100_000
    _load(app, ctl, dm_cut)
    s = _apply(ctl, cat=(1, 'group'))
    assert len(s.selected_ids) == lengths[0] == 34_120
    assert (s.kind, s.complete, s.n_expected, s.incomplete_reason) == ('group', True, 34_120, None)
    assert s.definition == {'finder': 'catalogue', 'group': 1, 'subhalo': False, 'catalogue_file': 'fof_tab_002.hdf5'}
    assert s.describe() == "catalogue group 1 of fof_tab_002.hdf5 at snapshot_002.hdf5 · complete · 34,120 particles"

    # group 30 runs past the end of the cut file: the set holds what is there and says so
    inside = min(start[29] + lengths[29], rows) - start[29]
    assert 0 < inside < lengths[29]
    s = _apply(ctl, cat=(30, 'group'))
    assert len(s.selected_ids) == inside
    assert (s.complete, s.n_expected, s.incomplete_reason) == (False, lengths[29], 'missing_in_file')
    assert f"members missing from the file: {inside:,} of {lengths[29]:,}" in s.describe()
    assert s.completeness_warning() == (f"The snapshot files hold only {inside:,} of this group's "
                                        f"{lengths[29]:,} members; results describe these particles only, "
                                        f"not the whole group.")


# ----------------------------------------------------------------------------- files, picks, List top

def test_load_ids_keeps_a_set_files_meaning(app, ctl, tmp_path):
    _write(tmp_path)
    _load(app, ctl, tmp_path)
    fof = {'finder': 'fof', 'group': 3, 'b': 0.2, 'min_size': 20, 'link_types': [1]}
    at = {'file': 'snapshot_007.hdf5', 'index': 7, 'time': 0.5, 'redshift': 1.0}
    p = selection_sets.write_set(str(tmp_path / "snapshot_000.hdf5"), [5, 6, 7], "halo 3", kind="group",
                                 definition=fof, defined_at=at, complete=False, n_expected=40,
                                 incomplete_reason="display_sample", path=str(tmp_path / "halo3.hdf5"))
    assert ctl.load_ids_file(p)
    s = _active(ctl)
    assert s.selected_ids == {5, 6, 7}
    assert s.meaning() == {'kind': 'group', 'definition': fof, 'defined_at': at, 'complete': False,
                           'n_expected': 40, 'incomplete_reason': 'display_sample'}

    v1 = tmp_path / "old.hdf5"
    with h5py.File(v1, "w") as f:                       # exactly what version 1 wrote
        f["ParticleIDs"] = np.array([3, 1, 2], dtype=np.uint64)
        f.attrs.update(dict(format=selection_sets.FORMAT, version=1, name="old", colour="#ff0000",
                            made_in="CosmoViewer Immersive", source_snapshot="snapshot_000.hdf5", n=3))
    assert ctl.load_ids_file(str(v1))
    assert (s.kind, s.definition, s.complete, s.selected_ids) == ('ids', {}, None, {1, 2, 3})

    txt = tmp_path / "ids.txt"
    txt.write_text("# made somewhere else\n4\n5\n")
    assert ctl.load_ids_file(str(txt))
    assert (s.kind, s.definition, s.complete) == ('ids', {'source': 'file', 'file': str(txt)}, None)
    assert s.describe() == "particle IDs from ids.txt · 2 particles"


def test_save_ids_as_a_set_file_keeps_the_meaning(app, ctl, tmp_path):
    centres = _write(tmp_path)
    _load(app, ctl, tmp_path)
    ctl.view.set_camera_centre(centres[1])
    s = _apply(ctl, radius=1.0, full=False)
    meaning, ids = s.meaning(), set(s.selected_ids)
    st = selection_sets.read_set(ctl.save_ids_file(str(tmp_path / "sphere.hdf5"), s))
    assert (st.kind, st.definition, st.complete, st.incomplete_reason) == ('region', s.definition, False,
                                                                           'display_sample')
    txt = ctl.save_ids_file(str(tmp_path / "sphere.txt"), s)
    assert Path(txt).read_text().splitlines()[0] == f"# {s.name}: {s.describe()}; saved at snapshot_000.hdf5"
    s.clear()
    assert ctl.load_ids_file(str(tmp_path / "sphere.hdf5"))
    assert s.meaning() == meaning and s.selected_ids == ids


def test_picks_are_ids_and_list_top_belongs_to_its_snapshot(app, ctl, tmp_path):
    _write(tmp_path, "snapshot_000.hdf5", time=0.5)
    _write(tmp_path, "snapshot_001.hdf5", time=1.0)
    _load(app, ctl, tmp_path)
    c, cp, sp = ctl, ctl.view.colouring_panel, ctl.view.selection_panel
    c.view.resize(1200, 800)
    assert cp.select_property('Potential')
    c._update_frame()
    pos = c._pick_candidates[0]
    x, y, depth, front = c.view.screen_coords(pos)
    ok = np.flatnonzero(front & (x > 5) & (x < c.view.view_3d.width() - 5)
                        & (y > 5) & (y < c.view.view_3d.height() - 5))
    k = int(ok[np.argmin(depth[ok])])
    sp.chk_pick.setChecked(True)
    c.view.pick_requested.emit(float(x[k]), float(y[k]), False)
    s = _active(c)
    assert len(s.selected_ids) == 1
    assert (s.kind, s.definition, s.complete) == ('ids', {'source': 'pick'}, True)
    assert s.defined_at['file'] == 'snapshot_000.hdf5' and s.describe() == "picked particles · complete · 1 particle"

    # the list is made for snapshot 0; another snapshot is shown before a row is chosen
    c._list_top()
    assert _wait(app, lambda: sp.combo_top.count() > 0, 60)
    _show(app, c, 1)
    sp.radio_new.setChecked(True)
    c._pick_from_list(2)
    pot, pid = _read(tmp_path / "snapshot_000.hdf5", "Potential")
    assert s.selected_ids == {int(pid[np.argsort(-pot, kind='stable')[2]])}
    assert (s.kind, s.complete, s.n_expected, s.incomplete_reason) == ('query', True, 1, None)
    (crit,) = s.definition['criteria']
    assert {k: crit[k] for k in ('kind', 'field', 'n', 'rank', 'scope')} == \
        {'kind': 'top', 'field': 'Potential', 'n': sp.combo_top.count(), 'rank': 3, 'scope': 'full'}
    assert (s.defined_at['file'], s.defined_at['index']) == ('snapshot_000.hdf5', 0)
    assert s.describe().startswith("3rd highest Potential (from a list of 200) at snapshot_000.hdf5 · complete")


# ----------------------------------------------------------------------------- shown, sent, kept, exported

def test_the_panel_shows_what_the_active_set_is(app, ctl, tmp_path):
    from views.selection_panel import ParticleSelectionPanel as Panel
    centres = _write(tmp_path)
    _load(app, ctl, tmp_path)
    lbl = ctl.view.selection_panel.lbl_meaning
    assert lbl.full_text() == "Empty set"
    ctl.view.set_camera_centre(centres[0])
    s = _apply(ctl, radius=1.0, full=True)
    assert lbl.full_text() == s.describe() and lbl.toolTip() == s.describe()
    assert Panel.WARNING_COLOUR not in lbl.styleSheet()
    s = _apply(ctl, radius=1.0, full=False)
    assert lbl.full_text() == "⚠ " + s.describe()
    assert s.completeness_warning() in lbl.toolTip() and Panel.WARNING_COLOUR in lbl.styleSheet()
    lbl.resize(90, lbl.height())                     # one line: the end gives way, the tooltip keeps all
    lbl.set_full_text(lbl.full_text(), lbl.toolTip())   # (a hidden window gets its resize event when shown)
    assert lbl.text().endswith("…") and len(lbl.text()) < len(lbl.full_text())
    assert s.completeness_warning() in lbl.toolTip()


def test_send_to_explorer_writes_the_meaning(app, ctl, tmp_path, monkeypatch):
    centres = _write(tmp_path)
    _load(app, ctl, tmp_path)
    ctl.view.set_camera_centre(centres[2])
    s = _apply(ctl, radius=1.0, full=False)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: object())
    path = ctl._send_set_to_explorer()
    st = selection_sets.read_set(path)
    assert (st.kind, st.definition, st.defined_at, st.complete, st.n_expected, st.incomplete_reason) == \
        ('region', s.definition, s.defined_at, False, None, 'display_sample')
    assert st.description == s.describe() == st.describe()
    with h5py.File(path, "r") as f:
        assert int(f.attrs["version"]) == 2


def test_a_session_keeps_what_the_sets_mean_and_the_fof_settings(app, ctl, tmp_path, monkeypatch):
    _write(tmp_path, "snapshot_000.hdf5", time=0.5)
    centres = _write(tmp_path, "snapshot_001.hdf5", time=1.0)
    _load(app, ctl, tmp_path)
    _show(app, ctl, 1)
    snap = _run_fof(app, ctl)
    sp = ctl.view.selection_panel
    ctl._highlight_fof_halo(1)                       # set 1: FoF group 1 of snapshot_001
    group = ctl.model.tracker.get_set(0)
    assert group.kind == 'group'
    sp.combo_sets.setCurrentIndex(1)                 # set 2: a sphere searched in the display sample
    ctl.view.set_camera_centre(centres[3])
    sphere = _apply(ctl, radius=1.0, full=False)
    path = tmp_path / "s.json"
    assert ctl.save_session(str(path))
    st = json.loads(path.read_text())
    assert st['version'] == 2 and st['sets'][0]['kind'] == 'group'
    assert st['sets'][1]['incomplete_reason'] == 'display_sample'
    run = {'file': str(snap['filepath']), 'index': 1, 'settings': FOF,
           'linking_length': pytest.approx(snap['fof_linking_length']), 'n_groups': len(snap['fof_sizes'])}
    assert st['fof_runs'] == [run]

    c2 = _controller(monkeypatch, tmp_path)
    try:
        assert c2.open_session(str(path))
        assert _wait(app, lambda: not c2.loading and c2.model.snapshots and c2._session_pending is None, 300)
        for i, before in ((0, group), (1, sphere)):
            after = c2.model.tracker.get_set(i)
            assert after.selected_ids == before.selected_ids
            assert after.meaning() == before.meaning() and after.describe() == before.describe()
        assert 'fof' not in c2.model.get_snapshot(1)            # FoF is not run again on opening...
        assert c2.session_state()['fof_runs'] == [run]          # ...but a new save keeps its settings
    finally:
        _close(c2)


def test_an_old_session_still_opens(app, ctl, tmp_path):
    _write(tmp_path)
    old = {"format": "cosmoviewer-session", "version": 1, "files": [str(tmp_path / "snapshot_000.hdf5")],
           "snapshot_index": 0, "active_set": 0,
           "sets": [{"name": "old halo", "color": [255, 0, 0, 255], "enabled": True, "ids": [3, 1, 2]}]}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    assert ctl.open_session(str(path))
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots and ctl._session_pending is None, 300)
    s = ctl.model.tracker.get_set(0)
    assert s.name == "old halo" and s.selected_ids == {1, 2, 3}
    assert s.meaning() == {'kind': 'ids', 'definition': {}, 'defined_at': {}, 'complete': None,
                           'n_expected': None, 'incomplete_reason': None}
    assert s.describe() == "particle IDs · 3 particles"
    # a meaning this version cannot read: plain IDs, said in the console, never half restored
    bad = dict(old, version=2, sets=[dict(old['sets'][0], kind='halo', complete=True)])
    path.write_text(json.dumps(bad))
    assert ctl.open_session(str(path))
    assert _wait(app, lambda: ctl._session_pending is None, 60)
    assert (s.kind, s.complete, s.selected_ids) == ('ids', None, {1, 2, 3})


def test_trajectory_export_says_which_particles_these_are(app, ctl, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    _write(tmp_path, "snapshot_000.hdf5", time=0.5)
    centres = _write(tmp_path, "snapshot_001.hdf5", time=1.0)
    _load(app, ctl, tmp_path)
    ctl.view.set_camera_centre(centres[0])
    s = _apply(ctl, radius=0.5, full=False)
    for name in ("t.npz", "t.csv"):
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, n=name, **k: (str(tmp_path / n), ""))
        ctl._export_trajectories()
    pop = json.loads(str(np.load(tmp_path / "t.npz")["metadata"]))['population']
    assert pop == {'name': s.name, 'describe': s.describe(), 'warning': s.completeness_warning(),
                   'n': len(s.selected_ids), **s.meaning()}
    assert pop['kind'] == 'region' and pop['complete'] is False and pop['warning']
    first = (tmp_path / "t.csv").read_text().splitlines()[0]
    assert first.startswith("# ") and json.loads(first[2:])['population'] == pop


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_real_dm_l50_catalogue_group_is_complete(app, ctl, tmp_path):
    out = Path(G4_OUTPUT)
    for name in ("snapshot_002.hdf5", "fof_tab_002.hdf5"):
        (tmp_path / name).symlink_to(out / name)      # read only: nothing here sends or saves a set
    with h5py.File(out / "fof_tab_002.hdf5", "r") as f:
        n1 = int(f["Group/GroupLen"][0])
    ctl.max_particles = 200_000
    _load(app, ctl, tmp_path)
    s = _apply(ctl, cat=(1, 'group'))
    assert len(s.selected_ids) == n1 == s.n_expected
    assert (s.kind, s.complete, s.incomplete_reason) == ('group', True, None)
    assert s.describe() == f"catalogue group 1 of fof_tab_002.hdf5 at snapshot_002.hdf5 · complete · {n1:,} particles"
