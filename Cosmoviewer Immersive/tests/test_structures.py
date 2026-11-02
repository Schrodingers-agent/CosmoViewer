"""
Structures: the halo table, "select its group" in the inspector, and the
Structures menu.

A researcher selects a group to measure it, so a selected group must hold all
of its members (a population of kind "group", complete, with the group's
length as n_expected), whichever route led to it: a catalogue row, a FoF row,
or a particle's group in the inspector. A group whose members are not all in
the snapshot file must say so. The table must describe the snapshot on screen
and no other, label its units, and its "check against catalogue" must agree
with Gadget-4's own GroupLen, GroupMass and GroupPos to float32 precision.

Data: dm_cut (the first 400,000 particles of DM-L50-N128 snapshot_002 with the
full fof_tab_002 of 2,774 groups) in _dev_tools; synthetic snapshots and a
synthetic fof_subhalo_tab written here; and, when COSMOVIEWER_G4_OUTPUT points
at DM-L50-N128/output, the real snapshot_002 and fof_tab_002 (copied first:
the application may write next to the snapshots it opens).
"""

import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from controller.structures_controller import (GroupAccumulator, compare_with_catalogue,  # noqa: E402
                                              measure_catalogue_groups, stored_units)
from snapshot_io.group_catalogue import read_catalogue_fields, read_gadget4_catalogue  # noqa: E402

DM_CUT = ROOT.parent / "_dev_tools" / "inspector_check" / "dm_cut"
G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
BOX = 20.0
SETTINGS = {'b_linking': 0.2, 'min_size': 20, 'r_link': None, 'link_types': [1]}


# ----------------------------------------------------------------------------- data

def _params(f, comoving=1, length=3.085678e24, h=0.7):
    p = f.create_group("Parameters")
    p.attrs.update({"UnitLength_in_cm": length, "UnitMass_in_g": 1.989e43, "UnitVelocity_in_cm_per_s": 1e5,
                    "Omega0": 0.3, "OmegaLambda": 0.7, "HubbleParam": h, "ComovingIntegrationOn": comoving})


def _write_clustered(folder, name, n=60_000, seed=5, time=1.0):
    """Clumps of DM in a periodic box; the same particles in every file written with the same seed."""
    rng = np.random.default_rng(seed)
    centres = rng.random((30, 3)) * BOX
    c = (centres[rng.integers(0, 30, n)] + rng.normal(0, 0.4, (n, 3))) % BOX
    with h5py.File(Path(folder) / name, "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": time, "Redshift": 1 / time - 1, "NumFilesPerSnapshot": 1,
                        "MassTable": np.array([0.0, 1.0, 0, 0, 0, 0]),
                        "NumPart_ThisFile": np.array([0, n, 0, 0, 0, 0]),
                        "NumPart_Total": np.array([0, n, 0, 0, 0, 0])})
        _params(f)
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=c.astype(np.float32))
        g.create_dataset("Velocities", data=np.zeros((n, 3), np.float32))
        g.create_dataset("ParticleIDs", data=rng.permutation(n).astype(np.uint64) + 1)


def _independent_com(pos, centre_guess, box):
    """Periodic centre of mass by unwrapping about a point known to be near the group's centre."""
    d = pos.astype(np.float64) - centre_guess
    d -= box * np.round(d / box)
    return np.mod(centre_guess + d.mean(axis=0), box)


def _write_with_subhaloes(folder):
    """A snapshot written in group order (as Gadget-4 does) and its fof_subhalo_tab: 3 groups
    (group 1 straddles the box corner), 3 subhaloes, then particles in no group."""
    rng = np.random.default_rng(11)
    m = 0.25
    centres = np.array([[0.1, 19.9, 0.05], [7.0, 8.0, 9.0], [15.0, 3.0, 12.0]])
    sizes, sub = [300, 200, 100], [(0, 0, 200), (0, 200, 60), (1, 0, 150)]   # (group, first row in group, length)
    parts = [(centres[g] + rng.normal(0, 0.3, (sz, 3))) % BOX for g, sz in enumerate(sizes)]
    field = rng.random((400, 3)) * BOX
    pos = np.vstack(parts + [field]).astype(np.float32)
    n = len(pos)
    ids = rng.permutation(np.arange(1000, 1000 + n)).astype(np.uint64)
    with h5py.File(Path(folder) / "snapshot_000.hdf5", "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": 1.0, "Redshift": 0.0, "NumFilesPerSnapshot": 1,
                        "MassTable": np.array([0.0, m]),
                        "NumPart_ThisFile": np.array([0, n]), "NumPart_Total": np.array([0, n])})
        _params(f)
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=pos)
        g["Coordinates"].attrs.update(a_scaling=1.0, h_scaling=-1.0)
        g.create_dataset("Velocities", data=np.zeros((n, 3), np.float32))
        g.create_dataset("ParticleIDs", data=ids)
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    gcom = [_independent_com(pos[o:o + s], centres[k], BOX) for k, (o, s) in enumerate(zip(offs, sizes))]
    s_off = [offs[gg] + first for gg, first, _ in sub]
    s_len = [ln for _, _, ln in sub]
    scom = [_independent_com(pos[o:o + ln], centres[gg], BOX) for (gg, _, ln), o in zip(sub, s_off)]
    with h5py.File(Path(folder) / "fof_subhalo_tab_000.hdf5", "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": 1.0, "Redshift": 0.0, "Ngroups_Total": 3, "Nsubhalos_Total": 3,
                        "NumFiles": 1})
        _params(f)
        g = f.create_group("Group")
        g["GroupLen"] = np.array(sizes, np.int32)
        g["GroupLenType"] = np.array([[0, s] for s in sizes], np.int32)
        g["GroupOffsetType"] = np.array([[0, o] for o in offs], np.int64)
        g["GroupMass"] = np.array([m * s for s in sizes], np.float32)
        g["GroupPos"] = np.array(gcom, np.float32)
        s = f.create_group("Subhalo")
        s["SubhaloLen"] = np.array(s_len, np.int32)
        s["SubhaloLenType"] = np.array([[0, ln] for ln in s_len], np.int32)
        s["SubhaloOffsetType"] = np.array([[0, o] for o in s_off], np.int64)
        s["SubhaloMass"] = np.array([m * ln for ln in s_len], np.float32)
        s["SubhaloCM"] = np.array(scom, np.float32)
        s["SubhaloPos"] = np.array([pos[o] for o in s_off], np.float32)       # a member, as the potential minimum
        s["SubhaloGroupNr"] = np.array([gg for gg, _, _ in sub], np.int64)
        s["SubhaloRankInGr"] = np.array([0, 1, 0], np.int32)
    return pos, ids, offs, sizes, s_off, s_len, gcom, scom


# ----------------------------------------------------------------------------- the application

@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    import lod.store
    monkeypatch.setattr(lod.store, "CACHE_DIR", tmp_path / "lod_cache")
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c.max_particles = 10_000
    yield c
    c._shutdown_structures()
    c._shutdown_fof()
    c._shutdown_lod()
    c._cancel_record_job()
    for j in c._old_record_jobs:
        j.wait(5000)
    c._reset_tracking()
    c.view.close()


def _wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _load(app, ctl, folder):
    ctl._load_directory(Path(folder))
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots)
    app.processEvents()


def _show(app, ctl, index):
    ctl.view.slide_time.setValue(index)
    app.processEvents()
    assert ctl.current_snap_idx == index


def _open_table(app, ctl):
    ctl.view.act_halo_table.setChecked(True)
    app.processEvents()
    assert ctl._halo_table_open()
    return ctl.structures_panel


def _active(ctl):
    return ctl.model.tracker.get_set(ctl.view.selection_panel.active_set_idx)


def _cell(panel, row, key):
    m = panel.model
    col = next(k for k, c in enumerate(m.columns) if c.key == key)
    return m.data(m.index(row, col))


def _rows_ids(snapshot, cat_path, group, ptype=1):
    """ParticleIDs of a catalogue group read straight from the file by GroupOffsetType / GroupLenType."""
    with h5py.File(cat_path, "r") as f:
        off = int(f["Group/GroupOffsetType"][group - 1, ptype])
        ln = int(f["Group/GroupLenType"][group - 1, ptype])
    with h5py.File(snapshot, "r") as f:
        ids = f[f"PartType{ptype}/ParticleIDs"]
        return np.sort(ids[off:min(off + ln, ids.shape[0])].astype(np.uint64)), ln


def _dm_cut(tmp_path):
    if not DM_CUT.exists():
        pytest.fail("tracked test fixture _dev_tools/inspector_check/dm_cut is missing: restore it with git checkout -- _dev_tools/inspector_check/dm_cut")
    d = tmp_path / "dm_cut"
    shutil.copytree(DM_CUT, d)
    return d


def _run_fof(app, ctl, settings=SETTINGS):
    snap = ctl.model.get_snapshot(ctl.current_snap_idx)
    ctl._open_fof_dialog()
    ctl._run_fof_algorithm(settings)
    assert _wait(app, lambda: 'fof' in snap and not ctl._fof_worker.isRunning(), 300)
    app.processEvents()
    return snap


# ----------------------------------------------------------------------------- menu

def test_structures_menu_replaces_file_read_groups(app, ctl):
    bar = ctl.view.menuBar()
    titles = [a.text() for a in bar.actions()]
    assert titles.index('Structures') == titles.index('File') + 1
    menu = next(a.menu() for a in bar.actions() if a.text() == 'Structures')
    assert [a.text() for a in menu.actions() if not a.isSeparator()] == \
        ['Find FoF groups...', 'Read halo catalogue...', 'Halo table']
    file_menu = next(a.menu() for a in bar.actions() if a.text() == 'File')
    texts = [a.text() for a in file_menu.actions()]
    assert not any('groups' in t.lower() or 'fof' in t.lower() or 'catalogue' in t.lower() for t in texts)
    assert 'Save session...' in texts and 'Read simulation snapshot...' in texts
    seps = [a.isSeparator() for a in file_menu.actions()]
    assert not any(a and b for a, b in zip(seps, seps[1:])), "no doubled separators in the File menu"

    next(a for a in menu.actions() if a.text() == 'Find FoF groups...').trigger()
    assert ctl.fof_dialog is not None and ctl.fof_dialog.btn_highlight.text() == "Select halo (all members)"
    next(a for a in menu.actions() if a.text() == 'Read halo catalogue...').trigger()
    assert ctl.group_dialog is not None
    ctl.group_dialog.close()
    ctl.view.resize(1200, 900)                       # on screen, as in use: docks get show and hide events
    ctl.view.show()
    app.processEvents()
    halo = next(a for a in menu.actions() if a.text() == 'Halo table')
    halo.trigger()
    app.processEvents()
    assert ctl._halo_table_open() and halo.isChecked()
    ctl.structures_dock.close()                      # the dock's own close button unticks the menu
    app.processEvents()
    assert not halo.isChecked() and not ctl._halo_table_open()
    halo.trigger()                                   # and the menu opens it again
    app.processEvents()
    assert ctl._halo_table_open() and ctl.structures_dock.isVisible()


# ----------------------------------------------------------------------------- catalogue on dm_cut

def test_dm_cut_catalogue_table_selection_and_check(app, ctl, tmp_path):
    d = _dm_cut(tmp_path)
    ctl.max_particles = 100_000
    _load(app, ctl, d)
    snap = ctl.model.snapshots[0]
    assert len(snap['c']) < 400_000                  # premise: a display sample, not every particle
    p = _open_table(app, ctl)
    assert p.combo_source.count() == 1 and "fof_tab_002" in p.combo_source.itemText(0)
    assert p.model.rowCount() == 2774
    headers = [c.header for c in p.model.columns]
    assert any("10¹⁰ M☉/h" in h for h in headers) and any("Mpc/h (comoving)" in h for h in headers)
    assert "snapshot_002.hdf5 (on screen)" in p.lbl_note.text() and "fof_tab_002.hdf5" in p.lbl_note.text()
    assert (_cell(p, 0, 'group'), _cell(p, 0, 'n')) == ("1", "34,120")

    # group 1: exactly the catalogue's members, complete, and the view centred on it
    assert p.jump_to(1)
    s = _active(ctl)
    want, ln = _rows_ids(d / "snapshot_002.hdf5", d / "fof_tab_002.hdf5", 1)
    np.testing.assert_array_equal(np.sort(s.id_array()), want)
    assert (s.kind, s.complete, s.n_expected, len(s.selected_ids)) == ('group', True, 34_120, 34_120)
    assert s.definition == {'finder': 'catalogue', 'group': 1, 'subhalo': False, 'catalogue_file': 'fof_tab_002.hdf5'}
    assert s.defined_at['file'] == 'snapshot_002.hdf5' and s.defined_at['index'] == 0
    assert "complete" in p.lbl_selection.text() and "34,120" in p.lbl_selection.text()
    with h5py.File(d / "fof_tab_002.hdf5", "r") as f:
        gpos = f["Group/GroupPos"][0].astype(np.float64)
    assert np.allclose(ctl.view.camera_state()['center'], gpos, atol=0.05)

    # check against the catalogue: N equal, mass and centre within float32 precision
    p.btn_check.click()
    ch = ctl.structures_last_check
    assert int(ch.n_found[0]) == int(ch.n_cat[0]) == 34_120
    assert ch.mass_ok[0] and ch.pos_ok[0] and ch.pos_ulp[0] <= 4 and ch.mass_ulp[0] <= 2
    assert "agrees within float32 precision" in p.lbl_check.text()

    # a group that runs past the end of the cut snapshot: incomplete, and the panel says why
    _, cat = ctl._catalogue_for(snap)
    end = cat.group_offset[:, 1] + cat.group_len[:, 1]
    g = int(np.flatnonzero((cat.group_offset[:, 1] < 400_000) & (end > 400_000))[0]) + 1
    p.jump_to(g)
    s = _active(ctl)
    want, ln = _rows_ids(d / "snapshot_002.hdf5", d / "fof_tab_002.hdf5", g)
    assert 0 < len(want) < ln
    np.testing.assert_array_equal(np.sort(s.id_array()), want)
    assert (s.kind, s.complete, s.n_expected) == ('group', False, ln)
    # the set itself names the cause: members missing from the file, not the display sample
    assert s.incomplete_reason == 'missing_in_file'
    assert "display sample" not in s.describe() and "display sample" not in (s.completeness_warning() or "")
    text = p.lbl_selection.text()
    assert f"Only {len(want):,} of the {ln:,} members" in text and "marked incomplete" in text
    p.btn_check.click()
    assert "members missing from the file" in p.lbl_check.text()

    # a group wholly beyond the cut: refused, the set is left as it was
    version = s.version
    g_out = int(np.flatnonzero(cat.group_offset[:, 1] >= 400_000)[0]) + 1
    p.jump_to(g_out)
    assert s.version == version and "None of the" in p.lbl_selection.text()

    # measure every group: the 29 groups wholly inside the cut agree; the rest are said to be cut
    ctl._measure_all_structures()
    assert _wait(app, lambda: ctl._structures_job is None, 120)
    ch = ctl.structures_last_check
    inside = end <= 400_000
    assert np.array_equal(ch.complete, inside)
    assert ch.mass_ok[inside].all() and ch.pos_ok[inside].all()
    assert "fewer members in the file than the catalogue says" in p.lbl_check.text()


def test_table_sorts_filters_and_jumps_without_reselecting(app, ctl, tmp_path):
    d = _dm_cut(tmp_path)
    _load(app, ctl, d)
    p = _open_table(app, ctl)
    from PyQt6.QtCore import Qt
    n = p.model.column_values('n')
    p.jump_to(5)
    s = _active(ctl)
    version = s.version
    col_n = next(k for k, c in enumerate(p.model.columns) if c.key == 'n')
    p.table.sortByColumn(col_n, Qt.SortOrder.AscendingOrder)
    app.processEvents()
    shown = [int(_cell(p, r, 'n').replace(',', '')) for r in range(p.model.rowCount())]
    assert shown == sorted(shown)
    assert p.selected_group() == 5                       # the selection moved with its group...
    _wait(app, lambda: False, 0.3)
    assert s.version == version                          # ...and was not applied again
    p.spin_min_n.setValue(1000)
    assert p.model.rowCount() == int((n >= 1000).sum())
    assert p.selected_group() == 5
    small = int(np.flatnonzero(n < 1000)[0]) + 1
    assert not p.jump_to(small) and "minimum N" in p.lbl_note.text()
    assert not p.jump_to(10 ** 6) and "no group" in p.lbl_note.text()

    # Analyse: the Analyse view (controller/analysis_host.py) is part of the build, and the
    # button hands it the active set. A stub stands in for it so this test does not start the
    # Explorer; tests/test_analysis_host.py tests the real hand-over.
    assert callable(getattr(ctl, '_analyse_active_set', None))
    calls = []
    ctl._analyse_active_set = lambda: calls.append(_active(ctl).describe())
    p.jump_to(1)
    assert p.btn_analyse.isEnabled()
    p.btn_analyse.click()
    assert calls and calls[0].startswith("catalogue group 1 of fof_tab_002.hdf5")


# ----------------------------------------------------------------------------- FoF rows

def test_fof_rows_equal_the_fof_result_and_select_complete_members(app, ctl, tmp_path):
    _write_clustered(tmp_path, "snapshot_000.hdf5")
    _load(app, ctl, tmp_path)
    p = _open_table(app, ctl)
    assert p.model.rowCount() == 0 and "FoF has not been run on snapshot_000.hdf5" in p.lbl_note.text()
    snap = _run_fof(app, ctl)
    assert _wait(app, lambda: p.model.rowCount() > 0, 5)
    sizes = snap['fof_sizes']
    assert p.model.rowCount() == len(sizes) and "FoF on this snapshot" in p.combo_source.currentText()
    np.testing.assert_array_equal(p.model.column_values('n'), [sizes[g] for g in range(1, len(sizes) + 1)])
    assert _cell(p, 1, 'mass') == "—"                   # not measured yet: never shown as zero

    p.jump_to(2)
    s = _active(ctl)
    fof = snap['fof']
    members = np.unique(np.asarray(fof.ids)[np.asarray(fof.group_index) == 1].astype(np.uint64))
    np.testing.assert_array_equal(np.sort(s.id_array()), members)
    assert (s.kind, s.complete, s.n_expected) == ('group', True, sizes[2])
    assert s.definition['finder'] == 'fof' and s.definition['group'] == 2 and s.definition['b'] == 0.2

    # the selected group's mass and centre are measured; an independent computation agrees
    with h5py.File(tmp_path / "snapshot_000.hdf5", "r") as f:
        pos = f["PartType1/Coordinates"][:]
        rows = np.isin(f["PartType1/ParticleIDs"][:], members)
    com = _independent_com(pos[rows], pos[rows][0].astype(np.float64), BOX)
    row = p.model.row_of(2)
    assert float(_cell(p, row, 'mass').replace(',', '')) == pytest.approx(len(members) * 1.0)
    assert [float(_cell(p, row, a)) for a in "xyz"] == pytest.approx(com.tolist(), abs=1e-4)
    assert _cell(p, p.model.row_of(3), 'mass') == "—"

    # Measure all groups fills every row, with the same value for the group measured before
    ctl._measure_all_structures()
    assert _wait(app, lambda: ctl._structures_job is None, 120)
    mass = p.model.column_values('mass')
    assert np.all(np.isfinite(mass))
    np.testing.assert_allclose(mass, [sizes[g] for g in range(1, len(sizes) + 1)])
    assert float(_cell(p, p.model.row_of(2), 'x')) == pytest.approx(com[0], abs=1e-4)
    assert "Measured all" in p.lbl_check.text()


def test_snapshot_change_follows_the_honesty_rule(app, ctl, tmp_path):
    _write_clustered(tmp_path, "snapshot_000.hdf5", time=0.5)
    _write_clustered(tmp_path, "snapshot_001.hdf5", time=1.0)
    _load(app, ctl, tmp_path)
    _show(app, ctl, 1)
    p = _open_table(app, ctl)
    snap1 = _run_fof(app, ctl)
    assert _wait(app, lambda: p.model.rowCount() == len(snap1['fof_sizes']), 5)
    stale = ctl._structures_source

    _show(app, ctl, 0)
    assert _wait(app, lambda: p.model.rowCount() == 0, 5)
    assert p.combo_source.count() == 0
    note = p.lbl_note.text()
    assert "FoF groups were found in snapshot_001.hdf5, not in snapshot_000.hdf5 (on screen)" in note

    # a list made for snapshot 1 is never applied to snapshot 0
    s = _active(ctl)
    s.select_by_ids(np.array([1, 2, 3], dtype=np.uint64))
    ctl._structures_source = stale
    ctl._on_structure_chosen(1)
    assert s.selected_ids == {1, 2, 3}
    assert "refreshed" in p.lbl_note.text() and p.model.rowCount() == 0

    _show(app, ctl, 1)
    assert _wait(app, lambda: p.model.rowCount() == len(snap1['fof_sizes']), 5)
    p.jump_to(1)
    assert len(s.selected_ids) == snap1['fof_sizes'][1] and s.defined_at['file'] == 'snapshot_001.hdf5'


def test_big_snapshots_are_centred_when_the_tracking_lookup_lands(app, ctl, tmp_path, monkeypatch):
    import controller.structures_controller as sc
    monkeypatch.setattr(sc, "SYNC_LOOKUP_MAX", 0)       # as for a snapshot too big to look up at once
    _write_clustered(tmp_path, "snapshot_000.hdf5")
    _load(app, ctl, tmp_path)
    p = _open_table(app, ctl)
    snap = _run_fof(app, ctl)
    assert _wait(app, lambda: p.model.rowCount() > 0, 5)
    p.jump_to(1)
    assert ctl._structures_pending is not None
    assert _wait(app, lambda: ctl._structures_pending is None, 60)
    members = ctl._fof_members(snap, 1)
    with h5py.File(tmp_path / "snapshot_000.hdf5", "r") as f:
        pos = f["PartType1/Coordinates"][:][np.isin(f["PartType1/ParticleIDs"][:], members)]
    centre = _independent_com(pos, pos[0].astype(np.float64), BOX)
    d = np.asarray(ctl.view.camera_state()['center']) - centre
    assert np.abs(d - BOX * np.round(d / BOX)).max() < 0.05
    assert _cell(p, p.model.row_of(1), 'mass') != "—"


# ----------------------------------------------------------------------------- subhaloes

def test_subhalo_catalogue_rows_selection_and_check(app, ctl, tmp_path):
    pos, ids, offs, sizes, s_off, s_len, gcom, scom = _write_with_subhaloes(tmp_path)
    _load(app, ctl, tmp_path)
    p = _open_table(app, ctl)
    labels = [p.combo_source.itemText(i) for i in range(p.combo_source.count())]
    assert labels == ["Catalogue groups · fof_subhalo_tab_000.hdf5", "Catalogue subhaloes · fof_subhalo_tab_000.hdf5"]
    p.jump_to(1)                                         # group 1 straddles the box corner
    p.btn_check.click()
    ch = ctl.structures_last_check
    assert ch.complete[0] and ch.mass_ok[0] and ch.pos_ok[0]

    p.combo_source.setCurrentIndex(1)
    p.combo_source.activated.emit(1)
    assert [c.key for c in p.model.columns] == ['group', 'parent', 'rank', 'n', 'mass', 'x', 'y', 'z']
    assert [_cell(p, r, 'parent') for r in range(3)] == ["1", "1", "2"]
    p.jump_to(2)
    s = _active(ctl)
    np.testing.assert_array_equal(np.sort(s.id_array()), np.sort(ids[s_off[1]:s_off[1] + s_len[1]]))
    assert s.definition == {'finder': 'catalogue', 'group': 2, 'subhalo': True,
                            'catalogue_file': 'fof_subhalo_tab_000.hdf5'}
    assert s.complete is True and s.n_expected == s_len[1] and "catalogue subhalo 2" in s.describe()
    p.btn_check.click()
    ch = ctl.structures_last_check
    assert ch.pos_field == 'SubhaloCM' and ch.complete[0] and ch.mass_ok[0] and ch.pos_ok[0]

    # the source kind chosen is kept when the table is rebuilt
    ctl._structures_follow(force=True)
    assert "subhaloes" in p.combo_source.currentText()


# ----------------------------------------------------------------------------- inspector

def test_inspector_selects_the_catalogue_group_of_a_particle(app, ctl, tmp_path):
    d = _dm_cut(tmp_path)
    _load(app, ctl, d)
    ip = ctl.view.inspector_panel
    assert not ip.btn_cat_group.isEnabled() and not ip.btn_fof_group.isEnabled()
    want, _ = _rows_ids(d / "snapshot_002.hdf5", d / "fof_tab_002.hdf5", 3)
    ctl.inspect_particle(int(want[0]))
    assert _wait(app, lambda: ip.btn_cat_group.isEnabled(), 60)
    assert ip.btn_cat_group.text() == "Select its catalogue group (3)"
    assert not ip.btn_fof_group.isEnabled() and "FoF has not been run" in ip.btn_fof_group.toolTip()
    ip.btn_cat_group.click()
    s = _active(ctl)
    np.testing.assert_array_equal(np.sort(s.id_array()), want)
    assert (s.kind, s.complete, s.definition['group']) == ('group', True, 3)
    assert "catalogue group 3" in ip.lbl_status.text()


def test_inspector_shows_and_selects_the_fof_group_of_a_particle(app, ctl, tmp_path):
    _write_clustered(tmp_path, "snapshot_000.hdf5")
    _load(app, ctl, tmp_path)
    snap = _run_fof(app, ctl)
    fof = snap['fof']
    gi = np.asarray(fof.group_index)
    pid = int(np.asarray(fof.ids)[gi == 3][0])               # a member of FoF group 4
    ip = ctl.view.inspector_panel
    ctl.inspect_particle(pid)
    assert _wait(app, lambda: ip.btn_fof_group.isEnabled(), 60)
    ctl._refresh_inspector()
    row = next(r for r in ip.table_rows() if r[0] == 'FoF group (computed here)')
    assert row[1] == "4"
    assert ip.btn_fof_group.text() == "Select its FoF group (4)"
    assert not ip.btn_cat_group.isEnabled() and "No halo catalogue" in ip.btn_cat_group.toolTip()
    ip.btn_fof_group.click()
    s = _active(ctl)
    assert (s.kind, s.complete, s.n_expected, len(s.selected_ids)) == ('group', True, snap['fof_sizes'][4],
                                                                       snap['fof_sizes'][4])
    assert s.definition['finder'] == 'fof' and s.definition['group'] == 4

    outside = int(np.asarray(fof.ids)[gi < 0][0]) if np.any(gi < 0) else None
    if outside is not None:
        ctl.inspect_particle(outside)
        assert _wait(app, lambda: (ctl._record_at(0) or None) is not None and ctl._record_at(0).found, 60)
        ctl._refresh_inspector()
        assert not ip.btn_fof_group.isEnabled() and "no FoF group" in ip.btn_fof_group.toolTip()


# ----------------------------------------------------------------------------- measurement and units

def test_accumulator_gives_the_periodic_centre_of_mass_in_any_order():
    rng = np.random.default_rng(2)
    box = 10.0
    centre = np.array([9.95, 0.02, 5.0])                    # straddles two faces
    pos = (centre + rng.normal(0, 0.2, (5000, 3))) % box
    m = rng.random(5000) + 0.5
    acc = GroupAccumulator(3, box)
    order = rng.permutation(5000)
    for part in np.array_split(order, 7):                    # in pieces, in no particular order
        acc.add(np.full(len(part), 1), pos[part], m[part])
    r = acc.result()
    d = pos - centre
    d -= box * np.round(d / box)
    want = np.mod(centre + (m[:, None] * d).sum(0) / m.sum(), box)
    assert r.count.tolist() == [0, 5000, 0] and r.mass[1] == pytest.approx(m.sum())
    assert np.allclose(r.com[1], want, atol=1e-12) and np.isnan(r.com[0]).all()
    iso = GroupAccumulator(1, 0.0)                           # isolated run: no wrapping
    iso.add(np.zeros(3, int), np.array([[1.0, 0, 0], [3.0, 0, 0], [5.0, 0, 0]]), 1.0)
    assert iso.result().com[0].tolist() == [3.0, 0.0, 0.0]


def test_stored_units_from_parameters_and_attributes(tmp_path):
    d = _dm_cut(tmp_path)
    u = stored_units([str(d / "snapshot_002.hdf5")])
    assert (u.length, u.mass, u.comoving) == ("Mpc/h (comoving)", "10¹⁰ M☉/h", True)
    fn = tmp_path / "iso.hdf5"                               # an isolated run in kpc, without h
    with h5py.File(fn, "w") as f:
        f.create_group("Header").attrs.update({"BoxSize": 0.0, "Time": 0.3, "MassTable": np.zeros(6)})
        _params(f, comoving=0, length=3.085678e21, h=1.0)
        g = f.create_group("PartType0")
        g["Coordinates"] = np.zeros((2, 3), np.float32)
        g["Coordinates"].attrs.update(a_scaling=0.0, h_scaling=0.0)
        g["Masses"] = np.ones(2, np.float32)
        g["Masses"].attrs.update(a_scaling=0.0, h_scaling=0.0)
    u = stored_units([str(fn)])
    assert (u.length, u.mass, u.comoving) == ("kpc", "10¹⁰ M☉", False)


# ----------------------------------------------------------------------------- real data

@pytest.mark.skipif(not G4_OUTPUT, reason="set COSMOVIEWER_G4_OUTPUT to DM-L50-N128/output")
def test_real_dm_l50_catalogue_membership_and_check(app, ctl, tmp_path):
    src = Path(G4_OUTPUT)
    d = tmp_path / "dm_l50"
    d.mkdir()
    for name in ("snapshot_002.hdf5", "fof_tab_002.hdf5"):   # copies: the application may write next to them
        shutil.copy(src / name, d / name)
    snapshot, catalogue = d / "snapshot_002.hdf5", d / "fof_tab_002.hdf5"

    # every group of fof_tab_002, measured from the snapshot's particles in one pass
    cat = read_gadget4_catalogue(str(catalogue))
    fields, header, _ = read_catalogue_fields(str(catalogue))
    t0 = time.time()
    meas = measure_catalogue_groups([str(snapshot)], cat)
    took = time.time() - t0
    ch = compare_with_catalogue(meas, fields, cat, False, 50.0)
    assert len(ch.groups) == 2774
    assert ch.complete.all() and ch.mass_ok.all() and ch.pos_ok.all()
    print(f"\nDM-L50 fof_tab_002: {len(ch.groups):,} groups, N equal for all; worst mass "
          f"{np.nanmax(ch.mass_ulp):.3g} ulp; worst centre {np.nanmax(ch.pos_ulp):.3g} ulp "
          f"(1 ulp = {ch.ulp_size[0]:.3g} Mpc/h); {took:.2f} s")

    # in the application: rows chosen in the table hold exactly the offset rows
    ctl.max_particles = 200_000
    _load(app, ctl, d)
    p = _open_table(app, ctl)
    assert p.model.rowCount() == 2774
    for g in (1, 2, 100, 2774):
        p.jump_to(g)
        s = _active(ctl)
        want, ln = _rows_ids(snapshot, catalogue, g)
        np.testing.assert_array_equal(np.sort(s.id_array()), want)
        assert (s.complete, s.n_expected, len(want)) == (True, ln, ln)
    p.jump_to(1)
    p.btn_check.click()
    one = ctl.structures_last_check
    assert int(one.n_found[0]) == 34_120 == int(one.n_cat[0]) and one.mass_ok[0] and one.pos_ok[0]
