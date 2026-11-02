"""
Populations: a selection set records what it means, not only which particles it holds.

A set is a list of ParticleIDs plus its kind ("ids", "group", "region",
"query"), its definition, the snapshot it was defined at, and whether it is
complete (made from every particle) or was taken from the display sample.
These tests pin that contract (selection_sets.py, version 2), its file format
and backward compatibility, how editing a set by hand changes its meaning,
and that a FoF halo becomes a complete "group" population.
"""

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
from model.simulation_model import ParticleSelectionSet  # noqa: E402

FOF_DEF = {'finder': 'fof', 'group': 1, 'b': 0.2, 'min_size': 32, 'link_types': [1]}
AT = {'file': 'snapshot_002.hdf5', 'index': 2, 'time': 1.0, 'redshift': 0.0}


def test_file_round_trip_keeps_the_meaning(tmp_path):
    snap = tmp_path / "snapshot_002.hdf5"
    snap.write_bytes(b"")
    p = selection_sets.write_set(str(snap), np.arange(5), "halo 1", kind="group", definition=FOF_DEF,
                                 defined_at=AT, complete=True, n_expected=5)
    st = selection_sets.read_set(p)
    assert (st.kind, st.definition, st.defined_at, st.complete, st.n_expected) == ("group", FOF_DEF, AT, True, 5)
    assert st.describe() == "FoF group 1 (b = 0.2, N_min = 32, DM) at snapshot_002.hdf5 · complete · 5 particles"
    assert st.description == st.describe()           # written by default
    assert st.completeness_warning() is None
    with h5py.File(p, "r") as f:
        assert int(f.attrs["version"]) == 2


def test_numpy_values_in_definitions_are_written_as_plain_json(tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    snap.write_bytes(b"")
    d = {'shape': 'sphere', 'centre': np.array([1.0, 2.0, 3.0]), 'radius': np.float32(0.5), 'types': [np.int64(1)]}
    st = selection_sets.read_set(selection_sets.write_set(str(snap), [1, 2], "r", kind="region", definition=d))
    assert st.definition == {'shape': 'sphere', 'centre': [1.0, 2.0, 3.0], 'radius': 0.5, 'types': [1]}


def test_version_1_files_read_as_plain_ids_of_unknown_completeness(tmp_path):
    p = tmp_path / "old.hdf5"
    with h5py.File(p, "w") as f:                       # exactly what version 1 wrote
        f["ParticleIDs"] = np.array([3, 1, 2], dtype=np.uint64)
        f.attrs.update(dict(format=selection_sets.FORMAT, version=1, name="old", colour="#ff0000",
                            created="2026-10-18T12:00:00", made_in="CosmoViewer Immersive",
                            source_snapshot="snapshot_000.hdf5", description="selected in x", n=3))
    st = selection_sets.read_set(str(p))
    assert (st.kind, st.definition, st.defined_at, st.complete, st.n_expected) == ("ids", {}, {}, None, None)
    assert st.describe() == "particle IDs · 3 particles"
    assert st.completeness_warning() is None


def test_an_incomplete_set_says_so():
    st = selection_sets.SelectionSet(np.arange(8442, dtype=np.uint64), "halo", kind="group", definition=FOF_DEF,
                                     defined_at=AT, complete=False, n_expected=34120,
                                     incomplete_reason="display_sample")
    assert "display sample only: 8,442 of 34,120" in st.describe()
    w = st.completeness_warning()
    assert w and "8,442 of its 34,120" in w and "not the whole group" in w


def test_unknown_kind_is_refused(tmp_path):
    with pytest.raises(ValueError):
        selection_sets.write_set(str(tmp_path / "s.hdf5"), [1], "x", kind="halo")
    with pytest.raises(ValueError):
        ParticleSelectionSet(0).set_population([1], kind="halo")


def test_selection_set_records_and_changes_its_meaning():
    s = ParticleSelectionSet(0)
    v0 = s.version
    s.set_population(np.arange(10), 'group', FOF_DEF, AT, complete=True, n_expected=10)
    assert s.version == v0 + 1 and s.kind == 'group'
    group_text = s.describe()
    assert group_text.startswith("FoF group 1") and "complete" in group_text

    # editing by hand: plain IDs that remember what they were edited from, never "complete group"
    s.add_ids([100, 101])
    assert s.kind == 'ids' and s.complete is None and s.n_expected is None
    assert s.definition == {'edited_from': group_text, 'operation': 'add'}
    assert "edited (add) from: FoF group 1" in s.describe()
    s.intersect_ids(np.arange(5))
    assert s.definition['operation'] == 'subset'

    # replacing with plain IDs forgets the old meaning; clearing too
    s.set_population(np.arange(3), 'group', FOF_DEF, AT, complete=True, n_expected=3)
    s.select_by_ids([7, 8])
    assert (s.kind, s.definition, s.complete) == ('ids', {}, None)
    s.set_population(np.arange(3), 'group', FOF_DEF, AT, complete=True, n_expected=3)
    s.clear()
    assert (s.kind, s.definition, s.complete, len(s.selected_ids)) == ('ids', {}, None, 0)


def test_editing_an_incomplete_set_keeps_it_incomplete():
    """Adding or removing particles by hand does not make an incomplete set complete, nor
    'unknown': it stays incomplete for the same reason, and keeps warning."""
    s = ParticleSelectionSet(0)
    s.set_population(np.arange(10), 'group', FOF_DEF, AT, complete=False, n_expected=40,
                     incomplete_reason='display_sample')
    s.add_ids([100])
    assert (s.kind, s.complete, s.incomplete_reason) == ('ids', False, 'display_sample')
    assert "display sample" in (s.completeness_warning() or "")
    s.intersect_ids(np.arange(5))
    assert (s.complete, s.incomplete_reason) == (False, 'display_sample')
    assert s.completeness_warning()


def test_as_selection_set_carries_everything():
    s = ParticleSelectionSet(2, name="Halo 1")
    s.set_population(np.array([5, 3, 9]), 'group', FOF_DEF, AT, complete=True, n_expected=3)
    st = s.as_selection_set("#ffaa00")
    np.testing.assert_array_equal(np.sort(st.ids), [3, 5, 9])
    assert (st.name, st.colour, st.kind, st.definition, st.defined_at, st.complete, st.n_expected) == \
        ("Halo 1", "#ffaa00", 'group', FOF_DEF, AT, True, 3)
    assert st.describe() == s.describe() and st.path == ""


# ----------------------------------------------------------------------------- FoF halo -> group population

BOX = 20.0


def _write(folder, n=30_000, seed=5):
    rng = np.random.default_rng(seed)
    centres = rng.random((20, 3)) * BOX
    c = (centres[rng.integers(0, 20, n)] + rng.normal(0, 0.4, (n, 3))) % BOX
    with h5py.File(Path(folder) / "snapshot_000.hdf5", "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": 1.0, "Redshift": 0.0, "NumFilesPerSnapshot": 1,
                        "MassTable": np.array([0.0, 1.0, 0, 0, 0, 0]),
                        "NumPart_ThisFile": np.array([0, n, 0, 0, 0, 0]),
                        "NumPart_Total": np.array([0, n, 0, 0, 0, 0])})
        p = f.create_group("Parameters")
        p.attrs.update({"UnitLength_in_cm": 3.085678e24, "UnitMass_in_g": 1.989e43, "Omega0": 0.3,
                        "HubbleParam": 0.7, "UnitVelocity_in_cm_per_s": 1e5})
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=c.astype(np.float32))
        g.create_dataset("Velocities", data=np.zeros((n, 3), np.float32))
        g.create_dataset("ParticleIDs", data=rng.permutation(n).astype(np.uint64) + 1)


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_fof_halo_becomes_a_complete_group_population(app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    import lod.store
    monkeypatch.setattr(lod.store, "CACHE_DIR", tmp_path / "lod_cache")
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    try:
        _write(tmp_path)
        c._load_directory(tmp_path)
        end = time.time() + 120
        while time.time() < end and (c.loading or not c.model.snapshots):
            app.processEvents()
            time.sleep(0.02)
        snap = c.model.snapshots[0]
        c._open_fof_dialog()
        c._run_fof_algorithm({'b_linking': 0.2, 'min_size': 20, 'r_link': None, 'link_types': [1]})
        while time.time() < end and not ('FoFGroupIndex' in snap and not c._fof_worker.isRunning()):
            app.processEvents()
            time.sleep(0.02)
        c._highlight_fof_halo(1)
        s = c.model.tracker.get_set(c.view.selection_panel.active_set_idx)
        assert s.kind == 'group' and s.complete is True
        assert s.n_expected == snap['fof_sizes'][1] == len(s.selected_ids)
        assert s.definition['finder'] == 'fof' and s.definition['group'] == 1
        assert s.definition['b'] == 0.2 and s.definition['min_size'] == 20 and s.definition['link_types'] == [1]
        assert s.definition['linking_length_derived'] == pytest.approx(snap['fof_linking_length'])
        assert s.defined_at['file'] == 'snapshot_000.hdf5' and s.defined_at['index'] == 0
        assert s.describe().startswith("FoF group 1 (b = 0.2, N_min = 20, DM) at snapshot_000.hdf5 · complete")
    finally:
        c._shutdown_fof()
        c._shutdown_lod()
        c.view.close()
