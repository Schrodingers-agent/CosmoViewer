"""
Selecting a FoF group gives the whole group, and FoF results stay with the
snapshot they were computed for.

When a snapshot holds more particles than the particle budget, the display
sample contains only some members of each group. A selection made from a
group ("Highlight Halo", or the "FoF Group ID" criterion) must still hold every
member, as the catalogue criterion does, because the set is then tracked,
plotted, saved and analysed as if it were the group.

FoF runs on one snapshot. On any other snapshot the FoF window says whose
groups it holds, its actions do nothing silently no more, and the FoF
criterion is refused with a message instead of being dropped from the other
criteria (which would select a different population than the one asked for).
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

BOX = 20.0
N = 60_000
BUDGET = 10_000                       # display sample of 10,000 out of 60,000
SETTINGS = {'b_linking': 0.2, 'min_size': 20, 'r_link': None, 'link_types': [1]}


def _write(folder, name, n=N, seed=5, time=1.0):
    """Same particles in every file; distinct times keep the snapshot order fixed (the loader sorts by time)."""
    rng = np.random.default_rng(seed)
    centres = rng.random((30, 3)) * BOX
    c = (centres[rng.integers(0, 30, n)] + rng.normal(0, 0.4, (n, 3))) % BOX
    ids = rng.permutation(n).astype(np.uint64) + 1
    with h5py.File(Path(folder) / name, "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": time, "Redshift": 0.0, "NumFilesPerSnapshot": 1,
                        "MassTable": np.array([0.0, 1.0, 0, 0, 0, 0]),
                        "NumPart_ThisFile": np.array([0, n, 0, 0, 0, 0]),
                        "NumPart_Total": np.array([0, n, 0, 0, 0, 0])})
        p = f.create_group("Parameters")
        p.attrs.update({"UnitLength_in_cm": 3.085678e24, "UnitMass_in_g": 1.989e43, "Omega0": 0.3,
                        "HubbleParam": 0.7, "UnitVelocity_in_cm_per_s": 1e5})
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=c.astype(np.float32))
        g.create_dataset("Velocities", data=np.zeros((n, 3), np.float32))
        g.create_dataset("ParticleIDs", data=ids)


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    warnings = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: (warnings.append(str(a[2]) if len(a) > 2 else ""),
                                         QMessageBox.StandardButton.Ok)[1])
    import lod.store
    monkeypatch.setattr(lod.store, "CACHE_DIR", tmp_path / "lod_cache")
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c.max_particles = BUDGET
    c.warnings = warnings
    yield c
    c._shutdown_fof()
    c._shutdown_lod()
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


def _show(app, ctl, index):
    ctl.view.slide_time.setValue(index)          # as the user does, so the frame updates
    app.processEvents()
    assert ctl.current_snap_idx == index


def _run_fof(app, ctl):
    snap = ctl.model.get_snapshot(ctl.current_snap_idx)
    ctl._open_fof_dialog()
    ctl._run_fof_algorithm(SETTINGS)
    assert _wait(app, lambda: 'FoFGroupIndex' in snap and not ctl._fof_worker.isRunning())
    return snap


def _members(snap, g):
    fof = snap['fof']
    return np.unique(np.asarray(fof.ids)[np.asarray(fof.group_index) == g - 1].astype(np.uint64))


def _active_set(ctl):
    return ctl.model.tracker.get_set(ctl.view.selection_panel.active_set_idx)


def test_highlight_halo_selects_every_member(app, ctl, tmp_path):
    _write(tmp_path, "snapshot_000.hdf5")
    _load(app, ctl, tmp_path)
    snap = _run_fof(app, ctl)
    assert len(snap['id']) < N                  # premise: the display sample is partial
    g = 1
    shown_members = int(np.sum(snap['FoFGroupIndex'] == g))
    assert shown_members < snap['fof_sizes'][g]  # ...and holds only part of group 1

    ctl._highlight_fof_halo(g)
    s = _active_set(ctl)
    assert len(s.selected_ids) == snap['fof_sizes'][g]
    np.testing.assert_array_equal(np.sort(s.id_array()), _members(snap, g))


def test_fof_criterion_selects_every_member(app, ctl, tmp_path):
    _write(tmp_path, "snapshot_000.hdf5")
    _load(app, ctl, tmp_path)
    snap = _run_fof(app, ctl)
    sp = ctl.view.selection_panel
    sp.chk_fof.setChecked(True)
    sp.spin_fof_id.setValue(2)
    ctl._apply_selection_criteria()
    s = _active_set(ctl)
    assert len(s.selected_ids) == snap['fof_sizes'][2]
    np.testing.assert_array_equal(np.sort(s.id_array()), _members(snap, 2))


def test_fof_criterion_with_a_group_that_does_not_exist_is_refused(app, ctl, tmp_path):
    _write(tmp_path, "snapshot_000.hdf5")
    _load(app, ctl, tmp_path)
    snap = _run_fof(app, ctl)
    s = _active_set(ctl)
    s.select_by_ids(np.array([1, 2, 3], dtype=np.uint64))
    sp = ctl.view.selection_panel
    sp.chk_fof.setChecked(True)
    sp.spin_fof_id.setValue(len(snap['fof_sizes']) + 1)
    ctl._apply_selection_criteria()
    assert s.selected_ids == {1, 2, 3}
    assert ctl.warnings and "FoF group" in ctl.warnings[-1]


def test_fof_results_stay_with_the_snapshot_they_were_computed_for(app, ctl, tmp_path):
    _write(tmp_path, "snapshot_000.hdf5", time=0.5)
    _write(tmp_path, "snapshot_001.hdf5", time=1.0)
    _load(app, ctl, tmp_path)
    assert [Path(s['filepath']).name for s in ctl.model.snapshots] == ["snapshot_000.hdf5", "snapshot_001.hdf5"]
    _show(app, ctl, 1)
    snap1 = _run_fof(app, ctl)
    dlg = ctl.fof_dialog
    assert dlg.btn_highlight.isEnabled() and dlg.combo_halos.count() == len(snap1['fof_sizes'])
    assert "snapshot_001" in dlg.lbl_status.text()

    # On a snapshot FoF has not run on, the window says whose groups it holds
    _show(app, ctl, 0)
    assert not dlg.btn_highlight.isEnabled()
    assert not dlg.btn_recentre.isEnabled()
    assert dlg.combo_halos.count() == 0
    assert "snapshot_001" in dlg.lbl_status.text() and "snapshot_000" in dlg.lbl_status.text()

    # Highlight there leaves the set alone and says why
    s = _active_set(ctl)
    s.select_by_ids(np.array([1, 2, 3], dtype=np.uint64))
    ctl._highlight_fof_halo(1)
    assert s.selected_ids == {1, 2, 3}

    # FoF criterion AND radius there: refused with a message, not silently reduced to the radius
    sp = ctl.view.selection_panel
    sp.chk_fof.setChecked(True)
    sp.spin_fof_id.setValue(1)
    sp.chk_radius.setChecked(True)
    sp.spin_radius.setValue(2.0)
    sp.chk_full.setChecked(False)
    ctl._apply_selection_criteria()
    assert s.selected_ids == {1, 2, 3}
    assert ctl.warnings and "FoF" in ctl.warnings[-1] and "snapshot_000" in ctl.warnings[-1]

    # Back on the snapshot FoF ran on, its groups are offered again
    _show(app, ctl, 1)
    assert dlg.btn_highlight.isEnabled() and dlg.combo_halos.count() == len(snap1['fof_sizes'])
    ctl._highlight_fof_halo(1)
    assert len(s.selected_ids) == snap1['fof_sizes'][1]


def test_fof_palette_covers_every_snapshots_groups():
    # The palette is shared by all snapshots with FoF and indexed modulo its length. It used to
    # be sized for the latest run only, so after FoF on a snapshot with fewer groups, groups of an
    # earlier snapshot beyond that count were drawn in the grey of "in no group".
    from types import SimpleNamespace
    from controller.app_controller import CosmoViewerController
    many = {'fof_sizes': {g: 40 for g in range(1, 3034)}}
    few = {'filepath': '/x/snapshot_002.hdf5'}
    fof = SimpleNamespace(result=SimpleNamespace(linking_length=0.1), ids=np.zeros(0, np.uint64))
    stub = SimpleNamespace(model=SimpleNamespace(snapshots=[many, few]), fof_dialog=None,
                           _update_frame=lambda: None)
    state = np.random.get_state()[1].copy()
    CosmoViewerController._on_fof_done(stub, few, (np.zeros(0, np.int32), 2774, {g: 40 for g in range(1, 2775)}, fof))
    assert len(stub.fof_colors_palette) >= 3034
    assert not np.allclose(stub.fof_colors_palette[3000], [0.2, 0.2, 0.2, 0.2])
    np.testing.assert_array_equal(np.random.get_state()[1], state)     # global random state untouched
