"""
Follow selection centres on the active set (GUI walkthrough of 2026-10-26, bug B9).

The centre used to be the mean of every enabled set's particles, so with a second set enabled
the view was pulled off the set being followed. Two clumps far apart, one set on each; the
view centre after a snapshot change must be the active set's mean, read independently from
the file with h5py, whichever set is active. Synthetic snapshots, headless.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_host import app, _wait  # noqa: E402,F401
from measurements.models import write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402
import selection_sets  # noqa: E402

N = 500                       # particles per clump


def _write_run(folder: Path) -> list:
    """Clump A (IDs 1..N) near (1, 1, 1) and clump B (IDs N+1..2N) near (8, 8, 8), both
    moving by 0.5 in x per snapshot."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.05, (N, 3)) + 1.0
    b = rng.normal(0.0, 0.05, (N, 3)) + 8.0
    code_mass = 1.989e43 / MSUN_G
    paths = []
    for i in range(3):
        pos = np.vstack([a, b]) + [0.5 * i, 0.0, 0.0]
        p = folder / f"snapshot_{i:03d}.hdf5"
        write_gadget4_snapshot(p, {1: dict(pos=pos, vel=np.zeros_like(pos), mass=None,
                                           ids=np.arange(1, 2 * N + 1))},
                               time=0.5 + 0.25 * i, mass_table=[0, 1.0 / code_mass, 0, 0, 0, 0])
        paths.append(p)
    return paths


def _mean_of_ids(path, ids):
    with h5py.File(path, "r") as f:
        pid = f["PartType1/ParticleIDs"][:]
        pos = f["PartType1/Coordinates"][:].astype(np.float64)
    return pos[np.isin(pid, ids)].mean(axis=0)


@pytest.fixture
def ctl(app, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(selection_sets, "FALLBACK_ROOT", tmp_path / "home" / "sets")
    monkeypatch.setattr(selection_sets, "INSTANCES_DIR", tmp_path / "home" / "explorer_instances")
    run = tmp_path / "run"
    run.mkdir()
    paths = _write_run(run)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c._load_directory(run)
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 120)
    app.processEvents()
    c.test_paths = paths
    yield c
    c._shutdown_analysis_host()
    c._cancel_record_job()
    for j in c._old_record_jobs:
        j.wait(5000)
    c._reset_tracking()
    c._shutdown_lod()


def _tracking_done(c):
    return c._track_worker is None or c._track_worker.isFinished()


@pytest.mark.parametrize("active", [0, 1])
def test_follow_centres_on_the_active_set_only(app, ctl, active):
    c = ctl
    ids = [np.arange(1, N + 1), np.arange(N + 1, 2 * N + 1)]       # set 1: clump A, set 2: clump B
    for k in (0, 1):
        s = c.model.tracker.get_set(k)
        s.select_by_ids(ids[k])
        s.enabled = True
    sp = c.view.selection_panel
    sp.combo_sets.setCurrentIndex(active)
    assert sp.active_set_idx == active
    sp.chk_follow.setChecked(True)
    c._update_frame()
    assert _wait(app, lambda: _tracking_done(c) and all(
        c._tracked(i, c.model.tracker.get_set(k)) is not None for i in range(3) for k in (0, 1)))
    for snap in (2, 1):
        c.view.slide_time.setValue(snap)
        for _ in range(3):
            app.processEvents()
        want = _mean_of_ids(c.test_paths[snap], ids[active])
        other = _mean_of_ids(c.test_paths[snap], ids[1 - active])
        got = np.array(c.view.camera_state()["center"])
        assert np.max(np.abs(got - want)) < 1e-3, (snap, got, want)
        assert np.linalg.norm(got - other) > 5.0
        # the scene's centre (also used by movies) is the same
        scene = c._build_scene(snap, c._ui_state())
        np.testing.assert_allclose(scene.centre, want, atol=1e-3)
