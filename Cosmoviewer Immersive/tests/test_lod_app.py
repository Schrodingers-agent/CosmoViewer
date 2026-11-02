"""
Level of detail through the real controller and window (Qt offscreen).

Checks that the 3D view draws the level-of-detail set and nothing else does:
the snapshot's base sample, FoF input, movies and selections are unchanged.
Set COSMOVIEWER_G4_OUTPUT to DM-L50-N128/output to also run on real data.
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

from PyQt6.QtGui import QVector3D  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from lod.store import id_uniforms  # noqa: E402

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
BOX = 20.0


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    import lod.store
    monkeypatch.setattr(lod.store, "CACHE_DIR", tmp_path / "lod_cache")
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c.view.resize(1000, 700)
    yield c
    c._shutdown_lod()
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


def _write_snapshot(folder, n=200_000, seed=3):
    rng = np.random.default_rng(seed)
    centres = rng.random((30, 3)) * BOX
    c = (centres[rng.integers(0, 30, n)] + rng.normal(0, 0.4, (n, 3))) % BOX
    ids = rng.permutation(n).astype(np.uint64) + 1
    with h5py.File(Path(folder) / "snapshot_000.hdf5", "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"BoxSize": BOX, "Time": 1.0, "Redshift": 0.0, "NumFilesPerSnapshot": 1,
                        "MassTable": np.array([0.0, 1.0, 0, 0, 0, 0]),
                        "NumPart_ThisFile": np.array([0, n, 0, 0, 0, 0]),
                        "NumPart_Total": np.array([0, n, 0, 0, 0, 0])})
        p = f.create_group("Parameters")
        p.attrs.update({"UnitLength_in_cm": 3.085678e24, "UnitMass_in_g": 1.989e43, "Omega0": 0.3, "HubbleParam": 0.7,
                        "UnitVelocity_in_cm_per_s": 1e5})
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=c)
        g.create_dataset("Velocities", data=rng.normal(0, 100, (n, 3)).astype(np.float32))
        g.create_dataset("ParticleIDs", data=ids)
    return c, ids


def _drawn(c):
    p = c.view.scatter_3d.pos
    return np.empty((0, 3)) if p is None else np.asarray(p)


def _zoom(c, centre, distance):
    c.view.view_3d.setCameraPosition(pos=QVector3D(*map(float, centre)), distance=float(distance))


def test_view_draws_lod_set_and_base_sample_is_untouched(app, ctl, tmp_path):
    coords, ids = _write_snapshot(tmp_path)
    ctl._set_lod_budget(20_000)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots, 120)
    snap = ctl.model.snapshots[0]
    base_ids = snap['id'].copy()
    assert abs(len(base_ids) - 50_000) < 900  # base sample: budget, but at least MIN_BASE_SAMPLE (u < 1/4)
    np.testing.assert_array_equal(np.sort(base_ids), np.sort(ids[id_uniforms(ids, 0) < 50_000 / len(ids)]))
    assert set(snap['rows']) == {(0, 1)}
    np.testing.assert_array_equal(snap['id'], ids[snap['rows'][(0, 1)]])

    assert _wait(app, lambda: ctl._lod_view is not None, 60)
    far = ctl._lod_view[3]
    assert far.n_drawn <= 20_000 * 1.1 and len(_drawn(ctl)) == far.n_drawn
    assert "Detail level" in ctl.view.lbl_info.text() and "drawn near the focus" in ctl.view.lbl_info.text()

    # a small rotation keeps the drawn set: no reload, no redraw
    frames = []
    orig_update = ctl._update_frame
    ctl._update_frame = lambda *a, **k: (frames.append(1), orig_update(*a, **k))
    o = ctl.view.view_3d.opts
    ctl.view.view_3d.setCameraPosition(azimuth=float(o["azimuth"]) + 1.0)
    ck = ctl._lod_cam_key(ctl._lod_camera())
    assert _wait(app, lambda: ctl._lod_view[2] == ck and not ctl._lod_query.isRunning(), 60)
    assert ctl._lod_view[3] is far and not frames
    ctl._update_frame = orig_update

    # zoom on a dense spot: more of the particles there are drawn
    centre = coords[0]
    _zoom(ctl, centre, 1.0)
    assert _wait(app, lambda: ctl._lod_view is not None and ctl._lod_view[3] is not far, 60)
    near = ctl._lod_view[3]
    assert near.fraction > far.fraction
    drawn = _drawn(ctl)
    assert len(drawn) == near.n_drawn > 0
    # every drawn point is a particle of the file, at its own position
    rows = near.data['rows'][(0, 1)]
    np.testing.assert_array_equal(near.data['id'], ids[rows])
    np.testing.assert_allclose(drawn, coords[rows].astype(np.float32))

    # nothing else changed: base sample, and movies draw the base sample
    np.testing.assert_array_equal(snap['id'], base_ids)
    scene = ctl._build_scene(0, ctl._ui_state(), compute=True)
    assert len(scene.pos) == len(base_ids)


def test_fof_colours_use_full_resolution_groups(app, ctl, tmp_path):
    _write_snapshot(tmp_path, n=60_000)
    ctl._set_lod_budget(500_000)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots, 120)
    snap = ctl.model.snapshots[0]
    from snapshot_io.fof_finder import FoFGroupsEngine
    result = FoFGroupsEngine.detect_groups_in_snapshot(snap['filepath'], snap['id'], min_group_size=20)
    ctl._on_fof_done(snap, result)
    assert _wait(app, lambda: ctl._lod_view is not None, 60)
    shown = ctl._display_snapshot(0)
    fof = result[3]
    np.testing.assert_array_equal(shown['FoFGroupIndex'], np.asarray(fof.lookup(shown['id'])) + 1)
    np.testing.assert_array_equal(snap['FoFGroupIndex'], result[0])   # base mapping unchanged
    ctl._set_color_mode('fof_group')
    # grouped particles are the lit layer, the rest the unlit one: together, every drawn particle
    assert (len(np.asarray(ctl.view.scatter_3d.color)) + len(np.asarray(ctl.view.scatter_3d_lit.color))
            == len(shown['id']))


def test_big_snapshot_sample_comes_from_store(app, ctl, tmp_path, monkeypatch):
    import snapshot_io.snapshot_loader as sl
    monkeypatch.setattr(sl, "BIG_SNAPSHOT", 100_000)
    coords, ids = _write_snapshot(tmp_path)
    ctl._set_lod_budget(30_000)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots, 120)
    snap = ctl.model.snapshots[0]
    rows = snap['rows'][(0, 1)]
    assert abs(len(rows) - 50_000) < 2_000            # uniform random sample of the requested size
    np.testing.assert_array_equal(snap['id'], ids[rows])
    np.testing.assert_allclose(snap['c'], coords[rows].astype(np.float32))
    assert snap['n_total'] == len(ids)


def test_disabling_lod_restores_base_sample(app, ctl, tmp_path):
    _write_snapshot(tmp_path, n=100_000)
    ctl._set_lod_budget(20_000)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: ctl._lod_view is not None, 120)
    ctl._lod_toggle.setChecked(False)
    app.processEvents()
    assert len(_drawn(ctl)) == len(ctl.model.snapshots[0]['id'])


@pytest.mark.skipif(not G4_OUTPUT, reason="set COSMOVIEWER_G4_OUTPUT to DM-L50-N128/output")
def test_real_run_zoom_reads_more_detail(app, ctl):
    ctl._set_lod_budget(200_000)
    ctl._load_directory(Path(G4_OUTPUT))
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots, 300)
    ctl.view.slide_time.setValue(len(ctl.model.snapshots) - 1)
    assert _wait(app, lambda: ctl._lod_view is not None and ctl._lod_view[0] == ctl.current_snap_idx, 120)
    far = ctl._lod_view[3]
    snap = ctl.model.get_snapshot(ctl.current_snap_idx)
    _zoom(ctl, snap['c'][0], 2.0)
    assert _wait(app, lambda: ctl._lod_view[3] is not far, 60)
    assert ctl._lod_view[3].fraction > far.fraction


def test_new_set_waits_until_the_camera_stops(app, ctl, tmp_path):
    """A set that arrives while the camera moves is held, then drawn (or replaced) once it stops."""
    coords, _ = _write_snapshot(tmp_path, n=100_000)
    ctl._set_lod_budget(20_000)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: ctl._lod_view is not None and not ctl._lod_query.isRunning(), 120)
    far = ctl._lod_view[3]
    frames = []
    orig_update = ctl._update_frame
    ctl._update_frame = lambda *a, **k: (frames.append(1), orig_update(*a, **k))
    ctl._lod_timer.stop()                   # drive the polls by hand
    _zoom(ctl, coords[0], 1.0)
    ck = ctl._lod_cam_key(ctl._lod_camera())
    ctl._lod_last_cam = ck
    ctl._lod_force = True
    ctl._lod_poll()
    job = ctl._lod_query
    job.wait()
    o = ctl.view.view_3d.opts               # the camera moves on before the set arrives
    ctl.view.view_3d.setCameraPosition(azimuth=float(o["azimuth"]) + 20.0)
    app.processEvents()
    assert _wait(app, lambda: ctl._lod_pending is not None, 30)
    assert ctl._lod_view[3] is far and not frames          # nothing drawn mid-move
    for _ in range(3):                                      # camera stops: polls settle and query
        ctl._lod_poll()
    assert _wait(app, lambda: ctl._lod_view[3] is not far and not ctl._lod_query.isRunning(), 60)
    assert ctl._lod_pending is None and frames
    assert ctl._lod_view[2] == ctl._lod_cam_key(ctl._lod_camera())
    ctl._update_frame = orig_update
