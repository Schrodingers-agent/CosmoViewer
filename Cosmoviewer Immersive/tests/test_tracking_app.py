"""
Tracking and movies through the real controller and window (Qt offscreen).

Synthetic tests always run. Real-data tests run when COSMOVIEWER_G4_OUTPUT points
at DM-L50-N128/output and COSMOVIEWER_CG_OUTPUT at CollidingGalaxiesSFR/output.
On-screen OpenGL drawing is not checked here: the tests inspect the data handed
to the 3D items and the PNG frames written by the movie maker.
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

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
CG_OUTPUT = os.environ.get("COSMOVIEWER_CG_OUTPUT")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app, monkeypatch):
    # Message boxes would block a headless run.
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
    c._shutdown_lod()                     # stop LOD jobs before Qt objects go away
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


def _load(app, c, folder):
    c._load_directory(Path(folder))
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 300)
    app.processEvents()


def _tracking_done(c):
    return c._track_worker is None or c._track_worker.isFinished()


def _highlight(c):
    p = c.view.scatter_3d_highlight.pos
    return np.empty((0, 3)) if p is None else np.asarray(p)


def _write_moving_run(folder, n_snap=4, n=4000, box=10.0):
    """Particles drifting +1 in x per snapshot in a periodic box; IDs shuffled per file."""
    rng = np.random.default_rng(5)
    x0 = rng.random((n, 3)) * box
    for s in range(n_snap):
        order = rng.permutation(n)
        with h5py.File(Path(folder) / f"snapshot_{s:03d}.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = box
            h.attrs["Time"] = 0.25 * (s + 1)
            h.attrs["Redshift"] = 1 / (0.25 * (s + 1)) - 1
            h.attrs["NumFilesPerSnapshot"] = 1
            h.attrs["MassTable"] = np.array([0, 1.0, 0, 0, 0, 0])
            h.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0])
            h.attrs["NumPart_Total"] = np.array([0, n, 0, 0, 0, 0])
            g = f.create_group("PartType1")
            pos = np.mod(x0 + [s * 1.0, 0, 0], box)
            g["Coordinates"] = pos[order].astype(np.float32)
            g["Velocities"] = np.zeros((n, 3), np.float32)
            g["ParticleIDs"] = (np.arange(n, dtype=np.uint64) + 100)[order]
    return x0


def test_tracking_follows_full_resolution_and_movie(app, ctl, tmp_path):
    x0 = _write_moving_run(tmp_path)
    c = ctl
    c_loader_max = 1000                       # force decimation: 1 in 4 displayed
    from snapshot_io import snapshot_loader as sl
    orig = sl.AsyncSnapshotLoader.__init__

    def small(self, files, max_particles=150_000):
        orig(self, files, c_loader_max)
    sl.AsyncSnapshotLoader.__init__ = small
    try:
        _load(app, c, tmp_path)
    finally:
        sl.AsyncSnapshotLoader.__init__ = orig
    assert len(c.model.snapshots) == 4 and abs(len(c.model.snapshots[0]['c']) - 1000) < 130
    # sampled by ParticleID hash: the same particles in every snapshot (no flicker in movies)
    assert all(set(s['id'].tolist()) == set(c.model.snapshots[0]['id'].tolist()) for s in c.model.snapshots)

    # Select 50 IDs by text, as a user would.
    sp = c.view.selection_panel
    sp.chk_ids.setChecked(True)
    sp.edit_ids.setText("100-149")
    sp.btn_apply.click()
    assert len(c.model.tracker.get_set(0).selected_ids) == 50
    assert _wait(app, lambda: _tracking_done(c))
    app.processEvents()
    for i in range(4):
        assert c._tracked(i, c.model.tracker.get_set(0)).n_found == 50

    # Every selected particle is drawn in every snapshot, not just those in the sample.
    for i in range(4):
        c.view.slide_time.setValue(i)
        app.processEvents()
        hl = _highlight(c)
        assert len(hl) == 50
        want = np.mod(x0[:50] + [i * 1.0, 0, 0], 10.0).astype(np.float32)
        np.testing.assert_allclose(np.sort(hl[:, 1]), np.sort(want[:, 1]), rtol=1e-6)
        in_sample = np.isin(c.model.snapshots[i]['id'], np.arange(100, 150)).sum()
        assert in_sample < 50                               # the old behaviour would have shown these only

    # Follow selection: changing snapshot moves the view centre to the set's (periodic) mean.
    from model.tracking import periodic_mean
    c.view.slide_time.setValue(2)
    app.processEvents()
    want_c = periodic_mean(np.mod(x0[:50] + [2.0, 0, 0], 10.0), 10.0)
    got_c = np.array(c.view.camera_state()['center'])
    d = got_c - want_c
    d -= 10.0 * np.round(d / 10.0)
    assert np.max(np.abs(d)) < 1e-3

    # Trails: one line per particle (up to the limit) plus the centre path.
    sp.spin_trails.setValue(5)
    sp.chk_trails.setChecked(True)
    app.processEvents()
    scene = c._build_scene(2, c._ui_state())
    assert len(scene.lines) == 6
    seg, col = c._lines_to_segments(scene.lines)
    assert len(seg) == 2 * 6 * 2                           # 3 snapshots -> 2 segments per line
    unw = scene.lines[0][0]
    np.testing.assert_allclose(np.diff(unw[:, 0]), 1.0, atol=1e-5)   # no periodic jumps

    # Console: per-snapshot summary and export.
    tv = c.view.telemetry_view
    for line in ("track", f"track export {tmp_path / 'set1.csv'}"):
        tv.input.setText(line)
        tv.input.returnPressed.emit()
        app.processEvents()
    text = tv.log_edit.toPlainText()
    assert "Set 1 'Selection Set 1': 50 IDs" in text and "snapshot_003.hdf5: 50 (type 1: 50)" in text
    assert len(open(tmp_path / "set1.csv").read().splitlines()) == 2 + 4 * 50

    # Clear removes the highlights and the trails.
    sp.btn_clear.click()
    app.processEvents()
    assert len(_highlight(c)) == 0

    # A set changed by other code (as FoF 'highlight halo' does) is looked up too.
    c.model.tracker.get_set(2).select_by_ids(np.arange(200, 220))
    c._update_frame()
    assert _wait(app, lambda: _tracking_done(c) and c._tracked(0, c.model.tracker.get_set(2)) is not None)
    assert c._tracked(3, c.model.tracker.get_set(2)).n_found == 20
    c.model.tracker.get_set(2).clear()

    # Evolving movie centred on a selection, through the real controller.
    sp.edit_ids.setText("100-109")
    sp.btn_apply.click()
    assert _wait(app, lambda: _tracking_done(c))
    from movie.maker import MovieSettings
    s = MovieSettings(kind="evolving", first=0, last=3, n_smooth=1, width=200, height=150,
                      basename=str(tmp_path / "mov" / "f"), draw_label=True)
    c._make_movie(s)
    assert _wait(app, lambda: c._movie_worker.isFinished(), 120)
    app.processEvents()
    meta = json.load(open(tmp_path / "mov" / "f_movie.json"))
    assert len(meta["frames"]) == 4
    for i, fr in enumerate(meta["frames"]):
        want_c = periodic_mean(np.mod(x0[:10] + [i * 1.0, 0, 0], 10.0), 10.0)
        d = np.array(fr["camera"]["center"]) - want_c
        d -= 10.0 * np.round(d / 10.0)
        assert np.max(np.abs(d)) < 1e-3
    assert meta["selection_sets"]["set_1"]["n_ids"] == 10


def test_draw_selected_only_redraws_at_once(app, ctl, tmp_path):
    # The checkbox used to change nothing on screen until something else redrew the view.
    _write_moving_run(tmp_path, n_snap=1)
    c = ctl
    _load(app, c, tmp_path)
    sp = c.view.selection_panel
    sp.chk_ids.setChecked(True)
    sp.edit_ids.setText("100-149")
    sp.btn_apply.click()
    assert _wait(app, lambda: _tracking_done(c), 60)
    app.processEvents()

    def drawn():
        p = c.view.scatter_3d.pos
        return 0 if p is None else len(p)
    assert drawn() > 1000
    sp.chk_draw_only.setChecked(True)
    app.processEvents()
    assert drawn() == 0                       # the set itself is drawn as the highlight
    assert len(_highlight(c)) == 50
    sp.chk_draw_only.setChecked(False)
    app.processEvents()
    assert drawn() > 1000


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_real_dm_halo_tracked_through_app(app, ctl, tmp_path):
    out = Path(G4_OUTPUT)
    c = ctl
    # A decimated display sample (~1 in 13, as with GadgetViewer's default), so the halo
    # is mostly outside it and only the full-resolution lookup can find all of it.
    old = getattr(c, 'max_particles', None)
    c.max_particles = 160_000
    try:
        _load(app, c, out)
    finally:
        c.max_particles = old
    names = [Path(s['filepath']).name for s in c.model.snapshots]
    assert names == ["snapshot_000.hdf5", "snapshot_001.hdf5", "snapshot_002.hdf5"]
    with h5py.File(out / "fof_tab_002.hdf5", "r") as f:
        off, n = int(f["Group/GroupOffsetType"][0, 1]), int(f["Group/GroupLenType"][0, 1])
        gpos = f["Group/GroupPos"][0]
    with h5py.File(out / "snapshot_002.hdf5", "r") as f:
        ids = f["PartType1/ParticleIDs"][off:off + n]
    np.savetxt(tmp_path / "halo0.txt", ids, fmt="%d")

    from model.tracking import read_id_file
    c.view.slide_time.setValue(2)
    app.processEvents()
    c._apply_ids_to_set(c.model.tracker.get_set(0), read_id_file(str(tmp_path / "halo0.txt")), "new")
    t0 = time.time()
    assert _wait(app, lambda: _tracking_done(c), 300)
    lookup_s = time.time() - t0
    app.processEvents()
    sample_hits = np.isin(c.model.snapshots[2]['id'], ids).sum()
    hl = _highlight(c)
    assert len(hl) == n == 34120
    assert sample_hits < n / 5                              # the display sample has ~1 in 13
    c.view.slide_time.setValue(0)                           # z = 3: progenitor particles
    app.processEvents()
    assert len(_highlight(c)) == n
    # Follow: the view centre is the progenitors' periodic mean.
    loc = c._tracked(0, c.model.tracker.get_set(0))
    from model.tracking import periodic_mean
    d = np.array(c.view.camera_state()['center']) - periodic_mean(loc.pos, 50.0)
    d -= 50.0 * np.round(d / 50.0)
    assert np.max(np.abs(d)) < 1e-3

    # Radius selection at full resolution about the halo centre, checked by brute force.
    c.view.slide_time.setValue(2)
    app.processEvents()
    c.view.set_camera_centre(gpos)
    sp = c.view.selection_panel
    sp.active_set_idx = 1
    sp.chk_radius.setChecked(True)
    sp.spin_radius.setValue(0.3)
    sp.chk_full.setChecked(True)
    sp.btn_apply.click()
    with h5py.File(out / "snapshot_002.hdf5", "r") as f:
        pos = f["PartType1/Coordinates"][:].astype(np.float64)
        all_ids = f["PartType1/ParticleIDs"][:]
    dd = pos - gpos
    dd -= 50.0 * np.round(dd / 50.0)
    want = set(all_ids[(dd ** 2).sum(axis=1) <= 0.09].tolist())
    assert c.model.tracker.get_set(1).selected_ids == want and len(want) > 1000

    # Trajectory export of set 1 (the halo) over the three snapshots.
    sp.active_set_idx = 0
    from model.tracking import build_trajectories, export_npz
    located = [c._tracked(i, c.model.tracker.get_set(0)) for i in range(3)]
    tr = build_trajectories(located, [s['filepath'] for s in c.model.snapshots])
    z = np.load(export_npz(tr, str(tmp_path / "halo0.npz")))
    assert z["pos"].shape == (3, n, 3) and np.all(z["types"] == 1)
    print(f"\nDM-L50 halo 0: background lookup of {n:,} IDs in 3 snapshots took {lookup_s:.2f} s")


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_real_sphere_across_a_face_is_one_piece_on_screen(app, ctl):
    # Maverick's report: r = 10 about (38.3, 4.741, 29.13) on DM-L50 snapshot_002 showed a big
    # piece at the y = 0 face and a thin cap at y = 50. Membership is periodic and stays so;
    # only the drawing moves the cap next to the rest.
    out = Path(G4_OUTPUT)
    c = ctl
    _load(app, c, out)
    c.view.slide_time.setValue(2)
    app.processEvents()
    centre = np.array([38.3, 4.741, 29.13])
    c.view.set_camera_centre(centre)
    sp = c.view.selection_panel
    sp.chk_radius.setChecked(True)
    sp.spin_radius.setValue(10.0)
    sp.chk_full.setChecked(True)
    sp.btn_apply.click()
    with h5py.File(out / "snapshot_002.hdf5", "r") as f:
        pos = f["PartType1/Coordinates"][:].astype(np.float64)
        all_ids = f["PartType1/ParticleIDs"][:]
    d = pos - centre
    d -= 50.0 * np.round(d / 50.0)
    inside = (d ** 2).sum(axis=1) <= 100.0
    s_set = c.model.tracker.get_set(0)
    assert s_set.selected_ids == set(all_ids[inside].tolist()) and len(s_set.selected_ids) == 39135
    assert pos[inside, 1].max() > 44.7                     # stored members include the wrapped cap
    assert _wait(app, lambda: _tracking_done(c), 300)
    app.processEvents()
    hl = _highlight(c)
    assert len(hl) == 39135
    assert hl[:, 1].min() > -5.3 and hl[:, 1].max() < 14.8  # drawn as one ball through y = 0
    assert np.linalg.norm(hl - centre, axis=1).max() <= 10.0 + 1e-3
    assert "wraps y" in s_set.describe()
    c._update_radius_marker()
    assert "crosses the box edge in y" in sp.lbl_radius_centre.text()
    c._centre_on_box()
    np.testing.assert_allclose(c.view.camera_state()['center'], [25, 25, 25])


@pytest.mark.skipif(not CG_OUTPUT, reason="COSMOVIEWER_CG_OUTPUT not set")
def test_real_colliding_galaxies_movie(app, ctl, tmp_path):
    out = Path(CG_OUTPUT)
    c = ctl
    old = getattr(c, 'max_particles', None)
    c.max_particles = 150_000             # the sample the selection centre below was chosen on
    try:
        _load(app, c, out)
    finally:
        c.max_particles = old
    n_snap = len(c.model.snapshots)
    assert n_snap >= 3
    # Gas within 2 kpc of a gas particle near the middle of one galaxy's gas disk.
    snap = c.model.snapshots[0]
    gas = snap['c'][snap['t'] == 0].astype(np.float64)
    near = gas[np.argsort(np.linalg.norm(gas - np.median(gas, axis=0), axis=1))[:2000]]
    c.view.set_camera_centre(gas[np.argmin(np.linalg.norm(gas - near.mean(axis=0), axis=1))])
    sp = c.view.selection_panel
    for t in range(6):
        c.view.layer_panel.layer_state.set_visible(t, t == 0)
    sp.chk_radius.setChecked(True)
    sp.spin_radius.setValue(2.0)
    sp.btn_apply.click()
    for t in range(6):
        c.view.layer_panel.layer_state.set_visible(t, True)
    s_set = c.model.tracker.get_set(0)
    assert len(s_set.selected_ids) > 100
    assert _wait(app, lambda: _tracking_done(c), 300)
    types = [np.bincount(c._tracked(i, s_set).types[c._tracked(i, s_set).found], minlength=6)
             for i in range(n_snap)]
    from movie.maker import MovieSettings
    s = MovieSettings(kind="evolving", first=0, last=n_snap - 1, n_smooth=2, width=480, height=360,
                      basename=str(tmp_path / "cg" / "frame"))
    sp.chk_trails.setChecked(True)
    c._make_movie(s)
    assert _wait(app, lambda: c._movie_worker.isFinished(), 600)
    app.processEvents()
    meta = json.load(open(tmp_path / "cg" / "frame_movie.json"))
    assert len(meta["frames"]) == n_snap
    print("\nCollidingGalaxies: selected", len(s_set.selected_ids), "gas particles; types per snapshot:",
          [t.tolist() for t in types])
