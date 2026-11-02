"""
Movie maker v2 through the real controller and window (Qt offscreen), on the
Gadget-4 example outputs: in-between frames, smoothed frames, overlays, lazy
reading, keyframes and the Quick tab.
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
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
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


def _load(app, c, folder, max_particles):
    c.max_particles = max_particles
    c._load_directory(Path(folder))
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 300)
    app.processEvents()


def _movie(app, c, s):
    c._make_movie(s)
    assert _wait(app, lambda: c._movie_worker is not None and c._movie_worker.isFinished(), 900)
    app.processEvents()


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_dm_halo_in_between_smoothed_lazy(app, ctl, tmp_path):
    from model.tracking import periodic_mean
    from movie.maker import MovieSettings
    out = Path(G4_OUTPUT)
    c = ctl
    _load(app, c, out, 100_000)                 # small loaded samples; the movie reads 400k per snapshot
    assert all(len(s['c']) < 110_000 for s in c.model.snapshots)   # every 20th particle
    with h5py.File(out / "fof_tab_002.hdf5", "r") as f:
        off, n = int(f["Group/GroupOffsetType"][0, 1]), int(f["Group/GroupLenType"][0, 1])
    with h5py.File(out / "snapshot_002.hdf5", "r") as f:
        ids = f["PartType1/ParticleIDs"][off:off + n]
    c._apply_ids_to_set(c.model.tracker.get_set(0), ids, "new")
    assert _wait(app, lambda: c._track_worker is None or c._track_worker.isFinished(), 300)

    sizes = []
    orig = c._build_scene

    def spy(i, ui, compute=False, snap=None):
        sc = orig(i, ui, compute, snap)
        if compute:
            sizes.append(len(sc.pos) + (0 if sc.lit_pos is None else len(sc.lit_pos)))
        return sc
    c._build_scene = spy
    s = MovieSettings(kind="evolving", first=0, last=2, interp_frames=3, n_smooth=0, width=320, height=240,
                      style="smoothed", scale_bar=True, scale_units="physical", colour_bar=True,
                      lazy=True, lazy_particles=400_000, basename=str(tmp_path / "dm" / "frame"))
    t0 = time.time()
    _movie(app, c, s)
    seconds = time.time() - t0
    meta = json.load(open(tmp_path / "dm" / "frame_movie.json"))
    fr = meta["frames"]
    assert len(fr) == 9 and [f["interpolated"] for f in fr] == [False, True, True, True] * 2 + [False]
    assert meta["particles_dropped_in_between"] == 0 and "lazy" in meta
    # Snapshots were read with about 400k particles (every 5th of 2.1M), not the 100k loaded.
    assert sizes and all(350_000 < k < 450_000 for k in sizes)
    assert meta["smooth_range_used"][1] > meta["smooth_range_used"][0] > 0
    # In-between halo centres lie between the snapshot centres (the halo moves smoothly).
    cen = np.array([f["selection_centre"] for f in fr])
    for a, b in ((0, 4), (4, 8)):
        seg = cen[b] - cen[a]
        seg -= 50.0 * np.round(seg / 50.0)
        for k in range(a + 1, b):
            d = cen[k] - cen[a]
            d -= 50.0 * np.round(d / 50.0)
            t = float(d @ seg / (seg @ seg))
            assert -0.2 < t < 1.2
    # Exact centres at the snapshots: the full-resolution members' periodic mean.
    loc = c._tracked(2, c.model.tracker.get_set(0))
    assert np.allclose(cen[8], periodic_mean(loc.pos.astype(np.float64), 50.0), atol=1e-4)
    print(f"\nDM-L50 9-frame smoothed lazy movie in {seconds:.1f} s; samples {sorted(set(sizes))}")


@pytest.mark.skipif(not CG_OUTPUT, reason="COSMOVIEWER_CG_OUTPUT not set")
def test_cg_quick_gif_and_keyframes(app, ctl, tmp_path):
    from movie.maker import MovieSettings
    c = ctl
    _load(app, c, CG_OUTPUT, 150_000)
    n_snap = len(c.model.snapshots)
    c._open_movie_dialog()
    dlg = c.movie_dialog
    # Quick: play through all snapshots as a GIF.
    dlg.tabs.setCurrentIndex(0)
    dlg.quick_kind.setCurrentIndex(1)
    dlg.quick_seconds.setValue(2)
    dlg.quick_size.setCurrentIndex(0)
    dlg.quick_gif.setChecked(True)
    dlg.quick_file.setText(str(tmp_path / "quick" / "cg.mp4"))
    dlg._sync_quick_suffix()
    assert dlg.quick_file.text().endswith("cg.gif")
    q = dlg.current_settings()
    assert q.kind == "evolving" and q.frame_count(n_snap) >= 2 * 15 - n_snap
    _movie(app, c, q)
    gif = tmp_path / "quick" / "cg.gif"
    assert gif.exists() and sorted(p.name for p in gif.parent.iterdir()) == ["cg.gif"]
    from PIL import Image
    im = Image.open(gif)
    assert im.size == (640, 480) and im.n_frames == q.frame_count(n_snap)

    # Custom: a camera path through two keyframes, from the first to the last snapshot.
    dlg.tabs.setCurrentIndex(1)
    c.view.slide_time.setValue(0)
    app.processEvents()
    dlg.btn_key_add.click()
    cam = c.view.camera_state()
    c.view.set_camera_state({**cam, 'distance': cam['distance'] * 0.5, 'azimuth': cam['azimuth'] + 120})
    c.view.slide_time.setValue(n_snap - 1)
    app.processEvents()
    dlg.btn_key_add.click()
    assert len(dlg.keyframes) == 2 and dlg.radio_path.isChecked()
    assert dlg.keyframes[1].snapshot == n_snap - 1
    assert dlg.keyframes[1].camera['distance'] == pytest.approx(cam['distance'] * 0.5, rel=1e-6)
    # "Show" puts the first keyframe back on screen.
    dlg.list_keys.setCurrentRow(0)
    dlg.btn_key_show.click()
    app.processEvents()
    assert c.current_snap_idx == 0
    assert c.view.camera_state()['distance'] == pytest.approx(cam['distance'], rel=1e-6)
    dlg.spin_path_frames.setValue(12)
    dlg.chk_scale.setChecked(True)
    dlg.edit_base.setText(str(tmp_path / "path" / "frame"))
    s = dlg.current_settings()
    assert s.kind == "keyframes" and s.nframes == 12
    _movie(app, c, s)
    meta = json.load(open(tmp_path / "path" / "frame_movie.json"))
    assert len(meta["frames"]) == 12
    assert meta["frames"][0]["snapshot_index"] == 0 and meta["frames"][-1]["snapshot_index"] == n_snap - 1
    assert meta["frames"][-1]["camera"]["distance"] == pytest.approx(cam['distance'] * 0.5, rel=1e-6)
    assert any(f["interpolated"] for f in meta["frames"])


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_dm_quick_and_custom_mp4(app, ctl, tmp_path):
    from movie.maker import can_write_mp4
    if not can_write_mp4():
        pytest.skip("no MP4 encoder (ffmpeg or OpenCV) installed")
    cv2 = pytest.importorskip("cv2")

    def decoded(path):
        cap = cv2.VideoCapture(str(path))
        n, shape = 0, None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            n, shape = n + 1, img.shape[:2]
        cap.release()
        return n, shape

    c = ctl
    _load(app, c, G4_OUTPUT, 100_000)
    c._open_movie_dialog()
    dlg = c.movie_dialog
    # MP4 is offered, and is the Quick default, whenever an encoder exists.
    assert dlg.quick_mp4.isEnabled() and dlg.quick_mp4.isChecked() and dlg.chk_mp4.isEnabled()
    dlg.tabs.setCurrentIndex(0)
    dlg.quick_kind.setCurrentIndex(0)
    dlg.quick_seconds.setValue(1)
    dlg.quick_size.setCurrentIndex(0)
    dlg.quick_file.setText(str(tmp_path / "quick" / "dm.mp4"))
    q = dlg.current_settings()
    assert q.kind == "rotating" and q.output_file.endswith("dm.mp4")
    _movie(app, c, q)
    mp4 = tmp_path / "quick" / "dm.mp4"
    assert sorted(p.name for p in mp4.parent.iterdir()) == ["dm.mp4"]
    assert decoded(mp4) == (q.nframes, (q.height + q.height % 2, q.width + q.width % 2))

    # Custom: frames kept and an MP4 encoded beside them, at an odd size.
    dlg.tabs.setCurrentIndex(1)
    dlg.radio_rotating.setChecked(True)
    dlg.spin_nframes.setValue(6)
    dlg.spin_w.setValue(321)
    dlg.spin_h.setValue(241)
    dlg.edit_base.setText(str(tmp_path / "custom" / "frame"))
    dlg.chk_mp4.setChecked(True)
    s = dlg.current_settings()
    assert s.encode_mp4 and s.keep_frames
    _movie(app, c, s)
    assert len(list((tmp_path / "custom").glob("frame_*.png"))) == 6
    assert decoded(tmp_path / "custom" / "frame.mp4") == (6, (242, 322))
