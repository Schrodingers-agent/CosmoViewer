"""
Smoothed 2D panel: zooming redraws the part on screen in detail, and images of
small snapshots use every particle (a 1-in-10 sample turns sparse gas into blobs).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from model.sph_projection import knn_smoothing_length, project  # noqa: E402
from test_gadgetviewer_features import _FINISHED, _wait, app, ctl, write_snapshot  # noqa: E402,F401

CAM = {'center': (0.1, 0.0, 0.0), 'distance': 5.0, 'fov': 60.0, 'elevation': 30.0, 'azimuth': 40.0}


def _blob(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, 3)), rng.uniform(1, 2, n), rng.uniform(0.01, 0.5, n), rng.uniform(1, 5, n))


def test_zoom_region_matches_the_same_part_of_a_fine_picture():
    pos, m, h, _ = _blob()
    W = 2 * CAM['distance'] * np.tan(np.radians(30.0))
    fine, _ = project(pos, m, h, CAM, 800)
    block = fine[400:600, 200:400]                       # rows = up, columns = right
    z, wz = project(pos, m, h, CAM, 200, region=(-W / 2 + W / 4, -W / 2 + W / 2, 0.0, W / 4))
    assert z.shape == (200, 200) and wz == pytest.approx(W / 4)
    d = np.abs(z - block) / block.max()
    assert np.median(d) < 1e-3 and d.max() < 0.03         # edges continue outward, no dark border
    assert z.sum() == pytest.approx(block.sum(), rel=0.01)
    r, wr = project(pos, m, h, CAM, 300, region=(-1.0, 2.0, -0.5, 0.5))
    assert r.shape == (100, 300) and wr == pytest.approx(3.0)   # square pixels


def test_whole_view_unchanged_by_region_argument():
    pos, m, h, q = _blob()
    W = 2 * CAM['distance'] * np.tan(np.radians(30.0))
    a, _ = project(pos, m, h, CAM, 200, weights=q)
    b, _ = project(pos, m, h, CAM, 200, weights=q, region=(-W / 2, W / 2, -W / 2, W / 2))
    # The same picture; only the outermost pixels differ (a zoom's edges continue outward).
    np.testing.assert_allclose(b[16:-16, 16:-16], a[16:-16, 16:-16], rtol=0.02, atol=0)


def test_sample_of_sparse_gas_is_blobby_and_every_particle_is_not():
    # Uniform gas with a smooth temperature gradient; smoothing lengths reach the
    # 32nd neighbour of the whole set, as in a snapshot file.
    rng = np.random.default_rng(3)
    n = 40000
    pos = rng.uniform(-1, 1, (n, 3))
    temp = 1e4 * (2.0 + pos[:, 0])
    h = knn_smoothing_length(pos, 32)
    m = np.ones(n)
    cam = {'center': (0, 0, 0), 'distance': 0.6 / np.tan(np.radians(30.0)), 'fov': 60.0,
           'elevation': 90.0, 'azimuth': -90.0}      # from above, screen right = +x
    full, w = project(pos, m, h, cam, 200, weights=temp)
    keep = rng.random(n) < 0.1
    samp, _ = project(pos[keep], m[keep] * 10, h[keep], cam, 200, weights=temp[keep])
    x = (np.arange(200) + 0.5) / 200 * w - w / 2
    truth = 1e4 * (2.0 + x)[None, :]                     # columns = screen right = +x seen from above
    err_full = np.nanstd((full - truth) / truth)
    err_samp = np.nanstd((samp - truth) / truth)
    assert err_samp > 2 * err_full


def _decimated(monkeypatch):
    import lod.budget
    monkeypatch.setattr(lod.budget, 'MIN_BASE_SAMPLE', 1000)


def _reload(app, ctl, folder):
    ctl.max_particles = 1500                             # a decimated display sample
    ctl._load_directory(folder)
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots, 300)
    ctl.view.combo_2d.setCurrentIndex(0)


def test_panel2d_in_the_window(app, ctl, tmp_path, monkeypatch):
    # One window for all of it: every window a test makes adds to the test-run crash
    # in pyqtgraph's PlotWidget construction.
    import controller.features_controller as fc
    _decimated(monkeypatch)
    write_snapshot(tmp_path / "snapshot_000.hdf5", n_gas=6000, n_dm=6000)
    _reload(app, ctl, tmp_path)
    _every_particle_and_zoom(app, ctl)
    monkeypatch.setattr(fc, 'PANEL2D_FULL_MAX', 5000)
    _reload(app, ctl, tmp_path)
    _big_snapshot_keeps_the_display_sample(app, ctl)
    monkeypatch.setattr(fc, 'PANEL2D_FULL_MAX', 2_000_000)
    monkeypatch.setattr(fc, 'PANEL2D_INLINE', 0)
    _reload(app, ctl, tmp_path)
    _drawn_off_the_window_thread(app, ctl)


def _every_particle_and_zoom(app, ctl):
    snap = ctl.model.get_snapshot(0)
    assert len(snap['c']) < 12000
    v = ctl.view
    assert v.extras.panel_wanted
    v.combo_2d.setCurrentIndex(1)
    app.processEvents()
    assert v.image_2d.isVisible()
    assert "every particle" in v.plot_2d.plotItem.titleLabel.text
    full = ctl._panel2d_full[1]
    assert len(full['c']) == 12000

    # Nothing changed: no redraw (a level-of-detail update must not reset a zoom)
    img_before = v.image_2d.image
    ctl._render_panel2d()
    assert v.image_2d.image is img_before

    # Zoom in to a quarter of the picture: the part on screen is drawn again, finer
    vb = v.plot_2d.getViewBox()
    (x0, x1), (y0, y1) = vb.viewRange()
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 8
    vb.setRange(xRange=(cx - half, cx + half), yRange=(cy - half, cy + half), padding=0)
    (zx0, zx1), _ = vb.viewRange()                       # the aspect lock may widen it
    ctl._on_panel2d_zoomed()
    r = v.image_2d.mapRectToParent(v.image_2d.boundingRect())
    assert r.width() == pytest.approx(1.5 * (zx1 - zx0))  # what is on screen plus a quarter each side
    assert r.width() < x1 - x0
    assert v.image_2d.image.shape[0] >= 200               # drawn again at screen resolution
    assert ctl._panel2d_region is not None
    zoom_img = v.image_2d.image
    ctl._render_panel2d()                                # frame update, same camera and settings
    assert v.image_2d.image is zoom_img

    # Render button: back to the whole view
    v.btn_2d_render.click()
    assert ctl._panel2d_region is None
    cam = v.camera_state()
    r = v.image_2d.mapRectToParent(v.image_2d.boundingRect())
    assert r.width() == pytest.approx(2 * cam['distance'] * np.tan(np.radians(cam['fov']) / 2))

    # Colour by temperature: the zoom is kept and drawn with the new colours
    vb.setRange(xRange=(cx - half, cx + half), yRange=(cy - half, cy + half), padding=0)
    ctl._on_panel2d_zoomed()
    cp = v.colouring_panel
    assert cp.select_property('Temperature')
    v.combo_2d.setCurrentIndex(2)
    app.processEvents()
    assert "Temperature" in v.plot_2d.plotItem.titleLabel.text


def _big_snapshot_keeps_the_display_sample(app, ctl):
    ctl.view.combo_2d.setCurrentIndex(1)
    app.processEvents()
    # The z-slice title also says "display sample" now: check that the smoothed image is shown.
    title = ctl.view.plot_2d.plotItem.titleLabel.text
    assert title.startswith("Smoothed") and "display sample" in title
    assert ctl._panel2d_full is None


def _drawn_off_the_window_thread(app, ctl):
    v = ctl.view
    v.combo_2d.setCurrentIndex(1)
    assert ctl._panel2d_job is not None                  # drawing; the window carries on
    ctl._rerender_panel2d()                              # asked again meanwhile: drawn next
    assert ctl._panel2d_pending is not None
    assert _wait(app, lambda: ctl._panel2d_job is None and ctl._panel2d_pending is None, 60)
    assert v.image_2d.isVisible() and "every particle" in v.plot_2d.plotItem.titleLabel.text
    vb = v.plot_2d.getViewBox()
    (x0, x1), (y0, y1) = vb.viewRange()
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 8
    vb.setRange(xRange=(cx - half, cx + half), yRange=(cy - half, cy + half), padding=0)
    ctl._on_panel2d_zoomed()
    assert _wait(app, lambda: ctl._panel2d_job is None, 60)
    assert ctl._panel2d_region is not None
    r = v.image_2d.mapRectToParent(v.image_2d.boundingRect())
    assert r.width() < x1 - x0
