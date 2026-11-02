"""
View tools and two colouring modes:
scale bar and time overlay, Jump to coordinates, sessions, orthographic view,
labelled points, simulation summary, anaglyph, the 2D panel corner toggle,
Lagrangian-origin colouring and brightness from local density.
"""

import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QImage, QMouseEvent  # noqa: E402

from test_gadgetviewer_features import _FINISHED, _load, _wait, app, ctl, write_snapshot  # noqa: E402,F401
from views import view_extras as VX  # noqa: E402
from controller import view_tools_controller as VT  # noqa: E402
from views.colouring_panel import LAGRANGE  # noqa: E402


# ----------------------------------------------------------------------------- pure functions

def test_nice_length_and_scale_bar_units():
    assert VX.nice_length(7.3) == 5.0 and VX.nice_length(1.9) == 1.0 and VX.nice_length(0.024) == 0.02
    assert VX.nice_length(0) == 0.0
    # code units: 0.2 * 1000 px at 0.01 per px -> target 2.0 -> bar of 2 = 200 px
    px, text = VX.scale_bar_for(2.0, 0.01, None)
    assert px == pytest.approx(200) and text == "2 (code units)"
    # 1 code length = 1 Mpc (1000 kpc), cosmological: target 2.0 code = 2000 kpc -> "2 cMpc/h"
    px, text = VX.scale_bar_for(2.0, 0.01, (1000.0, True, True))
    assert text == "2 cMpc/h" and px == pytest.approx(200)
    # 1 code length = 1 kpc: target 0.5 -> 500 pc
    px, text = VX.scale_bar_for(0.5, 0.001, (1.0, False, False))
    assert text == "500 pc" and px == pytest.approx(500)
    # units per pixel at the centre: 2 d tan(fov/2) / width
    assert VX.units_per_pixel({'distance': 10.0, 'fov': 90.0}, 200) == pytest.approx(0.1)


def test_is_cosmological_and_read_labels(tmp_path):
    assert VT.is_cosmological({'Time': 0.5, 'Redshift': 1.0})
    assert not VT.is_cosmological({'Time': 1.2, 'Redshift': 0.0})
    assert not VT.is_cosmological({'Time': 0.5})
    f = tmp_path / "labels.txt"
    f.write_text("# halo centres\nx y z name\n1, 2, 3, Halo A\n4 5 6\n7\t8\t9 big one  # comment\n")
    pos, names = VT.read_labels(str(f))
    np.testing.assert_allclose(pos, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert names == ["Halo A", "", "big one"]
    (tmp_path / "bad.txt").write_text("1 2\n")
    with pytest.raises(ValueError):
        VT.read_labels(str(tmp_path / "bad.txt"))


# ----------------------------------------------------------------------------- window pieces

def test_orthographic_keeps_framing_at_centre(app, ctl):
    v = ctl.view
    v.resize(1000, 700)
    v.view_3d.resize(800, 600)
    from PyQt6.QtGui import QVector3D
    v.view_3d.setCameraPosition(pos=QVector3D(5, 5, 5), distance=20, elevation=30, azimuth=40)
    cam = v.camera_state()
    right = VX.eye_offset(cam, 1.0) / cam['distance']          # unit screen-right vector
    pts = np.array([cam['center'], np.array(cam['center']) + 3.0 * right])
    xp, yp, _, _ = v.screen_coords(pts)
    v.extras.set_ortho(True)
    assert v.view_3d.ortho and v.extras.act_ortho.isChecked()
    xo, yo, _, front = v.screen_coords(pts)
    assert front.all()
    # the centre stays put and a length at the centre has the same size on screen
    assert xo[0] == pytest.approx(xp[0], abs=1e-3) and yo[0] == pytest.approx(yp[0], abs=1e-3)
    assert (xo[1] - xo[0]) == pytest.approx(xp[1] - xp[0], rel=1e-4)
    # ...and matches the scale bar's units per pixel
    upp = VX.units_per_pixel(cam, v.view_3d.width())
    assert (xo[1] - xo[0]) == pytest.approx(3.0 / upp, rel=1e-4)
    # nearer points do not grow in ortho: a point towards the camera projects like the centre
    toward = np.array(cam['center']) + 5.0 * ctl._view_direction(cam)
    xt, yt, _, _ = v.screen_coords(np.array([toward]))
    assert xt[0] == pytest.approx(xo[0], abs=1e-3) and yt[0] == pytest.approx(yo[0], abs=1e-3)
    v.extras.set_ortho(False)
    assert not v.view_3d.ortho


def test_anaglyph_and_side_by_side_exclude_each_other(app, ctl):
    v = ctl.view
    v.extras.act_anaglyph.setChecked(True)
    assert v.view_3d.anaglyph and not v.stereo_on and not v.stereo_bar.isHidden()
    v.act_stereo.setChecked(True)
    assert v.stereo_on and not v.view_3d.anaglyph and not v.extras.act_anaglyph.isChecked()
    v.act_stereo.setChecked(False)
    v.extras.act_anaglyph.setChecked(True)
    v.slide_eye.setValue(80)
    assert v.view_3d.eye_separation() == pytest.approx(0.08)
    v.extras.act_anaglyph.setChecked(False)
    assert v.stereo_bar.isHidden()


def _click(button, pos, release_pos=None):
    from PyQt6.QtWidgets import QApplication
    g = button.mapToGlobal(pos.toPoint())
    for t, p in ((QMouseEvent.Type.MouseButtonPress, pos), (QMouseEvent.Type.MouseMove, release_pos or pos),
                 (QMouseEvent.Type.MouseButtonRelease, release_pos or pos)):
        gp = button.mapToGlobal((p).toPoint())
        ev = QMouseEvent(t, p, QPointF(gp), Qt.MouseButton.LeftButton,
                         Qt.MouseButton.LeftButton if t != QMouseEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(button, ev)


def test_panel_toggle_click_hides_and_drag_sets_height(app, ctl):
    v = ctl.view
    v.resize(1200, 900)
    v.show()
    app.processEvents()
    t, panel = v.extras.toggle, v.panel2d_widget
    assert t is not None and t.parent() is v.view_3d
    assert v.view_3d.width() - 40 < t.x() and t.x() + t.width() <= v.view_3d.width() and t.y() < 20   # top right
    _click(t, QPointF(10, 10))
    assert panel.isHidden() and not v.extras.act_panel.isChecked()
    _click(t, QPointF(10, 10))
    assert not panel.isHidden() and v.extras.act_panel.isChecked()
    t.set_height(300)
    h0 = v.view_split.sizes()[1]
    assert h0 == pytest.approx(300, abs=3)
    _click(t, QPointF(10, 10), QPointF(10, -90))          # drag up by 100 px: 100 px taller, not toggled
    assert not panel.isHidden()
    assert v.view_split.sizes()[1] == pytest.approx(h0 + 100, abs=3)
    # hidden by the user stays hidden after full screen
    v.extras.act_panel.setChecked(False)
    v.act_fullscreen.setChecked(True)
    v.act_fullscreen.setChecked(False)
    assert panel.isHidden() and not t.isHidden()
    v.extras.act_panel.setChecked(True)
    assert not panel.isHidden()
    v.hide()


def test_overlay_paints_and_burns_into_screenshot(app, ctl):
    v = ctl.view
    v.view_3d.resize(400, 300)
    ov = v.extras.overlay
    v.extras.set_overlay_info("a = 0.5000   z = 1.000", (1000.0, True, True))
    v.extras.set_labels(np.array([v.camera_state()['center']]), ["centre"])
    bar = ov.scale_bar(400)
    assert bar is not None and bar[1].endswith("cMpc/h") and 0 < bar[0] <= 80 + 1e-6
    img = QImage(800, 600, QImage.Format.Format_RGB32)
    img.fill(0)
    out = v.extras.compose_screenshot(img)
    arr = np.frombuffer(out.constBits().asstring(out.sizeInBytes()), dtype=np.uint8).reshape(600, 800, 4)[:, :, :3]
    assert arr[560:, :400].max() > 100                 # scale bar and its label, bottom left (2x scale)
    assert arr[560:, 400:].max() > 100                 # time, bottom right
    assert arr[250:350, 380:480].max() > 100           # the label at the view centre
    ov.show_scale = ov.show_time = ov.show_labels = False
    img.fill(0)
    out = v.extras.compose_screenshot(img)
    arr = np.frombuffer(out.constBits().asstring(out.sizeInBytes()), dtype=np.uint8).reshape(600, 800, 4)
    assert arr[:, :, :3].max() == 0


# ----------------------------------------------------------------------------- with data

def _two_snapshots(tmp_path, **kw):
    for k in (0, 1):
        write_snapshot(tmp_path / f"snapshot_00{k}.hdf5", seed=k, **kw)
        with h5py.File(tmp_path / f"snapshot_00{k}.hdf5", 'r+') as f:
            f['Header'].attrs['Time'] = 0.5 + 0.5 * k
            f['Header'].attrs['Redshift'] = 1.0 / (0.5 + 0.5 * k) - 1.0
    return [tmp_path / f"snapshot_00{k}.hdf5" for k in (0, 1)]


def test_overlay_info_follows_snapshot_and_jump(app, ctl, tmp_path):
    _two_snapshots(tmp_path)
    _load(app, ctl, tmp_path)
    ov = ctl.view.extras.overlay
    assert "z = 1.000" in ov.time_text
    u = ctl._length_unit(ctl.model.get_snapshot(0))
    assert u is not None and u[0] == pytest.approx(1000.0, rel=1e-3) and u[1] and u[2]   # 1 Mpc code length
    ctl.view.set_time_slider_value(1)
    ctl._on_frame_slider_changed(1)
    assert "z = 0.000" in ov.time_text
    ctl.jump_to(1.0, 2.0, 3.0, 7.5)
    cam = ctl.view.camera_state()
    assert cam['center'] == pytest.approx((1.0, 2.0, 3.0)) and cam['distance'] == pytest.approx(7.5)
    ctl.jump_to(4.0, 5.0, 6.0)
    assert ctl.view.camera_state()['distance'] == pytest.approx(7.5)


def test_summary_lists_header_and_properties(app, ctl, tmp_path):
    _two_snapshots(tmp_path)
    _load(app, ctl, tmp_path)
    text = ctl.summary_text()
    for s in ("Snapshots loaded: 2", "BoxSize = 10", "MassTable", "Density", "Cosmological: yes",
              "snapshot_000.hdf5", "snapshot_001.hdf5"):
        assert s in text, s


def test_labels_file_through_controller(app, ctl, tmp_path):
    f = tmp_path / "pts.txt"
    f.write_text("1 1 1 A\n2 2 2 B\n")
    assert ctl.load_labels(str(f))
    np.testing.assert_allclose(ctl.view.extras.overlay.label_pos, [[1, 1, 1], [2, 2, 2]])
    assert ctl.view.extras.overlay.label_names == ["A", "B"]
    assert not ctl.load_labels(str(tmp_path / "missing.txt"))


def test_session_round_trip(app, ctl, tmp_path):
    from controller.app_controller import CosmoViewerController
    files = _two_snapshots(tmp_path)
    _load(app, ctl, tmp_path)
    c, v, cp, sp = ctl, ctl.view, ctl.view.colouring_panel, ctl.view.selection_panel
    c.view.set_time_slider_value(1)
    c._on_frame_slider_changed(1)
    assert cp.select_property('Density')
    cp.chk_log.setChecked(True)
    cp.set_range(0.01, 50.0)
    cp.chk_density.setChecked(True)
    cp.sliders[('lit', 'size')].setValue(7)
    cp.chk_unlit.setChecked(False)
    v.layer_panel.layer_state.set_visible(1, False)
    s0 = c.model.tracker.get_set(0)
    s0.name = "my halo"
    s0.select_by_ids(np.array([1, 2, 3, 10_000_005], dtype=np.uint64))
    c.model.tracker.get_set(2).select_by_ids(np.array([7], dtype=np.uint64))
    sp.chk_trails.setChecked(True)
    (tmp_path / "pts.txt").write_text("1 2 3 centre\n")
    c.load_labels(str(tmp_path / "pts.txt"))
    v.extras.set_ortho(True)
    v.extras.act_scale.setChecked(False)
    v.combo_2d.setCurrentIndex(1)
    c.jump_to(3.0, 4.0, 5.0, 12.0)
    path = tmp_path / "s.json"
    assert c.save_session(str(path))
    st = json.loads(path.read_text())
    assert st['format'] == VT.SESSION_FORMAT and sorted(st['files']) == sorted(str(f) for f in files)

    c2 = CosmoViewerController()
    try:
        assert c2.open_session(str(path))
        assert _wait(app, lambda: not c2.loading and c2.model.snapshots and c2._session_pending is None, 300)
        v2, cp2, sp2 = c2.view, c2.view.colouring_panel, c2.view.selection_panel
        assert c2.current_snap_idx == 1 and v2.slide_time.value() == 1
        st2 = cp2.state()
        assert st2['property'] == 'Density' and st2['log'] and st2['range'] == pytest.approx((0.01, 50.0))
        assert st2['density_brightness']
        dr = cp2.drawing_state()
        assert dr['lit']['size'] == 7 and not dr['show_unlit']
        assert not v2.layer_panel.layer_state.is_visible(1) and v2.layer_panel.layer_state.is_visible(0)
        assert c2.model.tracker.get_set(0).name == "my halo"
        assert c2.model.tracker.get_set(0).selected_ids == {1, 2, 3, 10_000_005}
        assert c2.model.tracker.get_set(2).selected_ids == {7}
        assert sp2.chk_trails.isChecked()
        np.testing.assert_allclose(v2.extras.overlay.label_pos, [[1, 2, 3]])
        assert v2.view_3d.ortho and not v2.extras.act_scale.isChecked() and v2.combo_2d.currentIndex() == 1
        cam = v2.camera_state()
        assert cam['center'] == pytest.approx((3.0, 4.0, 5.0)) and cam['distance'] == pytest.approx(12.0)
        # the restored sets are tracked through the snapshots
        assert _wait(app, lambda: all(c2._tracked(i, c2.model.tracker.get_set(0)) is not None for i in range(2)), 60)
        # opening it again with the same snapshots does not reload them
        assert c2.open_session(str(path)) and not c2.loading
    finally:
        c2._reset_tracking()
        c2._shutdown_lod()
        c2.view.close()
        _FINISHED.append(c2)
    # a session whose files are gone is refused with a message
    st['files'] = [str(tmp_path / "gone.hdf5")]
    (tmp_path / "bad.json").write_text(json.dumps(st))
    assert not c.open_session(str(tmp_path / "bad.json"))
    (tmp_path / "junk.json").write_text("{}")
    assert not c.open_session(str(tmp_path / "junk.json"))


def test_lagrangian_origin_colours(app, ctl, tmp_path):
    _two_snapshots(tmp_path, n_gas=2000, n_dm=3000)
    _load(app, ctl, tmp_path)
    c, cp = ctl, ctl.view.colouring_panel
    assert cp.select_property(LAGRANGE)
    assert not cp.combo_table.isEnabled()
    c.view.set_time_slider_value(1)
    c._on_frame_slider_changed(1)
    pos, ids, types, vals = c._pick_candidates
    with h5py.File(tmp_path / "snapshot_000.hdf5") as f:
        p0 = {}
        for t in (0, 1):
            g = f[f'PartType{t}']
            p0.update(zip(g['ParticleIDs'][:].tolist(), g['Coordinates'][:].astype(np.float64)))
    col = c.view.scatter_3d_lit.color
    exp = np.array([0.2 + 0.8 * np.clip(p0[int(i)] / 10.0, 0, 1) for i in ids])
    np.testing.assert_allclose(col[:, :3], exp, atol=1e-5)
    d = np.array([p0[int(i)] for i in ids]) - 5.0
    np.testing.assert_allclose(vals, np.sqrt((d ** 2).sum(1)), rtol=1e-6)
    # every drawn particle exists in the first snapshot, so all are lit
    snap = c.model.get_snapshot(1)
    assert len(ids) == len(snap['id'])
    # the first snapshot's own particles keep their own position
    c.view.set_time_slider_value(0)
    c._on_frame_slider_changed(0)
    pos0, ids0, _, _ = c._pick_candidates
    exp0 = 0.2 + 0.8 * np.clip(pos0 / 10.0, 0, 1)
    np.testing.assert_allclose(c.view.scatter_3d_lit.color[:, :3], exp0, atol=1e-5)


def test_density_brightness_keeps_hue(app, ctl, tmp_path):
    write_snapshot(tmp_path / "snapshot_000.hdf5", n_gas=3000, n_dm=3000)
    _load(app, ctl, tmp_path)
    c, cp = ctl, ctl.view.colouring_panel
    assert cp.select_property('Density')
    c._update_frame()
    plain = np.array(c.view.scatter_3d_lit.color, copy=True)
    cp.chk_density.setChecked(True)
    c._update_frame()
    lit = np.array(c.view.scatter_3d_lit.color, copy=True)
    assert lit.shape == plain.shape
    b = lit[:, :3].max(1) / np.maximum(plain[:, :3].max(1), 1e-9)
    assert b.min() >= 0.06 - 1e-4 and b.max() <= 1 + 1e-4 and b.std() > 0.05
    # same hue: colour divided by its brightness is unchanged
    np.testing.assert_allclose(lit[:, :3], plain[:, :3] * b[:, None], atol=2e-3)
    np.testing.assert_allclose(lit[:, 3], plain[:, 3])
    # brighter where denser (rank correlation with the kNN density)
    snap = c.model.get_snapshot(0)
    rho = c._knn_density(snap, 10.0, wait=True)[snap['t'] == 0]
    r1, r2 = np.argsort(np.argsort(rho)), np.argsort(np.argsort(b))
    assert np.corrcoef(r1, r2)[0, 1] > 0.95


# ----------------------------------------------------------------------------- cost of a frame

def test_scene_uses_arrays_without_copies_and_float32(app, ctl, tmp_path):
    write_snapshot(tmp_path / "snapshot_000.hdf5", n_gas=3000, n_dm=5000)
    _load(app, ctl, tmp_path)
    c = ctl
    snap = c._display_snapshot(0)
    sc = c._build_scene(0, c._ui_state())
    assert np.shares_memory(sc.pos, snap['c'])           # every type visible: no copy
    assert sc.colors.dtype == np.float32 and len(sc.colors) == len(snap['c'])
    np.testing.assert_allclose(sc.colors, np.asarray(c.TYPE_COLORS, dtype=np.float32)[snap['t']])
    c.view.layer_panel.layer_state.set_visible(1, False)
    sc = c._build_scene(0, c._ui_state())
    assert not np.shares_memory(sc.pos, snap['c']) and len(sc.pos) == int((snap['t'] != 1).sum())
    c.view.layer_panel.layer_state.set_visible(1, True)
    for mode in ('velocity', 'unknown'):
        ui = dict(c._ui_state(), color_mode=mode)
        assert c._build_scene(0, ui).colors.dtype == np.float32
    assert c.view.colouring_panel.select_property('Density')
    sc = c._build_scene(0, c._ui_state())
    assert sc.lit_colors.dtype == np.float32 and sc.colors.dtype == np.float32


def test_z_slice_redrawn_only_when_it_changes(app, ctl, tmp_path, monkeypatch):
    write_snapshot(tmp_path / "snapshot_000.hdf5", n_gas=3000, n_dm=5000)
    _load(app, ctl, tmp_path)
    c, v = ctl, ctl.view
    calls = []
    orig = v.update_scatter_2d
    monkeypatch.setattr(v, 'update_scatter_2d', lambda x, y, b: (calls.append(len(x)), orig(x, y, b)))
    c._slice_key = None
    c._update_frame()
    c._update_frame()                    # same snapshot and slider (e.g. a level-of-detail swap)
    assert len(calls) == 1
    v.slide_z.setValue(20)               # the slider redraws
    assert len(calls) == 2
    v.extras.act_panel.setChecked(False)
    v.slide_z.setValue(80)
    assert len(calls) == 2               # hidden: nothing drawn
    v.extras.act_panel.setChecked(True)  # shown again: drawn at once
    assert len(calls) == 3
    snap = c.model.get_snapshot(0)
    z = snap['c'][:, 2]
    total = float(c.model.g_max[2] - c.model.g_min[2])
    sz = c.model.g_min[2] + 0.8 * total
    assert calls[-1] == int((np.abs(z - sz) < total * 0.015).sum())
    monkeypatch.setattr(type(c), 'SLICE_MAX_POINTS', 50)
    v.slide_z.setValue(50)
    assert calls[-1] == 50


# ----------------------------------------------------------------------------- visibility layer labels

def _layer_labels(panel):
    from PyQt6.QtWidgets import QCheckBox
    return sorted(cb.text() for cb in panel.findChildren(QCheckBox) if cb.parent() is panel)


def test_layer_counts_follow_the_snapshot(app):
    """The VISIBILITY LAYERS counts are the snapshot's own. Found on CollidingGalaxiesSFR: with the same
    types in two snapshots the labels kept the first one's counts (stars stuck at 169,200 while they grew
    to 551,462), and a rebuild left the old checkbox under the new one until a later deferred delete."""
    from gadget_file_discovery import ParticleTypeInfo as P
    from views.layer_panel import DynamicLayerPanel
    p = DynamicLayerPanel()
    p.rebuild_for_snapshot([P(0, 'Gas', 600000, True), P(1, 'Dark Matter', 600000, True), P(4, 'Star Particles', 0, False)])
    p.rebuild_for_snapshot([P(0, 'Gas', 430800, True), P(1, 'Dark Matter', 600000, True), P(4, 'Star Particles', 169200, True)])
    # no event loop between: the old checkboxes must be gone already
    assert _layer_labels(p) == ['Dark Matter (600,000)', 'Gas (430,800)', 'Star Particles (169,200)']
    p.checkboxes[1].setChecked(False)
    p.rebuild_for_snapshot([P(0, 'Gas', 329257, True), P(1, 'Dark Matter', 600000, True), P(4, 'Star Particles', 270743, True)])
    assert _layer_labels(p) == ['Dark Matter (600,000)', 'Gas (329,257)', 'Star Particles (270,743)']
    assert not p.checkboxes[1].isChecked() and not p.layer_state.is_visible(1)   # a hidden type stays hidden


def test_layer_counts_follow_the_slider(app, ctl, tmp_path):
    write_snapshot(tmp_path / "snapshot_000.hdf5", n_gas=3000, n_dm=5000, seed=1)
    write_snapshot(tmp_path / "snapshot_001.hdf5", n_gas=2000, n_dm=5000, seed=2)
    _load(app, ctl, tmp_path)
    assert _wait(app, lambda: len(ctl.model.snapshots) == 2, 300)
    lp = ctl.view.layer_panel
    gas = {"snapshot_000.hdf5": 3000, "snapshot_001.hdf5": 2000}    # both at Time 1.0, so either order
    for i in (1, 0, 1):
        n_gas = gas[Path(ctl.model.snapshots[i]['filepath']).name]
        ctl.view.slide_time.setValue(i)
        assert _wait(app, lambda: ctl.current_snap_idx == i and f"({n_gas:,})" in lp.checkboxes[0].text(), 60), \
            _layer_labels(lp)
        assert _layer_labels(lp) == sorted([lp.checkboxes[0].text(), lp.checkboxes[1].text()])


def test_startup_shows_layers_and_colouring_only(app, ctl):
    """Asked for: every start opens on Visibility Layers and Colouring, Particle Selection and the inspector
    folded, whatever was open when the app was last closed (the sections no longer read the remembered
    state, so it is not written here: that would change the real settings of whoever runs the tests)."""
    from views.main_window import CosmoViewerWindow
    v = CosmoViewerWindow(ctl.model.tracker)
    try:
        assert v.colouring_section.is_expanded() and not v.colouring_panel.isHidden()
        assert not v.selection_section.is_expanded() and v.selection_panel.isHidden()
        assert not v.inspector_section.is_expanded() and v.inspector_panel.isHidden()
        assert not v.layer_panel.isHidden()
    finally:
        v.close()
    ctl.inspect_particle(1)                                  # a pick still unfolds the inspector
    assert ctl.view.inspector_section.is_expanded()


def test_follow_camera_label_not_under_render_button(app, ctl):
    """The macOS style put the Render button 9 px over the end of "Follow camera" ("Follow camer")."""
    v = ctl.view
    v.resize(1400, 900)
    v.show()
    app.processEvents()
    assert v.chk_2d_follow.geometry().right() < v.btn_2d_render.geometry().left()
