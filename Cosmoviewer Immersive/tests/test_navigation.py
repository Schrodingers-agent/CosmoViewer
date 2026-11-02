"""3D view navigation: the wheel zooms toward the cursor, Shift-drag and right-drag pan."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent, QVector3D, QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from views import view_extras  # noqa: E402

pytestmark = pytest.mark.skipif(not view_extras.HAS_GL, reason="no OpenGL")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    v = view_extras.CosmoGLView()
    v.resize(800, 500)
    v.setCameraPosition(pos=QVector3D(50, 50, 50), distance=120, elevation=25, azimuth=40)
    yield v
    v.close()


def _screen(v, p):
    m = v.projectionMatrix((0, 0, v.width(), v.height()), (0, 0, v.width(), v.height())) * v.viewMatrix()
    mvp = np.array(m.data(), dtype=np.float64).reshape(4, 4).T
    q = mvp @ np.r_[p, 1.0]
    return np.array([(q[0] / q[3] + 1) * 0.5 * v.width(), (1 - q[1] / q[3]) * 0.5 * v.height()])


def _centre(v):
    c = v.opts["center"]
    return np.array([c.x(), c.y(), c.z()])


@pytest.mark.parametrize("ortho", [False, True])
def test_focal_point_is_under_the_cursor(view, ortho):
    view.ortho = ortho
    for x, y in ((400, 250), (30, 40), (770, 480)):
        np.testing.assert_allclose(_screen(view, view.focal_point(x, y)), [x, y], atol=1e-3)   # float32 matrices: a thousandth of a pixel


@pytest.mark.parametrize("ortho", [False, True])
def test_zoom_keeps_the_point_under_the_cursor(view, ortho):
    view.ortho = ortho
    x, y = 60.0, 420.0                       # near a corner
    p = view.focal_point(x, y)
    gap = np.linalg.norm(_centre(view) - p)
    for _ in range(15):                      # zoom in 200x: the corner point stays put
        view.zoom_at(x, y, 0.7)
    assert view.opts["distance"] == pytest.approx(120 * 0.7 ** 15)
    np.testing.assert_allclose(_screen(view, p), [x, y], atol=0.05)   # float32 centre: drift < 0.05 px
    assert np.linalg.norm(_centre(view) - p) == pytest.approx(gap * 0.7 ** 15, abs=1e-3)  # centre moves to it


def test_wheel_zooms_toward_cursor_and_cmd_wheel_changes_fov(view, app):
    p = view.focal_point(100, 100)
    ev = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 240),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    view.wheelEvent(ev)
    assert view.opts["distance"] < 120
    np.testing.assert_allclose(_screen(view, p), [100, 100], atol=1e-3)
    fov, dist = view.opts["fov"], view.opts["distance"]
    ev = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 240),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
    view.wheelEvent(ev)
    assert view.opts["fov"] < fov and view.opts["distance"] == dist


@pytest.mark.parametrize("button,mods", [(Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier),
                                         (Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)])
def test_drag_pans(view, button, mods):
    az, el = view.opts["azimuth"], view.opts["elevation"]
    c0 = _centre(view)
    press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(400, 250), QPointF(400, 250), button, button, mods)
    view.mousePressEvent(press)
    move = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(450, 250), QPointF(450, 250), Qt.MouseButton.NoButton,
                       button, mods)
    view.mouseMoveEvent(move)
    assert np.linalg.norm(_centre(view) - c0) > 0                     # moved
    assert (view.opts["azimuth"], view.opts["elevation"]) == (az, el)  # did not orbit


def test_plain_drag_still_orbits(view):
    c0, az = _centre(view), view.opts["azimuth"]
    b = Qt.MouseButton.LeftButton
    view.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(400, 250), QPointF(400, 250), b, b,
                                     Qt.KeyboardModifier.NoModifier))
    view.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(450, 250), QPointF(450, 250),
                                    Qt.MouseButton.NoButton, b, Qt.KeyboardModifier.NoModifier))
    assert view.opts["azimuth"] != az
    np.testing.assert_allclose(_centre(view), c0)
