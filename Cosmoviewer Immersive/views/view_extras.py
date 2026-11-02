"""
Extras for the 3D view and its window:

- ``CosmoGLView``: the 3D view widget with an orthographic projection toggle and
  red/cyan anaglyph stereo. Anaglyph draws the scene twice per frame, once per
  eye with the camera moved half the eye separation sideways, writing only the
  red channel for the left eye and only green and blue for the right
  (glColorMask; the second clear leaves the red channel alone).
- ``ViewOverlay``: a scale bar, the time or redshift, and labelled points drawn
  over the 3D view, and burnt into screenshots.
- ``PanelToggle``: a button in the top-right corner of the view area that shows
  or hides the 2D panel when clicked and sets its height when dragged.
- ``ViewExtras``: installs the above into the main window with their menu items,
  and carries the signals the controller listens to.

Movie frames are drawn by movie/renderer.py, which has its own camera, so the
orthographic toggle, anaglyph and the overlays apply to the window and to
screenshots, not to movies.
"""

import math
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLineEdit, QPlainTextEdit,
                             QToolButton, QVBoxLayout, QWidget)

try:
    import pyqtgraph.opengl as gl
    from OpenGL import GL
    from pyqtgraph.Qt import QtGui
    HAS_GL = True
except Exception:  # no OpenGL: the window shows a placeholder instead of the 3D view
    HAS_GL = False


def nice_length(target: float) -> float:
    """1, 2 or 5 times a power of ten, at most ``target`` (``target`` > 0)."""
    if not (target > 0 and math.isfinite(target)):
        return 0.0
    p = 10.0 ** math.floor(math.log10(target))
    for m in (5.0, 2.0, 1.0):
        if m * p <= target:
            return m * p
    return p


def units_per_pixel(cam: dict, width_px: int) -> float:
    """Length per screen pixel at the view centre (the same for both projections here)."""
    return 2.0 * cam['distance'] * math.tan(math.radians(cam['fov']) / 2.0) / max(width_px, 1)


def scale_bar_for(target: float, upp: float, unit) -> Optional[tuple]:
    """
    (bar length in pixels, label) for a bar of at most ``target`` code lengths, with
    ``upp`` code lengths per pixel. ``unit`` is None (code units) or (kpc per code
    length, comoving?, per h?): the label then uses pc, kpc or Mpc, as "ckpc/h" etc.
    """
    if not (target > 0 and upp > 0):
        return None
    if unit is None:
        length = nice_length(target)
        return (length / upp, f"{length:g} (code units)") if length > 0 else None
    kpc, comoving, per_h = unit
    t_kpc = target * kpc
    name, per = ('Mpc', 1000.0) if t_kpc >= 1000.0 else (('kpc', 1.0) if t_kpc >= 1.0 else ('pc', 1e-3))
    length = nice_length(t_kpc / per)
    if length <= 0:
        return None
    label = f"{length:g} {'c' if comoving else ''}{name}{'/h' if per_h else ''}"
    return length * per / kpc / upp, label


def eye_offset(cam: dict, separation: float) -> np.ndarray:
    """The screen's right vector times separation × distance (as the side-by-side stereo uses)."""
    az = math.radians(cam['azimuth'])
    return np.array([-math.sin(az), math.cos(az), 0.0]) * separation * cam['distance']


if HAS_GL:
    class CosmoGLView(gl.GLViewWidget):
        """GLViewWidget with ``ortho`` and ``anaglyph`` switches (see the module docstring)."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.ortho = False
            self.anaglyph = False
            self.eye_separation = lambda: 0.05
            # Widgets laid over the view are placed from here, not by event filters: a Python
            # event filter on a parent can be called while Qt is deleting the window, which crashes.
            self.overlays: List[QWidget] = []

        def resizeEvent(self, ev):
            super().resizeEvent(ev)
            for w in self.overlays:
                w.place()

        # Navigation. pyqtgraph zooms toward the orbit centre only, so a corner of the box
        # can't be approached without first panning, and its pan needs a middle button or
        # Cmd-drag. Here the wheel zooms toward the point under the cursor, and Shift-drag
        # or right-drag pans; plain drag still orbits, Cmd-drag and middle-drag still pan.

        def focal_point(self, x: float, y: float) -> np.ndarray:
            """World point under widget pixel (x, y) in the plane through the orbit centre."""
            c = self.opts['center']
            centre = np.array([c.x(), c.y(), c.z()], dtype=np.float64)
            w, h = max(self.width(), 1), max(self.height(), 1)
            half_w = self.opts['distance'] * math.tan(0.5 * math.radians(self.opts['fov']))
            m = np.array(self.viewMatrix().data(), dtype=np.float64).reshape(4, 4).T   # column-major
            right, up = m[0, :3], m[1, :3]          # rows of the rotation: camera axes in world space
            nx, ny = 2.0 * x / w - 1.0, 1.0 - 2.0 * y / h
            return centre + right * nx * half_w + up * ny * half_w * h / w

        def zoom_at(self, x: float, y: float, factor: float):
            """Scale the distance by ``factor`` keeping the point under (x, y) where it is on screen."""
            p = self.focal_point(x, y)
            c = self.opts['center']
            centre = np.array([c.x(), c.y(), c.z()], dtype=np.float64)
            new = p + (centre - p) * factor
            self.opts['center'] = QtGui.QVector3D(float(new[0]), float(new[1]), float(new[2]))
            self.opts['distance'] *= factor
            self.update()

        def wheelEvent(self, ev):
            delta = ev.angleDelta().x() or ev.angleDelta().y()
            if delta == 0 or ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                return super().wheelEvent(ev)       # Cmd-scroll (Qt calls Cmd "Control" on macOS): field of view
            pos = ev.position()
            self.zoom_at(pos.x(), pos.y(), 0.999 ** delta)

        def mouseMoveEvent(self, ev):
            buttons, mods = ev.buttons(), ev.modifiers()
            pan = buttons == Qt.MouseButton.RightButton or (
                buttons == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier)
            if not pan:
                return super().mouseMoveEvent(ev)
            pos = ev.position()
            last = getattr(self, 'mousePos', pos)
            self.mousePos = pos
            d = pos - last
            self.pan(d.x(), d.y(), 0, relative='view')

        def projectionMatrix(self, region=None, viewport=None):
            if region is None:              # older call style: the whole widget
                region = viewport = (0, 0, self.width(), self.height())
            if not self.ortho:
                return super().projectionMatrix(region, viewport)
            x0, y0, w, h = viewport
            dist = self.opts['distance']
            # Same size at the view centre as the perspective view, so switching keeps the framing.
            r = dist * math.tan(0.5 * math.radians(self.opts['fov']))
            t = r * h / w
            left = r * ((region[0] - x0) * (2.0 / w) - 1)
            right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
            bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
            top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)
            tr = QtGui.QMatrix4x4()
            tr.ortho(left, right, bottom, top, -dist * 1000.0, dist * 1000.0)
            return tr

        def paintGL(self):
            if not self.anaglyph:
                return super().paintGL()
            region = self.getViewport()
            c = self.opts['center']
            centre = np.array([c.x(), c.y(), c.z()])
            cam = {'azimuth': self.opts['azimuth'], 'distance': self.opts['distance']}
            half = eye_offset(cam, 0.5 * float(self.eye_separation()))
            try:
                for sign, mask in ((-1.0, (True, False, False, True)), (1.0, (False, True, True, True))):
                    p = centre + sign * half
                    self.opts['center'] = QtGui.QVector3D(float(p[0]), float(p[1]), float(p[2]))
                    GL.glColorMask(*mask)
                    self.paint(region=region, viewport=region)
            finally:
                GL.glColorMask(True, True, True, True)
                self.opts['center'] = c
else:
    CosmoGLView = None


class ViewOverlay(QWidget):
    """Scale bar (bottom left), time (bottom right) and labelled points over the 3D view."""

    def __init__(self, window, view):
        super().__init__(view)
        self.window, self.view = window, view
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.show_scale, self.show_time, self.show_labels = True, True, True
        self.time_text = ""
        # None: code units; else (kpc per code length unit, comoving?, per h?)
        self.unit = None
        self.label_pos = np.empty((0, 3))
        self.label_names: List[str] = []
        self._cam_key = None
        self.setGeometry(view.rect())
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll_camera)
        self._timer.start()

    def place(self):
        self.setGeometry(self.view.rect())

    def _poll_camera(self):
        o = self.view.opts
        c = o['center']
        key = (c.x(), c.y(), c.z(), o['distance'], o['elevation'], o['azimuth'], o['fov'],
               getattr(self.view, 'ortho', False))
        if key != self._cam_key:
            self._cam_key = key
            self.update()

    def set_labels(self, pos: np.ndarray, names: List[str]):
        self.label_pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        self.label_names = list(names)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        try:
            self.paint_into(p, self.width(), self.height())
        finally:
            p.end()

    def scale_bar(self, width_px: int):
        """(bar length in pixels, label) or None."""
        cam = self.window.camera_state()
        if not cam:
            return None
        upp = units_per_pixel(cam, width_px)
        return scale_bar_for(0.2 * width_px * upp, upp, self.unit)

    def paint_into(self, p: QPainter, w: int, h: int):
        """Draw the overlays for a view of w × h widget pixels (also used for screenshots)."""
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(p.font())
        font.setPixelSize(12)
        p.setFont(font)
        white = QColor(230, 230, 230)
        if self.show_labels and len(self.label_pos):
            x, y, _, front = self.window.screen_coords(self.label_pos)
            p.setPen(QPen(QColor(255, 220, 90), 1.5))
            for i in np.flatnonzero(front & np.isfinite(x) & np.isfinite(y)):
                if -50 <= x[i] <= w + 50 and -50 <= y[i] <= h + 50:
                    p.drawEllipse(QPointF(float(x[i]), float(y[i])), 4.0, 4.0)
                    if i < len(self.label_names) and self.label_names[i]:
                        p.drawText(QPointF(float(x[i]) + 7, float(y[i]) - 5), self.label_names[i])
        if self.show_scale:
            bar = self.scale_bar(w)
            if bar is not None:
                px, text = bar
                x0, y0 = 16.0, h - 18.0
                p.setPen(QPen(white, 2))
                p.drawLine(QPointF(x0, y0), QPointF(x0 + px, y0))
                p.drawLine(QPointF(x0, y0 - 4), QPointF(x0, y0 + 4))
                p.drawLine(QPointF(x0 + px, y0 - 4), QPointF(x0 + px, y0 + 4))
                p.drawText(QPointF(x0, y0 - 8), text)
        if self.show_time and self.time_text:
            p.setPen(white)
            r = p.fontMetrics().boundingRect(self.time_text)
            p.drawText(QPointF(w - r.width() - 16, h - 12), self.time_text)

    def paint_onto(self, img, view_w: int):
        """Burn the overlays into a grabbed framebuffer of a view ``view_w`` widget pixels wide."""
        if img is None or img.isNull():
            return img
        s = img.width() / max(view_w, 1)
        p = QPainter(img)
        try:
            p.scale(s, s)
            self.paint_into(p, int(round(img.width() / s)), int(round(img.height() / s)))
        finally:
            p.end()
        return img


def _toggle_icon(shown: bool) -> QIcon:
    """A rectangle with an arrow: down = hide the panel below, up = show it."""
    pm = QPixmap(QSize(36, 36))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(220, 220, 220), 2.5))
    p.drawRoundedRect(QRectF(4, 4, 28, 28), 4, 4)
    p.setBrush(QColor(220, 220, 220))
    if shown:
        tri = [QPointF(11, 14), QPointF(25, 14), QPointF(18, 23)]
    else:
        tri = [QPointF(11, 22), QPointF(25, 22), QPointF(18, 13)]
    p.drawPolygon(QPolygonF(tri))
    p.end()
    return QIcon(pm)


class PanelToggle(QToolButton):
    """Click: show or hide the 2D panel. Drag up or down: set its height."""

    DRAG_PX = 4

    def __init__(self, parent: QWidget, splitter, panel: QWidget, on_change=None):
        super().__init__(parent)
        self.splitter, self.panel, self.on_change = splitter, panel, on_change
        self.setIconSize(QSize(18, 18))
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setStyleSheet("QToolButton { background-color: rgba(30,30,30,170); border: 1px solid #555;"
                           " border-radius: 4px; } QToolButton:hover { border-color: #007acc; }")
        self._press = None
        self._saved_height = 300
        self.wanted = True                     # what the user chose (full screen hides the panel too)
        self.refresh()
        self.place()

    def place(self):
        """Top-right corner of the view (called when the view is resized)."""
        self.move(self.parent().width() - self.width() - 8, 8)
        self.raise_()

    def refresh(self):
        shown = self.is_shown()
        self.setIcon(_toggle_icon(shown))
        self.setToolTip("2D panel: click to " + ("hide" if shown else "show") + " it, drag to set its height")

    def is_shown(self) -> bool:
        return not self.panel.isHidden()

    def set_shown(self, on: bool):
        self.wanted = on
        if on == self.is_shown():
            return
        if not on:
            sizes = self.splitter.sizes()
            if len(sizes) > 1 and sizes[1] > 20:
                self._saved_height = sizes[1]
            self.panel.hide()
        else:
            self.panel.show()
            total = sum(self.splitter.sizes()) or self.splitter.height()
            h = min(self._saved_height, max(total - 100, 50))
            self.splitter.setSizes([max(total - h, 50), h])
        self.refresh()
        if self.on_change:
            self.on_change(on)

    def set_height(self, h: int):
        total = sum(self.splitter.sizes()) or self.splitter.height()
        h = int(max(40, min(h, total - 60)))
        self.wanted = True
        if not self.is_shown():
            self.panel.show()
            self.refresh()
            if self.on_change:
                self.on_change(True)
        self.splitter.setSizes([total - h, h])
        self._saved_height = h

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            sizes = self.splitter.sizes()
            self._press = (ev.globalPosition().y(), sizes[1] if self.is_shown() and len(sizes) > 1 else 0, False)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press is not None:
            y0, h0, dragging = self._press
            dy = ev.globalPosition().y() - y0
            if dragging or abs(dy) > self.DRAG_PX:
                self._press = (y0, h0, True)
                self.set_height(int(h0 - dy))          # dragging up makes the panel taller
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        press, self._press = self._press, None
        if ev.button() == Qt.MouseButton.LeftButton and press is not None and not press[2]:
            self.set_shown(not self.is_shown())
        self.setDown(False)
        ev.accept()


class JumpDialog(QDialog):
    """View > Jump to coordinates: centre (and optionally distance) of the 3D view."""

    def __init__(self, parent, centre, distance):
        super().__init__(parent)
        self.setWindowTitle("Jump to coordinates")
        form = QFormLayout(self)
        self.edits = []
        for axis, v in zip("xyz", centre):
            e = QLineEdit(f"{v:.6g}")
            form.addRow(f"{axis} (code units)", e)
            self.edits.append(e)
        self.edit_dist = QLineEdit(f"{distance:.6g}")
        self.edit_dist.setToolTip("Camera distance from the centre; leave as is to keep the zoom")
        form.addRow("Distance", self.edit_dist)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        """(x, y, z, distance); raises ValueError on bad input."""
        try:
            xyz = [float(e.text()) for e in self.edits]
            d = float(self.edit_dist.text())
        except ValueError:
            raise ValueError("coordinates must be numbers and the distance positive") from None
        if not all(math.isfinite(v) for v in xyz + [d]) or d <= 0:
            raise ValueError("coordinates must be numbers and the distance positive")
        return xyz[0], xyz[1], xyz[2], d


class TextDialog(QDialog):
    """Read-only, copyable text (Simulation summary)."""

    def __init__(self, parent, title: str, text: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 520)
        lay = QVBoxLayout(self)
        self.text = QPlainTextEdit(text)
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Menlo", 11))
        lay.addWidget(self.text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


def _menu(menubar, title: str):
    for act in menubar.actions():
        if act.menu() is not None and act.text().replace('&', '') == title:
            return act.menu()
    return menubar.addMenu(title)


class ViewExtras(QObject):
    """Overlays, the 2D panel toggle, projection and stereo options, and their menu items."""

    session_save_requested = pyqtSignal(str)
    session_open_requested = pyqtSignal(str)
    labels_requested = pyqtSignal(str)
    jump_requested = pyqtSignal(float, float, float, float)
    summary_requested = pyqtSignal()
    panel_toggled = pyqtSignal(bool)          # the 2D panel was shown (True) or hidden

    def __init__(self, window):
        super().__init__(window)
        self.w = window
        self.overlay: Optional[ViewOverlay] = None
        self.toggle: Optional[PanelToggle] = None
        if getattr(window, 'view_3d', None) is not None and HAS_GL and isinstance(window.view_3d, gl.GLViewWidget):
            v = window.view_3d
            if isinstance(v, CosmoGLView):
                v.eye_separation = lambda: window.eye_separation
            self.overlay = ViewOverlay(window, v)
            self.overlay.raise_()
            window.lbl_colour_overlay.raise_()
            split = getattr(window, 'view_split', None)
            if split is not None:
                split.setHandleWidth(6)
                split.setStyleSheet("QSplitter::handle { background-color: #1c1c1c; }"
                                    " QSplitter::handle:hover { background-color: #007acc; }")
                self.toggle = PanelToggle(v, split, window.panel2d_widget, self._on_panel_toggled)
            if isinstance(v, CosmoGLView):
                v.overlays += [w for w in (self.overlay, self.toggle) if w is not None]
        self._build_menus()

    # ------------------------------------------------------------------ menus

    def _build_menus(self):
        w, mb = self.w, self.w.menuBar()
        file_menu, view_menu, options_menu = _menu(mb, 'File'), _menu(mb, 'View'), _menu(mb, 'Options')

        file_menu.addSeparator()
        a = QAction('Save session...', w)
        a.setShortcut(QKeySequence('Ctrl+S'))
        a.setToolTip("The snapshots, view, colouring, layers and selection sets, to reopen later")
        a.triggered.connect(self._ask_save_session)
        file_menu.addAction(a)
        a = QAction('Open session...', w)
        a.setShortcut(QKeySequence('Ctrl+Shift+O'))
        a.triggered.connect(self._ask_open_session)
        file_menu.addAction(a)
        aux = next((act.menu() for act in file_menu.actions()
                    if act.menu() is not None and act.text() == 'Auxiliary data'), file_menu)
        a = QAction('Read labelled points...', w)
        a.setToolTip("A text file with one point per line: x y z [name]; '#' starts a comment")
        a.triggered.connect(self._ask_labels)
        aux.addAction(a)
        self.act_clear_labels = QAction('Clear labelled points', w)
        self.act_clear_labels.triggered.connect(lambda: self.set_labels(np.empty((0, 3)), []))
        aux.addAction(self.act_clear_labels)

        a = QAction('Simulation summary...', w)
        a.triggered.connect(self.summary_requested)
        options_menu.addAction(a)

        view_menu.addSeparator()
        self.act_ortho = QAction('Orthographic projection', w, checkable=True)
        self.act_ortho.setShortcut(QKeySequence('Ctrl+5'))
        self.act_ortho.setToolTip("No perspective: sizes do not shrink with distance (window and screenshots)")
        self.act_ortho.toggled.connect(self.set_ortho)
        view_menu.addAction(self.act_ortho)
        self.act_anaglyph = QAction('Anaglyph stereo (red/cyan)', w, checkable=True)
        self.act_anaglyph.toggled.connect(self.set_anaglyph)
        view_menu.addAction(self.act_anaglyph)
        if hasattr(w, 'act_stereo'):
            w.act_stereo.toggled.connect(lambda on: on and self.act_anaglyph.setChecked(False))
        self.act_scale = QAction('Scale bar', w, checkable=True)
        self.act_scale.setChecked(True)
        self.act_scale.toggled.connect(lambda on: self._overlay_flag('show_scale', on))
        view_menu.addAction(self.act_scale)
        self.act_time = QAction('Time / redshift', w, checkable=True)
        self.act_time.setChecked(True)
        self.act_time.toggled.connect(lambda on: self._overlay_flag('show_time', on))
        view_menu.addAction(self.act_time)
        self.act_labels = QAction('Labelled points', w, checkable=True)
        self.act_labels.setChecked(True)
        self.act_labels.toggled.connect(lambda on: self._overlay_flag('show_labels', on))
        view_menu.addAction(self.act_labels)
        self.act_panel = QAction('2D panel', w, checkable=True)
        self.act_panel.setChecked(True)
        self.act_panel.setShortcut(QKeySequence('Ctrl+2'))
        self.act_panel.toggled.connect(self._set_panel)
        view_menu.addAction(self.act_panel)
        view_menu.addSeparator()
        a = QAction('Jump to coordinates...', w)
        a.setShortcut(QKeySequence('Ctrl+J'))
        a.triggered.connect(self._ask_jump)
        view_menu.addAction(a)

    # ------------------------------------------------------------------ switches

    def _overlay_flag(self, name: str, on: bool):
        if self.overlay is not None:
            setattr(self.overlay, name, on)
            self.overlay.update()

    def _on_panel_toggled(self, on: bool):
        if self.act_panel.isChecked() != on:
            self.act_panel.blockSignals(True)
            self.act_panel.setChecked(on)
            self.act_panel.blockSignals(False)
        self.panel_toggled.emit(on)

    def _set_panel(self, on: bool):
        if self.toggle is not None:
            self.toggle.set_shown(on)
        else:                                   # no 3D view (no OpenGL): the menu item alone
            self.w.panel2d_widget.setVisible(on)
            self.panel_toggled.emit(on)

    @property
    def panel_wanted(self) -> bool:
        return self.toggle.wanted if self.toggle is not None else not self.w.panel2d_widget.isHidden()

    def panel_shown(self) -> bool:
        return self.toggle.is_shown() if self.toggle is not None else not self.w.panel2d_widget.isHidden()

    def set_ortho(self, on: bool):
        v = getattr(self.w, 'view_3d', None)
        if v is None or not hasattr(v, 'ortho'):
            return
        if self.act_ortho.isChecked() != on:
            self.act_ortho.setChecked(on)
            return
        v.ortho = on
        r = getattr(self.w, 'view_3d_right', None)
        if r is not None and hasattr(r, 'ortho'):
            r.ortho = on
            r.update()
        v.update()

    def set_anaglyph(self, on: bool):
        v = getattr(self.w, 'view_3d', None)
        if v is None or not hasattr(v, 'anaglyph'):
            return
        if self.act_anaglyph.isChecked() != on:
            self.act_anaglyph.setChecked(on)
            return
        if on and getattr(self.w, 'stereo_on', False):
            self.w.act_stereo.setChecked(False)
        v.anaglyph = on
        self.w.stereo_bar.setVisible(on or getattr(self.w, 'stereo_on', False))
        v.update()

    def set_labels(self, pos, names):
        if self.overlay is not None:
            self.overlay.set_labels(pos, names)

    def set_overlay_info(self, time_text: str, unit):
        if self.overlay is not None:
            self.overlay.time_text, self.overlay.unit = time_text, unit
            self.overlay.update()

    def compose_screenshot(self, img):
        """Overlays burnt into a grabbed image of the (left) view."""
        if self.overlay is None or img is None:
            return img
        return self.overlay.paint_onto(img, self.w.view_3d.width())

    def state(self) -> dict:
        return {'ortho': self.act_ortho.isChecked(), 'anaglyph': self.act_anaglyph.isChecked(),
                'scale_bar': self.act_scale.isChecked(), 'time': self.act_time.isChecked(),
                'labels': self.act_labels.isChecked(), 'panel2d': self.panel_shown(),
                'panel2d_height': (self.w.view_split.sizes()[1] if self.panel_shown()
                                   and getattr(self.w, 'view_split', None) is not None else None)}

    def apply_state(self, st: dict):
        self.set_ortho(bool(st.get('ortho', False)))
        self.set_anaglyph(bool(st.get('anaglyph', False)))
        self.act_scale.setChecked(bool(st.get('scale_bar', True)))
        self.act_time.setChecked(bool(st.get('time', True)))
        self.act_labels.setChecked(bool(st.get('labels', True)))
        if self.toggle is not None:
            self.toggle.set_shown(bool(st.get('panel2d', True)))
            if st.get('panel2d') and st.get('panel2d_height'):
                self.toggle.set_height(int(st['panel2d_height']))

    # ------------------------------------------------------------------ dialogs

    def _ask_save_session(self):
        path, _ = QFileDialog.getSaveFileName(self.w, "Save session", "session.cosmoview.json",
                                              "CosmoViewer session (*.json)")
        if path:
            self.session_save_requested.emit(path)

    def _ask_open_session(self):
        path, _ = QFileDialog.getOpenFileName(self.w, "Open session", "", "CosmoViewer session (*.json);;All files (*)")
        if path:
            self.session_open_requested.emit(path)

    def _ask_labels(self):
        path, _ = QFileDialog.getOpenFileName(self.w, "Read labelled points", "",
                                              "Text files (*.txt *.dat *.csv *.tsv);;All files (*)")
        if path:
            self.labels_requested.emit(path)

    def _ask_jump(self):
        from PyQt6.QtWidgets import QMessageBox
        cam = self.w.camera_state()
        if not cam:
            return
        dlg = JumpDialog(self.w, cam['center'], cam['distance'])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            x, y, z, d = dlg.values()
        except ValueError as e:
            QMessageBox.warning(self.w, "Jump to coordinates", str(e))
            return
        self.jump_requested.emit(x, y, z, d)
