"""
COLOURING box of the side panel: colour particles by type or by any property.

GadgetViewer's "colour by property" plot type with its settings window (the
property, colour table, linear/log scale, range, and "use random colours for
integer properties" with a Randomize button), plus a colour bar. The panel
holds settings only; the controller computes colours in _build_scene.

Below it, how the two point layers are drawn (views/soft_points.py): "lit"
particles carry the colouring property (all particles when colouring by type)
and are always drawn on top; "unlit" ones lack it and can be drawn dim or
hidden. Each layer has its own size, blur and fade.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

from model.colour_tables import DEFAULT_TABLE, DIVERGING_TABLE, TABLES, colour_bar_image

TYPE_MODE = 'Particle type'
LAGRANGE = 'Lagrangian origin (first snapshot)'
BLEND_MODES = ('Glow', 'Solid')          # views.soft_points.BLEND_MODES (no OpenGL import here)
# layer -> (size px, blur 0..1, fade 0..1) defaults; lit matches the old single layer
DRAWING_DEFAULTS = {'lit': (4, 0.0, 1.0), 'unlit': (3, 0.5, 1.0)}
# Colour modes computed from the scene rather than read from the file.
# name -> (tooltip, integer?, preferred table)
SCENE_MODES: Dict[str, Tuple[str, bool, str]] = {
    'Line-of-sight velocity': (
        "Velocity along the viewing direction, relative to the mean velocity of the drawn particles.\n"
        "Positive (red) = moving away from the viewer, negative (blue) = towards. Code units as stored\n"
        "(for comoving runs multiply by a^0.5 for peculiar km/s). Updates when the view rotates.",
        False, DIVERGING_TABLE),
    'Local density (display sample)': (
        "k-nearest-neighbour density (k = 32) of the drawn particles, scaled by the sampling factor.\n"
        "A visual guide to structure, estimated from the display sample only; not for measurement.",
        False, 'magma'),
    'Distance from view centre': (
        "Distance from the centre of the 3D view (periodic in a box), code units as stored.",
        False, 'viridis'),
    LAGRANGE: (
        "Colour = where each particle was in the earliest loaded snapshot (red = x, green = y, blue = z\n"
        "across the box), found by ParticleID at full resolution. Matter that started together keeps\n"
        "one colour, so mixing and infall show up. The value (for picking and plots) is the distance\n"
        "of that first position from the centre of the box. Table and range do not apply.",
        False, 'viridis'),
}
# Modes whose colours are not a value on a colour table.
RGB_MODES = {LAGRANGE}


def _fmt(x: float) -> str:
    return f"{x:.4g}"


class ColouringPanel(QGroupBox):
    """Settings for colouring; emits ``colouring_changed`` on every edit."""

    colouring_changed = pyqtSignal()
    auto_range_requested = pyqtSignal()
    drawing_changed = pyqtSignal()           # point style only: no colours to recompute

    def __init__(self):
        super().__init__("COLOURING")
        self._integer: Dict[str, bool] = {}
        self._tooltips: Dict[str, str] = {}
        self.seed = 0

        lay = QVBoxLayout(self)
        lay.setSpacing(5)
        lay.setContentsMargins(10, 15, 10, 10)

        h = QHBoxLayout()
        h.addWidget(QLabel("Colour by:"))
        self.combo_prop = QComboBox()
        self.combo_prop.setMinimumContentsLength(14)
        self.combo_prop.currentTextChanged.connect(self._on_property_changed)
        h.addWidget(self.combo_prop, 1)
        lay.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Table:"))
        self.combo_table = QComboBox()
        self.combo_table.addItems(list(TABLES))
        self.combo_table.setCurrentText(DEFAULT_TABLE)
        self.combo_table.currentTextChanged.connect(self._changed)
        h.addWidget(self.combo_table, 1)
        self.chk_log = QCheckBox("Log")
        self.chk_log.toggled.connect(self._on_log_toggled)
        h.addWidget(self.chk_log)
        lay.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Range:"))
        self.edit_min = QLineEdit("0")
        self.edit_max = QLineEdit("1")
        for e in (self.edit_min, self.edit_max):
            e.setMaximumWidth(80)
            e.editingFinished.connect(self._changed)
        h.addWidget(self.edit_min)
        h.addWidget(QLabel("to"))
        h.addWidget(self.edit_max)
        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setToolTip("Set the range to the 0.5–99.5 percentiles of the drawn particles in this snapshot.\n"
                                 "The range then stays fixed across snapshots and in movies.")
        self.btn_auto.clicked.connect(lambda: self.auto_range_requested.emit())
        h.addWidget(self.btn_auto)
        lay.addLayout(h)

        h = QHBoxLayout()
        self.chk_random = QCheckBox("Random colours for integers")
        self.chk_random.setToolTip("Each integer value (e.g. a halo index) gets its own colour; 0 = in no group, drawn dim.")
        self.chk_random.toggled.connect(self._changed)
        h.addWidget(self.chk_random)
        self.btn_randomize = QPushButton("Randomize")
        self.btn_randomize.clicked.connect(self._randomize)
        h.addWidget(self.btn_randomize)
        lay.addLayout(h)

        self.chk_density = QCheckBox("Brightness from local density")
        self.chk_density.setToolTip(
            "Splotch-style pictures: the colour (hue) comes from the property above and the brightness\n"
            "from the local density (k = 32 neighbours in the display sample), log-scaled between the\n"
            "1st and 99.5th percentile of the coloured particles in each frame. For pictures, not measurement.")
        self.chk_density.toggled.connect(self._changed)
        lay.addWidget(self.chk_density)

        self.bar = QLabel()
        self.bar.setFixedHeight(12)
        self.bar.setScaledContents(True)
        lay.addWidget(self.bar)
        h = QHBoxLayout()
        self.lbl_lo = QLabel("")
        self.lbl_hi = QLabel("")
        self.lbl_hi.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(self.lbl_lo)
        h.addWidget(self.lbl_hi)
        lay.addLayout(h)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(self.lbl_info)

        self.chk_unlit = QCheckBox("Show uncoloured particles (dim)")
        self.chk_unlit.setChecked(True)
        self.chk_unlit.setToolTip("Particles without a value of the colouring property (the unlit layer).\n"
                                  "Off: only coloured (lit) particles are drawn, e.g. only black holes for BH_Mass.")
        self.chk_unlit.toggled.connect(lambda _: self.drawing_changed.emit())
        lay.addWidget(self.chk_unlit)
        lay.addWidget(self._drawing_box())

        self.set_properties([], {})
        self._sync_enabled()

    def _drawing_box(self) -> QWidget:
        from views.collapsible import CollapsibleSection
        box = QWidget()
        g = QGridLayout(box)
        g.setContentsMargins(0, 2, 0, 0)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(2)
        for col, text in enumerate(("", "Size", "Blur", "Fade")):
            if text:
                g.addWidget(QLabel(text), 0, col)
        self.sliders = {}
        for row, layer in enumerate(('lit', 'unlit'), start=1):
            lab = QLabel("Lit" if layer == 'lit' else "Unlit")
            lab.setToolTip("Lit: particles with a value of the colouring property (all, when colouring by type),\n"
                           "always drawn on top. Unlit: the rest.")
            g.addWidget(lab, row, 0)
            size, blur, fade = DRAWING_DEFAULTS[layer]
            for col, (name, lo, hi, val) in enumerate((('size', 1, 12, size), ('blur', 0, 100, int(blur * 100)),
                                                       ('fade', 0, 100, int(fade * 100))), start=1):
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setRange(lo, hi)
                sl.setValue(val)
                sl.setMinimumWidth(50)
                sl.valueChanged.connect(lambda _: self.drawing_changed.emit())
                g.addWidget(sl, row, col)
                self.sliders[(layer, name)] = sl
        g.addWidget(QLabel("Lit overlap:"), 3, 0, 1, 2)
        self.combo_blend = QComboBox()
        self.combo_blend.addItems(list(BLEND_MODES))
        self.combo_blend.setToolTip("Glow: lit particles add up (dense regions brighten; lower Fade to keep detail).\n"
                                    "Solid: the nearest lit particle is drawn in front of lit ones behind it.\n"
                                    "Either way lit particles are drawn over unlit ones.")
        self.combo_blend.currentTextChanged.connect(lambda _: self.drawing_changed.emit())
        g.addWidget(self.combo_blend, 3, 2, 1, 2)
        return CollapsibleSection("Point style", box, key="point_style", expanded=False)

    def drawing_state(self) -> dict:
        """Point style of the lit and unlit layers (plain values, safe for _ui_state)."""
        out = {'show_unlit': self.chk_unlit.isChecked(), 'blend': self.combo_blend.currentText()}
        for layer in ('lit', 'unlit'):
            out[layer] = {'size': self.sliders[(layer, 'size')].value(),
                          'blur': self.sliders[(layer, 'blur')].value() / 100.0,
                          'fade': self.sliders[(layer, 'fade')].value() / 100.0}
        return out

    # ------------------------------------------------------------------ properties list

    def set_properties(self, names: List[str], integer: Dict[str, bool], tooltips: Optional[Dict[str, str]] = None):
        """Offer ``names`` (file properties and attached ones) after the built-in modes."""
        current = self.combo_prop.currentText()
        self._integer = dict(integer)
        self._tooltips = dict(tooltips or {})
        for k, (tip, is_int, _) in SCENE_MODES.items():
            self._integer[k] = is_int
            self._tooltips[k] = tip
        items = [TYPE_MODE] + list(SCENE_MODES) + [n for n in names if n not in SCENE_MODES]
        self.combo_prop.blockSignals(True)
        self.combo_prop.clear()
        self.combo_prop.addItems(items)
        for i, n in enumerate(items):
            if n in self._tooltips:
                self.combo_prop.setItemData(i, self._tooltips[n], Qt.ItemDataRole.ToolTipRole)
        self.combo_prop.setCurrentText(current if current in items else TYPE_MODE)
        self.combo_prop.blockSignals(False)
        self._sync_enabled()

    def select_property(self, name: str):
        """Switch to ``name`` (used by View menu items and the FoF dialog)."""
        if self.combo_prop.findText(name) < 0:
            return False
        self.combo_prop.setCurrentText(name)
        return True

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        """Plain values for _ui_state (no widgets leave the GUI thread)."""
        try:
            lo, hi = float(self.edit_min.text()), float(self.edit_max.text())
        except ValueError:
            lo, hi = 0.0, 1.0
        prop = self.combo_prop.currentText() or TYPE_MODE
        return {'property': prop, 'table': self.combo_table.currentText(), 'log': self.chk_log.isChecked(),
                'range': (lo, hi), 'random': self.chk_random.isChecked() and self._integer.get(prop, False),
                'seed': self.seed, 'density_brightness': self.chk_density.isChecked() and prop != TYPE_MODE}

    def set_range(self, lo: float, hi: float):
        self.edit_min.setText(_fmt(lo))
        self.edit_max.setText(_fmt(hi))
        self._changed()

    def set_info(self, text: str):
        self.lbl_info.setText(text)

    # ------------------------------------------------------------------ slots

    def _on_property_changed(self, name: str):
        is_int = self._integer.get(name, False)
        self.chk_random.blockSignals(True)
        self.chk_random.setChecked(is_int)
        self.chk_random.blockSignals(False)
        if name in SCENE_MODES:
            self.combo_table.blockSignals(True)
            self.combo_table.setCurrentText(SCENE_MODES[name][2])
            self.combo_table.blockSignals(False)
        self.combo_prop.setToolTip(self._tooltips.get(name, ""))
        self._sync_enabled()
        if name == TYPE_MODE:
            self.set_info("")
        else:
            # A fresh property starts on its own range; with random colours only the line saying
            # how many drawn particles have a value is renewed (it named the previous property).
            self.auto_range_requested.emit()
        self._changed()

    def _on_log_toggled(self, _):
        self.auto_range_requested.emit()
        self._changed()

    def _randomize(self):
        self.seed += 1
        self._changed()

    def _sync_enabled(self):
        prop = self.combo_prop.currentText()
        by_value = prop != TYPE_MODE
        is_int = self._integer.get(prop, False)
        rnd = by_value and is_int and self.chk_random.isChecked()
        rnd = rnd or prop in RGB_MODES          # no colour table either
        for w in (self.combo_table, self.chk_log, self.edit_min, self.edit_max, self.btn_auto):
            w.setEnabled(by_value and not rnd)
        self.chk_density.setEnabled(by_value)
        self.chk_random.setEnabled(by_value and is_int)
        self.btn_randomize.setEnabled(rnd)
        self.bar.setVisible(by_value and not rnd)
        self.lbl_lo.setVisible(by_value and not rnd)
        self.lbl_hi.setVisible(by_value and not rnd)

    def _changed(self, *_):
        self._sync_enabled()
        st = self.state()
        img = colour_bar_image(st['table'])
        q = QImage(img.data, img.shape[1], img.shape[0], 4 * img.shape[1], QImage.Format.Format_RGBA8888).copy()
        self.bar.setPixmap(QPixmap.fromImage(q))
        lo, hi = st['range']
        self.lbl_lo.setText(_fmt(lo) + (" (log)" if st['log'] else ""))
        self.lbl_hi.setText(_fmt(hi))
        self.colouring_changed.emit()
