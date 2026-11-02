"""
Particle inspector panel: everything stored about one picked particle in the
current snapshot, how each value changed since another snapshot, and the
history of one chosen property over all loaded snapshots.

The panel only displays; controller/inspector_controller.py reads the values
(snapshot_io/particle_record.py) and fills it.
"""

from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

COMPARE_CHOICES = ("previous snapshot", "next snapshot", "first snapshot")


def fmt(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}"


def fmt_rel(v: Optional[float]) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{100.0 * v:+.3g}%"


class ParticleInspectorPanel(QGroupBox):
    PEN = pg.mkPen('#00aaff', width=2)
    inspected_chosen = pyqtSignal(int)        # row in the recent-picks list
    options_changed = pyqtSignal()
    history_property_changed = pyqtSignal(str)
    export_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    select_catalogue_group_requested = pyqtSignal()
    select_fof_group_requested = pyqtSignal()

    def __init__(self):
        super().__init__("PARTICLE INSPECTOR")
        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(10, 15, 10, 10)

        h = QHBoxLayout()
        h.addWidget(QLabel("Particle:"))
        self.combo_recent = QComboBox()
        self.combo_recent.setToolTip("Recently picked particles. Pick one in the 3D view or from List top.")
        self.combo_recent.activated.connect(lambda i: self.inspected_chosen.emit(i))
        h.addWidget(self.combo_recent, 1)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(lambda: self.clear_requested.emit())
        h.addWidget(self.btn_clear)
        lay.addLayout(h)

        self.lbl_header = QLabel("Pick a particle (PARTICLE SELECTION > Pick lit particles) to inspect it.")
        self.lbl_header.setWordWrap(True)
        self.lbl_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_header)

        # The particle's group, as a population: every member, at full resolution, into the active set.
        # One above the other: side by side they do not fit the side panel's width.
        hg = QVBoxLayout()
        hg.setSpacing(2)
        self.btn_cat_group = QPushButton("Select its catalogue group")
        self.btn_cat_group.clicked.connect(lambda: self.select_catalogue_group_requested.emit())
        hg.addWidget(self.btn_cat_group)
        self.btn_fof_group = QPushButton("Select its FoF group")
        self.btn_fof_group.clicked.connect(lambda: self.select_fof_group_requested.emit())
        hg.addWidget(self.btn_fof_group)
        lay.addLayout(hg)
        self.set_group_actions(None, None, "Pick a particle first.", "Pick a particle first.")

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Change since:"))
        self.combo_compare = QComboBox()
        self.combo_compare.addItems(COMPARE_CHOICES)
        self.combo_compare.setToolTip("Δ = value in this snapshot − value in the chosen snapshot.\n"
                                      "Positions use the nearest periodic image in a box.")
        self.combo_compare.currentIndexChanged.connect(lambda _: self.options_changed.emit())
        h2.addWidget(self.combo_compare, 1)
        lay.addLayout(h2)
        h3 = QHBoxLayout()
        self.chk_physical = QCheckBox("Physical units")
        self.chk_physical.setToolTip("Multiply by a^a_scaling h^h_scaling from each dataset's attributes.\n"
                                     "Physical deltas include the cosmic expansion between snapshots;\n"
                                     "the stored (comoving) deltas do not.")
        self.chk_physical.toggled.connect(lambda _: self.options_changed.emit())
        h3.addWidget(self.chk_physical)
        self.chk_changed = QCheckBox("Only changed")
        self.chk_changed.setToolTip("Hide properties whose value is the same in both snapshots")
        self.chk_changed.toggled.connect(lambda _: self.options_changed.emit())
        h3.addWidget(self.chk_changed)
        lay.addLayout(h3)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Property", "Value", "Δ", "Δ %"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(220)
        self.table.cellClicked.connect(self._on_row_clicked)
        lay.addWidget(self.table)

        h4 = QHBoxLayout()
        h4.addWidget(QLabel("History of:"))
        self.combo_history = QComboBox()
        self.combo_history.setToolTip("Value of this property in every loaded snapshot. Click a table row to choose it.")
        self.combo_history.currentTextChanged.connect(lambda s: self.history_property_changed.emit(s))
        h4.addWidget(self.combo_history, 1)
        lay.addLayout(h4)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(140)
        self.plot.setMaximumHeight(180)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot([], [], pen=self.PEN, symbol='o', symbolSize=5,
                                    symbolBrush='#00aaff')
        self.marker = self.plot.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='#ffcc00')
        lay.addWidget(self.plot)

        h5 = QHBoxLayout()
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        h5.addWidget(self.lbl_status, 1)
        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.setToolTip("Every property of this particle in every loaded snapshot, with deltas")
        self.btn_export.clicked.connect(lambda: self.export_requested.emit())
        h5.addWidget(self.btn_export)
        lay.addLayout(h5)

    # ------------------------------------------------------------------ state

    def compare_mode(self) -> str:
        return COMPARE_CHOICES[self.combo_compare.currentIndex()]

    def physical(self) -> bool:
        return self.chk_physical.isChecked()

    def only_changed(self) -> bool:
        return self.chk_changed.isChecked()

    def set_recent(self, labels: Sequence[str], current: int):
        self.combo_recent.blockSignals(True)
        self.combo_recent.clear()
        self.combo_recent.addItems(list(labels))
        if 0 <= current < len(labels):
            self.combo_recent.setCurrentIndex(current)
        self.combo_recent.blockSignals(False)

    def set_history_choices(self, names: List[str]):
        cur = self.combo_history.currentText()
        if [self.combo_history.itemText(i) for i in range(self.combo_history.count())] == names:
            return
        self.combo_history.blockSignals(True)
        self.combo_history.clear()
        self.combo_history.addItems(names)
        if cur in names:
            self.combo_history.setCurrentText(cur)
        self.combo_history.blockSignals(False)

    def history_property(self) -> str:
        return self.combo_history.currentText()

    def set_group_actions(self, catalogue_group: Optional[int], fof_group: Optional[int],
                          catalogue_tip: str = "", fof_tip: str = ""):
        """The two "select its group" buttons: a group number enables a button and is shown on it;
        None disables it, and the tooltip says why."""
        for btn, g, what, tip in ((self.btn_cat_group, catalogue_group, "catalogue", catalogue_tip),
                                  (self.btn_fof_group, fof_group, "FoF", fof_tip)):
            btn.setEnabled(g is not None)
            btn.setText(f"Select its {what} group ({g:,})" if g is not None else f"Select its {what} group")
            btn.setToolTip(tip)

    # ------------------------------------------------------------------ filling

    def set_rows(self, rows: List[tuple]):
        """rows: (name, value_text, delta_text, rel_text, tooltip)."""
        self.table.setRowCount(len(rows))
        for r, (name, val, d, rel, tip) in enumerate(rows):
            for c, txt in enumerate((name, val, d, rel)):
                it = QTableWidgetItem(txt)
                if c:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if tip:
                    it.setToolTip(tip)
                self.table.setItem(r, c, it)

    def table_rows(self) -> List[List[str]]:
        return [[self.table.item(r, c).text() if self.table.item(r, c) else '' for c in range(4)]
                for r in range(self.table.rowCount())]

    def set_history(self, t: np.ndarray, v: np.ndarray, current: int, xlabel: str, ylabel: str,
                    joined: bool = True):
        """joined=False draws the points alone (a label such as a group number)."""
        ok = np.isfinite(t) & np.isfinite(v)
        self.curve.setPen(self.PEN if joined else None)
        self.curve.setData(t[ok], v[ok])
        if 0 <= current < len(t) and ok[current]:
            self.marker.setData([t[current]], [v[current]])
        else:
            self.marker.setData([], [])
        self.plot.setLabel('bottom', xlabel)
        self.plot.setLabel('left', ylabel)

    def clear(self, text: str = ""):
        self.table.setRowCount(0)
        self.curve.setData([], [])
        self.marker.setData([], [])
        if text:
            self.lbl_header.setText(text)
        self.lbl_status.setText("")

    def reset(self, text: str):
        """No particle at all (Clear): besides the table and curve, the history choice, the axis labels
        and the plot's range go, so nothing of the previous particle is left on the panel."""
        self.clear(text)
        self.combo_history.blockSignals(True)
        self.combo_history.clear()
        self.combo_history.blockSignals(False)
        self.plot.setLabel('bottom', '')
        self.plot.setLabel('left', '')
        self.plot.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)   # an empty plot keeps its last range
        self.plot.enableAutoRange()

    def _on_row_clicked(self, row: int, _col: int):
        it = self.table.item(row, 0)
        if it is not None and self.combo_history.findText(it.text()) >= 0:
            self.combo_history.setCurrentText(it.text())
