"""
PropertyPlotWindow: GadgetViewer's "Make plot" (property_plot.F90 & graph.F90).

A histogram of one particle property, or a scatter plot of one property
against another. As in GadgetViewer, particles in the current sample are
drawn (here in grey rather than red) and each enabled selection set in its own colour, and the plot
updates by itself when the snapshot, the sample or a selection changes.
"""

from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QVBoxLayout, QWidget)
import pyqtgraph as pg

# GadgetViewer draws the sample in red; grey is used here because selection set 1
# defaults to red and would be indistinguishable from the sample.
SAMPLE_RGBA = (170, 170, 170, 90)


class PropertyPlotWindow(QMainWindow):

    plot_requested = pyqtSignal(str, str, str, bool, bool)  # (plot_type, x_prop, y_prop, log_x, log_y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make Plot — Particle Property Analysis")
        self.resize(900, 620)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        ctrl = QFrame()
        ch = QHBoxLayout(ctrl)
        ch.addWidget(QLabel("Plot:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Scatter plot", "Histogram"])
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        ch.addWidget(self.combo_type)
        ch.addWidget(QLabel("X:"))
        self.combo_x = QComboBox()
        self.combo_x.setMinimumContentsLength(14)
        ch.addWidget(self.combo_x, 1)
        self.chk_log_x = QCheckBox("Log")
        ch.addWidget(self.chk_log_x)
        self.lbl_y = QLabel("Y:")
        ch.addWidget(self.lbl_y)
        self.combo_y = QComboBox()
        self.combo_y.setMinimumContentsLength(14)
        ch.addWidget(self.combo_y, 1)
        self.chk_log_y = QCheckBox("Log")
        ch.addWidget(self.chk_log_y)
        self.chk_auto = QCheckBox("Auto update")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip("Redraw when the snapshot, sample or a selection changes")
        ch.addWidget(self.chk_auto)
        self.btn_update = QPushButton("Update Plot")
        self.btn_update.setStyleSheet("background-color: #007acc; font-weight: bold; color: white;")
        self.btn_update.clicked.connect(self._on_update_clicked)
        ch.addWidget(self.btn_update)
        layout.addWidget(ctrl)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#080808')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot_widget.addLegend(offset=(10, 10))
        layout.addWidget(self.plot_widget)
        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.lbl_note)
        self.has_plotted = False

    # ------------------------------------------------------------------ controls

    def set_properties(self, names: List[str]):
        for combo in (self.combo_x, self.combo_y):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(names))
            if cur in names:
                combo.setCurrentText(cur)
            combo.blockSignals(False)
        if self.combo_y.currentIndex() == self.combo_x.currentIndex() and self.combo_y.count() > 1:
            self.combo_y.setCurrentIndex(1)

    def _on_type_changed(self, idx: int):
        is_scatter = (idx == 0)
        for w in (self.lbl_y, self.combo_y, self.chk_log_y):
            w.setVisible(is_scatter)

    def request(self) -> Tuple[str, str, str, bool, bool]:
        p_type = "scatter" if self.combo_type.currentIndex() == 0 else "histogram"
        return (p_type, self.combo_x.currentText(), self.combo_y.currentText(),
                self.chk_log_x.isChecked(), self.chk_log_y.isChecked())

    def _on_update_clicked(self):
        self.plot_requested.emit(*self.request())

    # ------------------------------------------------------------------ drawing

    def _start(self, x_title: str, y_title: str, log_x: bool, log_y: bool):
        self.plot_widget.clear()
        self.legend.clear()
        self.plot_widget.setLabel('bottom', ("log10 " if log_x else "") + x_title)
        self.plot_widget.setLabel('left', ("log10 " if log_y else "") + y_title)
        self.has_plotted = True

    def draw_scatter_layers(self, layers: List[Tuple[str, np.ndarray, np.ndarray, tuple]],
                            x_title: str, y_title: str, log_x: bool, log_y: bool, note: str = ""):
        self._start(x_title, y_title, log_x, log_y)
        for label, x, y, rgba in layers:
            if len(x) == 0:
                continue
            size = 3 if rgba == SAMPLE_RGBA else 4
            item = pg.ScatterPlotItem(x=x, y=y, size=size, pen=None, brush=pg.mkBrush(*rgba), name=label)
            self.plot_widget.addItem(item)
        self.lbl_note.setText(note)

    def draw_histogram_layers(self, layers: List[Tuple[str, np.ndarray, np.ndarray, tuple]],
                              x_title: str, log_x: bool, note: str = ""):
        self._start(x_title, "Particle count", log_x, False)
        for label, edges, counts, rgba in layers:
            fill = (rgba[0], rgba[1], rgba[2], 60)
            curve = pg.PlotCurveItem(edges, counts, stepMode='center', fillLevel=0,
                                     brush=fill, pen=pg.mkPen(rgba[:3], width=2), name=label)
            self.plot_widget.addItem(curve)
        self.lbl_note.setText(note)

    def show_message(self, text: str):
        """GadgetViewer leaves the graph blank when a property cannot be loaded; say why."""
        self.plot_widget.clear()
        self.legend.clear()
        self.lbl_note.setText(text)

    # Back-compatible single-layer calls
    def draw_scatter(self, x_vals, y_vals, x_title, y_title):
        self.draw_scatter_layers([("sample", x_vals, y_vals, SAMPLE_RGBA)], x_title, y_title, False, False)

    def draw_histogram(self, bin_edges, counts, x_title):
        self.draw_histogram_layers([("sample", bin_edges, counts, SAMPLE_RGBA)], x_title, False)
