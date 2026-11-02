"""
plot_panel.py
─────────────
H5Web-style unified toolbar + Matrix / Histogram / Line view panel.

Layout (top → bottom)
─────────────────────
  ViewToolbar   — single bar:[Matrix|Histogram|Line]  [nav slider]
                                [X scale] [Y scale] [Grid] [Aspect]
                                [Cell width] [Auto|Scientific|Exact]
                                [↓ Export] [?]
  Info bar      — dataset metadata (teal)
  Middle row    — MetaStrip  +  QStackedWidget (PlotWidget | MatrixView)
  Status bar    — live stats (right-aligned)

Colour palette
──────────────
  Primary   #4ecdc4  teal   (~177° HSL)  — active tabs, highlights, slider
  Secondary #e07b54  coral  (~19°  HSL)  — Export button  (warm split-complement)
  These two colours sit ~158° apart on the wheel — maximum readability contrast
  on dark backgrounds without visual aggression.

Public API
──────────
  configure_for_plugin(cls)
  render(PlotRequest, vset=None)
  render_matrix(data_arrays, path)
  reset_view()                       (the orchestrator never touches pw or lbl_info)
  set_status(msg)
  current_tab() / current_scale_x/y()

Signals
───────
  tab_changed(str)          forwarded from toolbar
  scale_changed(str, str)   forwarded from toolbar
"""

from __future__ import annotations

import os
import re

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLayout,
    QPushButton, QSlider, QComboBox, QSizePolicy, QApplication,
    QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget,
    QFileDialog, QSpinBox, QAbstractItemView, QDialog,
    QMenu, QButtonGroup, QRadioButton, QWidgetAction, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect, QSize, QPoint
from PyQt6.QtGui import QFont, QColor, QAction, QActionGroup
import pyqtgraph as pg

from plugin_base import PlotRequest, CurveSpec, MarkerSpec, RegionSpec
from data_utils import latex_to_html
from lod_index  import LodIndex, LodCacheKey


# ─────────────────────────────────────────────────────────────────────────────
#  STYLE WHITELIST VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

_VALID_VIS    = {"Line", "Points", "Both"}
_VALID_INTERP = {"Linear", "Step", "Constant"}   # Constant → Step (legacy alias)
_INTERP_NORM  = {"Constant": "Step"}             # normalise old name


def _validate_plot_style(vis: str, interp: str) -> tuple[str, str]:
    """
    Whitelist-check a (vis_mode, interp_mode) pair from a plugin's
    DEFAULT_STYLE_* attribute.  Returns a safe, normalised pair.

    Invalid values are replaced with "Line" / "Linear" and a warning
    is printed to stdout so plugin developers get immediate feedback
    without crashing the UI thread.
    """
    if vis not in _VALID_VIS:
        print(f"[PlotPanel] Warning: invalid vis_mode {vis!r} — "
              f"falling back to 'Line'. Valid: {sorted(_VALID_VIS)}")
        vis = "Line"
    if interp not in _VALID_INTERP:
        print(f"[PlotPanel] Warning: invalid interp_mode {interp!r} — "
              f"falling back to 'Linear'. Valid: {sorted(_VALID_INTERP)}")
        interp = "Linear"
    return vis, _INTERP_NORM.get(interp, interp)


# ─────────────────────────────────────────────────────────────────────────────
#  BIN RESOLUTION DIALOG  (Phase 3 — CosmoViewer Tools context menu)
# ─────────────────────────────────────────────────────────────────────────────

class BinResolutionDialog(QDialog):
    """
    Premium dark-themed dialog for overriding the histogram bin count.

    Layout
    ──────
      Title label  (teal, bold)
      Subtitle     (current bin count hint)
      QSlider      (horizontal, min=4, max=500, tick marks every 50)
      QSpinBox     (linked bidirectionally to slider)
      [Apply]  [Cancel]

    The slider and spinbox are kept in sync via valueChanged.
    Hard-enforced bounds: [4, 500].
    """

    _DIALOG_SS = """
        QDialog {
            background: #0d0d0d;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
        }
        QLabel#dlg_title {
            color: #4ecdc4;
            font-size: 14px;
            font-weight: bold;
        }
        QLabel#dlg_sub {
            color: #666;
            font-size: 11px;
        }
        QLabel#dlg_val {
            color: #c8c8c8;
            font-size: 12px;
            font-weight: bold;
            min-width: 38px;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #2a2a2a;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #4ecdc4;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::sub-page:horizontal {
            background: #4ecdc4;
            border-radius: 2px;
        }
        QSpinBox {
            background: #1a1a1a;
            color: #c8c8c8;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 3px 6px;
            font-size: 13px;
            min-width: 62px;
        }
        QSpinBox::up-button, QSpinBox::down-button {
            background: #222;
            border: none;
            width: 18px;
        }
        QPushButton#btn_apply {
            background: #4ecdc4;
            color: #000;
            border: none;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
            padding: 6px 22px;
        }
        QPushButton#btn_apply:hover  { background: #67d8d0; }
        QPushButton#btn_apply:pressed { background: #3ab8b0; }
        QPushButton#btn_cancel {
            background: #222;
            color: #888;
            border: 1px solid #333;
            border-radius: 4px;
            font-size: 12px;
            padding: 6px 18px;
        }
        QPushButton#btn_cancel:hover { color: #c8c8c8; border-color: #555; }
    """

    _MIN = 4
    _MAX = 500

    def __init__(self, parent: QWidget, current_bins: int = 64):
        super().__init__(parent)
        self.setWindowTitle("CosmoViewer — Override Histogram Resolution")
        self.setModal(True)
        self.setFixedSize(390, 210)
        self.setStyleSheet(self._DIALOG_SS)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(10)

        # Title + subtitle
        lbl_title = QLabel("Override Histogram Resolution")
        lbl_title.setObjectName("dlg_title")
        root.addWidget(lbl_title)

        lbl_sub = QLabel(
            f"Current auto-computed bin count: {current_bins}  "
            f"(bounds: {self._MIN}–{self._MAX})")
        lbl_sub.setObjectName("dlg_sub")
        root.addWidget(lbl_sub)

        root.addSpacing(4)

        # Slider + spinbox row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(self._MIN, self._MAX)
        self._slider.setValue(current_bins)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(50)
        ctrl_row.addWidget(self._slider, stretch=1)

        self._spin = QSpinBox()
        self._spin.setRange(self._MIN, self._MAX)
        self._spin.setValue(current_bins)
        ctrl_row.addWidget(self._spin)

        root.addLayout(ctrl_row)
        root.addSpacing(6)

        # Sync slider ↔ spinbox
        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("btn_cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_apply = QPushButton("Apply")
        btn_apply.setObjectName("btn_apply")
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self.accept)
        btn_row.addWidget(btn_apply)

        root.addLayout(btn_row)

    def value(self) -> int:
        """Return the currently selected bin count."""
        return self._spin.value()


# ─────────────────────────────────────────────────────────────────────────────
#  THEME MANAGER — semantic role → visual attributes
# ─────────────────────────────────────────────────────────────────────────────

class ThemeManager:
    """
    Maps semantic roles to visual attributes based on the active app theme.

    Plugins assign semantic roles (e.g. role="primary_data") instead of
    hardcoding hex colours.  This class resolves roles to concrete visual
    properties at render time.

    If a spec has no role (empty string), its existing color/width/fill_color
    fields are used as fallbacks — ensuring backward compat for external
    consumers or testing.
    """

    CURVE_ROLES: dict[str, dict] = {
        "primary_data":   {"color": "#00b4d8", "width": 1.6,
                           "fill_color": "#00b4d841"},
        "secondary_data": {"color": "#a8dadc", "width": 1.6,
                           "fill_color": "#a8dadc30"},
        "ghost_context":  {"color": "#4ecdc4", "width": 1.0},
        "diag_primary":   {"color": "#4ecdc4", "width": 1.0},
        "diag_secondary": {"color": "#e07b54", "width": 1.0},
    }

    MARKER_ROLES: dict[str, dict] = {
        "domain_min":       {"color": "#ff6b6b", "dash": True},
        "domain_max":       {"color": "#ffe66d", "dash": True},
        "threshold_marker": {"color": "#ffe66d", "dash": True},
        "median_marker":    {"color": "#ff6b6b", "dash": True},
    }

    REGION_ROLES: dict[str, dict] = {
        "highlight_region": {"color": "#ffe66d18"},
    }

    @classmethod
    def resolve_curve(cls, spec: CurveSpec) -> CurveSpec:
        """Apply role-based visual attributes to a CurveSpec."""
        if not spec.role:
            return spec
        props = cls.CURVE_ROLES.get(spec.role, {})
        if props:
            spec.color = props.get("color", spec.color)
            spec.width = props.get("width", spec.width)
            if "fill_color" in props:
                spec.fill_color = props["fill_color"]
        return spec

    @classmethod
    def resolve_marker(cls, spec: MarkerSpec) -> MarkerSpec:
        """Apply role-based visual attributes to a MarkerSpec."""
        if not spec.role:
            return spec
        props = cls.MARKER_ROLES.get(spec.role, {})
        if props:
            spec.color = props.get("color", spec.color)
            spec.dash  = props.get("dash", spec.dash)
        return spec

    @classmethod
    def resolve_region(cls, spec: RegionSpec) -> RegionSpec:
        """Apply role-based visual attributes to a RegionSpec."""
        if not spec.role:
            return spec
        props = cls.REGION_ROLES.get(spec.role, {})
        if props:
            spec.color = props.get("color", spec.color)
        return spec



# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAX_MATRIX_DISPLAY = 10_000   # rows rendered in the matrix table


# ─────────────────────────────────────────────────────────────────────────────
#  CELL FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _as_columns(arr: np.ndarray) -> np.ndarray:
    """View a dataset as rows × columns: 1-D → N×1, N×K stays, higher dims flattened per row."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(len(arr), -1)


def _column_labels(k: int) -> list:
    return ["x", "y", "z"] if k == 3 else [str(j) for j in range(k)]


def _format_cell(v: float, notation: str) -> str:
    """Format a single float for the matrix table."""
    if not np.isfinite(v):
        return "NaN" if np.isnan(v) else ("∞" if v > 0 else "-∞")
    if notation == "Scientific":
        return f"{v:.6e}"
    elif notation == "Exact":
        return f"{v:.10g}"
    else:   # Auto
        if v == 0.0:
            return "0"
        abs_v = abs(v)
        if 1e-3 <= abs_v < 1e4:
            return f"{v:.6g}"
        return f"{v:.4e}"


# ─────────────────────────────────────────────────────────────────────────────
#  MATRIX VIEW
# ─────────────────────────────────────────────────────────────────────────────

class MatrixView(QFrame):
    """
    QTableWidget-based matrix display.

    For 1-D datasets (shape [N]) renders as a single-column table with
    row indices 0…N-1 on the left, mirroring H5Web's Matrix tab.

    Display is capped at MAX_MATRIX_DISPLAY rows; full data is available
    via the Export button. Uses setUpdatesEnabled(False) during population
    to keep load time < 0.5 s even at 10 K rows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("matrix_view")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Truncation notice
        self._notice = QLabel("")
        self._notice.setObjectName("matrix_notice")
        self._notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._notice.hide()
        lay.addWidget(self._notice)

        # Table
        self._table = QTableWidget()
        self._table.setObjectName("matrix_table")
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.setShowGrid(True)
        lay.addWidget(self._table, stretch=1)

        self._data:     np.ndarray | None = None
        self._comp:     np.ndarray | None = None
        self._comp_label: str             = "Loaded"
        self._notation: str               = "Auto"
        self._path:     str               = ""

    # ── Public ───────────────────────────────────────────────────────────────

    def load(
        self,
        array:      np.ndarray,
        path:       str,
        notation:   str = "Auto",
        comparison: np.ndarray | None = None,
        comp_label: str = "Loaded",
        labels:     list | None = None,
        cells:      list | None = None,
    ):
        self._labels     = labels      # column names; None = x/y/z or 0..K-1
        self._cells      = cells       # rows of strings: a text table shown as given
        self._data       = array
        self._comp       = comparison
        self._comp_label = comp_label
        self._notation   = notation
        self._path       = path
        self._rebuild()

    def set_notation(self, notation: str):
        if self._data is None or notation == self._notation:
            return
        self._notation = notation
        self._reformat_cells()

    def set_cell_width(self, width: int):
        self._table.setColumnWidth(0, width)

    # ── Private ──────────────────────────────────────────────────────────────

    def _rebuild(self):
        if getattr(self, "_cells", None):
            self._rebuild_text()
            return
        arr    = self._data
        comp   = self._comp
        n      = len(arr)
        n_show = min(n, MAX_MATRIX_DISPLAY)
        two_col = comp is not None

        if n > MAX_MATRIX_DISPLAY:
            self._notice.setText(
                f"⚠  Showing first {n_show:,} of {n:,} particles — "
                f"use  ↓ Export  to get the full dataset"
            )
            self._notice.show()
        else:
            self._notice.hide()

        self._table.setUpdatesEnabled(False)
        self._table.clearContents()

        if two_col:
            # Two-column mode: Pipeline | Loaded (+ optional Δ)
            n_comp   = len(comp)
            n_both   = min(n_show, n_comp)   # rows with both values
            n_total  = max(n_show, min(n_comp, MAX_MATRIX_DISPLAY))
            dataset_name = self._path.split("/")[-1]

            self._table.setColumnCount(3)
            self._table.setRowCount(n_total)
            self._table.setHorizontalHeaderLabels([dataset_name, self._comp_label, "Δ rel"]
            )
            self._table.setVerticalHeaderLabels([str(i) for i in range(n_total)])

            arr_slice  = arr[:n_show]
            comp_slice = comp[:min(n_comp, MAX_MATRIX_DISPLAY)]

            for i in range(n_total):
                has_pipe = i < n_show
                has_comp = i < len(comp_slice)

                # Pipeline column
                cell0 = QTableWidgetItem(
                    _format_cell(float(arr_slice[i]), self._notation)
                    if has_pipe else "—"
                )
                cell0.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(i, 0, cell0)

                # Loaded column
                cell1 = QTableWidgetItem(
                    _format_cell(float(comp_slice[i]), self._notation)
                    if has_comp else "—"
                )
                cell1.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(i, 1, cell1)

                # Δ rel column
                if has_pipe and has_comp:
                    pv = float(arr_slice[i])
                    cv = float(comp_slice[i])
                    if pv != 0 and np.isfinite(pv) and np.isfinite(cv):
                        rel = abs((pv - cv) / pv)
                        delta_str = _format_cell(rel, "Scientific")
                        cell2 = QTableWidgetItem(delta_str)
                        # Colour-code: green < 1e-4, amber < 1e-2, red otherwise
                        if rel < 1e-4:
                            cell2.setForeground(QColor("#4ecdc4"))
                        elif rel < 1e-2:
                            cell2.setForeground(QColor("#ffe66d"))
                        else:
                            cell2.setForeground(QColor("#e07b54"))
                    else:
                        cell2 = QTableWidgetItem("—")
                else:
                    cell2 = QTableWidgetItem("—")
                cell2.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(i, 2, cell2)

        else:
            # Single-dataset mode: one column per component (N×K → K columns)
            arr_slice = _as_columns(arr[:n_show])
            k = arr_slice.shape[1]
            self._table.setColumnCount(k)
            self._table.setRowCount(n_show)
            labels = getattr(self, "_labels", None)
            self._table.setHorizontalHeaderLabels(
                labels if labels and len(labels) == k else _column_labels(k))
            self._table.setVerticalHeaderLabels([str(i) for i in range(n_show)])
            for i in range(n_show):
                for j in range(k):
                    item = QTableWidgetItem(
                        _format_cell(float(arr_slice[i, j]), self._notation))
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self._table.setItem(i, j, item)

        self._table.setUpdatesEnabled(True)
        self._table.resizeColumnsToContents()

    def _rebuild_text(self):
        """Text table (analysis reports): cells shown exactly as given."""
        rows = self._cells
        k = max(len(r) for r in rows)
        self._notice.hide()
        self._table.setUpdatesEnabled(False)
        self._table.clearContents()
        self._table.setColumnCount(k)
        self._table.setRowCount(len(rows))
        labels = getattr(self, "_labels", None)
        self._table.setHorizontalHeaderLabels(
            labels if labels and len(labels) == k else _column_labels(k))
        self._table.setVerticalHeaderLabels([str(i) for i in range(len(rows))])
        colours = {"FAIL": "#e07b54", "WARN": "#ffe66d", "PASS": "#4ecdc4"}
        for i, row in enumerate(rows):
            for j, text in enumerate(row):
                item = QTableWidgetItem(str(text))
                if str(text) in colours:
                    item.setForeground(QColor(colours[str(text)]))
                self._table.setItem(i, j, item)
        self._table.setUpdatesEnabled(True)
        self._table.resizeColumnsToContents()

    def _reformat_cells(self):
        if self._data is None or getattr(self, "_cells", None):
            return
        n_show = min(len(self._data), MAX_MATRIX_DISPLAY)
        arr    = _as_columns(self._data[:n_show])
        self._table.setUpdatesEnabled(False)
        for i, row in enumerate(arr):
            for j, v in enumerate(row if self._comp is None else row[:1]):
                item = self._table.item(i, j)
                if item:
                    item.setText(_format_cell(float(v), self._notation))
            # Also reformat comparison column if present
            if self._comp is not None and i < len(self._comp):
                item2 = self._table.item(i, 1)
                if item2:
                    item2.setText(
                        _format_cell(float(self._comp[i]), self._notation)
                    )
        self._table.setUpdatesEnabled(True)


# ─────────────────────────────────────────────────────────────────────────────
#  VIEW TOOLBAR
# ─────────────────────────────────────────────────────────────────────────────

class _FlowLayout(QLayout):
    """
    Lays its items out left to right and starts a new row when the next one
    does not fit, so a narrow window wraps the toolbar onto a second row
    instead of clipping it or forcing the window wider than the screen.

    Every item stays visible and keeps its size hint; nothing scrolls.
    Items whose size hint is empty (all their contents hidden) take no room.
    With right_align_last, the last item sits at the right-hand end of its
    row (the Export group, as the stretch placed it before).
    """

    def __init__(self, parent=None, row_height: int = 44, spacing: int = 4):
        super().__init__(parent)
        self._items: list = []
        self._row_height = row_height
        self.setSpacing(spacing)
        self.right_align_last = False

    # QLayout interface
    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def _shown(self):
        return [it for it in self._items
                if not it.isEmpty() and it.sizeHint().width() > 0]

    def sizeHint(self):
        """One row holding everything."""
        m = self.contentsMargins()
        shown = self._shown()
        w = sum(it.sizeHint().width() for it in shown) + self.spacing() * max(0, len(shown) - 1)
        h = max([self._row_height] + [it.sizeHint().height() for it in shown])
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        """As narrow as the widest single item: the rest wraps."""
        m = self.contentsMargins()
        shown = self._shown()
        w = max([0] + [it.minimumSize().width() for it in shown])
        return QSize(w + m.left() + m.right(), self._row_height + m.top() + m.bottom())

    def _do_layout(self, rect, apply: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        shown = self._shown()
        rows, row, x = [], [], 0
        for it in shown:
            w = it.sizeHint().width()
            if row and x + self.spacing() + w > r.width():
                rows.append(row)
                row, x = [], 0
            x += (self.spacing() if row else 0) + w
            row.append(it)
        if row:
            rows.append(row)
        y = r.y()
        for k, row in enumerate(rows):
            h = max([self._row_height] + [it.sizeHint().height() for it in row])
            if apply:
                x = r.x()
                for it in row:
                    hint = it.sizeHint()
                    if self.right_align_last and it is shown[-1]:
                        x = max(x, r.right() + 1 - hint.width())
                    it.setGeometry(QRect(QPoint(x, y + (h - hint.height()) // 2), hint))
                    x += hint.width() + self.spacing()
            y += h
        return y - r.y() + m.top() + m.bottom()


class ViewToolbar(QFrame):
    """
    Single unified H5Web-style toolbar.

    Sections (left → right)
    ───────────────────────
      A  View tabs       Matrix | Histogram | Line
      ─  separator
      B  Nav controls    ⏮ ◀ [slider] ▶ ⏭ ✏  (Line only)
      ─  separator
      C  Scale combos    X ≡ scale▾   Y ≡ scale▾   (not Matrix)
         Grid toggle     ⊞ Grid                     (not Matrix)
         Aspect button   Aspect ▾                   (not Matrix, disconnected)
      ─  separator
      D  Matrix extras   Cell width [120][Auto|Scientific|Exact]  (Matrix only)
         ─ stretch ─
      ─  separator
      E  Export + Help   [↓ Export]  [?]

    Signals
    ───────
      view_changed(str)         active view name
      scale_changed(str, str)   axis ("x"|"y"), mode ("log"|"linear")
      notation_changed(str)     "Auto" | "Scientific" | "Exact"
      cell_width_changed(int)   spinbox value
      grid_toggled(bool)        grid on/off
      export_clicked()          export button pressed
    """

    view_changed       = pyqtSignal(str)
    scale_changed      = pyqtSignal(str, str)
    notation_changed   = pyqtSignal(str)
    cell_width_changed = pyqtSignal(int)
    grid_toggled       = pyqtSignal(bool)
    export_clicked     = pyqtSignal()
    load_matrix_clicked= pyqtSignal()   # user clicked the compared-arrays chip (choose other files)
    clear_diag_clicked = pyqtSignal()   # user clicked ✕ on active button

    LOAD_IDLE_TEXT = "Compare arrays…"  # never shown in the toolbar: the chip is hidden while idle
    swap_z_order_clicked = pyqtSignal() # ⇅ toggle: flip pipeline/diagnostic z-order
    unit_changed       = pyqtSignal(str) # Phase 2: user selected a new display unit
    style_changed      = pyqtSignal(str, str)  # (vis_mode, interp_mode)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("view_toolbar")
        # One 44 px row when the window is wide enough; otherwise the groups wrap onto
        # further rows (never a scrolling strip, and never wider than the window).
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

        self._active_view   = ""
        self._view_btns:    dict[str, QPushButton] = {}
        self._notation_btns:dict[str, QPushButton] = {}

        flow = _FlowLayout(self, row_height=44, spacing=4)
        flow.setContentsMargins(10, 0, 10, 0)
        self._flow = flow

        # ── A. View tabs ──────────────────────────────────────────────────
        lay = self._chunk(flow)
        self._tab_container = QWidget()
        self._tab_container.setObjectName("tab_container")
        self._tab_lay = QHBoxLayout(self._tab_container)
        self._tab_lay.setContentsMargins(0, 0, 0, 0)
        self._tab_lay.setSpacing(0)
        lay.addWidget(self._tab_container)

        lay = self._chunk(flow)
        lay.addWidget(self._vsep())


        # ── C. Scale + Grid + Aspect (plot modes) ─────────────────────────
        self._scale_section = QWidget()
        sl = QHBoxLayout(self._scale_section)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(5)

        sl.addWidget(self._axis_lbl("X"))
        self._combo_x = self._scale_combo()
        self._combo_x.currentTextChanged.connect(
            lambda v: self.scale_changed.emit("x", v.lower()))
        sl.addWidget(self._combo_x)

        sl.addSpacing(4)
        sl.addWidget(self._axis_lbl("Y"))
        self._combo_y = self._scale_combo()
        self._combo_y.currentTextChanged.connect(
            lambda v: self.scale_changed.emit("y", v.lower()))
        sl.addWidget(self._combo_y)

        sl.addSpacing(6)

        self._btn_grid = QPushButton("⊞  Grid")
        self._btn_grid.setObjectName("tool_btn_toggle")
        self._btn_grid.setCheckable(True)
        self._btn_grid.setChecked(True)
        self._btn_grid.toggled.connect(self.grid_toggled)
        sl.addWidget(self._btn_grid)

        btn_aspect = QPushButton("Style  ▾")
        btn_aspect.setObjectName("tool_btn")
        btn_aspect.setToolTip("Plot style: visibility and interpolation options")

        # ── Build the Style drop-down menu (radio buttons) ───────────────
        style_menu = QMenu(btn_aspect)
        style_menu.setStyleSheet(
            "QMenu { background:#1a1a1a; border:1px solid #333; min-width:0px; }"
            "QMenu::item:selected { background:transparent; }"
            "QMenu::separator { height:1px; background:#333; margin:2px 0; }"
        )

        # Shared radio-button stylesheet
        _rb_ss = (
            "QRadioButton { color:#c8c8c8; font-size:11px;"
            "  padding:2px 8px; background:transparent; }"
            "QRadioButton:hover { color:#ffffff; }"
            "QRadioButton:disabled { color:#3a3a3a; }"
            "QRadioButton::indicator { width:11px; height:11px; }"
        )

        def _add_header(text: str, font_size: int = 9) -> None:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"QLabel {{ color:#555; font-size:{font_size}px;"
                "  padding:4px 8px 1px 8px; background:transparent; }"   # plain string: one brace
            )
            wa = QWidgetAction(style_menu)
            wa.setDefaultWidget(lbl)
            style_menu.addAction(wa)

        def _add_radio(label: str, grp: QButtonGroup,
                       store: dict, checked: bool = False) -> None:
            rb = QRadioButton(label)
            rb.setChecked(checked)
            rb.setStyleSheet(_rb_ss)
            rb.setAutoExclusive(False)   # QButtonGroup handles exclusivity
            grp.addButton(rb)
            store[label] = rb
            wa = QWidgetAction(style_menu)
            wa.setDefaultWidget(rb)
            style_menu.addAction(wa)

        self._vis_bg    = QButtonGroup(style_menu)
        self._interp_bg = QButtonGroup(style_menu)
        self._vis_bg.setExclusive(True)
        self._interp_bg.setExclusive(True)
        self._rb_vis:    dict[str, QRadioButton] = {}
        self._rb_interp: dict[str, QRadioButton] = {}
        # False on the Histogram tab: bins are always drawn as bins.
        self._interp_allowed: bool = True

        _add_header("VISIBILITY", 9)
        for _lbl in ("Line", "Points", "Both"):
            _add_radio(_lbl, self._vis_bg, self._rb_vis, _lbl == "Line")

        style_menu.addSeparator()

        _add_header("INTERPOLATION", 8)   # ← smaller font → narrower menu
        for _lbl in ("Linear", "Step"):
            _add_radio(_lbl, self._interp_bg, self._rb_interp, _lbl == "Linear")

        # ── Callbacks with cross-group logic ────────────────────────────
        def _emit_style() -> None:
            vis    = next((k for k, r in self._rb_vis.items()    if r.isChecked()), "Line")
            interp = next((k for k, r in self._rb_interp.items() if r.isChecked()), "Linear")
            if interp == "Step":
                vis = "Line"  # Automatically shift to line for the renderer (no UI changes)
            self.style_changed.emit(vis, interp)

        def _on_vis_toggled(checked: bool) -> None:
            if not checked:
                return
            vis = next((k for k, r in self._rb_vis.items() if r.isChecked()), "Line")
            # Points → disable interpolation group and silently force Linear
            points = (vis == "Points")
            for r in self._rb_interp.values():
                r.setEnabled(not points and self._interp_allowed)
            if points:
                self._rb_interp["Linear"].setChecked(True)
            _emit_style()

        def _on_interp_toggled(checked: bool) -> None:
            if not checked:
                return
            _emit_style()

        for _r in self._rb_vis.values():
            _r.toggled.connect(_on_vis_toggled)
        for _r in self._rb_interp.values():
            _r.toggled.connect(_on_interp_toggled)

        btn_aspect.setMenu(style_menu)
        sl.addWidget(btn_aspect)


        lay.addWidget(self._scale_section)

        # ── Phase 2: Unit conversion dropdown ─────────────────────────────
        # Hidden by default; shown only when PlotRequest.unit_choices is
        # non-empty (i.e., Histogram tab of a supported plugin).
        # update_unit_dropdown() is the single public entry point.
        # Signals are blocked during programmatic updates to avoid re-triggering
        # compute cycles.
        self._sep_before_unit = self._vsep()
        lay = self._chunk(flow)
        lay.addWidget(self._sep_before_unit)

        self._unit_section = QWidget()
        ul = QHBoxLayout(self._unit_section)
        ul.setContentsMargins(0, 0, 0, 0)
        ul.setSpacing(5)

        lbl_unit = QLabel("Unit")
        lbl_unit.setStyleSheet("color:#555; font-size:10px;")
        ul.addWidget(lbl_unit)

        self._combo_unit = QComboBox()
        self._combo_unit.setObjectName("scale_combo")
        self._combo_unit.setMinimumWidth(110)
        self._combo_unit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._combo_unit.setToolTip(
            "Display unit for this plot.\n"
            "Conversion scalar is a pure float applied to the raw linear array\n"
            "before any log transform — mathematically exact."
        )
        self._combo_unit.currentTextChanged.connect(self._on_unit_changed)
        ul.addWidget(self._combo_unit)

        lay.addWidget(self._unit_section)

        # Initially hidden — revealed by update_unit_dropdown()
        self._unit_section.hide()
        self._sep_before_unit.hide()

        # ── Sep before matrix section ─────────────────────────────────────
        self._sep_before_matrix = self._vsep()
        lay = self._chunk(flow)
        lay.addWidget(self._sep_before_matrix)

        # ── D. Matrix extras ──────────────────────────────────────────────
        self._matrix_section = QWidget()
        ml = QHBoxLayout(self._matrix_section)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(5)

        lbl_cw = QLabel("Cell width")
        lbl_cw.setStyleSheet("color:#555; font-size:10px;")
        ml.addWidget(lbl_cw)

        self._cell_width = QSpinBox()
        self._cell_width.setObjectName("cell_width_spin")
        self._cell_width.setRange(60, 400)
        self._cell_width.setValue(120)
        self._cell_width.setFixedWidth(60)
        self._cell_width.valueChanged.connect(self.cell_width_changed)
        ml.addWidget(self._cell_width)

        ml.addSpacing(10)

        lbl_not = QLabel("Notation")
        lbl_not.setStyleSheet("color:#555; font-size:10px;")
        ml.addWidget(lbl_not)

        positions = ["left", "mid", "right"]
        for mode, pos in zip(("Auto", "Scientific", "Exact"), positions):
            btn = QPushButton(mode)
            btn.setObjectName("notation_btn")
            btn.setProperty("position", pos)
            btn.setCheckable(True)
            btn.setChecked(mode == "Auto")
            btn.clicked.connect(lambda _, m=mode: self._set_notation(m))
            self._notation_btns[mode] = btn
            ml.addWidget(btn)

        lay.addWidget(self._matrix_section)

        # ── Export and what follows sit at the right-hand end of their row ──
        lay = self._chunk(flow)
        flow.right_align_last = True
        lay.addWidget(self._vsep())

        # ── E. Export + Load Matrix + Help ───────────────────────────────
        self._btn_export = QPushButton("↓  Export")
        self._btn_export.setObjectName("export_btn")
        self._btn_export.setToolTip("Export current dataset to CSV")
        self._btn_export.clicked.connect(self.export_clicked)
        lay.addWidget(self._btn_export)

        # ── Compared arrays (diagnostic override) — shown only while active ──
        #    Started from Options > "Compare arrays from files (advanced)…";
        #    the toolbar has no idle button for it.
        #    Active: solid muted coral  elided-filename  [✕]
        self._load_container = QWidget()
        self._load_container.setObjectName("load_container")
        self._load_container.setFixedWidth(180)
        lc_lay = QHBoxLayout(self._load_container)
        lc_lay.setContentsMargins(0, 0, 0, 0)
        lc_lay.setSpacing(0)

        self._btn_load_main = QPushButton(self.LOAD_IDLE_TEXT)
        self._btn_load_main.setObjectName("load_btn_idle")
        self._btn_load_main.setToolTip(
            "Arrays from .npy, .csv or .txt files, shown instead of the snapshot.\n"
            "Click to choose other files; ✕ returns to the snapshot.\n"
            "The field tree is disabled meanwhile."
        )
        self._btn_load_main.clicked.connect(self.load_matrix_clicked)
        lc_lay.addWidget(self._btn_load_main, stretch=1)

        self._btn_load_clear = QPushButton("✕")
        self._btn_load_clear.setObjectName("load_btn_clear")
        self._btn_load_clear.setFixedSize(24, 28)
        self._btn_load_clear.setToolTip("Clear the compared arrays and return to the snapshot")
        self._btn_load_clear.clicked.connect(self.clear_diag_clicked)
        self._btn_load_clear.hide()
        lc_lay.addWidget(self._btn_load_clear)

        lay.addWidget(self._load_container)
        self._load_container.hide()          # idle: nothing in the toolbar

        # ── ⇅ Z-order toggle ─────────────────────────────────────────────
        # Swaps which layer is in front: pipeline output vs loaded diagnostic.
        # Checkable: unchecked = pipeline on top, checked = diagnostic on top.
        # Hidden until a diagnostic file is loaded; always one click away.
        self._btn_swap = QPushButton("⇅")
        self._btn_swap.setObjectName("tool_btn_toggle")
        self._btn_swap.setCheckable(True)
        self._btn_swap.setFixedSize(28, 28)
        self._btn_swap.setToolTip(
            "Swap z-order: flip which layer is in front.\n"
            "Pipeline output ↔ Loaded diagnostic file."
        )
        self._btn_swap.toggled.connect(
            lambda _: self.swap_z_order_clicked.emit()
        )
        self._btn_swap.hide()
        lay.addWidget(self._btn_swap)

        btn_help = QPushButton("?")
        btn_help.setObjectName("help_btn")
        btn_help.setFixedSize(26, 26)
        btn_help.setToolTip("How to read this view")
        btn_help.clicked.connect(self._show_help)
        lay.addWidget(btn_help)

        # Initial visibility (no plugin loaded yet)
        self._apply_visibility("Histogram")

    HELP_TEXT = (
        "<b>Choosing data.</b> Pick a field in the tree on the left; the Analyses "
        "section holds the measurements (Lagrangian radii, energy, profiles, phase "
        "diagram, sanity report).<br><br>"
        "<b>Views.</b> Matrix lists the values; Histogram bins them; Line plots them "
        "against their row number. X and Y set linear or log scales, ⊞ the grid.<br><br>"
        "<b>Histogram bins.</b> Click a bar to see its range, and, once the bin index "
        "is built, how many particles it holds and their rows in the file.<br><br>"
        "<b>Units.</b> Options sets the unit system and Physical or Comoving "
        "coordinates, and the centre used by the Analyses. The info bar under the "
        "plot says which h and a factors were applied.<br><br>"
        "<b>Samples.</b> A status line that says SAMPLE 1-in-K shows a strided "
        "sample of a large dataset, not every particle. Measurements are science "
        "only when they used every particle; otherwise they say they are a "
        "preview.<br><br>"
        "<b>Export.</b> ↓ Export writes what is on screen to CSV."
    )

    def _show_help(self):
        QMessageBox.information(self, "Dataset Explorer: reading this view", self.HELP_TEXT)

    # ── View tabs (public) ────────────────────────────────────────────────────

    def set_view_tabs(self, tabs: list[str], default: str = ""):
        """Rebuild view-mode tab buttons from a list (called by configure_for_plugin)."""
        for btn in self._view_btns.values():
            self._tab_lay.removeWidget(btn)
            # Hide now: deleteLater alone leaves the old button drawn over the new ones
            # until the event loop next deletes objects (processEvents does not).
            btn.hide()
            btn.deleteLater()
        self._view_btns.clear()

        self._active_view = default or (tabs[0] if tabs else "")

        for name in tabs:
            btn = QPushButton(name)
            btn.setObjectName("tab_btn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, n=name: self._switch_view(n))
            self._view_btns[name] = btn
            self._tab_lay.addWidget(btn)

        self._refresh_tab_btns()
        self._apply_visibility(self._active_view)

    def active_view(self) -> str:
        return self._active_view

    # ── Scale (public) ────────────────────────────────────────────────────────

    def set_scale_defaults(self, scale_x: str, scale_y: str):
        self._combo_x.blockSignals(True)
        self._combo_y.blockSignals(True)
        self._combo_x.setCurrentText(scale_x.capitalize())
        self._combo_y.setCurrentText(scale_y.capitalize())
        self._combo_x.blockSignals(False)
        self._combo_y.blockSignals(False)

    def current_scale_x(self) -> str:
        return self._combo_x.currentText().lower()

    def current_scale_y(self) -> str:
        return self._combo_y.currentText().lower()

    # ── Nav (public) ─────────────────────────────────────────────────────────

    # ── Notation (public) ─────────────────────────────────────────────────────

    def notation(self) -> str:
        for mode, btn in self._notation_btns.items():
            if btn.isChecked():
                return mode
        return "Auto"

    def set_load_active(self, filename: str):
        """Switch load button to active (diagnostic override) state."""
        # Elide filename to fit the fixed container width
        max_chars  = 18
        label      = filename if len(filename) <= max_chars else filename[:max_chars - 1] + "…"
        self._btn_load_main.setText(label)
        self._btn_load_main.setObjectName("load_btn_active")
        self._btn_load_main.style().unpolish(self._btn_load_main)
        self._btn_load_main.style().polish(self._btn_load_main)
        self._btn_load_clear.show()
        self._load_container.show()

    def set_load_idle(self):
        """Switch back to idle: the compared-arrays chip leaves the toolbar."""
        self._btn_load_main.setText(self.LOAD_IDLE_TEXT)
        self._btn_load_main.setObjectName("load_btn_idle")
        self._btn_load_main.style().unpolish(self._btn_load_main)
        self._btn_load_main.style().polish(self._btn_load_main)
        self._btn_load_clear.hide()
        self._load_container.hide()

    def show_swap_button(self):
        """Show ⇅ when a diagnostic file is loaded."""
        self._btn_swap.setChecked(False)   # reset to pipeline-on-top
        self._btn_swap.show()

    def hide_swap_button(self):
        """Hide ⇅ when diagnostic is cleared."""
        self._btn_swap.hide()
        self._btn_swap.setChecked(False)

    def diag_on_top(self) -> bool:
        """True when the toggle is checked = diagnostic layer is in front."""
        return self._btn_swap.isChecked()

    def update_unit_dropdown(self, choices: list[str], current: str) -> None:
        """
        Populate the unit combo with choices and select current.

        Called by PlotPanel.render() after every successful render.
        Shows the unit section when choices is non-empty; hides it otherwise.

        Signals are blocked during the programmatic update so that setting
        the combo value doesn't fire unit_changed and trigger a redundant
        compute cycle.
        """
        show = bool(choices)
        self._unit_section.setVisible(show)
        self._sep_before_unit.setVisible(show)
        if not show:
            return

        self._combo_unit.blockSignals(True)
        self._combo_unit.clear()
        self._combo_unit.addItems(choices)
        idx = self._combo_unit.findText(current)
        if idx >= 0:
            self._combo_unit.setCurrentIndex(idx)
        self._combo_unit.blockSignals(False)

    def current_unit(self) -> str:
        """Return the currently selected unit label (empty string if hidden)."""
        if not self._unit_section.isVisible():
            return ""
        return self._combo_unit.currentText()

    def sync_style(self, vis: str, interp: str) -> None:
        """
        Programmatically update the Style radio buttons WITHOUT emitting
        style_changed.  Called by PlotPanel.render() to keep the toolbar
        in sync with the plugin's DEFAULT_STYLE_* schema.

        blockSignals prevents the toggled callbacks from firing, which
        would otherwise trigger a redundant _on_style_changed → re-render loop.
        """
        all_rbs = [*self._rb_vis.values(), *self._rb_interp.values()]
        for rb in all_rbs:
            rb.blockSignals(True)

        if vis in self._rb_vis:
            self._rb_vis[vis].setChecked(True)
        if interp in self._rb_interp:
            self._rb_interp[interp].setChecked(True)

        # Mirror the Points→disable-interp constraint
        disabled = (vis == "Points") or not self._interp_allowed
        for rb in self._rb_interp.values():
            rb.setEnabled(not disabled)

        for rb in all_rbs:
            rb.blockSignals(False)

    def set_interp_allowed(self, allowed: bool) -> None:
        """Grey out Interpolation where it does not apply (the Histogram tab)."""
        self._interp_allowed = bool(allowed)
        points = self._rb_vis["Points"].isChecked()
        for rb in self._rb_interp.values():
            rb.setEnabled(self._interp_allowed and not points)

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_unit_changed(self, label: str) -> None:
        """Forwarded from combo — only fires on genuine user interaction."""
        if label:
            self.unit_changed.emit(label)

    def _switch_view(self, name: str):
        self._active_view = name
        self._refresh_tab_btns()
        self._apply_visibility(name)
        self.view_changed.emit(name)

    def _refresh_tab_btns(self):
        for name, btn in self._view_btns.items():
            active = (name == self._active_view)
            btn.setProperty("active", "true" if active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _apply_visibility(self, view: str):
        """Show/hide toolbar sections depending on active view mode."""
        is_matrix = (view == "Matrix")

        self._scale_section.setVisible(not is_matrix)
        self._sep_before_matrix.setVisible(is_matrix)
        self._matrix_section.setVisible(is_matrix)

        # Unit dropdown is only meaningful on plot views, never on Matrix.
        # update_unit_dropdown() controls show/hide for non-matrix views.
        if is_matrix:
            self._unit_section.hide()
            self._sep_before_unit.hide()

    def _set_notation(self, mode: str):
        for m, btn in self._notation_btns.items():
            btn.setChecked(m == mode)
        self.notation_changed.emit(mode)

    def _axis_lbl(self, letter: str) -> QLabel:
        lbl = QLabel(f"{letter}  ≡")
        lbl.setStyleSheet("color:#555; font-size:11px; padding:0 2px;")
        return lbl

    def _scale_combo(self) -> QComboBox:
        cb = QComboBox()
        cb.setObjectName("scale_combo")
        cb.addItems(["Linear", "Log"])
        cb.setCursor(Qt.CursorShape.PointingHandCursor)
        return cb

    @staticmethod
    def _chunk(flow: "_FlowLayout") -> QHBoxLayout:
        """A group of controls that stays together on one row of the toolbar."""
        box = QWidget()
        box.setObjectName("toolbar_group")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        flow.addWidget(box)
        return lay

    def _vsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background:#2a2a2a; max-width:1px; margin:6px 4px;")
        return sep


# ─────────────────────────────────────────────────────────────────────────────
#  COSMO HEADER BAR  (Phase 1 — "The Time Machine")
# ─────────────────────────────────────────────────────────────────────────────

class CosmoHeaderBar(QFrame):
    """
    Thin 22 px bar below the toolbar separator showing cosmological context.

    Layout (cosmological run)
    ─────────────────────────
      🌌  Snapshot: 015  │  z = 2.000  │  Age: 3.32 Gyr  │  Lookback: 10.45 Gyr

    Layout (non-cosmological run)
    ──────────────────────────────
      🔧  Non-cosmological run  │  t = 1.234 Gyr

    Colour palette
    ──────────────
      Background : #0a0a12  (deep indigo — distinct from teal primary + coral secondary)
      Text       : #7b8ec8  (muted indigo)
      Value      : #b0c0e8  (lighter indigo — stands out from label)
      Age        : #4ecdc4  (teal — inherits the primary accent)
      Error      : #e07b54  (coral — inherits the secondary accent)

    Hidden until set_cosmology_context() is called with a real dict.
    set_cosmology_context() never raises — errors are shown inline in the bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cosmo_header_bar")
        self.setFixedHeight(22)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(0)

        self._lbl_icon = QLabel("🌌")
        self._lbl_icon.setStyleSheet("font-size:11px; padding-right:8px;")
        lay.addWidget(self._lbl_icon)

        self._lbl_snapshot = self._seg_label()
        self._lbl_redshift = self._seg_label()
        self._lbl_age      = self._seg_label()
        self._lbl_lookback = self._seg_label()
        self._lbl_error    = self._seg_label(value_color="#e07b54")

        for lbl in (self._lbl_snapshot, self._lbl_redshift,
                    self._lbl_age, self._lbl_lookback, self._lbl_error):
            lay.addWidget(lbl)

        lay.addStretch()
        self.hide()

    # ── Public ────────────────────────────────────────────────────────────────

    def set_cosmology_context(self, ctx: dict) -> None:
        """Update bar from the dict returned by cosmology_context(). Never raises."""
        try:
            self._update(ctx)
        except Exception:
            pass   # bar failure must never crash the render pipeline

    # ── Private ───────────────────────────────────────────────────────────────

    def _update(self, ctx: dict) -> None:
        is_cosmo = bool(ctx.get("is_cosmological", False))
        snap_num = ctx.get("snapshot_num")
        time_gyr = ctx.get("time_gyr")
        age      = ctx.get("age_gyr")
        lookback = ctx.get("lookback_gyr")
        cosmo_nm = ctx.get("cosmo_name", "Planck18")
        err      = ctx.get("error")
        redshift = ctx.get("redshift")

        self._lbl_icon.setText("🌌" if is_cosmo else "🔧")

        # Snapshot number
        if snap_num is not None:
            self._seg(self._lbl_snapshot,
                      "Snapshot:", f"{snap_num:03d}", sep=True)
            self._lbl_snapshot.show()
        else:
            self._lbl_snapshot.hide()

        # Redshift (cosmo) or simulation time (non-cosmo)
        if is_cosmo and redshift is not None:
            self._seg(self._lbl_redshift, "z =", f"{redshift:.3f}", sep=True)
            self._lbl_redshift.show()
        elif not is_cosmo:
            if time_gyr is not None:
                self._seg(self._lbl_redshift,
                          "Non-cosmological run  │  t =",
                          f"{time_gyr:.3f} Gyr", sep=False)
            else:
                self._seg(self._lbl_redshift,
                          "Non-cosmological run", "", sep=False)
            self._lbl_redshift.show()
        else:
            self._lbl_redshift.hide()

        # Age of Universe
        if is_cosmo and age is not None:
            self._lbl_age.setText(
                f"<span style='color:#7b8ec8'>Age: </span>"
                f"<span style='color:#4ecdc4'>{age:.2f} Gyr</span>"
                f"<span style='color:#1c1c2e'>  │  </span>"
            )
            self._lbl_age.setToolTip(
                f"Age of the Universe at z = {redshift:.3f}\n"
                f"Calculated using {cosmo_nm} cosmology (Planck 2018)."
            )
            self._lbl_age.show()
        else:
            self._lbl_age.hide()

        # Lookback time
        if is_cosmo and lookback is not None:
            self._lbl_lookback.setText(
                f"<span style='color:#7b8ec8'>Lookback: </span>"
                f"<span style='color:#b0c0e8'>{lookback:.2f} Gyr</span>"
            )
            self._lbl_lookback.setToolTip(
                f"Lookback time from z = 0 to z = {redshift:.3f}\n"
                f"Calculated using {cosmo_nm} cosmology (Planck 2018)."
            )
            self._lbl_lookback.show()
        else:
            self._lbl_lookback.hide()

        # Error note (astropy absent or unphysical z)
        if err:
            short = err[:55] + "…" if len(err) > 55 else err
            self._lbl_error.setText(
                f"  <span style='color:#e07b54'>⚠ {short}</span>"
            )
            self._lbl_error.show()
        else:
            self._lbl_error.hide()

        self.show()

    def _seg(self, lbl: "QLabel", key: str, val: str, sep: bool) -> None:
        """Render a key/value segment with optional trailing divider."""
        tail = "<span style='color:#1c1c2e'>  │  </span>" if sep else ""
        lbl.setText(
            f"<span style='color:#7b8ec8'>{latex_to_html(key)} </span>"
            f"<span style='color:#b0c0e8'>{latex_to_html(val)}</span>"
            f"{tail}"
        )

    @staticmethod
    def _seg_label(value_color: str = "#b0c0e8") -> QLabel:
        lbl = QLabel("")
        lbl.setStyleSheet(
            "font-size:10px; padding:0;"
            "font-family:'SF Mono','Menlo','Consolas',monospace;"
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        return lbl


# ─────────────────────────────────────────────────────────────────────────────
#  META STRIP  (H5Web left column: n 584281 / x [D0])
# ─────────────────────────────────────────────────────────────────────────────

class MetaStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(72)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 14, 4, 8)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        row_n = QHBoxLayout()
        row_n.setSpacing(4)
        lbl_n = QLabel("n")
        lbl_n.setStyleSheet("color:#666; font-style:italic; font-size:13px;")
        self.lbl_count = QLabel("—")
        self.lbl_count.setStyleSheet("color:#c8c8c8; font-size:11px;")
        row_n.addWidget(lbl_n)
        row_n.addWidget(self.lbl_count)
        row_n.addStretch()
        lay.addLayout(row_n)

        row_x = QHBoxLayout()
        row_x.setSpacing(4)
        lbl_x = QLabel("x")
        lbl_x.setStyleSheet("color:#666; font-size:11px;")
        self.lbl_dim = QLabel("D0")
        self.lbl_dim.setStyleSheet(
            "color:#000; background:#4ecdc4; border-radius:3px;"
            "padding:1px 7px; font-size:10px; font-weight:bold;")
        row_x.addWidget(lbl_x)
        row_x.addWidget(self.lbl_dim)
        row_x.addStretch()
        lay.addLayout(row_x)

        lay.addStretch()

    def update(self, n: int, dim_label: str = "D0"):
        self.lbl_count.setText(f"{n:,}")
        self.lbl_dim.setText(dim_label)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT PANEL  (assembles everything — public API for the orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class PlotPanel(QWidget):
    """
    Top-level panel.  The orchestrator only ever calls the public methods
    below — it never touches internal widgets directly.
    """

    tab_changed             = pyqtSignal(str)
    scale_changed           = pyqtSignal(str, str)
    diagnostic_loaded       = pyqtSignal(object)
    diagnostic_cleared      = pyqtSignal()
    swap_z_order            = pyqtSignal()
    unit_changed            = pyqtSignal(str)
    style_changed           = pyqtSignal(str, str)   # (vis_mode, interp_mode)
    custom_bins_requested   = pyqtSignal(int)        # Phase 3: context-menu override

    _RGBA_RE = re.compile(r'^#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$')

    def __init__(self, parent=None):
        super().__init__(parent)

        self._export_array: np.ndarray | None = None
        self._export_path:  str               = ""
        # Plot tabs export what the plot shows, labelled as such:
        # (comment lines, [(column name, array, fmt), ...]) or None.
        self._export_columns = None
        self._export_text = None
        self._export_script = None       # Analyses: Python that reproduces the result

        # ── Style state cache ─────────────────────────────────────────────
        self._vis_mode:    str = "Line"
        self._interp_mode: str = "Linear"

        # ── Histogram bin tracking (for override dialog default value) ────
        self._current_n_bins: int = 64

        # ── Inspector state ───────────────────────────────────────────────
        # LOD-1 raw array for the Line inspector.
        # Integer-round snap (no argmin) reads y directly from lod1_raw.
        self._insp_lod1_raw: object = None   # np.ndarray | None (full LOD-1)
        self._insp_n_lod1:   int    = 0      # len(lod1_raw) = N_lod1
        self._insp_log_x:  bool              = False
        self._insp_active: bool              = False  # True only on Line tabs
        self._insp_dtype:  str               = "float64"
        # Global-index inspector fields
        self._insp_vset:         object = None  # VirtualSnapshotSet | None
        self._insp_dataset_path: str    = ""
        self._insp_n_total:      int    = 0
        self._insp_stride:       int    = 1
        self._insp_reading:      bool   = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────
        self._toolbar = ViewToolbar()
        self._toolbar.view_changed.connect(self._on_view_changed)
        self._toolbar.scale_changed.connect(self.scale_changed)
        self._toolbar.notation_changed.connect(self._on_notation_changed)
        self._toolbar.cell_width_changed.connect(self._on_cell_width_changed)
        self._toolbar.grid_toggled.connect(self._on_grid_toggled)
        self._toolbar.export_clicked.connect(self._do_export)
        self._toolbar.load_matrix_clicked.connect(self._do_load_diagnostic)
        self._toolbar.clear_diag_clicked.connect(self._on_clear_diagnostic)
        self._toolbar.swap_z_order_clicked.connect(self.swap_z_order)  # forward
        self._toolbar.unit_changed.connect(self.unit_changed)          # forward Phase 2
        self._toolbar.style_changed.connect(self._on_style_changed)   # Task 2
        outer.addWidget(self._toolbar)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#222; max-height:1px;")
        outer.addWidget(sep)

        # ── Phase 1: Cosmology context bar (hidden until file loaded) ─────
        self._cosmo_bar = CosmoHeaderBar()
        outer.addWidget(self._cosmo_bar)

        # ── Diagnostic override banner (hidden until a matrix is loaded) ──
        # Visible whenever diagnostic_override is active.
        # Shows: [DIAG] filename  shape/dtype[✕ clear]
        self._diag_banner = QFrame()
        self._diag_banner.setObjectName("diag_banner")
        self._diag_banner.setFixedHeight(26)
        db_lay = QHBoxLayout(self._diag_banner)
        db_lay.setContentsMargins(10, 0, 6, 0)
        db_lay.setSpacing(6)

        lbl_tag = QLabel("⚡ ARRAYS FROM FILES")
        lbl_tag.setObjectName("diag_tag_lbl")
        db_lay.addWidget(lbl_tag)

        self._lbl_diag_file = QLabel("")
        self._lbl_diag_file.setObjectName("diag_file_lbl")
        db_lay.addWidget(self._lbl_diag_file)

        db_lay.addStretch()

        self._lbl_diag_shape = QLabel("")
        self._lbl_diag_shape.setObjectName("diag_shape_lbl")
        db_lay.addWidget(self._lbl_diag_shape)

        self._diag_banner.hide()
        outer.addWidget(self._diag_banner)

        # ── Info bar ──────────────────────────────────────────────────────
        # Bug 2 fix: SizePolicy.Ignored tells the layout to stop honouring
        # this label's sizeHint() when computing minimum window width.
        # The label still displays its full text (word-wrapped) but will
        # never force the QMainWindow wider than its initial resize() size.
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("info_bar")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_info.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        outer.addWidget(self.lbl_info)

        # ── Middle: MetaStrip + stacked content ───────────────────────────
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)

        self._meta = MetaStrip()
        mid.addWidget(self._meta)

        self._stack = QStackedWidget()

        # Page 0 — pyqtgraph plot
        self.pw = pg.PlotWidget()
        self.pw.setBackground("#080808")
        self.pw.showGrid(x=True, y=True, alpha=0.18)
        self._stack.addWidget(self.pw)

        # ── Phase 3: CosmoViewer Tools context menu ───────────────────────
        # Graft our submenu onto the ViewBox's native right-click menu so the
        # full PyQtGraph menu still appears; we just append at the bottom.
        _vb = self.pw.getViewBox()
        _vb.menu.addSeparator()
        self._cv_ctx_menu = QMenu("CosmoViewer Tools", _vb.menu)
        _vb.menu.addMenu(self._cv_ctx_menu)
        self._act_override_bins = self._cv_ctx_menu.addAction(
            "Override Histogram Resolution…")
        self._act_override_bins.triggered.connect(self._on_override_bins_dialog)
        _vb.menu.aboutToShow.connect(self._update_context_menu_state)

        # ── Inspector (H5Web-style crosshair + tooltip) ───────────────────
        #
        # Architecture
        # ────────────
        # Crosshair: pg.InfiniteLine added to the ViewBox (NOT PlotItem).
        # ViewBox items survive pw.clear() / _hard_reset; PlotItem items are
        # wiped on every render cycle.  This means we create the line once
        # here and never have to recreate it.
        #
        # Tooltip: QLabel child of self.pw (pixel-coordinate child widget).
        # Placed via .move() in pixel space — not a scene item — so it is
        # immune to axis transforms and always renders at the correct screen
        # position regardless of log/linear mode.
        #
        # 60 Hz throttle: a QTimer(singleShot=True, interval=16 ms) debounces
        # the raw sigMouseMoved signal.  The handler stores the latest scene
        # position and the timer fires the actual argmin search at most once
        # per 16 ms frame.  NumPy argmin on 4 000 floats takes ~5 µs so there
        # is no risk of GUI starvation.

        # ── Inspector state (runtime) ─────────────────────────────────────
        # _insp_active updated by render() — see __init__ for cache fields.

        # ── Histogram LOD inspector state ──────────────────────────────────────
        # Set by render() from PlotRequest; then CosmoExplorer._ensure_bin_index
        # says whether the index is ready, building, or unavailable, and the
        # build's signals (_on_lod_ready/_failed/_progress) keep that current.
        # An open bin popup is redrawn on each change, so it never sits on
        # "building" after the build has ended.
        self._lod:             object          = None   # LodIndex | None
        self._lod_state:       str             = "unavailable"  # building|ready|failed|unavailable
        self._lod_msg:         str             = "no particle list for this view."
        self._lod_build:       object          = None   # LodBuildWorker being waited on
        self._lod_progress:    tuple           = (0, 0) # (files read, files in all)
        self._lod_cgs_edges:   object          = None   # bin edges the index must have
        self._lod_unit_label:  str             = ""
        self._tip_bin:         object          = None   # bin shown in the popup, or None
        self._tip_anchor:      object          = None   # scene point the popup points at
        self._unit_scalar:     float           = 1.0
        self._bin_edges_lod:   object          = None   # np.ndarray | None
        self._snap_path_lod:   str             = ""
        self._ds_path_lod:     str             = ""
        # _lod_key is built ONCE per render() call from the PlotRequest fields
        # and stored here.  _on_inspector_click() reads self._lod_key directly
        # so it never calls os.path.getmtime() at click-time (which would be
        # a network metadata operation on NFS/Lustre mounts, adding 1-50 ms).
        self._lod_key:         object          = None   # LodCacheKey | None

        # ── Line inspector global-read state ───────────────────────────────────
        # Set by render() for Line tabs; cleared on tab switch and reset_view().
        # vset is a read-only accessor — no Qt objects, no render state.
        self._insp_vset:         object = None  # VirtualSnapshotSet | None
        self._insp_dataset_path: str    = ""
        self._insp_n_total:      int    = 0
        # _insp_reading: True while read_particle() is executing on the main
        # thread.  Guards the 60 Hz timer from stacking calls on slow NFS.
        self._insp_reading:      bool   = False

        # ── Crosshair ─────────────────────────────────────────────────────
        self._insp_vline = pg.InfiniteLine(
            angle    = 90,
            movable  = False,
            pen      = pg.mkPen("#888888", width=1,
                                style=Qt.PenStyle.SolidLine),
        )
        self._insp_vline.setZValue(1000)   # always on top
        self.pw.getViewBox().addItem(self._insp_vline)
        self._insp_vline.hide()

        # ── Tooltip ───────────────────────────────────────────────────────
        # Child of pw so it clips correctly to the plot boundary.
        # WA_TransparentForMouseEvents prevents it from stealing mouse events.
        # Dark card in the app's palette (the Line hover readout uses it too).
        self._insp_tip = QLabel(self.pw)
        self._insp_tip.setObjectName("inspector_tip")
        self._insp_tip.setTextFormat(Qt.TextFormat.RichText)
        self._insp_tip.setStyleSheet(
            "QLabel#inspector_tip {"
            " background:#141418;"
            " color:#c8c8c8;"
            " border:1px solid #2c2c34;"
            " border-left:2px solid #4ecdc4;"
            " border-radius:6px;"
            " padding:7px 11px;"
            " font-size:11px;"
            "}"
        )
        self._insp_tip.setWordWrap(True)        # long messages wrap, not widen
        self._insp_tip.setMaximumWidth(380)
        self._insp_tip.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._insp_tip.hide()

        # ── 60 Hz throttle ────────────────────────────────────────────────
        self._insp_pending: object = None   # last scene QPointF received
        self._insp_timer = QTimer(self)
        self._insp_timer.setSingleShot(True)
        self._insp_timer.setInterval(16)    # ≈ 60 Hz
        self._insp_timer.timeout.connect(self._inspector_compute)

        # sigMouseMoved fires with a QPointF in scene coordinates
        self.pw.scene().sigMouseMoved.connect(self._inspector_mouse_moved)
        # Histogram click inspector
        self.pw.scene().sigMouseClicked.connect(self._on_inspector_click)

        # Page 1 — matrix table
        self._matrix_view = MatrixView()
        self._stack.addWidget(self._matrix_view)

        mid.addWidget(self._stack, stretch=1)
        outer.addLayout(mid, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────
        # Bug 2 fix: status strings can be 150+ characters (cosmo factor,
        # unit scalar, particle counts).  Without Ignored policy, QLabel
        # sets a sizeHint() equal to the full text width and the layout
        # stretches the QMainWindow to match — causing the ultra-wide window.
        # Ignored means: "use whatever space the layout gives you; don't
        # demand more."  Text is right-aligned so it clips from the left
        # gracefully if the window is narrower than the string.
        self.lbl_status = QLabel("Open a snapshot and select a dataset.")
        self.lbl_status.setObjectName("status_bar")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_status.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        outer.addWidget(self.lbl_status)

    # ═════════════════════════════════════════════════════════════════════════
    #  PUBLIC API  (called by CosmoExplorer)
    # ═════════════════════════════════════════════════════════════════════════

    def configure_for_plugin(self, plugin_cls):
        """
        Rebuild tab bar and reset scale defaults when the user selects a new
        dataset.  Matrix is always prepended so it works for every plugin.
        """
        tabs = ["Matrix"] + list(plugin_cls.TABS)
        self._toolbar.set_view_tabs(tabs, default=plugin_cls.TABS[0])
        self._toolbar.set_scale_defaults(
            plugin_cls.DEFAULT_SCALE_X,
            plugin_cls.DEFAULT_SCALE_Y,
        )
        # Clear unit dropdown when switching plugins (new plugin may have
        # different or no unit choices)
        self._toolbar.update_unit_dropdown([], "")

    def set_cosmology_context(self, ctx: dict) -> None:
        """
        Phase 1 — update the cosmology header bar.

        ctx is the dict returned by astro_units.cosmology_context(), augmented
        with snapshot_num by the orchestrator.  Delegates to
        CosmoHeaderBar.set_cosmology_context() which never raises.
        """
        self._cosmo_bar.set_cosmology_context(ctx)

    def render(self, request: PlotRequest, vset=None):
        """Switch to plot view and draw a PlotRequest.

        vset : VirtualSnapshotSet | None — injected by density_explorer for
               Line-tab inspector reads.  Not stored in PlotRequest (which is
               a pure-data boundary object with no live file handles).
        """
        self._stack.setCurrentIndex(0)

        # Preserve zoom state on PRESENTATION_CHANGE renders (unit,
        # scale-label, bin-count).  Save the current view range NOW, before
        # _hard_reset clears it.  Only preserve when log-mode is unchanged —
        # a linear→log switch cannot meaningfully reuse a linear range.
        _is_presentation = getattr(request, 'is_presentation', False)
        # The tab must be unchanged too: Histogram and Line have different x axes (the
        # value against the particle index), so a kept range put every point off screen.
        _log_unchanged   = (
            request.log_x == getattr(self, '_last_log_x', None)
            and request.log_y == getattr(self, '_last_log_y', None)
            and request.tab == getattr(self, '_last_tab', None)
        )
        _saved_range = (
            self.pw.viewRange()
            if _is_presentation and _log_unchanged
            else None
        )

        self._hard_reset()

        # ── Declarative style sync (Phase 3) ─────────────────────────────
        # Apply plugin-requested style BEFORE drawing so _draw_curve uses
        # the correct _vis_mode/_interp_mode.  blockSignals is inside sync_style.
        self._toolbar.set_interp_allowed(request.tab != "Histogram")
        if request.req_vis_mode:
            vis, interp = _validate_plot_style(
                request.req_vis_mode, request.req_interp_mode)
            self._vis_mode    = vis
            self._interp_mode = interp
            self._toolbar.sync_style(vis, interp)

        # Track active bin count so the override dialog shows the right default
        if request.n_bins > 0:
            self._current_n_bins = request.n_bins

        pw = self.pw
        pi = pw.getPlotItem()

        pw.setLogMode(x=request.log_x, y=request.log_y)

        # ── Track 2.3: AutoSIPrefix guard ─────────────────────────────────
        # Disable the automatic SI prefix (e.g. "×1e+12") on the Y axis for
        # dimensionless datasets like ParticleIDs.  Re-enable for all others
        # so that dimensional datasets (Density, Mass, etc.) are unaffected.
        # On a log axis pyqtgraph scales the prefix by the log values themselves
        # ("g (x1e+27)" under ticks of 1e12-1e16), so log axes never get one.
        _left_axis = pw.getAxis('left')
        _left_axis.enableAutoSIPrefix(request.unit_dimension != "dimensionless"
                                      and not request.log_y)
        pw.getAxis('bottom').enableAutoSIPrefix(not request.log_x)

        pw.setTitle(
            f'<span style="color:#fff;font-size:13pt;font-weight:bold;">'
            f'{request.title}</span>',
            justify="center",
        )
        # ── Track 3.2: Route all axis labels and UI strings through the
        # LaTeX → HTML converter before handing to Qt widgets.
        pw.setLabel("bottom", latex_to_html(request.x_label))
        pw.setLabel("left",   latex_to_html(request.y_label))

        needs_legend = any(c.label for c in request.curves)
        legend       = pi.addLegend(offset=(10, 10)) if needs_legend else None

        # ── Task 3.1: draw hooks (markers, regions) BEFORE curves ────────
        # This guarantees scientific annotations survive even if curve styling
        # raises an exception — the render cycle continues past the failure.
        for marker in request.markers:
            ThemeManager.resolve_marker(marker)
            self._draw_marker(pw, marker, request.log_x, request.log_y)
        for region in request.regions:
            ThemeManager.resolve_region(region)
            self._draw_region(pw, region, request.log_x)

        for curve  in request.curves:
            ThemeManager.resolve_curve(curve)
            # A histogram's own curve is drawn on its bin edges (see _histogram_outline).
            hist_edges = (request.bin_edges if request.tab == "Histogram"
                          and curve.role == "primary_data" else None)
            self._draw_curve(pw, curve, legend, request.log_y, hist_edges=hist_edges)
        for image  in request.images:  self._draw_image(pw, image)

        self.lbl_info.setText(latex_to_html(request.info))
        self.lbl_status.setText(latex_to_html(request.status))
        self._meta.update(request.n_points, request.dim_label)

        # Export what this view shows, and only that: a view with no curves (an analysis that
        # failed, or a picture such as the phase diagram) clears the previous view's data,
        # which Export used to write under this view's name.
        self._export_array = None
        self._export_columns = None
        self._export_text = None
        self._export_script = None
        if request.curves:
            self._export_array = request.curves[0].y
            self._export_path  = request.tab
            self._export_columns = self._plot_export_columns(request)
            self._export_script = getattr(request, "script", None)
            self._export_none_reason = ""
        elif request.title == "Error":
            self._export_path = ""
            self._export_none_reason = "Nothing to export: this view failed (see the status line)."
        else:
            self._export_path = ""
            self._export_none_reason = ("Nothing to export from this picture; "
                                        "the Matrix tab exports its table.")

        # ── Inspector cache ───────────────────────────────────────────────
        if request.tab == "Line" and request.curves:
            self._insp_lod1_raw = request.lod1_raw
            self._insp_n_lod1   = request.lod1_n_lod1
            self._insp_log_x    = request.log_x
            self._insp_dtype    = request.insp_dtype or "float64"
            self._insp_active   = True
            # Global-index fields for read_particle
            self._insp_vset         = vset
            self._insp_dataset_path = getattr(request, "dataset_path", "")
            self._insp_n_total      = getattr(request, "lod1_n_total", 0)
            self._insp_stride       = getattr(request, "lod1_stride", 1) or 1
        else:
            self._insp_active       = False
            self._insp_lod1_raw     = None
            self._insp_n_lod1       = 0
            self._insp_vset         = None
            self._insp_dataset_path = ""
            self._insp_n_total      = 0
            self._insp_stride       = 1
            self._insp_vline.hide()
            self._insp_tip.hide()

        # ── Wire LOD inspector fields from PlotRequest ───────────────────
        # Runs on every render() call (Histogram and non-Histogram alike).
        # On Histogram tab: copy the four LOD fields from the request and
        #   build _lod_key once so _on_inspector_click() needs no getmtime().
        # On other tabs: clear all fields so stale state from a previous
        #   Histogram render never leaks into a Line or Matrix render.
        if request.tab == "Histogram":
            self._unit_scalar   = request.unit_scalar
            self._bin_edges_lod = request.bin_edges
            self._snap_path_lod = request.snapshot_path
            self._ds_path_lod   = request.dataset_path
            self._lod_unit_label = request.unit_label or ""
            scalar = float(request.unit_scalar) or 1.0
            self._lod_cgs_edges = (None if request.bin_edges is None else
                                   np.asarray(request.bin_edges, dtype=np.float64) / scalar)
            # No index is assumed until the orchestrator says which one applies
            # (CosmoExplorer._ensure_bin_index, right after this render).
            self._lod       = None
            self._lod_build = None
            self._lod_state = "unavailable"
            self._lod_msg   = "no particle list for this view."
            self._tip_bin   = None
            # Build _lod_key from the request fields — the ONE getmtime() call
            # per render.  Skip when paths are absent (diagnostic override).
            if self._snap_path_lod and self._ds_path_lod:
                try:
                    self._lod_key = LodCacheKey.build(
                        snapshot_path = self._snap_path_lod,
                        dataset_path  = request.lod_key_path or self._ds_path_lod,
                        cache_dir     = LodIndex.CACHE_DIR,
                    )
                except Exception:
                    self._lod_key = None
            else:
                self._lod_key = None
            self._protect_lod_key()
            # A cache-hit render may carry an index; use it only if it is this one.
            if request.lod is not None:
                self._on_lod_ready(request.lod)
        else:
            # Non-Histogram tab: invalidate all LOD state.
            self._lod           = None
            self._lod_build     = None
            self._lod_state     = "unavailable"
            self._lod_cgs_edges = None
            self._tip_bin       = None
            self._lod_key       = None
            self._protect_lod_key()
            self._bin_edges_lod = None
            self._snap_path_lod = ""
            self._ds_path_lod   = ""

        # Phase 2 — update unit dropdown from PlotRequest metadata.
        # blockSignals is handled inside update_unit_dropdown; no spurious
        # unit_changed signal fires during this programmatic update.
        self._toolbar.update_unit_dropdown(
            request.unit_choices,
            request.unit_label,
        )

        # Restore the saved view range AFTER all drawing (including
        # setLogMode above).  Calling setLogMode can trigger an internal
        # autoRange — applying the range here guarantees it wins last.
        if _saved_range is not None:
            _vb = self.pw.getViewBox()
            _vb.disableAutoRange()
            self.pw.getPlotItem().setRange(
                xRange=_saved_range[0], yRange=_saved_range[1], padding=0)

        # Track last-rendered log mode so the next render can detect scale changes.
        self._last_log_x = request.log_x
        self._last_tab = request.tab
        self._last_log_y = request.log_y

        QApplication.processEvents()

    def render_matrix(self, data_arrays: dict[str, np.ndarray], path: str,
                      labels: list | None = None, notes: list | None = None,
                      info: str = "", status: str = "", cells: list | None = None,
                      script: str | None = None):
        """
        Switch to matrix view and populate the table.

        If called while diagnostic override is active this method is never
        reached — the orchestrator branches on Matrix tab before this.

        labels / notes / info / status come from analysis plugins: named
        columns, export header lines and their own info and status text.
        cells (rows of strings) makes it a text table, exported as CSV text.
        """
        self._stack.setCurrentIndex(1)

        arr = data_arrays.get(path)
        if arr is None:
            self.set_status(f"No data found for path: {path}")
            return

        self._export_array = arr
        self._export_path  = path
        self._export_columns = None
        self._export_text = None
        self._export_script = script

        self._matrix_view.load(arr, path, self._toolbar.notation(), labels=labels, cells=cells)
        if cells:
            self._export_text = (list(notes or []), list(labels or []), cells)
            self._meta.update(len(cells), "D0")
            self.lbl_info.setText(latex_to_html(info) if info else path)
            self.set_status(status or f"Matrix — {path}  |  {len(cells):,} rows")
            QApplication.processEvents()
            return
        self._meta.update(len(arr), "D0")

        if labels:
            cols = _as_columns(arr).astype(np.float64)
            self._export_columns = (list(notes or []), [
                (name.replace(",", " "), cols[:, j], "%.10e") for j, name in enumerate(labels)])
            self.lbl_info.setText(latex_to_html(info) if info else path)
            self.set_status(status or f"Matrix — {path}  |  {len(arr):,} rows")
            QApplication.processEvents()
            return

        self.lbl_info.setText(
            f"{path}  |  shape: {arr.shape}  |  dtype: {arr.dtype}  |  "
            f"min: {float(np.nanmin(arr)):.4e}  |  "
            f"max: {float(np.nanmax(arr)):.4e}  |  "
            f"mean: {float(np.nanmean(arr)):.4e}"
        )
        self.set_status(
            f"Matrix — {path}  |  {len(arr):,} values  |  "
            f"showing first {min(len(arr), MAX_MATRIX_DISPLAY):,}"
        )

        QApplication.processEvents()

    def reset_view(self):
        """Clear plot and info bar (called on new file open)."""
        self._hard_reset()
        self.lbl_info.setText("")
        self._export_array = None
        self._export_path  = ""
        self._export_columns = None
        self._export_text = None
        self._export_script = None
        # Clear inspector so stale arrays from the previous file are never used
        self._insp_active       = False
        self._insp_lod1_raw     = None
        self._insp_n_lod1       = 0
        self._insp_vline.hide()
        self._insp_tip.hide()
        # Clear LOD state so a stale index from the previous file
        # does not answer inspector clicks on the newly loaded dataset.
        self._lod           = None
        self._lod_build     = None
        self._lod_state     = "unavailable"
        self._lod_cgs_edges = None
        self._tip_bin       = None
        self._lod_key       = None
        self._bin_edges_lod = None
        self._snap_path_lod = ""
        self._ds_path_lod   = ""
        # Clear Line inspector state.
        self._insp_vset         = None
        self._insp_dataset_path = ""
        self._insp_n_total      = 0
        self._insp_stride       = 1
        self._insp_reading      = False

    def show_diagnostic_banner(self, files: list):
        """
        Update the diagnostic override banner.

        files : list of (ndarray, filename) tuples — 1 or 2 entries.
        """
        names  = [f[1] for f in files]
        shapes = "  ".join(f"{f[1]}: {f[0].shape}" for f in files)

        self._lbl_diag_file.setText("  ⇄  ".join(names))
        self._lbl_diag_shape.setText(shapes)
        self._diag_banner.show()

        # Button label: show filename if 1 file, count if 2
        btn_label = names[0] if len(names) == 1 else f"{len(names)} files loaded"
        self._toolbar.set_load_active(btn_label)

    def hide_diagnostic_banner(self):
        """Hide the diagnostic override banner."""
        self._diag_banner.hide()
        self._toolbar.set_load_idle()
        self.hide_swap_button()

    def show_swap_button(self):
        """Show ⇅ toggle (only called when exactly 2 files are loaded)."""
        self._toolbar.show_swap_button()

    def hide_swap_button(self):
        """Hide ⇅ toggle."""
        self._toolbar.hide_swap_button()

    def render_diagnostic(self, arr: np.ndarray, filename: str):
        """
        Render a raw array bypassing all plugin logic.

        Shape routing  (Snag 1 from PM spec — never crash on unexpected shape)
        ─────────────
          1-D  [N]      → line plot (Index vs Value)
          2-D  [H, W]   → heatmap via pg.ImageItem
          >2-D          → reject with friendly error in status bar

        Metadata fallback  (Snag 2 from PM spec)
        ─────────────────
          No HDF5 attributes available.  All labels default to generic strings.
          Bounds are derived directly from the array min/max.
        """
        ndim = arr.ndim

        if ndim > 2:
            self.set_status(
                f"⚠  Cannot render {ndim}-D array '{filename}'  —  "
                f"only 1-D and 2-D arrays are supported.  "
                f"Shape: {arr.shape}"
            )
            return

        self._export_array = arr.flatten()
        self._export_path  = filename
        self._export_columns = None
        self._export_text = None
        self._export_script = None

        # Cache for swap re-render
        self._diag_arr      = arr
        self._diag_filename = filename

        if ndim == 1:
            self._render_diagnostic_1d(arr, filename)
        else:
            self._render_diagnostic_2d(arr, filename)

    def set_status(self, msg: str):
        self.lbl_status.setText(msg)
        QApplication.processEvents()

    # ── Phase 3: context menu helpers ─────────────────────────────────────────

    def _update_context_menu_state(self) -> None:
        """Enable the bin-override action only when a Histogram is active."""
        self._act_override_bins.setEnabled(self.current_tab() == "Histogram")

    def _on_override_bins_dialog(self) -> None:
        """Spawn the dark-themed slider+spinbox bin-count dialog."""
        dlg = BinResolutionDialog(self, self._current_n_bins)
        if dlg.exec():
            self.custom_bins_requested.emit(dlg.value())

    def current_tab(self) -> str:
        return self._toolbar.active_view()

    def current_scale_x(self) -> str:
        return self._toolbar.current_scale_x()

    def current_scale_y(self) -> str:
        return self._toolbar.current_scale_y()

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE — signal handlers
    # ═════════════════════════════════════════════════════════════════════════

    def _on_view_changed(self, view: str):
        """Forward to orchestrator as tab_changed."""
        self.tab_changed.emit(view)


    def _on_style_changed(self, vis_mode: str, interp_mode: str) -> None:
        """
        Task 4: Intercept style-changed signal, update cached state, apply
        visual mutations to all currently rendered primary/secondary/ghost data items.

        Non-PlotDataItem items (markers, regions, crosshair) are skipped — 
        they must never receive symbol/pen mutations.
        """
        # Task 4.1 — update cache
        self._vis_mode    = vis_mode
        self._interp_mode = interp_mode

        # Task 4.1 — iterate over all items on the plot
        for item in self.pw.getPlotItem().listDataItems():
            # Task 4.2 — strict type check: only PlotDataItem
            if not isinstance(item, pg.PlotDataItem):
                continue
            # Task 4.2 — filter by semantic role: primary, secondary, and ghost_context (Line tab)
            role = getattr(item, '_plot_role', '')
            if role not in ('primary_data', 'secondary_data', 'ghost_context'):
                continue

            color = getattr(item, '_plot_color', '#00b4d8')
            self._apply_style_to_item(item, color, vis_mode, interp_mode)

        # Force an immediate repaint so 1-D Line mode refreshes without
        # requiring a user interaction (pan / zoom) to trigger a Qt paint event.
        self.pw.getViewBox().update()
        QApplication.processEvents()

        self.style_changed.emit(vis_mode, interp_mode)

    def _on_clear_diagnostic(self):
        """✕ clicked — emit to orchestrator which handles full state restoration."""
        self.diagnostic_cleared.emit()

    def _on_notation_changed(self, notation: str):
        self._matrix_view.set_notation(notation)

    def _on_cell_width_changed(self, width: int):
        self._matrix_view.set_cell_width(width)

    def _on_grid_toggled(self, on: bool):
        self.pw.showGrid(x=on, y=on, alpha=0.18)

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE — diagnostic matrix loader
    # ═════════════════════════════════════════════════════════════════════════

    def _do_load_diagnostic(self):
        """
        Multi-select file dialog — user picks 0, 1, or 2 files.

          0 selected : no-op
          1 selected : solo view, no swap button
          2 selected : overlay view, swap button visible
          3+ selected: first 2 used, rest ignored with a status notice

        Emits diagnostic_loaded(list of (ndarray, filename) tuples).
        The orchestrator stores the list and calls DiagnosticPlugin.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Compare arrays from files — select 1 or 2 files", "",
            "NumPy / CSV Files (*.npy *.csv *.txt);;All Files (*)",
        )
        if not paths:
            return

        if len(paths) > 2:
            self.set_status(
                f"⚠  {len(paths)} files selected — only the first 2 will be used."
            )
            paths = paths[:2]

        files: list[tuple] =[]

        for path in paths:
            filename = os.path.basename(path)
            self.set_status(f"Parsing {filename}…")
            QApplication.processEvents()

            try:
                ext = os.path.splitext(path)[1].lower()
                arr = (np.load(path, allow_pickle=False).astype(np.float64)
                       if ext == ".npy" else self._parse_csv(path))
            except Exception:
                self.set_status(f"⚠  {filename} could not be read as numbers: use a .npy array, "
                                f"or a .csv or .txt file of numbers.")
                continue

            if arr is None or arr.size == 0:
                self.set_status(f"⚠  No numeric data in {filename}")
                continue

            if arr.ndim > 2:
                self.set_status(
                    f"⚠  {filename}: {arr.ndim}-D array not supported "
                    f"(only 1-D and 2-D)."
                )
                continue

            files.append((arr, filename))

        if not files:
            return

        self.diagnostic_loaded.emit(files)


    @staticmethod
    def _parse_csv(path: str) -> np.ndarray:
        """
        Robustly read a CSV to a 1-D float64 array.

        Algorithm
        ─────────
        1. Read lines; skip blank lines and lines starting with '#'.
        2. Inspect the first surviving line to detect header vs data
           and to count columns.
        3. Determine skip_header (0 or 1) and which column to extract.
        4. Read with numpy.genfromtxt using those parameters.

        Never discards data rows.
        """
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        # Find first non-blank, non-comment line
        first_data_lineno = None   # 0-based index into all_lines
        for i, line in enumerate(all_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_data_lineno = i
                break

        if first_data_lineno is None:
            return np.array([], dtype=np.float64)

        first_line = all_lines[first_data_lineno].strip()
        parts      = [p.strip() for p in first_line.split(",")]
        n_cols     = len(parts)

        # Header detection: try to parse the last field as a float
        has_header = False
        try:
            float(parts[-1])
        except ValueError:
            has_header = True

        skip_header = first_data_lineno + (1 if has_header else 0)

        # Column selection
        if n_cols == 1:
            usecols = (0,)
        elif n_cols == 2:
            usecols = (1,)          # our export: (index, value)
        else:
            usecols = (n_cols - 1,) # last column as conservative default

        data = np.genfromtxt(
            path,
            delimiter   = ",",
            skip_header = skip_header,
            usecols     = usecols,
            dtype       = np.float64,
            invalid_raise = False,   # NaN for bad rows, don't crash
        )

        if data.ndim == 2:
            data = data[:, 0]

        # Remove any all-NaN rows at the end (genfromtxt artefact on empty lines)
        last_valid = np.where(np.isfinite(data))[0]
        if len(last_valid):
            data = data[: last_valid[-1] + 1]

        return data.astype(np.float64)

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE — export
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _plot_export_columns(request):
        """Columns for exporting a Histogram or Line plot, or None."""
        curve = request.curves[0]
        y = np.asarray(curve.y, dtype=np.float64)
        what = request.title or "dataset"
        if request.tab == "Histogram" and request.bin_edges is not None:
            edges = np.asarray(request.bin_edges, dtype=np.float64)
            if len(edges) != len(y) + 1:
                return None
            counts = np.nan_to_num(y, nan=0.0)   # empty bins are NaN for the log renderer
            unit = request.unit_label or request.x_label
            notes = [f"# Histogram of {what}; bin edges in {unit}",
                     f"# {request.y_label}; total = {int(counts.sum()):,}"]
            return notes, [("bin_low", edges[:-1], "%.8e"), ("bin_high", edges[1:], "%.8e"),
                           ("count", counts, "%d")]
        if request.tab == "Line":
            x = np.asarray(curve.x, dtype=np.float64)
            notes = [f"# Line view of {what}: min-max downsampled for display, "
                     f"{len(y):,} points; NOT the full dataset (use the Matrix tab for that)",
                     "# particle_index is approximate (position in the strided sample "
                     "mapped to the full array)"]
            return notes, [("particle_index", x, "%.0f"), (what.replace(",", " "), y, "%.8e")]
        return PlotPanel._curve_export_columns(request)

    @staticmethod
    def _curve_export_columns(request):
        """
        Columns for exporting any other plot (the analyses' views): every curve drawn, each with
        its x values. Curves sharing their x values share one x column; others get their own,
        padded with nan to the longest. Unlabelled curves after a labelled one with the same x
        are its error band (named "+ err" / "- err").
        """
        def plain(text):
            return re.sub(r"<[^>]+>", "", str(text or "")).strip()

        def clean(text):                         # a column name: no commas
            return plain(text).replace(",", " ")

        xs = [np.asarray(c.x, dtype=np.float64).ravel() for c in request.curves]
        ys = [np.asarray(c.y, dtype=np.float64).ravel() for c in request.curves]
        if any(len(x) != len(y) for x, y in zip(xs, ys)):
            return None
        names, base = [], None
        for i, c in enumerate(request.curves):
            if c.label:
                base = (i, clean(c.label))
                names.append(base[1])
                continue
            j = base[0] if base else None
            if j is not None and xs[j].shape == xs[i].shape and np.array_equal(xs[j], xs[i]):
                sign = "+" if np.nanmean(ys[i] - ys[j]) >= 0 else "-"
                names.append(f"{base[1]} {sign} err")
            else:
                names.append(clean(request.y_label) or f"curve {i + 1}")
        groups = []                              # [x array, [curve indices]]
        for i, x in enumerate(xs):
            for g in groups:
                if g[0].shape == x.shape and np.array_equal(g[0], x):
                    g[1].append(i)
                    break
            else:
                groups.append([x, [i]])
        n = max(len(x) for x in xs)

        def pad(a):
            return np.concatenate([a, np.full(n - len(a), np.nan)]) if len(a) < n else a

        x_name = clean(request.x_label) or "x"
        cols = []
        for k, (x, members) in enumerate(groups):
            cols.append((x_name if k == 0 else f"{x_name} for {names[members[0]]}", pad(x), "%.10e"))
            cols += [(names[i], pad(ys[i]), "%.10e") for i in members]
        notes = [f"# {plain(request.title) or 'plot'}: the {request.tab} view as drawn, "
                 f"{len(xs)} curve(s)" + ("; columns padded with nan" if len(groups) > 1 else "")]
        notes += [f"# {line}" for line in (plain(request.status), plain(request.info)) if line]
        if request.y_label:
            notes.append(f"# y: {plain(request.y_label)}")
        notes.append("# the Matrix tab exports the full result table with its provenance")
        return notes, cols

    def _do_export(self):
        """
        Export the current view to a CSV file.

        Matrix tab and loaded arrays: index, value for every element.
        Histogram tab: bin_low, bin_high, count.  Line tab: the downsampled
        curve, with a comment saying so.
        Uses numpy.savetxt for C-level write performance (~1–2 s for 25 M rows).
        """
        if self._export_array is None:
            self.set_status(getattr(self, "_export_none_reason", "")
                            or "Nothing to export — select a dataset first.")
            return

        default_name = (
            self._export_path.replace("/", "_").lstrip("_") + ".csv"
        )
        filters = "CSV Files (*.csv);;All Files (*)"
        if getattr(self, "_export_script", None):
            # Analyses: a script that recomputes the result headlessly (plan F4)
            filters = "CSV Files (*.csv);;Python script that reproduces it (*.py);;All Files (*)"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export Dataset to CSV", default_name, filters,
        )
        if not path:
            return
        if chosen.startswith("Python") and not path.endswith(".py"):
            path = os.path.splitext(path)[0] + ".py"
        self.export_to(path)

    def export_to(self, path: str):
        """Write the current export to path (CSV; .py writes the analysis script)."""
        if self._export_array is None:
            self.set_status(getattr(self, "_export_none_reason", "")
                            or "Nothing to export — select a dataset first.")
            return
        if path.endswith(".py") and getattr(self, "_export_script", None):
            csv_path = os.path.splitext(path)[0] + ".csv"
            text = self._export_script.replace("OUT = 'result.csv'", f"OUT = {csv_path!r}")
            try:
                with open(path, "w") as fh:
                    fh.write(text)
                self.set_status(f"Wrote a script that reproduces {self._export_path} → "
                                f"{os.path.basename(path)} (it writes {os.path.basename(csv_path)})")
            except Exception as exc:
                self.set_status(f"Export failed: {exc}")
            return

        self.set_status("Exporting…  (may take a few seconds for large arrays)")
        QApplication.processEvents()

        if getattr(self, "_export_text", None) is not None:
            import csv
            notes, header, rows = self._export_text
            try:
                with open(path, "w", newline="") as fh:
                    for n in notes:
                        fh.write(n + "\n")
                    w = csv.writer(fh)
                    if header:
                        w.writerow(header)
                    w.writerows(rows)
                self.set_status(f"Exported {len(rows):,} rows of {self._export_path} "
                                f"with provenance header → {os.path.basename(path)}")
            except Exception as exc:
                self.set_status(f"Export failed: {exc}")
            return

        if self._export_columns is not None:
            notes, cols = self._export_columns
            try:
                np.savetxt(
                    path, np.column_stack([c[1] for c in cols]),
                    delimiter = ",",
                    header    = "\n".join(notes + [",".join(c[0] for c in cols)]),
                    comments  = "",
                    fmt       = [c[2] for c in cols],
                )
                if self._export_path.startswith("Analyses/"):
                    self.set_status(
                        f"Exported {len(cols[0][1]):,} rows of {self._export_path} "
                        f"with provenance header → {os.path.basename(path)}")
                else:
                    self.set_status(
                        f"Exported {len(cols[0][1]):,} {self._export_path.lower()} rows "
                        f"(what the plot shows, not the dataset) → {os.path.basename(path)}")
            except Exception as exc:
                self.set_status(f"Export failed: {exc}")
            return

        try:
            arr          = self._export_array
            dataset_name = self._export_path.split("/")[-1]
            cols         = _as_columns(arr).astype(np.float64)
            k            = cols.shape[1]
            indices      = np.arange(len(arr), dtype=np.int64)
            out          = np.column_stack([indices, cols])
            names        = ([dataset_name] if k == 1 else
                            [f"{dataset_name}[{c}]" for c in _column_labels(k)])
            np.savetxt(
                path, out,
                delimiter = ",",
                header    = "index," + ",".join(names),
                comments  = "",
                fmt       = ["%d"] + ["%.8e"] * k,
            )
            self.set_status(
                f"Exported {len(arr):,} rows → {os.path.basename(path)}"
            )
        except Exception as exc:
            self.set_status(f"Export failed: {exc}")

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE — LOD histogram inspector
    # ═════════════════════════════════════════════════════════════════════════

    def _protect_lod_key(self) -> None:
        """Keep the on-screen field's index out of reach of the cache cap."""
        LodIndex.PROTECTED_KEYS.clear()
        if self._lod_key is not None:
            LodIndex.PROTECTED_KEYS.add(self._lod_key.hex_key)

    def _lod_is_current(self, lod: object) -> bool:
        """True when `lod` was built for the histogram on screen (same file,
        dataset view and bin edges); a bin-count change makes it stale."""
        return (lod is not None and self._lod_cgs_edges is not None
                and hasattr(lod, "matches") and lod.matches(self._lod_key, self._lod_cgs_edges))

    def set_bin_index_building(self, worker: object) -> None:
        """The orchestrator started (or is already running) `worker` for this histogram."""
        if self._lod_state == "building" and self._lod_build is worker:
            return
        self._lod_build    = worker
        self._lod_state    = "building"
        self._lod_progress = (0, 0)
        self._refresh_bin_tip()

    def set_bin_index_unavailable(self, why: str) -> None:
        """No particle list can be given for this histogram; `why` says so in the popup."""
        self._lod       = None
        self._lod_build = None
        self._lod_state = "unavailable"
        self._lod_msg   = why
        self._refresh_bin_tip()

    def _on_lod_ready(self, lod: object) -> None:
        """
        Slot for LodBuildWorker.buildComplete (and for an index already built).

        Called on the main thread. Takes the index only if it belongs to the
        histogram on screen: a build for an earlier binning is ignored.
        """
        if not self._lod_is_current(lod):
            return
        self._lod       = lod
        self._lod_build = None
        self._lod_state = "ready"
        self._refresh_bin_tip()

    def _on_lod_failed(self, worker: object, why: str) -> None:
        """Slot for LodBuildWorker.buildFailed: say so instead of waiting for ever."""
        if worker is not self._lod_build:
            return
        self._lod_build = None
        self._lod_state = "failed"
        self._lod_msg   = why
        self._refresh_bin_tip()

    def _on_lod_progress(self, worker: object, done: int, total: int) -> None:
        """Slot for LodBuildWorker.progress: snapshot files read so far."""
        if worker is not self._lod_build:
            return
        self._lod_progress = (done, total)
        self._refresh_bin_tip()

    def _on_inspector_click(self, event) -> None:
        """
        Slot connected to pw.scene().sigMouseClicked.

        Active only on the Histogram tab. Finds the bin under the click (with
        np.histogram's rule, on the display-unit edges the bars were drawn
        with) and shows a popup with that bin's range and, once the index is
        built, how many particles it holds and their row numbers in the
        snapshot dataset. While the index is being built the popup says so,
        with progress, and fills itself in when the build ends; if the build
        fails or no list can be given, it says why. A click outside the plot
        area closes it.
        """
        # Guard 1: only act on left-button single clicks
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Guard 2: only on Histogram tab
        if self.current_tab() != "Histogram":
            return

        # Guard 3: bin_edges must be populated (comes from PlotRequest)
        if self._bin_edges_lod is None:
            return

        # Map click scene position → view (data) coordinate
        vb        = self.pw.getViewBox()
        scene_pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(scene_pos):
            self._tip_bin = None
            self._insp_tip.hide()
            return
        mp = vb.mapSceneToView(scene_pos)

        # In log-x mode pyqtgraph's view coordinate is log10 of the value.
        log_x = bool(vb.state.get("logMode", [False, False])[0])
        try:
            display_val = 10.0 ** mp.x() if log_x else float(mp.x())
        except (OverflowError, ValueError):
            return

        # Bin under the click, on the display-unit edges the bars were drawn with.
        bin_edges = np.asarray(self._bin_edges_lod, dtype=np.float64)
        n_bins    = len(bin_edges) - 1
        if n_bins < 1 or not (bin_edges[0] <= display_val <= bin_edges[-1]):
            self._tip_bin = None              # beside the histogram: nothing to show
            self._insp_tip.hide()
            return
        bin_index = int(np.searchsorted(bin_edges, display_val, side="right")) - 1
        self._tip_bin    = max(0, min(bin_index, n_bins - 1))
        self._tip_anchor = scene_pos
        self._refresh_bin_tip(force=True)

    def _refresh_bin_tip(self, force: bool = False) -> None:
        """Redraw the bin popup from the current index state, if it is open."""
        if self._tip_bin is None or not (force or self._insp_tip.isVisible()):
            return
        self._insp_tip.setText(self._bin_tip_html(self._tip_bin))
        self._insp_tip.adjustSize()
        # Position near the click, flipped left/up when it would leave the plot.
        pt    = self.pw.mapFromScene(self._tip_anchor)
        tw, th = self._insp_tip.width(), self._insp_tip.height()
        px, py = int(pt.x()) + 14, int(pt.y()) - th // 2
        if px + tw + 6 > self.pw.width():
            px = int(pt.x()) - tw - 14
        py = max(4, min(py, self.pw.height() - th - 4))
        self._insp_tip.move(max(4, px), py)
        self._insp_tip.show()
        self._insp_tip.raise_()

    def _bin_tip_html(self, b: int) -> str:
        """Popup text for bin `b` (0-based) in the current index state."""
        edges  = np.asarray(self._bin_edges_lod, dtype=np.float64)
        n_bins = len(edges) - 1
        unit   = self._lod_unit_label
        mono   = 'font-family:"SF Mono",Menlo,Consolas,monospace;'
        head = (f"<div style='color:#e8e8ec; font-weight:bold; font-size:12px;'>"
                f"Bin {b + 1} <span style='color:#6a6a72; font-weight:normal;'>of {n_bins}</span></div>"
                f"<div style='color:#8a8a92; {mono}'>{edges[b]:.4g} – {edges[b + 1]:.4g}"
                f"{'&nbsp;' + unit if unit else ''}</div>")

        state = self._lod_state
        if state == "ready" and self._lod is not None and self._lod_key is not None:
            indices = self._lod.query(self._lod_key, b)
            if indices is None:
                state, self._lod_msg = "failed", "the index files have gone from the cache."
            else:
                n_in = len(indices)
                if n_in == 0:
                    return head + "<div style='color:#8a8a92; margin-top:4px;'>No particles in this bin</div>"
                sample = ", ".join(str(int(i)) for i in indices[:8])
                more   = (f" <span style='color:#6a6a72;'>… +{n_in - 8:,} more</span>"
                          if n_in > 8 else "")
                return (head +
                        f"<div style='margin-top:4px;'><span style='color:#4ecdc4; font-weight:bold;'>"
                        f"{n_in:,}</span> particle{'s' if n_in != 1 else ''}</div>"
                        f"<div style='color:#a0a0a8; {mono} font-size:10px;'>rows {sample}{more}</div>"
                        f"<div style='color:#5a5a62; font-size:10px;'>row numbers (from 0) in "
                        f"{self._ds_path_lod}</div>")
        if state == "building":
            done, total = self._lod_progress
            step = (f"{done} of {total} snapshot files read" if total > 1 else "reading the snapshot")
            return (head +
                    f"<div style='color:#e0b354; margin-top:4px;'>Listing the particles in each bin…</div>"
                    f"<div style='color:#6a6a72; font-size:10px;'>{step}; this fills in when done</div>")
        if state == "failed":
            return (head +
                    f"<div style='color:#e07b54; margin-top:4px;'>Could not list the particles</div>"
                    f"<div style='color:#8a8a92; font-size:10px;'>{self._html_escape(self._lod_msg)}</div>")
        return (head +
                f"<div style='color:#8a8a92; font-size:10px; margin-top:4px;'>"
                f"No particle list: {self._html_escape(self._lod_msg)}</div>")

    @staticmethod
    def _html_escape(text: str) -> str:
        return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE — inspector (crosshair + tooltip)
    # ═════════════════════════════════════════════════════════════════════════

    def _inspector_mouse_moved(self, scene_pos) -> None:
        """
        Slot connected to pw.scene().sigMouseMoved (QPointF, scene coords).

        Called on every raw mouse-move event — potentially hundreds per second.
        We only store the position and arm the 16 ms timer; the actual data
        lookup runs in _inspector_compute, at most once per frame.
        """
        self._insp_pending = scene_pos
        if not self._insp_timer.isActive():
            self._insp_timer.start()

    def _inspector_compute(self) -> None:
        """
        Throttled inspector logic — fires at most 60 times per second.

        Steps
        ─────
        1. Bail if inspector is inactive (histogram, matrix, no data loaded).
        2. Check that the mouse is inside the ViewBox scene rect; hide if not.
        3. Map scene coords → view (data) coords.
        4. Reverse log-x transform if needed: view_x = log10(data_x) when
           log-x is active, so data_x = 10^view_x.
        5. argmin snap: find the cached curve point with the nearest x.
        6. Build tooltip HTML and position it, flipping left if near right edge.
        """
        scene_pos = self._insp_pending
        if scene_pos is None or not self._insp_active:
            return
        if self._insp_n_total == 0 and self._insp_lod1_raw is None:
            return

        vb = self.pw.getViewBox()

        # ── Step 2: hide if mouse left the plot area ─────────────────────
        if not vb.sceneBoundingRect().contains(scene_pos):
            self._insp_vline.hide()
            self._insp_tip.hide()
            return

        # ── Step 3: scene → view (data) coordinate ───────────────────────
        mp     = vb.mapSceneToView(scene_pos)
        view_x = mp.x()

        # ── Step 4: reverse log transform ────────────────────────────────
        # data_x is the global 1-based particle index after 10^view_x inversion.
        if self._insp_log_x:
            data_x = 10.0 ** view_x
        else:
            data_x = view_x

        # ── Step 5: global index snap ─────────────────────────────────────
        # Clamp to the full population range [1, N_total].
        n_total    = self._insp_n_total if self._insp_n_total > 0 else self._insp_n_lod1
        x_snap     = max(1, min(int(round(data_x)), n_total))
        global_idx = x_snap - 1   # 0-based

        # ── Step 6: value lookup ──────────────────────────────────────────
        # Primary path: direct HDF5 read via read_particle.
        # Fallback: approximate from LOD-1 while prefix-sum builds on the
        # background thread (first hover after file open).
        if (
            self._insp_reading
            or self._insp_vset is None
            or not self._insp_dataset_path
        ):
            if self._insp_lod1_raw is not None and self._insp_n_lod1 > 0:
                stride   = max(1, self._insp_stride)
                lod1_idx = max(0, min(
                    int(round(global_idx / stride)),
                    self._insp_n_lod1 - 1,
                ))
                y_snap = float(self._insp_lod1_raw[lod1_idx])
            else:
                y_snap = float("nan")
        else:
            self._insp_reading = True
            try:
                y_snap = self._insp_vset.read_particle(
                    self._insp_dataset_path, global_idx)
            finally:
                self._insp_reading = False

        # ── Position crosshair ────────────────────────────────────────────
        vline_pos = (
            float(np.log10(x_snap)) if self._insp_log_x and x_snap > 0
            else float(x_snap)
        )
        self._insp_vline.setPos(vline_pos)
        self._insp_vline.show()

        # ── Step 7: build tooltip ─────────────────────────────────────────
        x_disp = f"{x_snap:,}"
        y_disp = f"{y_snap:.4e}"


        dtype_span = (
            f"<span style='color:#6a6a72; font-weight:normal; font-size:10px;'>"
            f"({self._insp_dtype})</span>"
        )
        self._insp_tip.setText(
            f"<div style='font-family:\"SF Mono\",Menlo,Consolas,monospace;'>"
            f"<div style='color:#8a8a92; font-size:11px; font-weight:normal;'>"
            f"x = {x_disp}</div>"
            f"<div style='color:#e8e8ec; font-size:12px; font-weight:bold;'>"
            f"{y_disp}&nbsp;{dtype_span}</div>"
            f"</div>"
        )
        self._insp_tip.adjustSize()

        # Map the snapped data point back to pixel coords in the pw widget
        tip_scene = vb.mapViewToScene(
            pg.Point(vline_pos, mp.y()))
        tip_px    = self.pw.mapFromScene(tip_scene)

        OFFSET  = 14   # horizontal gap between crosshair and tooltip
        tw      = self._insp_tip.width()
        th      = self._insp_tip.height()
        pw_w    = self.pw.width()
        pw_h    = self.pw.height()

        # Default: tooltip to the right of the crosshair
        tx = int(tip_px.x()) + OFFSET

        # Snag 3 (Edge Cutoff): flip to left side if tooltip would overflow
        if tx + tw + 4 > pw_w:
            tx = int(tip_px.x()) - tw - OFFSET

        # Vertical: centre on crosshair, clamped inside the widget
        ty = int(tip_px.y()) - th // 2
        ty = max(4, min(ty, pw_h - th - 4))

        self._insp_tip.move(tx, ty)
        self._insp_tip.show()
        self._insp_tip.raise_()   # always on top of any other pw children

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE — pyqtgraph helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _hard_reset(self):
        """
        Full pyqtgraph state wipe.  pw.clear() alone leaks log-mode, titles,
        labels, and legends into the next render.
        """
        pw = self.pw
        pi = pw.getPlotItem()
        pw.clear()
        pw.setLogMode(x=False, y=False)
        pi.setTitle("")
        pi.setLabel("bottom", "")
        pi.setLabel("left",   "")
        # Guard: scene() returns None when pyqtgraph has already cleaned up
        # the graphics scene (happens on second+ render).  Calling removeItem
        # on None throws AttributeError which silently kills the whole render.
        if pi.legend is not None:
            scene = pi.legend.scene()
            if scene is not None:
                scene.removeItem(pi.legend)
            pi.legend = None
        pw.enableAutoRange()



    # ─────────────────────────────────────────────────────────────────────────
    #  PRIVATE — style application (Task 2.2)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_style_to_item(
        self,
        item,
        color: str,
        vis_mode: str,
        interp_mode: str,
    ) -> None:
        """
        Encapsulate all visual mutation logic for a single PlotDataItem.

        Called immediately after item creation (_draw_curve Task 2.3) and again
        from the live-update slot (_on_style_changed Task 4.1) on every cached item.

        Visibility rules (Task 4.3 / 4.4)
        ─────────────────────────────────
          "Points"      → symbol='x', size=4, symbolPen=color, symbolBrush=transparent
                          pen=None  (no connecting line)
          "Line"        → symbol=None, pen=color
          "Both"        → symbol='x' + pen=color

        Interpolation rules
        ───────────────────
          This version of pyqtgraph requires len(x)==len(y)+1 for any
          stepMode value (old bin-edges API).  We never use pyqtgraph's
          native stepMode.  Instead we pre-compute a left-step staircase
          array via _make_step_xy() and call plain setData() — identical
          to scipy interp1d(kind='nearest') applied to the display path.
          _orig_x / _orig_y on the item hold the original data so that
          toggling Step ↔ Linear restores the correct arrays.
        """
        want_line   = vis_mode in ("Line",   "Both")
        want_points = vis_mode in ("Points", "Both")

        # ── Pen ───────────────────────────────────────────────────────────
        pen = pg.mkPen(color, width=getattr(item, '_plot_width', 1.6)) if want_line else None

        # ── Symbol ────────────────────────────────────────────────────────
        if want_points:
            sym, sym_size = 'x', 4
            sym_pen   = pg.mkPen(color)
            sym_brush = pg.mkBrush(None)
        else:
            sym, sym_size = None, 0
            sym_pen = sym_brush = None

        item.setSymbol(sym)
        item.setSymbolSize(sym_size)
        item.setSymbolPen(sym_pen)
        item.setSymbolBrush(sym_brush)
        item.setPen(pen)

        # ── Step / interpolation mode ─────────────────────────────────────
        # Build staircase x/y in Python and call setData() with no stepMode
        # kwarg.  _orig_x/_orig_y must have been stamped on the item by
        # _draw_curve before this method is called.
        orig_x = getattr(item, '_orig_x', None)
        orig_y = getattr(item, '_orig_y', None)
        hist_edges = getattr(item, '_hist_edges', None)
        if hist_edges is not None and orig_y is not None:
            # Histogram: the outline of the bins, whatever the interpolation.
            # Points marks the bin centres instead.
            if want_line:
                x_plot, y_plot = self._histogram_outline(
                    hist_edges, orig_y, getattr(item, '_hist_log_y', False))
            else:
                x_plot, y_plot = orig_x, orig_y
            try:
                item.setData(x_plot, y_plot)
            except Exception:
                pass
        elif orig_x is not None and orig_y is not None:
            if interp_mode == "Step":
                x_plot, y_plot = self._make_step_xy(orig_x, orig_y)
            else:
                x_plot, y_plot = orig_x, orig_y
            try:
                item.setData(x_plot, y_plot)
            except Exception:
                pass

    @staticmethod
    def _histogram_outline(edges, counts, log_y: bool = False):
        """
        The outline of histogram bars: each bin a flat top from its left edge
        to its right edge, joined by verticals, down to zero at both ends.

        Counts arrive with empty bins as NaN. On a linear count axis an empty
        bin is drawn at zero. A log axis cannot draw zero, so empty bins and
        the two ends go down to a floor at half the smallest non-zero count
        (0.5 for counts): below every real count, so the outline stays one
        line and no empty bin can be read as holding particles. A polyline
        through bin centres drew a lone bin between two empty ones as nothing.

        Input: n+1 edges, n counts → 2n+2 points.
        """
        e = np.asarray(edges, dtype=np.float64)
        c = np.asarray(counts, dtype=np.float64)
        filled = np.isfinite(c) & (c > 0)
        if log_y:
            base = 0.5 * float(c[filled].min()) if filled.any() else np.nan
        else:
            base = 0.0
        c = np.where(filled, c, base)
        x = np.repeat(e, 2)
        y = np.concatenate([[base], np.repeat(c, 2), [base]])
        return x, y

    @staticmethod
    def _make_step_xy(x: np.ndarray, y: np.ndarray):
        """
        Convert equal-length (x, y) arrays to a left-step staircase path.

        Each segment holds y[i] from x[i] until x[i+1] (horizontal), then
        jumps to y[i+1] (vertical).  The result is a plain polyline that any
        pyqtgraph version can render without needing native stepMode support.

        Input  : N points  → Output: 2N-1 points
        Example (N=3):
          x = [0, 1, 2],  y = [0, 10, 20]
          x_s = [0, 1, 1, 2, 2]   y_s = [0, 0, 10, 10, 20]
          draws: (0,0)→(1,0) horiz · (1,0)→(1,10) vert · (1,10)→(2,10) horiz …
        """
        n = len(x)
        if n < 2:
            return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        x_s = np.empty(2 * n - 1, dtype=np.float64)
        y_s = np.empty(2 * n - 1, dtype=np.float64)
        x_s[0::2] = x          # even positions: original x values
        x_s[1::2] = x[1:]      # odd positions:  repeated right-edge x
        y_s[0::2] = y           # even positions: original y values
        y_s[1::2] = y[:-1]      # odd positions:  hold previous y (horizontal)
        return x_s, y_s

    def _draw_curve(self, pw, spec: CurveSpec, legend, log_y: bool = False,
                    hist_edges=None):
        pen   = pg.mkPen(spec.color, width=spec.width,
                         style=Qt.PenStyle.DashLine if spec.dash
                               else Qt.PenStyle.SolidLine)
        brush = pg.mkBrush(self._parse_rgba(spec.fill_color)) if spec.fill else None
        kw    = dict(pen=pen, antialias=True)
        if spec.fill and brush is not None:
            # fillLevel in log-y mode:
            #   log_y=False → fill down to y=0  (linear baseline)
            #   log_y=True  → None lets pyqtgraph fill to the viewport bottom.
            #                 Using any fixed value was wrong: in log space it
            #                 puts the baseline at a specific y level, not the
            #                 axis edge.
            kw["fillLevel"] = None if log_y else 0.0
            kw["brush"]     = brush
        curve = pw.plot(spec.x, spec.y, **kw)

        # ── Task 2.3: stamp role + original data + apply cached style ────
        # _orig_x/_orig_y must be set BEFORE _apply_style_to_item so that
        # the staircase builder inside can read the unmodified source arrays.
        curve._plot_role  = spec.role
        curve._plot_width = spec.width
        curve._plot_color = spec.color
        curve._orig_x     = spec.x   # pristine source — never overwritten
        curve._orig_y     = spec.y
        if hist_edges is not None and len(hist_edges) == len(spec.y) + 1:
            curve._hist_edges = np.asarray(hist_edges, dtype=np.float64)
            curve._hist_log_y = log_y
        self._apply_style_to_item(
            curve, spec.color, self._vis_mode, self._interp_mode
        )

        # ── Downsampling and clipping strategy ───────────────────────────────
        #
        # Log-x mode (Line view)
        # ─────────────────────
        # The plugin pre-downsampled using _logspaced_minmax_downsample whose
        # bucket edges are proportional to the log-x visual width.
        # pyqtgraph's setDownsampling(auto=True, method="peak") uses LINEAR
        # bucket edges in raw-index space — it would undo our log-spaced work
        # and reintroduce the sawtooth.  DISABLED in log mode.
        #
        # setClipToView in log mode: viewRange() returns log10 bounds (e.g.
        # [0, 4.69] for indices 1→48 k).  Raw x values are linear integers —
        # comparing 1000 > 4.69 clips everything → blank plot.  DISABLED.
        #
        # Linear-x mode
        # ─────────────
        # Both setDownsampling and setClipToView are correct and useful.
        log_mode_active = log_y or pw.getViewBox().state.get(
            "logMode", [False, False]
        )[0]

        if not log_mode_active:
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)

        if legend and spec.label:
            legend.addItem(curve, spec.label)

    def _draw_marker(self, pw, spec: MarkerSpec,
                     log_x: bool = False, log_y: bool = False):
        """
        Draw an infinite line marker.

        When log scale is active pyqtgraph works in log10-transformed
        coordinate space internally.  A marker at physical position p on
        a log-x axis must therefore be placed at log10(p), not p — otherwise
        the line lands orders of magnitude off-screen and is never visible.

        Rules
        ─────
          axis=="x" and log_x  →  pos = log10(pos)   (if pos > 0)
          axis=="y" and log_y  →  pos = log10(pos)   (if pos > 0)
          pos ≤ 0              →  skip  (can't place a log marker at zero/negative)
        """
        pos = spec.pos

        if spec.axis == "x" and log_x:
            if pos <= 0:
                return          # un-plottable on a log axis — skip silently
            pos = float(np.log10(pos))
        elif spec.axis == "y" and log_y:
            if pos <= 0:
                return
            pos = float(np.log10(pos))

        line = pg.InfiniteLine(
            pos      = pos,
            angle    = 90 if spec.axis == "x" else 0,
            pen      = pg.mkPen(spec.color, width=1,
                                style=Qt.PenStyle.DashLine if spec.dash
                                      else Qt.PenStyle.SolidLine),
            label    = spec.label,
            labelOpts= {"color": spec.color, "position": 0.92},
        )
        pw.addItem(line)

    def _draw_region(self, pw, spec: RegionSpec, log_x: bool):
        x_min = np.log10(spec.x_min) if log_x and spec.x_min > 0 else spec.x_min
        x_max = np.log10(spec.x_max) if log_x and spec.x_max > 0 else spec.x_max
        r = pg.LinearRegionItem(
            values   =[x_min, x_max],
            brush    = pg.mkBrush(self._parse_rgba(spec.color)),
            pen      = pg.mkPen("#4ecdc4", width=1),
        )
        r.setMovable(False)
        pw.addItem(r)

    def _draw_image(self, pw, spec):
        """
        Render a 2-D ImageSpec as an origin-anchored pg.ImageItem.

        Alpha channel: data minimum → alpha=0 (transparent), maximum → alpha=255.
        This prevents background-void washout when two images overlap.

        Both images are placed at (0,0) with no transform — smaller images
        occupy the bottom-left region of larger ones (origin-anchored).
        """
        from plugin_base import ImageSpec as _IS
        arr_f = spec.data.astype(np.float64)
        vmin  = float(np.nanmin(arr_f))
        vmax  = float(np.nanmax(arr_f))
        span  = vmax - vmin if vmax != vmin else 1.0
        norm  = (arr_f - vmin) / span          # 0 at min, 1 at max

        lut     = self._build_lut(spec.colormap)
        lut_idx = (norm * 255).astype(np.uint8)
        alpha   = lut_idx                      # 0 at min → transparent

        rows, cols = arr_f.shape
        rgba       = np.zeros((cols, rows, 4), dtype=np.uint8)  # col-major
        norm_T     = lut_idx.T
        rgba[..., 0] = lut[norm_T, 0]
        rgba[..., 1] = lut[norm_T, 1]
        rgba[..., 2] = lut[norm_T, 2]
        rgba[..., 3] = alpha.T

        img = pg.ImageItem(rgba)
        img.setZValue(spec.z_value)
        rect = getattr(spec, "rect", None)
        if rect is not None:            # analysis maps: place the grid in axis units
            from PyQt6.QtCore import QRectF
            x0, y0, x1, y1 = (float(v) for v in rect)
            img.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        pw.getPlotItem().addItem(img)

    @staticmethod
    def _build_lut(colormap: str) -> np.ndarray:
        """
        Build a 256×3 uint8 LUT for diagnostic heatmaps.

        "teal"  → black (#000) → #4ecdc4  (file A / reference)
        "coral" → black (#000) → #e07b54  (file B / comparison)

        index 0   = data minimum → black (matches alpha=0 → invisible)
        index 255 = data maximum → full colour
        """
        lut = np.zeros((256, 3), dtype=np.uint8)
        t   = np.linspace(0.0, 1.0, 256)
        _colors = {
            "teal":  np.array([78,  205, 196], dtype=np.float32),
            "coral": np.array([224, 123,  84], dtype=np.float32),
        }
        color = _colors.get(colormap, _colors["teal"])
        for ch in range(3):
            lut[:, ch] = (t * color[ch]).astype(np.uint8)
        return lut

    @staticmethod
    def _parse_rgba(hex_str: str) -> tuple[int, int, int, int]:
        h = hex_str.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return r, g, b, 80
        elif len(h) == 8:
            r, g, b, a = (int(h[i:i+2], 16) for i in (0, 2, 4, 6))
            return r, g, b, a
        return 255, 255, 255, 80