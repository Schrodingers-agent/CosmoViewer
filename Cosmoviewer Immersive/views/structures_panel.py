"""
Halo table: the groups of the snapshot on screen, one row per group.

Rows come from the simulation's own Gadget-4 catalogue (fof_tab / fof_subhalo_tab)
or from friends-of-friends run in this window on the snapshot on screen; the
source selector says which, and never offers groups of another snapshot.
Selecting a row puts that group's complete membership into the active selection
set and centres the 3D view on it (controller/structures_controller.py does the
reading; this panel only displays and asks).

Numbers are the values stored in the files, in the run's code units, and every
column header names its unit (for example 10¹⁰ M☉/h, Mpc/h comoving). A value
that has not been measured yet is shown as "—", never as zero.

The table holds numpy columns and sorts and filters them itself, so a catalogue
of a million groups costs the view only the rows on screen.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                             QSpinBox, QTableView, QVBoxLayout, QWidget)

MISSING = "—"


@dataclass
class Column:
    """One table column: ``key`` names it for the controller, ``header`` (may hold a newline
    and the unit) and ``tip`` are what the reader sees, ``kind`` sets the format."""
    key: str
    header: str
    tip: str
    values: np.ndarray
    kind: str = 'float'          # 'int' (thousands separators), 'float' (5 significant digits), 'pos'


class GroupTableModel(QAbstractTableModel):
    """Columns of numbers, one row per group, sorted and filtered here (not by a proxy)."""

    def __init__(self):
        super().__init__()
        self.columns: List[Column] = []
        self.n_rows = 0
        self.pos_decimals = 4
        self.min_n = 0
        self._sort: Optional[Tuple[int, Qt.SortOrder]] = None
        self._order = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ content

    def set_columns(self, columns: Sequence[Column], n_rows: int, pos_decimals: int = 4):
        self.beginResetModel()
        self.columns, self.n_rows, self.pos_decimals = list(columns), int(n_rows), int(pos_decimals)
        self._sort = None if self._sort is None or self._sort[0] >= len(self.columns) else self._sort
        self._rebuild_order()
        self.endResetModel()

    def set_values(self, key: str, values: np.ndarray):
        """New values for one column (e.g. masses just measured); order and filter are kept."""
        for c, col in enumerate(self.columns):
            if col.key == key:
                col.values = values
                if self._sort is not None and self._sort[0] == c:
                    self.beginResetModel()
                    self._rebuild_order()
                    self.endResetModel()
                elif len(self._order):
                    self.dataChanged.emit(self.index(0, c), self.index(len(self._order) - 1, c))
                return

    def set_min_n(self, n: int):
        self.beginResetModel()
        self.min_n = int(n)
        self._rebuild_order()
        self.endResetModel()

    def column_values(self, key: str) -> Optional[np.ndarray]:
        col = next((c for c in self.columns if c.key == key), None)
        return None if col is None else col.values

    def _rebuild_order(self):
        idx = np.arange(self.n_rows, dtype=np.int64)
        n = self.column_values('n')
        if self.min_n > 0 and n is not None and len(n) == self.n_rows:
            idx = idx[np.asarray(n) >= self.min_n]
        if self._sort is not None and len(idx):
            c, order = self._sort
            v = np.asarray(self.columns[c].values, dtype=np.float64)[idx]
            nan = ~np.isfinite(v)
            # stable sort; values not measured yet go last in both directions
            key = np.where(nan, np.inf, -v if order == Qt.SortOrder.DescendingOrder else v)
            idx = idx[np.argsort(key, kind='stable')]
        self._order = idx

    # ------------------------------------------------------------------ rows and groups

    def group_at(self, row: int) -> Optional[int]:
        g = self.column_values('group')
        if g is None or not (0 <= row < len(self._order)):
            return None
        return int(g[self._order[row]])

    def row_of(self, group: int) -> Optional[int]:
        g = self.column_values('group')
        if g is None or not len(self._order):
            return None
        hit = np.flatnonzero(np.asarray(g)[self._order] == int(group))
        return int(hit[0]) if len(hit) else None

    def exists(self, group: int) -> bool:
        g = self.column_values('group')
        return g is not None and bool(np.any(np.asarray(g) == int(group)))

    def n_shown(self) -> int:
        return len(self._order)

    # ------------------------------------------------------------------ Qt model

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._order)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def _text(self, col: Column, v) -> str:
        v = float(v)
        if not np.isfinite(v):
            return MISSING
        if col.kind == 'int':
            return f"{int(v):,}"
        if col.kind == 'pos':
            return f"{v:.{self.pos_decimals}f}"
        return f"{v:,.5g}"

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        col = self.columns[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self._text(col, col.values[self._order[index.row()]])
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            return col.tip
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal or not (0 <= section < len(self.columns)):
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section].header
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.columns[section].tip
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if not (0 <= column < len(self.columns)):
            return
        self.layoutAboutToBeChanged.emit()
        # The selected row must still be the same group after sorting, not whatever group
        # now sits at its old row number: move the view's persistent indexes with their groups.
        old = self.persistentIndexList()
        items = [int(self._order[i.row()]) if 0 <= i.row() < len(self._order) else -1 for i in old]
        self._sort = (column, order)
        self._rebuild_order()
        where = np.full(self.n_rows, -1, dtype=np.int64)
        where[self._order] = np.arange(len(self._order))
        self.changePersistentIndexList(old, [self.index(int(where[it]), i.column()) if it >= 0 and where[it] >= 0
                                             else QModelIndex() for it, i in zip(items, old)])
        self.layoutChanged.emit()


class StructuresPanel(QWidget):
    """The halo table and its controls; lives in a dock of the main window."""

    source_chosen = pyqtSignal(object)        # the key of the chosen source
    group_chosen = pyqtSignal(int)            # 1-based group (or subhalo) number, chosen by the reader
    analyse_requested = pyqtSignal()
    check_requested = pyqtSignal()
    measure_all_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    STYLE = (
        "#StructuresPanel { background-color: #121212; }"
        "QTableView { background-color: #0a0a0a; alternate-background-color: #141414; color: #dddddd;"
        " gridline-color: #2a2a2a; selection-background-color: #007acc; selection-color: #ffffff; }"
        "QHeaderView { background-color: #1a1a1a; }"
        "QHeaderView::section { background-color: #1a1a1a; color: #aaaaaa; border: 1px solid #333;"
        " padding: 2px 4px; }"
        "QComboBox, QSpinBox { background-color: #1a1a1a; color: #dddddd; border: 1px solid #444; padding: 2px; }")

    def __init__(self):
        super().__init__()
        # Dark like the side panel, also when the dock floats on its own (no window behind it).
        self.setObjectName("StructuresPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(self.STYLE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.addWidget(QLabel("Groups from:"))
        self.combo_source = QComboBox()
        self.combo_source.setMinimumContentsLength(28)
        self.combo_source.setToolTip("The simulation's own catalogue for the snapshot on screen, or FoF run in this\n"
                                     "window on the snapshot on screen. Groups of other snapshots are never listed.")
        self.combo_source.activated.connect(lambda i: self.source_chosen.emit(self.combo_source.itemData(i)))
        top.addWidget(self.combo_source, 1)
        top.addSpacing(8)
        top.addWidget(QLabel("Min N:"))
        self.spin_min_n = QSpinBox()
        self.spin_min_n.setRange(0, 2_000_000_000)
        self.spin_min_n.setToolTip("Show only groups with at least this many member particles")
        self.spin_min_n.valueChanged.connect(self._on_min_n)
        top.addWidget(self.spin_min_n)
        top.addSpacing(8)
        top.addWidget(QLabel("Go to group:"))
        self.spin_jump = QSpinBox()
        self.spin_jump.setRange(1, 2_000_000_000)
        self.spin_jump.setToolTip("Group number as in the catalogue or the FoF list (1 = first)")
        top.addWidget(self.spin_jump)
        self.btn_jump = QPushButton("Go")
        self.btn_jump.clicked.connect(lambda: self.jump_to(self.spin_jump.value()))
        top.addWidget(self.btn_jump)
        lay.addLayout(top)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_note)

        self.model = GroupTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(140)
        self.table.selectionModel().currentRowChanged.connect(lambda cur, _prev: self._row_chosen(cur.row(), False))
        self.table.clicked.connect(lambda idx: self._row_chosen(idx.row(), True))
        lay.addWidget(self.table, 1)
        # A click and the current-row change it causes, or a run of arrow keys, choose one group.
        # A current-row change that leaves the same group current (sorting, filtering) chooses nothing;
        # clicking a row always chooses its group, so a set changed elsewhere can be put back.
        self._pending_row: Optional[int] = None
        self._pending_click = False
        self._last_chosen: Optional[int] = None
        self._quiet = False
        self._choose_timer = QTimer(self)
        self._choose_timer.setSingleShot(True)
        self._choose_timer.setInterval(120)
        self._choose_timer.timeout.connect(self._emit_chosen)

        self.lbl_selection = QLabel("")
        self.lbl_selection.setWordWrap(True)
        self.lbl_selection.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_selection)

        bar = QHBoxLayout()
        self.btn_analyse = QPushButton("Analyse")
        self.btn_analyse.clicked.connect(lambda: self.analyse_requested.emit())
        bar.addWidget(self.btn_analyse)
        self.btn_check = QPushButton("Check against catalogue")
        self.btn_check.setToolTip("Measure the selected group from its member particles at full resolution and\n"
                                  "compare with the catalogue: members found against its length, summed member\n"
                                  "mass against its mass, periodic centre of mass against its position.")
        self.btn_check.clicked.connect(lambda: self.check_requested.emit())
        bar.addWidget(self.btn_check)
        self.btn_measure = QPushButton("Measure all groups")
        self.btn_measure.clicked.connect(lambda: self.measure_all_requested.emit())
        bar.addWidget(self.btn_measure)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(lambda: self.cancel_requested.emit())
        self.btn_cancel.setVisible(False)
        bar.addWidget(self.btn_cancel)
        self.lbl_busy = QLabel("")
        bar.addWidget(self.lbl_busy, 1)
        lay.addLayout(bar)

        self.lbl_check = QLabel("")
        self.lbl_check.setWordWrap(True)
        self.lbl_check.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_check.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_check)
        self.set_actions(None)

    # ------------------------------------------------------------------ filling

    def set_sources(self, items: Sequence[Tuple[str, object]], current_key):
        """Source choices (label, key); ``current_key`` is shown. An empty list empties the selector."""
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        for label, key in items:
            self.combo_source.addItem(label, key)
        i = next((k for k, (_, key) in enumerate(items) if key == current_key), -1)
        if i >= 0:
            self.combo_source.setCurrentIndex(i)
        self.combo_source.setEnabled(bool(items))
        self.combo_source.blockSignals(False)

    def set_table(self, columns: Sequence[Column], n_rows: int, pos_decimals: int = 4):
        self._choose_timer.stop()
        self._pending_row, self._last_chosen = None, None
        self._quiet = True
        try:
            self.model.set_columns(columns, n_rows, pos_decimals)
            self.table.resizeColumnsToContents()
        finally:
            self._quiet = False

    def clear_table(self):
        self.set_table([], 0)

    def set_values(self, key: str, values: np.ndarray):
        group = self.selected_group()
        self._quiet = True
        try:
            self.model.set_values(key, values)
            if group is not None:
                self._select_row(self.model.row_of(group))
        finally:
            self._quiet = False

    def set_note(self, text: str, warn: bool = False):
        self.lbl_note.setText(text)
        self.lbl_note.setStyleSheet("color: #ffb347;" if warn else "color: #aaa;")

    def set_selection_text(self, text: str, warn: bool = False):
        self.lbl_selection.setText(text)
        self.lbl_selection.setStyleSheet("color: #ffb347;" if warn else "color: #00e5ff;")

    def set_check_text(self, html: str):
        self.lbl_check.setText(html)

    def set_busy(self, text: Optional[str]):
        """A background measurement is running (``text`` says what), or none (None)."""
        self.lbl_busy.setText(text or "")
        self.btn_cancel.setVisible(text is not None)
        self.btn_measure.setEnabled(text is None and self.btn_measure.property('available') is True)

    def set_actions(self, kind: Optional[str], analyse_tip: str = "", analyse_on: bool = False,
                    can_check: bool = False):
        """Buttons for a source kind ('catalogue', 'fof' or None)."""
        self.btn_analyse.setEnabled(analyse_on)
        self.btn_analyse.setToolTip(analyse_tip)
        self.btn_check.setVisible(kind == 'catalogue')
        self.btn_check.setEnabled(kind == 'catalogue' and can_check)
        self.btn_measure.setProperty('available', kind is not None)
        self.btn_measure.setEnabled(kind is not None and not self.btn_cancel.isVisible())
        self.btn_measure.setToolTip(
            "Measure every group from its member particles at full resolution and compare with the catalogue"
            if kind == 'catalogue' else
            "Measure the mass and the periodic centre of mass of every group from its members at full resolution")

    # ------------------------------------------------------------------ choosing a group

    def selected_group(self) -> Optional[int]:
        idx = self.table.selectionModel().currentIndex()
        return self.model.group_at(idx.row()) if idx.isValid() else None

    def _select_row(self, row: Optional[int]):
        sm = self.table.selectionModel()
        if row is None:
            sm.clearSelection()
            sm.clearCurrentIndex()
            return
        idx = self.model.index(row, 0)
        sm.setCurrentIndex(idx, sm.SelectionFlag.ClearAndSelect | sm.SelectionFlag.Rows)
        self.table.scrollTo(idx)

    def show_group(self, group: Optional[int]):
        """Mark ``group``'s row as the selected one without choosing it again."""
        self._choose_timer.stop()
        self._pending_row = None
        self._last_chosen = group
        self._quiet = True
        try:
            self._select_row(None if group is None else self.model.row_of(group))
        finally:
            self._quiet = False

    def jump_to(self, group: int) -> bool:
        """Choose group ``group`` as if its row had been clicked; says so when it is not listed."""
        row = self.model.row_of(group)
        if row is None:
            if self.model.exists(group):
                self.set_note(f"Group {group:,} is hidden by the minimum N of {self.model.min_n:,}; "
                              f"lower it to see the group.", warn=True)
            else:
                self.set_note(f"There is no group {group:,} in this list.", warn=True)
            return False
        self._choose_timer.stop()
        self._quiet = True
        try:
            self._select_row(row)
        finally:
            self._quiet = False
        self._pending_row, self._pending_click = row, True
        self._emit_chosen()
        return True

    def _row_chosen(self, row: int, click: bool):
        if self._quiet or row < 0:
            return
        self._pending_row = row
        self._pending_click = self._pending_click or click
        self._choose_timer.start()

    def _emit_chosen(self):
        row, click = self._pending_row, self._pending_click
        self._pending_row, self._pending_click = None, False
        g = None if row is None else self.model.group_at(row)
        if g is None or (not click and g == self._last_chosen):
            return
        self._last_chosen = g
        self.group_chosen.emit(g)

    def _on_min_n(self, n: int):
        group = self.selected_group()
        self._quiet = True
        try:
            self.model.set_min_n(n)
            if group is not None:
                self._select_row(self.model.row_of(group))
        finally:
            self._quiet = False
