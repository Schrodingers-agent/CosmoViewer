"""
density_explorer.py
───────────────────
Thin orchestrator — the only file that knows about both Qt AND HDF5.

Responsibilities
────────────────
  1. File open dialog → scan HDF5 structure → populate tree
  2. Tree click → look up plugin in REGISTRY
  3. Open HDF5 once → read all plugin.REQUIRES paths → close
  4. Call plugin.compute(data, attrs, header, params, tab, sx, sy, window)
  5. Pass PlotRequest → PlotPanel.render()
  6. Re-trigger compute on tab switch, scale change, or nav slider move

Diagnostic override (Options > Compare arrays from files)
──────────────────────────────────────────────────────────
  When the user loads 1 or 2 external files via that menu item:
    - The HDF5 pipeline is bypassed entirely.
    - DiagnosticPlugin.compute() is called instead.
    - The file tree is disabled until the user clicks ✕.

This file contains NO physics, NO unit conversion, NO pyqtgraph draw calls.
All of that lives in plugin_*.py and plot_panel.py respectively.
"""

from __future__ import annotations

import re
import sys
import os
import h5py
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QFrame, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QFileSystemWatcher
from PyQt6.QtGui import QFont, QColor, QAction, QActionGroup
import pyqtgraph as pg

from plugin_base import BasePlugin, PlotRequest, RecomputeReason
from plugin_registry import (build_tree_schema, get_plugin, REGISTRY,
                             register_generic_leaves, resolve_plugin_class)
from plot_panel import PlotPanel
from astro_units import cosmology_context
from io_manager import VirtualSnapshotSet, DataChunk, MAX_HIST_PARTICLES
from compute_worker import ComputeWorker, LodBuildWorker, _MatrixRequest
from lod_index import LodCacheKey, LodIndex
from analysis_plugins import AnalysisPlugin, CENTRE_LABELS, DEFAULT_CENTRE, available_analysis_leaves
from analysis_plugins_m2 import available_analysis_leaves_m2
from data_utils import coerce_attrs as _coerce_attrs
import selection_sets

pg.setConfigOptions(antialias=True, background="#080808", foreground="#c8c8c8")


# ─────────────────────────────────────────────────────────────────────────────
#  STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────

STYLESHEET = """
QMainWindow { background: #0d0d0d; }
QWidget     { background: #0d0d0d; color: #c8c8c8;
              font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
              font-size: 11px; }

/* ── sidebar ──────────────────────────────────────────────── */
QFrame#sidebar  { background: #111111; border-right: 1px solid #232323; }

QTreeWidget {
    background: #111111; border: none; outline: none;
    color: #c8c8c8; font-size: 11px;
}
QTreeWidget::item          { padding: 3px 6px; border-radius: 2px; }
QTreeWidget::item:hover    { background: #1c1c1c; }
QTreeWidget::item:selected { background: #163859; color: #ffffff; }
QTreeWidget::branch        { background: #111111; }

/* ── unified toolbar ──────────────────────────────────────── */
QFrame#view_toolbar { background: #0f0f0f; border-bottom: 1px solid #222; }
QWidget#tab_container { background: transparent; }
QWidget#toolbar_group { background: transparent; }

QPushButton#tab_btn {
    background: transparent; border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 18px; color: #555;
    border-radius: 0px; font-size: 12px;
}
QPushButton#tab_btn:hover { color: #ccc; }
QPushButton#tab_btn[active="true"] {
    color: #4ecdc4; border-bottom: 2px solid #4ecdc4;
}

QPushButton#nav_btn {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 3px; color: #888; font-size: 12px;
}
QPushButton#nav_btn:hover   { background: #222; border-color: #4ecdc4; color: #fff; }
QPushButton#nav_btn:pressed { background: #4ecdc4; color: #000; }

QPushButton#tool_btn {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    padding: 3px 10px; border-radius: 3px; color: #666; font-size: 11px;
}
QPushButton#tool_btn:hover { color: #ccc; border-color: #555; }

QPushButton#tool_btn_toggle {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    padding: 3px 10px; border-radius: 3px; color: #666; font-size: 11px;
}
QPushButton#tool_btn_toggle:hover   { color: #ccc; border-color: #555; }
QPushButton#tool_btn_toggle:checked {
    background: #0d2236; border-color: #4ecdc4; color: #4ecdc4;
}

QComboBox#scale_combo {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    padding: 3px 8px; border-radius: 3px; color: #c8c8c8; min-width: 76px;
}
QComboBox#scale_combo::drop-down  { border: none; width: 14px; }
QComboBox#scale_combo QAbstractItemView {
    background: #1a1a1a; border: 1px solid #333; color: #c8c8c8;
    selection-background-color: #163859;
}

QPushButton#notation_btn {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    padding: 3px 10px; border-radius: 3px; color: #666; font-size: 11px;
    min-width: 68px;
}
QPushButton#notation_btn:hover { color: #ccc; }
QPushButton#notation_btn[position="left"]  { border-radius: 3px 0px 0px 3px; }
QPushButton#notation_btn[position="mid"]   { border-radius: 0px; border-left: none; }
QPushButton#notation_btn[position="right"] { border-radius: 0px 3px 3px 0px; border-left: none; }
QPushButton#notation_btn:checked {
    background: #0d2236; border-color: #4ecdc4; color: #4ecdc4;
}

QSpinBox#cell_width_spin {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    padding: 2px 4px; border-radius: 3px; color: #c8c8c8;
}
QSpinBox#cell_width_spin::up-button, QSpinBox#cell_width_spin::down-button {
    background: #222; border: none; width: 14px;
}

QPushButton#export_btn {
    background: #1e1008; border: 1px solid #e07b54;
    padding: 4px 14px; border-radius: 3px;
    color: #e07b54; font-size: 11px; font-weight: bold;
}
QPushButton#export_btn:hover   { background: #e07b54; color: #0d0d0d; }
QPushButton#export_btn:pressed { background: #b85c34; color: #fff; }

QPushButton#load_btn_idle {
    background: transparent; border: 1px solid #e07b54;
    padding: 4px 10px; border-radius: 3px;
    color: #e07b54; font-size: 11px;
}
QPushButton#load_btn_idle:hover   { background: #1e1008; }
QPushButton#load_btn_idle:pressed { background: #e07b54; color: #0d0d0d; }

QPushButton#load_btn_active {
    background: #3d1f10; border: 1px solid #c06040;
    padding: 4px 10px; border-radius: 3px;
    color: #f0a080; font-size: 11px;
}
QPushButton#load_btn_active:hover { background: #4d2510; }

QWidget#load_container { background: transparent; }

QPushButton#help_btn {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 13px; color: #444; font-size: 12px; font-weight: bold;
}
QPushButton#help_btn:hover { color: #888; border-color: #444; }

QSlider#nav_slider::groove:horizontal {
    height: 4px; background: #252525; border-radius: 2px;
}
QSlider#nav_slider::handle:horizontal {
    background: #4ecdc4; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px;
}
QSlider#nav_slider::sub-page:horizontal { background: #4ecdc4; border-radius: 2px; }

QPushButton#open_btn {
    background: #1a1a1a; border: 1px solid #2e2e2e;
    padding: 7px 14px; border-radius: 0px; color: #fff;
}
QPushButton#open_btn:hover   { border-color: #007acc; }
QPushButton#open_btn:pressed { background: #007acc; }

QLabel#filename  { color: #007acc; font-weight: bold; font-size: 11px;
                   padding: 8px 8px 4px 10px; }
QLabel#info_bar  { color: #4ecdc4; font-size: 10px; padding: 4px 14px;
                   border-top: 1px solid #1a1a1a; }
QLabel#status_bar{ color: #444; font-size: 10px; padding: 4px 14px; }

QFrame#diag_banner { background: #1a0a04; border-bottom: 1px solid #6b3010; }
QFrame#cosmo_header_bar {
    background: #0a0a12;
    border-bottom: 1px solid #14142a;
}
QLabel#diag_tag_lbl {
    color: #0d0d0d; background: #e07b54;
    border-radius: 3px; padding: 1px 7px;
    font-size: 9px; font-weight: bold;
}
QLabel#diag_file_lbl  { color: #f0a080; font-size: 10px; font-weight: bold; padding: 4px 0px; }
QLabel#diag_shape_lbl { color: #664433; font-size: 10px; padding: 4px 8px; }

QFrame#sidebar[disabled="true"] { opacity: 0.6; }
QFrame#matrix_view { background: #0d0d0d; }

QLabel#matrix_notice {
    background: #1a0f08; color: #e07b54;
    font-size: 10px; padding: 5px 14px;
    border-bottom: 1px solid #2e1a0a;
}

QTableWidget#matrix_table {
    background: #0f0f0f; alternate-background-color: #111111;
    border: none; gridline-color: #1e1e1e; color: #c8c8c8;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 11px;
    selection-background-color: #0d2236; selection-color: #4ecdc4;
}
QTableWidget#matrix_table QHeaderView { background: #111111; }
QTableWidget#matrix_table QHeaderView::section {
    background: #151515; color: #4ecdc4; font-weight: bold; font-size: 11px;
    padding: 4px 8px; border: none;
    border-right: 1px solid #242424; border-bottom: 1px solid #242424;
}
QTableWidget#matrix_table QHeaderView::section:vertical {
    color: #444; font-weight: normal; text-align: right;
    padding: 2px 10px; min-width: 52px;
}
QScrollBar:vertical { background: #111; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical {
    background: #2e2e2e; border-radius: 4px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #4ecdc4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

# ── Group colours for the file tree ──────────────────────────────────────────
GROUP_COLORS = {
    "Config":     "#555555",
    "Header":     "#555555",
    "Parameters": "#555555",
    "PartType0":  "#00b4d8",
    "PartType1":  "#e63946",
    "PartType2":  "#e0dcc8",
    "PartType3":  "#f2cc8f",
    "PartType4":  "#a8dadc",
    "PartType5":  "#444444",
    "Analyses":   "#ffb703",
    "Sets":       "#7bd389",
}
LEAF_ACTIVE = "#00e5ff"


# ─────────────────────────────────────────────────────────────────────────────
#  HDF5 & ATTR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _scan_dataset_meta(group: h5py.Group, prefix: str = "") -> dict:
    """{dataset path: (shape, dtype, attrs)} below group, without reading data."""
    out = {}
    for name, item in group.items():
        path = f"{prefix}{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            out[path] = (item.shape, item.dtype, _coerce_attrs(item.attrs))
        elif isinstance(item, h5py.Group):
            out.update(_scan_dataset_meta(item, prefix=f"{path}/"))
    return out


def _scan_hdf5_group(group: h5py.Group, prefix: str = "") -> list[str]:
    """Recursively scan an HDF5 group and return a list of dataset paths relative to the group."""
    leaves = []
    for name, item in group.items():
        path = f"{prefix}{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            leaves.append(path)
        elif isinstance(item, h5py.Group):
            leaves.extend(_scan_hdf5_group(item, prefix=f"{path}/"))
    return leaves





# ─────────────────────────────────────────────────────────────────────────────
#  ASYNC PREFIX-SUM WORKER
# ─────────────────────────────────────────────────────────────────────────────

class _PrefixWorker(QThread):
    """
    Build VirtualSnapshotSet._per_file_prefix[grp_name] on a background thread.

    Opens each chunk file once to read NumPart_ThisFile — no particle data.
    On NVMe: ~10 µs/file.  On NFS: ~5 ms/file.

    The dict write is GIL-atomic in CPython; read_particle() polls the dict
    on every call and switches from the LOD-1 fallback to exact HDF5 reads
    as soon as the key appears.  No signal or lock required.
    """
    def __init__(self, vset, grp_name: str, parent=None):
        super().__init__(parent)
        self._vset     = vset
        self._grp_name = grp_name

    def run(self) -> None:
        try:
            prefix = self._vset._build_per_file_prefix(self._grp_name)
            if self._vset._per_file_prefix is None:
                self._vset._per_file_prefix = {}
            self._vset._per_file_prefix[self._grp_name] = prefix
        except Exception:
            pass   # Swallow; inspector falls back to LOD-1 path permanently


# ─────────────────────────────────────────────────────────────────────────────
#  FILE TREE PANEL
# ─────────────────────────────────────────────────────────────────────────────

class FileTreePanel(QFrame):
    leaf_clicked = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        # Scale sidebar width with screen: 16 % of available width,
        # clamped to [180, 260] px so it is usable on both small laptop
        # screens (1280 px wide) and large monitors (2560+ px wide).
        _screen_w = QApplication.primaryScreen().availableGeometry().width()
        self.setFixedWidth(max(180, min(260, int(_screen_w * 0.16))))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.btn_open = QPushButton("Open Snapshot…")
        self.btn_open.setObjectName("open_btn")
        self.btn_open.setFixedHeight(36)
        lay.addWidget(self.btn_open)

        self.lbl_filename = QLabel("No file loaded")
        self.lbl_filename.setObjectName("filename")
        self.lbl_filename.setWordWrap(True)
        lay.addWidget(self.lbl_filename)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#232323; max-height:1px;")
        lay.addWidget(sep)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setAnimated(True)
        self.tree.itemClicked.connect(self._on_click)
        lay.addWidget(self.tree, stretch=1)

        self._schema = build_tree_schema()
        # Options > "Show fields this file does not have": off lists only what the file can supply.
        self.show_absent = False
        self._populate(file_exists=False)

    ALL_PARTICLES = "All particles"

    def _populate(self, file_exists: bool, hdf5_structure: dict = None,
                  generic_leaves: dict = None, set_leaves: list = None, active_set: str = None):
        self.tree.clear()
        self._last_populate = (file_exists, hdf5_structure, generic_leaves)
        hdf5_structure = hdf5_structure or {}
        generic_leaves = generic_leaves or {}

        # Hand-written leaves first, then generic leaves for every other numeric dataset in the
        # file. Leaves the file cannot supply, and groups left without leaves, are not listed,
        # unless show_absent is on: then they are listed greyed out and cannot be chosen.
        schema = [(g, list(l) + [x for x in generic_leaves.get(g, []) if x not in l])
                  for g, l in self._schema]
        known = {g for g, _ in schema}
        schema += [(g, l) for g, l in generic_leaves.items() if g not in known]

        for group, leaves in schema:
            def present(leaf):
                is_gen = leaf in generic_leaves.get(group, [])
                return file_exists and (is_gen or leaf in hdf5_structure.get(group, []))
            if not self.show_absent:
                leaves = [leaf for leaf in leaves if present(leaf)]
                if not leaves:
                    continue
            g_item = QTreeWidgetItem([group])
            g_item.setForeground(0, QColor(GROUP_COLORS.get(group, "#555")))
            f = QFont()
            f.setBold(True)
            g_item.setFont(0, f)
            g_item.setData(0, Qt.ItemDataRole.UserRole, None)

            for leaf in leaves:
                is_generic = leaf in generic_leaves.get(group, [])
                in_file = present(leaf)
                l_item  = QTreeWidgetItem([leaf])
                if is_generic:
                    l_item.setToolTip(0, "Generic view: units come from the file's own "
                                         "attributes; no quantity-specific plugin.")
                elif group == "Analyses":
                    l_item.setToolTip(0, leaf.replace("Lagr. radii", "Lagrangian radii")
                                      + ": computed from every particle (measurements/). "
                                        "Centre: Options menu.")
                l_item.setData(0, Qt.ItemDataRole.UserRole, (group, leaf))
                if in_file:
                    l_item.setForeground(0, QColor(LEAF_ACTIVE))
                else:
                    l_item.setForeground(0, QColor("#363636"))
                    l_item.setFlags(l_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                g_item.addChild(l_item)

            self.tree.addTopLevelItem(g_item)
            if leaves:
                g_item.setExpanded(True)
        if set_leaves:
            self._add_sets_group(set_leaves, active_set)

    def set_show_absent(self, on: bool):
        """List the fields and analyses this file does not have (greyed), or leave them out.

        The tree is rebuilt from the last file's structure; the Sets group and the current
        choice are kept as they are."""
        self.show_absent = bool(on)
        current = self.tree.currentItem()
        current = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        sets = None
        for i in range(self.tree.topLevelItemCount()):
            if self.tree.topLevelItem(i).text(0) == "Sets":
                sets = self.tree.takeTopLevelItem(i)
                break
        self._populate(*self._last_populate)
        if sets is not None:
            self.tree.addTopLevelItem(sets)
            sets.setExpanded(True)
        if current:
            for i in range(self.tree.topLevelItemCount()):
                g_item = self.tree.topLevelItem(i)
                for j in range(g_item.childCount()):
                    if g_item.child(j).data(0, Qt.ItemDataRole.UserRole) == current:
                        self.tree.setCurrentItem(g_item.child(j))   # no leaf_clicked: nothing recomputes
                        return

    def _add_sets_group(self, set_leaves: list, active: str = None):
        """Selection sets of this run (selection_sets.py); the active one restricts Analyses."""
        g_item = QTreeWidgetItem(["Sets"])
        g_item.setForeground(0, QColor(GROUP_COLORS["Sets"]))
        f = QFont()
        f.setBold(True)
        g_item.setFont(0, f)
        g_item.setToolTip(0, "Particle sets saved by Immersive Mode (Send to Explorer) or a script. "
                             "Choosing one restricts the Analyses to exactly those particles.")
        # (leaf, tooltip) or (leaf, tooltip, shown text): the leaf is the set's key, the shown
        # text adds what the set is (its kind) and a warning when it is known to be incomplete.
        for entry in [(self.ALL_PARTICLES, "Analyses use every particle")] + list(set_leaves):
            leaf, tip = entry[0], entry[1]
            shown = entry[2] if len(entry) > 2 else leaf
            it = QTreeWidgetItem([("● " if leaf == (active or self.ALL_PARTICLES) else "   ") + shown])
            it.setToolTip(0, tip)
            it.setForeground(0, QColor("#ffb703" if "⚠" in shown else LEAF_ACTIVE))
            it.setData(0, Qt.ItemDataRole.UserRole, ("Sets", leaf))
            g_item.addChild(it)
        self.tree.addTopLevelItem(g_item)
        g_item.setExpanded(True)

    def set_sets(self, set_leaves: list, active: str = None):
        """Replace the Sets group only, keeping the rest of the tree."""
        for i in range(self.tree.topLevelItemCount()):
            if self.tree.topLevelItem(i).text(0) == "Sets":
                self.tree.takeTopLevelItem(i)
                break
        if set_leaves:
            self._add_sets_group(set_leaves, active)

    def select_leaf(self, group: str, leaf: str) -> bool:
        """Select and open a tree leaf as if clicked. False if absent or disabled."""
        for i in range(self.tree.topLevelItemCount()):
            g_item = self.tree.topLevelItem(i)
            for j in range(g_item.childCount()):
                item = g_item.child(j)
                if (item.data(0, Qt.ItemDataRole.UserRole) == (group, leaf)
                        and item.flags() & Qt.ItemFlag.ItemIsEnabled):
                    self.tree.setCurrentItem(item)
                    self.leaf_clicked.emit(group, leaf)
                    return True
        return False

    def available_leaves(self) -> list:
        """'group/leaf' for every enabled leaf, for messages."""
        out = []
        for i in range(self.tree.topLevelItemCount()):
            g_item = self.tree.topLevelItem(i)
            for j in range(g_item.childCount()):
                item = g_item.child(j)
                if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                    out.append("/".join(item.data(0, Qt.ItemDataRole.UserRole)))
        return out

    def load_file(self, path: str, hdf5_structure: dict):
        self.lbl_filename.setText(os.path.basename(path))
        self._populate(file_exists=True, hdf5_structure=hdf5_structure)

    def _on_click(self, item: QTreeWidgetItem, _col: int):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data:
            self.leaf_clicked.emit(*data)


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class CosmoExplorer(QMainWindow):

    # For a host window (Immersive Mode) that shows the Explorer as its Analyse view:
    show_in_3d_requested = pyqtSignal()        # "show the active population in the 3D view"
    population_proposed = pyqtSignal(object)   # a selection_sets.SelectionSet the user asked to select
    _hosted = False                            # set_hosted(): the host decides snapshot and population

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CosmoViewer — Dataset Explorer")

        # ── Resolution-aware window geometry ──────────────────────────────
        # Use the primary screen's *available* geometry (excludes taskbar /
        # dock / menu bar) so the window never spawns wider or taller than
        # the usable desktop area.
        #
        # Target: 92 % of available width, 88 % of available height,
        # capped at the original design size of 1 300 × 780 px.
        # On a 1 280 × 800 laptop: 1 177 × 704 px  (fits without scrolling)
        # On a 1 920 × 1 080 HD:   1 300 × 780 px  (original design size)
        # On a 2 560 × 1 440 QHD:  1 300 × 780 px  (plenty of room)
        #
        # setMaximumWidth prevents the user from dragging the window wider
        # than the screen — the pathological case that caused the original
        # overflow bug on small displays.
        _screen   = QApplication.primaryScreen().availableGeometry()
        _init_w   = min(1300, int(_screen.width()  * 0.92))
        _init_h   = min(780,  int(_screen.height() * 0.88))
        self.resize(_init_w, _init_h)
        self.setMaximumWidth(_screen.width())
        # Centre on screen so the window never starts half off-screen
        self.move(
            _screen.left() + (_screen.width()  - _init_w) // 2,
            _screen.top()  + (_screen.height() - _init_h) // 2,
        )

        self.setStyleSheet(STYLESHEET)

        # ── I/O state ─────────────────────────────────────────────────────
        # _vset replaces the old _hdf5_path string.  All file I/O routes
        # through VirtualSnapshotSet which handles N-chunk distributed
        # snapshots transparently alongside plain single-file Gadget-4 files.
        self._vset:           VirtualSnapshotSet | None = None
        self._hdf5_structure: dict                      = {}
        self._plugin:         BasePlugin | None         = None
        self._plugin_cls                                = None
        self._n_particles:    int                       = 0
        # Keep _PrefixWorker instances alive untill run() completes.
        self._prefix_workers: set                       = set()
        # Every started ComputeWorker, held until its finished signal has been delivered
        # (and the level-of-detail build it started, until that ends). A replaced worker is
        # otherwise held only by a reference cycle through its own signal connections, so
        # the garbage collector could free it after run() returns while its queued signals
        # still wait, and delivering them then crashed the process (SIGSEGV in
        # PyQtSlot::call).
        self._live_workers:   set                       = set()
        # The build of the click-a-bin index for the histogram on screen, or None
        # (_ensure_bin_index). Also held in _live_workers until it has finished.
        self._lod_worker:     LodBuildWorker | None     = None

        # Cached from the most recently opened file (in RAM — no re-read).
        self._cached_header: dict = {}   # Header group attrs (coerced)
        self._cached_params: dict = {}   # Parameters group attrs (coerced)

        # Phase 2 — persists the user's unit choice per plugin class so that
        # switching datasets and coming back does not reset the dropdown.
        # Key: plugin class (type), Value: selected unit menu label (str).
        self._unit_labels: dict[type, str] = {}

        # Phase 3 — custom histogram bin count from context-menu override.
        # Cleared whenever the user selects a different dataset from the tree.
        self._custom_bin_count: int | None = None

        # Per-(group, dataset) plugin instance cache.
        # Key   : (group: str, dataset: str)
        # Value : live plugin instance carrying accumulated hysteresis state.
        # Invalidated exactly once: when a new file is opened (_open_file).
        # Never cleared on scale/tab/unit changes — those are re-renders of
        # the same dataset and must reuse the cached instance.
        self._plugin_cache: dict[tuple[str, str], BasePlugin] = {}

        # Active background compute worker.
        # Only one worker runs at a time.  On new selection the current worker
        # is cancelled (cooperative flag) before the next is created.
        # The type annotation uses object to avoid a forward-ref issue; the
        # runtime type is always ComputeWorker | None.
        self._active_worker: object | None = None

        # ── Diagnostic override state ──────────────────────────────────────
        # Active when the user loads 1 or 2 external files (Options > Compare arrays from files).
        # While active, _diag_override=True blocks HDF5 tree clicks and routes
        # tab/scale signals through _compute_and_render_diagnostic() instead.
        self._diag_override: bool = False
        self._diag_files:    list =[]    # list of (ndarray, filename) — 1 or 2
        self._diag_z_swap:   bool = False # False = file A (teal) on top

        # ── Global context state (drives menu bar toggles) ────────────────
        # Stored here on the orchestrator; passed as context= to every
        # plugin.compute() call.  blockSignals() guards against re-entrance.
        self._global_context: dict = {
            "unit_system":  "CGS",      # "CGS" | "Galactic" | "Cosmological"
            "coordinates":  "Physical", # "Physical" | "Comoving"
            "centre":       DEFAULT_CENTRE,  # Analyses only; see analysis_plugins.CENTRE_LABELS
            "selection_set": None,           # Analyses only: {"name", "path", "ids"} or None
        }
        # Selection sets of the open run (selection_sets.py): display leaf -> file
        self._set_files: dict = {}
        self._set_watcher = QFileSystemWatcher(self)
        self._set_watcher.directoryChanged.connect(lambda _d: self._refresh_sets(announce=True))

        self._build_ui()
        self._wire_signals()
        self._create_menu()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        lay  = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.setCentralWidget(root)

        self.sidebar    = FileTreePanel()
        self.plot_panel = PlotPanel()

        lay.addWidget(self.sidebar)
        lay.addWidget(self.plot_panel, stretch=1)

    def _wire_signals(self):
        self.sidebar.btn_open.clicked.connect(self._open_file)
        self.sidebar.leaf_clicked.connect(self._on_leaf_clicked)
        self.plot_panel.tab_changed.connect(self._on_tab_changed)
        self.plot_panel.scale_changed.connect(self._on_scale_changed)
        self.plot_panel.diagnostic_loaded.connect(self._on_diagnostic_loaded)
        self.plot_panel.diagnostic_cleared.connect(self._on_diagnostic_cleared)
        self.plot_panel.swap_z_order.connect(self._on_swap_z_order)
        # Phase 2: unit dropdown selection → re-compute with new scalar
        self.plot_panel.unit_changed.connect(self._on_unit_changed)
        # Phase 3: context-menu bin-count override
        self.plot_panel.custom_bins_requested.connect(self._on_custom_bins_requested)

    # ── Native OS menu bar ─────────────────────────────────────────────────

    def _create_menu(self) -> None:
        """
        Build the native macOS / OS menu bar Options menu.

        Two QActionGroups with setExclusive(True) give radio-button behaviour:
          • Unit System  : CGS (default) | Galactic | Cosmological
          • Coordinates  : Physical (default) | Comoving

        Risk mitigation (infinite recursion)
        ─────────────────────────────────────
        Signal handlers call _compute_and_render() which may programmatically
        update UI state.  blockSignals(True) on each action during that update
        prevents re-entrant triggering.  The guards live in each handler below.
        """
        bar      = self.menuBar()
        opt_menu = bar.addMenu("Options")

        # ── Unit System group ─────────────────────────────────────────────
        unit_group = QActionGroup(self)
        unit_group.setExclusive(True)

        for label, is_default in [("CGS", True), ("Galactic", False),
                                   ("Cosmological", False)]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(is_default)
            unit_group.addAction(act)
            opt_menu.addAction(act)

        opt_menu.addSeparator()

        # ── Coordinates group ─────────────────────────────────────────────
        coord_group = QActionGroup(self)
        coord_group.setExclusive(True)

        for label, is_default in [("Physical", True), ("Comoving", False)]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(is_default)
            coord_group.addAction(act)
            opt_menu.addAction(act)

        opt_menu.addSeparator()

        # ── Centre group (used by the Analyses leaves) ────────────────────
        centre_group = QActionGroup(self)
        centre_group.setExclusive(True)
        for method, label in CENTRE_LABELS.items():
            act = QAction(f"Centre: {label}", self)
            act.setCheckable(True)
            act.setChecked(method == DEFAULT_CENTRE)
            act.setData(method)
            centre_group.addAction(act)
            opt_menu.addAction(act)
        centre_group.triggered.connect(self._on_centre_changed)
        self._centre_action_group = centre_group

        opt_menu.addSeparator()
        opt_menu.setToolTipsVisible(True)

        # ── Field tree: only what this file can supply, unless asked ──────
        act_absent = QAction("Show fields this file does not have", self)
        act_absent.setCheckable(True)
        act_absent.setChecked(False)
        act_absent.setToolTip("List, greyed out, the fields and analyses this snapshot cannot supply")
        act_absent.toggled.connect(self.sidebar.set_show_absent)
        opt_menu.addAction(act_absent)
        self._show_absent_action = act_absent

        # ── Arrays from files instead of the snapshot (diagnostic override) ──
        act_compare = QAction("Compare arrays from files (advanced)…", self)
        act_compare.setToolTip("Show one or two .npy, .csv or .txt arrays instead of the snapshot;\n"
                               "✕ in the toolbar returns to the snapshot")
        act_compare.triggered.connect(lambda _=False: self.plot_panel._do_load_diagnostic())
        # A developer diagnostic, kept out of the everyday menu:
        # listed only when COSMOVIEWER_ADVANCED=1. The action and its code stay.
        if os.environ.get("COSMOVIEWER_ADVANCED", "") == "1":
            opt_menu.addAction(act_compare)
        self._compare_arrays_action = act_compare

        unit_group.triggered.connect(self._on_unit_system_changed)
        coord_group.triggered.connect(self._on_coordinates_changed)

        self._unit_action_group  = unit_group
        self._coord_action_group = coord_group

    def _on_unit_system_changed(self, action: QAction) -> None:
        """
        User toggled a Unit System radio item (CGS / Galactic / Cosmological).

        Clears the persisted unit label for the current plugin so that
        _resolve_astropy_units() auto-selects the best unit for the new system
        on the next render cycle.  The user can then refine via the dropdown.
        """
        system = action.text()
        self._global_context["unit_system"] = system

        # Clear persisted label so Galactic/Cosmological auto-selection fires
        if self._plugin_cls is not None:
            self._unit_labels.pop(self._plugin_cls, None)

        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if not self._diag_override:
            # Coordinate mode feeds Stage 2 cosmo_factor (a^a_exp) — cached
            # CGS data was computed with the OLD coordinate mode; must re-read.
            self._spawn_worker(RecomputeReason.DATA_CHANGE)

    def _on_centre_changed(self, action: QAction) -> None:
        """User picked a centre method. Only analyses use it, so only they recompute."""
        self._global_context["centre"] = action.data()
        if isinstance(self._plugin, AnalysisPlugin) and not self._diag_override:
            self._spawn_worker(RecomputeReason.DATA_CHANGE)

    def _on_coordinates_changed(self, action: QAction) -> None:
        """
        User toggled a Coordinates radio item (Physical / Comoving).

        Comoving mode bypasses the a^a_exp factor in Stage 2, shifting the
        density distribution to comoving values (lower for a < 1).
        """
        self._global_context["coordinates"] = action.text()
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if not self._diag_override:
            # Unit system change clears _unit_labels and requires fresh Stage 2
            # CGS calibration context — must re-read.
            self._spawn_worker(RecomputeReason.DATA_CHANGE)

    # ── File open ──────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gadget-4 Snapshot", os.getcwd(),
            "HDF5 Files (*.hdf5 *.h5);;All Files (*)")
        if not path:
            return
        self.open_path(path)

    def open_path(self, path: str) -> bool:
        """Open a snapshot (any piece of a multi-file one). False on failure."""
        path = str(path)
        self.plot_panel.set_status("Scanning file structure…")

        # ── Build VirtualSnapshotSet ────────────────────────────────────────────
        # Automatically detects NumFilesPerSnapshot, discovers and validates
        # all chunk files, and reconstructs 64-bit particle counts.
        try:
            vset = VirtualSnapshotSet(path)
        except FileNotFoundError as exc:
            self.plot_panel.set_status(f"Missing chunk: {exc}")
            print(f"Cannot open {path}: missing chunk: {exc}")
            return False
        except Exception as exc:
            self.plot_panel.set_status(f"Failed to open: {exc}")
            print(f"Cannot open {path}: {type(exc).__name__}: {exc}")
            return False

        # ── Scan HDF5 structure for the sidebar tree (chunk 0 only) ──────────
        try:
            structure: dict = {}
            meta: dict = {}
            with h5py.File(vset.files[0], "r") as f:
                for g in f:
                    if isinstance(f[g], h5py.Group):
                        structure[g] = _scan_hdf5_group(f[g])
                        meta[g] = _scan_dataset_meta(f[g])
                    else:
                        structure[g] = []
            generic_leaves = register_generic_leaves(meta)
            # Analyses are enabled when the snapshot can supply them
            structure["Analyses"] = (available_analysis_leaves(vset)
                                     + available_analysis_leaves_m2(vset))
        except Exception as exc:
            self.plot_panel.set_status(f"Failed to scan structure: {exc}")
            print(f"Cannot read the structure of {path}: {type(exc).__name__}: {exc}")
            return False

        self._vset           = vset
        self._hdf5_structure = structure
        self._cached_header  = vset.global_header   # already coerced, in RAM
        self._cached_params  = vset.global_params
        self._plugin         = None
        self._plugin_cls     = None

        # Invalidate the plugin cache: new file → new particle counts → stale
        # hysteresis state.  This is the ONLY correct invalidation point.
        self._plugin_cache.clear()
        self._cancel_bin_index()      # an index of the previous file is no use now

        # Filename label: include chunk count for distributed snapshots
        n_chunks  = len(vset.files)
        base_name = os.path.basename(path)
        label     = base_name if n_chunks == 1 else f"{base_name}  ({n_chunks} chunks)"
        self.sidebar.lbl_filename.setText(label)
        self.sidebar._populate(file_exists=True, hdf5_structure=structure,
                               generic_leaves=generic_leaves)

        self.plot_panel.set_status(
            f"Loaded: {label} — select a dataset from the tree.")
        self.plot_panel.reset_view()

        # Populate the cosmology context bar from the cached header (no disk I/O)
        self._update_cosmo_header(path)

        # Selection sets of this run, and tell Immersive Mode which run is open here
        self._global_context["selection_set"] = None
        self._set_files = {}
        self._refresh_sets(announce=False)
        if not self._hosted:            # a hosted Explorer is not a separate window to send sets to
            selection_sets.register_instance(os.getpid(), vset.files)
        return True

    # ── Selection sets (shared with Immersive Mode) ──────────────────────

    def _refresh_sets(self, announce: bool = False) -> None:
        """Re-list the run's set files; a new one (sent from Immersive Mode) becomes active."""
        if self._vset is None:
            return
        snap = self._vset.files[0]
        dirs = selection_sets.candidate_dirs(snap)
        watch = [str(d) for d in dirs if d.is_dir()] + [os.path.dirname(os.path.abspath(snap))]
        old_dirs = self._set_watcher.directories()
        if old_dirs:
            self._set_watcher.removePaths(old_dirs)
        self._set_watcher.addPaths(watch)
        before = dict(getattr(self, "_set_mtimes", {}))
        files = {}
        for p in selection_sets.list_sets(snap):
            try:
                st = selection_sets.read_set(p)
            except Exception as exc:
                print(f"Cannot read set {p}: {exc}")
                continue
            leaf = f"{st.name} ({st.n:,})"
            while leaf in files:
                leaf += "'"
            files[leaf] = p
        self._set_files = files
        self._set_mtimes = {p: os.path.getmtime(p) for p in files.values()}
        active = self._global_context.get("selection_set")
        if self._hosted:    # Analyse view of Immersive Mode: the host decides the population, so a
            # new file is only listed, and the host's population (kept outside the run) stays listed
            if active and active["path"] not in files.values():
                self._set_files[f"{active['name']} ({len(active['ids']):,})"] = active["path"]
            self.sidebar.set_sets(self._set_leaves(), self._active_leaf())
            return
        # new files, or a set re-sent under the same name (rewritten file)
        new = [p for p in files.values() if before.get(p) != self._set_mtimes[p]]
        if announce and new:
            self._activate_set(new[0], announced=True)
            return
        if active and active["path"] not in files.values():
            if os.path.exists(active["path"]):
                # kept outside the run (a host's population in ~/.cosmoviewer/sets/_live): it is
                # still the population, so keep it listed, never drop it silently
                self._set_files[f"{active['name']} ({len(active['ids']):,})"] = active["path"]
            else:
                self._global_context["selection_set"] = None       # its file was deleted
                active = None
        self.sidebar.set_sets(self._set_leaves(), self._active_leaf())

    # Short marks for the Sets group: why a set is incomplete (selection_sets.INCOMPLETE_REASONS).
    _INCOMPLETE_MARK = {"display_sample": "⚠ sample only", "missing_in_file": "⚠ members missing"}

    def _set_leaves(self) -> list:
        """(leaf, tooltip, shown text) per set file: the shown text says what kind of set it is
        ("Halo 1 (34,120) · group") and marks one known to be incomplete; the tooltip has its
        full description, the warning, where it was made and its file."""
        out = []
        for leaf, p in self._set_files.items():
            try:
                st = selection_sets.read_set(p)
            except Exception:
                out.append((leaf, p, leaf))
                continue
            warning = st.completeness_warning()
            shown = f"{leaf} · {st.kind}"
            if st.complete is False:
                shown += " · " + self._INCOMPLETE_MARK.get(st.incomplete_reason, "⚠ incomplete")
            tip = (st.describe() + (f"\n⚠ {warning}" if warning else "")
                   + f"\n{st.n:,} particle IDs, made in {st.made_in or '?'} {st.created}\n{p}")
            out.append((leaf, tip, shown))
        return out

    def _active_leaf(self):
        a = self._global_context.get("selection_set")
        if not a:
            return None
        return next((leaf for leaf, p in self._set_files.items() if p == a["path"]), None)

    def _activate_set(self, path, announced: bool = False) -> bool:
        """Restrict Analyses to the set in ``path`` (None: every particle)."""
        if path is None:
            self._global_context["selection_set"] = None
            msg = "Analyses use every particle."
        else:
            try:
                st = selection_sets.read_set(path)
            except Exception as exc:
                self.plot_panel.set_status(f"Cannot read set {os.path.basename(str(path))}: {exc}")
                return False
            if str(path) not in self._set_files.values():
                self._set_files[f"{st.name} ({st.n:,})"] = str(path)
            warning = st.completeness_warning()
            self._global_context["selection_set"] = {"name": st.name, "path": str(path), "ids": st.ids,
                                                     "mtime": os.path.getmtime(path),
                                                     # what the set means (selection_sets.py, version 2)
                                                     "kind": st.kind, "definition": st.definition,
                                                     "defined_at": st.defined_at, "complete": st.complete,
                                                     "n_expected": st.n_expected,
                                                     "incomplete_reason": st.incomplete_reason,
                                                     "describe": st.describe(), "warning": warning}
            msg = (("Received from Immersive Mode: " if announced else "")
                   + f"set '{st.name}': {st.describe()}. Analyses now use only these particles."
                   + (f"  ⚠ {warning}" if warning else ""))
        self.sidebar.set_sets(self._set_leaves(), self._active_leaf())
        if isinstance(self._plugin, AnalysisPlugin):
            self._spawn_worker(RecomputeReason.DATA_CHANGE)
        elif self._plugin is not None and path is not None:
            msg += " " + self._field_scope_note()
        self.plot_panel.set_status(msg)
        return True

    def _field_scope_note(self) -> str:
        """P3: what a field view (histogram, line, matrix of a dataset) shows while a set is active.
        Field views are never restricted to a set; saying so stops a whole-snapshot histogram
        being read as the set's."""
        from html import escape                 # the status line is rich text
        sel = self._global_context.get("selection_set")
        if not sel:
            return ""
        return escape(f"Field views show the whole snapshot, not set '{sel['name']}'; "
                      f"the active set applies to Analyses only.", quote=False)

    def _note_field_scope(self, plugin) -> None:
        """After a field view is drawn while a set is active, end its status line with the P3 note
        (the end stays visible: the status line clips from the left)."""
        if plugin is None or isinstance(plugin, AnalysisPlugin) or self._diag_override:
            return
        note = self._field_scope_note()
        if note:
            self.plot_panel.set_status(f"{self.plot_panel.lbl_status.text()}  |  {note}")

    def activate_population(self, pop) -> bool:
        """Restrict Analyses to a population (a selection_sets.SelectionSet), or to every particle (None).

        This is how a host (Immersive Mode, in the same process) hands over its active set.
        A population that is not a file yet is written once to ~/.cosmoviewer/sets/_live/,
        never next to the simulation, and then handled exactly like a set file, so the
        analysis, its provenance and its export are the same as for a set sent from a file."""
        if pop is None:
            return self._activate_set(None)
        path = getattr(pop, "path", "") or ""
        if not path or not os.path.exists(path):
            if self._vset is None:
                return False
            live = selection_sets.FALLBACK_ROOT / "_live" / f"{selection_sets._safe(pop.name)}.hdf5"
            path = selection_sets.write_set(self._vset.files[0], pop.ids, pop.name, pop.colour or "#ffffff",
                                            made_in=pop.made_in or "CosmoViewer", description=pop.description,
                                            path=str(live), kind=pop.kind, definition=pop.definition,
                                            defined_at=pop.defined_at, complete=pop.complete,
                                            n_expected=pop.n_expected,
                                            incomplete_reason=getattr(pop, "incomplete_reason", None))
        return self._activate_set(path)

    def closeEvent(self, event) -> None:
        selection_sets.unregister_instance(os.getpid())
        super().closeEvent(event)

    # ── Host mode: the Analyse view of Immersive Mode, in its window ──────

    def set_hosted(self, hosted: bool = True) -> None:
        """Run as a host window's Analyse view (Immersive Mode, in the same process).

        The host decides which snapshot and which population are analysed. So a
        hosted Explorer offers no "Open Snapshot…", does not record itself in
        ~/.cosmoviewer/explorer_instances (Immersive's "Send to Explorer" would then
        find its own process and send its sets nowhere), and set files appearing in
        the run folder never replace the host's population (_refresh_sets)."""
        self._hosted = bool(hosted)
        self.sidebar.btn_open.setVisible(not self._hosted)
        if self._hosted:
            selection_sets.unregister_instance(os.getpid())

    def open_snapshot_file(self) -> str:
        """The snapshot open here (its first file when it has several), or "" before one is opened."""
        return str(self._vset.files[0]) if self._vset is not None else ""

    def shown_leaf(self):
        """(group, leaf) of the field or analysis on show, as in the tree; None before one is chosen."""
        return next((key for key, p in self._plugin_cache.items() if p is self._plugin), None)

    def follow_snapshot(self, path: str) -> bool:
        """Show another snapshot in place of the open one, as the same analysis (host mode).

        What the researcher is looking at carries over: the population, the field or
        analysis chosen in the tree and its tab. Stepping through a run in the host
        then shows the same measurement at each snapshot, with the population traced
        by its particle IDs. A computation still running for the old snapshot is
        cancelled first, so its result cannot appear under the new one. When the
        snapshot cannot be opened, the previous one and its result stay, the status
        line says why, and False is returned."""
        shown = self.shown_leaf()
        tab = self.plot_panel.current_tab()
        population = self._global_context.get("selection_set")
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
            self._set_loading_state(False)
        if not self.open_path(path):
            return False
        if population is not None:
            if os.path.exists(population["path"]):
                self._activate_set(population["path"])
            else:
                self.plot_panel.set_status(f"The file of set '{population['name']}' has gone; "
                                           f"Analyses now use every particle.")
        if shown is not None and self.sidebar.select_leaf(*shown):
            if tab != self.plot_panel.current_tab() and tab in ["Matrix"] + list(self._plugin_cls.TABS):
                self.plot_panel._toolbar._switch_view(tab)
        return True

    def open_from_command_line(self, plan, dataset: str = "", tab: str = "", set_path: str = "") -> None:
        """Open the snapshot named at launch, then the field and tab, if given."""
        for err in plan.errors:
            print(f"Cannot open {err}")
        files = list(plan.files)
        if plan.folders and not files:
            self.plot_panel.set_status(
                "Give a snapshot file, not a folder: the Explorer opens one snapshot.")
            print("The Explorer opens one snapshot: name a file, not a folder.")
            return
        if not files:
            return
        # Pieces of one multi-file snapshot arrive together; any piece opens all.
        if len({_snapshot_base(f) for f in files}) > 1:
            print(f"The Explorer opens one snapshot; opening {files[0].name}, "
                  f"ignoring {len(files) - 1} other file(s).")
        if not self.open_path(str(files[0])):
            return
        if set_path:
            self._activate_set(os.path.abspath(set_path))
            if not dataset:
                analysis = next((leaf for leaf in self.sidebar.available_leaves()
                                 if leaf.startswith("Analyses/")), None)
                if analysis:                    # show the set's result straight away
                    self.sidebar.select_leaf(*analysis.split("/", 1))
        if not dataset:
            return
        group, _, leaf = dataset.strip("/").partition("/")
        if not self.sidebar.select_leaf(group, leaf):
            msg = f"No field {dataset} in this file."
            self.plot_panel.set_status(msg)
            print(msg + " Fields: " + ", ".join(self.sidebar.available_leaves()))
            return
        if tab:
            allowed = list(getattr(self._plugin_cls, "TABS", [])) + ["Matrix"]
            match = next((t for t in allowed if t.lower() == tab.lower()), None)
            if match is None:
                print(f"Tab {tab} is not available for {dataset}; choose from {', '.join(allowed)}.")
            elif match != self.plot_panel.current_tab():
                self.plot_panel._toolbar._switch_view(match)

    # ── Tree click ─────────────────────────────────────────────────────────

    def _on_leaf_clicked(self, group: str, dataset: str):
        if self._diag_override:
            self.plot_panel.set_status(
                "⚡ Diagnostic override active — click ✕ to restore HDF5 view.")
            return

        if self._vset is None:
            return

        if group == "Sets":
            self._activate_set(None if dataset == FileTreePanel.ALL_PARTICLES
                               else self._set_files.get(dataset))
            return

        plugin_cls = resolve_plugin_class(group, dataset)
        if plugin_cls is None:
            self.plot_panel.set_status(
                f"{group}/{dataset} — no plugin registered yet.")
            return

        # Cache-aware lookup: the constructor runs only on a cache miss.
        # Reusing the existing instance preserves _last_bin_count hysteresis
        # across re-renders of the same dataset.
        key = (group, dataset)
        if key not in self._plugin_cache:
            self._plugin_cache[key] = plugin_cls()
        plugin = self._plugin_cache[key]

        if plugin_cls is not self._plugin_cls:
            self._plugin_cls       = plugin_cls
            self._custom_bin_count = None   # clear override on dataset change
            self.plot_panel.configure_for_plugin(plugin_cls)

        self._plugin = plugin   # always re-bind (no-op when same dataset)
        self._spawn_worker(RecomputeReason.DATA_CHANGE)

    # ── Tab / scale / nav signals ──────────────────────────────────────────

    def _on_tab_changed(self, tab: str):
        # Cancel any in-flight worker before dispatching a new compute.
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if self._diag_override:
            self._compute_and_render_diagnostic()
        else:
            self._spawn_worker(RecomputeReason.PRESENTATION_CHANGE)

    def _on_scale_changed(self, _axis: str, _mode: str):
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if self._diag_override:
            self._compute_and_render_diagnostic()
        else:
            self._spawn_worker(RecomputeReason.PRESENTATION_CHANGE)

    # ── Phase 1: Cosmology context bar ────────────────────────────────────────

    def _update_cosmo_header(self, path: str) -> None:
        """
        Delegate to astro_units.cosmology_context() and push the result to
        the header bar.  Called once per file open.

        Reads strictly from self._cached_header / self._cached_params which
        are already in RAM (populated by VirtualSnapshotSet from chunk 0).
        Zero additional disk I/O is performed here.
        """
        if self._vset is None:
            return

        # Snapshot number: Gadget convention — last 3-or-4-digit run in the
        # stem, e.g. "snapshot_015.hdf5" → 15, "snap_0031.hdf5" → 31
        stem       = os.path.splitext(os.path.basename(path))[0]
        snap_match = re.search(r"(\d{3,4})$", stem)
        snap_num   = int(snap_match.group(1)) if snap_match else None

        ctx = cosmology_context(self._vset.global_header, self._vset.global_params)
        ctx["snapshot_num"] = snap_num

        self.plot_panel.set_cosmology_context(ctx)

    # ── Phase 2: Unit conversion handler ──────────────────────────────────────

    def _on_unit_changed(self, label: str) -> None:
        """
        User selected a new display unit from the toolbar dropdown.

        Persists the choice keyed by plugin class so navigating away and
        back preserves the selection.  Then re-runs the full compute cycle
        so the scalar is recalculated and applied to the raw linear array
        before it reaches pyqtgraph.

        This handler is NOT called during programmatic dropdown updates
        (ViewToolbar.update_unit_dropdown blocks signals), so no infinite
        compute loop is possible.
        """
        if self._plugin_cls is not None and label:
            self._unit_labels[self._plugin_cls] = label
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if not self._diag_override:
            self._spawn_worker(RecomputeReason.PRESENTATION_CHANGE)

    def _on_custom_bins_requested(self, n: int) -> None:
        """
        User confirmed a custom bin count via the 'Override Histogram
        Resolution…' context-menu dialog.

        Caches the value in _custom_bin_count (cleared on dataset switch)
        and immediately re-renders so the histogram updates in-place.
        The params dict injected into plugin.compute() will carry
        'custom_bins': n which Stage 5 intercepts before the FD rule.
        """
        self._custom_bin_count = n
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None
        if not self._diag_override:
            self._spawn_worker(RecomputeReason.PRESENTATION_CHANGE)

    # ── Unified compute dispatch ──────────────────────────────────────────────

    def _spawn_worker(self, reason: RecomputeReason = RecomputeReason.DATA_CHANGE) -> None:
        """
        Cancel any in-flight worker, build a new ComputeWorker, connect its
        plotReady signal, and start it.  The main thread immediately enters
        loading state (sidebar + toolbar disabled) so the user cannot fire a
        second selection while the HDF5 read is in progress.

        All re-renders — tab, scale, unit, bin-count, coordinate-system —
        route through this single dispatch point.  This eliminates the class
        of bug where a new signal path bypasses cancellation.

        *reason* gates the execution path:
          PRESENTATION_CHANGE + cache ready  → synchronous on main thread,
                                               zero QThread overhead, no flicker
          PRESENTATION_CHANGE (cold cache)   → falls through to worker which
                                               runs the full path and fills cache
          DATA_CHANGE                        → full HDF5 read via ComputeWorker
        """
        if self._plugin is None or self._vset is None:
            return

        # ── PRESENTATION_CHANGE fast path: skip QThread entirely ─────────────
        # plugin.can_render_from_cache() returns True only when _cached_cgs is
        # populated AND the tab is not "Matrix" (Matrix needs raw data_arrays).
        # _compute_from_cache() is pure numpy — no h5py, no Qt widget calls —
        # so running it synchronously on the main thread is safe.  Eliminates
        # ~1-2 ms thread scheduling overhead and prevents "Computing…" flicker.
        current_tab = self.plot_panel.current_tab()
        if (
            reason is RecomputeReason.PRESENTATION_CHANGE
            and self._plugin.can_render_from_cache(current_tab)
        ):
            if self._active_worker is not None:
                self._active_worker.cancel()
                self._active_worker = None
            params: dict = dict(self._vset.global_params)
            params["unit_label"] = self._unit_labels.get(self._plugin_cls, "")
            if self._custom_bin_count is not None:
                params["custom_bins"] = self._custom_bin_count
            try:
                request = self._plugin._compute_from_cache(
                    tab     = current_tab,
                    scale_x = self.plot_panel.current_scale_x(),
                    scale_y = self.plot_panel.current_scale_y(),
                    params  = params,
                    context = dict(self._global_context),
                )
            except Exception:
                return
            self._on_plot_ready(request)   # already on main thread — direct call
            return

        # ── DATA_CHANGE or cold PRESENTATION_CHANGE: use ComputeWorker ────────

        # Cancel the previous compute if still running
        if self._active_worker is not None:
            self._active_worker.cancel()
            self._active_worker = None

        # Build the params snapshot the worker needs.
        # Fresh copy — the worker may extend it freely without mutating
        # the orchestrator's cached global_params dict.
        params: dict = dict(self._vset.global_params)
        params["unit_label"] = self._unit_labels.get(self._plugin_cls, "")
        if self._custom_bin_count is not None:
            params["custom_bins"] = self._custom_bin_count

        worker = ComputeWorker(
            vset    = self._vset,
            plugin  = self._plugin,
            tab     = self.plot_panel.current_tab(),
            scale_x = self.plot_panel.current_scale_x(),
            scale_y = self.plot_panel.current_scale_y(),
            header  = self._vset.global_header,
            params  = params,
            window  = (0, 0),
            context = dict(self._global_context),  # snapshot — not a live ref
            reason  = reason,
        )
        worker.plotReady.connect(self._on_plot_ready)
        worker.finished.connect(lambda: self._on_worker_finished(worker))

        # Held until its finished signal is delivered; level-of-detail builds of
        # earlier workers that have ended are let go here, where nothing of theirs is queued.
        self._live_workers = {w for w in self._live_workers
                              if isinstance(w, ComputeWorker) or not w.isFinished()}
        self._live_workers.add(worker)
        self._active_worker = worker
        self._set_loading_state(True)
        worker.start()

    def _on_plot_ready(self, result: object) -> None:
        """
        Receive the finished result from the worker thread.

        All Qt widget calls happen here — never inside ComputeWorker.run().
        The result is either a PlotRequest (plot tabs) or a _MatrixRequest
        (Matrix tab, carrying raw arrays for render_matrix).
        """
        self._set_loading_state(False)

        if isinstance(result, _MatrixRequest):
            matrix_path = f"{result.plugin.GROUP}/{result.plugin.DATASET}"
            first_arr = next(iter(result.data_arrays.values()), None)
            if first_arr is not None:
                self._n_particles = len(first_arr)
            self.plot_panel.render_matrix(result.data_arrays, matrix_path,
                                          labels=result.labels, notes=result.notes,
                                          info=result.info, status=result.status,
                                          cells=getattr(result, "cells", None),
                                          script=getattr(result, "script", None))
            self._note_field_scope(result.plugin)
            return

        try:
            # vset is passed as a separate kwarg so PlotRequest
            # remains a pure-data object with no live HDF5 file handles.
            # render() stores it only for the Line tab inspector path.
            self.plot_panel.render(result, vset=self._vset)
        except Exception as exc:
            import traceback
            self.plot_panel.set_status(
                f"Render error: {exc}  —  "
                f"{traceback.format_exc().splitlines()[-1]}")
        else:
            self._note_field_scope(self._plugin)
            if getattr(result, "tab", "") == "Histogram":
                self._ensure_bin_index(result)

        # Kick off the per-file prefix-sum build immediately after a
        # Line render so the exact-value inspector path is ready before the user
        # has had time to move the mouse.  The worker is kept alive on self so
        # it is not garbage-collected before run() completes.
        if (
            getattr(result, "tab", "") == "Line"
            and self._vset is not None
            and getattr(result, "dataset_path", "")
        ):
            grp_name = result.dataset_path.split("/")[0]
            worker   = _PrefixWorker(self._vset, grp_name, parent=self)
            worker.finished.connect(lambda: self._prefix_workers.discard(worker))
            self._prefix_workers.add(worker)
            worker.start()

    # ── Click-a-bin index for the Histogram on screen ─────────────────────

    def _ensure_bin_index(self, request) -> None:
        """
        Make sure the index behind "click a histogram bin" exists, or is being
        built, for the histogram just shown, and tell the panel which.

        Every histogram reaches the screen through _on_plot_ready: a fresh read
        and every re-render from cache (tab, scale, unit or bin-count change).
        The build used to be started, and its "ready" signal connected, inside
        ComputeWorker.run; that thread had ended by the time the build finished,
        so the signal was never delivered and the popup said "Index building…"
        until a tab or scale change happened to pass the index along. Started
        here, on the main thread, its signals reach the panel.
        """
        panel  = self.plot_panel
        plugin = self._plugin
        cached = getattr(plugin, "_cached_cgs", None)
        edges  = getattr(request, "bin_edges", None)
        if (plugin is None or self._vset is None or cached is None or edges is None
                or not cached.snapshot_path or not cached.dataset_path):
            panel.set_bin_index_unavailable("this view has no particle list "
                                            "(no snapshot dataset behind it).")
            return
        if getattr(request, "bin_index_note", ""):
            # The histogram is not of the stored quantity times one factor
            # (e.g. temperature from internal energy with μ per particle): an
            # index of the stored values would answer with the wrong particles.
            panel.set_bin_index_unavailable(request.bin_index_note)
            return
        unit_scalar = float(request.unit_scalar) or 1.0
        cgs_edges   = np.asarray(edges, dtype=np.float64) / unit_scalar
        key = LodCacheKey.build(
            snapshot_path = cached.snapshot_path,
            dataset_path  = plugin.lod_key_name(cached.dataset_path),
            cache_dir     = LodIndex.CACHE_DIR,
        )
        lod = getattr(plugin, "_cached_lod", None)
        if lod is not None and lod.matches(key, cgs_edges):
            panel._on_lod_ready(lod)
            return
        w = self._lod_worker
        if w is not None and not w.isFinished() and w.matches(key, cgs_edges):
            panel.set_bin_index_building(w)               # already on its way
            return
        self._cancel_bin_index()
        w = LodBuildWorker(
            plugin       = plugin,
            vset         = self._vset,
            dataset_path = cached.dataset_path,
            bin_edges    = cgs_edges,
            key          = key,
            cgs_factor   = cached.cgs_factor,
        )
        w.buildComplete.connect(panel._on_lod_ready)
        w.buildFailed.connect(panel._on_lod_failed)
        w.progress.connect(panel._on_lod_progress)
        self._live_workers.add(w)          # held until it has finished
        w.finished.connect(lambda: self._live_workers.discard(w))
        self._lod_worker = w
        panel.set_bin_index_building(w)
        w.start()

    def _cancel_bin_index(self) -> None:
        """Stop the bin-index build in flight, if any; it stays held until it ends."""
        if self._lod_worker is not None:
            self._lod_worker.cancel()
            self._lod_worker = None

    def _on_worker_finished(self, worker: object) -> None:
        """
        Clean up the active_worker reference when the thread exits.

        The identity check guards against clearing a newer worker that
        replaced a cancelled one: if the user clicked a second dataset before
        the first worker exited, the first worker's finished signal must not
        clear the second worker's reference.
        """
        if self._active_worker is worker:
            self._active_worker = None
            self._set_loading_state(False)
        # finished is emitted just before the thread ends: wait for that, then let it go.
        # (Bin-index builds are started and held by _ensure_bin_index.)
        worker.wait()
        self._live_workers.discard(worker)

    def _set_loading_state(self, active: bool) -> None:
        """
        Disable/enable the file tree and toolbar while a compute is in-flight.

        Called immediately before worker.start() (active=True) and at the
        start of _on_plot_ready (active=False).

        The status bar update is sub-millisecond, so the user sees the
        'Computing…' label within the first frame after clicking — well
        before any HDF5 read begins.
        """
        self.sidebar.setEnabled(not active)
        if hasattr(self, 'plot_panel') and hasattr(self.plot_panel, '_toolbar'):
            self.plot_panel._toolbar.setEnabled(not active)
        if active:
            self.plot_panel.set_status("Computing…")

    # ── Diagnostic override handlers ───────────────────────────────────────

    def _on_diagnostic_loaded(self, files: list):
        """
        Called by PlotPanel when the user selects 1 or 2 files.
        files = list of (ndarray, filename) tuples.
        """
        from diagnostic_plugin import DiagnosticPlugin

        self._diag_override = True
        self._diag_files    = files
        self._diag_z_swap   = False

        self.sidebar.setEnabled(False)
        self.sidebar.setWindowOpacity(0.6)

        self.plot_panel.show_diagnostic_banner(files)
        self.plot_panel.configure_for_plugin(DiagnosticPlugin)

        if len(files) == 2:
            self.plot_panel.show_swap_button()
        else:
            self.plot_panel.hide_swap_button()

        self._compute_and_render_diagnostic()

    def _on_diagnostic_cleared(self):
        self._diag_override = False
        self._diag_files    =[]
        self._diag_z_swap   = False

        self.sidebar.setEnabled(True)
        self.sidebar.setWindowOpacity(1.0)

        self.plot_panel.hide_diagnostic_banner()

        if self._plugin is not None and self._vset is not None:
            self.plot_panel.configure_for_plugin(self._plugin_cls)
            self._spawn_worker(RecomputeReason.DATA_CHANGE)
        else:
            self.plot_panel.reset_view()
            self.plot_panel.set_status(
                "Diagnostic cleared — open a snapshot and select a dataset.")

    def _on_swap_z_order(self):
        self._diag_z_swap = not self._diag_z_swap
        self._compute_and_render_diagnostic()

    def _compute_and_render_diagnostic(self):
        """
        Diagnostic compute cycle — feeds DiagnosticPlugin instead of an HDF5 plugin.

        Matrix tab : renders file A's flattened array via render_matrix.
        Overlay tab: calls DiagnosticPlugin.compute() → PlotPanel.render().
        """
        from diagnostic_plugin import DiagnosticPlugin

        if not self._diag_files:
            return

        tab     = self.plot_panel.current_tab()
        scale_x = self.plot_panel.current_scale_x()
        scale_y = self.plot_panel.current_scale_y()

        if tab == "Matrix":
            arr_flat = self._diag_files[0][0].flatten()
            self.plot_panel.render_matrix({"_diag_A": arr_flat}, "_diag_A")
            return

        arr_A = self._diag_files[0][0]
        data: dict = {"diag_A": arr_A}
        params: dict = {
            "diag_name_A": self._diag_files[0][1],
            "diag_z_swap": self._diag_z_swap,
            # Synthetic n_global_map: for externally loaded arrays the global
            # count equals the local length (no distributed striding).
            "n_global_map": {"diag_A": len(arr_A)},
        }
        if len(self._diag_files) >= 2:
            arr_B = self._diag_files[1][0]
            data["diag_B"]                    = arr_B
            params["diag_name_B"]             = self._diag_files[1][1]
            params["n_global_map"]["diag_B"]  = len(arr_B)

        try:
            request = DiagnosticPlugin().compute(
                data    = data,
                attrs   = {},
                header  = {},
                params  = params,
                tab     = tab,
                scale_x = scale_x,
                scale_y = scale_y,
                window  = (0, 0),
            )
        except Exception as exc:
            self.plot_panel.set_status(f"Diagnostic render error: {exc}")
            return

        try:
            self.plot_panel.render(request)
        except Exception as exc:
            import traceback
            self.plot_panel.set_status(
                f"Render error: {exc}  —  "
                f"{traceback.format_exc().splitlines()[-1]}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _snapshot_base(path) -> str:
    m = re.match(r"^(.+)\.\d+\.(hdf5|h5)$", os.path.basename(str(path)))
    return os.path.join(os.path.dirname(str(path)), m.group(1)) if m else str(path)


def _parser():
    from cli_import import build_parser
    p = build_parser("dataset-explorer",
                     "Dataset Explorer: histograms, curves and tables of one snapshot's fields.")
    p.add_argument("--dataset", default="", metavar="GROUP/FIELD",
                   help="field to open, as in the tree, e.g. PartType0/Density "
                        "or \"PartType1/Velocities [x]\"")
    p.add_argument("--tab", default="", choices=["Histogram", "Line", "Matrix",
                                                 "histogram", "line", "matrix"],
                   help="view to show for --dataset")
    p.add_argument("--set", default="", metavar="SET.hdf5",
                   help="selection-set file (from Immersive Mode's Send to Explorer): "
                        "restrict the Analyses to its particles")
    return p


def main(argv=None):
    """
        python density_explorer.py                          window only
        python density_explorer.py <snapshot.hdf5>          open that snapshot
        python density_explorer.py <file> --dataset PartType0/Density --tab Line
        python density_explorer.py <file> --list            print what would open
    """
    import multiprocessing as mp
    from cli_import import parse_args, resolve_sources
    mp.freeze_support()
    args, qt_argv = parse_args(_parser(), list(sys.argv if argv is None else argv))
    plan = resolve_sources(args.sources)
    if args.list:
        print(plan.describe())
        sys.exit(1 if plan.errors else 0)
    app    = QApplication(qt_argv)
    window = CosmoExplorer()
    window.show()
    if args.sources:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: window.open_from_command_line(plan, args.dataset, args.tab, args.set))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()