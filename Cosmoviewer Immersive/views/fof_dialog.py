"""
FoFGroupFinderDialog: Structures > Find FoF groups, modelled on GadgetViewer's
Read Groups and FoF group finder windows (behaviour, not code).

The dialog collects the parameters (linked particle types, linking length as b
or as an absolute r_link, minimum group size), shows progress and the resulting
groups, and asks the controller to select, recentre on or colour by a group.
The defaults for a snapshot come from snapshot_io/fof_snapshot.py, and the run
itself happens in controller/app_controller.py.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QMessageBox,
    QCheckBox, QRadioButton, QButtonGroup, QProgressBar
)
from PyQt6.QtCore import pyqtSignal


class FoFGroupFinderDialog(QDialog):
    """
    UI Dialog for Automatic Friends-of-Friends (FoF) Group Detection.
    Matches Gadgetviewer's group detection interface.
    """

    # dict: b_linking, min_size, r_link (None = use b), link_types (list of PartType ints)
    fof_run_requested = pyqtSignal(object)
    halo_highlight_requested = pyqtSignal(int)  # group_id
    color_by_fof_requested = pyqtSignal()
    recentre_halo_requested = pyqtSignal(int)   # group_id
    fof_cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find FoF groups (friends-of-friends)")
        self.resize(450, 420)
        self.group_sizes: Dict[int, int] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Parameters GroupBox
        param_box = QGroupBox("FoF Algorithm Parameters")
        pb = QVBoxLayout(param_box)

        h0 = QHBoxLayout()
        h0.addWidget(QLabel("Link types:"))
        self.chk_types = {}
        for t, name in enumerate(["Gas", "DM", "Disk", "Bulge", "Stars", "BH"]):
            chk = QCheckBox(name)
            chk.setChecked(t == 1)
            self.chk_types[t] = chk
            h0.addWidget(chk)
        pb.addLayout(h0)

        self.radio_b = QRadioButton("b x mean separation (periodic boxes):")
        self.radio_fixed = QRadioButton("Fixed linking length:")
        self.radio_b.setChecked(True)
        mode = QButtonGroup(self)
        mode.addButton(self.radio_b)
        mode.addButton(self.radio_fixed)

        h1 = QHBoxLayout()
        h1.addWidget(self.radio_b)
        self.spin_b = QDoubleSpinBox()
        self.spin_b.setRange(0.01, 1.00)
        self.spin_b.setSingleStep(0.01)
        self.spin_b.setValue(0.20)
        h1.addWidget(self.spin_b)
        pb.addLayout(h1)

        h1b = QHBoxLayout()
        h1b.addWidget(self.radio_fixed)
        self.spin_length = QDoubleSpinBox()
        self.spin_length.setDecimals(4)
        self.spin_length.setRange(1e-4, 1e6)
        self.spin_length.setValue(1.0)
        h1b.addWidget(self.spin_length)
        self.lbl_length_unit = QLabel("code units")
        h1b.addWidget(self.lbl_length_unit)
        pb.addLayout(h1b)
        self.radio_b.toggled.connect(self._sync_mode)
        self._sync_mode()

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Min Group Size (N_min):"))
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(2, 100000)
        self.spin_min_size.setValue(32)  # Gadget-4 FOF_GROUP_MIN_LEN default
        h2.addWidget(self.spin_min_size)
        pb.addLayout(h2)

        self.btn_run = QPushButton("Run FoF Auto-Detection")
        self.btn_run.setFixedHeight(32)
        self.btn_run.setStyleSheet("background-color: #007acc; font-weight: bold; color: white;")
        self.btn_run.clicked.connect(self._on_run_clicked)
        pb.addWidget(self.btn_run)

        h_prog = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        h_prog.addWidget(self.progress)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.fof_cancel_requested.emit)
        h_prog.addWidget(self.btn_cancel)
        pb.addLayout(h_prog)
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        layout.addWidget(param_box)

        # Detected Halos GroupBox
        halos_box = QGroupBox("Detected FoF Halos")
        hb = QVBoxLayout(halos_box)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Select Halo:"))
        self.combo_halos = QComboBox()
        h3.addWidget(self.combo_halos)
        hb.addLayout(h3)

        btn_layout = QHBoxLayout()
        self.btn_highlight = QPushButton("Select halo (all members)")
        self.btn_highlight.setToolTip("Put every member of this group into the active set, from the full-resolution\n"
                                      "FoF result (not only the members drawn in the display sample)")
        self.btn_highlight.setEnabled(False)
        self.btn_highlight.clicked.connect(self._on_highlight_clicked)
        btn_layout.addWidget(self.btn_highlight)

        self.btn_recentre = QPushButton("Recentre View")
        self.btn_recentre.setEnabled(False)
        self.btn_recentre.clicked.connect(self._on_recentre_clicked)
        btn_layout.addWidget(self.btn_recentre)
        hb.addLayout(btn_layout)

        self.btn_color_fof = QPushButton("Color Particles by FoF Group")
        self.btn_color_fof.setEnabled(False)
        self.btn_color_fof.clicked.connect(self._on_color_fof_clicked)
        hb.addWidget(self.btn_color_fof)

        layout.addWidget(halos_box)

        self.lbl_status = QLabel("Ready to run FoF algorithm.")
        self.lbl_status.setStyleSheet("color: #00e5ff; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

    def _sync_mode(self):
        self.spin_b.setEnabled(self.radio_b.isChecked())
        self.spin_length.setEnabled(self.radio_fixed.isChecked())

    def configure_for_snapshot(self, defaults: dict):
        """Set link types and linking-length mode to suit the loaded run (see fof_snapshot.fof_defaults)."""
        present = set(defaults['present_types'])
        for t, chk in self.chk_types.items():
            chk.setEnabled(t in present)
            chk.setChecked(t in defaults['link_types'])
        periodic = defaults['periodic']
        self.radio_b.setEnabled(periodic)
        (self.radio_b if periodic else self.radio_fixed).setChecked(True)
        if defaults.get('suggested_length'):
            self.spin_length.setValue(defaults['suggested_length'])
        self.lbl_length_unit.setText(defaults.get('length_unit', 'code units'))
        if not periodic:
            self.lbl_status.setText("Isolated run (no periodic box): FoF uses a fixed linking length.")
        self._sync_mode()

    def _on_run_clicked(self):
        types = [t for t, chk in self.chk_types.items() if chk.isChecked()]
        if not types:
            QMessageBox.warning(self, "FoF", "Select at least one particle type to link.")
            return
        self.fof_run_requested.emit({
            'b_linking': self.spin_b.value(),
            'min_size': self.spin_min_size.value(),
            'r_link': self.spin_length.value() if self.radio_fixed.isChecked() else None,
            'link_types': types,
        })

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.progress.setVisible(running)
        self.btn_cancel.setVisible(running)
        self.btn_cancel.setEnabled(running)
        if running:
            self.progress.setValue(0)
            self.lbl_status.setText("Running FoF on the full-resolution snapshot...")

    def set_progress(self, fraction: float, text: str):
        """Progress of a running FoF: fraction in [0, 1] and what it is doing."""
        self.progress.setValue(int(round(1000 * min(max(fraction, 0.0), 1.0))))
        self.lbl_status.setText(f"{text}...")

    def update_results(self, n_groups: int, group_sizes: Dict[int, int],
                       linking_length: Optional[float] = None, n_linked: Optional[int] = None,
                       where: str = ""):
        """Show the groups of one snapshot (``where``: its file name, said in the status line)."""
        self.group_sizes = group_sizes
        self.combo_halos.clear()
        place = f" in {where}" if where else ""
        if n_groups == 0:
            self.lbl_status.setText(f"No FoF groups detected{place} with current parameters.")
            self._set_actions_enabled(False)
            return

        for grp_id, size in group_sizes.items():
            self.combo_halos.addItem(f"FoF Halo #{grp_id} ({size:,} particles)", grp_id)

        detail = ""
        if linking_length is not None and n_linked is not None:
            detail = f" ({n_linked:,} particles linked, l = {linking_length:.5g} {self.lbl_length_unit.text()})"
        self.lbl_status.setText(f"Detected {n_groups:,} FoF groups{place}{detail}.")
        self._set_actions_enabled(True)

    def show_groups_elsewhere(self, sources, here: str):
        """The snapshot on screen (``here``) has no FoF result; ``sources`` are the snapshots that have one.

        The group list is cleared and its actions disabled, so groups of one snapshot are never
        applied to another."""
        self.group_sizes = {}
        self.combo_halos.clear()
        self._set_actions_enabled(False)
        names = ", ".join(sources[:3]) + (f" and {len(sources) - 3} more" if len(sources) > 3 else "")
        self.lbl_status.setText(f"FoF groups were found in {names}, not in {here} (on screen). "
                                f"Run FoF here, or go back to that snapshot to use its groups.")

    def reset(self):
        """Back to the state before any FoF run (a new set of snapshots was loaded)."""
        self.group_sizes = {}
        self.combo_halos.clear()
        self._set_actions_enabled(False)
        self.lbl_status.setText("Ready to run FoF algorithm.")

    def _set_actions_enabled(self, on: bool):
        self.btn_highlight.setEnabled(on)
        self.btn_recentre.setEnabled(on)
        self.btn_color_fof.setEnabled(on)

    def _on_highlight_clicked(self):
        grp_id = self.combo_halos.currentData()
        if grp_id is not None:
            self.halo_highlight_requested.emit(grp_id)

    def _on_recentre_clicked(self):
        grp_id = self.combo_halos.currentData()
        if grp_id is not None:
            self.recentre_halo_requested.emit(grp_id)

    def _on_color_fof_clicked(self):
        self.color_by_fof_requested.emit()
