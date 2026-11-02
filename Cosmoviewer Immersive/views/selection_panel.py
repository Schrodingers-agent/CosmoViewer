"""
ParticleSelectionPanel: the six selection sets in the side panel, modelled on
GadgetViewer's particle selection window.

For the set chosen at the top it offers the switch, name and colour; the
criteria (radius, FoF group, property range, catalogue group, ID list),
combined as New, Add or Subset; what the set means; ID files and Analyse ▸;
picking lit particles; and tracking across snapshots. The
sets live in the model's ParticleTracker. The panel edits their names and
switches directly and leaves everything that reads particles to the controller,
through its signals.
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QComboBox,
    QLineEdit, QPushButton, QLabel, QCheckBox, QRadioButton, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy
from model.simulation_model import ParticleTracker, ParticleSelectionSet


class ParticleSelectionPanel(QGroupBox):
    """
    UI Panel for Particle Selection & Tracking across snapshot sequences.
    Particle selection, modelled on GadgetViewer's.
    """

    selection_changed = pyqtSignal()
    apply_requested = pyqtSignal()            # "Apply Criteria": run the checked criteria
    recentre_requested = pyqtSignal()
    box_centre_requested = pyqtSignal()
    draw_selected_only_toggled = pyqtSignal(bool)
    # Tracking across snapshots
    ids_load_requested = pyqtSignal()
    ids_save_requested = pyqtSignal()
    send_to_explorer_requested = pyqtSignal()
    analyse_requested = pyqtSignal()          # "Analyse ▸": the active set, in the Analyse tab
    export_tracks_requested = pyqtSignal()
    tracking_options_changed = pyqtSignal()
    colour_change_requested = pyqtSignal()
    # Picking lit particles (controller/picking_controller.py)
    pick_list_requested = pyqtSignal()
    pick_list_chosen = pyqtSignal(int)

    COLORS_HEX = [
        "#FFFFFF",  # Set 1: White
        "#FFFF00",  # Set 2: Yellow
        "#FF00FF",  # Set 3: Magenta
        "#00FFFF",  # Set 4: Cyan
        "#00FF00",  # Set 5: Green
        "#0000FF",  # Set 6: Blue
    ]

    def __init__(self, tracker: ParticleTracker):
        super().__init__("PARTICLE SELECTION (6 SETS)")
        self.tracker = tracker
        self.active_set_idx = 0
        self.draw_selected_only = False

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 15, 10, 10)

        # Set Selector Dropdown with Color indicator
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Set:"))
        self.combo_sets = QComboBox()
        for i, s in enumerate(tracker.sets):
            self.combo_sets.addItem(f"Set {i+1}: {s.name}")
        self.combo_sets.currentIndexChanged.connect(self._on_set_changed)
        h1.addWidget(self.combo_sets)
        layout.addLayout(h1)

        # What the active set is (its kind, definition, snapshot, completeness), on one line;
        # the full text, and the warning for a set known to be incomplete, in the tooltip.
        self.lbl_meaning = _OneLineLabel()
        layout.addWidget(self.lbl_meaning)

        # Enable & Rename
        h2 = QHBoxLayout()
        self.chk_enable = QCheckBox("Enable")
        self.chk_enable.setChecked(True)
        self.chk_enable.stateChanged.connect(self._on_enable_toggled)
        h2.addWidget(self.chk_enable)

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Selection Name")
        self.edit_name.textChanged.connect(self._on_name_edited)
        h2.addWidget(self.edit_name)
        self.btn_colour = QPushButton("Colour")
        self.btn_colour.setToolTip("Colour used to highlight this set in the view, plots and movies")
        self.btn_colour.clicked.connect(lambda: self.colour_change_requested.emit())
        h2.addWidget(self.btn_colour)
        layout.addLayout(h2)

        # Selection Criteria Mode (New, Add, Subset)
        mode_box = QGroupBox("Selection Mode")
        mb = QHBoxLayout(mode_box)
        mb.setContentsMargins(5, 5, 5, 5)
        self.radio_new = QRadioButton("New")
        self.radio_new.setChecked(True)
        self.radio_add = QRadioButton("Add")
        self.radio_subset = QRadioButton("Subset")
        mb.addWidget(self.radio_new)
        mb.addWidget(self.radio_add)
        mb.addWidget(self.radio_subset)
        layout.addWidget(mode_box)

        # Selection Criteria Inputs (Radius & FoF Group ID)
        crit_box = QGroupBox("Criteria")
        cb = QVBoxLayout(crit_box)
        cb.setContentsMargins(5, 5, 5, 5)

        h_rad = QHBoxLayout()
        self.chk_radius = QCheckBox("Radius r <= ")
        self.chk_radius.setToolTip("Distance from the centre of the 3D view (code units, periodic in a box)")
        h_rad.addWidget(self.chk_radius)
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setDecimals(4)
        self.spin_radius.setRange(0.0001, 100000.0)
        self.spin_radius.setValue(10.0)
        h_rad.addWidget(self.spin_radius)
        self.lbl_radius_unit = QLabel("code units")      # the run's length unit, set when it loads
        self.lbl_radius_unit.setToolTip("Lengths in the 3D view and in selections are as stored in the snapshot")
        h_rad.addWidget(self.lbl_radius_unit)
        self.btn_box_centre = QPushButton("Box centre")
        self.btn_box_centre.setToolTip("Move the 3D view's centre, which the radius is measured from,\n"
                                       "to the middle of the box")
        self.btn_box_centre.clicked.connect(lambda: self.box_centre_requested.emit())
        h_rad.addWidget(self.btn_box_centre)
        cb.addLayout(h_rad)
        # where the radius is measured from; the controller keeps it current as the camera moves
        self.lbl_radius_centre = QLabel("")
        self.lbl_radius_centre.setStyleSheet("color: #aaa;")
        self.lbl_radius_centre.setWordWrap(True)
        cb.addWidget(self.lbl_radius_centre)

        self.chk_full = QCheckBox("Search full snapshot, not the sample")
        self.chk_full.setToolTip("Radius selection reads every particle from the snapshot file(s);\n"
                                 "otherwise only the displayed (decimated) particles are searched.")
        self.chk_full.setChecked(True)
        cb.addWidget(self.chk_full)

        h_fof = QHBoxLayout()
        self.chk_fof = QCheckBox("FoF Group ID = ")
        h_fof.addWidget(self.chk_fof)
        self.spin_fof_id = QSpinBox()
        self.spin_fof_id.setRange(1, 999999)
        self.spin_fof_id.setValue(1)
        h_fof.addWidget(self.spin_fof_id)
        cb.addLayout(h_fof)

        # Property range (GadgetViewer: select by temperature, density, ...)
        h_prop = QHBoxLayout()
        self.chk_prop = QCheckBox("")
        self.chk_prop.setToolTip("Select particles whose property lies in [min, max].\n"
                                 "File properties are searched in the full snapshot when\n"
                                 "'Search full snapshot' is ticked; attached ones in the sample.")
        h_prop.addWidget(self.chk_prop)
        self.combo_prop = QComboBox()
        self.combo_prop.setMinimumContentsLength(10)
        h_prop.addWidget(self.combo_prop, 1)
        cb.addLayout(h_prop)
        h_rng = QHBoxLayout()
        h_rng.addSpacing(20)
        self.edit_prop_min = QLineEdit()
        self.edit_prop_min.setPlaceholderText("min")
        self.edit_prop_max = QLineEdit()
        self.edit_prop_max.setPlaceholderText("max")
        h_rng.addWidget(self.edit_prop_min)
        h_rng.addWidget(QLabel("to"))
        h_rng.addWidget(self.edit_prop_max)
        cb.addLayout(h_rng)

        # Halo catalogue group, all members at full resolution
        h_cat = QHBoxLayout()
        self.chk_cat = QCheckBox("Catalogue")
        self.chk_cat.setToolTip("Every member of a group (or subhalo) of the loaded halo catalogue,\n"
                                "read from the snapshot at full resolution (1 = first in the catalogue).\n"
                                "Without a catalogue, the membership stored in the snapshot is used\n"
                                "(EAGLE GroupNumber; a subhalo is numbered group × 100000 + SubGroupNumber).")
        h_cat.addWidget(self.chk_cat)
        self.combo_cat = QComboBox()
        self.combo_cat.addItems(["group", "subhalo"])
        h_cat.addWidget(self.combo_cat)
        self.spin_cat = QSpinBox()
        self.spin_cat.setRange(1, 2_000_000_000)
        h_cat.addWidget(self.spin_cat)
        cb.addLayout(h_cat)

        h_ids = QHBoxLayout()
        self.chk_ids = QCheckBox("IDs:")
        h_ids.addWidget(self.chk_ids)
        self.edit_ids = QLineEdit()
        self.edit_ids.setPlaceholderText("e.g. 1021, 5000-5010")
        h_ids.addWidget(self.edit_ids)
        cb.addLayout(h_ids)

        h_io = QHBoxLayout()
        self.btn_load_ids = QPushButton("Load IDs...")
        self.btn_load_ids.clicked.connect(lambda: self.ids_load_requested.emit())
        h_io.addWidget(self.btn_load_ids)
        self.btn_save_ids = QPushButton("Save IDs...")
        self.btn_save_ids.clicked.connect(lambda: self.ids_save_requested.emit())
        h_io.addWidget(self.btn_save_ids)
        self.btn_send = QPushButton("Analyse ▸")
        self.btn_send.setToolTip("Analyse this set in the Analyse tab")
        self.btn_send.clicked.connect(lambda: self.analyse_requested.emit())
        h_io.addWidget(self.btn_send)
        cb.addLayout(h_io)

        layout.addWidget(crit_box)

        # Action Buttons
        h_btns = QHBoxLayout()
        self.btn_apply = QPushButton("Apply Criteria")
        self.btn_apply.setStyleSheet("background-color: #007acc; font-weight: bold; color: white;")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        h_btns.addWidget(self.btn_apply)

        self.btn_clear = QPushButton("Clear Set")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        h_btns.addWidget(self.btn_clear)
        layout.addLayout(h_btns)

        # View Options: Recentre & Draw Selected Only
        h_view = QHBoxLayout()
        self.btn_recentre = QPushButton("Recentre View")
        self.btn_recentre.clicked.connect(lambda: self.recentre_requested.emit())
        h_view.addWidget(self.btn_recentre)

        self.chk_draw_only = QCheckBox("Draw Selected Only")
        self.chk_draw_only.stateChanged.connect(self._on_draw_only_toggled)
        h_view.addWidget(self.chk_draw_only)
        layout.addLayout(h_view)

        # Status readout
        self.lbl_status = QLabel("0 particles selected")
        self.lbl_status.setStyleSheet("color: #00e5ff; font-size: 10px;")
        layout.addWidget(self.lbl_status)

        # Pick lit (coloured) particles: by clicking them, or from a list of the highest values
        pick_box = QGroupBox("Pick lit particles")
        pb = QVBoxLayout(pick_box)
        pb.setContentsMargins(5, 5, 5, 5)
        self.chk_pick = QCheckBox("Click a particle in the 3D view to pick it")
        self.chk_pick.setToolTip("A click that does not move the camera picks the nearest lit particle under the\n"
                                 "cursor into the active set (Shift+click adds to it). Picked particles are\n"
                                 "followed through every snapshot by ID, like any selection.")
        pb.addWidget(self.chk_pick)
        h_top = QHBoxLayout()
        self.combo_top = QComboBox()
        self.combo_top.setMinimumContentsLength(16)
        self.combo_top.setToolTip("Lit particles with the highest values of the colouring property,\n"
                                  "from the full snapshot for file properties. Choose one to pick it\n"
                                  "(New or Add, as above) and centre the view on it.")
        self.combo_top.activated.connect(lambda i: self.pick_list_chosen.emit(i))
        h_top.addWidget(self.combo_top, 1)
        self.btn_top = QPushButton("List top")
        self.btn_top.clicked.connect(lambda: self.pick_list_requested.emit())
        h_top.addWidget(self.btn_top)
        pb.addLayout(h_top)
        self.lbl_pick = QLabel("")
        self.lbl_pick.setWordWrap(True)
        self.lbl_pick.setStyleSheet("color: #888; font-size: 10px;")
        pb.addWidget(self.lbl_pick)
        layout.addWidget(pick_box)

        # Tracking across snapshots (GadgetViewer: View > Follow selected particles)
        track_box = QGroupBox("Tracking across snapshots")
        tb = QVBoxLayout(track_box)
        tb.setContentsMargins(5, 5, 5, 5)
        self.chk_follow = QCheckBox("Follow selection when the snapshot changes")
        self.chk_follow.setChecked(True)
        self.chk_follow.stateChanged.connect(lambda _: self.tracking_options_changed.emit())
        tb.addWidget(self.chk_follow)
        h_tr = QHBoxLayout()
        self.chk_trails = QCheckBox("Trails, up to")
        self.chk_trails.stateChanged.connect(lambda _: self.tracking_options_changed.emit())
        h_tr.addWidget(self.chk_trails)
        self.spin_trails = QSpinBox()
        self.spin_trails.setRange(0, 100000)
        self.spin_trails.setValue(100)
        self.spin_trails.setToolTip("Number of particles per set drawn with a trail; "
                                    "the set's centre path is always drawn")
        self.spin_trails.valueChanged.connect(lambda _: self.tracking_options_changed.emit())
        h_tr.addWidget(self.spin_trails)
        h_tr.addWidget(QLabel("particles"))
        tb.addLayout(h_tr)
        self.btn_export = QPushButton("Export trajectories...")
        self.btn_export.clicked.connect(lambda: self.export_tracks_requested.emit())
        tb.addWidget(self.btn_export)
        self.lbl_track = QLabel("")
        self.lbl_track.setWordWrap(True)
        self.lbl_track.setStyleSheet("color: #888; font-size: 10px;")
        tb.addWidget(self.lbl_track)
        layout.addWidget(track_box)

        self._refresh_ui()

    def _on_set_changed(self, idx: int):
        self.active_set_idx = idx
        self._refresh_ui()

    def _on_enable_toggled(self, state: int):
        s_set = self.tracker.get_set(self.active_set_idx)
        s_set.enabled = (state == 2)
        self.selection_changed.emit()

    def _on_name_edited(self, text: str):
        s_set = self.tracker.get_set(self.active_set_idx)
        s_set.name = text or f"Selection {self.active_set_idx + 1}"
        self.combo_sets.setItemText(self.active_set_idx, f"Set {self.active_set_idx+1}: {s_set.name}")

    def _on_apply_clicked(self):
        self.apply_requested.emit()

    def _on_clear_clicked(self):
        s_set = self.tracker.get_set(self.active_set_idx)
        s_set.clear()
        self._refresh_ui()
        self.selection_changed.emit()

    def set_properties(self, names):
        current = self.combo_prop.currentText()
        self.combo_prop.blockSignals(True)
        self.combo_prop.clear()
        self.combo_prop.addItems(list(names))
        if current in names:
            self.combo_prop.setCurrentText(current)
        self.combo_prop.blockSignals(False)

    def property_range(self):
        """(name, lo, hi); an empty bound is open. Raises ValueError on bad numbers."""
        lo_t, hi_t = self.edit_prop_min.text().strip(), self.edit_prop_max.text().strip()
        lo = float(lo_t) if lo_t else -float('inf')
        hi = float(hi_t) if hi_t else float('inf')
        if lo > hi:
            raise ValueError("property minimum is above the maximum")
        return self.combo_prop.currentText(), lo, hi

    def set_tracking_text(self, text: str):
        self.lbl_track.setText(text)

    def _on_draw_only_toggled(self, state: int):
        self.draw_selected_only = (state == 2)
        self.draw_selected_only_toggled.emit(self.draw_selected_only)

    def get_selection_mode(self) -> str:
        if self.radio_new.isChecked():
            return "new"
        elif self.radio_add.isChecked():
            return "add"
        else:
            return "subset"

    def _refresh_ui(self):
        s_set = self.tracker.get_set(self.active_set_idx)
        self.edit_name.setText(s_set.name)
        self.chk_enable.setChecked(s_set.enabled)
        r, g, b = s_set.color[:3]
        self.btn_colour.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {'#000' if r + g + b > 380 else '#fff'};")
        self.lbl_status.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
        self.lbl_status.setText(f"Set {self.active_set_idx+1}: {len(s_set.selected_ids):,} particles selected")
        self._show_meaning(s_set)

    MEANING_COLOUR, WARNING_COLOUR = "#9aa5b1", "#ffb703"

    def _show_meaning(self, s_set):
        """One line saying what the active set is; amber, with the warning in the tooltip,
        when the set is known to hold only part of what it describes."""
        if not s_set.selected_ids:
            self.lbl_meaning.set_full_text("Empty set", "Apply criteria, pick particles or load IDs to fill it.")
            self.lbl_meaning.setStyleSheet(f"color: {self.MEANING_COLOUR}; font-size: 10px;")
            return
        text = s_set.describe()
        warning = s_set.completeness_warning()
        if warning:
            self.lbl_meaning.set_full_text("⚠ " + text, f"{text}\n\n⚠ {warning}")
            self.lbl_meaning.setStyleSheet(f"color: {self.WARNING_COLOUR}; font-size: 10px;")
        else:
            self.lbl_meaning.set_full_text(text, text)
            self.lbl_meaning.setStyleSheet(f"color: {self.MEANING_COLOUR}; font-size: 10px;")


class _OneLineLabel(QLabel):
    """A label that keeps to one line however long its text: the end is replaced by '…' to
    fit the width, and the full text is kept for the tooltip and for tests (full_text)."""

    def __init__(self):
        super().__init__("")
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def set_full_text(self, text: str, tooltip: str = ""):
        self._full = text
        self.setToolTip(tooltip or text)
        self._fit()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        width = max(self.width(), 40)
        self.setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width))
