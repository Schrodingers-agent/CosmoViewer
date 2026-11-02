"""
VISIBILITY LAYERS box of the side panel: one checkbox per particle type present
in the current snapshot, labelled with its name and count. The checkboxes write
to a LayerState, which the controller reads to decide which types to draw.
"""

from typing import Dict, List
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox
from PyQt6.QtCore import Qt, pyqtSignal

from gadget_file_discovery import ParticleTypeInfo
from model.simulation_model import LayerState


class DynamicLayerPanel(QGroupBox):
    """View component: dynamically generated checkboxes for particle type visibility."""

    layer_changed = pyqtSignal()

    # The scene's CosmoViewerController.TYPE_COLORS in 0-255, so each label matches its particles.
    TYPE_COLORS: Dict[int, tuple] = {
        0: (0, 204, 255),       # Cyan      - Gas
        1: (204, 26, 26),       # Red       - Dark Matter
        2: (255, 255, 230),     # Off-white - Disk Stars
        3: (255, 230, 128),     # Yellow    - Bulge Stars
        4: (128, 0, 255),       # Purple    - Stars
        5: (51, 51, 51),        # Dark grey - Black holes (boundary particles in old Gadget-2 runs)
    }

    def __init__(self):
        super().__init__("VISIBILITY LAYERS")
        self.setObjectName("Title")

        self.layer_state = LayerState()
        self.checkboxes: Dict[int, QCheckBox] = {}
        self.current_types = set()

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(10, 15, 10, 10)

    def rebuild_for_snapshot(self, particle_types: List[ParticleTypeInfo]):
        existing = {p.type_id for p in particle_types if p.exists}
        if existing == self.current_types:
            # Same types, new counts: gas turns into stars between snapshots, so the labels must follow.
            for ptype in particle_types:
                if ptype.exists and ptype.type_id in self.checkboxes:
                    self.checkboxes[ptype.type_id].setText(self._label(ptype))
            return

        self._clear_layout()

        for ptype in particle_types:
            if not ptype.exists:
                continue
            cb = self._create_checkbox(ptype)
            self.checkboxes[ptype.type_id] = cb
            self.main_layout.addWidget(cb)

        self.main_layout.addStretch()
        self.current_types = existing

    @staticmethod
    def _label(ptype: ParticleTypeInfo) -> str:
        return f"{ptype.type_name} ({ptype.count:,})"

    def _create_checkbox(self, ptype: ParticleTypeInfo) -> QCheckBox:
        cb = QCheckBox(self._label(ptype))
        cb.setChecked(self.layer_state.is_visible(ptype.type_id))
        cb.stateChanged.connect(
            lambda state, tid=ptype.type_id: self._on_toggled(tid, state)
        )
        if ptype.type_id in self.TYPE_COLORS:
            r, g, b = self.TYPE_COLORS[ptype.type_id]
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: rgb({r},{g},{b});
                    font-weight: bold;
                    font-size: 11px;
                }}
            """)
        return cb

    def _on_toggled(self, type_id: int, state: int):
        self.layer_state.set_visible(type_id, state == Qt.CheckState.Checked.value)
        self.layer_changed.emit()

    def _clear_layout(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            w = item.widget()
            if w:
                # Off the panel at once: deleteLater alone leaves the old checkbox drawn until the
                # event loop gets round to deleting it.
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.checkboxes.clear()

    def get_visible_types(self, existing_types: List[int]) -> List[int]:
        return self.layer_state.get_drawable_types(existing_types)
