"""
Collapsible sections for the side panel: a header that shows or hides a box.

The header is styled like the panel's group-box titles, so a collapsed panel
reads as the same title with an arrow. Whether each section is open is
remembered between sessions (QSettings), when settings can be written.
"""

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QGroupBox, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """``body`` under a clickable title; a QGroupBox body keeps its frame and loses its own title."""

    def __init__(self, title: str, body: QWidget, key: str = "", expanded: bool = True, parent=None):
        super().__init__(parent)
        self.body, self.key = body, key
        if key:
            try:
                expanded = QSettings("CosmoViewer", "SidePanel").value(f"open/{key}", expanded, type=bool)
            except Exception:  # unreadable settings: use the default
                pass
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.button.setStyleSheet("QToolButton { border: none; color: #007acc; font-weight: bold; "
                                  "text-align: left; padding: 2px 0; }")
        self.button.toggled.connect(self.set_expanded)
        if isinstance(body, QGroupBox):
            body.setTitle("")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.button)
        lay.addWidget(body)
        self.set_expanded(expanded)

    def is_expanded(self) -> bool:
        return self.button.isChecked()

    def set_expanded(self, on: bool):
        if self.button.isChecked() != on:
            self.button.setChecked(on)          # comes back here through toggled
            return
        self.button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)
        if self.key:
            try:
                QSettings("CosmoViewer", "SidePanel").setValue(f"open/{self.key}", on)
            except Exception:  # settings not writable: the state is simply not remembered
                pass
