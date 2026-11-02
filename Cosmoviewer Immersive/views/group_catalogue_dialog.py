"""
GroupCatalogueDialog: Structures > Read halo catalogue, after GadgetViewer's File > Read groups.

Only Gadget-4 HDF5 catalogues (fof_tab, fof_subhalo_tab) can be read, so only that format is
offered. VELOCIraptor and binary SubFind readers would need test files to be written against;
until those exist the dialog does not list them at all.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QGroupBox, QLineEdit
)
from PyQt6.QtCore import pyqtSignal


class GroupCatalogueDialog(QDialog):
    """Choose a halo catalogue file (Gadget-4 HDF5, the only format that can be read)."""

    catalog_load_requested = pyqtSignal(str, str)  # (format_type, file_path)

    FORMATS = {
        'GADGET4': 'Gadget-4 fof_tab / fof_subhalo_tab (*.hdf5)',
    }
    READABLE = ('GADGET4',)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Read Group Catalogue (Halo Finder)")
        self.resize(500, 260)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        grp_box = QGroupBox("Halo Finder Output Format")
        gb = QVBoxLayout(grp_box)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Catalog Format:"))
        self.combo_fmt = QComboBox()
        for k, v in self.FORMATS.items():
            self.combo_fmt.addItem(v, k)
            if k not in self.READABLE:
                self.combo_fmt.model().item(self.combo_fmt.count() - 1).setEnabled(False)
        h1.addWidget(self.combo_fmt)
        gb.addLayout(h1)

        h2 = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("Select halo catalog file...")
        h2.addWidget(self.edit_path)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        h2.addWidget(self.btn_browse)
        gb.addLayout(h2)

        layout.addWidget(grp_box)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Group Catalogue")
        self.btn_load.setStyleSheet("background-color: #007acc; font-weight: bold; color: white;")
        self.btn_load.clicked.connect(self._on_load_clicked)
        btn_layout.addWidget(self.btn_load)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def _on_browse_clicked(self):
        fmt_key = self.combo_fmt.currentData()
        if fmt_key in ('GADGET4', 'VELOCIRAPTOR'):
            f_filter = "HDF5 Files (*.hdf5 *.h5 *.catalog_groups);;All Files (*)"
        else:
            f_filter = "Binary Group Files (*.group_tab* *.sub_tab*);;All Files (*)"
        
        fname, _ = QFileDialog.getOpenFileName(self, "Select Group Catalogue File", "", f_filter)
        if fname:
            self.edit_path.setText(fname)

    def _on_load_clicked(self):
        fname = self.edit_path.text().strip()
        fmt_key = self.combo_fmt.currentData()
        if fname:
            self.catalog_load_requested.emit(fmt_key, fname)
            self.accept()
