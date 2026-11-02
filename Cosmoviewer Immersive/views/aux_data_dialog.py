"""
File > Auxiliary data > Read additional data: the fields of GadgetViewer's
window (property name, integer or real, file, format, ASCII header lines and
column or HDF5 dataset, and the particle type the data applies to).
"""

from typing import List

from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from snapshot_io.aux_data import AuxSpec

TYPE_NAMES = ['0 gas', '1 dark matter', '2', '3', '4 stars', '5 black holes']


class AuxDataDialog(QDialog):
    def __init__(self, parent, types_present: List[int]):
        super().__init__(parent)
        self.setWindowTitle("Read additional data")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. Potential")
        form.addRow("Property name:", self.edit_name)
        h = QHBoxLayout()
        self.radio_real = QRadioButton("Real")
        self.radio_real.setChecked(True)
        self.radio_int = QRadioButton("Integer")
        h.addWidget(self.radio_real)
        h.addWidget(self.radio_int)
        w = QWidget()
        w.setLayout(h)
        form.addRow("Data type:", w)
        h = QHBoxLayout()
        self.edit_path = QLineEdit()
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse)
        h.addWidget(self.edit_path)
        h.addWidget(b)
        w = QWidget()
        w.setLayout(h)
        form.addRow("File:", w)
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["HDF5", "ASCII"])
        self.combo_fmt.currentTextChanged.connect(self._sync)
        form.addRow("Format:", self.combo_fmt)
        self.edit_dset = QLineEdit()
        self.edit_dset.setPlaceholderText("dataset path, e.g. PartType1/Potential")
        form.addRow("HDF5 dataset:", self.edit_dset)
        self.spin_header = QSpinBox()
        self.spin_header.setRange(0, 100000)
        form.addRow("ASCII header lines:", self.spin_header)
        self.spin_col = QSpinBox()
        self.spin_col.setRange(1, 1000)
        form.addRow("ASCII column (1 = first):", self.spin_col)
        self.combo_type = QComboBox()
        for t in types_present:
            self.combo_type.addItem(TYPE_NAMES[t], t)
        form.addRow("Applies to type:", self.combo_type)
        lay.addLayout(form)
        note = QLabel("One value per particle of that type, in snapshot order. Files may be split like\n"
                      "snapshots (name.0.hdf5, name.1.hdf5 ...). The snapshot number (_NNN) in the name\n"
                      "is replaced to read the matching file for every snapshot.")
        note.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._sync()

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Additional data file", "", "All files (*)")
        if f:
            self.edit_path.setText(f)
            if not self.edit_name.text():
                from pathlib import Path
                self.edit_name.setText(Path(f).stem.split('.')[0])

    def _sync(self, *_):
        hdf = self.combo_fmt.currentText() == "HDF5"
        self.edit_dset.setEnabled(hdf)
        self.spin_header.setEnabled(not hdf)
        self.spin_col.setEnabled(not hdf)

    def _accept(self):
        if self.edit_name.text().strip() and self.edit_path.text().strip() and self.combo_type.count():
            self.accept()

    def spec(self) -> AuxSpec:
        return AuxSpec(name=self.edit_name.text().strip(), path=self.edit_path.text().strip(),
                       ptype=int(self.combo_type.currentData()),
                       fmt='hdf5' if self.combo_fmt.currentText() == "HDF5" else 'ascii',
                       dataset=self.edit_dset.text().strip(), column=self.spin_col.value() - 1,
                       header_lines=self.spin_header.value(), integer=self.radio_int.isChecked())
