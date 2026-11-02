"""
Command-line import for the Dataset Explorer: a snapshot, field and tab named
at launch.

    python3 -m pytest -q test_cli_import.py

Uses the Gadget-4 DM example in COSMOVIEWER_DM_OUTPUT when set, otherwise a
small file written here.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import cli_import  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")


def _write_snapshot(path, n=500):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["Time"] = 1.0
        h.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([0, n, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["MassTable"] = np.zeros(6)
        g = f.create_group("PartType1")
        g["Velocities"] = rng.normal(size=(n, 3)).astype(np.float32)
        g["ParticleIDs"] = np.arange(n, dtype=np.uint64)


def _wait(app, win, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        w = win._active_worker
        if w is None or not w.isRunning():
            app.processEvents()
            if win._active_worker is None or not win._active_worker.isRunning():
                return True
        time.sleep(0.02)
    return False


class TestExplorerLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = Path(tempfile.gettempdir()) / f"de_cli_{os.getpid()}"   # TMPDIR, TEMP or /tmp
        cls.tmp.mkdir(exist_ok=True)
        cls.snap = cls.tmp / "snapshot_000.hdf5"
        if DM and (Path(DM) / "snapshot_000.hdf5").exists():
            cls.snap = Path(DM) / "snapshot_000.hdf5"
        else:
            _write_snapshot(cls.snap)

    def _window(self):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        return win

    def test_opens_field_and_tab(self):
        win = self._window()
        plan = cli_import.resolve_sources([str(self.snap)])
        win.open_from_command_line(plan, "PartType1/Velocities [x]", "line")
        self.assertTrue(_wait(self.app, win))
        self.assertIsNotNone(win._vset)
        self.assertEqual(win._plugin_cls.DATASET, "Velocities")
        self.assertEqual(win._plugin_cls.COLUMN, 0)
        self.assertEqual(win.plot_panel.current_tab(), "Line")

    def test_unknown_field_says_so(self):
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap)]), "PartType9/Nothing")
        self.assertIsNotNone(win._vset)
        self.assertIn("No field PartType9/Nothing", win.plot_panel.lbl_status.text())

    def test_folder_is_refused_with_a_reason(self):
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap.parent)]))
        self.assertIsNone(win._vset)
        self.assertIn("not a folder", win.plot_panel.lbl_status.text())

    def test_list_runs_without_a_window(self):
        out = subprocess.run([sys.executable, str(HERE / "density_explorer.py"), str(self.snap),
                              "--list"], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(f"file:   {self.snap}", out.stdout)


if __name__ == "__main__":
    unittest.main()
