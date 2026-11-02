"""
The Lagrangian-radii Analyses leaves in the Explorer window.

    python3 -m pytest -q test_analyses_window.py

Checks that the window shows exactly what the headless measurement layer
computes, that leaves the file cannot supply are greyed out, that the Centre
menu recomputes, and that an oversized selection is refused, not sampled.
Uses COSMOVIEWER_DM_OUTPUT for the real-data case when set.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import analysis_plugins  # noqa: E402
import cli_import  # noqa: E402
from measurements import SnapshotView, find_centre, lagrangian_radii  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")


def _wait(app, win, timeout=180):
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


def _table(win):
    t = win.plot_panel._matrix_view._table
    labels = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    cells = [[t.item(i, j).text() for j in range(t.columnCount())] for i in range(t.rowCount())]
    return labels, cells


def _shown(values):
    """How the Matrix tab writes these numbers (Auto notation)."""
    from plot_panel import _format_cell
    return [_format_cell(float(v), "Auto") for v in values]


class TestAnalysesWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = tempfile.TemporaryDirectory()
        # Stars (PartType4, Masses dataset) and DM (PartType1, MassTable)
        p4, v4, m4 = plummer(5000, 1e5, 0.001, seed=1)
        p1, v1, _ = plummer(8000, 1e6, 0.003, seed=2)
        code_mass = 1.989e43 / MSUN_G
        cls.snap = Path(cls.tmp.name) / "snapshot_000.hdf5"
        write_gadget4_snapshot(cls.snap, {
            4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / code_mass, ids=np.arange(5000)),
            1: dict(pos=p1 + 2.0, vel=v1, mass=None, ids=np.arange(5000, 13000))},
            mass_table=[0, 1e6 / 8000 / code_mass, 0, 0, 0, 0])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _window(self):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        return win

    def _open(self, win, path, leaf, tab=""):
        win.open_from_command_line(cli_import.resolve_sources([str(path)]), leaf, tab)
        self.assertTrue(_wait(self.app, win))

    def _headless(self, path, types, method="shrinking_sphere"):
        p = SnapshotView(str(path)).particles(types, velocities=False, ids=False)
        return lagrangian_radii(p, find_centre(p, method))

    def test_leaves_follow_the_file(self):
        win = self._window()
        self.assertTrue(win.open_path(str(self.snap)))
        leaves = set(win.sidebar.available_leaves())
        self.assertIn("Analyses/Lagr. radii, all types", leaves)
        self.assertIn("Analyses/Lagr. radii, PartType4", leaves)
        self.assertIn("Analyses/Lagr. radii, PartType1", leaves)
        self.assertNotIn("Analyses/Lagr. radii, PartType0", leaves)

    def test_matrix_equals_headless_result(self):
        """The window shows exactly what the headless measurement layer computes (wiring check;
        the analytic Plummer tests in test_lagrangian_radii_and_centres.py check the numbers)."""
        win = self._window()
        self._open(win, self.snap, "Analyses/Lagr. radii, PartType4", "Matrix")
        labels, cells = _table(win)
        self.assertEqual(labels, ["mass fraction", "radius [kpc]", "sampling error [kpc]"])
        ref = self._headless(self.snap, [4])
        self.assertEqual([c[0] for c in cells], _shown(ref["fraction"]))
        self.assertEqual([c[1] for c in cells], _shown(ref["radius"].value))
        self.assertEqual([c[2] for c in cells], _shown(ref["radius_err_sampling"].value))
        self.assertIn("All 5,000 particles", win.plot_panel.lbl_status.text())
        cols = win.plot_panel._export_columns
        self.assertIsNotNone(cols)
        np.testing.assert_array_equal(cols[1][1][1], ref["radius"].value)   # export is exact
        self.assertTrue(any("centre: shrinking sphere" in n for n in cols[0]))

    def test_profile_tab_and_centre_menu(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Lagr. radii, all types", "Profile")
        self.assertEqual(win.plot_panel.current_tab(), "Profile")
        status = win.plot_panel.lbl_status.text()
        self.assertIn("All 13,000 particles", status)
        self.assertIn("shrinking_sphere", status)
        act = next(a for a in win._centre_action_group.actions() if a.data() == "density")
        act.trigger()
        self.assertTrue(_wait(self.app, win))
        self.assertIn("centre: density", win.plot_panel.lbl_status.text())
        self.assertEqual(win._global_context["centre"], "density")

    def test_masstable_type_uses_its_mass(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Lagr. radii, PartType1", "Matrix")
        _, cells = _table(win)
        ref = self._headless(self.snap, [1])
        self.assertEqual([c[1] for c in cells], _shown(ref["radius"].value))

    def test_oversized_selection_is_refused_not_sampled(self):
        old = analysis_plugins.MAX_ANALYSIS_PARTICLES
        analysis_plugins.MAX_ANALYSIS_PARTICLES = 1000
        self.addCleanup(setattr, analysis_plugins, "MAX_ANALYSIS_PARTICLES", old)
        win = self._window()
        self._open(win, self.snap, "Analyses/Lagr. radii, all types", "Profile")
        self.assertIn("need every particle", win.plot_panel.lbl_status.text())

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_real_dm_snapshot(self):
        path = Path(DM) / "snapshot_000.hdf5"
        win = self._window()
        self._open(win, path, "Analyses/Lagr. radii, all types", "Matrix")
        _, cells = _table(win)
        ref = self._headless(path, [1])
        self.assertEqual([c[1] for c in cells], _shown(ref["radius"].value))
        self.assertIn("All 2,097,152 particles", win.plot_panel.lbl_status.text())


if __name__ == "__main__":
    unittest.main()
