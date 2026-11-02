"""
The sanity, profile and energy Analyses leaves in the Explorer window.

    python3 -m pytest -q test_analyses_window_profiles_energy.py

The sanity report, radial profiles and energy budget must show exactly what
the headless measurement layer computes, and the sanity report must show a
planted defect. Uses COSMOVIEWER_SFR_OUTPUT for the real-data case when set.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_CACHE = tempfile.TemporaryDirectory()
os.environ["COSMOVIEWER_RESULTS_CACHE"] = _CACHE.name     # keep tests out of ~/.cosmoviewer

from PyQt6.QtWidgets import QApplication  # noqa: E402

import cli_import  # noqa: E402
from measurements import SnapshotView, find_centre, lagrangian_radii  # noqa: E402
from measurements.energy import energy_budget  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.profiles import radial_profile  # noqa: E402
from measurements.sanity import sanity_report  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402
from test_analyses_window import _shown, _table, _wait  # noqa: E402

SFR = os.environ.get("COSMOVIEWER_SFR_OUTPUT", "")


class TestAnalysesWindowProfilesEnergy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = tempfile.TemporaryDirectory()
        p4, v4, m4 = plummer(6000, 1e5, 0.01, seed=1)
        code_mass = 1.989e43 / MSUN_G
        cls.snap = Path(cls.tmp.name) / "snapshot_000.hdf5"
        cls.bad = Path(cls.tmp.name) / "snapshot_001.hdf5"
        ids = np.arange(1, 6001)
        write_gadget4_snapshot(cls.snap, {4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / code_mass, ids=ids)})
        dup = ids.copy()
        dup[5] = dup[6]
        write_gadget4_snapshot(cls.bad, {4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / code_mass, ids=dup)})

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

    def test_leaves_follow_the_file(self):
        win = self._window()
        self.assertTrue(win.open_path(str(self.snap)))
        leaves = set(win.sidebar.available_leaves())
        for leaf in ("Sanity report", "Energy, all types", "Profile, all types", "Profile, PartType4"):
            self.assertIn(f"Analyses/{leaf}", leaves)
        self.assertNotIn("Analyses/Profile, PartType0", leaves)

    def test_sanity_report_shows_each_check(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Sanity report", "Report")
        labels, cells = _table(win)
        self.assertEqual(labels, ["status", "check", "scope", "particles", "detail"])
        ref = sanity_report(self.snap)
        self.assertEqual([(c[0], c[1], c[2]) for c in cells],
                         [(c.status, c.check, c.scope) for c in ref.checks])
        self.assertIn("Sanity PASS", win.plot_panel.lbl_status.text())

    def test_sanity_report_flags_duplicate_ids(self):
        win = self._window()
        self._open(win, self.bad, "Analyses/Sanity report", "Report")
        _, cells = _table(win)
        row = next(c for c in cells if c[1] == "ids")
        self.assertEqual(row[0], "FAIL")
        self.assertEqual(row[3], "2")
        self.assertIn("Sanity FAIL", win.plot_panel.lbl_status.text())
        # the export carries the provenance header and the same rows
        notes, header, rows = win.plot_panel._export_text
        self.assertTrue(notes[0].startswith("# Sanity FAIL"))
        self.assertEqual(header[0], "status")
        self.assertEqual(len(rows), len(cells))

    def test_profile_matrix_equals_headless(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Profile, PartType4", "Matrix")
        labels, cells = _table(win)
        self.assertEqual(labels[4], "density [Msun/kpc3]")
        p = SnapshotView(str(self.snap)).particles([4])
        c = find_centre(p, "shrinking_sphere")
        r10 = float(lagrangian_radii(p, c, [0.10])["radius"].value[0])
        c = find_centre(p, "shrinking_sphere", velocity_radius=r10)
        ref = radial_profile(p, c, n_bins=30)
        self.assertEqual([row[4] for row in cells], _shown(ref["density"].value))
        self.assertEqual([row[8] for row in cells], _shown(ref["sigma_r"].value))
        self.assertEqual([row[12] for row in cells], _shown(ref["beta"]))
        self.assertIn("All 6,000 particles", win.plot_panel.lbl_status.text())

    def test_profile_plot_tabs(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Profile, all types", "Density")
        self.assertEqual(win.plot_panel.current_tab(), "Density")
        self.assertIn("All 6,000 particles", win.plot_panel.lbl_status.text())
        for tab in ("Enclosed mass", "Dispersion", "Anisotropy"):
            win.plot_panel._toolbar._view_btns[tab].click()       # as a user would
            self.assertTrue(_wait(self.app, win))
            self.assertEqual(win.plot_panel.current_tab(), tab)
            status = win.plot_panel.lbl_status.text()
            self.assertIn("All 6,000 particles", status, tab)
            self.assertNotIn("Error", status, tab)

    def test_energy_budget_equals_headless(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Energy, all types", "Budget")
        labels, cells = _table(win)
        self.assertEqual(labels, ["quantity", "value", "error", "unit"])
        ref = energy_budget(str(self.snap))
        row = {c[0]: c for c in cells}
        self.assertEqual(row["K, kinetic"][1], _shown([ref["K"].value])[0])
        self.assertEqual(row["W, potential"][1], _shown([ref["W"].value])[0])
        self.assertEqual(row["Q = K / |W|"][1], _shown([ref["Q"]])[0])
        self.assertIn("W: direct", win.plot_panel.lbl_status.text())

    def _run_script(self, win, name):
        import subprocess
        out = Path(self.tmp.name)
        win.plot_panel.export_to(str(out / f"{name}.csv"))
        win.plot_panel.export_to(str(out / f"{name}.py"))
        self.assertIn("reproduces", win.plot_panel.lbl_status.text())
        run = subprocess.run([sys.executable, str(out / f"{name}.py")], capture_output=True,
                             text=True, timeout=600, cwd=str(out))
        self.assertEqual(run.returncode, 0, run.stderr)
        return np.loadtxt(out / f"{name}.csv", delimiter=",", skiprows=1)

    def test_profile_script_reproduces_the_window(self):
        """Plan F4 acceptance: the exported script reproduces the GUI's numbers exactly."""
        win = self._window()
        self._open(win, self.snap, "Analyses/Profile, PartType4", "Matrix")
        notes, cols = win.plot_panel._export_columns
        gui = np.column_stack([c[1] for c in cols])
        script = self._run_script(win, "profile")
        np.testing.assert_array_equal(np.isnan(script), np.isnan(gui))
        np.testing.assert_array_equal(np.nan_to_num(script), np.nan_to_num(
            np.array([[float(f"{x:.10e}") for x in row] for row in gui])))

    def test_energy_script_reproduces_the_window(self):
        win = self._window()
        self._open(win, self.snap, "Analyses/Energy, all types", "Budget")
        ref = energy_budget(str(self.snap))
        got = self._run_script(win, "energy")
        want = [ref["K"].value, ref["U"].value, ref["W"].value, ref["E"].value, ref["Q"]]
        np.testing.assert_array_equal(got[:, 0], [float(f"{x:.10e}") for x in want])

    def test_sanity_script_runs(self):
        win = self._window()
        self._open(win, self.bad, "Analyses/Sanity report", "Report")
        got = self._run_script(win, "sanity")
        self.assertEqual(list(got), [float(c.count) for c in sanity_report(self.bad).checks])

    def test_energy_over_run(self):
        from analysis_plugins_m2 import pick_snapshots
        from measurements.energy import conservation
        from measurements.runset import Run
        self.assertEqual(pick_snapshots(range(335), 60)[:2], [0, 6])
        self.assertEqual(len(pick_snapshots(range(335), 60)), 60)
        self.assertEqual(pick_snapshots(range(335), 60)[-1], 334)
        win = self._window()
        self._open(win, self.snap, "Analyses/Energy over run", "Energy drift")
        self.assertIn("2 snapshots · every particle in each", win.plot_panel.lbl_status.text())
        win.plot_panel._toolbar._view_btns["Matrix"].click()
        self.assertTrue(_wait(self.app, win))
        labels, cells = _table(win)
        self.assertEqual(labels[3], "dE/|E0|")
        _, drift = conservation(Run.from_folder(self.tmp.name), types=[4])
        self.assertEqual([c[3] for c in cells], _shown(next(iter(drift.values()))["dE_rel"]))

    @unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
    def test_phase_diagram_leaf(self):
        from measurements.phase import property_histogram2d
        path = Path(SFR) / "snapshot_100.hdf5"
        win = self._window()
        self._open(win, path, "Analyses/Phase diagram, gas", "Phase diagram")
        self.assertIn("All 144,813 gas particles", win.plot_panel.lbl_status.text())
        win.plot_panel._toolbar._view_btns["Matrix"].click()
        self.assertTrue(_wait(self.app, win))
        _, cells = _table(win)
        ref = property_histogram2d(path, 0, "Density", "Temperature", bins=120)
        g = ref.values["grid"]
        self.assertEqual([c[4] for c in cells], _shown(g[np.nonzero(g)]))

    @unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
    def test_real_sfr_snapshot(self):
        win = self._window()
        self._open(win, Path(SFR) / "snapshot_100.hdf5", "Analyses/Sanity report", "Report")
        self.assertIn("Sanity PASS", win.plot_panel.lbl_status.text())
        self.assertIn("all 1,600,000 particles", win.plot_panel.lbl_status.text())


if __name__ == "__main__":
    unittest.main()
