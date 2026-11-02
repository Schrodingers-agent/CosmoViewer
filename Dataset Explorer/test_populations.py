"""
The Explorer side of populations (selection_sets.py, version 2).

    python3 -m pytest -q test_populations.py

A host (Immersive Mode in the same process) hands its active set over with
CosmoExplorer.activate_population(): nothing is written next to the
simulation, the analysis is restricted exactly as for a set file, and what
the set means (kind, definition, snapshot, completeness) reaches the status
line and the exported notes. A set known to be incomplete carries a warning
into every analysis made from it.
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

from PyQt6.QtWidgets import QApplication  # noqa: E402

import selection_sets  # noqa: E402
from measurements import SnapshotView, find_centre, lagrangian_radii  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402
from test_analyses_window import _shown, _table, _wait  # noqa: E402

FOF_DEF = {'finder': 'fof', 'group': 1, 'b': 0.2, 'min_size': 32, 'link_types': [4]}


class TestPopulationsInWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = tempfile.TemporaryDirectory()
        cls._old = (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR)
        selection_sets.FALLBACK_ROOT = Path(cls.tmp.name) / "home" / "sets"
        selection_sets.INSTANCES_DIR = Path(cls.tmp.name) / "home" / "explorer_instances"
        p4, v4, m4 = plummer(3000, 1e5, 0.001, seed=4)
        code_mass = 1.989e43 / MSUN_G
        cls.run_dir = Path(cls.tmp.name) / "output"
        cls.run_dir.mkdir()
        cls.snap = cls.run_dir / "snapshot_000.hdf5"
        write_gadget4_snapshot(cls.snap, {4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / code_mass, ids=np.arange(3000))})
        cls.ids = np.arange(0, 3000, 2).astype(np.uint64)

    @classmethod
    def tearDownClass(cls):
        selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR = cls._old
        cls.tmp.cleanup()

    def _window(self):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        self.assertTrue(win.open_path(str(self.snap)))
        return win

    def _population(self, complete=True, n_expected=None):
        return selection_sets.SelectionSet(self.ids, "Halo 1", "#ffaa00", made_in="test", kind="group",
                                           definition=FOF_DEF, defined_at={"file": "snapshot_000.hdf5", "index": 0},
                                           complete=complete, n_expected=n_expected or len(self.ids),
                                           incomplete_reason="display_sample" if complete is False else None)

    def test_activate_population_restricts_exactly_and_writes_nothing_next_to_the_run(self):
        win = self._window()
        self.assertTrue(win.activate_population(self._population()))
        win.sidebar.select_leaf("Analyses", "Lagr. radii, PartType4")
        win.plot_panel._toolbar._switch_view("Matrix")
        self.assertTrue(_wait(self.app, win))
        _, cells = _table(win)
        p = SnapshotView(str(self.snap)).particles([4], velocities=False, ids=True)
        p = p.subset(np.isin(p.ids.astype(np.uint64), self.ids))
        ref = lagrangian_radii(p, find_centre(p, "shrinking_sphere"))
        self.assertEqual([c[1] for c in cells], _shown(ref["radius"].value))
        self.assertEqual(sorted(x.name for x in self.run_dir.iterdir()), ["snapshot_000.hdf5"])   # nothing written here
        sel = win._global_context["selection_set"]
        self.assertEqual((sel["kind"], sel["complete"], sel["n_expected"]), ("group", True, len(self.ids)))
        self.assertTrue(sel["path"].startswith(str(selection_sets.FALLBACK_ROOT / "_live")))
        notes = win.plot_panel._export_columns[0]
        self.assertTrue(any("is: FoF group 1 (b = 0.2, N_min = 32, stars) at snapshot_000.hdf5 · complete" in n
                            for n in notes), notes)
        self.assertFalse(any(n.startswith("WARNING") for n in notes))

    def test_an_incomplete_population_warns_in_the_status_and_the_notes(self):
        win = self._window()
        self.assertTrue(win.activate_population(self._population(complete=False, n_expected=6000)))
        self.assertIn("display sample", win.plot_panel.lbl_status.text())
        win.sidebar.select_leaf("Analyses", "Lagr. radii, PartType4")
        win.plot_panel._toolbar._switch_view("Matrix")
        self.assertTrue(_wait(self.app, win))
        notes = win.plot_panel._export_columns[0]
        self.assertTrue(any("WARNING: This set was taken from the display sample" in n and "1,500 of its 6,000" in n
                            for n in notes), notes)

    def test_none_means_every_particle(self):
        win = self._window()
        win.activate_population(self._population())
        self.assertIsNotNone(win._global_context["selection_set"])
        win.activate_population(None)
        self.assertIsNone(win._global_context["selection_set"])


if __name__ == "__main__":
    unittest.main()
