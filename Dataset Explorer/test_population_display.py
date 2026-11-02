"""
The Explorer says what each set is, and when it is incomplete.

    python3 -m pytest -q test_population_display.py

- The Sets group shows each set's kind ("Halo 1 (100) · group") and marks one
  known to be incomplete with the reason in short ("⚠ sample only",
  "⚠ members missing", "⚠ incomplete"); the tooltip has the full description,
  the warning, where it was made and its file. The leaf key (name and count) is
  unchanged, so choosing a set works as before.
- An analysis made on an incomplete set ends its status line with
  "⚠ incomplete set"; why it is incomplete reaches the analysis context.
- A field view (histogram, line or matrix of a dataset) drawn while a set is
  active says that it shows the whole snapshot, not the set.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import selection_sets  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402
from test_analyses_window import _wait  # noqa: E402

FOF_DEF = {'finder': 'fof', 'group': 1, 'b': 0.2, 'min_size': 32, 'link_types': [4]}
CAT_DEF = {'finder': 'catalogue', 'group': 30, 'subhalo': False, 'catalogue_file': 'fof_tab_000.hdf5'}
REGION = {'shape': 'sphere', 'centre': [2.0, 2.0, 2.0], 'radius': 0.5, 'units': 'kpc', 'periodic': False,
          'types': [4], 'scope': 'sample'}
AT = {'file': 'snapshot_000.hdf5', 'index': 0}
NOTE = "Field views show the whole snapshot, not set 'Halo 1'; the active set applies to Analyses only."


class TestPopulationDisplay(unittest.TestCase):
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
        cls.snap = Path(cls.tmp.name) / "snapshot_000.hdf5"
        write_gadget4_snapshot(cls.snap, {4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / (1.989e43 / MSUN_G),
                                                  ids=np.arange(3000))})
        cls.ids = np.arange(0, 3000, 2).astype(np.uint64)

    @classmethod
    def tearDownClass(cls):
        selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR = cls._old
        cls.tmp.cleanup()

    def _run(self, name):
        """A run folder of its own (sets are listed per run folder)."""
        run = Path(self.tmp.name) / name
        run.mkdir()
        shutil.copy(self.snap, run / "snapshot_000.hdf5")
        return run / "snapshot_000.hdf5"

    def _window(self, snap):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        self.assertTrue(win.open_path(str(snap)))
        return win

    def _population(self, name="Halo 1", **meaning):
        meaning = {'kind': 'group', 'definition': FOF_DEF, 'defined_at': AT, 'complete': True,
                   'n_expected': len(self.ids), **meaning}
        return selection_sets.SelectionSet(self.ids, name, "#ffaa00", made_in="test", **meaning)

    @staticmethod
    def _sets_items(win):
        """{leaf key: (text shown, tooltip)} of the Sets group."""
        tree = win.sidebar.tree
        for i in range(tree.topLevelItemCount()):
            g = tree.topLevelItem(i)
            if g.text(0) == "Sets":
                return {g.child(j).data(0, Qt.ItemDataRole.UserRole)[1]: (g.child(j).text(0), g.child(j).toolTip(0))
                        for j in range(g.childCount())}
        return {}

    def test_sets_group_shows_each_sets_kind_and_incompleteness(self):
        snap = self._run("labels")
        ids = np.arange(100, dtype=np.uint64)
        write = selection_sets.write_set
        write(str(snap), ids, "Halo 1", kind="group", definition=FOF_DEF, defined_at=AT, complete=True, n_expected=100)
        write(str(snap), ids[:50], "Sphere", kind="region", definition=REGION, defined_at=AT, complete=False,
              incomplete_reason="display_sample")
        write(str(snap), ids[:30], "Cut halo", kind="group", definition=CAT_DEF, defined_at=AT, complete=False,
              n_expected=60, incomplete_reason="missing_in_file")
        write(str(snap), ids[:20], "Unsure", kind="query", definition={'criteria': [{'kind': 'sanity', 'check': 'x'}]},
              defined_at=AT, complete=False)
        with h5py.File(snap.parent / selection_sets.SETS_FOLDER / "old.hdf5", "w") as f:     # version 1
            f["ParticleIDs"] = np.array([1, 2, 3], dtype=np.uint64)
            f.attrs.update(dict(format=selection_sets.FORMAT, version=1, name="old", n=3))
        win = self._window(snap)
        items = self._sets_items(win)
        shown = {k: text.lstrip("● ") for k, (text, _) in items.items()}
        self.assertEqual(shown["Halo 1 (100)"], "Halo 1 (100) · group")
        self.assertEqual(shown["Sphere (50)"], "Sphere (50) · region · ⚠ sample only")
        self.assertEqual(shown["Cut halo (30)"], "Cut halo (30) · group · ⚠ members missing")
        self.assertEqual(shown["Unsure (20)"], "Unsure (20) · query · ⚠ incomplete")
        self.assertEqual(shown["old (3)"], "old (3) · ids")

        cut = selection_sets.read_set(str(snap.parent / selection_sets.SETS_FOLDER / "Cut_halo.hdf5"))
        tip = items["Cut halo (30)"][1]
        self.assertTrue(tip.startswith(cut.describe() + "\n⚠ The snapshot files hold only 30 of this group's 60"), tip)
        self.assertIn(str(snap.parent / selection_sets.SETS_FOLDER / "Cut_halo.hdf5"), tip)
        self.assertNotIn("⚠", items["Halo 1 (100)"][1])

        # the shown text is not the key: choosing a set works as before, and it becomes the marked one
        self.assertTrue(win.sidebar.select_leaf("Sets", "Sphere (50)"))
        sel = win._global_context["selection_set"]
        self.assertEqual((sel["kind"], sel["complete"], sel["incomplete_reason"]), ("region", False, "display_sample"))
        self.assertIn("⚠ This set was taken from the display sample", win.plot_panel.lbl_status.text())
        self.assertTrue(self._sets_items(win)["Sphere (50)"][0].startswith("● "))

    def test_an_analysis_on_an_incomplete_set_says_so_in_its_status_line(self):
        win = self._window(self.snap)
        self.assertTrue(win.activate_population(self._population(
            definition=CAT_DEF, complete=False, n_expected=6000, incomplete_reason="missing_in_file")))
        sel = win._global_context["selection_set"]
        self.assertEqual((sel["complete"], sel["incomplete_reason"]), (False, "missing_in_file"))
        for leaf in ("Lagr. radii, PartType4", "Profile, PartType4"):          # analysis_plugins and _m2
            self.assertTrue(win.sidebar.select_leaf("Analyses", leaf))
            self.assertTrue(_wait(self.app, win))
            self.assertIn("set 'Halo 1' · ⚠ incomplete set", win.plot_panel.lbl_status.text(), leaf)
        win.plot_panel._toolbar._switch_view("Matrix")
        self.assertTrue(_wait(self.app, win))
        notes = win.plot_panel._export_columns[0]
        self.assertTrue(any("WARNING: The snapshot files hold only 1,500 of this group's 6,000 members" in n
                            for n in notes), notes)
        # a complete set carries no such mark
        self.assertTrue(win.activate_population(self._population(name="Halo 2")))
        self.assertTrue(_wait(self.app, win))
        status = win.plot_panel.lbl_status.text()
        self.assertIn("set 'Halo 2'", status)
        self.assertNotIn("incomplete set", status)

    def test_a_field_view_says_it_shows_the_whole_snapshot_not_the_set(self):
        win = self._window(self.snap)
        self.assertTrue(win.sidebar.select_leaf("PartType4", "Masses"))
        self.assertTrue(_wait(self.app, win))
        self.assertNotIn("Field views show", win.plot_panel.lbl_status.text())
        self.assertTrue(win.activate_population(self._population()))     # made active while a field is open
        self.assertIn(NOTE, win.plot_panel.lbl_status.text())
        for tab in ("Line", "Histogram", "Matrix"):
            win.plot_panel._toolbar._switch_view(tab)
            self.assertTrue(_wait(self.app, win))
            status = win.plot_panel.lbl_status.text()
            self.assertTrue(status.endswith(NOTE), (tab, status))
            if tab == "Histogram":                                         # it does count every particle
                self.assertIn("Particles: 3,000", win.plot_panel.lbl_info.text())
        # an analysis uses the set and needs no such note
        self.assertTrue(win.sidebar.select_leaf("Analyses", "Lagr. radii, PartType4"))
        self.assertTrue(_wait(self.app, win))
        status = win.plot_panel.lbl_status.text()
        self.assertIn("set 'Halo 1'", status)
        self.assertNotIn("Field views show", status)
        # with every particle chosen again, a field view has nothing to explain
        self.assertTrue(win.sidebar.select_leaf("Sets", "All particles"))
        self.assertTrue(win.sidebar.select_leaf("PartType4", "Masses"))
        self.assertTrue(_wait(self.app, win))
        self.assertNotIn("Field views show", win.plot_panel.lbl_status.text())


if __name__ == "__main__":
    unittest.main()
