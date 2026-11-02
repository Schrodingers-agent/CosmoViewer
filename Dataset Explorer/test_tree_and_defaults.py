"""
Presentation defaults of the Explorer:

  * the field tree lists only what the open file can supply; Options >
    "Show fields this file does not have" lists the rest, greyed, and keeps
    the Sets group as it was;
  * the array comparison (formerly the toolbar's "Load Matrix") is reached
    from Options > "Compare arrays from files (advanced)…", not the toolbar,
    and its banner and ✕ still work;
  * positions (Coordinates, GroupPos, ...) are histogrammed on a linear
    axis, and generic lengths are offered in kpc first.

    QT_QPA_PLATFORM=offscreen python3 -m pytest -q test_tree_and_defaults.py
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

import selection_sets  # noqa: E402
from generic_plugin import configure_units, make_generic_leaves  # noqa: E402

DM_CUT = HERE.parent / "_dev_tools" / "inspector_check" / "dm_cut" / "snapshot_002.hdf5"


def _leaves(win, skip=("Sets",)):
    """(group, leaf, enabled) for every leaf in the tree, groups in ``skip`` left out."""
    out = []
    tree = win.sidebar.tree
    for i in range(tree.topLevelItemCount()):
        g = tree.topLevelItem(i)
        if g.text(0) in skip:
            continue
        for j in range(g.childCount()):
            it = g.child(j)
            out.append((g.text(0), it.text(0), bool(it.flags() & Qt.ItemFlag.ItemIsEnabled)))
    return out


def _options(win):
    return next(a.menu() for a in win.menuBar().actions() if a.text() == "Options")


def _groups(win):
    tree = win.sidebar.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


class _Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = tempfile.TemporaryDirectory()
        cls._old = (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR)
        selection_sets.FALLBACK_ROOT = Path(cls.tmp.name) / "home" / "sets"
        selection_sets.INSTANCES_DIR = Path(cls.tmp.name) / "home" / "explorer_instances"

    @classmethod
    def tearDownClass(cls):
        selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR = cls._old
        cls.tmp.cleanup()

    def _window(self):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        return win


class TestTreeShowsWhatTheFileHas(_Window):
    @classmethod
    def setUpClass(cls):
        assert DM_CUT.exists(), "tracked test fixture _dev_tools/inspector_check/dm_cut is missing: restore it with git checkout -- _dev_tools/inspector_check/dm_cut"
        super().setUpClass()
        # A copy, so the set file below is written in the temporary folder, not the repository.
        cls.snap = Path(cls.tmp.name) / "run" / "snapshot_002.hdf5"
        cls.snap.parent.mkdir()
        shutil.copy(DM_CUT, cls.snap)

    def test_dark_matter_only_snapshot_lists_only_its_fields(self):
        win = self._window()
        self.assertTrue(win.open_path(str(self.snap)))
        shown = _leaves(win)
        self.assertEqual(len(shown), 14)                                  # P8: 14 usable leaves
        self.assertTrue(all(enabled for _, _, enabled in shown))          # nothing greyed by default
        groups = _groups(win)
        self.assertIn("PartType1", groups)
        for g in groups:                                                  # no empty group headers
            item = win.sidebar.tree.topLevelItem(groups.index(g))
            self.assertGreater(item.childCount(), 0, g)
        self.assertNotIn(("Analyses", "Lagr. radii, PartType0", True), shown)
        self.assertFalse(any(g == "Analyses" and "PartType0" in leaf for g, leaf, _ in shown))

        act = win._show_absent_action
        self.assertEqual(act.text(), "Show fields this file does not have")
        self.assertTrue(act.isCheckable() and not act.isChecked())
        self.assertIn(act, _options(win).actions())

        act.setChecked(True)
        all_leaves = _leaves(win)
        greyed = [x for x in all_leaves if not x[2]]
        self.assertEqual(len(greyed), 37)                                 # P8: 37 greyed leaves
        self.assertEqual([x for x in all_leaves if x[2]], shown)          # the usable ones unchanged
        self.assertIn(("Analyses", "Lagr. radii, PartType0", False), all_leaves)
        self.assertEqual([x for x in win.sidebar.available_leaves() if not x.startswith("Sets/")],
                         [f"{g}/{leaf}" for g, leaf, _ in shown])        # greyed ones cannot be chosen

        act.setChecked(False)
        self.assertEqual(_leaves(win), shown)

    def test_toggle_keeps_the_sets_group_and_the_current_leaf(self):
        p = selection_sets.write_set(str(self.snap), np.arange(1, 101, dtype=np.uint64), "first hundred",
                                     made_in="test")
        self.addCleanup(os.remove, p)
        win = self._window()
        self.assertTrue(win.open_path(str(self.snap)))
        self.assertEqual(_groups(win)[-1], "Sets")
        sets_before = [win.sidebar.tree.topLevelItem(len(_groups(win)) - 1).child(j).text(0)
                       for j in range(win.sidebar.tree.topLevelItem(len(_groups(win)) - 1).childCount())]
        self.assertTrue(any("first hundred" in s for s in sets_before))
        leaf = next(f"{g}/{x}" for g, x, _ in _leaves(win) if g == "PartType1")
        group, name = leaf.split("/", 1)
        tree = win.sidebar.tree
        target = next(tree.topLevelItem(i).child(j) for i in range(tree.topLevelItemCount())
                      for j in range(tree.topLevelItem(i).childCount())
                      if tree.topLevelItem(i).child(j).data(0, Qt.ItemDataRole.UserRole) == (group, name))
        tree.setCurrentItem(target)

        for on in (True, False):
            win._show_absent_action.setChecked(on)
            self.assertEqual(_groups(win)[-1], "Sets")
            last = tree.topLevelItem(tree.topLevelItemCount() - 1)
            self.assertEqual([last.child(j).text(0) for j in range(last.childCount())], sets_before)
            self.assertEqual(tree.currentItem().data(0, Qt.ItemDataRole.UserRole), (group, name))


class TestNoFileOpen(_Window):
    def test_empty_until_a_file_is_open_unless_asked(self):
        win = self._window()
        self.assertEqual(_leaves(win), [])
        win._show_absent_action.setChecked(True)
        listed = _leaves(win)
        self.assertGreater(len(listed), 0)
        self.assertFalse(any(enabled for _, _, enabled in listed))


class TestCompareArraysFromTheMenu(_Window):
    def test_menu_opens_the_comparison_and_the_toolbar_clears_it(self):
        import plot_panel
        win = self._window()
        bar = win.plot_panel._toolbar
        # The toolbar no longer offers it while idle.
        self.assertFalse(bar._load_container.isVisibleTo(win))
        self.assertFalse(any("Load Matrix" in b.text() for b in bar.findChildren(QPushButton)))

        act = win._compare_arrays_action
        self.assertEqual(act.text(), "Compare arrays from files (advanced)…")
        # Deliberately cut from the Options menu; listed only with
        # COSMOVIEWER_ADVANCED=1 (TestCompareArraysOnlyWhenAdvanced). The action still works.
        self.assertNotIn(act, _options(win).actions())

        arr = Path(self.tmp.name) / "phase_space.npy"
        np.save(arr, np.arange(12, dtype=float).reshape(3, 4))
        old = plot_panel.QFileDialog.getOpenFileNames
        plot_panel.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(arr)], ""))
        try:
            act.trigger()
        finally:
            plot_panel.QFileDialog.getOpenFileNames = old
        self.app.processEvents()
        self.assertTrue(win._diag_override)
        self.assertEqual(len(win._diag_files), 1)
        self.assertTrue(win.plot_panel._diag_banner.isVisibleTo(win))
        self.assertIn("phase_space.npy", win.plot_panel._lbl_diag_file.text())
        self.assertTrue(bar._load_container.isVisibleTo(win))             # name and ✕ while active
        self.assertTrue(bar._btn_load_clear.isVisibleTo(win))
        self.assertIn("phase_space", bar._btn_load_main.text())
        self.assertFalse(win.sidebar.isEnabled())

        bar._btn_load_clear.click()
        self.app.processEvents()
        self.assertFalse(win._diag_override)
        self.assertFalse(win.plot_panel._diag_banner.isVisibleTo(win))
        self.assertFalse(bar._load_container.isVisibleTo(win))
        self.assertTrue(win.sidebar.isEnabled())


class TestCompareArraysOnlyWhenAdvanced(_Window):
    def test_listed_in_options_only_with_cosmoviewer_advanced(self):
        old = os.environ.get("COSMOVIEWER_ADVANCED")
        os.environ["COSMOVIEWER_ADVANCED"] = "1"
        try:
            win = self._window()
        finally:
            if old is None:
                os.environ.pop("COSMOVIEWER_ADVANCED", None)
            else:
                os.environ["COSMOVIEWER_ADVANCED"] = old
        self.assertIn(win._compare_arrays_action, _options(win).actions())


class TestPositionDefaults(unittest.TestCase):
    GADGET4_COORDS = {"to_cgs": 3.085678e24, "length_scaling": 1.0, "mass_scaling": 0.0,
                      "velocity_scaling": 0.0, "a_scaling": 1.0, "h_scaling": -1.0}

    def test_positions_are_linear_and_in_kpc(self):
        from astropy import units as u
        for group, dataset in (("PartType1", "Coordinates"), ("Group", "GroupPos"),
                               ("Subhalo", "SubhaloPos"), ("Subhalo", "SubhaloCM"),
                               ("Subhalo", "CentreOfMass"), ("Subhalo", "CenterOfMass")):
            cls = make_generic_leaves(group, dataset, (500, 3), "float32")[f"{dataset} [x]"]
            self.assertEqual((cls.DEFAULT_SCALE_X, cls.DEFAULT_SCALE_Y), ("linear", "linear"), dataset)
            self.assertTrue(cls.HISTOGRAM_LINEAR_X, dataset)
            configure_units(cls, self.GADGET4_COORDS)
            self.assertEqual(cls.ALLOWED_UNITS[0], u.kpc, dataset)          # default in the CGS system
            self.assertEqual([str(x) for x in cls.ALLOWED_UNITS], ["kpc", "cm", "Mpc"])

        # Positive positions (a box from 0 to 50): linear bins, every particle counted, in kpc
        cls = make_generic_leaves("PartType1", "Coordinates", (1000, 3), "float64")["Coordinates [x]"]
        configure_units(cls, self.GADGET4_COORDS)
        pos = np.random.default_rng(5).uniform(0.0, 50.0, (1000, 3))
        key = "PartType1/Coordinates"
        req = cls().compute({key: pos}, {key: self.GADGET4_COORDS},
                            {"Time": 1.0, "Redshift": 0.0, "HubbleParam": 0.7},
                            {"n_global_map": {key: 1000}, "HubbleParam": 0.7},
                            "Histogram", "linear", "linear", (0, 1000))
        self.assertFalse(req.log_x)
        self.assertEqual(int(np.nan_to_num(req.curves[0].y).sum()), 1000)
        self.assertEqual(req.unit_label, "kpc")
        edges = np.asarray(req.bin_edges)
        self.assertTrue(np.allclose(np.diff(edges), np.diff(edges)[0]))     # equal widths: linear bins

    def test_other_fields_keep_their_defaults(self):
        from astropy import units as u
        vel = make_generic_leaves("PartType1", "Velocities", (10, 3), "float32")["Velocities |·|"]
        self.assertEqual(vel.DEFAULT_SCALE_X, "log")
        self.assertFalse(vel.HISTOGRAM_LINEAR_X)
        radius = make_generic_leaves("Group", "Group_R_Crit200", (10,), "float32")["Group_R_Crit200"]
        self.assertEqual(radius.DEFAULT_SCALE_X, "log")                    # radii span decades: log
        configure_units(radius, {"to_cgs": 3.085678e24, "length_scaling": 1.0, "mass_scaling": 0.0,
                                 "velocity_scaling": 0.0})
        self.assertEqual(radius.ALLOWED_UNITS[0], u.kpc)                   # but lengths in kpc
        dens = make_generic_leaves("PartType0", "Density", (10,), "float32")["Density"]
        configure_units(dens, {"to_cgs": 6.77e-22, "length_scaling": -3.0, "mass_scaling": 1.0,
                               "velocity_scaling": 0.0})
        self.assertEqual(dens.ALLOWED_UNITS[0], u.g / u.cm ** 3)            # densities unchanged
        self.assertEqual(dens.DEFAULT_SCALE_X, "log")


if __name__ == "__main__":
    unittest.main()
