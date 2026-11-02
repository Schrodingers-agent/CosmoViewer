"""
Selection sets shared with Immersive Mode (selection_sets.py) and the Explorer's Sets group.

    python3 -m pytest -q test_selection_sets.py

Checks the file format, where files go (next to the snapshots, or under
~/.cosmoviewer when that folder is read-only), that the Analyses restricted to
a set equal the headless measurement on exactly those particles, that a set
sent while the window is open is picked up, and that --set works.
Uses COSMOVIEWER_DM_OUTPUT for the real-data case when set.
"""

import filecmp
import os
import stat
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
import selection_sets  # noqa: E402
from measurements import SnapshotView, find_centre, lagrangian_radii  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402
from test_analyses_window import _shown, _table, _wait  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")
IMMERSIVE_COPY = HERE.parent / "Cosmoviewer Immersive" / "selection_sets.py"


class TestFormat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self._old = (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR)
        selection_sets.FALLBACK_ROOT = self.home / "sets"
        selection_sets.INSTANCES_DIR = self.home / "explorer_instances"
        self.addCleanup(self._restore)
        self.run_dir = Path(self.tmp.name) / "output"
        self.run_dir.mkdir()
        self.snap = self.run_dir / "snapshot_002.hdf5"
        self.snap.write_bytes(b"")

    def _restore(self):
        selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR = self._old

    def test_round_trip_next_to_the_snapshots(self):
        ids = np.array([5, 3, 3, 2 ** 40, 7], dtype=np.uint64)
        p = selection_sets.write_set(str(self.snap), ids, "halo 0 / z=0", "#ff8000",
                                     made_in="test", description="by hand")
        self.assertEqual(Path(p).resolve().parent, (self.run_dir / "cosmoviewer_sets").resolve())
        st = selection_sets.read_set(p)
        np.testing.assert_array_equal(st.ids, np.array([3, 5, 7, 2 ** 40], dtype=np.uint64))
        self.assertEqual((st.name, st.colour, st.made_in, st.source_snapshot, st.description),
                         ("halo 0 / z=0", "#ff8000", "test", "snapshot_002.hdf5", "by hand"))
        self.assertEqual(selection_sets.list_sets(str(self.snap)), [p])
        # nothing else in the output folder changed
        self.assertEqual(sorted(x.name for x in self.run_dir.iterdir()), ["cosmoviewer_sets", "snapshot_002.hdf5"])

    def test_same_name_replaces_and_list_is_newest_first(self):
        a = selection_sets.write_set(str(self.snap), [1], "A")
        time.sleep(0.02)
        b = selection_sets.write_set(str(self.snap), [2], "B")
        self.assertEqual(selection_sets.list_sets(str(self.snap)), [b, a])
        a2 = selection_sets.write_set(str(self.snap), [9, 10], "A")
        self.assertEqual(a2, a)
        self.assertEqual(selection_sets.read_set(a).n, 2)

    @unittest.skipIf(os.geteuid() == 0 if hasattr(os, "geteuid") else False, "root ignores permissions")
    def test_read_only_output_uses_the_home_folder(self):
        os.chmod(self.run_dir, stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(os.chmod, self.run_dir, stat.S_IRWXU)
        try:                        # Windows ignores the read-only bit on folders
            (self.run_dir / "probe").touch()
        except OSError:
            pass
        else:
            (self.run_dir / "probe").unlink()
            self.skipTest("this system lets us write to a read-only folder")
        p = selection_sets.write_set(str(self.snap), [1, 2], "ro")
        self.assertTrue(str(p).startswith(str(selection_sets.FALLBACK_ROOT)))
        self.assertEqual(selection_sets.list_sets(str(self.snap)), [p])

    def test_fallback_when_the_first_folder_cannot_be_made(self):
        (self.run_dir / "cosmoviewer_sets").write_text("a file, not a folder")
        p = selection_sets.write_set(str(self.snap), [1, 2], "x")
        self.assertTrue(str(p).startswith(str(selection_sets.FALLBACK_ROOT)))
        self.assertEqual(selection_sets.list_sets(str(self.snap)), [p])

    def test_other_hdf5_files_are_not_sets(self):
        d = self.run_dir / "cosmoviewer_sets"
        d.mkdir()
        with h5py.File(d / "x.hdf5", "w") as f:
            f["ParticleIDs"] = np.arange(3)
        self.assertEqual(selection_sets.list_sets(str(self.snap)), [])
        with self.assertRaises(ValueError):
            selection_sets.read_set(str(d / "x.hdf5"))

    def test_explorer_instances(self):
        self.assertIsNone(selection_sets.explorer_for(str(self.snap)))
        selection_sets.register_instance(os.getpid(), [str(self.snap)])
        self.assertEqual(selection_sets.explorer_for(str(self.snap)), os.getpid())
        other = Path(self.tmp.name) / "elsewhere" / "snap.hdf5"
        self.assertIsNone(selection_sets.explorer_for(str(other)))
        selection_sets.register_instance(2 ** 22 + 12345, [str(self.snap)])   # not a live process
        selection_sets.unregister_instance(os.getpid())
        self.assertIsNone(selection_sets.explorer_for(str(self.snap)))
        self.assertEqual(list(selection_sets.INSTANCES_DIR.glob("*.json")), [])   # stale entry removed

    @unittest.skipUnless(IMMERSIVE_COPY.exists(), "Immersive Mode not next to the Explorer")
    def test_both_apps_have_the_same_module(self):
        self.assertTrue(filecmp.cmp(HERE / "selection_sets.py", IMMERSIVE_COPY, shallow=False))


class TestSetsInWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = tempfile.TemporaryDirectory()
        cls._old = (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR)
        selection_sets.FALLBACK_ROOT = Path(cls.tmp.name) / "home" / "sets"
        selection_sets.INSTANCES_DIR = Path(cls.tmp.name) / "home" / "explorer_instances"
        p4, v4, m4 = plummer(4000, 1e5, 0.001, seed=4)
        p1, v1, _ = plummer(6000, 1e6, 0.003, seed=5)
        code_mass = 1.989e43 / MSUN_G
        cls.snap = Path(cls.tmp.name) / "snapshot_000.hdf5"
        write_gadget4_snapshot(cls.snap, {
            4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / code_mass, ids=np.arange(4000)),
            1: dict(pos=p1 + 2.0, vel=v1, mass=None, ids=np.arange(4000, 10000))},
            mass_table=[0, 1e6 / 6000 / code_mass, 0, 0, 0, 0])
        # set: every third star plus some DM (DM is ignored by the PartType4 analysis)
        cls.ids = np.concatenate([np.arange(0, 4000, 3), np.arange(4000, 4500)]).astype(np.uint64)
        cls.set_path = selection_sets.write_set(str(cls.snap), cls.ids, "thirds", "#00ff00", made_in="test")

    @classmethod
    def tearDownClass(cls):
        selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR = cls._old
        cls.tmp.cleanup()

    def _window(self):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        return win

    def _headless(self, types, ids):
        p = SnapshotView(str(self.snap)).particles(types, velocities=False, ids=True)
        p = p.subset(np.isin(p.ids.astype(np.uint64), ids))
        return lagrangian_radii(p, find_centre(p, "shrinking_sphere"))

    def test_sets_group_lists_the_file(self):
        win = self._window()
        self.assertTrue(win.open_path(str(self.snap)))
        leaves = win.sidebar.available_leaves()
        self.assertIn("Sets/All particles", leaves)
        self.assertIn("Sets/thirds (1,834)", leaves)
        self.assertIn(str(os.getpid()) + ".json", [p.name for p in selection_sets.INSTANCES_DIR.glob("*.json")])

    def test_restricted_analysis_equals_headless_on_the_set(self):
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap)]),
                                   "Analyses/Lagr. radii, PartType4", "Matrix", self.set_path)
        self.assertTrue(_wait(self.app, win))
        _, cells = _table(win)
        ref = self._headless([4], self.ids)
        self.assertEqual(len(np.arange(0, 4000, 3)), ref.provenance.n_used)
        self.assertEqual([c[1] for c in cells], _shown(ref["radius"].value))
        status = win.plot_panel.lbl_status.text()
        self.assertIn("All 1,334 particles", status)
        self.assertIn("set 'thirds'", status)
        notes = win.plot_panel._export_columns[0]
        self.assertTrue(any("restricted to set 'thirds': 1,334 of its 1,834 IDs" in n for n in notes))
        # back to every particle
        win.sidebar.select_leaf("Sets", "All particles")
        self.assertTrue(_wait(self.app, win))
        _, cells = _table(win)
        p = SnapshotView(str(self.snap)).particles([4], velocities=False, ids=False)
        full = lagrangian_radii(p, find_centre(p, "shrinking_sphere"))
        self.assertEqual([c[1] for c in cells], _shown(full["radius"].value))

    def test_cli_set_alone_opens_an_analysis(self):
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap)]), "", "", self.set_path)
        self.assertTrue(_wait(self.app, win))
        self.assertIn("set 'thirds'", win.plot_panel.lbl_status.text())

    def test_a_set_sent_while_open_becomes_active(self):
        win = self._window()
        self.assertTrue(win.open_path(str(self.snap)))
        win.sidebar.select_leaf("Analyses", "Lagr. radii, PartType4")
        self.assertTrue(_wait(self.app, win))
        ids = np.arange(0, 2000, dtype=np.uint64)
        selection_sets.write_set(str(self.snap), ids, "inner half by ID", made_in="test")
        end = time.time() + 20
        while time.time() < end and (win._global_context.get("selection_set") or {}).get("name") != "inner half by ID":
            self.app.processEvents()
            time.sleep(0.05)
        self.assertEqual(win._global_context["selection_set"]["name"], "inner half by ID")
        self.assertTrue(_wait(self.app, win))
        self.assertIn("set 'inner half by ID'", win.plot_panel.lbl_status.text())
        self.assertIn("Sets/inner half by ID (2,000)", win.sidebar.available_leaves())

    def test_set_of_other_types_only_says_so(self):
        p = selection_sets.write_set(str(self.snap), np.arange(5000, 5010), "dm only", made_in="test")
        self.addCleanup(os.remove, p)
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap)]),
                                   "Analyses/Lagr. radii, PartType4", "Profile", p)
        self.assertTrue(_wait(self.app, win))
        self.assertIn("None of the 10 particles of set 'dm only'", win.plot_panel.lbl_status.text())

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_real_dm_halo_set_matches_headless(self):
        path = Path(DM) / "snapshot_002.hdf5"
        cat = Path(DM) / "fof_tab_002.hdf5"
        with h5py.File(cat, "r") as f:
            off, ln = int(f["Group/GroupOffsetType"][0, 1]), int(f["Group/GroupLenType"][0, 1])
        with h5py.File(path, "r") as f:
            ids = f["PartType1/ParticleIDs"][off:off + ln].astype(np.uint64)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sp = selection_sets.write_set(str(path), ids, "group 1", path=str(Path(tmp.name) / "g1.hdf5"))
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(path)]),
                                   "Analyses/Lagr. radii, all types", "Matrix", sp)
        self.assertTrue(_wait(self.app, win))
        _, cells = _table(win)
        p = SnapshotView(str(path)).particles([1], velocities=False, ids=True)
        p = p.subset(np.isin(p.ids.astype(np.uint64), ids))
        ref = lagrangian_radii(p, find_centre(p, "shrinking_sphere"))
        self.assertEqual(ref.provenance.n_used, ln)
        self.assertEqual([c[1] for c in cells], _shown(ref["radius"].value))
        self.assertIn(f"All {ln:,} particles", win.plot_panel.lbl_status.text())


if __name__ == "__main__":
    unittest.main()
