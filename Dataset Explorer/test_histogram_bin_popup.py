"""
Clicking a histogram bin: the popup that lists the particles in that bin.

    QT_QPA_PLATFORM=offscreen python3 -m pytest -q test_histogram_bin_popup.py

Reported 2026-10-27: the popup was a white card that said "⏳ Index building…"
for ever. Causes found:
  0. The hang itself: the index was built and written, but the signal saying
     so was connected from inside the compute thread, which had already ended,
     so it was never delivered. Only a tab or scale change picked it up.
  1. The popup was a static label. Clicked while the index was being built, it
     said "building" and never changed, even after the build had finished.
  2. A build that failed printed a traceback on the terminal only; the popup
     said "building" on every later click.
  3. The index was started only by a fresh read of the dataset; a histogram
     drawn from cache (tab, scale, unit or bin-count change) started none, and
     after a bin-count change the old index answered for the new bins.
  4. The click was always read as on a log axis; on a linear axis (positions,
     velocity components) it picked the wrong bin.
  5. With a unit made by a non-linear transform (temperature from internal
     energy, with mu per particle) the index, built on the stored values,
     returned particles of other bins.
Each test below failed before the fix. Counts are checked against an
independent np.histogram of the values read straight from the file.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import density_explorer  # noqa: E402
import lod_index  # noqa: E402
import selection_sets  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402

N = 6000


class _Click:
    """Stand-in for pyqtgraph's mouse-click event at a scene point."""
    def __init__(self, scene_pos):
        self._p = scene_pos

    def button(self):
        return Qt.MouseButton.LeftButton

    def scenePos(self):
        return self._p


class HistogramBinPopupTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])
        cls.tmp = Path(tempfile.mkdtemp())
        cls._saved = (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR,
                      lod_index.LodIndex.CACHE_DIR)
        selection_sets.FALLBACK_ROOT = cls.tmp / "sets"
        selection_sets.INSTANCES_DIR = cls.tmp / "instances"
        pos, vel, _ = plummer(N, 1e6, 0.003, seed=3)
        rng = np.random.default_rng(7)
        cls.snap = cls.tmp / "output" / "snapshot_000.hdf5"
        cls.snap.parent.mkdir(parents=True)
        write_gadget4_snapshot(cls.snap, {0: dict(
            pos=pos + 2.0, vel=vel, mass=np.full(N, 1e-4), ids=np.arange(N),
            fields={"Density": rng.lognormal(0.0, 2.0, N),
                    "InternalEnergy": rng.lognormal(3.0, 1.0, N),
                    "ElectronAbundance": rng.uniform(0.0, 1.2, N)})})

    @classmethod
    def tearDownClass(cls):
        (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR,
         lod_index.LodIndex.CACHE_DIR) = cls._saved

    def setUp(self):
        # A fresh index folder per test, so no test is answered from another's files.
        lod_index.LodIndex.CACHE_DIR = tempfile.mkdtemp(dir=self.tmp)
        self.win = density_explorer.CosmoExplorer()
        self.win.resize(1200, 800)
        self.win.show()
        self.assertTrue(self.win.open_path(str(self.snap)))
        self.pp = self.win.plot_panel

    def tearDown(self):
        w = self.win._lod_worker
        if w is not None:
            w.cancel()
            w.wait(10000)
        self.win.close()
        self.pump(0.05)

    # ── helpers ───────────────────────────────────────────────────────────

    def pump(self, seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.01)
        return until is None or until()

    def show(self, leaf):
        self.assertTrue(self.win.sidebar.select_leaf("PartType0", leaf))
        self.assertTrue(self.pump(60, lambda: self.win._active_worker is None
                                  and self.pp._bin_edges_lod is not None))
        self.assertEqual(self.pp.current_tab(), "Histogram")

    def click_value(self, value):
        """Click the plot at data value `value` on the x axis."""
        vb = self.pp.pw.getViewBox()
        log_x = vb.state["logMode"][0]
        x = np.log10(value) if log_x else value
        y = vb.viewRange()[1]
        scene = vb.mapViewToScene(QPointF(x, 0.5 * (y[0] + y[1])))
        self.pp._on_inspector_click(_Click(scene))

    def tip(self):
        return self.pp._insp_tip.text() if self.pp._insp_tip.isVisible() else ""

    def file_values(self, dataset, column=None):
        with h5py.File(self.snap, "r") as f:
            ds = f["PartType0"][dataset]
            v = ds[:].astype(float)
            to_cgs = float(ds.attrs.get("to_cgs", 1.0))
        return (v if column is None else v[:, column]) * to_cgs

    def wait_ready(self):
        self.assertTrue(self.pump(60, lambda: self.pp._lod_state == "ready"),
                        f"index state {self.pp._lod_state}: {self.pp._lod_msg}")

    def middle_bin(self):
        edges = np.asarray(self.pp._bin_edges_lod, float)
        b = (len(edges) - 1) // 2
        return b, edges, np.sqrt(edges[b] * edges[b + 1]) if edges[b] > 0 else 0.5 * (edges[b] + edges[b + 1])

    # ── tests ─────────────────────────────────────────────────────────────

    def test_finished_index_reaches_the_panel(self):
        """Cause 0, the hang itself: the build finished and wrote its files, but its
        "ready" signal was connected inside the compute thread, which had ended, so
        it was never delivered; only a later tab or scale change picked it up."""
        self.show("Density")
        self.assertTrue(self.pump(60, lambda: self.pp._lod is not None),
                        "the finished index never reached the panel")

    def test_popup_fills_itself_in_when_the_build_ends(self):
        """Cause 1: clicked while building, the popup updates without a second click."""
        gate = threading.Event()
        original = lod_index.LodIndex.build

        def slow_build(index, *a, **k):
            gate.wait(30)
            return original(index, *a, **k)
        lod_index.LodIndex.build = slow_build
        try:
            self.show("Density")
            self.assertEqual(self.pp._lod_state, "building")
            b, edges, x = self.middle_bin()
            self.click_value(x)
            self.assertIn("Listing the particles", self.tip())
            gate.set()
            self.wait_ready()
        finally:
            gate.set()
            lod_index.LodIndex.build = original
        expect = np.histogram(self.file_values("Density") * self.pp._unit_scalar, bins=edges)[0][b]
        self.assertIn(f"{expect:,}</span> particle", self.tip())
        self.assertNotIn("Listing", self.tip())

    def test_failed_build_says_so(self):
        """Cause 2: a build that fails is reported in the popup, not left 'building'."""
        original = lod_index.LodIndex.build

        def broken(index, *a, **k):
            raise OSError("disk full (test)")
        lod_index.LodIndex.build = broken
        try:
            self.show("Density")
            self.assertTrue(self.pump(30, lambda: self.pp._lod_state == "failed"))
        finally:
            lod_index.LodIndex.build = original
        _, _, x = self.middle_bin()
        self.click_value(x)
        self.assertIn("Could not list the particles", self.tip())
        self.assertIn("disk full (test)", self.tip())

    def test_bin_count_change_gets_its_own_index(self):
        """Cause 3: a re-render from cache with other bins builds and uses a matching index."""
        self.show("Density")
        self.wait_ready()
        self.win._on_custom_bins_requested(37)          # re-rendered from cache, no file read
        self.assertTrue(self.pump(10, lambda: len(self.pp._bin_edges_lod) == 38))
        self.wait_ready()
        values = self.file_values("Density") * self.pp._unit_scalar
        edges = np.asarray(self.pp._bin_edges_lod, float)
        counts = np.histogram(values, bins=edges)[0]
        for b in range(37):
            rows = self.pp._lod.query(self.pp._lod_key, b)
            self.assertEqual(len(rows), counts[b], f"bin {b}")
        b = int(np.argmax(counts))
        self.click_value(np.sqrt(edges[b] * edges[b + 1]))
        self.assertIn(f"Bin {b + 1} <span", self.tip())
        self.assertIn(f"{counts[b]:,}</span> particle", self.tip())

    def test_tab_round_trip_reuses_the_index(self):
        """A histogram shown again from cache gets the finished index, not a new build."""
        self.show("Density")
        self.wait_ready()
        built = self.pp._lod
        self.pp._toolbar._switch_view("Line")
        self.pump(2)
        self.pp._toolbar._switch_view("Histogram")
        self.assertTrue(self.pump(10, lambda: self.pp._lod_state == "ready"))
        self.assertIs(self.pp._lod, built)

    def test_rows_are_the_particles_in_the_bin(self):
        """The listed rows are exactly the file rows whose values fall in the bin."""
        self.show("Density")
        self.wait_ready()
        values = self.file_values("Density") * self.pp._unit_scalar
        edges = np.asarray(self.pp._bin_edges_lod, float)
        b = int(np.argmax(np.histogram(values, bins=edges)[0]))
        rows = np.sort(self.pp._lod.query(self.pp._lod_key, b))
        inside = np.flatnonzero((values >= edges[b]) & (values < edges[b + 1]))
        np.testing.assert_array_equal(rows, inside)

    def test_click_on_a_linear_axis_picks_the_bin_under_it(self):
        """Cause 4: positions are binned on a linear axis; the click must not be read as log."""
        self.show("Coordinates [x]")
        self.assertFalse(self.pp.pw.getViewBox().state["logMode"][0])
        self.wait_ready()
        edges = np.asarray(self.pp._bin_edges_lod, float)
        values = self.file_values("Coordinates", column=0) * self.pp._unit_scalar
        counts = np.histogram(values, bins=edges)[0]
        b = int(np.argmax(counts))                 # the densest bin, so the count is not 0
        self.click_value(0.5 * (edges[b] + edges[b + 1]))
        self.assertIn(f"Bin {b + 1} <span", self.tip())
        self.assertIn(f"{counts[b]:,}</span> particle", self.tip())

    def test_temperature_unit_gives_no_wrong_list(self):
        """Cause 5: in Temperature (K) the popup says why there is no list, never a wrong one."""
        self.show("InternalEnergy")
        if self.pp._lod_unit_label != "Temperature (K)":
            self.win._on_unit_changed("Temperature (K)")
            self.assertTrue(self.pump(30, lambda: self.pp._lod_unit_label == "Temperature (K)"))
        self.assertEqual(self.pp._lod_state, "unavailable")
        self.pp.pw.getViewBox().autoRange()      # a unit change keeps the old zoom
        _, _, x = self.middle_bin()
        self.click_value(x)
        self.assertIn("No particle list", self.tip())
        self.assertIn("Temperature (K)", self.tip())
        self.assertNotIn("particles</", self.tip())

    def test_click_beside_the_plot_closes_the_popup(self):
        self.show("Density")
        _, _, x = self.middle_bin()
        self.click_value(x)
        self.assertTrue(self.pp._insp_tip.isVisible())
        outside = self.pp.pw.getViewBox().sceneBoundingRect().topLeft() - QPointF(5, 5)
        self.pp._on_inspector_click(_Click(outside))
        self.assertFalse(self.pp._insp_tip.isVisible())

    def test_popup_is_dark_like_the_app(self):
        sheet = self.pp._insp_tip.styleSheet().lower()
        self.assertNotIn("#ffffff", sheet)
        self.assertIn("background:#141418", sheet)


if __name__ == "__main__":
    unittest.main()
