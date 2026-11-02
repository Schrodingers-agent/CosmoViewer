"""
Histogram binning and drawing: the "disjointed" Analyse histograms.

Four causes, each with a test that fails on the code before the fix:
  1. Freedman-Diaconis was applied to the linear values while the bins were
     log-spaced, so data spanning decades hit MAX_BINS and the log bins in its
     tail were mostly empty (EAGLE black holes: 3,562 values, 100 bins).
  2. Whole-number data (GroupNumber, SubGroupNumber, BH_CumlNumSeeds) had bins
     narrower than 1, so bins without an integer in them were always empty.
  3. Non-negative data with zeros (StarFormationRate, TotalMassFromSNIa) was
     put on a linear axis over [0, max], crushing values spanning decades.
     EAGLE's 2**30 "in no group" flag stretched GroupNumber over nine decades.
  4. The curve was a polyline through bin centres with empty bins as NaN, so
     it broke at every empty bin and a lone bin between two empty ones was
     drawn as nothing.

    python3 -m pytest -q test_histogram_binning.py
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from data_utils import adaptive_bin_count, integer_bin_edges  # noqa: E402
from generic_plugin import make_generic_leaves  # noqa: E402
from plugin_base import BasePlugin  # noqa: E402


class _LogHist(BasePlugin):
    """A hand-written-style plugin: positive quantity, log axis, MAX_BINS 100."""
    GROUP = "PartType5"
    DATASET = "X"
    TABS = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {}


def _run(plugin, values, dtype=None, key=None):
    key = key or f"{plugin.GROUP}/{plugin.DATASET}"
    values = np.asarray(values)
    params = {"n_global_map": {key: len(values)},
              "_source_dtypes": {key: str(dtype or values.dtype)}}
    return plugin.compute({key: values.astype(np.float64)}, {key: {}}, {"Time": 1.0},
                          params, "Histogram", "log", "log", (0, len(values)))


def _generic(dataset, values, group="PartType5"):
    cls = make_generic_leaves(group, dataset, np.shape(values), np.asarray(values).dtype)[dataset]
    return cls()


def _counts(req):
    return np.nan_to_num(np.asarray(req.curves[0].y), nan=0.0)


def _interior_empty(counts):
    nz = np.flatnonzero(counts)
    return int((counts[nz[0]:nz[-1] + 1] == 0).sum())


class TestBinCount(unittest.TestCase):
    def test_fd_on_log_values_for_log_bins(self):
        """Cause 1: 3,562 log-normal values over six decades (the EAGLE BH count)."""
        rng = np.random.default_rng(3)
        x = 10 ** rng.normal(6.5, 1.0, 3562)
        k_log, _ = adaptive_bin_count(x, 8, 100, log=True)
        # The old rule, FD on the linear values: far above MAX_BINS.
        k_lin = np.ptp(x) / (2 * np.subtract(*np.percentile(x, [75, 25])) * len(x) ** (-1 / 3))
        self.assertGreater(k_lin, 100)
        self.assertLess(k_log, 50)
        # FD in log10: W = 2 IQR n^(-1/3), k = range / W
        lx = np.log10(x)
        iqr = np.subtract(*np.percentile(lx, [75, 25]))
        k_fd = int(np.ceil(np.ptp(lx) / (2 * iqr * len(x) ** (-1 / 3))))
        self.assertEqual(k_log, min(k_fd, int(np.ceil(2 * len(x) ** (1 / 3)))))

    def test_long_tails_do_not_multiply_bins_for_small_sets(self):
        """3,562 signed values with long tails (an EAGLE BH velocity component):
        FD alone gave 70-86 bins; capped at Rice it gives 31."""
        rng = np.random.default_rng(11)
        x = rng.standard_t(2, 3562) * 100.0
        k_fd = int(np.ceil(np.ptp(x) / (2 * np.subtract(*np.percentile(x, [75, 25]))
                                        * len(x) ** (-1 / 3))))
        self.assertGreater(k_fd, 60)
        k, how = adaptive_bin_count(x, 8, 100)
        self.assertEqual(k, 31)
        self.assertEqual(how, "FD, capped at Rice")
        # Large sets are not affected: Rice is above MAX_BINS from n ≈ 125k.
        big = rng.standard_t(2, 200_000)
        self.assertEqual(adaptive_bin_count(big, 8, 100)[0], 100)

    def test_small_log_set_has_few_empty_bins(self):
        rng = np.random.default_rng(5)
        x = 10 ** rng.normal(-1.0, 1.2, 3562)
        req = _run(_LogHist(), x)
        self.assertTrue(req.log_x)
        self.assertLess(len(req.bin_edges) - 1, 50)
        self.assertLessEqual(_interior_empty(_counts(req)), 3)
        self.assertEqual(int(_counts(req).sum()), len(x))

    def test_large_floats_are_not_taken_for_integers(self):
        """Above 2**53 every float is whole: entropy in CGS must stay on log bins."""
        rng = np.random.default_rng(2)
        x = 10 ** rng.uniform(20, 30, 5000)
        req = _run(_LogHist(), x.astype(np.float32), dtype="float32")
        self.assertTrue(req.log_x)
        self.assertNotIn("whole-number", req.status)
        self.assertEqual(req.bin_edges[0], x.astype(np.float32).astype(float).min())


class TestWholeNumbers(unittest.TestCase):
    def test_edges_are_half_integers_and_every_bin_holds_one(self):
        for lo, hi, k, log in [(1, 61, 31, True), (1, 61, 31, False), (0, 110, 100, False),
                               (32, 30000, 100, True), (2, 5, 100, True)]:
            e = integer_bin_edges(lo, hi, k, log=log)
            self.assertEqual(e[0], lo - 0.5)
            # Equal linear widths may run past the largest value by under one bin.
            self.assertGreaterEqual(e[-1], hi + 0.5)
            self.assertLess(e[-2], hi)
            np.testing.assert_array_equal(e % 1, 0.5)
            self.assertTrue(np.all(np.diff(e) >= 1), (lo, hi, k, log))
            self.assertLessEqual(len(e) - 1, k)
            if not log:
                self.assertEqual(len(set(np.diff(e))), 1)   # equal widths

    def test_seed_counts_no_picket_fence(self):
        """Cause 2: BH_CumlNumSeeds, uint32 1..61, mostly 1."""
        rng = np.random.default_rng(1)
        x = np.minimum(rng.geometric(0.9, 3562), 61).astype(np.uint32)
        x[:3] = [1, 2, 3]
        req = _run(_generic("BH_CumlNumSeeds", x), x)
        c = _counts(req)
        e = req.bin_edges
        self.assertIn("whole-number edges", req.status)
        np.testing.assert_array_equal(e % 1, 0.5)
        for b in range(len(c)):                # a bin is empty only if no value lies in it
            inside = (x >= e[b]) & (x < e[b + 1])
            self.assertEqual(c[b], inside.sum())
        self.assertEqual(int(c.sum()), len(x))

    def test_subgroup_number_keeps_zero_on_a_linear_axis(self):
        x = np.r_[np.zeros(2424), np.arange(1, 61), np.full(214, 2 ** 30)].astype(np.int32)
        req = _run(_generic("SubGroupNumber", x), x)
        self.assertFalse(req.log_x)
        self.assertEqual(req.bin_edges[0], -0.5)
        self.assertEqual(int(_counts(req).sum()), 2424 + 60)

    def test_eagle_no_group_flag_is_counted_not_binned(self):
        """Cause 3b: 2**30 marks 'in no group'; binned, it stretched the axis to 1e9."""
        x = np.r_[np.arange(1, 1407).repeat(2), np.full(180, 2 ** 30)].astype(np.int32)
        req = _run(_generic("GroupNumber", x), x)
        self.assertLess(req.bin_edges[-1], 2000)
        self.assertIn("180 flagged", req.status)
        self.assertIn("in no group", req.status)
        self.assertEqual(int(_counts(req).sum()), 2 * 1406)
        # Other datasets do not lose the value.
        req = _run(_generic("BH_CumlNumSeeds", x.astype(np.uint32)), x.astype(np.uint32))
        self.assertEqual(int(_counts(req).sum()), len(x))


class TestZerosAndSigns(unittest.TestCase):
    def test_zero_heavy_positive_data_stays_on_log_axis(self):
        """Cause 3: StarFormationRate, 99.7 % zeros, the rest over three decades."""
        rng = np.random.default_rng(4)
        x = np.r_[np.zeros(40000), 10 ** rng.uniform(-7, -3.3, 1000)].astype(np.float32)
        req = _run(_generic("StarFormationRate", x, "PartType0"), x)
        self.assertTrue(req.log_x)
        self.assertIn("40,000 zero or negative, not on the log axis", req.status)
        self.assertEqual(int(_counts(req).sum()), 1000)
        self.assertLessEqual(_interior_empty(_counts(req)), 2)

    def test_signed_data_stays_linear(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, 5000).astype(np.float32)
        req = _run(_generic("OnEquationOfState", x, "PartType0"), x)
        self.assertFalse(req.log_x)
        self.assertEqual(int(_counts(req).sum()), len(x))

    def test_log_bin_centres_are_geometric(self):
        x = 10 ** np.random.default_rng(8).uniform(0, 4, 2000)
        req = _run(_LogHist(), x)
        e = np.asarray(req.bin_edges)
        np.testing.assert_allclose(req.curves[0].x, np.sqrt(e[:-1] * e[1:]))


class TestStarFormationRateUnit(unittest.TestCase):
    def test_eagle_sfr_is_shown_in_msun_per_yr(self):
        """EAGLE stores Msun/yr with CGSConversionFactor 6.30e25 (to g/s). The CGS
        value was labelled Msun/yr, so the axis read 6.3e25 times too high."""
        from generic_plugin import configure_units
        x = np.r_[np.zeros(50), np.linspace(7e-5, 5.5e-4, 200)].astype(np.float32)
        cls = make_generic_leaves("PartType0", "StarFormationRate", x.shape, x.dtype)["StarFormationRate"]
        attrs = {"CGSConversionFactor": 6.304278922345484e25,
                 "h-scale-exponent": 0.0, "aexp-scale-exponent": 0.0}
        configure_units(cls, attrs)
        key = "PartType0/StarFormationRate"
        req = cls().compute({key: x.astype(np.float64)}, {key: attrs},
                            {"Time": 1.0, "HubbleParam": 0.6777, "ExpansionFactor": 1.0},
                            {"n_global_map": {key: len(x)}, "_source_dtypes": {key: "float32"}},
                            "Histogram", "log", "log", (0, len(x)))
        self.assertIn("yr", req.unit_label)
        # astropy's Msun/yr is 6.3009e25 g/s, EAGLE's factor 6.3043e25: 0.05 % apart
        np.testing.assert_allclose([req.bin_edges[0], req.bin_edges[-1]],
                                   [x[x > 0].min(), x.max()], rtol=1e-3)


class TestOutline(unittest.TestCase):
    def setUp(self):
        from plot_panel import PlotPanel
        self.outline = PlotPanel._histogram_outline

    def test_linear_counts_axis_draws_empty_bins_at_zero(self):
        edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x, y = self.outline(edges, np.array([3.0, np.nan, 2.0, np.nan]))
        self.assertTrue(np.all(np.isfinite(y)))
        np.testing.assert_array_equal(x, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        np.testing.assert_array_equal(y, [0, 3, 3, 0, 0, 2, 2, 0, 0, 0])

    def test_log_counts_axis_keeps_a_lone_bin(self):
        """Cause 4: a bin between two empty ones keeps a flat top on a log axis,
        and the outline is unbroken: empty bins go down to a floor of 0.5."""
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        x, y = self.outline(edges, np.array([np.nan, 1.0, np.nan]), log_y=True)
        self.assertTrue(np.all(np.isfinite(y)))
        np.testing.assert_array_equal(x, [1, 1, 2, 2, 3, 3, 4, 4])
        np.testing.assert_array_equal(y, [0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5])

    def test_log_floor_is_below_every_count(self):
        x, y = self.outline(np.arange(5.0), np.array([40.0, np.nan, 7.0, 3.0]), log_y=True)
        self.assertEqual(y[0], 1.5)
        self.assertEqual(y[3], 1.5)            # the empty bin
        self.assertLess(y[0], 3.0)


class TestDrawnHistogram(unittest.TestCase):
    """The histogram on screen is the bin outline, whatever the style menu says."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        import density_explorer
        import lod_index
        import selection_sets
        from measurements.models import plummer, write_gadget4_snapshot
        cls.app = QApplication.instance() or QApplication(sys.argv[:1])
        cls.tmp = Path(tempfile.mkdtemp())
        cls._mods = (selection_sets, lod_index)
        cls._saved = (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR,
                      lod_index.LodIndex.CACHE_DIR)
        selection_sets.FALLBACK_ROOT = cls.tmp / "sets"
        selection_sets.INSTANCES_DIR = cls.tmp / "instances"
        lod_index.LodIndex.CACHE_DIR = str(cls.tmp / "lod")
        n = 3000
        pos, vel, _ = plummer(n, 1e6, 0.003, seed=3)
        rng = np.random.default_rng(9)
        cls.seeds = np.minimum(rng.geometric(0.85, n), 40).astype(np.uint32)
        cls.snap = cls.tmp / "output" / "snapshot_000.hdf5"
        cls.snap.parent.mkdir(parents=True)
        write_gadget4_snapshot(cls.snap, {0: dict(
            pos=pos + 2.0, vel=vel, mass=np.full(n, 1e-4), ids=np.arange(n),
            fields={"Density": rng.lognormal(0.0, 2.0, n), "NumSeeds": cls.seeds})})
        cls.density_explorer = density_explorer

    @classmethod
    def tearDownClass(cls):
        selection_sets, lod_index = cls._mods
        (selection_sets.FALLBACK_ROOT, selection_sets.INSTANCES_DIR,
         lod_index.LodIndex.CACHE_DIR) = cls._saved

    def setUp(self):
        self.win = self.density_explorer.CosmoExplorer()
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

    def primary(self):
        items = [i for i in self.pp.pw.getPlotItem().listDataItems()
                 if getattr(i, "_plot_role", "") == "primary_data"]
        self.assertEqual(len(items), 1)
        return items[0]

    def test_drawn_curve_is_the_bin_outline(self):
        self.show("NumSeeds")
        edges = np.asarray(self.pp._bin_edges_lod, float)
        np.testing.assert_array_equal(edges % 1, 0.5)
        x, y = self.primary().getData()
        # pyqtgraph hands back log10 values on a log axis
        vb = self.pp.pw.getViewBox()
        if vb.state["logMode"][0]:
            x = 10 ** x
        np.testing.assert_allclose(x, np.repeat(edges, 2))
        counts = np.histogram(self.seeds, bins=edges)[0]
        tops = y[1:-1:2] if not vb.state["logMode"][1] else 10 ** y[1:-1:2]
        filled = counts > 0
        np.testing.assert_allclose(tops[filled], counts[filled])

    def test_interpolation_is_greyed_out_on_histograms_only(self):
        self.show("Density")
        tb = self.pp._toolbar
        self.assertFalse(any(rb.isEnabled() for rb in tb._rb_interp.values()))
        before = self.primary().getData()[0].copy()
        self.pp._on_style_changed("Line", "Step")            # still bins
        np.testing.assert_array_equal(self.primary().getData()[0], before)
        tb._switch_view("Line")
        self.assertTrue(self.pump(30, lambda: self.win._active_worker is None
                                  and self.pp.current_tab() == "Line"))
        self.pump(0.5)
        self.assertTrue(all(rb.isEnabled() for rb in tb._rb_interp.values()))


if __name__ == "__main__":
    unittest.main()
