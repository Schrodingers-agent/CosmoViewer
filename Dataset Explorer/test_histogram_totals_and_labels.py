"""
Histogram totals, sample labels, the LOD index,
export columns, EAGLE metadata merging and label rendering.

    python3 -m pytest -q test_histogram_totals_and_labels.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from plugin_base import BasePlugin  # noqa: E402
from data_utils import latex_to_html, run_cosmology  # noqa: E402
import lod_index  # noqa: E402


class _Hist(BasePlugin):
    GROUP = "PartType0"
    DATASET = "X"
    TABS = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {}


def _run(values, tab="Histogram", n_global=None):
    p = _Hist()
    key = "PartType0/X"
    params = {"n_global_map": {key: n_global or len(values)}}
    return p.compute({key: np.asarray(values, dtype=np.float64)}, {key: {}},
                     {"Time": 1.0}, params, tab, "log", "log", (0, len(values)))


class TestHistogramTotals(unittest.TestCase):
    def test_every_positive_value_is_counted(self):
        rng = np.random.default_rng(1)
        for trial in range(20):
            x = 10 ** rng.uniform(-30, 30, 5000) * rng.uniform(0.5, 2)
            req = _run(x)
            counts = np.nan_to_num(req.curves[0].y, nan=0)
            self.assertEqual(int(counts.sum()), len(x), f"trial {trial}")
            self.assertEqual(req.bin_edges[0], x.min())
            self.assertEqual(req.bin_edges[-1], x.max())

    def test_full_read_is_not_labelled_as_sample(self):
        req = _run(np.arange(1, 1001))
        self.assertEqual(req.y_label, "Count")
        self.assertNotIn("SAMPLE", req.status)

    def test_strided_read_is_labelled(self):
        req = _run(np.arange(1, 1001), n_global=15_000)
        self.assertIn("1-in-15", req.y_label)
        self.assertIn("SAMPLE: 1-in-15 of 15,000", req.status)

    def test_line_sample_is_labelled(self):
        req = _run(np.arange(1, 1001), tab="Line", n_global=15_000)
        self.assertNotIn("Full array", req.status)
        self.assertIn("SAMPLE: 1,000 of 15,000", req.status)
        req = _run(np.arange(1, 1001), tab="Line")
        self.assertIn("Full array", req.status)


class TestLodIndex(unittest.TestCase):
    def test_excluded_particles_are_not_indexed(self):
        edges = np.logspace(0, 2, 5)
        data = np.array([0.0, -1.0, np.nan, 1.0, 5.0, 100.0, 1000.0, 50.0])
        ids = lod_index._bin_ids(data, edges, 4)
        self.assertEqual(list(ids[:3]), [-1, -1, -1])
        self.assertEqual(ids[3], 0)          # first edge
        self.assertEqual(ids[5], 3)          # last edge goes in the last bin
        self.assertEqual(ids[6], -1)         # beyond the last edge
        counts, _ = np.histogram(data[np.isfinite(data)], bins=edges)
        np.testing.assert_array_equal(lod_index._bin_counts(ids, 4), counts)

    def test_build_matches_histogram(self):
        rng = np.random.default_rng(2)
        data = np.concatenate([np.zeros(300), 10 ** rng.uniform(-2, 3, 2000)])
        rng.shuffle(data)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.hdf5")
            with h5py.File(path, "w") as f:
                f.create_group("Header").attrs["NumPart_ThisFile"] = np.array([len(data), 0, 0, 0, 0, 0])
                f["PartType0/X"] = data

            class _V:
                files = [path]
                particle_counts = {"PartType0": len(data)}
            pos = data[data > 0]
            edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 21)
            edges[0], edges[-1] = pos.min(), pos.max()
            key = lod_index.LodCacheKey.build(snapshot_path=path, dataset_path="PartType0/X", cache_dir=d)
            lod = lod_index.LodIndex()
            lod.build(_V(), "PartType0/X", edges, 1.0, key, lambda: False)
            counts, _ = np.histogram(pos, bins=edges)
            for b in range(20):
                idx = lod.query(key, b)
                self.assertEqual(len(idx), counts[b])
                vals = data[idx]
                self.assertTrue(np.all((vals >= edges[b]) & (vals <= edges[b + 1])))
            self.assertTrue(lod.is_valid(key, 1.0, 20))


class TestExportColumns(unittest.TestCase):
    def test_histogram_export_has_edges_and_counts(self):
        from plot_panel import PlotPanel
        x = np.arange(1, 1001, dtype=float)
        req = _run(x)
        notes, cols = PlotPanel._plot_export_columns(req)
        names = [c[0] for c in cols]
        self.assertEqual(names, ["bin_low", "bin_high", "count"])
        self.assertEqual(int(cols[2][1].sum()), 1000)
        np.testing.assert_array_equal(cols[0][1][1:], cols[1][1][:-1])

    def test_line_export_says_downsampled(self):
        from plot_panel import PlotPanel
        req = _run(np.arange(1, 1001, dtype=float), tab="Line")
        notes, cols = PlotPanel._plot_export_columns(req)
        self.assertIn("NOT the full dataset", notes[0])
        self.assertEqual(cols[0][0], "particle_index")


class TestEagleMetadata(unittest.TestCase):
    def test_runtimepars_and_units_fill_parameters(self):
        from io_manager import VirtualSnapshotSet
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap_028_z000p000.0.hdf5")
            with h5py.File(path, "w") as f:
                h = f.create_group("Header")
                for k, v in dict(Time=1.0, ExpansionFactor=1.0, Redshift=0.0, HubbleParam=0.6777,
                                 Omega0=0.307, OmegaLambda=0.693, NumFilesPerSnapshot=1).items():
                    h.attrs[k] = v
                h.attrs["NumPart_ThisFile"] = np.array([3, 0, 0, 0, 0, 0])
                h.attrs["NumPart_Total"] = np.array([3, 0, 0, 0, 0, 0])
                h.attrs["NumPart_Total_HighWord"] = np.zeros(6, int)
                f.create_group("Parameters").create_group("ChemicalElements")
                rp = f.create_group("RuntimePars")
                rp.attrs["ComovingIntegrationOn"] = 1
                rp.attrs["UnitMass_in_g"] = np.inf
                u = f.create_group("Units")
                u.attrs["UnitLength_in_cm"] = 3.085678e24
                u.attrs["UnitMass_in_g"] = 1.989e43
                u.attrs["UnitVelocity_in_cm_per_s"] = 1e5
                f["PartType0/Density"] = np.ones(3)
            v = VirtualSnapshotSet(path)
            self.assertEqual(int(v.global_params["ComovingIntegrationOn"]), 1)
            self.assertEqual(float(v.global_params["UnitMass_in_g"]), 1.989e43)   # not RuntimePars' inf
            self.assertTrue(run_cosmology(v.global_header, v.global_params)["is_cosmological"])


class TestLabels(unittest.TestCase):
    def test_identifiers_are_not_subscripted(self):
        self.assertEqual(latex_to_html("a_scaling=1.0"), "a_scaling=1.0")
        self.assertEqual(latex_to_html("phase_space_305_r.npy"), "phase_space_305_r.npy")
        self.assertEqual(latex_to_html("x_0"), "x<sub>0</sub>")


if __name__ == "__main__":
    unittest.main()
