"""
The generic view for fields without a
hand-written plugin, the Matrix view of vector data, the LOD cache cap and
the temperature μ.

    python3 -m pytest -q test_generic_fields_and_temperature.py

Tests that need the Gadget-4 example outputs look for them in
COSMOVIEWER_DM_OUTPUT / COSMOVIEWER_SFR_OUTPUT and skip when absent.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import lod_index  # noqa: E402
from generic_plugin import configure_units, make_generic_leaves  # noqa: E402
from internal_energy_plugin import InternalEnergyPlugin, mean_molecular_weight  # noqa: E402
from astro_units import BOLTZMANN_CGS, PROTON_MASS_CGS  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")
SFR = os.environ.get("COSMOVIEWER_SFR_OUTPUT", "")


class TestGenericLeaves(unittest.TestCase):
    def test_vector_gets_components_and_magnitude(self):
        leaves = make_generic_leaves("PartType1", "Velocities", (10, 3), "float32")
        self.assertEqual(set(leaves), {"Velocities |·|", "Velocities [x]",
                                       "Velocities [y]", "Velocities [z]"})
        v = np.arange(30, dtype=float).reshape(10, 3)
        self.assertTrue(np.allclose(leaves["Velocities |·|"]().reduce_vector(v),
                                    np.linalg.norm(v, axis=1)))
        self.assertTrue(np.array_equal(leaves["Velocities [y]"]().reduce_vector(v), v[:, 1]))

    def test_positions_have_no_magnitude(self):
        leaves = make_generic_leaves("PartType1", "Coordinates", (10, 3), "float64")
        self.assertNotIn("Coordinates |·|", leaves)

    def test_ids_are_line_only_and_strings_skipped(self):
        ids = make_generic_leaves("PartType1", "ParticleIDs", (10,), "uint64")
        self.assertEqual(next(iter(ids.values())).TABS, ["Line"])
        self.assertEqual(make_generic_leaves("Header", "Name", (3,), "S10"), {})

    def test_lod_keys_differ_per_view(self):
        leaves = make_generic_leaves("PartType1", "Velocities", (10, 3), "float32")
        keys = {cls().lod_key_name("PartType1/Velocities") for cls in leaves.values()}
        self.assertEqual(len(keys), 4)

    def test_units_from_gadget4_exponents(self):
        cls = make_generic_leaves("PartType1", "Velocities", (10, 3), "float32")["Velocities [x]"]
        configure_units(cls, {"to_cgs": 1e5, "length_scaling": 1.0,
                              "mass_scaling": 0.0, "velocity_scaling": 1.0 - 1.0})
        self.assertEqual(str(cls.BASE_CGS_UNIT), "cm")
        configure_units(cls, {"to_cgs": 1e5, "length_scaling": 0.0,
                              "mass_scaling": 0.0, "velocity_scaling": 1.0})
        self.assertTrue(cls.BASE_CGS_UNIT.is_equivalent("km/s"))

    def test_no_unit_attributes_means_code_units(self):
        cls = make_generic_leaves("Group", "GroupLen", (5,), "int32")["GroupLen"]
        configure_units(cls, {})
        self.assertIsNone(cls.BASE_CGS_UNIT)
        self.assertIn("code units", cls.DISPLAY_NAME)

    def test_signed_data_is_binned_linearly_and_fully_counted(self):
        cls = make_generic_leaves("PartType1", "Velocities", (1000, 3), "float64")["Velocities [x]"]
        configure_units(cls, {})
        v = np.random.default_rng(3).normal(size=(1000, 3))
        key = "PartType1/Velocities"
        req = cls().compute({key: v}, {key: {}}, {"Time": 1.0},
                            {"n_global_map": {key: 1000}}, "Histogram", "log", "log", (0, 1000))
        self.assertFalse(req.log_x)
        self.assertEqual(int(np.nan_to_num(req.curves[0].y).sum()), 1000)


class TestMatrixColumns(unittest.TestCase):
    def test_vector_rows_become_columns(self):
        from plot_panel import _as_columns, _column_labels
        self.assertEqual(_as_columns(np.arange(4)).shape, (4, 1))
        self.assertEqual(_as_columns(np.zeros((4, 3))).shape, (4, 3))
        self.assertEqual(_column_labels(3), ["x", "y", "z"])
        self.assertEqual(_column_labels(2), ["0", "1"])

    def test_matrix_view_loads_vector_data(self):
        from PyQt6.QtWidgets import QApplication
        import plot_panel
        app = QApplication.instance() or QApplication([])  # noqa: F841
        view = plot_panel.MatrixView() if hasattr(plot_panel, "MatrixView") else None
        if view is None:
            self.skipTest("MatrixView class not exposed under that name")
        v = np.arange(12, dtype=float).reshape(4, 3)
        view.load(v, "PartType1/Velocities")
        self.assertEqual(view._table.columnCount(), 3)
        self.assertEqual(view._table.item(2, 1).text().strip().rstrip("0").rstrip("."),
                         plot_panel._format_cell(7.0, "Auto").strip().rstrip("0").rstrip("."))


def _fake_set(d, key, size, when):
    for suffix in (".offsets.npy", ".indices.npy", ".edges.npy", ".meta.json"):
        p = os.path.join(d, key + suffix)
        with open(p, "wb") as f:
            f.write(b"\0" * (size // 4))
        os.utime(p, (when, when))


class TestCacheCap(unittest.TestCase):
    def setUp(self):
        lod_index.LodIndex.PROTECTED_KEYS.clear()

    def test_oldest_sets_go_first_and_others_stay(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            keys = [f"{i:016x}" for i in range(5)]
            for i, k in enumerate(keys):
                _fake_set(d, k, 4000, now - 1000 * (5 - i))      # keys[0] oldest
            open(os.path.join(d, "notes.txt"), "w").write("x" * 50_000)
            open(os.path.join(d, keys[0] + ".indices.npy.tmp"), "w").write("x")
            removed = lod_index.enforce_cache_cap(d, cap_bytes=12_000)
            self.assertEqual([k for k, _ in removed], keys[:2])
            left = lod_index.cache_sets(d)
            self.assertEqual(set(left), set(keys[2:]))
            self.assertTrue(os.path.exists(os.path.join(d, "notes.txt")))
            self.assertTrue(os.path.exists(os.path.join(d, keys[0] + ".indices.npy.tmp")))

    def test_protected_and_kept_sets_survive(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            keys = [f"{i:016x}" for i in range(3)]
            for i, k in enumerate(keys):
                _fake_set(d, k, 4000, now - 1000 * (3 - i))
            lod_index.LodIndex.PROTECTED_KEYS.add(keys[0])
            removed = lod_index.enforce_cache_cap(d, cap_bytes=1, keep={keys[2]})
            self.assertEqual([k for k, _ in removed], [keys[1]])
            self.assertEqual(set(lod_index.cache_sets(d)), {keys[0], keys[2]})

    def test_under_cap_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            _fake_set(d, "0" * 16, 4000, time.time())
            self.assertEqual(lod_index.enforce_cache_cap(d, cap_bytes=10 ** 9), [])


class TestTemperatureMu(unittest.TestCase):
    def test_formula_limits(self):
        self.assertAlmostEqual(mean_molecular_weight(0.0), 4 / (1 + 3 * 0.76), places=12)
        ne_full = 1 + (1 - 0.76) / (2 * 0.76)          # H and He fully ionised
        self.assertAlmostEqual(mean_molecular_weight(ne_full), 0.588, places=3)

    def _compute(self, with_ne):
        p = InternalEnergyPlugin()
        u = np.array([1e12, 2e12, 3e12])
        data = {"PartType0/InternalEnergy": u}
        if with_ne:
            data["PartType0/ElectronAbundance"] = np.array([0.0, 0.5, 1.158])
        r = p._run_stage1_stage2(data, {k: {} for k in data}, {"Time": 1.0}, {})
        return p, r, data

    def test_per_particle_mu_from_electron_abundance(self):
        p, r, data = self._compute(True)
        T = p.custom_unit_transform(r.cgs_data, "Temperature (K)")
        mu = mean_molecular_weight(data["PartType0/ElectronAbundance"])
        ref = (2 / 3) * mu * PROTON_MASS_CGS * data["PartType0/InternalEnergy"] / BOLTZMANN_CGS
        self.assertTrue(np.allclose(T, ref, rtol=1e-12))
        self.assertIn("ElectronAbundance", p.get_custom_status(T, "Temperature (K)", 1.0))

    def test_without_electron_abundance_says_assumed(self):
        p, r, _ = self._compute(False)
        T = p.custom_unit_transform(r.cgs_data, "Temperature (K)")
        ref = (2 / 3) * 0.588 * PROTON_MASS_CGS * r.cgs_data / BOLTZMANN_CGS
        self.assertTrue(np.allclose(T, ref))
        self.assertIn("assumed", p.get_custom_status(r.cgs_data, "erg/g", 1.0))

    def test_status_with_temperature_unit_is_not_double_converted(self):
        p, r, _ = self._compute(True)
        T = p.custom_unit_transform(r.cgs_data, "Temperature (K)")
        self.assertEqual(p.get_custom_status(T, "Temperature (K)", 1.0),
                         p.get_custom_status(r.cgs_data, "erg/g", 1.0))

    @unittest.skipUnless(SFR and os.path.exists(os.path.join(SFR, "snapshot_100.hdf5")),
                         "COSMOVIEWER_SFR_OUTPUT not set")
    def test_matches_h5py_on_colliding_galaxies(self):
        from io_manager import VirtualSnapshotSet
        snap = os.path.join(SFR, "snapshot_100.hdf5")
        vs = VirtualSnapshotSet(snap)
        p = InternalEnergyPlugin()
        data, attrs = {}, {}
        for path in p.REQUIRES:
            try:
                data[path] = vs.extract_array(path, 25_000_000).data
                attrs[path] = vs.get_attributes(path)
            except KeyError:
                pass
        r = p._run_stage1_stage2(data, attrs, vs.global_header, vs.global_params)
        T = p.custom_unit_transform(r.cgs_data, "Temperature (K)")
        with h5py.File(snap) as f:
            ds = f["PartType0/InternalEnergy"]
            u = ds[:] * ds.attrs["to_cgs"]
            ne = f["PartType0/ElectronAbundance"][:].astype(float)
        ref = (2 / 3) * mean_molecular_weight(ne) * PROTON_MASS_CGS * u / BOLTZMANN_CGS
        self.assertEqual(len(T), len(ref))
        self.assertLess(np.max(np.abs(T - ref) / ref), 1e-6)


@unittest.skipUnless(DM and os.path.exists(os.path.join(DM, "snapshot_000.hdf5")),
                     "COSMOVIEWER_DM_OUTPUT not set")
class TestDMGenericAgainstH5py(unittest.TestCase):
    def test_velocity_magnitude_histogram_counts_everything(self):
        from io_manager import VirtualSnapshotSet
        snap = os.path.join(DM, "snapshot_000.hdf5")
        vs = VirtualSnapshotSet(snap)
        path = "PartType1/Velocities"
        with h5py.File(snap) as f:
            ds = f[path]
            cls = make_generic_leaves("PartType1", "Velocities", ds.shape, ds.dtype)["Velocities |·|"]
            configure_units(cls, dict(ds.attrs))
            n = ds.shape[0]
        chunk = vs.extract_array(path, 25_000_000)
        req = cls().compute({path: chunk.data}, {path: vs.get_attributes(path)},
                            vs.global_header, {**vs.global_params,
                                               "n_global_map": {path: chunk.n_global}},
                            "Histogram", "log", "log", (0, n))
        self.assertEqual(int(np.nan_to_num(req.curves[0].y).sum()), n)


if __name__ == "__main__":
    unittest.main()
