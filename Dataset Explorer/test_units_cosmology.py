"""
Cosmological scaling (h and a exponents) and the run's cosmology.

Real-file tests use the Gadget-4 example outputs. Point these at them:
    COSMOVIEWER_DM_OUTPUT   = .../examples/DM-L50-N128/output
    COSMOVIEWER_SFR_OUTPUT  = .../examples/CollidingGalaxiesSFR/output
They are skipped when unset.

Run from the Dataset Explorer folder:
    python3 -m pytest -q test_units_cosmology.py
"""

import os
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import run_cosmology, scaling_exponents, coerce_attrs  # noqa: E402
from astro_units import cosmology_context  # noqa: E402
from plugin_base import BasePlugin  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT")
SFR = os.environ.get("COSMOVIEWER_SFR_OUTPUT")


class _Probe(BasePlugin):
    """Minimal plugin so Stage 1–2 can run on any dataset."""
    GROUP = "PartType1"
    DATASET = "Coordinates"
    TABS = ["Histogram"]
    TREE_CONTRIBUTIONS = {}

    def reduce_vector(self, array):
        return array[:, 0]


def _stage12(path, group, dataset, n=2000):
    with h5py.File(path, "r") as f:
        header = coerce_attrs(f["Header"].attrs)
        params = coerce_attrs(f["Parameters"].attrs) if "Parameters" in f else {}
        ds = f[f"{group}/{dataset}"]
        raw = ds[:n].astype(np.float64)
        attrs = coerce_attrs(ds.attrs)
    p = _Probe()
    p.GROUP, p.DATASET = group, dataset
    key = f"{group}/{dataset}"
    r = p._run_stage1_stage2({key: raw}, {key: attrs}, header, params)
    return r, raw, attrs, header, params


class TestScalingNames(unittest.TestCase):
    def test_gadget4_and_eagle_names(self):
        self.assertEqual(scaling_exponents({"h_scaling": -1.0, "a_scaling": 1.0}), (-1.0, 1.0))
        self.assertEqual(scaling_exponents({"h-scale-exponent": 2.0, "aexp-scale-exponent": -3.0}), (2.0, -3.0))
        self.assertEqual(scaling_exponents({}), (0.0, 0.0))


class TestRunCosmology(unittest.TestCase):
    def test_gadget4_cosmological_uses_parameters(self):
        r = run_cosmology({"Time": 0.25, "Redshift": 3.0},
                          {"ComovingIntegrationOn": 1, "HubbleParam": 0.678, "Omega0": 0.308,
                           "OmegaLambda": 0.692})
        self.assertTrue(r["is_cosmological"])
        self.assertAlmostEqual(r["a"], 0.25)
        self.assertAlmostEqual(r["h"], 0.678)

    def test_non_cosmological_time_is_not_a(self):
        r = run_cosmology({"Time": 4.0}, {"ComovingIntegrationOn": 0, "HubbleParam": 1.0})
        self.assertFalse(r["is_cosmological"])
        self.assertEqual(r["a"], 1.0)

    def test_eagle_flag_from_runtimepars_merge(self):
        # VirtualSnapshotSet merges RuntimePars/ComovingIntegrationOn into params
        r = run_cosmology({"Time": 1.0, "ExpansionFactor": 1.0, "HubbleParam": 0.6777,
                           "Omega0": 0.307, "OmegaLambda": 0.693}, {"ComovingIntegrationOn": 1})
        self.assertTrue(r["is_cosmological"])
        self.assertAlmostEqual(r["h"], 0.6777)

    def test_no_flag_inferred_from_header(self):
        self.assertTrue(run_cosmology({"Time": 0.5, "Omega0": 0.3, "HubbleParam": 0.7})["is_cosmological"])
        self.assertFalse(run_cosmology({"Time": 4.0, "Omega0": 0.0})["is_cosmological"])

    def test_cosmology_context_uses_file_parameters(self):
        hdr = {"Time": 1.0, "Redshift": 0.0}
        par = {"ComovingIntegrationOn": 1, "HubbleParam": 0.678, "Omega0": 0.308,
               "OmegaLambda": 0.692, "OmegaBaryon": 0.0482}
        ctx = cosmology_context(hdr, par)
        from astropy.cosmology import LambdaCDM
        ref = LambdaCDM(H0=67.8, Om0=0.308, Ode0=0.692, Tcmb0=0.0).age(0).value
        self.assertAlmostEqual(ctx["age_gyr"], ref, places=6)
        self.assertIn("file:", ctx["cosmo_name"])


@unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
class TestDMRealFile(unittest.TestCase):
    def test_coordinates_become_physical_cm(self):
        path = os.path.join(DM, "snapshot_000.hdf5")       # z ≈ 3, a ≈ 0.25
        r, raw, attrs, header, params = _stage12(path, "PartType1", "Coordinates")
        a = float(header["Time"])
        h = float(params["HubbleParam"])
        expected = raw[:, 0] * float(attrs["to_cgs"]) * h ** -1 * a ** 1
        np.testing.assert_allclose(r.cgs_data, expected, rtol=1e-12)
        self.assertTrue(r.cosmo_applied)
        self.assertTrue(r.is_cosmological)
        self.assertAlmostEqual(r.h, 0.678)

    def test_velocities_use_sqrt_a(self):
        path = os.path.join(DM, "snapshot_000.hdf5")
        r, raw, attrs, header, params = _stage12(path, "PartType1", "Velocities")
        a = float(header["Time"])
        self.assertEqual(float(attrs["a_scaling"]), 0.5)
        np.testing.assert_allclose(r.cgs_data, raw[:, 0] * float(attrs["to_cgs"]) * a ** 0.5, rtol=1e-12)


@unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
class TestIsolatedRealFile(unittest.TestCase):
    def test_time_is_not_used_as_scale_factor(self):
        path = sorted(Path(SFR).glob("snapshot_*.hdf5"))[-1]
        r, raw, attrs, header, params = _stage12(str(path), "PartType0", "Density")
        self.assertFalse(r.is_cosmological)
        self.assertEqual(r.a, 1.0)
        self.assertEqual(r.redshift, 0.0)
        np.testing.assert_allclose(r.cgs_data, raw * float(attrs["to_cgs"]), rtol=1e-12)
        ctx = cosmology_context(header, params)
        self.assertFalse(ctx["is_cosmological"])


    def test_formation_time_is_time_not_scale_factor(self):
        from scale_factor_at_birth_plugin import ScaleFactorAtBirthPlugin
        path = sorted(Path(SFR).glob("snapshot_*.hdf5"))[-1]
        with h5py.File(path, "r") as f:
            header = coerce_attrs(f["Header"].attrs)
            params = coerce_attrs(f["Parameters"].attrs)
            raw = f["PartType4/StellarFormationTime"][:].astype(np.float64)
        p = ScaleFactorAtBirthPlugin()
        key = "PartType4/StellarFormationTime"
        req = p.compute({key: raw}, {key: {}}, header, params, "Histogram", "log", "log", (0, len(raw)))
        unit_time_gyr = params["UnitLength_in_cm"] / params["UnitVelocity_in_cm_per_s"] / 3.15576e16
        self.assertIn("formation times, not scale factors", req.info)
        self.assertNotIn("z ≈", req.info)
        self.assertIn(f"{np.median(raw) * unit_time_gyr:.4f} Gyr", req.info)
        self.assertIn("Gyr", req.x_label)


if __name__ == "__main__":
    unittest.main()
