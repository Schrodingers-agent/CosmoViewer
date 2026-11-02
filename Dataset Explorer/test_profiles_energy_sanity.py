"""
Radial profiles, the energy budget and the snapshot sanity report (measurements/).

    python3 -m pytest -q test_profiles_energy_sanity.py

S3 radial profiles: density, σ_r and β of Plummer spheres, Hernquist density,
a planted anisotropy, and Σ(R) of a projected Plummer sphere, each pooled over
seeds and compared with the analytic value in units of the quoted error.

S2 energy budget: the spline kernel, direct and Monte Carlo pair potentials
against each other and the Plummer value W = −3πGM²/(32a), Q = 0.50 ± 0.01 at
N = 10⁵, drift over a synthetic run, and agreement with Gadget-4's energy.txt.

S1 sanity report: every planted defect in a synthetic Gadget-4-style file is
caught by its own check and by no other, a clean file has no FAIL or WARN, and
the real DM-L50 snapshot_002 reports its two particles on the box edge.
Real-data cases use COSMOVIEWER_DM_OUTPUT when set.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from measurements.centre import Centre  # noqa: E402
from measurements.models import (  # noqa: E402
    anisotropic_velocities, hernquist, hernquist_mass, plummer, plummer_density,
    plummer_mass, plummer_sigma, write_gadget4_snapshot)
from measurements.energy import (  # noqa: E402
    code_energy_unit, conservation, energy_budget, potential_energy, read_energy_txt,
    spline_kernel)
from measurements.models import G_KPC_KMS2_MSUN  # noqa: E402
from measurements.profiles import radial_profile, surface_density  # noqa: E402
from measurements.runset import Run  # noqa: E402
from measurements.snapshot_view import SnapshotView  # noqa: E402
from measurements.snapshot_view import ParticleSet  # noqa: E402
from measurements.sanity import GADGET_G, GADGET_HUBBLE, sanity_report  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")
SFR = os.environ.get("COSMOVIEWER_SFR_OUTPUT", "")

BOX = 100.0          # code units (kpc)
N_GAS, N_DM = 2000, 3000


def _clean_parts(seed=0):
    rng = np.random.default_rng(seed)
    gas = dict(pos=rng.uniform(0, BOX, (N_GAS, 3)), vel=rng.normal(0, 50, (N_GAS, 3)),
               mass=np.full(N_GAS, 1e-4), ids=np.arange(1, N_GAS + 1),
               fields={"Density": rng.uniform(1e-6, 1e-3, N_GAS),
                       "InternalEnergy": np.r_[0.0, rng.uniform(1, 100, N_GAS - 1)],
                       "SmoothingLength": rng.uniform(0.5, 2.0, N_GAS)})
    dm = dict(pos=rng.uniform(0, BOX, (N_DM, 3)), vel=rng.normal(0, 80, (N_DM, 3)),
              mass=None, ids=np.arange(N_GAS + 1, N_GAS + N_DM + 1))
    return {0: gas, 1: dm}


def _write(path, parts, **kw):
    kw.setdefault("box", BOX)
    kw.setdefault("mass_table", [0, 5e-4, 0, 0, 0, 0])
    kw.setdefault("config", {"PERIODIC": b""})
    kw.setdefault("extra_params", {"SofteningClassOfPartType0": 0, "SofteningClassOfPartType1": 0,
                                   "SofteningComovingClass0": 0.5, "SofteningMaxPhysClass0": 0.5})
    return write_gadget4_snapshot(path, parts, **kw)


def _bad(report):
    return {(c.check, c.scope, c.status) for c in report.checks if c.status in ("FAIL", "WARN")}


class TestSanityPlantedDefects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _report(self, name, mutate=None, after=None, **kw):
        parts = _clean_parts()
        if mutate:
            mutate(parts)
        path = _write(self.dir / f"{name}.hdf5", parts, **kw)
        if after:
            with h5py.File(path, "r+") as f:
                after(f)
        return sanity_report(path)

    def assertOnly(self, report, expected):
        self.assertEqual(_bad(report), expected, report.to_text())

    def test_clean_file_passes(self):
        r = self._report("clean")
        self.assertOnly(r, set())
        self.assertEqual(r.worst, "PASS")
        self.assertTrue(r.provenance.is_science)
        self.assertEqual(r.provenance.n_total, N_GAS + N_DM)

    def test_header_count_mismatch(self):
        def after(f):
            f["Header"].attrs["NumPart_Total"] = np.array([N_GAS + 1, N_DM, 0, 0, 0, 0], np.uint64)
        r = self._report("count_header", after=after)
        self.assertIn(("counts", "PartType0", "FAIL"), _bad(r))

    def test_dataset_length_mismatch(self):
        def after(f):
            del f["PartType0/Density"]
            f["PartType0"].create_dataset("Density", data=np.ones(N_GAS - 3))
        r = self._report("count_dataset", after=after)
        self.assertOnly(r, {("counts", "PartType0", "FAIL")})
        self.assertIn("Density has 1,997 rows", r.find("counts", "PartType0")[0].detail)

    def test_nan_velocity(self):
        def m(p):
            p[1]["vel"][17, 1] = np.nan
        r = self._report("nan", m)
        self.assertOnly(r, {("finite", "PartType1", "FAIL")})
        c = r.find("finite", "PartType1")[0]
        self.assertEqual(c.count, 1)
        self.assertIn("index 17", c.examples[0])

    def test_inf_density(self):
        def m(p):
            p[0]["fields"]["Density"][5] = np.inf
        r = self._report("inf", m)
        self.assertOnly(r, {("finite", "PartType0", "FAIL")})

    def test_negative_mass_and_density(self):
        def m(p):
            p[0]["mass"][3] = -1e-4
            p[0]["fields"]["Density"][9] = 0.0
        r = self._report("negative", m)
        self.assertOnly(r, {("positive", "PartType0", "FAIL")})
        details = " ".join(c.detail for c in r.find("positive", "PartType0"))
        self.assertIn("Masses: 1 values", details)
        self.assertIn("Density: 1 values", details)

    def test_duplicate_ids_across_types(self):
        def m(p):
            p[1]["ids"][0] = p[0]["ids"][10]
        r = self._report("dup_ids", m)
        self.assertOnly(r, {("ids", "all types", "FAIL")})
        c = r.find("ids")[0]
        self.assertEqual(c.count, 2)
        self.assertIn("types [0, 1]", c.examples[0])

    def test_duplicate_position(self):
        def m(p):
            p[1]["pos"][100] = p[0]["pos"][200]
        r = self._report("dup_pos", m)
        self.assertOnly(r, {("positions", "all types", "WARN")})
        self.assertEqual(r.find("positions")[0].count, 2)

    def test_particle_on_box_edge(self):
        def m(p):
            p[1]["pos"][42, 2] = BOX
        r = self._report("edge", m)
        self.assertOnly(r, {("box", "PartType1", "WARN")})
        c = r.find("box", "PartType1")[0]
        self.assertEqual(c.count, 1)
        self.assertIn(f"ID {N_GAS + 1 + 42}", c.examples[0])

    def test_particle_outside_box(self):
        def m(p):
            p[0]["pos"][7, 0] = BOX * 1.01
            p[0]["pos"][8, 1] = -0.5
        r = self._report("outside", m)
        self.assertOnly(r, {("box", "PartType0", "FAIL")})
        self.assertEqual(r.find("box", "PartType0")[0].count, 2)

    def test_box_not_checked_without_periodic(self):
        def m(p):
            p[0]["pos"][7, 0] = BOX * 3
        r = self._report("not_periodic", m, config={})
        self.assertOnly(r, set())
        self.assertEqual(r.find("box")[0].status, "INFO")

    def test_speed_above_c(self):
        def m(p):
            p[1]["vel"][5] = [4e5, 0, 0]
        r = self._report("fast_c", m)
        self.assertOnly(r, {("velocities", "PartType1", "FAIL")})

    def test_speed_outlier(self):
        def m(p):
            p[1]["vel"][5] = [8e4, 0, 0]
        r = self._report("fast", m)
        self.assertOnly(r, {("velocities", "PartType1", "WARN")})
        self.assertIn("index 5", r.find("velocities", "PartType1")[0].examples[0])

    def test_masstable_and_masses_both_set(self):
        r = self._report("both_mass", mass_table=[1e-4, 5e-4, 0, 0, 0, 0])
        self.assertOnly(r, {("mass", "PartType0", "WARN")})

    def test_type_without_any_mass(self):
        r = self._report("no_mass", mass_table=[0, 0, 0, 0, 0, 0])
        self.assertOnly(r, {("mass", "PartType1", "FAIL")})

    def test_cosmological_mass_budget(self):
        h, om = 0.7, 0.3
        unit_l = 3.085678e21
        v = (BOX * unit_l) ** 3                       # comoving, code length
        # In code units (mass per h, length per h) M = Ω₀ ρ_crit(h = 1) V: h cancels
        rho = 3 * GADGET_HUBBLE ** 2 / (8 * np.pi * GADGET_G)
        m_code = om * rho * v / 1.989e43 / N_DM

        def m(p):
            del p[0]
        kw = dict(cosmological=True, hubble=h, omega0=om, omega_lambda=1 - om, time=0.5,
                  extra_params={})
        ok = self._report("cosmo_ok", m, mass_table=[0, m_code, 0, 0, 0, 0], **kw)
        self.assertOnly(ok, set())
        self.assertIn("1.00000 ×", ok.find("mass", "all types")[0].detail)
        off = self._report("cosmo_off", m, mass_table=[0, 1.05 * m_code, 0, 0, 0, 0], **kw)
        self.assertOnly(off, {("mass", "all types", "WARN")})


def _true_centre(vel=True):
    return Centre(pos=np.zeros(3), method="true", n_used=0, vel=np.zeros(3) if vel else None)


def _shell_density(mass_fn, lo, hi, *args):
    return (mass_fn(hi, *args) - mass_fn(lo, *args)) / (4 / 3 * np.pi * (hi ** 3 - lo ** 3))


def _pulls(z):
    z = np.asarray(z, float)
    return z[np.isfinite(z)]


class TestRadialProfiles(unittest.TestCase):
    M, A, N, SEEDS = 1e5, 1.0, 50_000, 8

    def _plummer(self, seed):
        pos, vel, m = plummer(self.N, self.M, self.A, seed=seed)
        return ParticleSet(pos=pos, mass=m, vel=vel, n_total=self.N)

    def assertPulls(self, z, label, bias=0.15, spread=(0.6, 1.25), within2=0.90):
        z = _pulls(z)
        msg = f"{label}: n={len(z)} mean={z.mean():.3f} std={z.std():.3f}"
        self.assertGreater(len(z), 50, msg)
        self.assertLess(abs(z.mean()), bias, msg)
        self.assertTrue(spread[0] < z.std() < spread[1], msg)
        self.assertGreater(np.mean(np.abs(z) < 2), within2, msg)

    def test_plummer_density_sigma_and_isotropy(self):
        zr, zs, zb = [], [], []
        for seed in range(self.SEEDS):
            r = radial_profile(self._plummer(seed), _true_centre(), n_bins=20,
                               r_min=0.05, r_max=10.0)
            lo, hi = r["r_lo"].value, r["r_hi"].value
            zr += list((r["density"].value - _shell_density(plummer_mass, lo, hi, self.M, self.A))
                       / r["density_err"].value)
            rr = np.linspace(lo, hi, 200).T
            w = plummer_density(rr, self.M, self.A) * rr ** 2
            s_true = np.sqrt((w * plummer_sigma(rr, self.M, self.A) ** 2).sum(1) / w.sum(1))
            zs += list((r["sigma_r"].value - s_true) / r["sigma_r_err"].value)
            zb += list(r["beta"] / r["beta_err"])
            np.testing.assert_allclose(r["mass_enclosed"].value[-1],
                                       plummer_mass(10.0, self.M, self.A), rtol=0.02)
            self.assertTrue(r.provenance.is_science)
        self.assertPulls(zr, "Plummer density")
        self.assertPulls(zs, "Plummer sigma_r")
        # β errors are conservative (they ignore the σ_r–σ_t correlation)
        self.assertPulls(zb, "Plummer beta = 0", spread=(0.5, 1.25))

    def test_hernquist_density(self):
        z = []
        for seed in range(self.SEEDS):
            pos, m = hernquist(self.N, self.M, self.A, seed=seed)
            p = ParticleSet(pos=pos, mass=m, n_total=self.N)
            r = radial_profile(p, _true_centre(vel=False), n_bins=20, r_min=0.01, r_max=50.0)
            lo, hi = r["r_lo"].value, r["r_hi"].value
            z += list((r["density"].value - _shell_density(hernquist_mass, lo, hi, self.M, self.A))
                      / r["density_err"].value)
            self.assertTrue(np.isnan(r["beta"]).all())
        self.assertPulls(z, "Hernquist density")

    def test_planted_anisotropy(self):
        z = []
        for seed in range(self.SEEDS):
            pos, _, m = plummer(self.N, self.M, self.A, seed=seed)
            vel = anisotropic_velocities(pos, 100.0, 50.0, seed=seed + 100)
            p = ParticleSet(pos=pos, mass=m, vel=vel, n_total=self.N)
            r = radial_profile(p, _true_centre(), bins="equal_count", n_bins=10)
            self.assertTrue(np.all(np.abs(np.diff(r["n"])) <= 1), r["n"])
            z += list((r["beta"] - 0.75) / r["beta_err"])
        self.assertPulls(z, "planted beta = 0.75", spread=(0.5, 1.25))

    def test_surface_density_of_projected_plummer(self):
        z = []
        for seed in range(self.SEEDS):
            r = surface_density(self._plummer(seed), _true_centre(), n_bins=20,
                                R_min=0.05, R_max=10.0)
            lo, hi = r["R_lo"].value, r["R_hi"].value
            cum = lambda R: self.M * R ** 2 / (R ** 2 + self.A ** 2)   # projected M(<R)
            true = (cum(hi) - cum(lo)) / (np.pi * (hi ** 2 - lo ** 2))
            z += list((r["surface_density"].value - true) / r["surface_density_err"].value)
        self.assertPulls(z, "Plummer surface density")

    def test_resolution_radius_from_softening(self):
        r = radial_profile(self._plummer(0), _true_centre(), n_bins=20, r_min=0.05,
                           r_max=10.0, softening=0.1)
        self.assertAlmostEqual(r["r_resolved"].value, 0.28)
        np.testing.assert_array_equal(r["resolved"], r["r_lo"].value >= 0.28)


def _plummer_W(M, a):
    return -3 * np.pi * G_KPC_KMS2_MSUN * M ** 2 / (32 * a)     # Literature (from memory)


class TestEnergy(unittest.TestCase):
    M, A = 1e5, 1.0

    def test_spline_kernel(self):
        eps = 0.5
        h = 2.8 * eps
        for q in (0.5, 1.0):
            lo, hi = spline_kernel([q * h * (1 - 1e-9), q * h * (1 + 1e-9)], eps)
            self.assertAlmostEqual(lo, hi, places=6)
        self.assertAlmostEqual(spline_kernel([0.0], eps)[0], 1 / eps)
        r = np.array([h, 2 * h, 10.0])
        np.testing.assert_allclose(spline_kernel(r, eps), 1 / r)
        np.testing.assert_allclose(spline_kernel(r, 0.0), 1 / r)
        g = spline_kernel(np.linspace(1e-6, h, 1000), eps)          # monotone, below 1/r
        self.assertTrue(np.all(np.diff(g) < 0))
        self.assertTrue(np.all(g <= 1 / np.linspace(1e-6, h, 1000) + 1e-12))

    def test_pairs_agree_with_direct(self):
        z = []
        for seed in range(6):
            pos, _, m = plummer(5000, self.M, self.A, seed=seed)
            eps = np.where(np.arange(5000) % 2, 0.05, 0.2)      # mixed softening
            wd, _, _ = potential_energy(pos, m, eps, "direct")
            wp, ep, _ = potential_energy(pos, m, eps, "pairs", n_pairs=1_000_000, seed=seed)
            z.append((wp - wd) / ep)
        z = np.array(z)
        self.assertTrue(np.all(np.abs(z) < 3), z)

    def test_plummer_virial_ratio_at_1e5(self):
        """Plan acceptance: Plummer Q = 0.50 ± 0.01 at N = 10⁵."""
        pos, vel, m = plummer(100_000, self.M, self.A, seed=3)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        code_mass = 1.989e43 / 1.98840987e33
        path = write_gadget4_snapshot(Path(tmp.name) / "snapshot_000.hdf5", {
            1: dict(pos=pos, vel=vel, mass=m / code_mass, ids=np.arange(1, 100_001))})
        r = energy_budget(path)
        self.assertEqual(r.provenance.parameters["potential"], "pairs")
        self.assertTrue(r.is_science)
        self.assertLess(abs(r["Q"] - 0.5), 0.01, (r["Q"], r["Q_err"]))
        self.assertLess(r["Q_err"], 0.003)
        W_th = _plummer_W(self.M, self.A)
        # finite-N scatter of W about the model is ~1 / √N; allow 3 % plus 3σ
        self.assertLess(abs(r["W"].value / W_th - 1), 0.03 + 3 * r["W_err"].value / abs(W_th))
        np.testing.assert_allclose(r["K"].value, 0.5 * np.sum(m * np.einsum("ij,ij->i", vel, vel)),
                                   rtol=1e-9)        # masses pass through code units

    def test_drift_over_a_synthetic_run(self):
        """Same particles in three snapshots: rotated (E unchanged) and with
        velocities raised 1 % (ΔK = 2.01 % of K). Common pairs make ΔW exact."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pos, vel, m = plummer(40_000, self.M, self.A, seed=7)
        code_mass = 1.989e43 / 1.98840987e33
        th = 0.3
        rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        ids = np.arange(1, 40_001)
        perm = np.random.default_rng(0).permutation(40_000)    # file order differs
        snaps = [(pos, vel), (pos @ rot.T, vel @ rot.T), (pos @ rot.T, 1.01 * vel @ rot.T)]
        for k, (x, v) in enumerate(snaps):
            sel = perm if k else np.arange(40_000)
            write_gadget4_snapshot(Path(tmp.name) / f"snapshot_{k:03d}.hdf5", {
                1: dict(pos=x[sel], vel=v[sel], mass=m[sel] / code_mass, ids=ids[sel])},
                time=0.1 * k)
        table, drift = conservation(Run.from_folder(tmp.name), potential="pairs",
                                    n_pairs=1_000_000)
        d = drift[next(iter(drift))]
        K0 = 0.5 * np.sum(m * np.einsum("ij,ij->i", vel, vel))
        E0 = d["E"][0]
        self.assertLess(abs(d["dE_rel"][1]), 1e-9)          # same pairs: W identical
        np.testing.assert_allclose(d["dE_rel"][2] * abs(E0), 0.0201 * K0, rtol=1e-6)
        L = np.array([[table.series("L", i)[next(iter(drift))][2][k] for i in range(3)]
                      for k in range(3)])
        np.testing.assert_allclose(L[1], rot @ L[0], rtol=1e-9)    # L turns with the system
        np.testing.assert_allclose(d["dL_rel"][1], np.linalg.norm(rot @ L[0] - L[0])
                                   / np.linalg.norm(L[0]), rtol=1e-9)
        self.assertLess(d["com_shift_kpc"][1], 0.01)
        self.assertEqual(table.failures, [])

    @unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
    def test_sfr_snapshot_000_equals_energy_txt(self):
        path = Path(SFR) / "snapshot_000.hdf5"
        etxt = Path(SFR) / "energy.txt"
        if not (path.exists() and etxt.exists()):
            self.skipTest("snapshot_000 or energy.txt not staged")
        view = SnapshotView(path)
        r = energy_budget(view, potential="none")
        e = read_energy_txt(etxt)
        self.assertEqual(e["time"][0], view.time_info()["code"])
        cu = code_energy_unit(view)
        types = list(r["types"])
        # energy.txt keeps 8 significant digits: rounding is up to 5e-8 relative
        np.testing.assert_allclose(r["K"].value / cu, e["kinetic"][0], rtol=6e-8)
        np.testing.assert_allclose(r["U"].value / cu, e["thermal"][0], rtol=6e-8)
        np.testing.assert_allclose(r["K_type"].value / cu, e["kinetic_type"][0, types], rtol=6e-8)
        mu = 1.989e43 / 1.98840987e33
        np.testing.assert_allclose(r["mass_type"].value / mu, e["mass_type"][0, types], rtol=2e-7)

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_dm_kinetic_energy_matches_energy_txt(self):
        """energy.txt K is ½ Σ m v_pec² in code units (inferred: the rows are not
        at snapshot times, so the file is interpolated in log K – log a)."""
        etxt = Path(DM) / "energy.txt"
        if not etxt.exists():
            self.skipTest("energy.txt not staged")
        e = read_energy_txt(etxt)
        for name in ("snapshot_000.hdf5", "snapshot_001.hdf5"):
            path = Path(DM) / name
            if not path.exists():
                continue
            view = SnapshotView(path)
            r = energy_budget(view, potential="none")
            self.assertTrue(np.isnan(r["W"].value))
            k_file = np.exp(np.interp(np.log(view.a), np.log(e["time"]), np.log(e["kinetic"])))
            np.testing.assert_allclose(r["K"].value / code_energy_unit(view), k_file, rtol=1e-3)


class TestResultsCache(unittest.TestCase):
    """Plan F3 acceptance: a 3-run synthetic set gives the same values cached and
    uncached, and touching one file recomputes only its rows."""

    def test_cached_equals_uncached_and_touch_recomputes_one(self):
        from measurements.cache import ResultCache
        from measurements.runset import RunSet
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "runs"
        code_mass = 1.989e43 / 1.98840987e33
        for k in range(3):
            d = root / f"run{k}" / "output"
            d.mkdir(parents=True)
            for s_ in range(2):
                pos, vel, m = plummer(3000, 1e5, 1.0 + 0.1 * k, seed=10 * k + s_)
                write_gadget4_snapshot(d / f"snapshot_{s_:03d}.hdf5", {
                    1: dict(pos=pos, vel=vel, mass=m / code_mass, ids=np.arange(1, 3001))},
                    time=0.1 * s_)
        rs = RunSet.from_folder(str(root))
        self.assertEqual(len(rs.runs), 3)
        cache = ResultCache(Path(tmp.name) / "cache")
        kw = dict(potential="pairs", n_pairs=200_000)
        plain, _ = conservation(rs, **kw)
        first, _ = conservation(rs, cache=cache, **kw)
        self.assertEqual((cache.hits, cache.misses), (0, 6))
        again, _ = conservation(rs, cache=cache, **kw)
        self.assertEqual((cache.hits, cache.misses), (6, 6))
        key = lambda t: sorted((r["run"], r["snapshot"], r["quantity"], r["index"], r["value"])
                               for r in t.rows)
        self.assertEqual(key(plain), key(first))
        self.assertEqual(key(plain), key(again))
        # touching one snapshot recomputes only that one
        f = root / "run1" / "output" / "snapshot_001.hdf5"
        st = os.stat(f)
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        conservation(rs, cache=cache, **kw)
        self.assertEqual((cache.hits, cache.misses), (11, 7))
        # different arguments are a different result
        conservation(rs, cache=cache, potential="pairs", n_pairs=100_000)
        self.assertEqual(cache.misses, 13)


class TestPhaseDiagram(unittest.TestCase):
    """Plan S5 acceptance: weighted 2-D histogram totals equal the total mass, and
    the ρ–T diagram of CollidingGalaxiesSFR equals an independent h5py computation."""

    def test_weights_are_accounted_for(self):
        from measurements.phase import property_histogram2d
        parts = _clean_parts()
        parts[0]["fields"]["Density"][:3] = [0.0, -1.0, np.nan]      # dropped on a log axis
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = _write(Path(tmp.name) / "phase.hdf5", parts)
        r = property_histogram2d(path, 0, "Density", "InternalEnergy", bins=40,
                                 y_range=(0.0, 1.8))                  # cuts some particles off
        v = r.values
        self.assertEqual(v["n_dropped"], 3)
        self.assertAlmostEqual(v["in_range"] + v["out_of_range"] + v["dropped"], v["total"],
                               delta=1e-9 * v["total"])
        self.assertGreater(v["out_of_range"], 0)
        self.assertAlmostEqual(v["grid"].sum(), v["in_range"], delta=1e-9 * v["total"])
        mass = SnapshotView(path).particles([0]).mass
        self.assertAlmostEqual(v["total"], mass.sum(), delta=1e-9 * mass.sum())
        counts = property_histogram2d(path, 0, "Density", "Velocities|·|", weight="count", bins=10)
        self.assertEqual(counts.values["in_range"] + counts.values["dropped"], N_GAS)

    @unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
    def test_rho_T_equals_independent_h5py(self):
        from measurements.phase import property_histogram2d
        path = Path(SFR) / "snapshot_100.hdf5"
        r = property_histogram2d(path, 0, "Density", "Temperature", bins=80,
                                 x_range=(-34, -21), y_range=(1, 8))
        v = r.values
        with h5py.File(path) as f:
            g = f["PartType0"]
            rho = g["Density"][:] * g["Density"].attrs["to_cgs"]
            u = g["InternalEnergy"][:] * g["InternalEnergy"].attrs["to_cgs"]
            mu = 4 / (1 + 3 * 0.76 + 4 * 0.76 * g["ElectronAbundance"][:])
            T = (2 / 3) * mu * 1.67262192369e-24 * u / 1.380649e-16
            m = g["Masses"][:] * g["Masses"].attrs["to_cgs"] / 1.988409870698051e33
        H, _, _ = np.histogram2d(np.log10(rho), np.log10(T), bins=[v["x_edges"], v["y_edges"]],
                                 weights=m)
        np.testing.assert_allclose(v["grid"], H.T, rtol=1e-12, atol=0)
        self.assertEqual(v["out_of_range"], 0.0)
        np.testing.assert_allclose(v["in_range"], m.sum(), rtol=1e-12)


class TestSanityRealData(unittest.TestCase):
    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_dm_snapshot_002_edge_particles(self):
        path = Path(DM) / "snapshot_002.hdf5"
        if not path.exists():
            self.skipTest("snapshot_002.hdf5 not staged")
        r = sanity_report(path)
        self.assertOnly = TestSanityPlantedDefects.assertOnly.__get__(self)
        self.assertOnly(r, {("box", "PartType1", "WARN")})
        c = r.find("box", "PartType1")[0]
        self.assertEqual(c.count, 2)
        # independent read
        with h5py.File(path) as f:
            x = f["PartType1/Coordinates"][:]
            ids = f["PartType1/ParticleIDs"][:]
            edge = np.flatnonzero((x == f["Header"].attrs["BoxSize"]).any(axis=1))
        self.assertEqual(c.examples, [f"index {i} (ID {ids[i]})" for i in edge])
        self.assertIn("1.00000 ×", r.find("mass", "all types")[0].detail)

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_dm_snapshot_000_clean(self):
        r = sanity_report(Path(DM) / "snapshot_000.hdf5")
        self.assertEqual(r.worst, "PASS", r.to_text())


if __name__ == "__main__":
    unittest.main()
