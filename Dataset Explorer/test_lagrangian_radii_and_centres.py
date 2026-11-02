"""
The measurement layer (measurements/): Lagrangian radii, centres,
SnapshotView frames and units, and run sets.

    python3 -m pytest -q test_lagrangian_radii_and_centres.py

Synthetic tests compare against Plummer models whose answers are known
analytically, with tolerances set by the expected sampling error, not by
what happens to pass. Tests on real Gadget-4 outputs look for them in
COSMOVIEWER_DM_OUTPUT (DM-L50-N128/output) and COSMOVIEWER_SFR_OUTPUT
(CollidingGalaxiesSFR/output) and skip when absent.
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

from measurements import (  # noqa: E402
    Centre, RunSet, SnapshotView, discover_snapshots, find_centre,
    lagrangian_radii, minimum_image)
from measurements.models import (  # noqa: E402
    plummer, plummer_lagrangian_radius, write_gadget4_snapshot)
from measurements.snapshot_view import KPC_CM, MSUN_G  # noqa: E402

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")
SFR = os.environ.get("COSMOVIEWER_SFR_OUTPUT", "")

A = 0.001            # Plummer scale radius, kpc (1 pc)
M = 1.0e5            # Msun
MASS_CODE = 1.989e43 / MSUN_G     # Msun per code mass unit (1.989e43 g)
MPC_CODE = 3.085678e24 / KPC_CM   # kpc per Gadget "Mpc" (1.4e-7 longer than the IAU Mpc)


def _write_plummer(path, n, seed, offset=(0.0, 0.0, 0.0), box=0.0, **kw):
    pos, vel, m = plummer(n, M, A, seed=seed)
    pos = pos + np.asarray(offset)
    if box:
        pos = np.mod(pos, box)
    write_gadget4_snapshot(path, {1: dict(pos=pos, vel=vel, mass=m / MASS_CODE,
                                          ids=np.arange(n))}, box=box, **kw)
    return path


def _origin(pos=(0.0, 0.0, 0.0)):
    return Centre(pos=np.asarray(pos, float), method="known", n_used=0)


class TestPlummerLagrangianRadii(unittest.TestCase):
    """Acceptance: Plummer Lagrangian radii agree with r_f = a/sqrt(f^-2/3 - 1)
    within the expected sampling error at N = 1e4 and 1e5."""

    def _deviations(self, n, seeds):
        devs, ratios = [], []
        with tempfile.TemporaryDirectory() as d:
            for s in seeds:
                p = SnapshotView(_write_plummer(f"{d}/p.hdf5", n, s)).particles()
                r = lagrangian_radii(p, _origin())
                true = plummer_lagrangian_radius(r["fraction"], A)
                devs.append((r["radius"].value - true) / r["radius_err_sampling"].value)
                ratios.append(r["radius"].value / true)
        return np.array(devs), np.array(ratios)

    def test_coverage_n1e4(self):
        devs, ratios = self._deviations(10_000, range(100))
        inside = np.mean(np.abs(devs) < 2.0)
        # 800 deviations; a correct 2σ error gives 95.4% with binomial sd 0.7%
        self.assertGreater(inside, 0.93)
        self.assertLess(inside, 0.98)
        # no bias: mean deviation per fraction consistent with 0 (sd of mean = 0.1)
        self.assertTrue(np.all(np.abs(devs.mean(axis=0)) < 0.35), devs.mean(axis=0))

    def test_coverage_n1e5(self):
        devs, _ = self._deviations(100_000, range(20))
        self.assertGreater(np.mean(np.abs(devs) < 2.0), 0.90)
        self.assertTrue(np.all(np.abs(devs) < 4.0))

    def test_bootstrap_agrees_with_sampling_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = SnapshotView(_write_plummer(f"{d}/p.hdf5", 20_000, 7)).particles()
        r = lagrangian_radii(p, _origin(), n_bootstrap=300)
        ratio = r["radius_err_bootstrap"].value / r["radius_err_sampling"].value
        # bootstrap sd from 300 resamples is itself uncertain by ~4%; the
        # half-width estimate of the sampling error is discrete at small f
        self.assertTrue(np.all((ratio > 0.75) & (ratio < 1.3)), ratio)

    def test_measured_centre_keeps_radii_unbiased(self):
        """A wrong centre inflates Lagrangian radii by about δ²/r. With the
        density centre that bias must stay well inside the sampling error; the
        noisier shrinking-sphere centre is allowed up to 1σ (measured 0.2–0.7σ)."""
        devs = {"density": [], "shrinking_sphere": []}
        with tempfile.TemporaryDirectory() as d:
            for s in range(30):
                p = SnapshotView(_write_plummer(f"{d}/p.hdf5", 10_000, 100 + s,
                                                offset=(3.0, -2.0, 1.0))).particles()
                for m in devs:
                    r = lagrangian_radii(p, find_centre(p, m))
                    true = plummer_lagrangian_radius(r["fraction"], A)
                    devs[m].append((r["radius"].value - true) / r["radius_err_sampling"].value)
        # 30 seeds: standard error of a mean deviation is about 0.18
        mean_d = np.array(devs["density"]).mean(axis=0)
        mean_s = np.array(devs["shrinking_sphere"]).mean(axis=0)
        self.assertTrue(np.all(np.abs(mean_d) < 0.6), mean_d)
        self.assertTrue(np.all(np.abs(mean_s) < 1.0), mean_s)


class TestCentres(unittest.TestCase):
    """Acceptance: centring recovers a planted offset within the expected error."""

    def test_planted_offset(self):
        offset = np.array([12.5, -7.25, 3.0])
        errs = {m: [] for m in ("com", "shrinking_sphere", "density")}
        with tempfile.TemporaryDirectory() as d:
            for s in range(30):
                p = SnapshotView(_write_plummer(f"{d}/p.hdf5", 10_000, 200 + s,
                                                offset=offset)).particles()
                for m in errs:
                    errs[m].append(find_centre(p, m).pos - offset)
        for m, e in errs.items():
            e = np.array(e)
            rms = np.sqrt(np.mean(e ** 2))
            # unbiased: each component's mean within 3 standard errors of 0
            self.assertTrue(np.all(np.abs(e.mean(axis=0)) < 3 * e.std(axis=0, ddof=1) / np.sqrt(len(e))
                                   + 1e-12), (m, e.mean(axis=0)))
            # below half the 1% Lagrangian radius (0.22 a) for every method;
            # measured rms: com 0.04 a, shrinking sphere 0.05 a, density 0.02 a
            self.assertLess(rms, 0.1 * A, m)
            if m == "density":
                self.assertLess(rms, 0.03 * A)

    def test_com_error_matches_theory(self):
        # error of a mass-weighted mean is sigma_x / sqrt(N); a Plummer sphere
        # has infinite <x^2>, so compare against the empirical spread instead
        with tempfile.TemporaryDirectory() as d:
            p = SnapshotView(_write_plummer(f"{d}/p.hdf5", 10_000, 5)).particles()
        c = find_centre(p, "com")
        sigma = p.pos.std(axis=0) / np.sqrt(len(p))
        self.assertTrue(np.all(np.abs(c.pos) < 5 * sigma), (c.pos, sigma))

    def test_periodic_wrap_matches_unwrapped(self):
        box = 10.0       # kpc; a 1 pc system on a corner (farthest particle ~0.2 kpc)
        corner = np.array([0.0, box, 0.0])
        with tempfile.TemporaryDirectory() as d:
            free = SnapshotView(_write_plummer(f"{d}/a.hdf5", 20_000, 9, offset=corner)).particles()
            wrapped = SnapshotView(_write_plummer(f"{d}/b.hdf5", 20_000, 9, offset=corner,
                                                  box=box)).particles()
        self.assertGreater(wrapped.box, 0)
        for m in ("com", "shrinking_sphere", "density"):
            cf, cw = find_centre(free, m), find_centre(wrapped, m)
            diff = minimum_image(cw.pos - cf.pos, wrapped.box)   # box in kpc, not code units
            self.assertLess(np.abs(diff).max(), 1e-9, m)
            rf, rw = lagrangian_radii(free, cf), lagrangian_radii(wrapped, cw)
            np.testing.assert_allclose(rw["radius"].value, rf["radius"].value, rtol=1e-9)

    def test_two_clusters_density_centre_warns(self):
        """Two equal Plummer spheres 100 a apart: the density-weighted mean sits
        between them and must say so; the shrinking sphere finds one of them."""
        p1, v1, m1 = plummer(10_000, M, A, seed=1)
        p2, v2, m2 = plummer(10_000, M, A, seed=2)
        sep = np.array([100 * A, 0, 0])
        with tempfile.TemporaryDirectory() as d:
            path = write_gadget4_snapshot(f"{d}/two.hdf5", {1: dict(
                pos=np.vstack([p1, p2 + sep]), vel=np.vstack([v1, v2]),
                mass=np.concatenate([m1, m2]) / MASS_CODE, ids=np.arange(20_000))})
            p = SnapshotView(path).particles()
        cd = find_centre(p, "density")
        self.assertIn("warning", cd.info)
        cs = find_centre(p, "shrinking_sphere")
        self.assertLess(min(np.linalg.norm(cs.pos), np.linalg.norm(cs.pos - sep)), 0.1 * A)
        single = SnapshotView  # a single cluster gives no warning
        with tempfile.TemporaryDirectory() as d:
            q = single(_write_plummer(f"{d}/one.hdf5", 10_000, 3)).particles()
        self.assertNotIn("warning", find_centre(q, "density").info)

    def test_particle_and_potential_centres(self):
        with tempfile.TemporaryDirectory() as d:
            p = SnapshotView(_write_plummer(f"{d}/p.hdf5", 1000, 1)).particles()
        c = find_centre(p, "particle", particle_id=17)
        np.testing.assert_array_equal(c.pos, p.pos[17])
        pot = np.einsum("ij,ij->i", p.pos, p.pos)
        c = find_centre(p, "potential_min", potential=pot)
        np.testing.assert_array_equal(c.pos, p.pos[np.argmin(pot)])


class TestSnapshotViewSynthetic(unittest.TestCase):
    def test_masstable_and_frames(self):
        n = 1000
        pos, vel, _ = plummer(n, M, A, seed=3)
        a = 0.5
        with tempfile.TemporaryDirectory() as d:
            path = write_gadget4_snapshot(
                f"{d}/c.hdf5", {1: dict(pos=pos + 1.0, vel=vel, mass=None, ids=np.arange(n))},
                time=a, box=10.0, cosmological=True, hubble=0.7, omega0=0.3, omega_lambda=0.7,
                unit_length_cm=3.085678e24, mass_table=[0, 0.01, 0, 0, 0, 0])
            s = SnapshotView(path)
            phys, com = s.particles(frame="physical"), s.particles(frame="comoving")
        # Mpc/h comoving -> kpc: ×1000/h; physical additionally × a
        np.testing.assert_allclose(com.pos, (pos + 1.0) * MPC_CODE / 0.7, rtol=1e-12)
        np.testing.assert_allclose(phys.pos, com.pos * a, rtol=1e-12)
        self.assertAlmostEqual(com.box, 10.0 * MPC_CODE / 0.7, places=6)
        self.assertAlmostEqual(phys.box, 10.0 * MPC_CODE / 0.7 * a, places=6)
        np.testing.assert_allclose(phys.mass, 0.01 * MASS_CODE / 0.7, rtol=1e-12)
        np.testing.assert_allclose(phys.vel, vel * np.sqrt(a), rtol=1e-12)
        self.assertTrue(any("MassTable" in x for x in phys.notes))
        self.assertTrue(phys.time["cosmological"])
        self.assertAlmostEqual(phys.time["z"], 1.0)

    def test_preview_is_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            s = SnapshotView(_write_plummer(f"{d}/p.hdf5", 10_000, 4))
            full, prev = s.particles(), s.particles(max_particles=1000)
        self.assertEqual(len(prev), 1000)
        r_full = lagrangian_radii(full, _origin())
        r_prev = lagrangian_radii(prev, _origin())
        self.assertTrue(r_full.is_science)
        self.assertFalse(r_prev.is_science)
        self.assertIn("PREVIEW", r_prev.provenance.status_line())
        self.assertIn("All 10,000", r_full.provenance.status_line())
        np.testing.assert_array_equal(prev.ids, full.ids[::10])


class TestRunSet(unittest.TestCase):
    def test_two_runs_three_snapshots(self):
        scales = {"runA": 1.0, "runB": 2.0}
        with tempfile.TemporaryDirectory() as d:
            for name, sc in scales.items():
                out = Path(d, name, "output")
                out.mkdir(parents=True)
                for k in range(3):
                    pos, vel, m = plummer(5000, M, A, seed=k)
                    write_gadget4_snapshot(
                        out / f"snapshot_{k:03d}.hdf5",
                        {1: dict(pos=pos * sc * (1 + k), vel=vel, mass=m / MASS_CODE,
                                 ids=np.arange(5000))},
                        time=0.1 * k, extra_params={"ScaleRun": sc})
                # files that are not snapshots
                write_gadget4_snapshot(out / "fof_tab_000.hdf5", {}, time=0.0)
                write_gadget4_snapshot(out / "bak-snapshot_000.hdf5", {}, time=0.0)
            self.assertEqual([k for k, _ in discover_snapshots(f"{d}/runA/output")], [0, 1, 2])
            rs = RunSet.from_folder(d)
            self.assertEqual([r.name for r in rs.runs], ["runA", "runB"])
            self.assertEqual(rs.varying_parameters(), {"ScaleRun": [1.0, 2.0]})

            def measure(snap):
                p = snap.particles()
                return lagrangian_radii(p, _origin(), fractions=[0.5])

            table = rs.apply(measure)
            table.to_csv(f"{d}/out.csv")
            self.assertTrue(Path(d, "out.csv").stat().st_size > 0)
        self.assertEqual(table.failures, [])
        series = table.series("radius", 0)
        for name, sc in scales.items():
            snaps, times, r50 = series[name]
            np.testing.assert_array_equal(snaps, [0, 1, 2])
            np.testing.assert_allclose(times, [0.0, 0.1, 0.2])
            true = plummer_lagrangian_radius(0.5, A) * sc * np.arange(1, 4)
            np.testing.assert_allclose(r50, true, rtol=0.05)

    def test_failures_are_recorded_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            _write_plummer(f"{d}/snapshot_000.hdf5", 100, 0)
            rs = RunSet.from_folder(d)

            def bad(snap):
                raise RuntimeError("boom")

            table = rs.apply(bad)
            self.assertEqual(len(table), 0)
            self.assertEqual(len(table.failures), 1)
            self.assertIn("boom", table.failures[0]["error"])
            with self.assertRaises(RuntimeError):
                rs.apply(bad, on_error="raise")


def _plugin_stage2(vset, group, dataset, coordinates):
    """The value the Histogram tab computes for one dataset (plugin Stage 1-2)."""
    from generic_plugin import make_generic_leaves
    path = f"{group}/{dataset}"
    raw = vset.extract_array(path).data
    leaves = make_generic_leaves(group, dataset, raw.shape, vset.get_dtype(path))
    attrs = {path: vset.get_attributes(path)}
    out = {}
    for label, cls in leaves.items():
        if "|·|" in label:
            continue
        res = cls()._run_stage1_stage2({path: raw}, attrs, vset.global_header,
                                       dict(vset.global_params), {"coordinates": coordinates})
        out[getattr(cls, "COLUMN", None)] = res.cgs_data
    return out


class TestRealOutputs(unittest.TestCase):
    """Real outputs: SnapshotView equals the plugin path on every field (consistency), and the
    DM run matches physical anchors: box size, total mass from Omega0 rho_crit, GroupPos."""

    def _check_all_fields(self, path):
        s = SnapshotView(path)
        checked = 0
        with h5py.File(path, "r") as f:
            groups = [g for g in f if g.startswith("PartType")]
            names = {g: [k for k in f[g] if isinstance(f[g][k], h5py.Dataset)] for g in groups}
        for g in groups:
            t = int(g.replace("PartType", ""))
            for name in names[g]:
                for frame, coords in (("physical", "Physical"), ("comoving", "Comoving")):
                    mine = s.field_cgs(t, name, frame)
                    theirs = _plugin_stage2(s.vset, g, name, coords)
                    for col, arr in theirs.items():
                        ref = mine if col is None else mine[:, col]
                        np.testing.assert_array_equal(ref, arr, err_msg=f"{g}/{name} {frame} {col}")
                        checked += 1
        return checked

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_every_field_dm(self):
        self.assertGreater(self._check_all_fields(f"{DM}/snapshot_000.hdf5"), 0)

    @unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
    def test_every_field_sfr(self):
        self.assertGreater(self._check_all_fields(f"{SFR}/snapshot_100.hdf5"), 0)

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_dm_box_and_total_mass(self):
        from astropy import constants as const, units as u
        s = SnapshotView(f"{DM}/snapshot_000.hdf5")
        p = s.particles(velocities=False, ids=False)
        h, a = 0.678, s.a
        self.assertAlmostEqual(p.box, 50.0 * MPC_CODE / h * a, places=6)
        self.assertAlmostEqual(s.box_size("comoving"), 50.0 * MPC_CODE / h, places=6)
        rho_crit = (3 * (100 * h * u.km / u.s / u.Mpc) ** 2 / (8 * np.pi * const.G)).to(u.Msun / u.Mpc ** 3)
        expected = 0.308 * rho_crit.value * (50.0 / h) ** 3
        # MassTable was set by Gadget-4 from its own constants; agree to 0.1%
        self.assertLess(abs(p.mass.sum() / expected - 1), 1e-3)

    @unittest.skipUnless(DM, "set COSMOVIEWER_DM_OUTPUT")
    def test_fof_centre_of_mass_matches_catalogue(self):
        """Gadget-4's GroupPos is the centre of mass of the group; ours must agree
        for every group, including those that straddle the periodic boundary."""
        s = SnapshotView(f"{DM}/snapshot_000.hdf5")
        p = s.particles(frame="comoving", ids=False)
        with h5py.File(f"{DM}/fof_tab_000.hdf5", "r") as f:
            off = f["Group/GroupOffsetType"][:, 1]
            ln = f["Group/GroupLenType"][:, 1]
            gpos = f["Group/GroupPos"][:].astype(float)
        conv = s.conversion(s.attrs(1, "Coordinates"), "comoving") / 3.085678e21
        worst, straddle = 0.0, 0
        for i in range(len(ln)):
            sub = p.subset(slice(off[i], off[i] + ln[i]))
            if np.ptp(sub.pos, axis=0).max() > p.box / 2:
                straddle += 1
            d = minimum_image(find_centre(sub, "com").pos - gpos[i] * conv, p.box)
            worst = max(worst, np.abs(d).max())
        # GroupPos is float32: at 50 Mpc/h one ulp is 3.8e-6 Mpc/h = 5.6 kpc/1000
        self.assertLess(worst / conv, 4 * np.spacing(np.float32(50.0)))
        self.assertGreater(straddle, 0)   # the periodic case was exercised

    @unittest.skipUnless(SFR, "set COSMOVIEWER_SFR_OUTPUT")
    def test_sfr_stars_and_masstable_types(self):
        s = SnapshotView(f"{SFR}/snapshot_100.hdf5")
        stars = s.particles([4])
        halo = s.particles([1])
        self.assertEqual(len(stars), 455_187)
        self.assertEqual(stars.box, 0.0)
        self.assertTrue(any("PartType4 mass attributes" in n or "PartType0 mass attributes" in n
                            for n in halo.notes))
        self.assertAlmostEqual(halo.time["Gyr"], 1.203125 * 3.085678e21 / 1e5 / 3.15576e16, places=4)
        c = find_centre(stars, "shrinking_sphere")
        r = lagrangian_radii(stars, c)
        self.assertTrue(r.is_science)
        self.assertTrue(np.all(np.diff(r["radius"].value) > 0))


if __name__ == "__main__":
    unittest.main()
