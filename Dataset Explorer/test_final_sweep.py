"""
Regression tests for the final bug sweep of the Dataset Explorer (2026-10-27).

Each test failed on the code before the sweep and passes after it:

1. In the Comoving frame, particle sets carried Gadget's stored velocity
   w = v/√a instead of a velocity in km/s, and v_circ used the comoving radius,
   so profiles and dispersions in Comoving mode were not physical km/s.
2. Energy over run listed snapshots only in the open file's folder, so a
   series written into snapdir_NNN/ folders gave one snapshot.
3. The solar Ne marker was 1.40e-3, about 10 % above Asplund et al. (2009)
   log ε(Ne) = 7.93; the other elements' markers follow that table.
4. Max Temperature called the highest temperature ever reached the gas's
   "cold / warm / hot" phase.
5. The toolbar's "?" button was not connected to anything.

Synthetic files only; no environment variables needed.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from measurements.models import write_gadget4_snapshot  # noqa: E402


def _cosmological(path, a=0.5, n=3000, h=0.7):
    rng = np.random.default_rng(3)
    r = 0.05 * rng.pareto(2.5, n)                       # a concentrated clump, code units (Mpc/h)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    pos = 25.0 + d * r[:, None]
    vel = rng.normal(0.0, 200.0, (n, 3))                # stored w
    part = dict(pos=pos, vel=vel, mass=np.full(n, 1e-3), ids=np.arange(1, n + 1))
    return write_gadget4_snapshot(str(path), {1: part}, time=a, box=50.0, cosmological=True,
                                  hubble=h, omega0=0.3, omega_lambda=0.7,
                                  unit_length_cm=3.085678e24)


# ── 1. comoving-frame velocities and v_circ ──────────────────────────────────

def test_comoving_frame_velocities_are_physical_km_s(tmp_path):
    from measurements import SnapshotView
    a = 0.5
    view = SnapshotView(_cosmological(tmp_path / "c.hdf5", a=a))
    phys = view.particles([1], frame="physical")
    com = view.particles([1], frame="comoving")
    assert np.allclose(com.pos * a, phys.pos, rtol=1e-12)            # lengths differ by a
    assert np.allclose(com.vel, phys.vel, rtol=1e-12)                # velocities do not
    raw = view.field_cgs(1, "Velocities", "physical") / 1e5
    assert np.allclose(phys.vel, raw, rtol=1e-12)                    # √a w, in km/s


def test_comoving_profile_has_the_physical_circular_velocity(tmp_path):
    from measurements import SnapshotView, find_centre
    from measurements.profiles import radial_profile
    view = SnapshotView(_cosmological(tmp_path / "c.hdf5", a=0.5))
    out = {}
    for frame in ("physical", "comoving"):
        p = view.particles([1], frame=frame)
        out[frame] = radial_profile(p, find_centre(p, "com"), n_bins=12)
    vp, vc = out["physical"]["v_circ"].value, out["comoving"]["v_circ"].value
    ok = np.isfinite(vp) & np.isfinite(vc)
    assert ok.sum() >= 8
    assert np.allclose(vc[ok], vp[ok], rtol=1e-6)
    sp, sc = out["physical"]["sigma_r"].value, out["comoving"]["sigma_r"].value
    ok = np.isfinite(sp) & np.isfinite(sc)
    assert np.allclose(sc[ok], sp[ok], rtol=1e-6)


# ── 2. snapshots in snapdir_NNN folders ─────────────────────────────────────

def test_energy_over_run_finds_snapdir_series(tmp_path):
    from measurements.runset import discover_snapshots
    for k in range(3):
        d = tmp_path / f"snapdir_{k:03d}"
        d.mkdir()
        for piece in range(2):
            (d / f"snapshot_{k:03d}.{piece}.hdf5").write_bytes(b"")
    (tmp_path / "fof_tab_000.hdf5").write_bytes(b"")
    found = discover_snapshots(str(tmp_path / "snapdir_001"))       # the open file's folder
    assert [n for n, _ in found] == [0, 1, 2]
    assert all(p.endswith(".0.hdf5") for _, p in found)
    assert discover_snapshots(str(tmp_path)) == found


def test_single_folder_series_still_listed(tmp_path):
    from measurements.runset import discover_snapshots
    for k in (0, 1):
        (tmp_path / f"snapshot_{k:03d}.hdf5").write_bytes(b"")
    (tmp_path / "bak-snapshot_000.hdf5").write_bytes(b"")
    assert [n for n, _ in discover_snapshots(str(tmp_path))] == [0, 1]


# ── 3. solar Ne marker ──────────────────────────────────────────────────────

def test_solar_neon_marker_follows_asplund_2009():
    import abundance_neon_plugin as ne
    import abundance_oxygen_plugin as ox
    x_sun = 0.7381
    mass_fraction = lambda log_eps, a_mass: x_sun * a_mass * 10 ** (log_eps - 12)
    # the oxygen marker shows how the markers were made from the table
    assert ox._SOLAR_O == pytest.approx(mass_fraction(8.69, 15.999), rel=0.02)
    assert ne._SOLAR_NE == pytest.approx(mass_fraction(7.93, 20.180), rel=0.02)


# ── 4. Max Temperature wording ──────────────────────────────────────────────

def test_max_temperature_info_is_thermal_history_not_phase():
    from max_temperature_plugin import MaxTemperaturePlugin
    info = MaxTemperaturePlugin().get_custom_info(np.array([1e3, 2e4, 1e6, 3e6]))
    assert "Cold" not in info and "phase" in info
    assert "Never above 10⁴ K: 25.0%" in info and "Ever above 5×10⁵ K: 50.0%" in info


# ── 5. the "?" button ───────────────────────────────────────────────────────

def test_help_button_opens_help(monkeypatch):
    from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton
    app = QApplication.instance() or QApplication([])
    from plot_panel import ViewToolbar
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    tb = ViewToolbar()
    btn = next(b for b in tb.findChildren(QPushButton) if b.objectName() == "help_btn")
    assert "not yet connected" not in btn.toolTip()
    btn.click()
    assert shown and "SAMPLE 1-in-K" in shown[0][2]
    tb.deleteLater()
    app.processEvents()
