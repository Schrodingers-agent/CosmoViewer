"""
Particle inspector: one particle's stored values, deltas between snapshots and
history, checked against direct h5py reads.

Real-data tests use COSMOVIEWER_G4_OUTPUT (DM-L50-N128/output) and
COSMOVIEWER_CG_OUTPUT (CollidingGalaxiesSFR/output), as the other suites do.
"""

import csv
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from snapshot_io.particle_record import (cosmic_time_gyr, history, locate_rows, read_records,  # noqa: E402
                                         record_delta, snapshot_clock)

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
CG_OUTPUT = os.environ.get("COSMOVIEWER_CG_OUTPUT")


def _write(path, parts, box=10.0, time_=0.5, comoving=1, mass_table=(0, 2.0, 0, 0, 0, 0)):
    """parts: {ptype: dict(ids=..., pos=..., extra datasets...)}."""
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = box
        h.attrs["Time"] = time_
        h.attrs["MassTable"] = np.array(mass_table, float)
        h.attrs["NumFilesPerSnapshot"] = 1
        p = f.create_group("Parameters")
        p.attrs.update(dict(HubbleParam=0.7, ComovingIntegrationOn=comoving, Omega0=0.3, OmegaLambda=0.7,
                            UnitLength_in_cm=3.085678e21, UnitVelocity_in_cm_per_s=1e5,
                            UnitMass_in_g=1.989e43))
        for t, d in parts.items():
            g = f.create_group(f"PartType{t}")
            g["ParticleIDs"] = np.asarray(d["ids"], np.uint64)
            g["Coordinates"] = np.asarray(d["pos"], np.float32)
            g["Coordinates"].attrs.update(a_scaling=1.0, h_scaling=-1.0)
            g["Velocities"] = np.asarray(d.get("vel", np.zeros_like(d["pos"])), np.float32)
            g["Velocities"].attrs.update(a_scaling=0.5, h_scaling=0.0)
            for k, v in d.items():
                if k not in ("ids", "pos", "vel"):
                    g[k] = np.asarray(v)
                    g[k].attrs.update(a_scaling=-3.0, h_scaling=2.0)


def _gas_star_pair(tmp_path):
    """Snapshot A: ID 7 is gas at x = 9.9; snapshot B: ID 7 is a star at x = 0.2 (crossed the box edge)."""
    rng = np.random.default_rng(0)
    n = 50
    ids = np.arange(1, n + 1)
    pos = rng.random((n, 3)) * 10
    pos[6] = [9.9, 5.0, 5.0]
    dens = rng.random(n) + 1
    u = rng.random(n) * 100
    a = tmp_path / "snapshot_000.hdf5"
    _write(a, {0: dict(ids=ids, pos=pos, Masses=np.full(n, 0.1), Density=dens, InternalEnergy=u),
               1: dict(ids=ids + 1000, pos=pos[::-1])}, time_=0.5)
    keep = ids != 7
    pos_b = pos.copy()
    pos_b[6] = [0.2, 5.0, 5.0]
    b = tmp_path / "snapshot_001.hdf5"
    _write(b, {0: dict(ids=ids[keep], pos=pos_b[keep], Masses=np.full(n - 1, 0.1), Density=dens[keep] * 2,
                       InternalEnergy=u[keep]),
               1: dict(ids=ids + 1000, pos=pos[::-1]),
               4: dict(ids=[7], pos=pos_b[6:7], Masses=[0.09], StellarFormationTime=[0.7])}, time_=1.0)
    return str(a), str(b), pos, dens


def test_locate_rows_matches_brute_force_across_files_and_chunks(tmp_path):
    rng = np.random.default_rng(3)
    files, truth, nid = [], {}, 0
    running = np.zeros(6, int)
    for i in range(3):
        parts = {}
        for t in (0, 1, 4):
            n = int(rng.integers(20, 90))
            ids = rng.permutation(np.arange(nid, nid + n)) + 10
            nid += n
            parts[t] = dict(ids=ids, pos=rng.random((n, 3)))
            for r, pid in enumerate(ids):
                truth[int(pid)] = (i, t, r, r + running[t])
            running[t] += n
        fn = tmp_path / f"snap_000.{i}.hdf5"
        _write(fn, parts)
        files.append(str(fn))
    want = rng.choice(list(truth), 40, replace=False).tolist() + [999999]
    keys, fi, pt, row, grow = locate_rows(files, want, chunk_rows=17)
    for k, a, b, c, d in zip(keys, fi, pt, row, grow):
        if int(k) == 999999:
            assert (a, b, c, d) == (-1, -1, -1, -1)
        else:
            assert (a, b, c, d) == truth[int(k)]


def test_record_values_equal_the_file(tmp_path):
    a, _, pos, dens = _gas_star_pair(tmp_path)
    rec = read_records([a], [7, 1003])
    g = rec[7]
    assert g.found and g.ptype == 0
    assert g.values["Density"] == pytest.approx(float(np.float64(dens[6])))
    assert [g.values[f"Coordinates {c}"] for c in "xyz"] == pytest.approx(pos[6].astype(np.float32).tolist())
    assert g.values["Masses"] == pytest.approx(0.1)
    assert "Temperature" in g.values and g.values["Temperature"] > 0
    dm = rec[1003]
    assert dm.ptype == 1 and dm.values["Masses"] == 2.0            # MassTable
    assert "Density" not in dm.values


def test_delta_type_change_and_periodic_displacement(tmp_path):
    a, b, pos, dens = _gas_star_pair(tmp_path)
    ra, rb = read_records([a], [7])[7], read_records([b], [7])[7]
    d = record_delta(rb, ra)
    assert d.type_change == ("gas", "star")
    ch = {c.name: c for c in d.changes}
    # 9.9 -> 0.2 across a 10-wide box is +0.3, not -9.7
    assert ch["Coordinates x"].delta == pytest.approx(0.3, abs=1e-5)
    assert d.displacement == pytest.approx(0.3, abs=1e-5)
    assert ch["Masses"].delta == pytest.approx(-0.01, abs=1e-7)
    assert ch["Masses"].relative == pytest.approx(-0.1, rel=1e-5)
    assert "StellarFormationTime" in d.only_now and "Density" in d.only_before
    # comoving run: dt is cosmic time between a = 0.5 and a = 1
    assert d.time_unit == "Gyr"
    assert d.dt == pytest.approx(cosmic_time_gyr(1.0, 0.3, 0.7, 0.7) - cosmic_time_gyr(0.5, 0.3, 0.7, 0.7))
    assert ch["Masses"].rate == pytest.approx(-0.01 / d.dt, rel=1e-5)


def test_physical_units_use_the_dataset_attributes(tmp_path):
    a, b, pos, dens = _gas_star_pair(tmp_path)
    ra = read_records([a], [7])[7]
    # Coordinates: a^1 h^-1 at a = 0.5, h = 0.7
    assert ra.physical("Coordinates x") == pytest.approx(ra.values["Coordinates x"] * 0.5 / 0.7)
    # Density written with a^-3 h^2
    assert ra.physical("Density") == pytest.approx(ra.values["Density"] * 0.5 ** -3 * 0.7 ** 2)
    assert ra.physical("Temperature") == ra.values["Temperature"]   # derived in K already
    t, v = history([ra, read_records([b], [7])[7]], "Density")
    assert np.isfinite(v[0]) and np.isnan(v[1])                       # a star has no Density


def test_physical_value_of_dimensionless_gadget4_fields(tmp_path):
    """Gadget-4 writes Metallicity, ElectronAbundance and StellarFormationTime without a/h attributes (seen
    in CollidingGalaxiesSFR), so "Physical units" showed "–" for them. They have nothing to scale; any other
    field without the attributes still has no physical value."""
    fn = tmp_path / "snapshot_000.hdf5"
    _write(fn, {4: dict(ids=[5], pos=[[1, 2, 3]], Masses=[0.1])})
    with h5py.File(fn, "a") as f:
        g = f["PartType4"]
        g["Metallicity"] = np.array([0.02]); g["StellarFormationTime"] = np.array([0.3])
        g["Unknown"] = np.array([7.0])                      # no attributes and not known to be dimensionless
    r = read_records([str(fn)], [5])[5]
    assert r.physical("Metallicity") == pytest.approx(0.02) and r.physical("StellarFormationTime") == pytest.approx(0.3)
    assert r.values["Unknown"] == 7.0 and r.physical("Unknown") is None


def test_cosmic_time_against_analytic_and_astropy():
    assert cosmic_time_gyr(1.0, 1.0, 0.0, 1.0) == pytest.approx(2.0 / 3.0 * 977.7922216807891 / 100, rel=1e-5)
    import astropy.cosmology as astropy     # a requirement: the independent check must not skip
    c = astropy.LambdaCDM(H0=67.77, Om0=0.307, Ode0=0.693, Tcmb0=0)
    for a in (0.1, 0.25, 0.5, 1.0):
        assert cosmic_time_gyr(a, 0.307, 0.693, 0.6777) == pytest.approx(c.age(1 / a - 1).value, rel=2e-4)
    c2 = astropy.LambdaCDM(H0=70, Om0=0.3, Ode0=0.5, Tcmb0=0)        # open, curvature term
    assert cosmic_time_gyr(0.5, 0.3, 0.5, 0.7) == pytest.approx(c2.age(1.0).value, rel=2e-4)


def test_non_comoving_clock_uses_code_time(tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    _write(fn, {1: dict(ids=[1], pos=[[0, 0, 0]])}, box=0.0, time_=1.5, comoving=0)
    c = snapshot_clock(str(fn))
    assert not c.comoving and c.redshift is None and c.time == 1.5
    # kpc / (km/s) = 0.9778 Gyr
    assert c.time_gyr == pytest.approx(1.5 * 3.085678e21 / 1e5 / 3.15576e16)


# --------------------------------------------------------------------------- in the app

@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
    c._cancel_record_job()
    for j in c._old_record_jobs:
        j.wait(5000)
    c._reset_tracking()
    c.view.close()


def _wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _load(app, c, folder):
    c._load_directory(Path(folder))
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 300)
    app.processEvents()


def _all_read(c):
    return all((c._inspect_pid, c._snap_key(s)) in c._records for s in c.model.snapshots)


def _row(c, name):
    for r in c.view.inspector_panel.table_rows():
        if r[0] == name:
            return r
    return None


def test_app_inspector_shows_values_deltas_history_and_exports(app, ctl, tmp_path):
    a, b, pos, dens = _gas_star_pair(tmp_path)
    c = ctl
    _load(app, c, tmp_path)
    assert len(c.model.snapshots) == 2
    c.current_snap_idx = 1
    c._update_frame()
    c.inspect_particle(7)
    assert _wait(app, lambda: _all_read(c), 60)
    c._refresh_inspector()
    ip = c.view.inspector_panel
    assert "star" in ip.lbl_header.text() and "gas → star" in ip.lbl_header.text()
    x = _row(c, "Coordinates x")
    assert float(x[1]) == pytest.approx(0.2, abs=1e-6) and float(x[2]) == pytest.approx(0.3, abs=1e-5)
    assert _row(c, "Density")[2] == "gone"
    assert _row(c, "StellarFormationTime")[2] == "new"
    ip.chk_physical.setChecked(True)
    x = _row(c, "Coordinates x")
    assert float(x[1]) == pytest.approx(0.2 * 1.0 / 0.7, rel=1e-5)
    ip.chk_physical.setChecked(False)
    ip.combo_history.setCurrentText("Coordinates x")
    xs, ys = ip.curve.getData()
    assert len(xs) == 2 and ys[0] == pytest.approx(9.9, abs=1e-5)
    out = c._export_inspected(str(tmp_path / "p7.csv"))
    rows = list(csv.reader(open(out)))
    head = rows[1]
    assert rows[2][head.index("type")] == "gas" and rows[3][head.index("type")] == "star"
    assert float(rows[3][head.index("d_Coordinates x")]) == pytest.approx(0.3, abs=1e-5)


def test_only_changed_keeps_new_and_gone_properties(app, ctl, tmp_path):
    """Found following a CollidingGalaxiesSFR gas particle into a star: with "Only changed" ticked the
    table kept "gone" (Density) but dropped "new" (StellarFormationTime), though both are changes."""
    _gas_star_pair(tmp_path)
    c = ctl
    _load(app, c, tmp_path)
    c.current_snap_idx = 1
    c._update_frame()
    c.inspect_particle(7)
    assert _wait(app, lambda: _all_read(c), 60)
    ip = c.view.inspector_panel
    ip.chk_changed.setChecked(True)
    names = [r[0] for r in ip.table_rows()]
    assert _row(c, "StellarFormationTime")[2] == "new" and _row(c, "Density")[2] == "gone"
    assert "Coordinates x" in names and "Coordinates y" not in names     # y and z are the same in both
    ip.chk_changed.setChecked(False)


def test_clear_leaves_nothing_of_the_particle(app, ctl, tmp_path):
    """Found on CollidingGalaxiesSFR: after Clear the history plot still read "Coordinates x" with the old
    particle's axis labels and range, and the History box still offered its properties."""
    _gas_star_pair(tmp_path)
    c = ctl
    _load(app, c, tmp_path)
    c.inspect_particle(7)
    assert _wait(app, lambda: _all_read(c), 60)
    ip = c.view.inspector_panel
    ip.combo_history.setCurrentText("Coordinates x")
    assert ip.plot.getPlotItem().getAxis('left').labelText == "Coordinates x"
    ip.plot.getViewBox().setXRange(100, 200)          # as a wheel zoom over the plot does
    ip.btn_clear.click()
    app.processEvents()
    assert ip.combo_recent.count() == 0 and ip.table.rowCount() == 0
    assert ip.combo_history.count() == 0 and ip.history_property() == ""
    ax = ip.plot.getPlotItem()
    assert ax.getAxis('left').labelText == "" and ax.getAxis('bottom').labelText == ""
    assert all(ip.plot.getViewBox().autoRangeEnabled())
    (x0, x1), _ = ip.plot.getViewBox().viewRange()
    assert x1 < 100                                    # not the old particle's (or the zoomed) range
    c.current_snap_idx = 1
    c._update_frame()
    app.processEvents()
    assert ip.table.rowCount() == 0 and ip.combo_history.count() == 0     # a new snapshot brings nothing back


def test_redshift_zero_and_group_history_as_points(app, ctl, tmp_path):
    """Found following DM-L50 particles: the z = 0 snapshot (Time = 0.9999999999999997) was headed
    "z = 4.441e-16", and the history of "Group (catalogue)" joined group 0 -> 18 -> 1 with lines as if the
    particle had passed through the groups in between."""
    from PyQt6.QtCore import Qt
    _write(tmp_path / "snapshot_000.hdf5", {1: dict(ids=[1, 2], pos=[[1, 1, 1], [2, 2, 2]])}, time_=0.9999999999999997)
    c = ctl
    _load(app, c, tmp_path)
    c.inspect_particle(1)
    assert _wait(app, lambda: _all_read(c), 60)
    ip = c.view.inspector_panel
    assert "z = 0," in ip.lbl_header.text() and "e-16" not in ip.lbl_header.text()
    t = np.array([2.2, 5.9, 13.8])
    ip.set_history(t, np.array([0.0, 18.0, 1.0]), 2, "cosmic time (Gyr)", "Group (catalogue)", joined=False)
    assert ip.curve.opts['pen'].style() == Qt.PenStyle.NoPen and len(ip.curve.getData()[0]) == 3
    ip.set_history(t, np.array([1.0, 2.0, 3.0]), 2, "cosmic time (Gyr)", "Coordinates x")
    assert ip.curve.opts['pen'].style() != Qt.PenStyle.NoPen
    rec = c._record_at(0)
    rec.values['Group (catalogue)'] = 5.0                  # the controller draws a label's history unjoined
    ip.combo_history.addItem('Group (catalogue)'); ip.combo_history.setCurrentText('Group (catalogue)')
    c._show_history(rec)
    assert ip.curve.opts['pen'].style() == Qt.PenStyle.NoPen


def test_pick_makes_the_particle_inspected(app, ctl, tmp_path):
    a, b, pos, dens = _gas_star_pair(tmp_path)
    c = ctl
    _load(app, c, tmp_path)
    c._pick_ids(np.array([1003], dtype=np.uint64), 'new', "ID 1003 (DM)")
    assert c._inspect_pid == 1003
    assert _wait(app, lambda: _all_read(c), 60)
    c._refresh_inspector()
    assert _row(c, "Masses")[1] == "2"
    assert c.view.inspector_panel.combo_recent.itemText(0) == "ID 1003 (DM)"


# --------------------------------------------------------------------------- real outputs

@pytest.mark.skipif(not G4_OUTPUT, reason="set COSMOVIEWER_G4_OUTPUT to DM-L50-N128/output")
def test_real_dm_l50_record_and_delta_match_direct_reads():
    files = [str(Path(G4_OUTPUT) / f"snapshot_00{i}.hdf5") for i in range(3)]
    with h5py.File(files[2], "r") as f:
        ids = f["PartType1/ParticleIDs"][:]
        pick = [int(ids[0]), int(ids[len(ids) // 2]), int(ids[-1])]
    recs = [read_records([fn], pick) for fn in files]
    for fn, rec in zip(files, recs):
        with h5py.File(fn, "r") as f:
            allid = f["PartType1/ParticleIDs"][:]
            for pid in pick:
                r = int(np.flatnonzero(allid == pid)[0])
                c = f["PartType1/Coordinates"][r]
                v = f["PartType1/Velocities"][r]
                got = rec[pid]
                assert got.row == r and got.ptype == 1
                assert [got.values[f"Coordinates {k}"] for k in "xyz"] == pytest.approx(c.tolist())
                assert [got.values[f"Velocities {k}"] for k in "xyz"] == pytest.approx(v.tolist())
                assert got.values["Masses"] == pytest.approx(float(f["Header"].attrs["MassTable"][1]))
    box = 50.0
    for pid in pick:
        d = record_delta(recs[2][pid], recs[1][pid])
        raw = np.array([recs[2][pid].values[f"Coordinates {k}"] - recs[1][pid].values[f"Coordinates {k}"]
                        for k in "xyz"])
        raw -= box * np.round(raw / box)
        assert d.displacement == pytest.approx(float(np.linalg.norm(raw)), rel=1e-9)
        assert d.time_unit == "Gyr" and d.dt > 0


@pytest.mark.skipif(not CG_OUTPUT, reason="set COSMOVIEWER_CG_OUTPUT to CollidingGalaxiesSFR/output")
def test_real_colliding_galaxies_gas_that_became_a_star():
    a, b = str(Path(CG_OUTPUT) / "snapshot_100.hdf5"), str(Path(CG_OUTPUT) / "snapshot_101.hdf5")
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        gas = fa["PartType0/ParticleIDs"][:]
        stars_b = fb["PartType4/ParticleIDs"][:]
        both = np.intersect1d(gas, stars_b)
        assert len(both) > 0
        pid = int(both[0])
        rb = int(np.flatnonzero(stars_b == pid)[0])
        sft = float(fb["PartType4/StellarFormationTime"][rb])
    ra, rbr = read_records([a], [pid])[pid], read_records([b], [pid])[pid]
    d = record_delta(rbr, ra)
    assert d.type_change == ("gas", "star")
    assert rbr.values["StellarFormationTime"] == pytest.approx(sft)
    assert d.time_unit == "code time" and d.dt == pytest.approx(rbr.clock.time - ra.clock.time)
