"""
Multi-file and EAGLE-style HDF5 snapshots.

EAGLE writes one snapshot as <base>.<i>.hdf5 pieces, names the arrays
Velocity and Mass, keeps sub-groups such as ElementAbundance inside
PartType0, and stores units in a Units group.
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from snapshot_io import snapshot_loader as sl  # noqa: E402
from lod.store import id_uniforms  # noqa: E402
from model.units import GadgetUnits  # noqa: E402
from gadget_file_discovery import SnapshotInspector  # noqa: E402

MPC_IN_CM = 3.085678e24
EAGLE_SNAP = os.environ.get("COSMOVIEWER_EAGLE_SNAPSHOT")   # path to a snap_*.0.hdf5


def _write_eagle_like(folder, nfiles=4, n_gas=300, n_dm=400, n_star=50, dm_mass=0.97, seed=0):
    """Returns the full (undecimated) arrays in loader order: type, then piece."""
    rng = np.random.default_rng(seed)
    per_type = {0: n_gas, 1: n_dm, 4: n_star}
    chunks = {t: np.array_split(np.arange(n), nfiles) for t, n in per_type.items()}
    full = {t: {'c': rng.random((n, 3)) * 16.94, 'v': rng.normal(0, 50, (n, 3)).astype(np.float32),
                'id': np.arange(n, dtype=np.uint64) + 10_000 * t,
                'm': np.full(n, dm_mass) if t == 1 else rng.random(n).astype(np.float32)}
            for t, n in per_type.items()}
    totals = np.zeros(6, dtype=np.uint32)
    for t, n in per_type.items():
        totals[t] = n
    for i in range(nfiles):
        with h5py.File(folder / f"snap_028_z000p000.{i}.hdf5", "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = 16.94
            h.attrs["Time"] = 1.0
            h.attrs["ExpansionFactor"] = 1.0
            h.attrs["Redshift"] = 0.0
            h.attrs["HubbleParam"] = 0.6777
            h.attrs["NumFilesPerSnapshot"] = nfiles
            h.attrs["MassTable"] = np.array([0, dm_mass, 0, 0, 0, 0], dtype=np.float64)
            h.attrs["NumPart_Total"] = totals
            h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
            this = np.zeros(6, dtype=np.int32)
            u = f.create_group("Units")
            u.attrs["UnitLength_in_cm"] = MPC_IN_CM
            u.attrs["UnitMass_in_g"] = 1.989e43
            u.attrs["UnitVelocity_in_cm_per_s"] = 1e5
            p = f.create_group("Parameters")
            p.create_group("ChemicalElements").attrs["BG_NELEMENTS"] = 9
            for t in per_type:
                rows = chunks[t][i]
                this[t] = len(rows)
                g = f.create_group(f"PartType{t}")
                ds = g.create_dataset("Coordinates", data=full[t]['c'][rows], chunks=True, compression="gzip")
                ds.attrs["h-scale-exponent"] = -1.0
                ds.attrs["aexp-scale-exponent"] = 1.0
                g["Velocity"] = full[t]['v'][rows]
                g["ParticleIDs"] = full[t]['id'][rows]
                if t != 1:
                    g["Mass"] = full[t]['m'][rows]
                if t == 0:
                    ea = g.create_group("ElementAbundance")
                    ea["Hydrogen"] = np.full(len(rows), 0.75, np.float32)
            h.attrs["NumPart_ThisFile"] = this
    order = [0, 1, 4]
    return {k: np.concatenate([full[t][k] for t in order]) for k in ('c', 'v', 'id', 'm')}, \
        np.concatenate([np.full(per_type[t], t) for t in order])


def _files(folder):
    return sorted(folder.glob("snap_*.hdf5"))


def test_pieces_are_one_snapshot(tmp_path):
    _write_eagle_like(tmp_path, nfiles=4)
    sets, errors = sl.group_snapshot_files(_files(tmp_path))
    assert errors == []
    assert len(sets) == 1
    assert [p.name for p in sets[0]] == [f"snap_028_z000p000.{i}.hdf5" for i in range(4)]


def test_missing_piece_is_reported_not_loaded(tmp_path):
    _write_eagle_like(tmp_path, nfiles=4)
    (tmp_path / "snap_028_z000p000.2.hdf5").unlink()
    sets, errors = sl.group_snapshot_files(_files(tmp_path))
    assert sets == []
    assert len(errors) == 1 and "missing 1 of 4" in errors[0]


@pytest.mark.parametrize("max_particles", [10_000, 100, 37])
def test_full_and_decimated_reads_match_reference(tmp_path, max_particles):
    full, types = _write_eagle_like(tmp_path, nfiles=4)
    d = sl.load_hdf5_snapshot(_files(tmp_path), 0, max_particles)
    assert "error" not in d, d.get("error")
    n = len(types)
    # the display sample: particles whose ParticleID hash u is below max_particles / n (lod/store.py)
    keep = id_uniforms(full['id'], 0) < max_particles / n if n > max_particles else np.ones(n, bool)
    np.testing.assert_array_equal(d['id'], full['id'][keep])
    np.testing.assert_allclose(d['c'], full['c'][keep])
    np.testing.assert_allclose(d['v'], full['v'][keep])
    np.testing.assert_allclose(d['m'], full['m'][keep], rtol=1e-6)
    np.testing.assert_array_equal(d['t'], types[keep])
    assert d['n_total'] == n and d['step'] == pytest.approx(max(n / max_particles, 1.0))
    assert len(d['files']) == 4


def test_eagle_units_and_scaling(tmp_path):
    _write_eagle_like(tmp_path)
    d = sl.load_hdf5_snapshot(_files(tmp_path), 0, 10_000)
    units = GadgetUnits(d['unit_source'])
    assert units.UnitLength_in_cm == pytest.approx(MPC_IN_CM)
    assert units.HubbleParam == pytest.approx(0.6777)
    assert d['coord_scaling']['h-scale-exponent'] == -1.0


def test_inspector_counts_whole_snapshot(tmp_path):
    _write_eagle_like(tmp_path, nfiles=4, n_gas=300, n_dm=400, n_star=50)
    info = SnapshotInspector.inspect(tmp_path / "snap_028_z000p000.0.hdf5")
    assert [i.count for i in info] == [300, 400, 0, 0, 50, 0]
    assert info[0].has_velocities and info[0].has_masses
    assert info[1].has_velocities and not info[1].has_masses


def test_loader_thread_reads_pieces_in_parallel(tmp_path):
    full, types = _write_eagle_like(tmp_path, nfiles=4)
    loader = sl.AsyncSnapshotLoader(_files(tmp_path), max_particles=100)
    got = {}
    loader.finished.connect(lambda snaps, units, out: got.update(snaps=snaps, units=units))
    loader.run()
    assert loader.errors == []
    assert loader.n_requested == 1
    assert len(got['snaps']) == 1
    keep = id_uniforms(full['id'], 0) < 100 / len(types)
    np.testing.assert_array_equal(got['snaps'][0]['id'], full['id'][keep])


def test_separate_snapshots_stay_separate(tmp_path):
    for k in range(2):
        sub = tmp_path / f"s{k}"
        sub.mkdir()
        _write_eagle_like(sub, nfiles=2, seed=k)
        for p in sub.iterdir():
            p.rename(tmp_path / p.name.replace("snap_028", f"snap_02{k}"))
    sets, errors = sl.group_snapshot_files(_files(tmp_path))
    assert errors == [] and len(sets) == 2 and all(len(s) == 2 for s in sets)


@pytest.mark.skipif(not EAGLE_SNAP, reason="set COSMOVIEWER_EAGLE_SNAPSHOT to an EAGLE snap_*.0.hdf5")
def test_real_eagle_snapshot():
    files = sorted(Path(EAGLE_SNAP).parent.glob(Path(EAGLE_SNAP).name.split('.')[0] + ".*.hdf5"))
    sets, errors = sl.group_snapshot_files(files)
    assert errors == [] and len(sets) == 1
    d = sl.load_hdf5_snapshot(sets[0], 0, 150_000)
    assert "error" not in d, d.get("error")
    assert 140_000 <= len(d['c']) <= 160_000
    assert set(np.unique(d['t'])) <= {0, 1, 4, 5}
