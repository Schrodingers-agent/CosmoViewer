"""The cached ParticleID index gives exactly what the chunked scan of the ID columns gives."""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lod.store as lod_store  # noqa: E402
from snapshot_io import id_index  # noqa: E402
from snapshot_io.id_lookup import locate_ids  # noqa: E402

DM_SNAPSHOT = os.environ.get("COSMOVIEWER_DM_SNAPSHOT")
EAGLE_SNAPSHOT = os.environ.get("COSMOVIEWER_EAGLE_SNAPSHOT")   # first piece of a multi-file snapshot


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lod_store, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(id_index, "MIN_PARTICLES", 0)
    return tmp_path / "cache"


def _write(folder, nfiles=3, seed=0, dup=True):
    """Sparse 64-bit IDs over types 0, 1 and 4 in several files, rows shuffled, a few repeated IDs."""
    rng = np.random.default_rng(seed)
    pool = rng.choice(2**40, 40_000, replace=False).astype(np.uint64) + np.uint64(5_000_000_000)
    files, k = [], 0
    for i in range(nfiles):
        p = Path(folder) / f"snap_000.{i}.hdf5"
        files.append(str(p))
        with h5py.File(p, "w") as f:
            h = f.create_group("Header")
            h.attrs["MassTable"] = np.array([0.0, 0.5, 0, 0, 0, 0])
            h.attrs["NumFilesPerSnapshot"] = nfiles
            for t, n in ((0, 3000), (1, 6000), (4, 500)):
                ids = pool[k:k + n].copy()
                k += n
                if dup and t == 1:
                    ids[-3:] = pool[:3]                # repeats of IDs written earlier (3 per file)
                g = f.create_group(f"PartType{t}")
                g.create_dataset("ParticleIDs", data=ids, chunks=(min(n, 1024),), compression="gzip")
                g.create_dataset("Coordinates", data=rng.random((n, 3)).astype(np.float32))
                g.create_dataset("Velocities", data=rng.random((n, 3)).astype(np.float32))
                if t != 1:
                    g.create_dataset("Masses", data=rng.random(n).astype(np.float32))
    return files, pool[:k]


def _same(a, b):
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_array_equal(a.found, b.found)
    np.testing.assert_array_equal(a.types, b.types)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.vel, b.vel)
    np.testing.assert_array_equal(a.mass, b.mass)
    assert a.duplicates == b.duplicates and a.scaling == b.scaling


@pytest.mark.parametrize("chunk_rows", [37, 1 << 22])
def test_index_equals_scan(tmp_path, cache, chunk_rows):
    files, pool = _write(tmp_path)
    rng = np.random.default_rng(1)
    want = np.concatenate([rng.choice(pool, 2000, replace=False), pool[:3], [1, 2, 2**63]])  # repeats, absent
    scan = locate_ids(files, want, chunk_rows=chunk_rows, use_index=False)
    assert id_index.open_index(files, build=False) is None
    indexed = locate_ids(files, want, chunk_rows=chunk_rows)
    assert id_index.open_index(files, build=False) is not None     # built and cached
    _same(indexed, scan)
    assert scan.duplicates == 9 and (~scan.found).sum() == 3
    _same(locate_ids(files, want, chunk_rows=chunk_rows), scan)    # reopened from the cache


def test_index_is_rebuilt_when_a_file_changes_and_skipped_when_too_big(tmp_path, cache, monkeypatch):
    files, pool = _write(tmp_path, dup=False)
    first = id_index.open_index(files)
    with h5py.File(files[1], "a") as f:
        f["PartType0/ParticleIDs"][0] = 123
    assert id_index.open_index(files, build=False) is None
    _same(locate_ids(files, [123, int(pool[5])]), locate_ids(files, [123, int(pool[5])], use_index=False))
    assert first is not None
    monkeypatch.setattr(lod_store, "CACHE_LIMIT_BYTES", 10)
    assert id_index.open_index(files) is None                      # would not fit the cache limit


@pytest.mark.parametrize("path", [p for p in (DM_SNAPSHOT, EAGLE_SNAPSHOT) if p],
                         ids=lambda p: Path(p).name)
def test_index_equals_scan_on_real_snapshot(path, cache, monkeypatch):
    from snapshot_io.snapshot_loader import group_snapshot_files
    sets, _ = group_snapshot_files([Path(path)] + sorted(Path(path).parent.glob(Path(path).name.split(".")[0] + ".*.hdf5")))
    files = [str(f) for f in sets[0]]
    if id_index.open_index(files, build=False) is None and sum(
            h5py.File(f, "r")["Header"].attrs["NumPart_ThisFile"].sum() for f in files) * 12 > 4e9:
        pytest.skip("index would not fit the cache")
    with h5py.File(files[-1], "r") as f:
        g = next(f[k] for k in ("PartType1", "PartType0", "PartType4") if k in f)
        n = g["ParticleIDs"].shape[0]
        pick = np.unique(np.r_[g["ParticleIDs"][:: max(n // 3000, 1)], g["ParticleIDs"][n - 1]])
    _same(locate_ids(files, pick), locate_ids(files, pick, use_index=False))
