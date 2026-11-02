"""
A sorted ParticleID index of one snapshot, cached on disk, so particles are
found by binary search instead of by reading every ID column again.

Tracking looks particles up by ID in every snapshot (snapshot_io/id_lookup.py).
Without an index each lookup reads all ParticleIDs columns; on a compressed
multi-gigabyte snapshot that is the slow part. The index holds the IDs sorted
(uint64) and, for each, its place in the snapshot as a global row: rows are
counted over (file piece, type, row), the order locate_ids scans in. Built
once per snapshot (one read of the ID columns and a sort), it lives in the LOD
cache directory (lod/store.py), keyed by the files' paths, sizes and
modification times, and counts against the same size limit with the same
least-recently-used eviction.

Cost: 12 bytes per particle on disk (8 for the ID, 4 for the row), and about
twice that in memory while it is built. Snapshots whose index would not fit in
half the cache limit, or with more than MAX_PARTICLES particles, get no index
and keep the chunked scan. Snapshots below MIN_PARTICLES do not need one.

The index reproduces the scan exactly, including repeated IDs: the first
occurrence in scan order is the one used, and the extra rows are counted.
"""

import hashlib
import json
import os
import shutil
import time
from typing import Optional, Sequence

import h5py
import numpy as np

import lod.store as lod_store

FORMAT_VERSION = 1
MIN_PARTICLES = 1_000_000          # below this a scan takes well under a second
MAX_PARTICLES = int(float(os.environ.get('COSMOVIEWER_ID_INDEX_MAX', 200e6)))  # bounds memory while building
BYTES_PER_PARTICLE = 12
_READ_ROWS = 1 << 22


def _layout(files: Sequence[str]) -> np.ndarray:
    """Rows with ParticleIDs per (piece, type); 0 where a type has no IDs or no Coordinates."""
    counts = np.zeros((len(files), 6), dtype=np.int64)
    for i, fname in enumerate(files):
        with h5py.File(fname, 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is not None and 'ParticleIDs' in g and 'Coordinates' in g:
                    counts[i, t] = g['ParticleIDs'].shape[0]
    return counts


def _key(files: Sequence[str]) -> str:
    h = hashlib.sha1(f"idx{FORMAT_VERSION}".encode())
    for fname in files:
        st = os.stat(fname)
        h.update(f"|{os.path.abspath(fname)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return 'idx-' + h.hexdigest()[:20]


class IDIndex:
    """Sorted IDs and their global rows (memory-mapped), plus the row offsets per (piece, type)."""

    def __init__(self, ids: np.ndarray, rows: np.ndarray, counts: np.ndarray):
        self.ids, self.rows, self.counts = ids, rows, counts
        self.offsets = np.concatenate([[0], np.cumsum(counts.ravel())])   # piece-major, then type

    def lookup(self, targets: np.ndarray):
        """
        For sorted unique ``targets``: found mask, and piece, type, row of each found one
        (first occurrence in scan order), plus the number of extra rows of found IDs.
        """
        lo = np.searchsorted(self.ids, targets, side='left')
        hi = np.searchsorted(self.ids, targets, side='right')
        found = hi > lo
        g = np.asarray(self.rows[lo[found]]).astype(np.int64)  # stable sort: first is the earliest row
        duplicates = int((hi - lo)[found].sum() - found.sum())
        cell = np.searchsorted(self.offsets, g, side='right') - 1
        piece, ptype = np.divmod(cell, 6)
        return found, piece, ptype, g - self.offsets[cell], duplicates


def _build(files: Sequence[str], counts: np.ndarray) -> tuple:
    n = int(counts.sum())
    ids = np.empty(n, dtype=np.uint64)
    k = 0
    for i, fname in enumerate(files):
        with h5py.File(fname, 'r') as f:
            for t in range(6):
                if counts[i, t] == 0:
                    continue
                d = f[f'PartType{t}']['ParticleIDs']
                for s in range(0, int(counts[i, t]), _READ_ROWS):
                    e = min(s + _READ_ROWS, int(counts[i, t]))
                    ids[k:k + e - s] = d[s:e]
                    k += e - s
    order = np.argsort(ids, kind='stable').astype(np.uint32)
    return ids[order], order


def open_index(files: Sequence[str], build: bool = True) -> Optional[IDIndex]:
    """The snapshot's index from the cache, built first if needed and allowed; None if none."""
    files = [str(f) for f in files]
    try:
        counts = _layout(files)
        n = int(counts.sum())
        if n < MIN_PARTICLES or n > MAX_PARTICLES or n >= 1 << 32 \
                or n * BYTES_PER_PARTICLE > lod_store.CACHE_LIMIT_BYTES // 2:
            return None
        path = lod_store.CACHE_DIR / _key(files)
        meta = path / 'meta.json'
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                if m.get('version') == FORMAT_VERSION and m.get('n') == n:
                    os.utime(meta)
                    return IDIndex(np.load(path / 'ids.npy', mmap_mode='r'),
                                   np.load(path / 'rows.npy', mmap_mode='r'), counts)
            except (OSError, ValueError):
                pass
        if not build:
            return None
        t0 = time.time()
        ids, rows = _build(files, counts)
        lod_store._evict(ids.nbytes + rows.nbytes)
        tmp = path.with_name(path.name + '.part')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        np.save(tmp / 'ids.npy', ids)
        np.save(tmp / 'rows.npy', rows)
        (tmp / 'meta.json').write_text(json.dumps({'version': FORMAT_VERSION, 'n': n, 'files': files,
                                                   'build_seconds': time.time() - t0, 'created': time.time()}))
        shutil.rmtree(path, ignore_errors=True)
        tmp.rename(path)
        return IDIndex(ids, rows, counts)
    except OSError:
        return None
