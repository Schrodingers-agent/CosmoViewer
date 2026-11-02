"""
Read-once cache of particle property values: one array per (snapshot, property, type).

Why: large runs (EAGLE, for example) store every dataset gzip-compressed in
chunks of 8192 rows. A display sample or LOD view set touches nearly every
chunk, so reading "only the sampled rows" still decompresses the whole dataset,
and did so again for every new view set. Reading each dataset once,
sequentially, and keeping the values makes every later set (any zoom level,
any sample) a plain array lookup by (piece, row).

- Types with fewer than DISK_MIN_ROWS rows are kept in memory (bounded,
  least recently used first out).
- Larger ones are written once to ~/.cosmoviewer/property_cache/<snapshot key>/
  as .npy files, filled by one process per file, and opened as memory maps.
  The key is made from the files' paths, sizes and modification times, so a
  changed snapshot is never served stale values. The folder is capped at
  COSMOVIEWER_PROPERTY_CACHE_GB (default 8 GB); the least recently used
  arrays are removed first. Deleting the folder is always safe.

Values are stored as read (dataset dtype; float32 for derived temperature and
vector magnitudes, int32 / int64 for the snapshot's halo membership). Rows are
the file rows of snap['rows'] (or the loader's plan), which
snapshot_io.properties checks against ParticleIDs before trusting them.
"""

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

CACHE_DIR = Path(os.environ.get('COSMOVIEWER_PROPERTY_CACHE', Path.home() / '.cosmoviewer' / 'property_cache'))
CACHE_LIMIT_BYTES = int(float(os.environ.get('COSMOVIEWER_PROPERTY_CACHE_GB', '8')) * 1e9)
DISK_MIN_ROWS = 8_000_000        # a type with fewer rows than this is kept in memory only
MEMORY_LIMIT_BYTES = 1_500_000_000
SLAB_ROWS = 1 << 22

_lock = threading.Lock()
_memory: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_counts: Dict[str, np.ndarray] = {}
_building: Dict[Path, threading.Event] = {}


def snapshot_key(files: Sequence[str]) -> str:
    h = hashlib.sha1(b'props-v1')
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.abspath(f)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def type_counts(files: Sequence[str]) -> np.ndarray:
    """Rows of each type in each file, shape (n_files, 6)."""
    key = snapshot_key(files)
    with _lock:
        if key in _counts:
            return _counts[key]
    counts = np.zeros((len(files), 6), dtype=np.int64)
    for i, fn in enumerate(files):
        with h5py.File(fn, 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    counts[i, t] = g['Coordinates'].shape[0]
    with _lock:
        _counts[key] = counts
    return counts


def _dtype(files: Sequence[str], counts: np.ndarray, info, t: int) -> np.dtype:
    if info.derived == 'temperature' or info.component == -1:
        return np.dtype(np.float32)
    if info.derived == 'snap_group':
        return np.dtype(np.int32)                  # GroupNumber is int32; 0 = none
    if info.derived == 'snap_subhalo':
        return np.dtype(np.int64)                  # group * SUBHALO_STRIDE + subhalo
    for i, fn in enumerate(files):
        if counts[i, t] == 0:
            continue
        with h5py.File(fn, 'r') as f:
            g = f[f'PartType{t}']
            path = info.dataset
            if info.derived == 'masstable':
                path = next((k for k in ('Masses', 'Mass') if k in g), None)
                if path is None:
                    return np.dtype(np.float64)
            if path in g:
                dt = g[path].dtype
                return np.dtype(np.float64) if dt.kind in 'iub' and dt.itemsize > 4 else dt
    return np.dtype(np.float64)


def _fill(fn: str, t: int, info, out: np.ndarray, missing) -> None:
    from snapshot_io.properties import _values_from_group
    with h5py.File(fn, 'r') as f:
        g = f.get(f'PartType{t}')
        n = len(out)
        for s in range(0, n, SLAB_ROWS):
            e = min(s + SLAB_ROWS, n)
            v = _values_from_group(f, g, t, info, np.arange(s, e))
            out[s:e] = missing if v is None else v


def _fill_task(args) -> int:
    """Worker process: fill rows [start, start + n) of the .npy at ``path`` from one file."""
    path, fn, t, info, start, n, missing = args
    arr = np.load(path, mmap_mode='r+')
    _fill(fn, t, info, arr[start:start + n], missing)
    arr.flush()
    del arr
    return n


def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def cache_usage() -> List[Tuple[Path, int, float]]:
    out = []
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob('*/*.npy'):
            try:
                st = p.stat()
                out.append((p, st.st_size, st.st_atime))
            except OSError:
                pass
    return out


def _evict(need: int):
    files = sorted(cache_usage(), key=lambda x: x[2])
    total = sum(s for _, s, _ in files)
    for p, s, _ in files:
        if total + need <= CACHE_LIMIT_BYTES:
            break
        if p.name.endswith('.partial.npy'):
            continue
        try:
            p.unlink()
            total -= s
        except OSError:
            pass


def _remember(key, arr: np.ndarray):
    with _lock:
        _memory[key] = arr
        _memory.move_to_end(key)
        total = sum(a.nbytes for a in _memory.values())
        while total > MEMORY_LIMIT_BYTES and len(_memory) > 1:
            _, old = _memory.popitem(last=False)
            total -= old.nbytes


def values_for_type(files: Sequence[str], info, t: int,
                    progress: Optional[Callable[[int, int], None]] = None,
                    cancelled: Optional[Callable[[], bool]] = None,
                    workers: Optional[int] = None) -> Optional[np.ndarray]:
    """
    All values of ``info`` for type ``t``, concatenated over the files in order
    (index = row offset of the file + row). None if the type has no such property.
    """
    files = [str(f) for f in files]
    if t not in info.types:
        return None
    counts = type_counts(files)
    n = int(counts[:, t].sum())
    if n == 0:
        return None
    skey = snapshot_key(files)
    mkey = (skey, info.name, t)
    with _lock:
        if mkey in _memory:
            _memory.move_to_end(mkey)
            return _memory[mkey]
    dt = _dtype(files, counts, info, t)
    missing = np.nan if dt.kind == 'f' else 0
    starts = np.concatenate([[0], np.cumsum(counts[:, t])])

    if n < DISK_MIN_ROWS:
        arr = np.empty(n, dtype=dt)
        for i, fn in enumerate(files):
            if counts[i, t]:
                _fill(fn, t, info, arr[starts[i]:starts[i + 1]], missing)
                if progress:
                    progress(i + 1, len(files))
        _remember(mkey, arr)
        return arr

    path = CACHE_DIR / skey / f"{_safe(info.name)}_t{t}.npy"
    while True:
        with _lock:
            ev = _building.get(path)
            if ev is None:
                if path.exists():
                    break
                ev = _building[path] = threading.Event()
                owner = True
            else:
                owner = False
        if owner:
            try:
                _build(path, files, counts, starts, info, t, n, dt, missing, progress, cancelled, workers)
            finally:
                with _lock:
                    _building.pop(path, None)
                ev.set()
            if not path.exists():
                return None                              # cancelled
            break
        ev.wait()                                        # another thread is building it
    try:
        os.utime(path)                                   # recently used: evicted last
    except OSError:
        pass
    return np.load(path, mmap_mode='r')


def _build(path: Path, files, counts, starts, info, t, n, dt, missing, progress, cancelled, workers):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.parent / 'snapshot.json'
    if not meta.exists():
        meta.write_text(json.dumps({'files': files, 'created': time.time()}, indent=1))
    _evict(n * dt.itemsize)
    tmp = path.with_name(path.stem + '.partial.npy')
    arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=dt, shape=(n,))
    del arr
    tasks = [(str(tmp), fn, t, info, int(starts[i]), int(counts[i, t]), missing)
             for i, fn in enumerate(files) if counts[i, t]]
    nw = min(workers or max(1, (os.cpu_count() or 2) // 2), len(tasks))   # half the cores: UI and LOD keep the rest
    done_rows = 0
    ok = True
    if nw <= 1:
        for k, task in enumerate(tasks):
            if cancelled and cancelled():
                ok = False
                break
            _fill_task(task)
            if progress:
                progress(k + 1, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futs = [pool.submit(_fill_task, task) for task in tasks]
            for k, fut in enumerate(as_completed(futs)):
                done_rows += fut.result()
                if progress:
                    progress(k + 1, len(tasks))
                if cancelled and cancelled():
                    for f2 in futs:
                        f2.cancel()
                    ok = False
                    break
    if ok:
        os.replace(tmp, path)
    else:
        try:
            tmp.unlink()
        except OSError:
            pass


def gather(snap: dict, info, plan: Optional[list] = None,
           progress: Optional[Callable[[int, int], None]] = None,
           cancelled: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """``info`` for every particle of the set, aligned with snap['c'] (NaN where a type lacks it)."""
    from snapshot_io.properties import sample_rows
    files = [str(f) for f in (snap.get('files') or [snap['filepath']])]
    plan = sample_rows(snap) if plan is None else plan
    n_sample = len(snap['c'])
    if sum(len(r) for _, _, r in plan) != n_sample:
        raise ValueError("rows do not match the set")
    counts = type_counts(files)
    piece_of = {fn: i for i, fn in enumerate(files)}
    out = np.full(n_sample, np.nan)
    wanted = sorted({t for _, t, _ in plan} & set(info.types))
    full: Dict[int, Optional[np.ndarray]] = {}
    for k, t in enumerate(wanted):
        def prog(i, m, k=k):
            if progress:
                progress(k * 100 + int(100 * i / max(m, 1)), 100 * len(wanted))
        full[t] = values_for_type(files, info, t, prog, cancelled)
        if cancelled and cancelled():
            raise InterruptedError
    pos = 0
    for fn, t, rows in plan:
        arr = full.get(t)
        if arr is not None and len(rows):
            i = piece_of[str(fn)]
            off = int(counts[:i, t].sum())
            out[pos:pos + len(rows)] = arr[off + rows]
        pos += len(rows)
    return out


def ids_at(files: Sequence[str], t: int, global_rows: np.ndarray) -> np.ndarray:
    """ParticleIDs at type-``t`` rows counted over all files (as values_for_type indexes)."""
    files = [str(f) for f in files]
    counts = type_counts(files)
    starts = np.concatenate([[0], np.cumsum(counts[:, t])])
    global_rows = np.asarray(global_rows, dtype=np.int64)
    out = np.zeros(len(global_rows), dtype=np.uint64)
    from snapshot_io.properties import _read_rows
    for i, fn in enumerate(files):
        m = (global_rows >= starts[i]) & (global_rows < starts[i + 1])
        if m.any():
            with h5py.File(fn, 'r') as f:
                out[m] = _read_rows(f[f'PartType{t}/ParticleIDs'], global_rows[m] - starts[i])
    return out


def top_values(files: Sequence[str], info, types: Sequence[int], k: int = 200, largest: bool = True,
               progress=None, cancelled=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, types, values) of the ``k`` particles with the largest (or smallest) values, full resolution."""
    cand = []
    for t in sorted(set(types) & set(info.types)):
        arr = values_for_type(files, info, t, progress, cancelled)
        if arr is None:
            continue
        v = np.asarray(arr, dtype=np.float64)
        ok = np.flatnonzero(np.isfinite(v))
        if len(ok) == 0:
            continue
        key = -v[ok] if largest else v[ok]
        kk = min(k, len(ok))
        sel = ok[np.argpartition(key, kk - 1)[:kk]]
        cand.append((t, sel, v[sel]))
    if not cand:
        return np.empty(0, np.uint64), np.empty(0, np.int8), np.empty(0)
    tt = np.concatenate([np.full(len(s), t, np.int8) for t, s, _ in cand])
    rr = np.concatenate([s for _, s, _ in cand])
    vv = np.concatenate([v for _, _, v in cand])
    order = np.argsort(-vv if largest else vv, kind='stable')[:k]
    tt, rr, vv = tt[order], rr[order], vv[order]
    ids = np.zeros(len(rr), dtype=np.uint64)
    for t in np.unique(tt):
        m = tt == t
        sub = np.argsort(rr[m])
        got = ids_at(files, int(t), rr[m][sub])
        tmp = np.empty_like(got)
        tmp[sub] = got
        ids[m] = tmp
    return ids, tt, vv
