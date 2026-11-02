"""
LOD store: a uniform random sample of a snapshot, ordered for level of detail,
plus a block index of the snapshot files for reading full-resolution detail.

Every particle of the snapshot gets a fixed random number u in [0, 1), a hash
of its ParticleID (and the seed), so the same particle gets the same u in every
snapshot of a run, wherever Gadget wrote it: a movie draws the same particles
in every frame. Files without ParticleIDs fall back to a number generated from
(seed, file piece, particle type, block start). The store keeps the particles with
u < fraction (fraction = min(1, cap / N)), sorted by octree cell and, inside a
cell, by u. Taking the first part of every cell is then the same as keeping
u below a threshold, and detail read later from the files with
u < f (f >= fraction) contains every stored particle: levels are nested from
the coarsest store level down to the full snapshot.

Small snapshots (fraction 1) are built in memory in well under a second.
Large ones are cached on disk under ~/.cosmoviewer/lod_cache, keyed by the
files' paths, sizes and modification times, and opened as memory maps, so
reopening is fast and only the parts looked at are paged in.

Positions are kept as float32: this is display data. Nothing here is used for
FoF, tracking or any quantity reported to the user.
"""

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from lod.budget import background_workers, lower_priority
from lod.cells import CellTable, _ranges, choose_depth, cell_index, morton

FORMAT_VERSION = 3              # 3: u from ParticleIDs, stored with each row
STORE_CAP = 16_000_000          # particles kept in the store (~51 bytes each, ~0.8 GB on disk at the cap)
DISK_CACHE_MIN = 8_000_000      # below this many particles the store is rebuilt in memory (fast)
CACHE_DIR = Path(os.environ.get('COSMOVIEWER_LOD_CACHE', Path.home() / '.cosmoviewer' / 'lod_cache'))
CACHE_LIMIT_BYTES = int(float(os.environ.get('COSMOVIEWER_LOD_CACHE_GB', '8')) * 1e9)
BLOCK_ROWS = 1 << 14            # rows per block of the file index (2 EAGLE HDF5 chunks; 16k measured
                                # to read 1.5-2x fewer rows on zoom than 64k, PERFORMANCE.md)
TASK_ROWS = 1 << 22             # rows per build task (one process reads ~4M rows at a time)
SUB = 4                         # each block's occupancy is a 4x4x4 bitmap inside its bounding box

VELOCITY_NAMES = ('Velocities', 'Velocity')
MASS_NAMES = ('Masses', 'Mass')
FIELDS = ('pos', 't', 'id', 'v', 'm', 'piece', 'row', 'u')
DERIVED = ('rank',)             # rank: position of each row in (type, piece, row) order, for subset()


def _first(group, names):
    for n in names:
        if n in group and isinstance(group[n], h5py.Dataset):
            return group[n]
    return None


def block_uniforms(seed: int, piece: int, ptype: int, block_start: int, n: int) -> np.ndarray:
    """Fallback u of rows block_start .. block_start+n of one piece and type (files without IDs)."""
    return np.random.default_rng([seed, piece, ptype, block_start]).random(n, dtype=np.float32)


_M64 = (1 << 64) - 1


def id_uniforms(ids: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    u of particles with these ParticleIDs: the splitmix64 hash of (ID, seed), top 24 bits,
    as float32 in [0, 1). Uniform for any ID scheme (consecutive, sparse or 64-bit).
    """
    x = np.asarray(ids).astype(np.uint64) ^ np.uint64((seed * 0x9E3779B97F4A7C15) & _M64)
    with np.errstate(over='ignore'):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return (z >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / (1 << 24))


def row_uniforms(g, seed: int, piece: int, ptype: int, start: int, stop: int):
    """
    (u, ids) of rows start..stop of one type group (h5py group). u comes from ParticleIDs when
    the group has them; otherwise from BLOCK_ROWS-aligned blocks of rows, and ids is None.
    """
    if 'ParticleIDs' in g:
        ids = g['ParticleIDs'][start:stop]
        return id_uniforms(ids, seed), ids
    b0 = start - start % BLOCK_ROWS
    u = np.concatenate([block_uniforms(seed, piece, ptype, b, BLOCK_ROWS) for b in range(b0, stop, BLOCK_ROWS)])
    return u[start - b0:stop - b0], None


def snapshot_layout(files: Sequence[str]) -> dict:
    """Rows per (piece, type), box size and mass table, from the files' metadata only."""
    counts = np.zeros((len(files), 6), dtype=np.int64)
    for i, f in enumerate(files):
        with h5py.File(f, 'r') as h:
            for t in range(6):
                g = h.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    counts[i, t] = g['Coordinates'].shape[0]
            if i == 0:
                hdr = dict(h['Header'].attrs) if 'Header' in h else {}
    box = np.atleast_1d(hdr.get('BoxSize', 0.0)).astype(np.float64)
    mt = np.atleast_1d(hdr.get('MassTable', np.zeros(6))).astype(np.float64)
    return {'files': [str(f) for f in files], 'counts': counts, 'n_total': int(counts.sum()),
            'box': float(box[0]) if box.size else 0.0, 'mass_table': np.resize(mt, 6)}


# ---------------------------------------------------------------------------- reading blocks

def _sub_bits(pos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
    span = np.maximum(hi - lo, 1e-30)
    ijk = np.clip(((pos - lo) * (SUB / span)).astype(np.int64), 0, SUB - 1)
    cells = np.unique(ijk[:, 0] + SUB * (ijk[:, 1] + SUB * ijk[:, 2]))
    bits = 0
    for c in cells:
        bits |= 1 << int(c)
    return bits


def _read_rows(g, sel: np.ndarray, start: int, stop: int, mass_table: float, n: int, all_ids=None):
    """id, v, m of the selected rows (sel indexes rows start..stop) of one type group.
    ``all_ids``: ParticleIDs of rows start..stop if already read (None: read them, or row numbers)."""
    if all_ids is not None:
        ids = all_ids[sel]
    elif 'ParticleIDs' in g:
        ids = g['ParticleIDs'][start:stop][sel]
    else:
        ids = (start + np.nonzero(sel)[0]).astype(np.uint64)
    vd = _first(g, VELOCITY_NAMES)
    v = vd[start:stop][sel] if vd is not None else np.zeros((n, 3), dtype=np.float32)
    md = _first(g, MASS_NAMES)
    if md is not None:
        m = md[start:stop][sel]
    else:
        m = np.full(n, mass_table if mass_table > 0 else 1.0, dtype=np.float32)
    return ids.astype(np.uint64), v.astype(np.float32), m.astype(np.float32)


def scan_task(args) -> dict:
    """
    Read rows [start, stop) of one type in one piece, block by block: record
    each block's bounding box and occupancy, and keep the rows with u < fraction.
    """
    filepath, piece, ptype, start, stop, fraction, seed, mass_table = args
    out = {k: [] for k in FIELDS}
    blk = {k: [] for k in ('piece', 'ptype', 'start', 'stop', 'lo', 'hi', 'bits')}
    try:
        with h5py.File(filepath, 'r') as f:
            g = f[f'PartType{ptype}']
            cd = g['Coordinates']
            for b0 in range(start, stop, BLOCK_ROWS):
                b1 = min(b0 + BLOCK_ROWS, stop)
                pos = cd[b0:b1]
                lo, hi = pos.min(axis=0).astype(np.float64), pos.max(axis=0).astype(np.float64)
                blk['piece'].append(piece); blk['ptype'].append(ptype)
                blk['start'].append(b0); blk['stop'].append(b1)
                blk['lo'].append(lo); blk['hi'].append(hi)
                blk['bits'].append(_sub_bits(pos, lo, hi))
                u, all_ids = row_uniforms(g, seed, piece, ptype, b0, b1)
                sel = u < fraction
                n = int(sel.sum())
                if n == 0:
                    continue
                ids, v, m = _read_rows(g, sel, b0, b1, mass_table, n, all_ids)
                out['u'].append(u[sel])
                out['pos'].append(pos[sel].astype(np.float32))
                out['t'].append(np.full(n, ptype, dtype=np.uint8))
                out['id'].append(ids); out['v'].append(v); out['m'].append(m)
                out['piece'].append(np.full(n, piece, dtype=np.uint16))
                out['row'].append((b0 + np.nonzero(sel)[0]).astype(np.uint32))
    except Exception as e:
        return {'error': f"{Path(filepath).name} PartType{ptype}: {type(e).__name__}: {e}"}
    empty = {'pos': np.empty((0, 3), np.float32), 'v': np.empty((0, 3), np.float32), 'u': np.empty(0, np.float32)}
    res = {k: (np.concatenate(v) if v else empty.get(k, np.empty(0))) for k, v in out.items()}
    res['blocks'] = {
        'piece': np.array(blk['piece'], dtype=np.int32), 'ptype': np.array(blk['ptype'], dtype=np.int8),
        'start': np.array(blk['start'], dtype=np.int64), 'stop': np.array(blk['stop'], dtype=np.int64),
        'lo': np.array(blk['lo']).reshape(-1, 3), 'hi': np.array(blk['hi']).reshape(-1, 3),
        'bits': np.array(blk['bits'], dtype=np.uint64)}
    return res


# ---------------------------------------------------------------------------- the store

@dataclass
class LODStore:
    layout: dict
    fraction: float                       # stored particles / all particles
    seed: int
    arrays: Dict[str, np.ndarray]         # FIELDS, in cell order
    cells: CellTable
    blocks: Dict[str, np.ndarray]
    path: Optional[Path] = None           # cache directory, None if in memory
    build_seconds: float = 0.0
    info: Dict = field(default_factory=dict)

    @property
    def n_stored(self) -> int:
        return len(self.arrays['t'])

    def below(self, cells: np.ndarray, f) -> np.ndarray:
        """
        Store rows of these cells with u < f (one f, or one per cell), in cell order. The rule is
        exact, so a particle is drawn or not by its own u alone, the same in every snapshot.
        """
        cells = np.asarray(cells, dtype=np.int64)
        ct, u = self.cells, self.arrays['u']
        f = np.broadcast_to(np.asarray(f, dtype=np.float64), cells.shape)
        cnt, st = ct.counts[cells], ct.starts[cells]
        # u is sorted inside a cell: read a little past the expected count, then cut exactly.
        est = cnt * np.minimum(f / max(self.fraction, 1e-30), 1.0)
        n = np.minimum(cnt, np.ceil(est * 1.05 + 3.0 * np.sqrt(est) + 2.0).astype(np.int64))
        edge = np.flatnonzero(n < cnt)
        if len(edge):
            short = np.asarray(u[st[edge] + n[edge]]) < f[edge]      # more of the cell is below f
            n[edge[short]] = cnt[edge[short]]
        rows = _ranges(st, n)
        return rows[np.asarray(u[rows]) < np.repeat(f, n)]

    def subset(self, rows: np.ndarray) -> dict:
        """Display arrays of the given store rows, ordered by (type, piece, row)."""
        rows = np.asarray(rows, dtype=np.int64)
        rows.sort()                       # memory-map friendly gather
        a = self.arrays
        # Order by the precomputed rank instead of sorting: mark the ranks present, read them
        # back in order (two passes over a flag per stored row, ~4x faster than a lexsort).
        rank = np.asarray(a['rank'][rows])
        where = np.empty(self.n_stored, dtype=np.int32)
        where[rank] = np.arange(len(rows), dtype=np.int32)
        present = np.zeros(self.n_stored, dtype=bool)
        present[rank] = True
        order = where[np.flatnonzero(present)]
        take = lambda k: np.take(np.asarray(a[k][rows]), order, axis=0)
        return make_view(take('pos'), take('v'), take('m'), take('id'), take('t'), take('piece'), take('row'))


def make_view(pos, v, m, ids, t, piece, row) -> dict:
    """A display set in the snap dict's layout: c, v, m, id, t and rows {(piece, type): rows}."""
    rows = {}
    if len(t):
        key = t.astype(np.int64) * 65536 + piece
        cut = np.flatnonzero(np.diff(key)) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(key)]):
            rows[(int(piece[s]), int(t[s]))] = row[s:e].astype(np.int64)
    return {'c': pos, 'v': v, 'm': m, 'id': ids, 't': t, 'rows': rows}


def _domain(pos: np.ndarray, box: float) -> Tuple[np.ndarray, float]:
    """The cube the cells cover: the periodic box if every particle is in it, else a bounding cube."""
    if len(pos) == 0:
        return np.zeros(3), 1.0
    lo, hi = pos.min(axis=0).astype(np.float64), pos.max(axis=0).astype(np.float64)
    if box > 0 and lo.min() >= -1e-6 * box and hi.max() <= box * (1 + 1e-6):
        return np.zeros(3), box
    size = float(np.max(hi - lo)) * (1 + 1e-6) or 1.0
    return lo, size


def _cache_key(files: Sequence[str], cap: int, seed: int) -> str:
    h = hashlib.sha1(f"v{FORMAT_VERSION}|{cap}|{seed}|{BLOCK_ROWS}".encode())
    for f in files:
        st = os.stat(f)
        h.update(f"|{os.path.abspath(f)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def _tasks(layout: dict, fraction: float, seed: int) -> List[tuple]:
    tasks = []
    for i, f in enumerate(layout['files']):
        for t in range(6):
            n = int(layout['counts'][i, t])
            for s in range(0, n, TASK_ROWS):
                tasks.append((f, i, t, s, min(s + TASK_ROWS, n), fraction, seed, float(layout['mass_table'][t])))
    return tasks


def build_store(files: Sequence[str], cap: int = STORE_CAP, seed: int = 0, workers: Optional[int] = None,
                progress: Optional[Callable[[int, int], None]] = None,
                cancelled: Callable[[], bool] = lambda: False,
                use_cache: Optional[bool] = None) -> Optional[LODStore]:
    """Open the cached store for these files or build it. Returns None if cancelled."""
    t0 = time.time()
    files = [str(f) for f in files]
    layout = snapshot_layout(files)
    n_total = layout['n_total']
    fraction = min(1.0, cap / n_total) if n_total else 1.0
    if use_cache is None:
        use_cache = n_total >= DISK_CACHE_MIN
    key = _cache_key(files, cap, seed) if use_cache else None
    if key:
        cached = open_cached(CACHE_DIR / key, layout)
        if cached is not None:
            cached.build_seconds = time.time() - t0
            cached.info['from_cache'] = True
            return cached

    tasks = _tasks(layout, fraction, seed)
    results: List[Optional[dict]] = [None] * len(tasks)
    if workers is None:          # a process pool only pays off for big snapshots (spawn costs ~1 s on macOS)
        workers = background_workers() if n_total >= DISK_CACHE_MIN else 1
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        for k, a in enumerate(tasks):
            if cancelled():
                return None
            results[k] = scan_task(a)
            if progress:
                progress(k + 1, len(tasks))
    else:
        import multiprocessing
        # spawn, not fork: builds run in a QThread, and forking a threaded process can deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=lower_priority) as ex:
            futs = {ex.submit(scan_task, a): k for k, a in enumerate(tasks)}
            for done, fut in enumerate(as_completed(futs)):
                if cancelled():
                    ex.shutdown(wait=False, cancel_futures=True)
                    return None
                results[futs[fut]] = fut.result()
                if progress:
                    progress(done + 1, len(tasks))
    for r in results:
        if 'error' in r:
            raise IOError(r['error'])
    t_read = time.time() - t0

    arrays = {k: np.concatenate([r[k] for r in results]) for k in FIELDS}
    blocks = {k: np.concatenate([r['blocks'][k] for r in results]) for k in results[0]['blocks']}
    del results
    origin, size = _domain(arrays['pos'], layout['box'])
    depth = choose_depth(len(arrays['t']))
    # Order: leaf cell, then u.
    u = arrays['u']
    keys = morton(cell_index(arrays['pos'], origin, size, depth))
    ukey = np.minimum((u / max(fraction, 1e-30)) * (1 << 30), (1 << 30) - 1).astype(np.uint64)
    perm = np.argsort((keys << np.uint64(30)) | ukey, kind='stable')
    keys = keys[perm]
    del u, ukey
    cells = CellTable.from_sorted_keys(keys, origin, size, depth, seed)
    del keys
    arrays['rank'] = _type_piece_row_rank(arrays, perm)
    info = {'n_total': n_total, 'read_seconds': t_read, 'from_cache': False, 'workers': workers}
    if key:
        path = _write_cache(CACHE_DIR / key, arrays, perm, cells, blocks, layout, fraction, seed)
        del arrays
        store = open_cached(path, layout)
        store.info.update(info)
    else:
        arrays = {k: (v if k in DERIVED else v[perm]) for k, v in arrays.items()}
        store = LODStore(layout, fraction, seed, arrays, cells, blocks, None, 0.0, info)
    store.build_seconds = time.time() - t0
    return store


# ---------------------------------------------------------------------------- disk cache

def _type_piece_row_rank(arrays: dict, perm: np.ndarray) -> np.ndarray:
    """For the rows in store order (arrays[k][perm]): each row's rank in (type, piece, row) order."""
    key = ((arrays['t'].astype(np.int64) << 48) | (arrays['piece'].astype(np.int64) << 32)
           | arrays['row'].astype(np.int64))[perm]
    rank = np.empty(len(key), dtype=np.uint32)
    rank[np.argsort(key)] = np.arange(len(key), dtype=np.uint32)
    return rank


def _write_cache(path: Path, arrays, perm, cells: CellTable, blocks, layout, fraction, seed) -> Path:
    need = sum(a.nbytes for a in arrays.values()) + sum(b.nbytes for b in blocks.values())
    _evict(need)
    tmp = path.with_name(path.name + '.part')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for k, a in arrays.items():
        out = np.lib.format.open_memmap(tmp / f'{k}.npy', mode='w+', dtype=a.dtype, shape=a.shape)
        step = 4_000_000
        for s in range(0, len(perm), step):
            out[s:s + step] = a[s:s + step] if k in DERIVED else a[perm[s:s + step]]
        out.flush()
        del out
    for k, a in blocks.items():
        np.save(tmp / f'blk_{k}.npy', a)
    for k in ('keys', 'starts', 'counts', 'jitter'):
        np.save(tmp / f'cell_{k}.npy', getattr(cells, k))
    meta = {'version': FORMAT_VERSION, 'files': layout['files'], 'fraction': fraction, 'seed': seed,
            'origin': cells.origin.tolist(), 'size': cells.size, 'depth': cells.depth,
            'n_total': layout['n_total'], 'block_rows': BLOCK_ROWS, 'created': time.time()}
    (tmp / 'meta.json').write_text(json.dumps(meta, indent=1))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    return path


def open_cached(path: Path, layout: dict) -> Optional[LODStore]:
    meta_file = path / 'meta.json'
    if not meta_file.exists():
        return None
    try:
        meta = json.loads(meta_file.read_text())
        if meta.get('version') != FORMAT_VERSION or meta.get('block_rows') != BLOCK_ROWS:
            return None
        arrays = {k: np.load(path / f'{k}.npy', mmap_mode='r') for k in FIELDS + DERIVED}
        blocks = {k: np.load(path / f'blk_{k}.npy') for k in ('piece', 'ptype', 'start', 'stop', 'lo', 'hi', 'bits')}
        c = {k: np.load(path / f'cell_{k}.npy') for k in ('keys', 'starts', 'counts', 'jitter')}
    except (OSError, ValueError):
        return None
    cells = CellTable(np.array(meta['origin']), meta['size'], meta['depth'], c['keys'], c['starts'],
                      c['counts'], c['jitter'])
    os.utime(meta_file)                   # for least-recently-used eviction
    return LODStore(layout, meta['fraction'], meta['seed'], arrays, cells, blocks, path,
                    info={'n_total': meta['n_total']})


def cache_usage() -> List[Tuple[Path, int, float]]:
    """(directory, bytes, last used) of every cached store, oldest first."""
    out = []
    if CACHE_DIR.exists():
        for d in CACHE_DIR.iterdir():
            if d.is_dir() and (d / 'meta.json').exists():
                size = sum(f.stat().st_size for f in d.iterdir())
                out.append((d, size, (d / 'meta.json').stat().st_mtime))
    return sorted(out, key=lambda x: x[2])


def _evict(need: int):
    """Remove least recently used stores (only this cache's own directories) to stay under the limit."""
    entries = cache_usage()
    total = sum(s for _, s, _ in entries)
    for d, s, _ in entries:
        if total + need <= CACHE_LIMIT_BYTES:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= s
