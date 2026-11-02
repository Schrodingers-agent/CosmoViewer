"""
lod_index.py
────────────
LOD-2: Persistent, memory-efficient CSR bin-index sidecar.
Built in two parallel passes, with particles indexed globally.

Architecture
────────────
The index maps histogram bins to the **global HDF5 particle indices** that fall
inside them.  "Global" means the row index in the full, unsampled dataset:
    f[dataset_path][global_index]   → that particle's value
A query result of [14823, 9102847, …] names actual HDF5 rows, usable for
subsequent targeted reads of coordinates, metallicity, etc.

CSR structure (identical to sparse-matrix row layout):
    offsets[b]   : first position in `indices` belonging to bin b
    offsets[b+1] : one-past-last position in `indices` belonging to bin b
    particle indices in bin b = indices[offsets[b] : offsets[b+1]]

Build algorithm — parallel two-pass
────────────────────────────────────
Pass 1 — ThreadPoolExecutor across chunk files:
    Each task reads one chunk, counts particles per bin, and returns
    (local_counts[n_bins], n_this_file).  No shared mutable state.

Pre-computation (O(n_files) arithmetic, no I/O):
    chunk_global_offsets  — prefix sum of per-chunk n_this_file values.
    chunk_bin_offsets[i,b] — write-start in `indices` for chunk i, bin b.
    These ranges are non-overlapping by construction — pass 2 writes with
    no locks.

Pass 2 — SINGLE_PASS=True (default):
    Pass 1 tasks also return bin_ids per chunk (cached in RAM).  Pass 2
    fills `indices` from those cached arrays without a second disk read.
    Peak RAM: N_total × 8 bytes (bin_ids int32 + global_idx uint32)
              + max_workers × max_chunk_size × 8 bytes (argsort temp).
    Actual worst-case: N × 16 + max_workers × chunk_size × 8.

    SINGLE_PASS=False (memory-constrained nodes):
    Each chunk file is opened a second time by _fill_chunk tasks.  Peak RAM
    is bounded by max_workers × max_chunk_size × (dtype_bytes + 8).

Bin-edge unit convention
────────────────────────
.edges.npy stores bin edges in **CGS units** (raw × cgs_factor, before any
display unit_scalar is applied).  This keeps the sidecar self-consistent with
the cgs_factor stored in meta and robust to plugin unit changes.
_on_inspector_click converts: cgs_val = display_val / unit_scalar, then
calls np.searchsorted(cgs_edges, cgs_val).

Cancellation
────────────
cancelled_fn() is checked between pass 1 and pass 2, and after pass 2 before
any disk write.  All writes target the in-memory `indices` array first — no
partial sidecar is ever written to disk.

Sidecar files (16-char hex key prefix):
    <key>.offsets.npy   — int64[n_bins + 1]
    <key>.indices.npy   — uint32 or uint64[N_total]
    <key>.edges.npy     — float64[n_bins + 1]  (CGS-unit bin edges)
    <key>.meta.json     — validity sentinel (written last)

Petascale note
──────────────
np.argsort(kind="stable") per chunk is O(n_chunk × log B).  For
n_chunk > _RADIX_N_THRESHOLD a radix sort over B bins is O(n_chunk) — a
future Numba/Cython hook.

Future: replace the VirtualSnapshotSet import with a _SnapshotProtocol to
decouple lod_index from io_manager for offline / CLI use.

Per-file particle counts (future use)
──────────────────────────────────────
Currently read lazily inside _count_chunk (pass 1).  Three future features
require per-file counts known before any I/O:
  1. Parallel extract_array with closed-form strides (prerequisite).
  2. Cross-dataset particle lookup: map global index → (chunk_file, local_offset).
  3. Chunk-level progress reporting for long streaming reads.
When any of these is built, add a lazy-cached property to VirtualSnapshotSet
rather than populating at __init__ time (which would regress NFS load time).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar

import h5py
import numpy as np


# ── Petascale radix-sort threshold ────────────────────────────────────────────
_RADIX_N_THRESHOLD: int = 500_000_000


# ─────────────────────────────────────────────────────────────────────────────
#  LOD CACHE KEY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LodCacheKey:
    """
    Immutable cache identity computed once by ComputeWorker at construction.

    hex_key   : sha256[:16] of "snapshot_path:dataset_path"
    mtime     : os.path.getmtime(snapshot_path), resolved at worker init
    cache_dir : directory for sidecar files; overridable in tests
    """
    hex_key:   str
    mtime:     float
    cache_dir: str

    @property
    def prefix(self) -> str:
        """Full filesystem prefix shared by all sidecar files."""
        return os.path.join(self.cache_dir, self.hex_key)

    @staticmethod
    def build(
        snapshot_path: str,
        dataset_path:  str,
        cache_dir:     str,
    ) -> "LodCacheKey":
        raw     = f"{snapshot_path}:{dataset_path}".encode()
        hex_key = hashlib.sha256(raw).hexdigest()[:16]
        try:
            mtime = os.path.getmtime(snapshot_path)
        except OSError:
            mtime = 0.0
        return LodCacheKey(hex_key=hex_key, mtime=mtime, cache_dir=cache_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CACHE SIZE CAP
# ─────────────────────────────────────────────────────────────────────────────

# Maximum total size of the sidecar folder.  Beyond it the least recently
# used index sets are deleted (they are rebuilt on demand).  Approved: 10 GB.
CACHE_CAP_BYTES: int = 10 * 1024 ** 3

_SIDECAR_RE = re.compile(r"^([0-9a-f]{16})\.(offsets\.npy|indices\.npy|edges\.npy|meta\.json)$")


def cache_sets(cache_dir: str) -> dict:
    """{hex_key: (bytes, last_used, [paths])} for complete or partial sidecar sets.

    Only files named like sidecars are counted; .tmp files (a build in
    progress) and anything else in the folder are left alone.
    """
    sets: dict = {}
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return sets
    for name in names:
        m = _SIDECAR_RE.match(name)
        if not m:
            continue
        path = os.path.join(cache_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        size, used, paths = sets.get(m.group(1), (0, 0.0, []))
        sets[m.group(1)] = (size + st.st_size, max(used, st.st_mtime), paths + [path])
    return sets


def enforce_cache_cap(cache_dir: str, cap_bytes: int = None, keep=()) -> list:
    """Delete least recently used sidecar sets until the folder is ≤ cap.

    keep: hex keys never deleted (the field on screen, the index just built).
    Each deletion is printed.  Returns [(hex_key, bytes)] that were deleted.
    """
    cap_bytes = CACHE_CAP_BYTES if cap_bytes is None else cap_bytes
    sets = cache_sets(cache_dir)
    total = sum(v[0] for v in sets.values())
    removed = []
    if total <= cap_bytes:
        return removed
    keep = set(keep) | set(LodIndex.PROTECTED_KEYS)
    for hex_key, (size, used, paths) in sorted(sets.items(), key=lambda kv: kv[1][1]):
        if total <= cap_bytes:
            break
        if hex_key in keep:
            continue
        # meta.json first: a set without it is already invalid, so a crash
        # part-way never leaves something that looks usable.
        paths = sorted(paths, key=lambda p: not p.endswith(".meta.json"))
        freed = 0
        for path in paths:
            try:
                n = os.path.getsize(path)
                os.unlink(path)
                freed += n
            except OSError as exc:
                print(f"LOD cache: could not delete {path}: {exc}")
        total -= freed
        removed.append((hex_key, freed))
        print(f"LOD cache: deleted index {hex_key} ({freed / 1024 ** 3:.2f} GB, "
              f"last used {_fmt_time(used)}) to stay under {cap_bytes / 1024 ** 3:.0f} GB")
    if total > cap_bytes:
        print(f"LOD cache: still {total / 1024 ** 3:.2f} GB after cleanup; the rest "
              f"is in use or not a sidecar file")
    return removed


def _fmt_time(t: float) -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


# ─────────────────────────────────────────────────────────────────────────────
#  LOD INDEX METADATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _LodIndexMeta:
    n_particles:    int
    n_bins:         int
    dataset_path:   str   # stored for debuggability
    snapshot_mtime: float
    cgs_factor:     float
    index_dtype:    str   # numpy dtype str, e.g. "<u4"
    # 2: particles outside the histogram (zeros in log mode, NaN, outside the
    #    edges) are left out of the index instead of clamped into the end bins.
    format:         int = 2


_LOD_FORMAT = 2


def _same_edges(a, b) -> bool:
    """Bin edges equal to rounding (both CGS, same length)."""
    if a is None or b is None:
        return False
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-9, atol=0.0))


def same_bin_target(key_a, edges_a, key_b, edges_b) -> bool:
    """Same sidecar (snapshot + dataset view) and the same bin edges."""
    if key_a is None or key_b is None:
        return False
    return key_a.hex_key == key_b.hex_key and _same_edges(edges_a, edges_b)


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL WORKER FUNCTIONS
#  Must be module-level (not methods) for clean use with ThreadPoolExecutor.
# ─────────────────────────────────────────────────────────────────────────────

def _bin_ids(data: np.ndarray, bin_edges: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Bin index per particle with np.histogram's rules; -1 where the histogram
    does not count the particle (non-finite, below the first edge or above
    the last).  The last bin includes its right edge.  The outer edges get a
    1e-12 relative tolerance because they arrive here via a unit round trip.
    """
    lo = bin_edges[0] * (1 - 1e-12) if bin_edges[0] > 0 else bin_edges[0]
    hi = bin_edges[-1] * (1 + 1e-12) if bin_edges[-1] > 0 else bin_edges[-1]
    ids = (np.digitize(data, bin_edges) - 1).astype(np.int32)
    ids[ids >= n_bins] = n_bins - 1           # value == last edge (or within tolerance)
    ids[ids < 0] = 0                          # within tolerance below the first edge
    outside = ~np.isfinite(data) | (data < lo) | (data > hi)
    ids[outside] = -1
    return ids


def _bin_counts(ids: np.ndarray, n_bins: int) -> np.ndarray:
    return np.bincount(ids[ids >= 0].astype(np.intp), minlength=n_bins).astype(np.int64)


def _read_chunk_data(
    chunk_path: str,
    grp_name:   str,
    ds_name:    str,
    cgs_factor: float,
    reduce_fn:  Callable | None = None,
) -> tuple[np.ndarray, int]:
    """
    Open one HDF5 chunk, read the dataset, apply cgs_factor.
    reduce_fn turns an N×K dataset into the 1-D quantity the plot bins
    (the plugin's reduce_vector), so the index matches the histogram.

    Returns
    ───────
    (data_cgs, n_this_file)
        data_cgs    : float64 CGS array for this chunk's particles.
                      Empty array if dataset absent or count mismatch.
        n_this_file : particle count from NumPart_ThisFile for offset
                      arithmetic.  Always advances the global offset,
                      even when data_cgs is empty due to a mismatch.
    """
    with h5py.File(chunk_path, "r", rdcc_nbytes=0) as f:
        if grp_name not in f or ds_name not in f[grp_name]:
            return np.empty(0, dtype=np.float64), 0

        ds       = f[grp_name][ds_name]
        n_actual = len(ds)

        # Read NumPart_ThisFile for correct global offset arithmetic.
        # Falls back to n_actual when the Header attribute is absent.
        n_this_file = n_actual
        if "Header" in f:
            try:
                part_idx    = int(grp_name.replace("PartType", ""))
                npart_arr   = np.asarray(
                    f["Header"].attrs.get("NumPart_ThisFile",
                                          np.zeros(6, dtype=np.int64)))
                n_this_file = int(npart_arr[part_idx])
            except Exception:
                pass

        if n_this_file != n_actual:
            warnings.warn(
                f"LOD build: {chunk_path!r} declares NumPart_ThisFile="
                f"{n_this_file} for {grp_name} but dataset length is "
                f"{n_actual}.  Skipping {n_actual} particles from this chunk; "
                f"global offset advanced by {n_this_file} to preserve "
                f"downstream chunk correctness.",
                RuntimeWarning, stacklevel=4,
            )
            return np.empty(0, dtype=np.float64), n_this_file

        data = ds[:].astype(np.float64)

    if data.ndim > 1:
        data = (reduce_fn(data) if reduce_fn is not None
                else np.sqrt(np.sum(data ** 2, axis=-1))).astype(np.float64)
    data *= cgs_factor
    return data, n_this_file


def _count_chunk(
    chunk_path: str,
    grp_name:   str,
    ds_name:    str,
    bin_edges:  np.ndarray,
    cgs_factor: float,
    n_bins:     int,
    reduce_fn:  Callable | None = None,
) -> tuple[np.ndarray, int]:
    """
    Pass 1 (SINGLE_PASS=False): count particles per bin.

    Returns (local_counts[n_bins], n_this_file).
    """
    data, n_this_file = _read_chunk_data(chunk_path, grp_name, ds_name,
                                          cgs_factor, reduce_fn)
    if len(data) == 0:
        return np.zeros(n_bins, dtype=np.int64), n_this_file

    ids    = _bin_ids(data, bin_edges, n_bins)
    return _bin_counts(ids, n_bins), n_this_file


def _count_chunk_cached(
    chunk_path: str,
    grp_name:   str,
    ds_name:    str,
    bin_edges:  np.ndarray,
    cgs_factor: float,
    n_bins:     int,
    reduce_fn:  Callable | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Pass 1 (SINGLE_PASS=True): count particles per bin AND cache bin_ids.

    Returns (local_counts[n_bins], n_this_file, bin_ids[n_actual]).
    bin_ids are int32 to halve the cached memory footprint.
    """
    data, n_this_file = _read_chunk_data(chunk_path, grp_name, ds_name,
                                          cgs_factor, reduce_fn)
    if len(data) == 0:
        return (np.zeros(n_bins, dtype=np.int64), n_this_file,
                np.empty(0, dtype=np.int32))

    ids    = _bin_ids(data, bin_edges, n_bins)
    return _bin_counts(ids, n_bins), n_this_file, ids


def _fill_from_cache(
    bin_ids:               np.ndarray,   # int32[n_chunk], from pass 1
    chunk_global_offset:   int,
    chunk_bin_offsets_row: np.ndarray,   # int64[n_bins]
    indices:               np.ndarray,   # shared pre-allocated output
    idx_dtype:             np.dtype,
    n_bins:                int,
) -> None:
    """
    Pass 2 (SINGLE_PASS=True): fill `indices` from cached bin_ids.

    No disk I/O.  Writes to non-overlapping slices of `indices` (guaranteed
    by chunk_bin_offsets_row), so no lock is required even under
    ThreadPoolExecutor concurrency.
    """
    n = len(bin_ids)
    if n == 0:
        return

    g_idx = (
        np.arange(n, dtype=np.uint64) + np.uint64(chunk_global_offset)
    ).astype(idx_dtype)

    # Particles the histogram does not count (id -1) are not indexed.
    keep = bin_ids >= 0
    if not keep.all():
        bin_ids, g_idx = bin_ids[keep], g_idx[keep]
        n = len(bin_ids)
        if n == 0:
            return

    order       = np.argsort(bin_ids, kind="stable")
    sorted_ids  = bin_ids[order]
    sorted_gidx = g_idx[order]

    unique_bins, starts = np.unique(sorted_ids, return_index=True)
    # np.split produces contiguous per-bin subarrays in one vectorised call,
    # eliminating a Python loop over all n_bins (many may be empty).
    subarrays = np.split(sorted_gidx, starts[1:])
    ends      = np.append(starts[1:], n)

    for k, b in enumerate(unique_bins):
        n_fill   = int(ends[k]) - int(starts[k])
        write_at = int(chunk_bin_offsets_row[b])
        indices[write_at : write_at + n_fill] = subarrays[k]


def _fill_chunk(
    chunk_path:            str,
    grp_name:              str,
    ds_name:               str,
    bin_edges:             np.ndarray,
    cgs_factor:            float,
    chunk_global_offset:   int,
    chunk_bin_offsets_row: np.ndarray,   # int64[n_bins]
    indices:               np.ndarray,   # shared pre-allocated output
    idx_dtype:             np.dtype,
    n_bins:                int,
    n_this_file:           int,
    reduce_fn:             Callable | None = None,
) -> None:
    """
    Pass 2 (SINGLE_PASS=False): re-read chunk from disk and fill `indices`.

    Writes to non-overlapping slices — no lock required.
    Skips silently if the dataset is absent or count mismatches (already
    warned in pass 1).
    """
    data, n_actual_this = _read_chunk_data(chunk_path, grp_name, ds_name,
                                            cgs_factor, reduce_fn)
    if len(data) == 0 or n_actual_this != n_this_file:
        return

    bin_ids = _bin_ids(data, bin_edges, n_bins)

    _fill_from_cache(
        bin_ids               = bin_ids,
        chunk_global_offset   = chunk_global_offset,
        chunk_bin_offsets_row = chunk_bin_offsets_row,
        indices               = indices,
        idx_dtype             = idx_dtype,
        n_bins                = n_bins,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LOD INDEX
# ─────────────────────────────────────────────────────────────────────────────

class LodIndex:
    """
    CSR bin-index: O(N) parallel build, O(1) query.

    Public API
    ──────────
    LodIndex.is_valid(key, cgs_factor, n_bins) -> bool
    LodIndex.build(vset, dataset_path, bin_edges, cgs_factor, key,
                   cancelled_fn) -> None
    LodIndex.query(key, bin_index) -> np.ndarray | None
    LodIndex.load(key) -> tuple | None
    """

    CACHE_DIR: ClassVar[str] = os.path.join(
        os.path.expanduser("~"), ".cosmoviewer", "lod_cache"
    )

    # ── Parallel build configuration ──────────────────────────────────────────
    # MAX_READERS: concurrent HDF5 file handles per pass.  4 saturates a single
    #   NVMe PCIe lane without thrashing.  On Lustre, set to the stripe count.
    # SINGLE_PASS: True  → cache bin_ids from pass 1, skip second disk read.
    #              False → re-read each chunk in pass 2 (lower peak RAM).
    # Peak RAM (SINGLE_PASS=True): N_total × 16 bytes
    #   + max_workers × max_chunk_size × 8 bytes (argsort temp per worker).
    # For N=200M, uint32 indices: ≈ 3.2 GB + worker temps.
    # Set SINGLE_PASS=False on <8 GB nodes.
    # Hex keys of indices currently on screen; enforce_cache_cap never deletes them.
    PROTECTED_KEYS: ClassVar[set] = set()

    MAX_READERS:  ClassVar[int]  = 4
    SINGLE_PASS:  ClassVar[bool] = True

    def __init__(self) -> None:
        # What this index was built (or found valid) for; set by LodBuildWorker.
        # PlotPanel answers a click from it only when both match the histogram
        # on screen, so a bin-count change can never be answered from old bins.
        self.key:       LodCacheKey | None = None
        self.cgs_edges: np.ndarray  | None = None

    def matches(self, key: "LodCacheKey | None", cgs_edges) -> bool:
        """True when this index is for `key` and binned on `cgs_edges`."""
        return same_bin_target(self.key, self.cgs_edges, key, cgs_edges)

    # ── Validity check ────────────────────────────────────────────────────────

    def is_valid(
        self,
        key:        LodCacheKey,
        cgs_factor: float,
        n_bins:     int,
        bin_edges:  np.ndarray | None = None,
    ) -> bool:
        """
        Return True iff the on-disk sidecar is current and compatible.

        Guards (all must pass):
          1. meta.json exists and parses cleanly
          2. snapshot mtime matches within 1 s
          3. cgs_factor matches within rel_tol=1e-9
          4. n_bins matches exactly (stale binning → must rebuild)
          5. when bin_edges is given, the stored edges equal it (rel 1e-9):
             the same bin count over a different range is a different index
        """
        meta_path = key.prefix + ".meta.json"
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        ok = (
            abs(meta.get("snapshot_mtime", -1.0) - key.mtime) < 1.0
            and math.isclose(meta.get("cgs_factor", 0.0), cgs_factor,
                             rel_tol=1e-9)
            and meta.get("n_bins", -1) == n_bins
            and meta.get("format", 1) == _LOD_FORMAT
        )
        if ok and bin_edges is not None:
            try:
                stored = np.load(key.prefix + ".edges.npy")
            except (OSError, ValueError):
                return False
            ok = _same_edges(stored, bin_edges)
        if ok:
            # Mark as recently used so the cache cap removes it last.
            try:
                os.utime(meta_path)
            except OSError:
                pass
        return ok

    # ── Build  (runs on LodBuildWorker thread — never on main thread) ─────────

    def build(
        self,
        vset:         object,         # VirtualSnapshotSet (duck-typed)
        dataset_path: str,
        bin_edges:    np.ndarray,     # CGS-unit edges (display_edges / unit_scalar)
        cgs_factor:   float,
        key:          LodCacheKey,
        cancelled_fn: Callable[[], bool],
        reduce_fn:    Callable | None = None,
        progress_fn:  Callable[[int, int], None] | None = None,
    ) -> bool:
        """
        Build the globally-indexed CSR bin-index via a parallel two-pass algorithm
        and write atomically to the sidecar files.

        Returns True when the sidecar was written, False when there was nothing
        to index, the build was cancelled, or the files could not be replaced.
        progress_fn(files_read, n_files), if given, is called from this thread
        as pass 1 reads each snapshot file (the slow, disk-bound part).

        bin_edges must be in CGS units (raw_value × cgs_factor), matching the
        unit convention of the data read from HDF5.  The caller (ComputeWorker)
        divides display-unit bin_edges by unit_scalar before passing here.

        Cancellation is checked between passes and before any disk write.
        Mid-pass cancellation is safe — all writes target the in-memory
        `indices` array; no partial sidecar is written.
        """
        grp_name, ds_name = dataset_path.split("/", 1)
        n_bins    = len(bin_edges) - 1
        n_chunks  = len(vset.files)
        max_w     = min(self.MAX_READERS, max(1, n_chunks))
        n_total   = int(vset.particle_counts.get(grp_name, 0))
        idx_dtype = np.dtype(np.uint32 if n_total < 2**32 else np.uint64)

        if n_bins == 0 or n_chunks == 0:
            return False

        # ── Pass 1 ────────────────────────────────────────────────────────────
        if self.SINGLE_PASS:
            _p1 = partial(
                _count_chunk_cached,
                grp_name=grp_name, ds_name=ds_name,
                bin_edges=bin_edges, cgs_factor=cgs_factor, n_bins=n_bins,
                reduce_fn=reduce_fn,
            )
        else:
            _p1 = partial(
                _count_chunk,
                grp_name=grp_name, ds_name=ds_name,
                bin_edges=bin_edges, cgs_factor=cgs_factor, n_bins=n_bins,
                reduce_fn=reduce_fn,
            )

        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = [pool.submit(_p1, path) for path in vset.files]
            if progress_fn is not None:
                progress_fn(0, n_chunks)
                for done, _ in enumerate(as_completed(futures), 1):
                    progress_fn(done, n_chunks)
            p1_results = [fu.result() for fu in futures]   # file order; re-raises

        # Unpack results
        local_counts_list = [r[0] for r in p1_results]
        n_this_list       = [r[1] for r in p1_results]
        bin_ids_list      = [r[2] for r in p1_results] if self.SINGLE_PASS else None

        # Prefix-sum of n_this_file → global particle offsets per chunk
        chunk_global_offsets = np.zeros(n_chunks + 1, dtype=np.int64)
        for i, n in enumerate(n_this_list):
            chunk_global_offsets[i + 1] = chunk_global_offsets[i] + n

        # Global bin counts → CSR offsets
        local_counts  = np.array(local_counts_list, dtype=np.int64)  # (n_chunks, n_bins)
        global_counts = local_counts.sum(axis=0)                      # (n_bins,)
        offsets       = np.zeros(n_bins + 1, dtype=np.int64)
        np.cumsum(global_counts, out=offsets[1:])

        # chunk_bin_offsets[i, b] = write-start in `indices` for chunk i, bin b.
        # Each chunk owns an exclusive, non-overlapping slice per bin.
        chunk_bin_offsets = np.zeros((n_chunks, n_bins), dtype=np.int64)
        if n_chunks > 1:
            np.cumsum(local_counts[:-1], axis=0, out=chunk_bin_offsets[1:])
        chunk_bin_offsets += offsets[:n_bins]   # broadcast bin base offsets

        # ── Cancellation checkpoint ────────────────────────────────────────────
        if cancelled_fn():
            return False

        # ── Pass 2 ────────────────────────────────────────────────────────────
        n_idx_total = int(offsets[-1])
        indices     = np.empty(n_idx_total, dtype=idx_dtype)

        if self.SINGLE_PASS:
            # Fill from cached bin_ids — no disk I/O.
            # Sequential here: the dominant cost was disk I/O in pass 1.
            # Could also be parallelised (writes are non-overlapping).
            for i in range(n_chunks):
                if cancelled_fn():
                    return False
                _fill_from_cache(
                    bin_ids               = bin_ids_list[i],
                    chunk_global_offset   = int(chunk_global_offsets[i]),
                    chunk_bin_offsets_row = chunk_bin_offsets[i],
                    indices               = indices,
                    idx_dtype             = idx_dtype,
                    n_bins                = n_bins,
                )
            del bin_ids_list  # release cached pass-1 arrays
        else:
            # Two-pass: re-read each chunk; parallel, no locks.
            _p2 = partial(
                _fill_chunk,
                grp_name=grp_name, ds_name=ds_name,
                bin_edges=bin_edges, cgs_factor=cgs_factor,
                indices=indices, idx_dtype=idx_dtype, n_bins=n_bins,
                reduce_fn=reduce_fn,
            )
            fill_args = [
                (vset.files[i],
                 int(chunk_global_offsets[i]),
                 chunk_bin_offsets[i],
                 int(n_this_list[i]))
                for i in range(n_chunks)
            ]
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                list(pool.map(
                    lambda a: _p2(
                        a[0],
                        chunk_global_offset   = a[1],
                        chunk_bin_offsets_row = a[2],
                        n_this_file           = a[3],
                    ),
                    fill_args,
                ))

        # ── Cancellation checkpoint ────────────────────────────────────────────
        if cancelled_fn():
            return False

        # ── Atomic sidecar write ───────────────────────────────────────────────
        # .edges.npy stores bin edges in CGS units (consistent with cgs_factor
        # in meta and with the raw HDF5 data read in _count_chunk / _fill_chunk).
        os.makedirs(key.cache_dir, exist_ok=True)
        for suffix, arr in [
            (".offsets.npy", offsets),
            (".indices.npy", indices),
            (".edges.npy",   bin_edges.astype(np.float64)),
        ]:
            target = key.prefix + suffix
            tmp    = target + ".tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            try:
                os.replace(tmp, target)
            except PermissionError:
                # Windows: os.replace fails on a mapped file.
                # Existing valid sidecar is retained; new build discarded.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return False

        del indices, offsets  # release before meta write — peak RAM drops here

        meta = _LodIndexMeta(
            n_particles    = n_idx_total,
            n_bins         = n_bins,
            dataset_path   = dataset_path,   # stored for debuggability
            snapshot_mtime = key.mtime,
            cgs_factor     = float(cgs_factor),
            index_dtype    = idx_dtype.str,
        )
        meta_tmp = key.prefix + ".meta.json.tmp"
        with open(meta_tmp, "w") as f:
            json.dump(dataclasses.asdict(meta), f)
        try:
            os.replace(meta_tmp, key.prefix + ".meta.json")
        except PermissionError:
            try:
                os.unlink(meta_tmp)
            except OSError:
                pass
            return False
        return True

    # ── Query  (runs on main thread, O(1)) ───────────────────────────────────

    def query(
        self,
        key:       LodCacheKey,
        bin_index: int,
    ) -> np.ndarray | None:
        """
        Return global particle indices for bin `bin_index`, or None.

        The returned indices are global HDF5 row positions — they can be
        used directly in f[dataset_path][returned_indices] for subsequent
        targeted reads of any particle property.

        Memory access: two 8-byte reads from offsets (≤ 2 page faults) +
        one contiguous slice of indices.  The .copy() detaches from the
        memmap before returning so the caller does not pin mapped pages.
        """
        offsets_path = key.prefix + ".offsets.npy"
        indices_path = key.prefix + ".indices.npy"
        if not os.path.exists(offsets_path) or not os.path.exists(indices_path):
            return None
        try:
            offsets = np.load(offsets_path, mmap_mode="r")
            indices = np.load(indices_path, mmap_mode="r")
            n_bins  = len(offsets) - 1
            b       = max(0, min(int(bin_index), n_bins - 1))
            lo      = int(offsets[b])
            hi      = int(offsets[b + 1])
            return indices[lo:hi].copy()
        except Exception:
            return None

    # ── Load raw memmaps ──────────────────────────────────────────────────────

    def load(
        self,
        key: LodCacheKey,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Return (offsets, indices, edges) as read-only memmaps, or None.

        edges are in CGS units.  Used by PlotPanel to hold references that
        keep the mmap pages alive for the duration of an inspector session.
        """
        offsets_path = key.prefix + ".offsets.npy"
        indices_path = key.prefix + ".indices.npy"
        edges_path   = key.prefix + ".edges.npy"
        if not all(os.path.exists(p) for p in
                   (offsets_path, indices_path, edges_path)):
            return None
        try:
            return (
                np.load(offsets_path, mmap_mode="r"),
                np.load(indices_path, mmap_mode="r"),
                np.load(edges_path,   mmap_mode="r"),
            )
        except Exception:
            return None
