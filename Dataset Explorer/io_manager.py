"""
io_manager.py
─────────────
Distributed Multi-File I/O layer for CosmoViewer.

Implements VirtualSnapshotSet: a transparent abstraction over N-chunk
Gadget-4 / VirgoDB / EAGLE distributed snapshots.

Public API
──────────
    vset = VirtualSnapshotSet(filepath)      # filepath = any chunk (e.g. .0.hdf5)
    arr  = vset.extract_array(path, max_p)   # strided read across all chunks
    attrs = vset.get_attributes(path)        # attrs dict from chunk 0
    dtype = vset.get_dtype(path)             # on-disk dtype string from chunk 0

Constants
─────────
    MAX_HIST_PARTICLES = 25_000_000
        Memory budget for Histogram reads.  25M float64 values ≈ 200 MB RAM.
        Passed as max_particles for Histogram tab; N_GHOST_PTS for Line tab.

Striding guarantee
──────────────────
    residual_offset tracks the fractional offset across file boundaries so
    sampling is phase-continuous: no aliasing at chunk edges.

    residual_offset' = (stride - (N_chunk - offset) % stride) % stride

Backwards compatibility
───────────────────────
    Single-file snapshots (NumFilesPerSnapshot == 1 or attribute absent, or
    filename has no numeric suffix) work without any code changes in callers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator

import h5py
import numpy as np

from data_utils import coerce_attrs as _coerce_attrs

# (group, keys) copied into global_params when Parameters lacks them.
_PARAM_FALLBACKS = (
    ("Units",       ("UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s")),
    ("RuntimePars", ("ComovingIntegrationOn", "HubbleParam", "Omega0",
                     "OmegaLambda", "OmegaBaryon")),
)


# ── Public constant ────────────────────────────────────────────────────────────

MAX_HIST_PARTICLES: int = 25_000_000
"""
Memory budget for Histogram data extraction.
25M × float64 ≈ 200 MB — safe for any modern workstation.
The FD / Rice binning engine needs the full distribution to compute
a meaningful IQR; aggressive downsampling here would corrupt the statistics.
"""


@dataclass
class DataChunk:
    """
    Unified I/O result from VirtualSnapshotSet.extract_array().

    Attributes
    ──────────
    data     : np.ndarray — strided / downsampled array (N_local particles).
               Shape (M,) or (M, K) for vector datasets, always float64.
    n_global : int        — true 64-bit total particle count from the
               Header bitwise reconstruction.  Unaffected by striding.
               Use this for all "how many particles are in the simulation?"
               reporting; use len(data) for "how many did we render?".
    """
    data:     np.ndarray
    n_global: int


# ── Main class ─────────────────────────────────────────────────────────────────

class VirtualSnapshotSet:
    """
    Transparent abstraction over a distributed N-chunk HDF5 snapshot.

    Parameters
    ──────────
    filepath : str
        Path to *any* chunk of the snapshot (conventionally chunk 0).
        May also be a plain single-file snapshot with no numeric suffix.

    Attributes
    ──────────
    files            : list[str]   — ordered list of all chunk file paths
    global_header    : dict        — coerced Header attrs from chunk 0
    global_params    : dict        — coerced Parameters attrs from chunk 0
    particle_counts  : dict[str, int]
        64-bit total particle count per PartType key, e.g.
        {"PartType0": 4294968296, "PartType1": 0, ...}

    Raises
    ──────
    FileNotFoundError
        If any expected chunk file in the sequence is absent on disk.
    """

    def __init__(self, filepath: str) -> None:
        self.files:           list[str]      = []
        self.global_header:   dict           = {}
        self.global_params:   dict           = {}
        self.particle_counts: dict[str, int] = {}
        # Per-group prefix sum for the global → (file, local) mapping.
        # None until first read_particle() call; populated lazily per group.
        # Reset to None whenever a new VirtualSnapshotSet is constructed
        # (i.e. on every file open) — no explicit invalidation required.
        self._per_file_prefix: dict[str, np.ndarray] | None = None

        # ── Step 1: parse the filepath ─────────────────────────────────────
        base, chunk_idx, ext = self._parse_filepath(filepath)

        # ── Step 2: read Header / Parameters from chunk 0 ─────────────────
        # chunk 0 is always `filepath` itself if the regex matched, otherwise
        # it IS the file (single-chunk path).
        chunk0_path = filepath
        self._read_chunk0(chunk0_path)

        # ── Step 3: discover + validate all chunk files ────────────────────
        n_files = int(self.global_header.get("NumFilesPerSnapshot", 1))

        if base is None or n_files == 1:
            # Single-file snapshot or unrecognised naming — use as-is
            self.files = [filepath]
        else:
            self._discover_files(base, ext, n_files)
            self._validate_files()

        # ── Step 4: build 64-bit particle count map ────────────────────────
        self._build_particle_map()

    # ── Initialisation helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_filepath(filepath: str) -> tuple[str | None, int | None, str | None]:
        """
        Extract base prefix, chunk index, and extension from filepath.

        Examples
        ────────
            snap_015.0.hdf5  →  ("snap_015.", 0, "hdf5")
            snapshot_000.31.h5  →  ("snapshot_000.", 31, "h5")
            single.hdf5        →  (None, None, None)   ← no numeric suffix
        """
        m = re.match(r"^(.*?)(\d+)\.(hdf5|h5)$", filepath)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
        return None, None, None

    def _read_chunk0(self, path: str) -> None:
        """Open chunk 0 and cache global_header + global_params."""
        with h5py.File(path, "r") as f:
            if "Header" in f:
                self.global_header = _coerce_attrs(f["Header"].attrs)
            if "Parameters" in f:
                self.global_params = _coerce_attrs(f["Parameters"].attrs)
            # EAGLE keeps the run's units in Units and ComovingIntegrationOn
            # in RuntimePars; its Parameters group has no attributes.  Fill
            # only keys Parameters lacks, and only these names (RuntimePars
            # also holds stale values such as UnitMass_in_g = inf).
            for group, keys in _PARAM_FALLBACKS:
                if group in f:
                    extra = _coerce_attrs(f[group].attrs)
                    for k in keys:
                        if k not in self.global_params and k in extra:
                            self.global_params[k] = extra[k]

    def _discover_files(self, base: str, ext: str, n_files: int) -> None:
        """
        Generate expected chunk filenames 0 … n_files-1 from the base prefix.

        Example:  base="snap_015.", ext="hdf5", n_files=4
            → ["snap_015.0.hdf5", "snap_015.1.hdf5", "snap_015.2.hdf5",
               "snap_015.3.hdf5"]
        """
        self.files = [f"{base}{i}.{ext}" for i in range(n_files)]

    def _validate_files(self) -> None:
        """
        Assert every expected chunk exists on disk.

        Raises
        ──────
        FileNotFoundError
            Message names the first missing chunk so the user can diagnose
            incomplete downloads / NFS mounts without counting files manually.
        """
        for path in self.files:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing chunk {path!r} in distributed snapshot.  "
                    f"Expected {len(self.files)} files in sequence.")

    def _build_particle_map(self) -> None:
        """
        Reconstruct true 64-bit particle counts from the split 32-bit words
        stored in the Gadget-4 / EAGLE Header.

        Gadget stores N > 2^32 particles using two 32-bit unsigned ints:
            NumPart_Total[i]          — low  32 bits
            NumPart_Total_HighWord[i] — high 32 bits  (absent → 0)

        64-bit reconstruction:
            total = (high_word << 32) | low_word
        """
        low_words  = self.global_header.get("NumPart_Total",         [0] * 6)
        high_words = self.global_header.get("NumPart_Total_HighWord", [0] * 6)

        for i in range(6):
            try:
                low  = int(np.asarray(low_words)[i])
                high = int(np.asarray(high_words)[i]) if i < len(high_words) else 0
            except (IndexError, TypeError):
                low, high = 0, 0
            self.particle_counts[f"PartType{i}"] = (high << 32) | low

    # ── Targeted single-particle read ──────────────────────────────────────

    def _build_per_file_prefix(self, grp_name: str) -> np.ndarray:
        """
        Build a prefix-sum array from per-file NumPart_ThisFile values.

        Opens each chunk file exactly once to read one Header attribute;
        no dataset data is transferred.  For a typical NVMe drive this
        costs ~10 µs/file; on NFS ~5 ms/file (512 chunks ≈ 2.5 s worst case).

        Returns
        ───────
        int64[n_files + 1] where prefix[i] = global index of first
        particle in chunk i.  prefix[0] = 0 always.
        """
        try:
            part_idx = int(grp_name.replace("PartType", ""))
        except ValueError:
            part_idx = None      # catalogue group: count rows per file instead

        counts: list[int] = []
        for path in self.files:
            try:
                with h5py.File(path, "r") as f:
                    if part_idx is None:
                        g = f.get(grp_name)
                        first = next((d for d in g.values() if isinstance(d, h5py.Dataset)), None) if g else None
                        counts.append(len(first) if first is not None else 0)
                        continue
                    npart = np.asarray(
                        f["Header"].attrs.get(
                            "NumPart_ThisFile", np.zeros(6, dtype=np.int64)))
                    counts.append(int(npart[part_idx]))
            except Exception:
                counts.append(0)

        prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=prefix[1:])
        return prefix

    def read_particle(
        self,
        dataset_path: str,
        global_idx:   int,
    ) -> float:
        """
        Return the raw code-unit value for one particle by global HDF5 index.

        global_idx is 0-based.  Uses O(log n_files) binary search on the
        per-file prefix sum to identify the chunk file, then one h5py scalar
        read `ds[local_idx]` — exactly one element transferred.

        The prefix sum is built lazily on the first call per PartType group
        and cached for the lifetime of this instance.  On a single-file
        snapshot the prefix is [0, N_total] and the binary search always
        resolves to file_idx=0 in O(1).

        Note: float(val) truncates uint64 values above 2^53.  Callers
        should check plugin dtype (ds_dtype) when displaying ParticleIDs.
        """
        grp_name, ds_name = dataset_path.split("/", 1)

        # Lazy prefix initialisation
        if self._per_file_prefix is None:
            self._per_file_prefix = {}
        if grp_name not in self._per_file_prefix:
            self._per_file_prefix[grp_name] = self._build_per_file_prefix(grp_name)

        prefix   = self._per_file_prefix[grp_name]
        # searchsorted(prefix, global_idx, 'right') - 1 gives the file containing
        # global_idx.  Clamp to valid file range.
        file_idx = int(np.searchsorted(prefix, global_idx, side="right")) - 1
        file_idx = max(0, min(file_idx, len(self.files) - 1))
        local_idx = global_idx - int(prefix[file_idx])

        try:
            with h5py.File(self.files[file_idx], "r") as f:
                if grp_name not in f or ds_name not in f[grp_name]:
                    return float("nan")
                val = f[grp_name][ds_name][local_idx]
            return float(val)
        except Exception:
            return float("nan")

    # ── Public I/O methods ─────────────────────────────────────────────────

    def extract_array(
        self,
        dataset_path: str,
        max_particles: int | None = None,
    ) -> DataChunk:
        """
        Read a dataset across all chunks with read-time striding.

        Parameters
        ──────────
        dataset_path : str
            HDF5 path, e.g. "PartType0/Density".
        max_particles : int | None
            Maximum number of particles to return.
            None  → no striding, full read (used for Histogram tab and Matrix).
            int   → stride = max(1, N_global // max_particles); used for Line tab.

        Returns
        ───────
        DataChunk
            .data     — shape (M,) or (M, K) depending on the dataset,
                        always dtype float64.
            .n_global — true 64-bit particle count from the header;
                        independent of striding or max_particles.

        Raises
        ──────
        KeyError
            If dataset_path is absent in every chunk (caller should catch
            and skip gracefully rather than crashing).
        TypeError
            If a chunk's dtype does not match chunk 0 — indicates file
            corruption or an incompatible simulation suite.
        """
        # Route to the correct N_global via the leading PartType group
        grp = dataset_path.split("/")[0]   # e.g. "PartType0"
        n_global = self.particle_counts.get(grp, 0)

        # Stride calculation
        if max_particles is None or n_global == 0 or max_particles >= n_global:
            global_stride = 1   # full read — no decimation
        else:
            global_stride = max(1, n_global // max_particles)

        output_chunks:  list[np.ndarray] = []
        residual_offset: int             = 0
        target_dtype:    np.dtype | None = None
        found_any:       bool            = False

        for chunk_path in self.files:
            with h5py.File(chunk_path, "r") as f:
                # Skip chunks that don't contain this dataset
                parts = dataset_path.split("/", 1)
                if len(parts) != 2:
                    continue
                grp_name, ds_name = parts
                if grp_name not in f or ds_name not in f[grp_name]:
                    continue

                ds = f[grp_name][ds_name]

                # Dtype consistency guard — checked against chunk 0 baseline
                if target_dtype is None:
                    target_dtype = ds.dtype
                    found_any    = True
                elif ds.dtype != target_dtype:
                    raise TypeError(
                        f"Dtype mismatch in {dataset_path!r}: "
                        f"chunk 0 has {target_dtype}, "
                        f"{chunk_path!r} has {ds.dtype}.  "
                        f"Possible file corruption or mixed-suite dataset.")

                # Per-chunk particle count (from this file's Header).
                # Catalogue groups (Group, Subhalo, IDs) have no PartType
                # index: their count is the dataset length.
                n_chunk = len(ds)
                if "Header" in f and grp_name.startswith("PartType"):
                    np_this = f["Header"].attrs.get("NumPart_ThisFile",
                                                    np.zeros(6, dtype=np.int64))
                    try:
                        part_idx = int(grp_name.replace("PartType", ""))
                        n_chunk = int(np.asarray(np_this)[part_idx])
                    except (IndexError, ValueError):
                        n_chunk = len(ds)

                # Strided slice from disk — the h5py slice stays on disk
                # until the memcpy; we never load the full chunk
                data = ds[residual_offset::global_stride]
                output_chunks.append(data.astype(np.float64))

                # Residual offset for next chunk (phase-continuous striding)
                # residual = (stride - (N_chunk - offset) % stride) % stride
                consumed = n_chunk - residual_offset
                if global_stride == 1:
                    residual_offset = 0
                elif consumed <= 0:
                    # This chunk ended before the next strided particle: the
                    # rest of the offset carries into the next chunk.
                    residual_offset = -consumed
                else:
                    residual_offset = (
                        global_stride - (consumed % global_stride)
                    ) % global_stride

        if not found_any:
            raise KeyError(
                f"Dataset {dataset_path!r} not found in any chunk of "
                f"this VirtualSnapshotSet ({len(self.files)} file(s)).")

        arr = np.concatenate(output_chunks) if output_chunks else np.empty(0, dtype=np.float64)
        if n_global == 0:
            # No header count for this group (catalogue files): the read was
            # a full read, so its length is the global count.
            n_global = len(arr)
        return DataChunk(data=arr, n_global=n_global)

    def iter_chunks(
        self,
        dataset_path:           str,
        max_particles_per_chunk: int | None = None,
    ) -> Iterator[DataChunk]:
        """
        LOD-3: Yield one DataChunk per HDF5 chunk file, optionally strided.

        Each file is opened and closed within the `with` block before the
        next yield.  `rdcc_nbytes=0` disables the HDF5 chunk cache entirely:
        each dataset page is read exactly once and not cached — this is the
        documented recommendation for streaming I/O where re-reading the same
        page is impossible by construction.

        Memory footprint: max(chunk_size) * dtype_bytes at any one time.
        GeneratorExit propagated through the `with` block's __exit__ closes
        the HDF5 file correctly if the caller breaks or raises inside the loop.

        Parameters
        ──────────
        dataset_path            : HDF5 path, e.g. "PartType0/Density"
        max_particles_per_chunk : if set, stride within each file so at most
                                  this many particles are returned per chunk.
                                  None = full read of each file.
        """
        grp_name, ds_name = dataset_path.split("/", 1)
        target_dtype: np.dtype | None = None

        for chunk_path in self.files:
            with h5py.File(chunk_path, "r", rdcc_nbytes=0) as f:
                if grp_name not in f or ds_name not in f[grp_name]:
                    continue
                ds = f[grp_name][ds_name]
                if target_dtype is None:
                    target_dtype = ds.dtype
                if max_particles_per_chunk is not None:
                    stride = max(1, len(ds) // max_particles_per_chunk)
                    data   = ds[::stride]
                else:
                    data = ds[:]
                n_global = self.particle_counts.get(grp_name, 0)
                # Yield the raw on-disk array without upcasting here.
                # The streaming consumer (_compute_streaming_histogram) performs
                # the one correct .astype(np.float64) upcast before histogramming.
                # Upcasting here would create a second redundant allocation while
                # both arrays are live — doubling per-chunk RAM footprint.
                yield DataChunk(
                    data     = data,
                    n_global = n_global,
                )

    def get_attributes(self, dataset_path: str) -> dict:
        """
        Return the HDF5 attribute dict for a dataset, read from chunk 0 only.

        Gadget-4 attributes (CGSConversionFactor, aexp-scale-exponent, etc.)
        are written identically across all chunks — reading only chunk 0
        gives O(1) overhead regardless of how many files exist.

        Returns an empty dict if the dataset or its attributes are absent.
        """
        with h5py.File(self.files[0], "r") as f:
            parts = dataset_path.split("/", 1)
            if len(parts) != 2:
                return {}
            grp_name, ds_name = parts
            if grp_name not in f or ds_name not in f[grp_name]:
                return {}
            return _coerce_attrs(f[grp_name][ds_name].attrs)

    def get_dtype(self, dataset_path: str) -> str:
        """
        Return the on-disk HDF5 dtype string for a dataset from chunk 0.

        O(1) I/O overhead — opens only chunk 0, reads the dtype attribute
        (no data transfer), and returns immediately.  The caller maps this
        string to data_dtypes[path] BEFORE extract_array upcasts to float64,
        preserving the original storage footprint for the inspector tooltip.

        Returns "" if the dataset is absent in chunk 0.
        """
        with h5py.File(self.files[0], "r") as f:
            parts = dataset_path.split("/", 1)
            if len(parts) != 2:
                return ""
            grp_name, ds_name = parts
            if grp_name not in f or ds_name not in f[grp_name]:
                return ""
            return str(f[grp_name][ds_name].dtype)
