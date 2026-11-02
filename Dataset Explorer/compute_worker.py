"""
compute_worker.py
─────────────────
QThread subclass that runs the HDF5 read + plugin.compute() pipeline
off the Qt main thread.

Thread boundary contract
────────────────────────
  IN  (constructor args, provided by the main thread before start()):
        vset, plugin, tab, scale_x, scale_y, header, params, window, context
        All passed by reference.  The main thread must NOT mutate any of
        these objects while the worker is running.  `params` is already a
        fresh copy built by _spawn_worker(), so this constraint is trivially
        satisfied.

  OUT (plotReady signal → main-thread slot _on_plot_ready):
        PlotRequest — a pure-data dataclass with no Qt objects.
        _MatrixRequest — a thin sentinel for the Matrix tab (see below).
        Both are safe to emit across the thread boundary.

  Cancellation
  ────────────
        cancel() sets _cancelled = True.  The worker checks this flag at
        every safe checkpoint (after each extract_array call and before
        emitting).  No QThread.terminate() is used — cooperative exit only.

h5py thread safety
──────────────────
  extract_array() opens a new h5py.File handle per chunk scoped to a
  `with` block inside the function.  No file handle ever crosses the
  thread boundary.  This is the SWMR-compatible multi-reader pattern.

Matrix tab
──────────
  render_matrix() is a Qt widget call and must run on the main thread.
  When tab == "Matrix" the worker emits a _MatrixRequest sentinel
  carrying the raw data arrays.  _on_plot_ready() routes this to
  plot_panel.render_matrix() on the main thread.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from plugin_base import BasePlugin, PlotRequest, RecomputeReason
from io_manager  import VirtualSnapshotSet, MAX_HIST_PARTICLES
from lod_index   import LodIndex, LodCacheKey, enforce_cache_cap, same_bin_target


# ─────────────────────────────────────────────────────────────────────────────
#  SENTINEL — Matrix tab
# ─────────────────────────────────────────────────────────────────────────────

class _MatrixRequest:
    """
    Emitted via plotReady when tab == "Matrix".

    The worker cannot call plot_panel.render_matrix() directly (Qt widget
    call on wrong thread).  Instead it fills this sentinel with the raw
    arrays and lets _on_plot_ready() dispatch to the main thread.
    """
    __slots__ = ("data_arrays", "plugin", "labels", "notes", "info", "status", "cells", "script")

    def __init__(self, data_arrays: dict, plugin: BasePlugin, labels: list | None = None,
                 notes: list | None = None, info: str = "", status: str = "",
                 cells: list | None = None) -> None:
        self.data_arrays = data_arrays
        self.plugin      = plugin
        # Optional, set by analysis plugins that build their own table:
        # column names, export header lines, info and status bar text.
        self.labels      = labels
        self.notes       = notes
        self.info        = info
        self.status      = status
        self.cells       = cells       # rows of display strings (text tables)
        self.script      = None        # Python that reproduces an Analyses result


# ───────────────────────────────────────────────────────────────────────────────
#  LOD BUILD WORKER  (fire-and-forget CSR index builder)
# ───────────────────────────────────────────────────────────────────────────────

class LodBuildWorker(QThread):
    """
    Fire-and-forget background thread that builds the CSR LOD-2 index.

    Receives vset + dataset_path instead of the LOD-1 cgs_data
    array.  The build streams the full HDF5 population via a parallel
    two-pass algorithm, producing global HDF5 particle indices rather than
    LOD-1 buffer offsets.  Inspector query results are therefore physically
    meaningful: index i refers to f[dataset_path][i] in the original file.

    Signals
    ───────
    buildComplete(object)    — the LodIndex on success (PlotPanel._on_lod_ready).
    buildFailed(object, str) — this worker and why it failed (PlotPanel._on_lod_failed).
    progress(object, int, int) — this worker, snapshot files read, files in all
                               (PlotPanel._on_lod_progress).
    The worker rides along on the last two so the panel can ignore a build it
    is no longer waiting for.

    Started by CosmoExplorer._ensure_bin_index for every Histogram on screen.

    Cancellation
    ────────────
    cancel() sets _cancelled = True.  LodIndex.build() checks cancelled_fn()
    between passes and before any disk write — no partial sidecar is written.
    """

    buildComplete: pyqtSignal = pyqtSignal(object)
    buildFailed:   pyqtSignal = pyqtSignal(object, str)
    progress:      pyqtSignal = pyqtSignal(object, int, int)

    def __init__(
        self,
        plugin:       BasePlugin,
        vset:         VirtualSnapshotSet,
        dataset_path: str,
        bin_edges:    np.ndarray,   # CGS-unit edges (display_edges / unit_scalar)
        key:          LodCacheKey,
        cgs_factor:   float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._plugin       = plugin
        self._vset         = vset
        self._dataset_path = dataset_path
        self._bin_edges    = bin_edges      # caller already owns a copy
        self._key          = key
        self._cgs_factor   = cgs_factor
        self._cancelled:   bool = False

    def cancel(self) -> None:
        self._cancelled = True

    def matches(self, key: LodCacheKey, cgs_edges) -> bool:
        """True when this worker builds the index for `key` binned on `cgs_edges`."""
        return same_bin_target(self._key, self._bin_edges, key, cgs_edges)

    def run(self) -> None:
        """Build the globally-indexed CSR index and emit buildComplete.

        Ends with exactly one of buildComplete or buildFailed unless cancelled,
        so the panel never waits on a build that has stopped."""
        if self._cancelled:
            return
        lod    = LodIndex()
        n_bins = len(self._bin_edges) - 1
        # The outer edges are the smallest and largest values on screen, which
        # the histogram counts. Here they arrive as display edges / unit_scalar
        # while the data are raw × cgs_factor, so either extreme can land an
        # ulp outside (seen on a real snapshot: one particle missing from the
        # last bin of PartType0/Coordinates [x]). Widen them by 1e-12, relative.
        edges = np.array(self._bin_edges, dtype=np.float64)
        if len(edges) > 1:
            edges[0]  -= abs(edges[0])  * 1e-12
            edges[-1] += abs(edges[-1]) * 1e-12
        if not lod.is_valid(self._key, self._cgs_factor, n_bins, edges):
            try:
                written = lod.build(
                    vset         = self._vset,
                    dataset_path = self._dataset_path,
                    bin_edges    = edges,
                    cgs_factor   = self._cgs_factor,
                    key          = self._key,
                    cancelled_fn = lambda: self._cancelled,
                    reduce_fn    = self._plugin.reduce_vector,
                    progress_fn  = lambda done, total: self.progress.emit(self, done, total),
                )
            except Exception as exc:
                # The plot is already shown; only the click-to-inspect index
                # is missing.  Say so in the popup and on the terminal.
                import traceback
                print("LOD index build failed; inspector unavailable for "
                      f"{self._dataset_path}:")
                traceback.print_exc()
                if not self._cancelled:
                    self.buildFailed.emit(self, f"{type(exc).__name__}: {exc}")
                return
            if self._cancelled:
                return
            if not written:
                self.buildFailed.emit(self, "the index files could not be written to "
                                            f"{self._key.cache_dir}")
                return
            # A new sidecar was written: keep the folder under the cap.
            try:
                enforce_cache_cap(self._key.cache_dir, keep={self._key.hex_key})
            except Exception:
                import traceback
                print("LOD cache cleanup failed:")
                traceback.print_exc()
        if self._cancelled:
            return
        lod.key       = self._key
        lod.cgs_edges = self._bin_edges
        self._plugin._cached_lod = lod
        self.buildComplete.emit(lod)


# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE WORKER
# ─────────────────────────────────────────────────────────────────────────────

class ComputeWorker(QThread):
    """
    Producer thread: reads HDF5, runs plugin.compute(), emits PlotRequest.

    Usage (main thread)
    ───────────────────
        worker = ComputeWorker(vset, plugin, tab, scale_x, scale_y,
                               header, params, window, context)
        worker.plotReady.connect(self._on_plot_ready)
        worker.finished.connect(lambda: self._on_worker_finished(worker))
        worker.start()

    Cancellation (main thread)
    ──────────────────────────
        worker.cancel()   # cooperative — does NOT call quit() or wait()
    """

    # Class-level signal: emits PlotRequest or _MatrixRequest.
    # `object` type avoids the import-time circular dependency that would
    # arise from declaring pyqtSignal(PlotRequest) in a non-Qt module.
    plotReady: pyqtSignal = pyqtSignal(object)

    def __init__(
        self,
        vset:    VirtualSnapshotSet,
        plugin:  BasePlugin,
        tab:     str,
        scale_x: str,
        scale_y: str,
        header:  dict,
        params:  dict,
        window:  tuple[int, int],
        context: dict | None = None,
        reason:  RecomputeReason = RecomputeReason.DATA_CHANGE,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._vset    = vset
        self._plugin  = plugin
        self._tab     = tab
        self._scale_x = scale_x
        self._scale_y = scale_y
        self._header  = header    # already-coerced dict from vset.global_header
        self._params  = params    # caller's copy — worker may extend freely
        self._window  = window
        self._context = context
        self._reason  = reason    # gates I/O vs cache-hit path
        self._cancelled: bool = False
        # Resolved once at construction; never re-stated
        self._snapshot_path: str = vset.files[0] if vset and vset.files else ""

    # ── Public API ────────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """
        Signal cooperative cancellation.

        Does NOT call quit() or wait().  The worker thread reaches the next
        cancellation checkpoint, skips the emit, and exits run() cleanly.
        The main thread's _on_worker_finished cleans up the reference.
        (The histogram's bin index is built by CosmoExplorer, not here.)
        """
        self._cancelled = True

    def _emit_error(self, stage: str, exc: Exception) -> None:
        """Emit an error PlotRequest (and print the traceback) unless cancelled."""
        import traceback
        traceback.print_exc()
        if not self._cancelled:
            self.plotReady.emit(BasePlugin.error_request(
                f"{stage} failed: {type(exc).__name__}: {exc}"))

    # ── Worker body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Worker body — executed on the worker thread.

        Branching:
          PRESENTATION_CHANGE + valid _cached_cgs  →  cache-hit path
            Skips all HDF5 I/O and Stages 1–2.  Calls plugin._compute_from_cache()
            which begins at Stage 3.  No h5py.File handles are opened.

          DATA_CHANGE  (or _cached_cgs is None)  →  full path
            Opens HDF5 handles, reads all plugin.REQUIRES arrays, then calls
            plugin.compute() which runs all six stages and populates _cached_cgs.

        The only output is a single plotReady emission if and only if
        _cancelled is False at every checkpoint.
        """
        plugin  = self._plugin
        tab     = self._tab
        scale_x = self._scale_x
        scale_y = self._scale_y

        # ── Cache-hit fast path ──────────────────────────────────────────
        # Delegate eligibility check entirely to the plugin — worker has zero
        # knowledge of _cached_cgs field name, _Stage12Result type, or tab
        # sentinel exclusions.
        if (
            self._reason is RecomputeReason.PRESENTATION_CHANGE
            and plugin.can_render_from_cache(tab)
        ):
            try:
                request = plugin._compute_from_cache(
                    tab     = tab,
                    scale_x = scale_x,
                    scale_y = scale_y,
                    params  = self._params,
                    context = self._context,
                )
            except Exception as exc:
                self._emit_error("re-render", exc)
                return

            if not self._cancelled:
                self.plotReady.emit(request)
            return

        # ── Full path: HDF5 read → all 6 stages ──────────────────────────

        data_arrays:  dict[str, np.ndarray] = {}
        data_attrs:   dict[str, dict]       = {}
        data_dtypes:  dict[str, str]        = {}
        n_global_map: dict[str, int]        = {}

        # ── Memory budget ────────────────────────────────────────────────
        # Line tab:      aggressive downsampling → N_GHOST_PTS budget
        # Histogram tab: full distribution needed for FD IQR → 25M budget
        # Matrix tab:    full array (render_matrix does its own sampling)
        max_particles = (plugin.N_GHOST_PTS
                         if tab == "Line" else MAX_HIST_PARTICLES)

        # ── HDF5 read ────────────────────────────────────────────────────
        # Each extract_array() call opens/closes its own h5py.File handles
        # per chunk (SWMR-compatible).  No handle crosses the boundary.
        for path in plugin.REQUIRES:
            if self._cancelled:
                return

            data_dtypes[path] = self._vset.get_dtype(path)

            try:
                chunk = self._vset.extract_array(path, max_particles)
            except KeyError:
                # Dataset absent in this snapshot — skip silently.
                # plugin.compute() will detect the missing key and return
                # an error PlotRequest.
                continue
            except Exception as exc:
                # IO error or dtype mismatch: report it instead of leaving
                # the status bar on "Computing…".
                self._emit_error(f"reading {path}", exc)
                return

            data_arrays[path]  = chunk.data
            n_global_map[path] = chunk.n_global
            data_attrs[path]   = self._vset.get_attributes(path)

        # ── Cancellation checkpoint ──────────────────────────────────────
        if self._cancelled:
            return

        # ── Matrix tab: emit sentinel, let main thread call render_matrix ─
        # (Analysis plugins build their own table in compute(), so they skip this.)
        if tab == "Matrix" and not getattr(plugin, "COMPUTES_MATRIX", False):
            self.plotReady.emit(_MatrixRequest(data_arrays, plugin))
            return

        # ── Inject per-render params ─────────────────────────────────────
        # self._params is already a caller-provided copy; extend it freely.
        self._params["_source_dtypes"] = data_dtypes
        self._params["n_global_map"]   = n_global_map
        # The snapshot path resolves once at worker construction and is
        # injected here so _Stage12Result.snapshot_path is populated on the
        # frozen boundary object used for LodCacheKey construction.
        self._params["_snapshot_path"] = self._snapshot_path
        # LOD-3: inject the live VirtualSnapshotSet so _pipeline_histogram
        # can route through _compute_streaming_histogram when the plugin sets
        # STREAMING_HISTOGRAM = True.  Never injected on the PRESENTATION_CHANGE
        # cache-hit path (that path skips this entire block), so
        # params.get("_vset") correctly returns None there — which causes
        # _pipeline_histogram to fall back to the LOD-1 in-memory path.
        self._params["_vset"] = self._vset

        # ── plugin.compute() — full 6-stage pipeline ─────────────────────
        try:
            request = plugin.compute(
                data    = data_arrays,
                attrs   = data_attrs,
                header  = self._header,
                params  = self._params,
                tab     = tab,
                scale_x = scale_x,
                scale_y = scale_y,
                window  = self._window,
                context = self._context,
            )
        except Exception as exc:
            # Plugin raised: show the error rather than a blank plot.
            self._emit_error(f"computing {plugin.GROUP}/{plugin.DATASET}", exc)
            return

        # ── Final cancellation checkpoint ────────────────────────────────
        # A new selection may have arrived while plugin.compute() was
        # running (slow FD binning on 25M particles).  Check one last time
        # before committing the result to the main thread.
        if self._cancelled:
            return

        self.plotReady.emit(request)
        # The click-a-bin index is started on the main thread when this plot
        # arrives (CosmoExplorer._ensure_bin_index), for this and for every
        # re-render from cache, so no Histogram is left without one.
