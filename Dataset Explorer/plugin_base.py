"""
plugin_base.py
──────────────
Template Method pipeline + pure-data contracts shared by plugins and renderer.
No Qt. No h5py.

Data flow
---------
    Orchestrator
        opens HDF5 once
        reads all arrays listed in plugin.REQUIRES
        calls plugin.compute(data, attrs, header, params, tab, scale_x, scale_y, window)
        receives PlotRequest  ← pure data, no Qt objects
        passes to PlotPanel.render(request)
        closes HDF5

6-Stage Execution Pipeline
──────────────────────────
    Stage 1: Ingestion & Shape Resolution
    Stage 2: Base Cosmological Calibration (CGS)
    Stage 3: Dimensional Unit Transformation
    Stage 4: Domain Conditioning & Log-Space Safety
    Stage 5: Statistical Reduction
    Stage 6: Assembly & Annotation

Child plugins declare class-level configuration and optionally override hooks.
They do NOT override compute() — only DiagnosticPlugin is exempt (it bypasses
the HDF5/CGS pipeline entirely).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, ClassVar, Iterator
import numpy as np

if TYPE_CHECKING:
    from io_manager import VirtualSnapshotSet

from data_utils import (
    compute_robust_domain,
    apply_cosmological_factors,
    adaptive_bin_count,
    integer_bin_edges,
    run_cosmology,
    scaling_exponents,
)
from astro_units import (
    get_unit_scalar,
    get_unit_choices,
    astropy_unit_choices,
    astropy_unit_scalar,
)


# ─────────────────────────────────────────────────────────────────────────────
#  ASTROPY STAGE-3 HELPERS  (module-level; no Qt, no h5py)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_galactic_unit_idx(allowed_units: list, unit_system: str) -> int:
    """
    Heuristically select the best unit from ALLOWED_UNITS for the given
    global unit system.

    Strategy
    ────────
    Scan each unit's unicode label for scale-indicating substrings:
      Galactic      → prefer 'kpc' or plain 'pc'  (M☉/kpc³ is canonical)
      Cosmological  → prefer 'Mpc'

    Falls back to index 0 if no match is found, so the dropdown always
    shows something sensible even for an unusual unit list.
    """
    labels = astropy_unit_choices(allowed_units)
    prefer = "kpc" if unit_system == "Galactic" else "Mpc"
    for i, lbl in enumerate(labels):
        if prefer in lbl:
            return i
    return 0


def _resolve_astropy_units(
    base_cgs_unit,
    allowed_units: list,
    requested_label: str,
    unit_system: str,
) -> tuple:
    """
    Resolve (choices, selected_label, scalar, axis_label) for the Astropy path.

    Selection priority
    ──────────────────
    1. If unit_system is not "CGS" AND requested_label is empty (system just
       changed — orchestrator cleared _unit_labels), auto-select the best unit
       for that system via _pick_galactic_unit_idx().
    2. If requested_label is present and valid (user made a manual dropdown
       choice), honour it regardless of unit_system.
    3. Otherwise fall back to index 0.

    The scalar is computed via astropy_unit_scalar() which catches
    UnitConversionError and emits a terminal warning instead of crashing.
    """
    choices = astropy_unit_choices(allowed_units)
    if not choices:
        return [], "", 1.0, str(base_cgs_unit)

    # Determine which unit to use
    if requested_label and requested_label in choices:
        idx = choices.index(requested_label)
    elif unit_system in ("Galactic", "Cosmological") and not requested_label:
        idx = _pick_galactic_unit_idx(allowed_units, unit_system)
    else:
        idx = 0

    selected_label = choices[idx]
    selected_unit  = allowed_units[idx]

    scalar, axis_label = astropy_unit_scalar(base_cgs_unit, selected_unit)
    return choices, selected_label, scalar, axis_label


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveSpec:
    """One line / filled area on the plot."""
    x:          np.ndarray
    y:          np.ndarray
    role:       str              = ""           # semantic role for ThemeManager
    color:      str              = "#00b4d8"    # fallback if no role
    width:      float            = 1.6          # fallback if no role
    label:      str              = ""
    fill:       bool             = False
    fill_color: str              = "#00b4d850"  # RGBA hex, fallback
    dash:       bool             = False


@dataclass
class MarkerSpec:
    """Vertical or horizontal infinite line marker."""
    pos:   float
    axis:  str          # "x" or "y"
    role:  str          = ""           # semantic role for ThemeManager
    color: str          = "#ff6b6b"    # fallback if no role
    label: str          = ""
    dash:  bool         = True


@dataclass
class RegionSpec:
    """Shaded rectangular region (LinearRegionItem)."""
    x_min: float
    x_max: float
    role:  str  = ""               # semantic role for ThemeManager
    color: str  = "#4ecdc418"      # RGBA hex, very transparent, fallback


@dataclass
class ImageSpec:
    """
    A 2-D array rendered as an origin-anchored heatmap via pg.ImageItem.

    Produced by DiagnosticPlugin (and future image-type plugins).
    PlotPanel._draw_image() consumes it.

    colormap  : "teal"  → black → #4ecdc4  (file 1 / primary)
                "coral" → black → #e07b54  (file 2 / comparison)
    z_value   : controls draw order; higher = on top.
                Use 10 for the foreground layer, 0 for background.
    alpha_min : minimum pixel alpha (0 = fully transparent at data minimum).
                Keeps the background void from washing out the other layer.
    label     : legend entry; empty = no legend item.
    """
    data:      np.ndarray   # 2-D float64 array, raw values
    colormap:  str  = "teal"
    z_value:   int  = 0
    label:     str  = ""
    rect:      tuple | None = None   # (x0, y0, x1, y1) in axis units; None = pixel grid at 0


@dataclass
class PlotRequest:
    """
    Everything the renderer needs.  Zero Qt objects inside.
    Produced by a plugin.compute(); consumed by PlotPanel.render().
    """
    # ── content ──────────────────────────────────────────────────────────────
    title:    str
    x_label:  str
    y_label:  str
    curves:   list[CurveSpec]   = field(default_factory=list)
    markers:  list[MarkerSpec]  = field(default_factory=list)
    regions:  list[RegionSpec]  = field(default_factory=list)
    images:   list[ImageSpec]   = field(default_factory=list)

    # ── axis config ───────────────────────────────────────────────────────────
    log_x:    bool  = True
    log_y:    bool  = True

    # ── UI strings ────────────────────────────────────────────────────────────
    status:   str   = ""          # → status bar (bottom right)
    info:     str   = ""          # → info bar (below toolbar)

    # ── metadata ──────────────────────────────────────────────────────────────
    tab:      str   = ""          # which tab produced this
    n_points: int   = 0           # for meta strip
    dim_label: str  = "D0"        # for meta strip [D0] badge
    n_bins:   int   = 0           # active histogram bin count (for override dialog)

    # ── unit conversion (Phase 2 — Dimensional Freedom) ───────────────────────
    unit_dimension: str       = ""
    unit_label:     str       = ""
    unit_choices:   list[str] = field(default_factory=list)

    # ── inspector metadata ────────────────────────────────────────────────────
    insp_dtype: str = ""

    # ── declarative style defaults ───────────────────────────────────────────
    # Plugin-requested initial visibility/interpolation for the active tab.
    # Empty string = no override (PlotPanel keeps its current cached state).
    # Valid: req_vis_mode in {"Line","Points","Both"}
    #        req_interp_mode in {"Linear","Step"}
    req_vis_mode:    str = ""
    req_interp_mode: str = ""

    # ── LOD inspector fields ─────────────────────────────────────────────────
    # lod is typed as object to avoid importing lod_index here (no circular dep).
    # PlotPanel casts it to LodIndex at use-site.
    lod:           object          = None   # LodIndex | None
    unit_scalar:   float           = 1.0    # Stage-3 scalar for CGS↔display inversion
    bin_edges:     object          = None   # np.ndarray | None — log_bins from histogram
    snapshot_path: str             = ""     # vset.files[0] — for LodCacheKey
    dataset_path:  str             = ""     # e.g. "PartType0/Density"
    lod_key_path:  str             = ""     # cache-key name; differs from dataset_path
                                            # when several views share one dataset
    bin_index_note: str            = ""     # why no particle list can be given for a
                                            # clicked bin, or "" when one can

    # ── Line inspector global-index fields ────────────────────────────────────
    # Enables read_particle() to address the correct HDF5 row from a hover.
    lod1_n_total:  int             = 0      # true N_total for this PartType
    lod1_stride:   int             = 1      # extract_array stride (N_total // N_lod1)
    # LOD-1 raw code-unit array for Line inspector direct reads.
    # Typed as object to avoid ndarray import cycle; PlotPanel casts at use-site.
    lod1_raw:      object          = None   # np.ndarray | None
    lod1_n_lod1:   int             = 0      # len(lod1_raw) = N_lod1

    # True when triggered by PRESENTATION_CHANGE (tab, scale, unit).
    # PlotPanel skips auto-range so the user's zoom state is preserved.
    is_presentation: bool          = False


# ─────────────────────────────────────────────────────────────────────────────
#  RECOMPUTE REASON — gates HDF5 I/O in ComputeWorker
# ─────────────────────────────────────────────────────────────────────────────

class RecomputeReason(Enum):
    """Controls which pipeline stages ComputeWorker executes.

    DATA_CHANGE         — new file or new dataset: full read + all six stages.
    PRESENTATION_CHANGE — tab / scale / unit / bin-count change: skip I/O and
                          Stages 1–2; resume from Stage 3 using _cached_cgs.
    """
    DATA_CHANGE         = auto()  # new file, new dataset
    PRESENTATION_CHANGE = auto()  # tab, scale, unit, bin-count only


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1+2 RESULT — frozen boundary object between I/O and rendering
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Stage12Result:
    """Immutable snapshot of everything computed during Stages 1 and 2.

    frozen=True prevents any Stage 3+ code path from rebinding these fields,
    making it impossible to accidentally corrupt the cached CGS array.
    Mutable containers (dicts, ndarrays) can still be mutated by reference,
    but that is not done anywhere in the pipeline.

    Produced exclusively by BasePlugin._run_stage1_stage2().
    Consumed by BasePlugin._compute_stages3plus() (via both the full path
    through compute() and the cache-hit path through _compute_from_cache()).
    """
    cgs_data:      np.ndarray   # post-Stage-2 CGS array  (raw * to_cgs * cosmo)
    raw:           np.ndarray   # pre-CGS array after reduce_vector (Line tab)
    ds_attrs:      dict         # dataset HDF5 attributes
    n:             int          # len(raw) — local rendered particle count
    n_global:      int          # true 64-bit global particle count
    to_cgs:        float        # Gadget-4 / EAGLE CGS conversion factor
    h_exp:         float        # h-scale-exponent
    a_exp:         float        # aexp-scale-exponent
    h:             float        # HubbleParam
    a:             float        # Time (scale factor)
    redshift:      float        # z — from Header or derived as 1/a - 1
    cosmo_applied: bool         # h_exp != 0 or a_exp != 0
    ds_dtype:      str          # on-disk dtype string (for inspector tooltip)
    source_dtypes: dict         # {path: dtype_str} for _source_dtypes injection
    n_global_map:  dict         # {path: n_global} for n_global_map injection
    # ── LOD plumbing ──────────────────────────────────────────────────────────
    cgs_factor:    float = 0.0  # to_cgs * cosmo_factor — combined CGS scale
    snapshot_path: str  = ""   # vset.files[0]; carried for LodCacheKey construction
    dataset_path:  str  = ""   # e.g. "PartType0/Density"
    # ── Run cosmology (data_utils.run_cosmology) ──────────────────────────────
    is_cosmological: bool = False
    time_code:     float = 0.0  # Header/Time as stored (a or code time)
    h_known:       bool = True  # False → HubbleParam absent; h = 1 was used
    comoving:      bool = False # Options ▸ Comoving: the a^a_exp factor was left out


# ─────────────────────────────────────────────────────────────────────────────
#  BASE PLUGIN — TEMPLATE METHOD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class BasePlugin(abc.ABC):
    """
    Template Method base for all quantity plugins.

    Routing & Identity (must override)
    ───────────────────────────────────
    GROUP            : HDF5 group, e.g. "PartType0"
    DATASET          : HDF5 dataset name, e.g. "Density"
    TABS             : list of tab names, e.g.["Histogram", "Line"]
    TREE_CONTRIBUTIONS : dict mapping group → [leaf, …]

    Optional Identity
    ─────────────────
    DISPLAY_NAME     : human-readable label (falls back to DATASET)
    DEFAULT_SCALE_X / DEFAULT_SCALE_Y : "log" or "linear"

    Pipeline Configuration
    ──────────────────────
    UNIT_DIMENSION       : physical dimension key, e.g. "density", "metallicity"
    ALLOW_ZEROS_IN_LOG   : True if zeros are physically meaningful (clip to 1e-15)
    SECONDARY_DATASET    : for future 2D Phase Space plots
    LINE_Y_LABEL         : custom y-axis label for Line tab

    Auto-derived
    ────────────
    REQUIRES : built from GROUP/DATASET/SECONDARY_DATASET unless explicitly set

    Optional Hooks (invoked by Base if overridden)
    ───────────────────────────────────────────────
    reduce_vector(array)                          → 1-D from N-D
    custom_unit_transform(cgs_array, target_unit) → non-scalar transform
    get_custom_markers(unit_scalar, cgs_data)     → extra Histogram markers
    get_custom_line_markers()                     → extra Line markers
    get_custom_regions(unit_scalar, vmin, vmax)   → shaded Histogram regions
    get_custom_info(cgs_data)                     → extra text for info bar
    get_custom_status(cgs_data, unit_label, scalar) → extra text for status bar
    """

    # ── Must override ─────────────────────────────────────────────────────────
    GROUP:    ClassVar[str]
    DATASET:  ClassVar[str]
    TABS:     ClassVar[list[str]]
    TREE_CONTRIBUTIONS: ClassVar[dict[str, list[str]]]

    # ── Optional overrides ────────────────────────────────────────────────────
    DEFAULT_SCALE_X:    ClassVar[str]  = "log"
    DEFAULT_SCALE_Y:    ClassVar[str]  = "log"
    DISPLAY_NAME:       ClassVar[str]  = ""

    # ── Pipeline configuration ────────────────────────────────────────────
    # Astropy schema (Phase 3 — preferred for dimensional quantities)
    #   BASE_CGS_UNIT : astropy Unit the plugin emits after Stage 2,
    #                   e.g. u.g / u.cm**3.  Set via try/except ImportError
    #                   in the child; None = no Astropy delegation.
    #   ALLOWED_UNITS : ordered list of target astropy Units for the dropdown.
    #                   None (not []) = sentinel meaning Astropy was intended
    #                   but ImportError occurred — triggers the safefall label.
    BASE_CGS_UNIT:      ClassVar[object]        = None
    ALLOWED_UNITS:      ClassVar[list | None]   = []
    # Legacy schema (active and retained for niche / dimensionless plugins)
    UNIT_DIMENSION:     ClassVar[str]  = ""
    ALLOW_ZEROS_IN_LOG: ClassVar[bool] = False
    SECONDARY_DATASET:  ClassVar[str]  = ""
    LINE_Y_LABEL:       ClassVar[str]  = ""

    # ── Pipeline constants ────────────────────────────────────────────────────
    N_BINS_MIN:   ClassVar[int] = 8    # absolute floor for all plugins
    N_GHOST_PTS:  ClassVar[int] = 4_000

    # ── Per-plugin statistical cap (override in child with MAX_BINS) ──────────
    MAX_BINS: ClassVar[int] = 100      # default; child plugins override

    # ── Declarative style defaults ────────────────────────────────────────────
    # Child plugins set these to dictate the initial toolbar state when their
    # tab is first rendered.  Plot panel validates and applies on every render.
    DEFAULT_STYLE_HISTOGRAM: ClassVar[list] = ["Line", "Linear"]
    DEFAULT_STYLE_LINE:      ClassVar[list] = ["Line", "Linear"]

    # ── Streaming histogram flag ─────────────────────────────────────────────
    # Set to True in child plugins to route _pipeline_histogram through
    # _compute_streaming_histogram instead of the LOD-1 in-memory path.
    # The wiring is complete; activation is a one-line flip per plugin.
    # No plugin enables it yet: none has a test showing its streamed counts
    # equal those of the LOD-1 path.
    STREAMING_HISTOGRAM: ClassVar[bool] = False

    # Bin on a linear x axis when the data has values ≤ 0 (velocity
    # components, potentials).  Off for hand-written plugins, which are all
    # positive quantities; on for the generic fallback plugin.
    HISTOGRAM_AUTO_LINEAR: ClassVar[bool] = False
    # Always bin on a linear x axis, whatever the sign of the data (positions).
    HISTOGRAM_LINEAR_X: ClassVar[bool] = False
    # {value: meaning} of stored flags that are not data; left out of the bins
    # and counted in the status line (generic_plugin sets EAGLE's group flags).
    HISTOGRAM_FLAG_VALUES: ClassVar[dict] = {}

    def __init__(self) -> None:
        """
        Initialise per-instance hysteresis state.

        _last_bin_count   — FD-computed bin count from the previous render.
        _last_n_particles — valid-particle count from the previous render.
        _cached_cgs       — frozen _Stage12Result from the last DATA_CHANGE
                            render; None until the first full read completes.
                            Set by compute(); consumed by _compute_from_cache().
                            Cleared implicitly when the plugin instance is
                            replaced (_plugin_cache.clear() on file open).
        _cached_lod       — LodIndex instance after first DATA_CHANGE build.
                            Set by LodBuildWorker via ComputeWorker after the
                            build completes (asynchronously after plotReady).
                            Cleared when _plugin_cache.clear() fires on file open.

        All reset to 0 / None on new plugin instance (dataset switch), so the
        first render always recomputes fresh.
        """
        self._last_bin_count:   int                    = 0
        self._last_n_particles: int                    = 0
        self._cached_cgs:       _Stage12Result | None  = None
        self._cached_lod:       object         | None  = None  # LodIndex | None

    def _resolve_bin_count(self, raw_k: int, n_valid: int) -> int:
        """
        Apply population-change hysteresis WITHOUT power-of-two snapping.

        The bin count only updates when the valid-particle count shifts by
        more than 10 % relative to the cached value, preventing histogram
        flicker during timeline scrubbing.

        First call (_last_bin_count == 0) always updates.
        """
        if self._last_bin_count == 0:
            self._last_bin_count   = raw_k
            self._last_n_particles = n_valid
            return raw_k

        pop_change = abs(n_valid - self._last_n_particles) / max(1, self._last_n_particles)
        if pop_change > 0.10:
            self._last_bin_count   = raw_k
            self._last_n_particles = n_valid
        return self._last_bin_count

    # ── Auto-derive REQUIRES from GROUP/DATASET ───────────────────────────────

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only auto-derive if the child did not explicitly declare REQUIRES
        if 'REQUIRES' not in cls.__dict__:
            grp = getattr(cls, 'GROUP', '')
            ds  = getattr(cls, 'DATASET', '')
            if grp and ds:
                paths =[f"{grp}/{ds}"]
                sec = getattr(cls, 'SECONDARY_DATASET', '')
                if sec:
                    paths.append(f"{grp}/{sec}")
                cls.REQUIRES = paths

    # ══════════════════════════════════════════════════════════════════════════
    #  TEMPLATE METHOD — compute()  (thin orchestrator, DATA_CHANGE path only)
    # ══════════════════════════════════════════════════════════════════════════

    def compute(
        self,
        data:    dict[str, np.ndarray],
        attrs:   dict[str, dict],
        header:  dict,
        params:  dict,
        tab:     str,
        scale_x: str,
        scale_y: str,
        window:  tuple[int, int],
        context: dict | None = None,
    ) -> PlotRequest:
        """
        Locked 6-stage execution pipeline.  Child plugins should NOT override
        this method.  Override hooks instead.

        DiagnosticPlugin is the sole exception — it bypasses this pipeline
        entirely because it reads no HDF5 data.

        On the DATA_CHANGE path this method runs Stages 1–2, stores the
        frozen result in self._cached_cgs, then continues from Stage 3.
        On the PRESENTATION_CHANGE path ComputeWorker calls
        _compute_from_cache() directly — compute() is NOT called.
        """
        result = self._run_stage1_stage2(data, attrs, header, params, context)
        if isinstance(result, PlotRequest):
            return result          # dataset absent — propagate error request
        self._cached_cgs = result  # populate cache for future cache-hit renders
        return self._compute_stages3plus(result, tab, scale_x, scale_y, params, context)

    # ── Stage 1 + 2 extractor  (sole producer of _Stage12Result) ─────────────

    def _run_stage1_stage2(
        self,
        data:    dict[str, np.ndarray],
        attrs:   dict[str, dict],
        header:  dict,
        params:  dict,
        context: dict | None = None,
    ) -> "_Stage12Result | PlotRequest":
        """
        Execute Stages 1 (Ingestion & Shape Resolution) and 2 (Base
        Cosmological Calibration).

        Returns a frozen _Stage12Result on success, or an error PlotRequest
        when the primary dataset is absent from the data dict.

        This method has NO access to self._vset or h5py — all inputs arrive
        through the data/attrs/header/params arguments supplied by
        ComputeWorker.
        """
        primary_path = f"{self.GROUP}/{self.DATASET}"

        # ══════════════════════════════════════════════════════════════════
        #  Stage 1: Ingestion & Shape Resolution
        # ══════════════════════════════════════════════════════════════════
        raw = data.get(primary_path)
        if raw is None:
            return self.error_request(
                f"{primary_path} not found in data dict.")

        # Vector data (e.g. Velocities N×3) → child determines reduction
        if raw.ndim > 1:
            raw = self.reduce_vector(raw)

        ds_attrs = attrs.get(primary_path, {})
        n = len(raw)   # N_local: particles in the strided render array

        # N_global: true 64-bit count from the header bitwise cache.
        # Falls back to len(raw) for single-chunk files or diagnostic override
        # (where n_global_map contains n_global == len(array) for each key).
        n_global: int = params.get("n_global_map", {}).get(primary_path, n)

        # ══════════════════════════════════════════════════════════════════
        #  Stage 2: Base Cosmological Calibration (CGS)
        # ══════════════════════════════════════════════════════════════════
        # Gadget-4: to_cgs;  EAGLE/VirgoDB: CGSConversionFactor
        to_cgs_raw = (ds_attrs.get("to_cgs")
                      or ds_attrs.get("CGSConversionFactor", 1.0))
        try:
            to_cgs = float(to_cgs_raw)
        except (TypeError, ValueError):
            to_cgs = 1.0

        # Scale-factor exponents: EAGLE h-/aexp-scale-exponent or Gadget-4
        # h_scaling / a_scaling (absent → 0)
        h_exp, a_exp = scaling_exponents(ds_attrs)

        # a and h of this run. Gadget-4 keeps HubbleParam in Parameters, and
        # in a non-cosmological run Header/Time is a time, so a = 1 there.
        cosmo = run_cosmology(header, params)
        a = cosmo["a"]
        h = cosmo["h"]

        # Context: if Comoving coordinates, bypass the a^a_exp factor so that
        # data remains in comoving units.  h^h_exp is still applied (Hubble
        # normalisation is always physical).  Passing a=1.0 / a_exp=0 is
        # equivalent to applying only the h correction.
        _coordinates = (context or {}).get("coordinates", "Physical")
        if _coordinates == "Comoving":
            cosmo_factor = apply_cosmological_factors(h, h_exp, 1.0, 0.0)
        else:
            cosmo_factor = apply_cosmological_factors(h, h_exp, a, a_exp)
        cgs_data = raw * to_cgs * cosmo_factor

        # On-disk dtype string for inspector tooltip
        _source_dtypes = params.get("_source_dtypes", {})
        ds_dtype       = _source_dtypes.get(primary_path, "float64")

        redshift = (float(header.get("Redshift", (1.0 / a - 1.0) if a > 0 else 0.0))
                    if cosmo["is_cosmological"] else 0.0)

        return _Stage12Result(
            cgs_data      = cgs_data,
            raw           = raw,
            ds_attrs      = ds_attrs,
            n             = n,
            n_global      = n_global,
            to_cgs        = to_cgs,
            h_exp         = h_exp,
            a_exp         = a_exp,
            h             = h,
            a             = a,
            redshift      = redshift,
            cosmo_applied = (h_exp != 0.0 or (a_exp != 0.0 and cosmo["is_cosmological"])),
            ds_dtype      = ds_dtype,
            source_dtypes = _source_dtypes,
            n_global_map  = params.get("n_global_map", {}),
            # LOD plumbing fields
            cgs_factor    = float(to_cgs * cosmo_factor),
            snapshot_path = params.get("_snapshot_path", ""),
            dataset_path  = primary_path,
            is_cosmological = cosmo["is_cosmological"],
            time_code     = cosmo["time_code"],
            h_known       = cosmo["h_known"],
            comoving      = _coordinates == "Comoving",
        )

    # ── Cache eligibility gate  (public — worker and orchestrator only) ───────

    def can_render_from_cache(self, tab: str) -> bool:
        """Return True when a PRESENTATION_CHANGE render can skip HDF5 I/O.

        Encapsulates both eligibility conditions so ComputeWorker and the
        orchestrator have zero knowledge of the field name or tab sentinel:
          1. _cached_cgs is not None  — a prior DATA_CHANGE run populated it.
          2. tab != "Matrix"          — Matrix tab emits _MatrixRequest with raw
             data_arrays from HDF5 and never calls _compute_from_cache().
        """
        return self._cached_cgs is not None and tab != "Matrix"

    # ── Cache-hit entry point  (PRESENTATION_CHANGE path) ────────────────────

    def _compute_from_cache(
        self,
        tab:     str,
        scale_x: str,
        scale_y: str,
        params:  dict,
        context: dict | None = None,
    ) -> PlotRequest:
        """
        Entry point for PRESENTATION_CHANGE renders (tab, scale, unit,
        bin-count).  Reads self._cached_cgs directly — the caller has zero
        dependency on _Stage12Result; no cached= argument required.

        This method has NO access to self._vset or h5py.
        """
        cached = self._cached_cgs   # read own state — caller is fully decoupled
        # Writes into params dict only — the frozen _Stage12Result is unchanged.
        params["_source_dtypes"] = cached.source_dtypes
        params["n_global_map"]   = cached.n_global_map
        request = self._compute_stages3plus(cached, tab, scale_x, scale_y, params, context)
        # Propagate the cached LOD index so PlotPanel can activate
        # the inspector immediately on PRESENTATION_CHANGE renders.
        request.lod           = self._cached_lod
        request.snapshot_path = cached.snapshot_path
        request.dataset_path  = cached.dataset_path
        # Tag as a presentation change so render() preserves zoom state.
        request.is_presentation = True
        return request

    # ── Stage 3–6  (shared convergence point — no I/O) ───────────────────────

    def _compute_stages3plus(
        self,
        r:       "_Stage12Result",
        tab:     str,
        scale_x: str,
        scale_y: str,
        params:  dict,
        context: dict | None = None,
    ) -> PlotRequest:
        """
        Execute Stages 3 through 6 (Unit Transformation → Assembly).

        Takes a frozen _Stage12Result as its sole source of physics data —
        it has NO access to self._vset, h5py, or any I/O mechanism.  All
        cosmological values come exclusively from *r*.
        """
        # ══════════════════════════════════════════════════════════════════
        #  Stage 3: Dimensional Unit Transformation
        # ══════════════════════════════════════════════════════════════════
        _choices        = []
        unit_label      = ""
        unit_scalar     = 1.0
        unit_axis_label = self.DISPLAY_NAME or self.DATASET
        # Working copy — the frozen r.cgs_data field is never rebound.
        cgs_data        = r.cgs_data
        custom_used     = False   # True when custom_unit_transform made the values

        if self.BASE_CGS_UNIT is not None and self.ALLOWED_UNITS:
            # ───────────────────────────────────────────────────────────────
            # PATH A: Astropy delegation
            # Plugin declared BASE_CGS_UNIT + ALLOWED_UNITS successfully.
            # ───────────────────────────────────────────────────────────────
            _choices, unit_label, unit_scalar, unit_axis_label = \
                _resolve_astropy_units(
                    base_cgs_unit   = self.BASE_CGS_UNIT,
                    allowed_units   = self.ALLOWED_UNITS,
                    requested_label = params.get("unit_label", ""),
                    unit_system     = (context or {}).get("unit_system", "CGS"),
                )
            # Hook: custom transform takes priority over Astropy delegation.
            # If the plugin overrides custom_unit_transform(), the pipeline
            # defers entirely to the child's physics and ignores the scalar.
            custom = self.custom_unit_transform(cgs_data, unit_label)
            if custom is not None:
                cgs_data    = custom
                unit_scalar = 1.0
                custom_used = True

        elif self.ALLOWED_UNITS is None:
            # ───────────────────────────────────────────────────────────────
            # PATH B: Safefall — Astropy was intended but ImportError occurred.
            # ALLOWED_UNITS = None is the sentinel set by child plugins when
            # their try/except ImportError block catches a missing astropy.
            # Pass raw CGS array to renderer; lock dropdown to disabled label.
            # ───────────────────────────────────────────────────────────────
            _choices   = ["CGS (Astropy Missing)"]
            unit_label = _choices[0]
            unit_scalar = 1.0
            unit_axis_label = self.DISPLAY_NAME or self.DATASET

        elif self.UNIT_DIMENSION:
            # ───────────────────────────────────────────────────────────────
            # PATH C: Legacy UNIT_DIMENSION (metallicity, niche plugins).
            # UNIT_DIMENSION is retained as a live, active ClassVar.
            # custom_unit_transform() still takes full precedence here too.
            # ───────────────────────────────────────────────────────────────
            _choices = get_unit_choices(self.UNIT_DIMENSION)
            if _choices:
                unit_label = params.get("unit_label", "") or _choices[0]
                if unit_label not in _choices:
                    unit_label = _choices[0]

                custom = self.custom_unit_transform(cgs_data, unit_label)
                if custom is not None:
                    cgs_data    = custom
                    unit_scalar = 1.0
                    custom_used = True
                    # Axis label: get_unit_scalar returns (scalar, axis_label);
                    # we only need the label — the scalar is irrelevant here.
                    _, unit_axis_label = get_unit_scalar(
                        self.UNIT_DIMENSION, unit_label)
                else:
                    unit_scalar, unit_axis_label = get_unit_scalar(
                        self.UNIT_DIMENSION, unit_label)

        # ── Unpack Stage 1/2 scalars from frozen result ─────────────────
        a             = r.a
        h             = r.h
        n             = r.n
        n_global      = r.n_global
        h_exp         = r.h_exp
        a_exp         = r.a_exp
        redshift      = r.redshift
        ds_attrs      = r.ds_attrs
        cosmo_applied = r.cosmo_applied

        # ── Build base info string ──────────────────────────────────────
        # The factors actually multiplied in Stage 2: in Comoving mode a^a_exp is left out.
        a_left_out = r.comoving and r.is_cosmological and a_exp != 0.0
        cosmo_factors = (f"h^{h_exp:g}" + ("" if a_left_out else f"×a^{a_exp:g}")
                         + f" (h={h:.4f}, a={a:.4f})"
                         + (f"; a^{a_exp:g} not applied: comoving" if a_left_out else ""))
        if cosmo_applied:
            cosmo_note = f"{cosmo_factors.replace(' (h=', ' applied  (h=', 1)}"
            if not r.h_known:
                cosmo_note += "  WARNING: HubbleParam not in file, h=1 used"
        elif r.is_cosmological:
            cosmo_note = "no h/a scaling attributes on this dataset"
        else:
            cosmo_note = "non-cosmological run: no h/a correction"

        custom_info_str = self.get_custom_info(cgs_data)
        # Report true global scale; annotate rendered count only when striding
        # was active (i.e. the two numbers differ — Line tab with N_GHOST_PTS).
        if n_global != n:
            particles_str = f"Particles: {n_global:,}  (Rendered: {n:,})"
        else:
            particles_str = f"Particles: {n_global:,}"
        info = (
            f"{self.GROUP}/{self.DATASET}  |  "
            + (f"a={a:.4f}  z={redshift:.3f}  |  " if r.is_cosmological
               else f"t={r.time_code:.4g} (code units)  |  ")
            + f"{particles_str}  |  "
            + (f"{custom_info_str}  |  " if custom_info_str else "")
            + f"{cosmo_note}  |  "
            + self._attr_summary(ds_attrs)
        )

        # ── Dispatch to tab handler (Stages 4–6) ────────────────────────
        if tab == "Histogram":
            request = self._pipeline_histogram(
                cgs_data, n, info, scale_y, params,
                r.to_cgs, cosmo_applied, h_exp, a_exp, h, a,
                unit_scalar, unit_axis_label, unit_label, _choices,
                r.cgs_factor, r.snapshot_path, r.dataset_path,
                n_global=n_global, cosmo_factors=cosmo_factors,
                source_dtype=r.ds_dtype,
            )
            if custom_used:
                # The bin index stores the dataset times one factor; these values
                # are not that (e.g. T from u with μ per particle), so its bins
                # would hold the wrong particles.
                request.bin_index_note = (
                    f"particle lists are not available in {unit_label or 'this unit'}; "
                    f"they are for the stored quantity. Choose another unit to use them.")
            return request
        else:
            return self._pipeline_line(
                r.raw, n, n_global, info, scale_x, scale_y, r.ds_dtype)

    # ── Stages 4–6: Histogram ────────────────────────────────────────────────

    def _pipeline_histogram(
        self,
        cgs_data:        np.ndarray,
        n:               int,
        info:            str,
        scale_y:         str,
        params:          dict,
        to_cgs:          float,
        cosmo_applied:   bool,
        h_exp:           float,
        a_exp:           float,
        h:               float,
        a:               float,
        unit_scalar:     float,
        unit_axis_label: str,
        unit_label:      str,
        unit_choices:    list[str],
        cgs_factor:      float = 0.0,
        snapshot_path:   str   = "",
        dataset_path:    str   = "",
        n_global:        int   = 0,
        cosmo_factors:   str   = "",
        source_dtype:    str   = "float64",
    ) -> PlotRequest:

        # Stage 3 (cont): apply unit scalar to LINEAR data before binning
        # Snag 3: this is the only correct place — never on log-transformed data.
        display_data = cgs_data * unit_scalar

        # ══════════════════════════════════════════════════════════════════
        #  Stage 4: Domain Conditioning & Log-Space Safety
        # ══════════════════════════════════════════════════════════════════
        # compute_robust_domain in "log" mode strips values ≤ 0.
        # Zeros naturally fall below log bin edges (not counted by histogram).
        finite = display_data[np.isfinite(display_data)]
        # Whole-number data (group numbers, counters) is binned on whole numbers,
        # linearly: log bins narrower than the gaps between integers alternate
        # full and empty. Auto-linear is for signed data; data that is only
        # zero or positive keeps the log axis, with the zeros counted in the
        # status line, since a linear axis over [0, max] crushes values that
        # span decades into the first bin.
        # Integers: stored as an integer type and still whole numbers after the
        # unit factor (beyond 2**53 every float is whole, so that test alone
        # would catch energies in erg).
        integer_data = bool(np.dtype(source_dtype).kind in "iu" and finite.size
                            and np.abs(finite).max() < 2.0 ** 53
                            and np.all(finite == np.round(finite)))
        # Flag values (EAGLE's 2**30 "in no group") are not quantities: binned
        # with the rest they stretch the axis over nine decades.
        n_flagged = 0
        if self.HISTOGRAM_FLAG_VALUES and finite.size:
            flagged = np.isin(finite, list(self.HISTOGRAM_FLAG_VALUES))
            n_flagged = int(flagged.sum())
            if n_flagged:
                display_data = display_data[np.isfinite(display_data)]
                display_data = display_data[~np.isin(display_data, list(self.HISTOGRAM_FLAG_VALUES))]
                finite = finite[~flagged]
                if not finite.size:
                    return self.error_request(
                        f"Every {self.DISPLAY_NAME or self.DATASET} value is a flag "
                        f"({self.HISTOGRAM_FLAG_VALUES}).")
        # Signed data, and whole numbers that include zero (SubGroupNumber 0 is
        # the central), go on a linear axis.
        nonpositive = bool(finite.size and (finite.min() <= 0 if integer_data
                                            else finite.min() < 0))
        linear_x = bool(self.HISTOGRAM_LINEAR_X
                        or (self.HISTOGRAM_AUTO_LINEAR and nonpositive))
        vmin, vmax = compute_robust_domain(display_data,
                                           scale_mode="linear" if linear_x else "log")
        if vmin is None:
            return self.error_request(
                f"No finite {'' if linear_x else 'positive '}"
                f"{self.DISPLAY_NAME or self.DATASET} values.")
        # The range of the data itself. compute_robust_domain pads a constant
        # dataset by ±10 % so it has bins; the status and markers report the data.
        _in_domain = finite if linear_x else finite[finite > 0]
        data_min, data_max = float(_in_domain.min()), float(_in_domain.max())

        # ══════════════════════════════════════════════════════════════════
        #  Stage 5: Statistical Reduction — FD Adaptive Binning
        # ══════════════════════════════════════════════════════════════════
        #
        # Primary rule: Freedman–Diaconis  W = 2·IQR·n^(-1/3)
        # Fallback     : Rice Rule         k = ⌈2·n^(1/3)⌉  (when IQR == 0)
        # Custom       : params['custom_bins'] overrides everything (bounded [4,500])
        # Hysteresis   : bin count frozen until valid-particle count shifts >10%
        #
        n_valid = (int(finite.size) if linear_x
                   else int(np.sum(np.isfinite(display_data) & (display_data > 0))))

        custom_bins_raw = params.get('custom_bins')
        if custom_bins_raw is not None:
            # User-requested override — bypass hysteresis entirely.
            # The hysteresis checks population change, which is always 0% when
            # the user just dragged the dialog slider (same dataset, same frame).
            # Applying _resolve_bin_count here would silently discard the input.
            # Also reset the cache so the next auto-render starts fresh.
            raw_k     = max(4, min(int(custom_bins_raw), 500))
            heuristic = "Custom"
            n_bins    = raw_k
            self._last_bin_count   = raw_k
            self._last_n_particles = n_valid
        else:
            # FD rule on the positive-finite display values, capped by MAX_BINS.
            # Hysteresis applied: bin count frozen until population shifts >10%.
            # FD on the variable the bins are spaced in: log10 of the positive
            # values for a log axis, the values themselves for a linear one.
            raw_k, heuristic = adaptive_bin_count(
                display_data,
                n_min = self.N_BINS_MIN,
                n_max = self.MAX_BINS,
                log   = not linear_x,
            )
            n_bins = self._resolve_bin_count(raw_k, n_valid)

        if integer_data:
            log_bins = integer_bin_edges(data_min, data_max, n_bins, log=not linear_x)
            n_bins   = len(log_bins) - 1
            heuristic += ", whole-number edges"
        elif linear_x:
            log_bins = np.linspace(vmin, vmax, n_bins + 1)
        else:
            log_bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins + 1)
        if not integer_data:
            # 10**log10(x) does not round-trip exactly; pin the outer edges so the
            # smallest and largest particles are counted (np.histogram includes
            # the right edge in the last bin).
            log_bins[0], log_bins[-1] = vmin, vmax

        # ── LOD-3: streaming histogram gate ──────────────────────────────
        # When STREAMING_HISTOGRAM is True AND the worker injected _vset,
        # accumulate counts one HDF5 chunk at a time instead of binning the
        # in-memory LOD-1 array.  This keeps peak RAM at
        #   max(chunk_size) * dtype_bytes + n_bins * 8
        # regardless of total particle count.
        #
        # On the PRESENTATION_CHANGE cache-hit path _vset is not injected
        # (ComputeWorker skips the HDF5 block), so we fall through to the
        # LOD-1 path using display_data which is already in RAM.
        _vset = params.get("_vset")  # VirtualSnapshotSet | None
        if self.STREAMING_HISTOGRAM and _vset is not None and dataset_path:
            # Combined scalar: cgs_factor (Stage 2) × unit_scalar (Stage 3).
            # _compute_streaming_histogram applies this in-place per chunk.
            _stream_scalar = cgs_factor * unit_scalar
            counts = self._compute_streaming_histogram(
                _vset, dataset_path, log_bins, _stream_scalar)
            # n_valid from display_data (LOD-1 sample) is a good-enough
            # approximation for status bar; it is not used for binning here.
        else:
            counts, _ = np.histogram(display_data, bins=log_bins)

        edges = log_bins
        # Bin centres where the axis draws them: geometric on a log axis.
        x_mid = (0.5 * (edges[:-1] + edges[1:]) if linear_x
                 else np.sqrt(edges[:-1] * edges[1:]))

        # Zero-Count Mask: NaN AFTER binning — log renderer skips gaps gracefully
        y = counts.astype(np.float64)
        y[y == 0] = np.nan

        peak_idx = int(np.nanargmax(counts))
        peak     = float(x_mid[peak_idx])
        avg_ppb  = n_valid // max(1, n_bins)

        # ══════════════════════════════════════════════════════════════════
        #  Stage 6: Assembly & Annotation
        # ══════════════════════════════════════════════════════════════════
        curve = CurveSpec(
            x    = x_mid,
            y    = y,
            role = "primary_data",
            fill = True,
        )

        # Default domain markers
        markers = [
            MarkerSpec(pos=data_min, axis="x", role="domain_min",
                       label=f"min={data_min:.2e}"),
            MarkerSpec(pos=data_max, axis="x", role="domain_max",
                       label=f"max={data_max:.2e}"),
        ]

        # Hook: child-injected markers and regions
        markers.extend(self.get_custom_markers(unit_scalar, cgs_data))
        regions = self.get_custom_regions(unit_scalar, vmin, vmax)

        # CGS detail for info bar
        if to_cgs != 0:
            exp_val  = int(np.floor(np.log10(abs(to_cgs))))
            mantissa = to_cgs / 10**exp_val
        else:
            exp_val  = 0
            mantissa = 0.0

        # :g, not :.0f — Gadget-4 velocities carry a^0.5 and EAGLE entropy h^-1.333
        cosmo_note_short = (cosmo_factors or f"h^{h_exp:g}×a^{a_exp:g} (h={h:.4f}, a={a:.4f})"
                            ) if cosmo_applied else ""

        info_with_cgs = (
            info
            + f"  |  CGS factor: {mantissa:.4f}×10^{exp_val}"
            + (f"  |  cosmo: {cosmo_note_short}" if cosmo_applied else "")
            + f"  |  unit scalar: {unit_scalar:.4e}"
        )

        # Status bar: heuristic label + avg pts/bin
        custom_status = self.get_custom_status(
            cgs_data, unit_label, unit_scalar)
        # A strided read (n < n_global) gives counts and domain of the sample only.
        sampled = (n_global > n and not (self.STREAMING_HISTOGRAM and _vset is not None))
        stride = max(1, round(n_global / max(1, n))) if sampled else 1
        sample_note = (f"SAMPLE: 1-in-{stride} of {n_global:,} particles; counts and domain "
                       f"are of the sample  |  " if sampled else "")
        status = (
            sample_note +
            f"Peak: {peak:.3e} {unit_label}  |  "
            f"Valid: {n_valid:,}/{n:,}"
            + (f" ({finite.size - n_valid:,} zero or negative, not on the log axis)"
               if not linear_x and finite.size > n_valid else "")
            + (f" ({n_flagged:,} flagged "
               + " / ".join(f"{v:g} = {m}" for v, m in self.HISTOGRAM_FLAG_VALUES.items())
               + ", not binned)" if n_flagged else "")
            + "  |  "
            f"Domain: [{data_min:.3e}, {data_max:.3e}]"
            + ("  (every value equal; bins span ±10 %)" if data_min == data_max else "")
            + "  |  "
            f"{n_bins} bins ({heuristic})  |  "
            f"~{avg_ppb:,} pts/bin"
            + (f"  |  {custom_status}" if custom_status else "")
        )

        # Declarative style: read from plugin's DEFAULT_STYLE_HISTOGRAM
        _hist_style = self.DEFAULT_STYLE_HISTOGRAM
        _req_vis    = _hist_style[0] if len(_hist_style) > 0 else "Line"
        _req_interp = _hist_style[1] if len(_hist_style) > 1 else "Linear"

        return PlotRequest(
            title           = self.DISPLAY_NAME or self.DATASET,
            x_label         = unit_axis_label,
            y_label         = (f"Count (1-in-{stride} sample)" if sampled else "Count"),
            curves          = [curve],
            markers         = markers,
            regions         = regions,
            log_x           = not linear_x,
            log_y           = (scale_y == "log"),
            status          = status,
            info            = info_with_cgs,
            tab             = "Histogram",
            n_points        = n,
            dim_label       = "D0",
            n_bins          = n_bins,
            unit_dimension  = self.UNIT_DIMENSION,
            unit_label      = unit_label,
            unit_choices    = unit_choices,
            req_vis_mode    = _req_vis,
            req_interp_mode = _req_interp,
            # LOD inspector fields
            unit_scalar     = unit_scalar,
            bin_edges       = log_bins.copy(),   # copy: worker may release cgs_data
            snapshot_path   = snapshot_path,
            dataset_path    = dataset_path,
            lod_key_path    = self.lod_key_name(dataset_path),
            # lod is None here; LodBuildWorker sets it asynchronously via
            # PlotPanel._on_lod_ready() after the build completes.
            lod             = None,
        )

    # ── Stages 4–6: Line ─────────────────────────────────────────────────────

    def _pipeline_line(
        self,
        raw:      np.ndarray,
        n:        int,
        n_global: int,
        info:     str,
        scale_x:  str,
        scale_y:  str,
        ds_dtype: str,
    ) -> PlotRequest:
        """
        Full-array line view with log-spaced downsampling for log-x mode.

        Log-x mode: _logspaced_minmax_downsample — bucket edges proportional
        to log-x visual width.  Each screen decade gets equal representation.
        x_global maps each LOD-1 sample to its approximate 1-based global
        HDF5 particle index, so the x-axis spans [1, N_total] regardless of
        how many particles were sampled for display.

        Linear-x mode: _minmax_downsample (linear buckets, already accepts x).
        """

        # ── Dimensionless guard ──────────────────────────────────────────────
        # Detects whether this plugin's quantity has no physical units.
        # Criteria (in priority order):
        #   1. UNIT_DIMENSION == "dimensionless"  (legacy UNIT_DIMENSION path)
        #   2. BASE_CGS_UNIT == astropy dimensionless_unscaled  (Astropy path)
        # Used to suppress "(code units)" labels and disable AutoSIPrefix.
        _is_dimensionless = False
        if self.UNIT_DIMENSION == "dimensionless":
            _is_dimensionless = True
        elif self.BASE_CGS_UNIT is not None and self.ALLOWED_UNITS:
            try:
                from astropy import units as _u
                if self.BASE_CGS_UNIT == _u.dimensionless_unscaled:
                    _is_dimensionless = True
            except ImportError:
                pass

        # Stage 4: Log-space safety for line mode
        plot_y = raw
        if self.ALLOW_ZEROS_IN_LOG and scale_y == "log":
            plot_y = np.clip(raw, 1e-15, None)

        # Stage 5: Downsampling
        # x_global maps each LOD-1 sample i to its approximate 1-based global
        # HDF5 particle index.  Bucket edges and emitted x-values are in global
        # index space [1, N_total] so the rendered curve spans the full dataset.
        stride   = max(1, n_global // n) if n > 0 else 1
        lod1_raw = np.array(plot_y, copy=False)   # safe ref for inspector fallback

        x_global = (
            np.arange(n, dtype=np.float64) * stride + stride // 2 + 1
        ).clip(1, max(1, n_global))

        if scale_x == "log":
            x_ds, y_ds = self._logspaced_minmax_downsample(
                plot_y, self.N_GHOST_PTS, x_in=x_global)
        else:
            x_ds, y_ds = self._minmax_downsample(
                plot_y, self.N_GHOST_PTS, x=x_global)

        n_ds   = len(y_ds)

        # ── Anchor the curve to the exact simulation boundaries ───────────────
        # The downsampler's argmin/argmax selects interior bucket points and is
        # not guaranteed to include x_global[0] or x_global[-1].  Without this
        # anchor, the rendered curve ends short of N_total while the inspector
        # clamps to N_total, producing an invisible region where the tooltip
        # fires but no curve exists.
        #
        # Boundary values: plot_y[0] and plot_y[-1] are the first/last LOD-1
        # code-unit samples — statistically representative of particles 1 and
        # N_total respectively, and physically correct as curve anchors.
        if len(x_ds) > 0 and n_global > 0:
            _n_total_f = float(n_global)
            if x_ds[0] != 1.0:
                x_ds = np.concatenate(([1.0],        x_ds))
                y_ds = np.concatenate(([float(plot_y[0])],  y_ds))
            if x_ds[-1] != _n_total_f:
                x_ds = np.concatenate((x_ds, [_n_total_f]))
                y_ds = np.concatenate((y_ds, [float(plot_y[-1])]))

        n_ds   = len(y_ds)
        vmin_y = float(np.nanmin(y_ds)) if n_ds else 0.0
        vmax_y = float(np.nanmax(y_ds)) if n_ds else 0.0

        # Stage 6: Assembly
        curve = CurveSpec(x=x_ds, y=y_ds, role="ghost_context")

        # Hook: child-injected markers for Line mode
        markers = self.get_custom_line_markers()

        # Y-axis label: custom LINE_Y_LABEL wins; then dimensionless guard
        # strips "(code units)"; fallback appends it for dimensional data.
        if self.LINE_Y_LABEL:
            y_label = self.LINE_Y_LABEL
        elif _is_dimensionless:
            y_label = self.DISPLAY_NAME or self.DATASET
        else:
            y_label = f"{self.DISPLAY_NAME or self.DATASET} (code units)"

        # Status bar: omit " code units" for dimensionless datasets so the
        # bar reads "Y: [min, max]" rather than "Y: [min, max] code units".
        unit_suffix = "" if _is_dimensionless else " code units"
        ds_method   = ("log-spaced min-max" if scale_x == "log"
                       else "linear min-max")
        # A log Y axis cannot draw values ≤ 0 (a signed quantity such as a velocity
        # component loses about half its points), so say how many are left out.
        n_hidden = int(np.count_nonzero(plot_y <= 0)) if scale_y == "log" else 0
        hidden_note = (f"  |  log Y leaves out {n_hidden:,} values ≤ 0"
                       if n_hidden else "")

        # Declarative style for Line tab
        _line_style = self.DEFAULT_STYLE_LINE
        _req_vis    = _line_style[0] if len(_line_style) > 0 else "Line"
        _req_interp = _line_style[1] if len(_line_style) > 1 else "Linear"

        return PlotRequest(
            title           = self.DISPLAY_NAME or self.DATASET,
            x_label         = "Particle Index",
            y_label         = y_label,
            curves          = [curve],
            markers         = markers,
            log_x           = (scale_x == "log"),
            log_y           = (scale_y == "log"),
            status          = (
                (f"Full array — {n:,} particles  |  " if n >= n_global else
                 f"SAMPLE: {n:,} of {n_global:,} particles (every {stride}th)  |  ")
                + f"Y: [{vmin_y:.3e}, {vmax_y:.3e}]{unit_suffix}  |  "
                f"{ds_method} → {n_ds:,} pts" + hidden_note
            ),
            info            = info,
            tab             = "Line",
            n_points        = n,
            dim_label       = "D0",
            unit_dimension  = self.UNIT_DIMENSION,
            insp_dtype      = ds_dtype,
            req_vis_mode    = _req_vis,
            req_interp_mode = _req_interp,
            # Global particle count and stride for read_particle
            lod1_n_total    = int(n_global),
            lod1_stride     = int(stride),
            # LOD-1 raw array for direct inspector reads
            lod1_raw        = lod1_raw,
            lod1_n_lod1     = int(n),
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  HOOKS — override in child plugins as needed
    # ══════════════════════════════════════════════════════════════════════════

    def lod_key_name(self, dataset_path: str) -> str:
        """Name used in the LOD cache key; override when one dataset has several views."""
        return dataset_path

    def reduce_vector(self, array: np.ndarray) -> np.ndarray:
        """
        Reduce a vector dataset (ndim > 1) to a scalar.
        Default: compute magnitude.  Override to select a Cartesian component.
        """
        return np.sqrt(np.sum(array**2, axis=-1))

    def custom_unit_transform(
        self, cgs_array: np.ndarray, target_unit: str,
    ) -> np.ndarray | None:
        """
        Non-scalar unit transformation (e.g. InternalEnergy → Temperature
        via the Ideal Gas Law).
        Return the transformed array, or None to use the standard scalar path.
        """
        return None

    def get_custom_markers(
        self, unit_scalar: float, cgs_data: np.ndarray,
    ) -> list[MarkerSpec]:
        """
        Return additional markers for the Histogram tab.
        Called during Stage 6.  Base already provides domain min/max markers.
        """
        return[]

    def get_custom_line_markers(self) -> list[MarkerSpec]:
        """
        Return additional markers for the Line tab.
        Called during Stage 6.  Base provides no Line markers by default.
        """
        return[]

    def get_custom_regions(
        self, unit_scalar: float, vmin: float, vmax: float,
    ) -> list[RegionSpec]:
        """
        Return shaded regions for the Histogram tab.
        Called during Stage 6.  Base provides no regions by default.
        """
        return[]

    def get_custom_info(self, cgs_data: np.ndarray) -> str:
        """
        Return custom metadata string for the info bar.
        Called before dispatch to Stages 4–6.  Empty string = nothing added.
        """
        return ""

    def get_custom_status(
        self, cgs_data: np.ndarray, unit_label: str, unit_scalar: float,
    ) -> str:
        """
        Return custom text appended to the histogram status bar.
        Called during Stage 6.  Empty string = nothing appended.
        """
        return ""

    # ══════════════════════════════════════════════════════════════════════════
    #  SHARED HELPERS (available to all subclasses)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def error_request(msg: str) -> PlotRequest:
        """Return a PlotRequest that tells the renderer something went wrong."""
        return PlotRequest(
            title   = "Error",
            x_label = "",
            y_label = "",
            status  = f"Error: {msg}",
            info    = msg,
        )

    @staticmethod
    def _compute_streaming_histogram(
        vset:         "VirtualSnapshotSet",
        dataset_path: str,
        bin_edges:    np.ndarray,
        cgs_scalar:   float,
    ) -> np.ndarray:
        """
        LOD-3: Accumulate histogram counts one HDF5 chunk at a time.

        Memory footprint = max(chunk_size) * dtype_bytes + n_bins * 8.
        Only one chunk resides in RAM at any point — the `del` calls and
        the generator's cooperative GC guarantee this.

        Particles outside bin_edges are clipped to the nearest edge bin by
        np.digitize's natural behaviour — this matches the LOD-1 domain-clipped
        histogram behaviour exactly.

        Parameters
        ──────────
        vset         : VirtualSnapshotSet (passed from ComputeWorker)
        dataset_path : HDF5 path, e.g. "PartType0/Density"
        bin_edges    : log_bins array (n_bins + 1 edges), already in display units
        cgs_scalar   : to_cgs * cosmo_factor * unit_scalar — combined scale,
                       precomputed in Stage 2 + Stage 3 so this loop is O(1) ops
                       per element.  in-place *= avoids a second allocation.
        """
        counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
        for chunk in vset.iter_chunks(dataset_path):
            converted = chunk.data.astype(np.float64)
            converted *= cgs_scalar                    # in-place: no second alloc
            local_counts, _ = np.histogram(converted, bins=bin_edges)
            counts += local_counts
            del converted, chunk                       # release before next chunk
        return counts

    @staticmethod
    def _attr_summary(attrs: dict) -> str:
        """
        Format numeric attributes as 'k=v' pairs.
        String attributes (e.g. VirgoDB VarDescription) are shown truncated.
        """
        parts =[]
        for k, v in sorted(attrs.items()):
            if isinstance(v, float):
                parts.append(f"{k}={v:.3g}")
            elif isinstance(v, str):
                truncated = v[:40] + "\u2026" if len(v) > 40 else v
                parts.append(f'{k}="{truncated}"')
        return "  |  ".join(parts)

    @staticmethod
    def _minmax_downsample(
        y:     np.ndarray,
        n_out: int,
        x:     np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Min-max (peak-preserving) downsampling.

        Matches H5Web's Line-view behaviour: for every display bucket the
        function retains BOTH the minimum and maximum value, so no spike
        is ever lost to uniform striding.

        Why uniform stride is wrong
        ───────────────────────────
        With n=25 M particles and n_out=4 000, uniform stride ≈ 6 258.
        A high-density particle that does not happen to land on a strided
        index is silently dropped.  Min-max keeps every extremum by
        construction.

        Algorithm
        ─────────
        1. Divide y into ⌊n_out/2⌋ equal-width buckets.
        2. In each bucket find argmin and argmax.
        3. Emit both indices in ascending order so the polyline stays
           monotone in x (required for pyqtgraph's clip-to-view to work).

        Parameters
        ──────────
        y     : 1-D array of values (typically raw code units)
        n_out : target number of output points (budget).  Actual output
                may be up to 2 × ⌊n_out/2⌋ — always ≤ n_out + 1.
        x     : optional matching x-coordinate array.  If None, integer
                indices 0..len(y)-1 are used.

        Returns
        ───────
        x_ds, y_ds : downsampled coordinate arrays, same dtype as inputs.
        """
        n = len(y)
        if x is None:
            x = np.arange(n, dtype=np.float64)

        # No downsampling needed
        if n <= n_out:
            return x.copy(), y.copy()

        n_buckets = max(1, n_out // 2)
        bucket_edges = np.linspace(0, n, n_buckets + 1, dtype=np.intp)

        x_ds_list: list[np.ndarray] = []
        y_ds_list: list[np.ndarray] =[]

        for i in range(n_buckets):
            lo = int(bucket_edges[i])
            hi = int(bucket_edges[i + 1])
            if lo >= hi:
                continue

            seg_y = y[lo:hi]
            seg_x = x[lo:hi]

            i_min = int(np.argmin(seg_y))
            i_max = int(np.argmax(seg_y))

            # Emit in ascending index order so x stays monotone
            if i_min <= i_max:
                x_ds_list.append(np.array([seg_x[i_min], seg_x[i_max]]))
                y_ds_list.append(np.array([seg_y[i_min], seg_y[i_max]]))
            else:
                x_ds_list.append(np.array([seg_x[i_max], seg_x[i_min]]))
                y_ds_list.append(np.array([seg_y[i_max], seg_y[i_min]]))

        if not x_ds_list:
            return x[:1].copy(), y[:1].copy()

        return np.concatenate(x_ds_list), np.concatenate(y_ds_list)

    @staticmethod
    def _logspaced_minmax_downsample(
        y:     np.ndarray,
        n_out: int,
        x_in:  np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Peak-preserving downsampling with LOG-SPACED bucket edges.

        When x_in is provided, bucket edges are computed in
        global-index space so that every screen decade of the full N_total
        axis gets equal bucket density.  x_in[j] is the global HDF5 particle
        index for LOD-1 sample j.  The returned x_ds values are global
        indices, matching H5Web's Line-view sample positions exactly.

        When x_in is None (legacy / test paths), implicit 1-based LOD-1
        indices are used — backwards-compatible.

        Parameters
        ──────────
        y     : 1-D float64 array (LOD-1 raw values)
        n_out : output budget; actual count ≤ n_out
        x_in  : optional float64[len(y)] of global particle indices (1-based)

        Returns
        ───────
        x_ds, y_ds : float64 arrays; x_ds contains global particle indices
                     and is strictly monotone for pyqtgraph clip-to-view.
        """
        n = len(y)
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        if n <= n_out:
            if x_in is not None:
                return x_in.astype(np.float64), y.astype(np.float64)
            return np.arange(1, n + 1, dtype=np.float64), y.astype(np.float64)

        n_buckets = max(1, n_out // 2)

        if x_in is not None:
            # ── Global-index bucket edges ─────────────────────────────────────
            # Raw log-spaced edges in global-index space
            x0, x1   = float(x_in[0]), float(x_in[-1])
            raw_edges = np.logspace(
                np.log10(max(x0, 1.0)),
                np.log10(max(x1, 2.0)),
                n_buckets + 1,
            )
            # Map global-index edges to LOD-1 array positions via searchsorted
            lod1_edges = np.searchsorted(x_in, raw_edges, side="left")
            lod1_edges = np.clip(lod1_edges, 0, n)
            int_edges  = np.unique(lod1_edges)
        else:
            # ── Legacy: implicit 1-based LOD-1 indices ──────────────────────
            raw_edges = np.logspace(0.0, np.log10(max(n, 2)), n_buckets + 1)
            int_edges = np.unique(
                np.clip(np.round(raw_edges).astype(np.intp), 1, n)
            )

        x_out: list[float] = []
        y_out: list[float] = []

        for i in range(len(int_edges) - 1):
            if x_in is not None:
                lo_0 = int(int_edges[i])
                hi_0 = min(int(int_edges[i + 1]), n)
            else:
                lo_1 = int(int_edges[i])
                hi_1 = int(int_edges[i + 1])
                lo_0 = lo_1 - 1
                hi_0 = min(hi_1, n)

            if lo_0 >= n:
                break

            seg = y[lo_0:hi_0]
            if len(seg) == 0:
                continue

            if len(seg) == 1:
                x_val = float(x_in[lo_0]) if x_in is not None else float(lo_0 + 1)
                x_out.append(x_val)
                y_out.append(float(seg[0]))
                continue

            i_min = int(np.argmin(seg))
            i_max = int(np.argmax(seg))

            if x_in is not None:
                xv_min = float(x_in[lo_0 + i_min])
                xv_max = float(x_in[lo_0 + i_max])
            else:
                lo_1 = int(int_edges[i])  # re-read for legacy path
                xv_min = float(lo_1 + i_min)
                xv_max = float(lo_1 + i_max)

            if i_min == i_max:
                x_out.append(xv_min)
                y_out.append(float(seg[i_min]))
                continue

            if i_min < i_max:
                x_out.append(xv_min)
                x_out.append(xv_max)
                y_out.append(float(seg[i_min]))
                y_out.append(float(seg[i_max]))
            else:
                x_out.append(xv_max)
                x_out.append(xv_min)
                y_out.append(float(seg[i_max]))
                y_out.append(float(seg[i_min]))

        return np.array(x_out, dtype=np.float64), np.array(y_out, dtype=np.float64)