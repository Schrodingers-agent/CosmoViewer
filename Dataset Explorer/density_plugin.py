"""
density_plugin.py
─────────────────
DensityPlugin — PartType0/Density

Declarative schema for the BasePlugin 6-stage pipeline.
No compute() override — all logic lives in BasePlugin.

Tabs
----
  Histogram : log-log PDF of density in g/cm³ (physics view)
  Line      : raw code-unit values vs particle index (H5Web Line view)

Astropy Schema (Phase 3)
------------------------
  BASE_CGS_UNIT  = u.g / u.cm**3
    The physical unit the pipeline emits after Stage 2 cosmological
    calibration.  Used by BasePlugin to compute conversion scalars via
    (1.0 * BASE_CGS_UNIT).to(target).value.

  ALLOWED_UNITS  = [u.g/u.cm**3, u.Msun/u.kpc**3, ...]
    Ordered list of display units offered in the UI dropdown.
    Labels are generated dynamically via unit.to_string('unicode').

  Safefall: If astropy is not installed, BASE_CGS_UNIT = None and
  ALLOWED_UNITS = None (sentinel).  The pipeline detects this, bypasses
  Stage 3, and locks the dropdown to "CGS (Astropy Missing)".

Conversion  (EAGLE/VirgoDB cosmological correction)
----------
  EAGLE encodes co-moving densities. The VarDescription reads:
    "Physical rho = Densities h^2 a^-3 U_M U_L^-3[g/cm^3]"

  Full physical CGS conversion (handled by BasePlugin Stage 2):
    ρ_physical = raw × CGSConversionFactor × h^(h_exp) × a^(a_exp)

  For Gadget-4 native files the two exponent attributes are absent
  (defaulting to 0), so the formula reduces to raw × to_cgs — no change.

Line view downsampling
----------------------
  Uses min-max downsampling (BasePlugin._minmax_downsample /
  _logspaced_minmax_downsample) to cap array size for UI thread safety
  while preserving all density spikes — matching H5Web Line-view exactly.
"""

from __future__ import annotations

from plugin_base import BasePlugin


class DensityPlugin(BasePlugin):

    # ── Routing & Identity ────────────────────────────────────────────────────
    GROUP    = "PartType0"
    DATASET  = "Density"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["Density"]}
    DISPLAY_NAME    = "Density"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    # ── Astropy Unit Schema (Phase 3) ─────────────────────────────────────────
    # Safefall: if astropy is absent, BASE_CGS_UNIT = None and
    # ALLOWED_UNITS = None (pipeline sentinel — triggers "CGS (Astropy Missing)").
    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.g / _u.cm**3
        ALLOWED_UNITS = [
            _u.g    / _u.cm**3,
            _u.Msun / _u.kpc**3,
            _u.Msun / _u.pc**3,
            _u.Msun / _u.Mpc**3,
        ]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None   # None = sentinel: Astropy intended but unavailable

    # ── Pipeline Configuration ────────────────────────────────────────────────
    ALLOW_ZEROS_IN_LOG = False    # density is always positive; no clipping needed
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    # No hooks needed — pure scalar pipeline.
    # REQUIRES is auto-derived: ["PartType0/Density"]