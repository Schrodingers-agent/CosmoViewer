"""
metallicity_plugin.py
─────────────────────
MetallicityPlugin — PartType0/Metallicity

Declarative schema + hooks for the BasePlugin 6-stage pipeline.

Tabs
----
  Histogram : PDF of metallicity in both mass fraction Z and solar units Z/Z☉
  Line      : raw values vs particle index (min-max downsampled ghost)

Physics notes
─────────────
  • Z = 0 particles are real — they are genuinely metal-free primordial gas.
    ALLOW_ZEROS_IN_LOG = True clips them to 1e-15 in Line log-y mode.
    In Histogram mode, zeros fall below the log bins and are not counted
    (but reported in the info bar via get_custom_info).
  • The distribution is typically bimodal: a narrow peak near Z ~ 0.001–0.003
    (enriched IGM / outer halo) and a broader peak near Z ~ Z☉ (star-forming
    ISM / galaxy centres).
  • Solar metallicity reference: Z☉ = 0.0134, Asplund et al. 2009
    photospheric — the value EAGLE adopts.
"""

from __future__ import annotations

import numpy as np

from plugin_base import BasePlugin, MarkerSpec, RegionSpec
from astro_units import (
    SOLAR_METALLICITY,
    metallicity_to_solar,
)


class MetallicityPlugin(BasePlugin):

    # ── Routing & Identity ────────────────────────────────────────────────────
    GROUP    = "PartType0"
    DATASET  = "Metallicity"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["Metallicity"]}
    DISPLAY_NAME    = "Metallicity"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    # ── Pipeline Configuration ────────────────────────────────────────────────
    # Metallicity is dimensionless — Astropy cannot express Z/Z☉ as a Unit.
    # BASE_CGS_UNIT = None + ALLOWED_UNITS = [] (empty list, NOT the None sentinel)
    # routes the pipeline to the legacy UNIT_DIMENSION path.
    # The plugin always gets final say: custom_unit_transform() takes precedence
    # over all automated scaling regardless of global context.
    BASE_CGS_UNIT  = None   # dimensionless: no Astropy delegation
    ALLOWED_UNITS  = []     # [] = legacy path; None = Astropy safefall (different)
    UNIT_DIMENSION = "metallicity"
    ALLOW_ZEROS_IN_LOG = True     # Z=0 = pristine gas, clip for log-y safety
    LINE_Y_LABEL       = "Metallicity Z  (mass fraction)"
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    # REQUIRES is auto-derived:["PartType0/Metallicity"]

    # ══════════════════════════════════════════════════════════════════════════
    #  HOOKS — metallicity-specific pipeline overrides
    # ══════════════════════════════════════════════════════════════════════════

    def get_custom_info(self, cgs_data: np.ndarray) -> str:
        """Metallicity-specific summary for the info bar."""
        n          = len(cgs_data)
        n_zero     = int(np.sum(cgs_data == 0.0))
        Z_pos      = cgs_data[cgs_data > 0.0]
        median_Z   = float(np.median(Z_pos)) if len(Z_pos) > 0 else 0.0
        median_sol = metallicity_to_solar(median_Z)
        n_above    = int(np.sum(cgs_data > SOLAR_METALLICITY))
        frac       = 100.0 * n_above / n if n > 0 else 0.0
        return (
            f"Z☉ = {SOLAR_METALLICITY:.4f}  |  "
            f"Metal-free (Z=0): {n_zero:,} ({100.*n_zero/n:.1f}%)  |  "
            f"Median Z = {median_Z:.3e} ({median_sol:.2f} Z☉)  |  "
            f"Above solar: {frac:.1f}%"
        )

    def get_custom_markers(
        self, unit_scalar: float, cgs_data: np.ndarray,
    ) -> list[MarkerSpec]:
        """Solar and median markers for Histogram tab."""
        solar_pos = SOLAR_METALLICITY * unit_scalar

        Z_pos      = cgs_data[cgs_data > 0.0]
        median_Z   = float(np.median(Z_pos)) if len(Z_pos) > 0 else 0.0
        median_sol = metallicity_to_solar(median_Z)
        median_pos = median_Z * unit_scalar

        # Marker labels show values in the current display unit
        solar_lbl  = f"Z☉={solar_pos:.4f}"
        median_lbl = f"median={median_pos:.2e} ({median_sol:.2f} Z☉)"

        return[
            MarkerSpec(pos=solar_pos,  axis="x", role="threshold_marker",
                       label=solar_lbl,  dash=True),
            MarkerSpec(pos=median_pos, axis="x", role="median_marker",
                       label=median_lbl, dash=True),
        ]

    def get_custom_line_markers(self) -> list[MarkerSpec]:
        """Solar metallicity horizontal reference for Line tab."""
        return[
            MarkerSpec(pos=SOLAR_METALLICITY, axis="y",
                       role="threshold_marker",
                       label=f"Z☉={SOLAR_METALLICITY:.4f}", dash=True),
        ]

    def get_custom_regions(
        self, unit_scalar: float, vmin: float, vmax: float,
    ) -> list[RegionSpec]:
        """Shade super-solar region in display-unit space."""
        solar_pos = SOLAR_METALLICITY * unit_scalar
        if vmax > solar_pos > 0:
            return[RegionSpec(
                x_min=solar_pos, x_max=vmax, role="highlight_region")]
        return