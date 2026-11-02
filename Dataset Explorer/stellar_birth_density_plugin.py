"""
stellar_birth_density_plugin.py — PartType4/BirthDensity
Gas density at the location where each star particle formed.
EAGLE attrs: CGSConversionFactor=6.77e-31 (no h/a exponents).

Creature comfort: star formation threshold marker at n_H ~ 0.1 cm⁻³
(Schaye & Dalla Vecchia 2008, EAGLE default).
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin, MarkerSpec
from astro_units import PROTON_MASS_CGS, SOLAR_HYDROGEN_FRAC

# EAGLE SF threshold: n_H = 0.1 cm⁻³ → ρ = n_H × m_p / X
_SF_THRESHOLD_CGS = 0.1 * PROTON_MASS_CGS / SOLAR_HYDROGEN_FRAC


class StellarBirthDensityPlugin(BasePlugin):
    GROUP    = "PartType4"
    DATASET  = "BirthDensity"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType4": ["BirthDensity"]}
    DISPLAY_NAME    = "Birth Density"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.g / _u.cm**3
        ALLOWED_UNITS = [_u.g / _u.cm**3, _u.M_sun / _u.kpc**3, _u.M_sun / _u.pc**3]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None

    ALLOW_ZEROS_IN_LOG = False
    MAX_BINS                = 64
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    # ── Creature comfort: SF density threshold marker ────────────────────
    def get_custom_markers(self, unit_scalar, cgs_data):
        return [
            MarkerSpec(pos=_SF_THRESHOLD_CGS * unit_scalar, axis="x",
                       role="threshold_marker",
                       label="SF threshold (n_H=0.1 cm⁻³)", dash=True),
        ]

    def get_custom_info(self, cgs_data):
        above = int(np.sum(cgs_data >= _SF_THRESHOLD_CGS))
        n = len(cgs_data)
        return f"Above SF threshold: {above:,}/{n:,} ({100.*above/n:.1f}%)"
