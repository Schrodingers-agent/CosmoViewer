"""
bh_accretion_rate_plugin.py — PartType5/BH_Mdot
Instantaneous black hole accretion rate.
EAGLE attrs: CGSConversionFactor=6.4459e+23, h-scale-exponent=-1.

Creature comfort: custom_info reports the dormant (zero-accretion) BH
fraction and median active accretion rate — a key diagnostic for AGN
feedback models.
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin
from astro_units import SOLAR_MASS_CGS, YEAR_CGS


class BHAccretionRatePlugin(BasePlugin):
    GROUP    = "PartType5"
    DATASET  = "BH_Mdot"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType5": ["BH_Mdot"]}
    DISPLAY_NAME    = "BH Accretion Rate"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.g / _u.s
        ALLOWED_UNITS = [_u.g / _u.s, _u.M_sun / _u.yr]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None

    ALLOW_ZEROS_IN_LOG = True
    MAX_BINS                = 32
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Step"]

    # ── Creature comfort: dormant BH diagnostic ──────────────────────────
    def get_custom_info(self, cgs_data):
        n = len(cgs_data)
        n_dormant = int(np.sum(cgs_data <= 0.0))
        active = cgs_data[cgs_data > 0.0]
        if len(active) > 0:
            med_msun_yr = float(np.median(active)) * YEAR_CGS / SOLAR_MASS_CGS
            return (f"Dormant (Ṁ=0): {n_dormant:,} ({100.*n_dormant/n:.1f}%)  |  "
                    f"Median active Ṁ: {med_msun_yr:.2e} M☉/yr")
        return f"Dormant (Ṁ=0): {n_dormant:,} ({100.*n_dormant/n:.1f}%)  |  All BHs dormant"

    def get_custom_status(self, cgs_data, unit_label, unit_scalar):
        active = cgs_data[cgs_data > 0.0]
        if len(active) > 0:
            max_msun_yr = float(np.max(active)) * YEAR_CGS / SOLAR_MASS_CGS
            return f"Peak Ṁ: {max_msun_yr:.2e} M☉/yr"
        return ""
