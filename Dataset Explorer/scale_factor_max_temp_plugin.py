"""
scale_factor_max_temp_plugin.py — PartType0/AExpMaximumTemperature
Scale factor a at which each gas particle reached its maximum temperature.

Creature comfort: custom_info shows the median redshift z = 1/a - 1 of
peak heating — diagnostic for when the CGM was shock-heated.
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin


class ScaleFactorMaxTempPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "AExpMaximumTemperature"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["AExpMaximumTemperature"]}
    DISPLAY_NAME    = "a(T_max)"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.dimensionless_unscaled
        ALLOWED_UNITS = [_u.dimensionless_unscaled]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None

    ALLOW_ZEROS_IN_LOG = False
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    def get_custom_info(self, cgs_data):
        pos = cgs_data[cgs_data > 0]
        if len(pos) == 0:
            return ""
        med_a = float(np.median(pos))
        med_z = 1.0 / med_a - 1.0
        return f"Median a(T_max) = {med_a:.4f}  →  z ≈ {med_z:.2f}"
