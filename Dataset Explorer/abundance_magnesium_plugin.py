"""
abundance_magnesium_plugin.py — PartType0/ElementAbundance/Magnesium
Mass fraction of Magnesium. Solar Mg: 7.08×10⁻⁴ (Asplund 2009).
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin, MarkerSpec

_SOLAR_MG = 7.08e-4


class AbundanceMagnesiumPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "ElementAbundance/Magnesium"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["ElementAbundance/Magnesium"]}
    DISPLAY_NAME    = "Magnesium Abundance"
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

    ALLOW_ZEROS_IN_LOG = True
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    def get_custom_markers(self, unit_scalar, cgs_data):
        return [MarkerSpec(pos=_SOLAR_MG * unit_scalar, axis="x",
                           role="threshold_marker",
                           label=f"Solar Mg = {_SOLAR_MG:.2e}", dash=True)]

    def get_custom_info(self, cgs_data):
        n = len(cgs_data)
        n_zero = int(np.sum(cgs_data <= 0))
        return f"Pristine (Mg=0): {n_zero:,} ({100.*n_zero/n:.1f}%)"
