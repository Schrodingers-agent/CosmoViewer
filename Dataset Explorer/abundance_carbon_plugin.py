"""
abundance_carbon_plugin.py — PartType0/ElementAbundance/Carbon
Mass fraction of Carbon in each gas particle.
Solar C mass fraction: 2.36×10⁻³ (Asplund 2009).
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin, MarkerSpec

_SOLAR_C = 2.36e-3


class AbundanceCarbonPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "ElementAbundance/Carbon"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["ElementAbundance/Carbon"]}
    DISPLAY_NAME    = "Carbon Abundance"
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
        return [MarkerSpec(pos=_SOLAR_C * unit_scalar, axis="x",
                           role="threshold_marker",
                           label=f"Solar C = {_SOLAR_C:.2e}", dash=True)]

    def get_custom_info(self, cgs_data):
        n = len(cgs_data)
        n_zero = int(np.sum(cgs_data <= 0))
        return f"Pristine (C=0): {n_zero:,} ({100.*n_zero/n:.1f}%)"
