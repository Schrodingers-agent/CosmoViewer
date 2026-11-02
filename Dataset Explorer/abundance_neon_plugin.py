"""
abundance_neon_plugin.py — PartType0/ElementAbundance/Neon
Mass fraction of Neon. Solar Ne: 1.26×10⁻³ (Asplund 2009: log ε(Ne) = 7.93 with
X☉ = 0.7381, the same way as the other elements' markers; it was 1.40×10⁻³, about 10 %
above that, while the other eight agree with Asplund 2009 within 3 %).
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin, MarkerSpec

_SOLAR_NE = 1.26e-3


class AbundanceNeonPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "ElementAbundance/Neon"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["ElementAbundance/Neon"]}
    DISPLAY_NAME    = "Neon Abundance"
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
        return [MarkerSpec(pos=_SOLAR_NE * unit_scalar, axis="x",
                           role="threshold_marker",
                           label=f"Solar Ne = {_SOLAR_NE:.2e}", dash=True)]

    def get_custom_info(self, cgs_data):
        n = len(cgs_data)
        n_zero = int(np.sum(cgs_data <= 0))
        return f"Pristine (Ne=0): {n_zero:,} ({100.*n_zero/n:.1f}%)"
