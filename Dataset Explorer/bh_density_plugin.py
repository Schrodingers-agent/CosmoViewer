"""
bh_density_plugin.py — PartType5/BH_Density
Local gas density around each black hole particle.
EAGLE attrs: CGSConversionFactor=6.77e-31, aexp=-3, h=2.
"""
from __future__ import annotations
from plugin_base import BasePlugin

class BHDensityPlugin(BasePlugin):
    GROUP    = "PartType5"
    DATASET  = "BH_Density"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType5": ["BH_Density"]}
    DISPLAY_NAME    = "BH Local Density"
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
    MAX_BINS                = 32
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Step"]
