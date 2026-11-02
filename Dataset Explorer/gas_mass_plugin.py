"""
gas_mass_plugin.py — PartType0/Mass
SPH particle mass for gas particles.
EAGLE attrs: CGSConversionFactor=1.989e+43, h-scale-exponent=-1.
"""
from __future__ import annotations
from plugin_base import BasePlugin

class GasMassPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "Mass"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["Mass"]}
    DISPLAY_NAME    = "Gas Particle Mass"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.g
        ALLOWED_UNITS = [_u.g, _u.M_sun]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None

    ALLOW_ZEROS_IN_LOG = False
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]
