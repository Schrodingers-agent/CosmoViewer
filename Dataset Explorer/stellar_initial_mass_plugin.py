"""
stellar_initial_mass_plugin.py — PartType4/InitialMass
Mass of each star particle at the time of formation (before stellar
evolution mass loss).
EAGLE attrs: CGSConversionFactor=1.989e+43, h-scale-exponent=-1.
"""
from __future__ import annotations
from plugin_base import BasePlugin


class StellarInitialMassPlugin(BasePlugin):
    GROUP    = "PartType4"
    DATASET  = "InitialMass"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType4": ["InitialMass"]}
    DISPLAY_NAME    = "Stellar Initial Mass"
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
    MAX_BINS                = 64
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]
