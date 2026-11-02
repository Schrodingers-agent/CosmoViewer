"""
bh_cuml_accr_mass_plugin.py — PartType5/BH_CumlAccrMass
Cumulative accreted mass onto each black hole particle.
"""
from __future__ import annotations
from plugin_base import BasePlugin

class BHCumlAccrMassPlugin(BasePlugin):
    GROUP    = "PartType5"
    DATASET  = "BH_CumlAccrMass"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType5": ["BH_CumlAccrMass"]}
    DISPLAY_NAME    = "BH Cumul. Accreted Mass"
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

    ALLOW_ZEROS_IN_LOG = True
    MAX_BINS                = 32
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Step"]
