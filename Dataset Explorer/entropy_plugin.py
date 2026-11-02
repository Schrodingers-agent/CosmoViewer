"""
entropy_plugin.py — PartType0/Entropy
Entropic function A = P / ρ^γ in complex code units.
EAGLE attrs: CGSConversionFactor=1.2970e+30, h-scale-exponent=-1.333.

BASE_CGS_UNIT = None: Entropy has non-trivial composite dimensions
(erg cm^(3γ-3) g^(-γ)) that are code-dependent.  Astropy delegation
is intentionally disabled; data is displayed in raw code units.
"""
from __future__ import annotations
from plugin_base import BasePlugin


class EntropyPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "Entropy"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["Entropy"]}
    DISPLAY_NAME    = "Entropy"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    BASE_CGS_UNIT      = None
    ALLOWED_UNITS      = []
    ALLOW_ZEROS_IN_LOG = False
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]
