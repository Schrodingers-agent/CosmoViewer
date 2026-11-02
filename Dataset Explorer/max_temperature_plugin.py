"""
max_temperature_plugin.py — PartType0/MaximumTemperature
Maximum temperature ever reached by each gas particle (K).

Creature comfort: custom_info splits the particles by the highest temperature
they have ever reached: below 10⁴ K, 10⁴–5×10⁵ K, and above 5×10⁵ K. This is
the thermal history (has the gas ever been heated, e.g. shocked or by
feedback), not its current ISM phase: a particle that was once hot and has
cooled counts as "ever hot". The current phase is the Temperature field's.
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin, MarkerSpec

_T_COLD_WARM = 1e4    # K — cold/warm boundary
_T_WARM_HOT  = 5e5    # K — warm/hot boundary


class MaxTemperaturePlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "MaximumTemperature"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["MaximumTemperature"]}
    DISPLAY_NAME    = "Max Temperature"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.K
        ALLOWED_UNITS = [_u.K]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None

    ALLOW_ZEROS_IN_LOG = False
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    # ── Creature comfort: ISM phase classification ───────────────────────
    def get_custom_info(self, cgs_data):
        n = len(cgs_data)
        if n == 0:
            return ""
        n_cold = int(np.sum(cgs_data < _T_COLD_WARM))
        n_warm = int(np.sum((cgs_data >= _T_COLD_WARM) & (cgs_data < _T_WARM_HOT)))
        n_hot  = int(np.sum(cgs_data >= _T_WARM_HOT))
        return (f"Never above 10⁴ K: {100.*n_cold/n:.1f}%  |  "
                f"Peak 10⁴–5×10⁵ K: {100.*n_warm/n:.1f}%  |  "
                f"Ever above 5×10⁵ K: {100.*n_hot/n:.1f}%  (highest temperature reached, not the current phase)")

    def get_custom_markers(self, unit_scalar, cgs_data):
        return [
            MarkerSpec(pos=_T_COLD_WARM * unit_scalar, axis="x",
                       role="threshold_marker", label="Peak 10⁴ K", dash=True),
            MarkerSpec(pos=_T_WARM_HOT * unit_scalar, axis="x",
                       role="threshold_marker", label="Peak 5×10⁵ K", dash=True),
        ]
