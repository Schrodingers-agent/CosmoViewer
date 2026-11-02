"""
scale_factor_at_birth_plugin.py — PartType4/StellarFormationTime

Cosmological runs store the scale factor a at which each star formed;
custom_info converts the median a → redshift z = 1/a - 1.

Non-cosmological runs (ComovingIntegrationOn = 0) store the formation time
in code time units instead.  Reading that as a would invent redshifts, so
the plugin then converts it to seconds (UnitLength / UnitVelocity) and
offers Gyr / Myr.
"""
from __future__ import annotations
import dataclasses
import numpy as np
from plugin_base import BasePlugin

_GYR_S = 3.15576e16


class ScaleFactorAtBirthPlugin(BasePlugin):
    GROUP    = "PartType4"
    DATASET  = "StellarFormationTime"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType4": ["StellarFormationTime"]}
    DISPLAY_NAME    = "Formation Scale Factor"
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
    MAX_BINS                = 64
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    def _run_stage1_stage2(self, data, attrs, header, params, context=None):
        r = super()._run_stage1_stage2(data, attrs, header, params, context)
        if not dataclasses.is_dataclass(r) or r.is_cosmological:
            self._is_time = False
            return r
        merged = {**header, **params}
        unit_time_s = (float(merged.get("UnitLength_in_cm", 3.085678e21))
                       / float(merged.get("UnitVelocity_in_cm_per_s", 1.0e5)))
        # Instance overrides: this file's values are times, not scale factors.
        self._is_time = True
        try:
            from astropy import units as u
            self.BASE_CGS_UNIT = u.s
            self.ALLOWED_UNITS = [u.Gyr, u.Myr, u.s]
        except ImportError:
            self.BASE_CGS_UNIT = None
            self.ALLOWED_UNITS = None
        self.DISPLAY_NAME = "Formation Time"
        return dataclasses.replace(r, cgs_data=r.raw * unit_time_s, to_cgs=unit_time_s,
                                   cgs_factor=unit_time_s)

    def get_custom_info(self, cgs_data):
        if getattr(self, "_is_time", False):
            if len(cgs_data) == 0:
                return ""
            return (f"Non-cosmological run: values are formation times, not scale factors  |  "
                    f"Median t_form = {np.median(cgs_data) / _GYR_S:.4f} Gyr  |  "
                    f"Earliest: {np.min(cgs_data) / _GYR_S:.4g} Gyr")
        pos = cgs_data[cgs_data > 0]
        if len(pos) == 0:
            return ""
        med_a = float(np.median(pos))
        med_z = 1.0 / med_a - 1.0
        earliest = float(np.min(pos))
        z_earliest = 1.0 / earliest - 1.0
        return (f"Median a_form = {med_a:.4f} (z ≈ {med_z:.2f})  |  "
                f"Earliest: a = {earliest:.4f} (z ≈ {z_earliest:.1f})")
