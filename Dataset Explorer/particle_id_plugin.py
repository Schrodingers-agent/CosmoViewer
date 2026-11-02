"""
particle_id_plugin.py — PartType4/ParticleIDs
Unique 64-bit particle identifiers.

Risk mitigation
───────────────
ParticleIDs are uint64 in HDF5 — cast to float64 causes precision
truncation beyond 2^53.  Histogram is scientifically meaningless for
IDs; only the Line tab is enabled.  DEFAULT_SCALE_Y forced to linear.
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin


class ParticleIDPlugin(BasePlugin):
    GROUP    = "PartType4"
    DATASET  = "ParticleIDs"
    TABS     = ["Line"]   # No Histogram — IDs are not a continuous variable
    TREE_CONTRIBUTIONS = {"PartType4": ["ParticleIDs"]}
    DISPLAY_NAME    = "Particle IDs"
    DEFAULT_SCALE_X = "linear"
    DEFAULT_SCALE_Y = "linear"   # linear only — log of IDs is meaningless

    try:
        from astropy import units as _u
        BASE_CGS_UNIT = _u.dimensionless_unscaled
        ALLOWED_UNITS = [_u.dimensionless_unscaled]
        del _u
    except ImportError:
        BASE_CGS_UNIT = None
        ALLOWED_UNITS = None

    ALLOW_ZEROS_IN_LOG = False
    LINE_Y_LABEL = "Particle ID (uint64 → float64)"
    MAX_BINS                = 64
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    def _run_stage1_stage2(self, data, attrs, header, params, context=None):
        r = super()._run_stage1_stage2(data, attrs, header, params, context)
        # Label with the on-disk dtype, and remember how many IDs exist in total
        # so the summary can say when it only describes a sample.
        self._n_global = getattr(r, "n_global", None)
        dtype = getattr(r, "ds_dtype", "")
        if dtype:
            self.LINE_Y_LABEL = f"Particle ID ({dtype} → float64)"
        return r

    def get_custom_info(self, cgs_data):
        id_min = float(np.min(cgs_data))
        id_max = float(np.max(cgs_data))
        n_unique = len(np.unique(cgs_data))
        n = len(cgs_data)
        dup = n - n_unique
        n_global = getattr(self, "_n_global", None) or n
        scope = "" if n >= n_global else f" (sample of {n:,} of {n_global:,}; not a full check)"
        precision = "  |  IDs above 2^53 lose precision as float64" if id_max >= 2.0 ** 53 else ""
        return (f"ID range{'' if n >= n_global else ' in sample'}: [{id_min:.0f}, {id_max:.0f}]  |  "
                f"Unique: {n_unique:,}/{n:,}{scope}"
                + (f"  |  ⚠ {dup:,} duplicates" if dup > 0 else "") + precision)
