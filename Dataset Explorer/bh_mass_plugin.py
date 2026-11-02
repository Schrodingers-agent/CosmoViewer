"""
bh_mass_plugin.py — black hole masses.

BHMassPlugin        PartType5/BH_Mass : subgrid black hole mass (EAGLE).
                    This is the physical SMBH mass; reference markers for
                    Sgr A* and M87* belong here.
BHParticleMassPlugin PartType5/Mass   : dynamical particle mass, which in
                    EAGLE includes the gas reservoir and is not the BH mass.
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin, MarkerSpec
from astro_units import SOLAR_MASS_CGS

# Reference SMBH masses (g)
_SGR_A_STAR_CGS = 4.15e6 * SOLAR_MASS_CGS   # Sgr A*  ~4.15×10⁶ M☉
_M87_CGS        = 6.5e9  * SOLAR_MASS_CGS   # M87*    ~6.5×10⁹ M☉


class BHMassPlugin(BasePlugin):
    GROUP    = "PartType5"
    DATASET  = "BH_Mass"
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType5": ["BH_Mass"]}
    DISPLAY_NAME    = "BH Mass (subgrid)"
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
    MAX_BINS                = 32
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Step"]

    # ── Creature comfort: SMBH reference markers ─────────────────────────
    def get_custom_markers(self, unit_scalar, cgs_data):
        return [
            MarkerSpec(pos=_SGR_A_STAR_CGS * unit_scalar, axis="x",
                       role="threshold_marker",
                       label="Sgr A* (4.15×10⁶ M☉)", dash=True),
            MarkerSpec(pos=_M87_CGS * unit_scalar, axis="x",
                       role="threshold_marker",
                       label="M87* (6.5×10⁹ M☉)", dash=True),
        ]

    def get_custom_info(self, cgs_data):
        med = float(np.median(cgs_data))
        med_sol = med / SOLAR_MASS_CGS
        return f"Median BH mass: {med_sol:.2e} M☉"


class BHParticleMassPlugin(BHMassPlugin):
    DATASET  = "Mass"
    TREE_CONTRIBUTIONS = {"PartType5": ["Mass"]}
    DISPLAY_NAME    = "BH Particle Mass (dynamical)"

    def get_custom_markers(self, unit_scalar, cgs_data):
        return []          # SMBH references compare to BH_Mass, not this

    def get_custom_info(self, cgs_data):
        med_sol = float(np.median(cgs_data)) / SOLAR_MASS_CGS
        return (f"Median particle mass: {med_sol:.2e} M☉  |  "
                f"dynamical mass, not the subgrid BH mass (see BH_Mass)")
