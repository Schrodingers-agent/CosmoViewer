"""
The run's code units, and G and c expressed in them.

The unit lengths, masses and velocities come from the snapshot's Parameters
group (Gadget-4) or Units group (EAGLE), passed in as ``unit_source``; any that
are missing fall back to Gadget's defaults (1 kpc, 1e10 Msun, 1 km/s). These
are the values per unit as the run states them: in a cosmological run one
length unit is really UnitLength_in_cm / h, one mass unit UnitMass_in_g / h,
and HubbleParam supplies h. G in code units does not depend on h, since the h
factors of length, mass and time cancel (with the default units G is about
43,020 whatever h is).
HubbleParam defaults to 0.7 when the file does not give it.
"""

from typing import Optional


class GadgetUnits:
    """Derives physical constants in code units from snapshot metadata."""

    def __init__(self, unit_source: Optional[dict] = None):
        self.UnitLength_in_cm = 3.085678e21      # 1 kpc
        self.UnitMass_in_g = 1.989e43           # 1e10 Msun
        self.UnitVelocity_in_cm_per_s = 1e5     # 1 km/s
        self.HubbleParam = 0.7

        if unit_source is not None:
            for attr, key in [
                ('UnitLength_in_cm', 'UnitLength_in_cm'),
                ('UnitMass_in_g', 'UnitMass_in_g'),
                ('UnitVelocity_in_cm_per_s', 'UnitVelocity_in_cm_per_s'),
                ('HubbleParam', 'HubbleParam'),
            ]:
                if key in unit_source:
                    setattr(self, attr, float(unit_source[key]))

        self.UnitTime_in_s = self.UnitLength_in_cm / self.UnitVelocity_in_cm_per_s
        self.UnitEnergy_in_erg = self.UnitMass_in_g * self.UnitVelocity_in_cm_per_s ** 2

        G_cgs = 6.67430e-8
        self.G = (G_cgs * self.UnitMass_in_g * self.UnitTime_in_s ** 2
                  / self.UnitLength_in_cm ** 3)

        c_cgs = 2.99792458e10
        self.c = c_cgs / self.UnitVelocity_in_cm_per_s
