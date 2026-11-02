"""
internal_energy_plugin.py — PartType0/InternalEnergy
Specific thermal energy per unit mass (erg/g in CGS).

Creature comfort: custom_unit_transform converts to Temperature (K) via
the Ideal Gas Law — T = (γ-1) × μ × m_p × u / k_B.  The speed of sound
is reported in the status bar as a diagnostic for ICM / CGM studies.

Uses the legacy UNIT_DIMENSION path (not Astropy) because Temperature is
not a simple dimensional conversion from erg/g — it requires knowledge of
the mean molecular weight μ and the adiabatic index γ.

μ per particle: when the snapshot stores ElectronAbundance (n_e / n_H, as
Gadget-4 writes it with cooling on), μ = 4 / (1 + 3X + 4X·n_e/n_H), the
formula Gadget uses.  X is the particle's hydrogen mass fraction from
ElementAbundance/Hydrogen when stored, else 0.76 (Gadget's default; the
file does not record it).  Without ElectronAbundance μ = 0.588 (fully
ionised) and the status bar says it was assumed.
"""
from __future__ import annotations
import numpy as np
from plugin_base import BasePlugin
from astro_units import (
    GAMMA_MONOATOMIC, MU_IONISED_PRIMORDIAL,
    PROTON_MASS_CGS, BOLTZMANN_CGS,
)


HYDROGEN_MASSFRAC_DEFAULT = 0.76

_NE_PATH = "PartType0/ElectronAbundance"
_X_PATH  = "PartType0/ElementAbundance/Hydrogen"


def mean_molecular_weight(ne, x_h=HYDROGEN_MASSFRAC_DEFAULT):
    """μ = 4 / (1 + 3X + 4X·n_e/n_H)  (Gadget's expression)."""
    return 4.0 / (1.0 + 3.0 * x_h + 4.0 * x_h * ne)


class InternalEnergyPlugin(BasePlugin):
    GROUP    = "PartType0"
    DATASET  = "InternalEnergy"
    # Optional reads; ComputeWorker skips paths absent from the snapshot.
    REQUIRES = ["PartType0/InternalEnergy", _NE_PATH, _X_PATH]
    TABS     = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS = {"PartType0": ["InternalEnergy"]}
    DISPLAY_NAME    = "Internal Energy"
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"

    # Legacy path: Temperature conversion requires physics, not Astropy.
    BASE_CGS_UNIT  = None
    ALLOWED_UNITS  = []
    UNIT_DIMENSION = "internal_energy"
    ALLOW_ZEROS_IN_LOG = False
    MAX_BINS                = 100
    DEFAULT_STYLE_HISTOGRAM = ["Line", "Linear"]

    # ── μ from the snapshot when it can be computed ──────────────────────
    def _run_stage1_stage2(self, data, attrs, header, params, context=None):
        r = super()._run_stage1_stage2(data, attrs, header, params, context)
        cgs = getattr(r, "cgs_data", None)
        if cgs is None:
            return r
        ne = data.get(_NE_PATH)
        x  = data.get(_X_PATH)
        mu, source = MU_IONISED_PRIMORDIAL, "μ = 0.588 assumed (fully ionised; no ElectronAbundance)"
        if ne is not None and len(ne) == len(cgs):
            if x is not None and len(x) == len(cgs):
                mu = mean_molecular_weight(np.asarray(ne, float), np.asarray(x, float))
                source = "μ per particle from ElectronAbundance and hydrogen fraction"
            else:
                mu = mean_molecular_weight(np.asarray(ne, float))
                source = "μ per particle from ElectronAbundance (X = 0.76 assumed)"
        elif ne is not None:
            source = ("μ = 0.588 assumed: ElectronAbundance sample does not line up "
                      "with InternalEnergy")
        # Keyed by the array itself: the cache path re-uses r.cgs_data, and a
        # different render (other tab, other sample) gets its own entry.
        self._mu_by_array = [(cgs, mu, source)] + [
            e for e in getattr(self, "_mu_by_array", []) if e[0] is not cgs][:3]
        return r

    def _mu_for(self, cgs_array):
        for arr, mu, source in getattr(self, "_mu_by_array", []):
            if arr is cgs_array:
                return mu, source
        return MU_IONISED_PRIMORDIAL, "μ = 0.588 assumed (fully ionised)"

    # ── Creature comfort: Temperature via Ideal Gas Law ──────────────────
    def custom_unit_transform(self, cgs_array, target_unit):
        if target_unit == "Temperature (K)":
            mu, source = self._mu_for(cgs_array)
            T = ((GAMMA_MONOATOMIC - 1.0) * mu
                 * PROTON_MASS_CGS * cgs_array / BOLTZMANN_CGS)
            self._last_T = (T, cgs_array, source)   # get_custom_status receives T
            return T
        return None

    def get_custom_status(self, cgs_data, unit_label, unit_scalar):
        # With "Temperature (K)" selected the pipeline passes T, not u.
        last = getattr(self, "_last_T", None)
        if last is not None and last[0] is cgs_data:
            T, u_cgs, source = last
        else:
            u_cgs = cgs_data
            mu, source = self._mu_for(u_cgs)
            T = ((GAMMA_MONOATOMIC - 1.0) * mu * PROTON_MASS_CGS * u_cgs / BOLTZMANN_CGS)
        ok = u_cgs > 0
        if not ok.any():
            return ""
        med_u = float(np.median(u_cgs[ok]))
        # Adiabatic sound speed: c_s = sqrt(γ(γ-1)u)  [cm/s → km/s]
        cs_kms = (GAMMA_MONOATOMIC * (GAMMA_MONOATOMIC - 1.0) * med_u) ** 0.5 / 1e5
        med_T = float(np.median(T[ok]))
        return f"Median T ≈ {med_T:.2e} K ({source})  |  Median cₛ ≈ {cs_kms:.1f} km/s"
