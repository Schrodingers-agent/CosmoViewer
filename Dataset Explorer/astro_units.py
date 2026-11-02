"""
astro_units.py
──────────────
Centralised astrophysical constants and unit-conversion helpers.

Rules
─────
  1. No Qt. No h5py. No numpy. No plugin logic.
     This file must import with zero dependencies beyond the stdlib.
  2. Every constant carries a source comment and CGS units.
  3. Conversion helpers are pure functions: in → out, no side-effects.
  4. All helpers guard against unphysical inputs and return NaN rather
     than raising, so plugins never crash on bad data — they just show NaN.

Source abbreviations used below
────────────────────────────────
  CODATA-2018  : NIST CODATA 2018 recommended values
  ASPLUND-2009 : Asplund et al. 2009, ARA&A 47, 481  (protosolar abundances)
  PDG-2022     : Particle Data Group 2022
  IAU-2015     : IAU 2015 nominal solar values (Resolution B3)
"""

from __future__ import annotations

import math


# ═════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL CONSTANTS  (all in CGS unless noted)
# ═════════════════════════════════════════════════════════════════════════════

# ── Mechanics ─────────────────────────────────────────────────────────────────
SPEED_OF_LIGHT_CGS:  float = 2.99792458e10    # cm/s          exact, CODATA-2018
GRAVITATIONAL_G_CGS: float = 6.67430e-8       # cm³/(g·s²)    CODATA-2018

# ── Thermodynamics ────────────────────────────────────────────────────────────
BOLTZMANN_CGS:       float = 1.380649e-16     # erg/K          exact, CODATA-2018
PLANCK_CGS:          float = 6.62607015e-27   # erg·s          exact, CODATA-2018

# ── Particle masses ───────────────────────────────────────────────────────────
PROTON_MASS_CGS:     float = 1.67262192369e-24  # g             CODATA-2018
ELECTRON_MASS_CGS:   float = 9.1093837015e-28   # g             CODATA-2018
ATOMIC_MASS_UNIT:    float = 1.66053906660e-24  # g  (1 u)      CODATA-2018

# ── Equation of state ─────────────────────────────────────────────────────────
GAMMA_MONOATOMIC:    float = 5.0 / 3.0   # adiabatic index for ideal monatomic gas
GAMMA_DIATOMIC:      float = 7.0 / 5.0   # adiabatic index for ideal diatomic gas

# Mean molecular weights for common plasma/gas compositions
#   Fully ionised primordial (76 % H, 24 % He by mass)
MU_IONISED_PRIMORDIAL: float = 0.588
#   Neutral primordial
MU_NEUTRAL_PRIMORDIAL: float = 1.22
#   Fully ionised solar (accounts for metals)
MU_IONISED_SOLAR:      float = 0.588   # metals contribute negligibly to μ


# ═════════════════════════════════════════════════════════════════════════════
#  ASTRONOMICAL LENGTH UNITS  (in cm)
# ═════════════════════════════════════════════════════════════════════════════

PARSEC_CGS:           float = 3.085677581491367e18   # cm/pc    IAU exact
KILOPARSEC_CGS:       float = PARSEC_CGS * 1.0e3     # cm/kpc
MEGAPARSEC_CGS:       float = PARSEC_CGS * 1.0e6     # cm/Mpc
AU_CGS:               float = 1.495978707e13         # cm/AU    IAU exact


# ═════════════════════════════════════════════════════════════════════════════
#  SOLAR / STELLAR REFERENCE VALUES
# ═════════════════════════════════════════════════════════════════════════════

SOLAR_MASS_CGS:       float = 1.98841586e33     # g         IAU-2015 nominal
SOLAR_LUMINOSITY_CGS: float = 3.828e33          # erg/s     IAU-2015 nominal
SOLAR_RADIUS_CGS:     float = 6.957e10          # cm        IAU-2015 nominal

# Solar metallicity: total heavy-element mass fraction Z☉
# Asplund et al. 2009 (protosolar, accounting for diffusion):  Z☉ = 0.0142
# Asplund et al. 2009 (present-day photospheric):              Z☉ = 0.0134
# EAGLE / VirgoDB simulations adopt the photospheric value.
SOLAR_METALLICITY:    float = 0.0134   # ASPLUND-2009 photospheric

# Solar hydrogen mass fraction (Asplund 2009 protosolar)
SOLAR_HYDROGEN_FRAC:  float = 0.7381   # X☉
# Solar helium mass fraction
SOLAR_HELIUM_FRAC:    float = 0.2485   # Y☉


# ═════════════════════════════════════════════════════════════════════════════
#  TIME UNITS  (in seconds)
# ═════════════════════════════════════════════════════════════════════════════

YEAR_CGS:         float = 3.15576e7    # s/yr   Julian year (365.25 d)
MEGAYEAR_CGS:     float = YEAR_CGS * 1.0e6
GIGAYEAR_CGS:     float = YEAR_CGS * 1.0e9
HUBBLE_TIME_CGS:  float = GIGAYEAR_CGS * 13.8   # s  approximate age of Universe


# ═════════════════════════════════════════════════════════════════════════════
#  COSMOLOGICAL DEFAULTS
# ═════════════════════════════════════════════════════════════════════════════

# Planck 2018 / EAGLE cosmology defaults — override from Header when available
HUBBLE_PARAM_DEFAULT: float = 0.6777   # h  (dimensionless)
OMEGA_M_DEFAULT:      float = 0.307
OMEGA_LAMBDA_DEFAULT: float = 0.693
OMEGA_B_DEFAULT:      float = 0.0482


# ═════════════════════════════════════════════════════════════════════════════
#  UNIT-CONVERSION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
#
#  Convention
#  ──────────
#  All helpers return NaN (not raise) on unphysical input so plugins can
#  pass raw simulation arrays without pre-filtering.
#  Use math.isnan() / np.isnan() on results if you need to flag bad values.


# ── Metallicity ──────────────────────────────────────────────────────────────

def metallicity_to_solar(z_fraction: float) -> float:
    """
    Convert a mass-fraction metallicity Z to solar units Z/Z☉.

    Parameters
    ----------
    z_fraction : float
        Metal mass fraction Z (dimensionless, typically 0–0.05).

    Returns
    -------
    float  — Z / Z☉   (1.0 = solar metallicity)
    NaN    — if z_fraction < 0 (unphysical) or Z☉ = 0 (guard).
    """
    if z_fraction < 0.0 or SOLAR_METALLICITY <= 0.0:
        return math.nan
    return z_fraction / SOLAR_METALLICITY


def metallicity_to_solar_log(z_fraction: float) -> float:
    """
    Convert a mass-fraction metallicity Z to log₁₀(Z/Z☉).

    Returns
    -------
    float  — log₁₀(Z / Z☉)
    NaN    — if z_fraction ≤ 0 (can't take log of zero/negative).
    """
    if z_fraction <= 0.0 or SOLAR_METALLICITY <= 0.0:
        return math.nan
    return math.log10(z_fraction / SOLAR_METALLICITY)


# ── Temperature ───────────────────────────────────────────────────────────────

def internal_energy_to_temperature(
        u_cgs:  float,
        mu:     float = MU_IONISED_PRIMORDIAL,
        gamma:  float = GAMMA_MONOATOMIC,
) -> float:
    """
    Convert specific internal energy u [erg/g] → temperature T[K].

    Derivation
    ──────────
    For an ideal gas:  u = (1/(γ-1)) × (k_B T) / (μ m_p)
    Rearranging:       T = (γ-1) × μ × m_p × u / k_B

    Parameters
    ----------
    u_cgs : float
        Specific internal energy in erg/g (code units × CGSConversionFactor).
    mu    : float
        Mean molecular weight (dimensionless).  Default: fully ionised primordial.
    gamma : float
        Adiabatic index.  Default: 5/3 (monatomic ideal gas).

    Returns
    -------
    float  — Temperature in Kelvin.
    NaN    — if u_cgs ≤ 0 or any parameter is non-positive.
    """
    if u_cgs <= 0.0 or mu <= 0.0 or gamma <= 1.0 or BOLTZMANN_CGS <= 0.0:
        return math.nan
    return (gamma - 1.0) * mu * PROTON_MASS_CGS * u_cgs / BOLTZMANN_CGS


def temperature_to_internal_energy(
        T_K:    float,
        mu:     float = MU_IONISED_PRIMORDIAL,
        gamma:  float = GAMMA_MONOATOMIC,
) -> float:
    """
    Inverse of internal_energy_to_temperature: T [K] → u [erg/g].

    Returns
    -------
    float  — Specific internal energy in erg/g.
    NaN    — if T_K ≤ 0 or any parameter is non-positive.
    """
    if T_K <= 0.0 or mu <= 0.0 or gamma <= 1.0:
        return math.nan
    return BOLTZMANN_CGS * T_K / ((gamma - 1.0) * mu * PROTON_MASS_CGS)


# ── Star formation rate ───────────────────────────────────────────────────────

def sfr_cgs_to_solar_per_year(sfr_cgs: float) -> float:
    """
    Convert star-formation rate from CGS (g/s) to M☉/yr.

    Parameters
    ----------
    sfr_cgs : float
        Star-formation rate in g/s.

    Returns
    -------
    float  — SFR in M☉/yr.
    NaN    — if sfr_cgs < 0.
    """
    if sfr_cgs < 0.0 or SOLAR_MASS_CGS <= 0.0 or YEAR_CGS <= 0.0:
        return math.nan
    return sfr_cgs * YEAR_CGS / SOLAR_MASS_CGS


def sfr_solar_per_year_to_cgs(sfr_msun_yr: float) -> float:
    """Inverse of sfr_cgs_to_solar_per_year."""
    if sfr_msun_yr < 0.0:
        return math.nan
    return sfr_msun_yr * SOLAR_MASS_CGS / YEAR_CGS


# ── Density ───────────────────────────────────────────────────────────────────

def density_cgs_to_hydrogen_number(
        rho_cgs: float,
        X:       float = SOLAR_HYDROGEN_FRAC,
) -> float:
    """
    Convert mass density ρ [g/cm³] → hydrogen number density n_H [cm⁻³].

    n_H = X × ρ / m_p

    Parameters
    ----------
    rho_cgs : float
        Physical mass density in g/cm³.
    X       : float
        Hydrogen mass fraction (dimensionless).  Default: solar X☉.

    Returns
    -------
    float  — Hydrogen number density in cm⁻³.
    NaN    — if rho_cgs ≤ 0 or X ≤ 0.
    """
    if rho_cgs <= 0.0 or X <= 0.0:
        return math.nan
    return X * rho_cgs / PROTON_MASS_CGS


def density_cgs_to_mean_number(
        rho_cgs: float,
        mu:      float = MU_IONISED_PRIMORDIAL,
) -> float:
    """
    Convert mass density ρ [g/cm³] → mean particle number density n [cm⁻³].

    n = ρ / (μ × m_p)

    Returns
    -------
    float  — Number density in cm⁻³.
    NaN    — if rho_cgs ≤ 0.
    """
    if rho_cgs <= 0.0 or mu <= 0.0:
        return math.nan
    return rho_cgs / (mu * PROTON_MASS_CGS)


# ── Pressure ─────────────────────────────────────────────────────────────────

def pressure_from_density_energy(
        rho_cgs: float,
        u_cgs:   float,
        gamma:   float = GAMMA_MONOATOMIC,
) -> float:
    """
    Ideal-gas pressure from density and specific internal energy.

    P = (γ - 1) × ρ × u

    Returns
    -------
    float  — Pressure in dyn/cm² (= erg/cm³).
    NaN    — if either input ≤ 0.
    """
    if rho_cgs <= 0.0 or u_cgs <= 0.0 or gamma <= 1.0:
        return math.nan
    return (gamma - 1.0) * rho_cgs * u_cgs


# ── Cosmological scale factor ─────────────────────────────────────────────────

def scale_factor_to_redshift(a: float) -> float:
    """
    Convert scale factor a → redshift z = 1/a - 1.

    Returns
    -------
    float  — Redshift z.
    NaN    — if a ≤ 0.
    """
    if a <= 0.0:
        return math.nan
    return 1.0 / a - 1.0


def redshift_to_scale_factor(z: float) -> float:
    """
    Convert redshift z → scale factor a = 1/(1+z).

    Returns
    -------
    float  — Scale factor a.
    NaN    — if z < -1 (unphysical).
    """
    if z < -1.0:
        return math.nan
    return 1.0 / (1.0 + z)


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK REFERENCE  (human-readable summary, printed on direct execution)
# ─────────────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — COSMOLOGY CONTEXT  (astropy.cosmology)
# ═════════════════════════════════════════════════════════════════════════════
#
#  cosmology_context() is the single entry point for the "Time Machine" bar.
#  It accepts the raw HDF5 Header and Parameters attribute dicts, reads
#  ComovingIntegrationOn itself, and returns a complete ctx dict the UI can
#  display directly.
#
#  Non-cosmological guard  (PM Snag 1)
#  ─────────────────────────────────────
#  If ComovingIntegrationOn == 0 or is absent, the run is an isolated box
#  (e.g. galaxy collision).  Calling Planck18.age(0) would print "13.8 Gyr"
#  which is nonsense for a Newtonian simulation.  Instead we compute the
#  raw physical simulation time:
#      t_phys [s] = Header/Time × UnitLength_in_cm / UnitVelocity_in_cm_per_s
#      t_gyr      = t_phys / GIGAYEAR_CGS
#
#  Astropy import guard
#  ────────────────────
#  All astropy calls are inside try/except.  If astropy is absent, ctx["error"]
#  is set to the ImportError message and age_gyr / lookback_gyr are None.
#  The UI gracefully shows the error note — it never crashes.
#
#  No Quantity objects escape
#  ─────────────────────────
#  .value is extracted immediately from every astropy result so this function
#  always returns plain Python floats.


def cosmology_context(header: dict, params: dict) -> dict:
    """
    Single entry point for the Time Machine header bar.

    Reads ComovingIntegrationOn from the merged header+params dict, then
    either computes the physical simulation time (non-cosmological) or calls
    Planck18.age / lookback_time (cosmological).

    Parameters
    ----------
    header : dict   coerced HDF5 Header group attributes
    params : dict   coerced HDF5 Parameters group attributes

    Returns
    -------
    dict with keys
      is_cosmological : bool
      age_gyr         : float | None   — None if non-cosmo or astropy absent
      lookback_gyr    : float | None
      cosmo_name      : str  | None    — "Planck18" for cosmo runs
      time_gyr        : float | None   — physical time for non-cosmo runs [Gyr]
      time_code       : float          — raw Header/Time value
      redshift        : float          — Header/Redshift (0 for non-cosmo)
      error           : str  | None    — None on success; message on failure
    """
    from data_utils import run_cosmology
    merged    = {**header, **params}
    run       = run_cosmology(header, params)
    is_cosmo  = run["is_cosmological"]
    redshift  = float(header.get("Redshift", 0.0))
    time_code = float(header.get("Time",     0.0))

    # ── Non-cosmological path ─────────────────────────────────────────────────
    if not is_cosmo:
        ul       = float(merged.get("UnitLength_in_cm",         3.085678e21))
        uv       = float(merged.get("UnitVelocity_in_cm_per_s", 1.0e5))
        time_gyr = (time_code * (ul / uv) / GIGAYEAR_CGS
                    if uv > 0 and GIGAYEAR_CGS > 0 else None)
        return {
            "is_cosmological": False,
            "age_gyr":         None,
            "lookback_gyr":    None,
            "cosmo_name":      None,
            "time_gyr":        time_gyr,
            "time_code":       time_code,
            "redshift":        redshift,
            "error":           None,
        }

    # ── Cosmological path: the file's own Ω0, ΩΛ, h; Planck18 only if absent ─
    try:
        if redshift < 0.0:
            raise ValueError(f"Unphysical redshift z = {redshift:.6f}")
        if run["omega0"] is not None and run["omega_lambda"] is not None and run["h_known"]:
            from astropy.cosmology import LambdaCDM   # lazy — inside try
            ob = run["omega_baryon"]
            cosmo = LambdaCDM(H0=100.0 * run["h"], Om0=run["omega0"], Ode0=run["omega_lambda"],
                              Ob0=ob if ob else None, Tcmb0=0.0)
            name = (f"file: h={run['h']:.4f}, Ωm={run['omega0']:.4f}, "
                    f"ΩΛ={run['omega_lambda']:.4f}")
        else:
            from astropy.cosmology import Planck18 as cosmo   # lazy — inside try
            name = "Planck18 (file has no Ω0/ΩΛ/h)"
        # .value extracted immediately — no Quantity object escapes this function
        return {
            "is_cosmological": True,
            "age_gyr":         float(cosmo.age(redshift).value),
            "lookback_gyr":    float(cosmo.lookback_time(redshift).value),
            "cosmo_name":      name,
            "time_gyr":        None,
            "time_code":       time_code,
            "redshift":        redshift,
            "error":           None,
        }
    except Exception as exc:
        return {
            "is_cosmological": True,
            "age_gyr":         None,
            "lookback_gyr":    None,
            "cosmo_name":      None,
            "time_gyr":        None,
            "time_code":       time_code,
            "redshift":        redshift,
            "error":           str(exc),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — UNIT CONVERSION  (astropy.units, scalar quarantine)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Architecture: strict Astropy quarantine
#  ────────────────────────────────────────
#  Astropy's Quantity objects must NEVER reach pyqtgraph.  Attaching units
#  to a 25 M-element array produces a Quantity array; passing that to
#  pyqtgraph causes GUI thread stalls.
#
#  Rule: Astropy is used ONLY to compute one float scalar per conversion.
#  The multiplication   converted = raw_array * scalar   is always pure numpy.
#
#  Two code paths
#  ──────────────
#  ① Astropy path  (dimensional quantities — density, velocity, etc.)
#       Plugins declare BASE_CGS_UNIT and ALLOWED_UNITS as astropy Unit objects.
#       astropy_unit_choices() / astropy_unit_scalar() handle them dynamically.
#  ② Legacy path   (dimensionless quantities — metallicity, and any niche plugin)
#       Plugins declare UNIT_DIMENSION (str) and use get_unit_choices() /
#       get_unit_scalar().  UNIT_DIMENSION is retained as a live, active field
#       so edge-case plugins can supply custom string-based unit registries.
#
#  Base units (what each plugin emits before conversion)
#  ──────────────────────────────────────────────────────
#    density     → g cm⁻³          (physical; cosmological correction applied)
#    metallicity → dimensionless Z  (mass fraction; no correction needed)

# ── Legacy label registry (UNIT_DIMENSION plugins only) ──────────────────────
# Density was removed — it is now handled by the Astropy path.
# Niche plugins that define UNIT_DIMENSION can add entries here.
_LEGACY_UNIT_LABELS: dict[str, list[tuple[str, str]]] = {
    "metallicity": [
        ("Z  (mass fraction)", "Metallicity Z  (mass fraction)"),
        ("Z / Z☉",             f"Metallicity  (Z / Z☉,  Z☉ = {SOLAR_METALLICITY:.4f})"),
    ],
    "internal_energy": [
        ("erg / g",            "Specific Internal Energy  (erg g⁻¹)"),
        ("Temperature (K)",    "Temperature  (K)"),
    ],
}


def get_unit_choices(dimension: str) -> list[str]:
    """
    Return menu labels for legacy UNIT_DIMENSION plugins (e.g. metallicity).

    Dimensional quantities (density, etc.) use astropy_unit_choices() instead.
    UNIT_DIMENSION is retained as a live field for niche / custom-dimension plugins.
    """
    return [lbl for lbl, _ in _LEGACY_UNIT_LABELS.get(dimension, [])]


def get_unit_scalar(dimension: str, target_label: str) -> tuple[float, str]:
    """
    Return (scalar, axis_label) for legacy UNIT_DIMENSION plugins.

    scalar     — pure float; multiply the raw numpy array by this value.
                 Never returns an astropy Quantity.
    axis_label — formatted axis label string for the pyqtgraph plot.

    Always returns (1.0, fallback_label) on unknown dimension or label
    so callers never crash on missing entries.

    Note: dimensional quantities (density, etc.) now use astropy_unit_scalar()
    which delegates to Astropy directly.  This function only handles
    dimensionless dimensions (metallicity) and any niche UNIT_DIMENSION plugins.
    """
    entries      = _LEGACY_UNIT_LABELS.get(dimension, [])
    default_axis = entries[0][1] if entries else (target_label or dimension)

    matched_axis = default_axis
    for lbl, axis in entries:
        if lbl == target_label:
            matched_axis = axis
            break
    else:
        return 1.0, default_axis   # unknown label → identity

    # Metallicity: dimensionless — pure Python arithmetic, no Astropy needed.
    if dimension == "metallicity":
        if target_label == "Z / Z☉" and SOLAR_METALLICITY > 0.0:
            return float(1.0 / SOLAR_METALLICITY), matched_axis
        return 1.0, matched_axis

    # Generic niche dimension: return identity scalar, use matched axis label.
    return 1.0, matched_axis


# ── Astropy-native helpers (Phase 3 — Astropy Delegation) ────────────────────
#
#  These two functions are the ONLY entry points for dimensional unit
#  conversion.  They are called by the BasePlugin Stage 3 Astropy path.
#  Astropy Quantity objects never escape — .value is extracted immediately.


def astropy_unit_choices(allowed_units: list) -> list[str]:
    """
    Generate UI dropdown labels from a plugin's ALLOWED_UNITS list.

    Uses Astropy's unit.to_string('unicode') for human-readable labels
    (e.g. 'g / cm³', 'M☉ / kpc³').  Falls back to str(unit) if to_string
    raises, so the dropdown never crashes on malformed unit objects.

    Parameters
    ----------
    allowed_units : list of astropy Unit objects

    Returns
    -------
    list[str] — one label per unit, matching the order of allowed_units.
    """
    labels = []
    for u_obj in allowed_units:
        try:
            labels.append(u_obj.to_string("unicode"))
        except Exception:
            labels.append(str(u_obj))
    return labels


def astropy_unit_scalar(
    base_cgs_unit,
    target_unit,
) -> tuple[float, str]:
    """
    Compute the float multiplier to convert from base_cgs_unit to target_unit.

    Returns (scalar, axis_label).  scalar is a plain Python float; target_unit
    is used only to produce the axis label string — no Quantity object escapes.

    Safe: catches astropy.units.UnitConversionError (and any other exception)
    and returns (1.0, label) with a terminal warning so the GUI thread never
    crashes on an incompatible plugin unit schema.

    Parameters
    ----------
    base_cgs_unit : astropy Unit   — the CGS base unit the plugin emits
    target_unit   : astropy Unit   — the display unit selected by the user

    Returns
    -------
    (float, str) — (scalar multiplier, axis label string)
    """
    try:
        label = target_unit.to_string("unicode")
    except Exception:
        label = str(target_unit)

    try:
        scalar = float((1.0 * base_cgs_unit).to(target_unit).value)
        return scalar, label
    except Exception as exc:
        import warnings
        warnings.warn(
            f"[CosmoViewer] UnitConversionError: {exc}  "
            f"({base_cgs_unit} → {target_unit})  — scalar falls back to 1.0",
            stacklevel=2,
        )
        return 1.0, label


if __name__ == "__main__":
    rows =[
        ("BOLTZMANN_CGS",         f"{BOLTZMANN_CGS:.6e}",       "erg/K"),
        ("PROTON_MASS_CGS",       f"{PROTON_MASS_CGS:.6e}",     "g"),
        ("SOLAR_MASS_CGS",        f"{SOLAR_MASS_CGS:.6e}",      "g"),
        ("SOLAR_METALLICITY Z☉",  f"{SOLAR_METALLICITY:.4f}",   "dimensionless"),
        ("SOLAR_HYDROGEN_FRAC X☉",f"{SOLAR_HYDROGEN_FRAC:.4f}", "dimensionless"),
        ("PARSEC_CGS",            f"{PARSEC_CGS:.6e}",          "cm"),
        ("YEAR_CGS",              f"{YEAR_CGS:.6e}",            "s"),
        ("GAMMA_MONOATOMIC",      f"{GAMMA_MONOATOMIC:.6f}",    "dimensionless"),
        ("MU_IONISED_PRIMORDIAL", f"{MU_IONISED_PRIMORDIAL:.3f}","dimensionless"),
    ]
    w = max(len(r[0]) for r in rows)
    print("\nastro_units.py — constant reference\n" + "─" * 52)
    for name, value, unit in rows:
        print(f"  {name:<{w}}  {value:>14}  [{unit}]")

    print("\n─── Conversion smoke-tests ──────────────────────────")
    z_test = 0.0134
    print(f"  metallicity_to_solar({z_test})        = "
          f"{metallicity_to_solar(z_test):.4f}  (expect 1.0000)")
    u_test = 1.0e12   # erg/g  — warm ISM
    T = internal_energy_to_temperature(u_test)
    print(f"  internal_energy_to_temperature(1e12) = {T:.3e} K")
    print(f"  round-trip u→T→u error               = "
          f"{abs(temperature_to_internal_energy(T) - u_test) / u_test:.2e}")
    print(f"  scale_factor_to_redshift(0.5)        = "
          f"{scale_factor_to_redshift(0.5):.4f}  (expect 1.0000)")
    print(f"  density_cgs_to_hydrogen_number(1e-27)= "
          f"{density_cgs_to_hydrogen_number(1e-27):.4e} cm⁻³")
    print()