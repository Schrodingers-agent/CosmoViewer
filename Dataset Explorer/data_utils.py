"""
data_utils.py
─────────────
Shared, pure-Python utilities. No Qt. No plugin logic.

  compute_robust_domain        — H5Web-ported NaN/log-safe domain
  apply_cosmological_factors   — h^h_exp × a^a_exp  (EAGLE/VirgoDB scale factors)
  scaling_exponents            — h/a exponents from EAGLE or Gadget-4 attribute names
  run_cosmology                — is the run cosmological; which a, h, Ω apply
  read_dataset_attrs           — reads only the attribute dict for one HDF5 dataset,
                                 no array data (O(1) I/O, safe for UI / Stage 2)
  _read_dataset_full           — INTERNAL: reads array + attrs; requires a
                                 max_particles budget; raises ValueError otherwise
  GadgetCodeUnits              — full unit system from header/parameter dicts
"""

import numpy as np
from astropy import units as u
from astropy.constants import G as G_newton


# ── CGS unit table ────────────────────────────────────────────────────────────
_CGS_UNITS: dict[str, u.Unit] = {
    "Density":              u.g / u.cm**3,
    "InternalEnergy":       u.erg / u.g,
    "Masses":               u.g,
    "Velocities":           u.cm / u.s,
    "Coordinates":          u.cm,
    "SmoothingLength":      u.cm,
    "Pressure":             u.g / (u.cm * u.s**2),
    "Temperature":          u.K,
    "StarFormationRate":    u.g / u.s,      # what CGSConversionFactor / to_cgs give
    "Metallicity":          u.dimensionless_unscaled,
    "ElectronAbundance":    u.dimensionless_unscaled,
}


# ─────────────────────────────────────────────────────────────────────────────
#  DOMAIN
# ─────────────────────────────────────────────────────────────────────────────

def compute_robust_domain(
        data: np.ndarray,
        scale_mode: str = "linear",
) -> tuple[float | None, float | None]:
    """
    H5Web-ported Min/Max domain computation.

    Rules
    -----
    1. Flatten and strip NaN / Inf.
    2. Log mode: additionally strip values <= 0.
    3. Empty result → (None, None).
    4. Nice-domain: pad ±10 % when min == max.
    """
    values = data.flatten()
    clean  = values[np.isfinite(values)]
    if len(clean) == 0:
        return None, None

    if scale_mode == "log":
        pos = clean[clean > 0]
        if len(pos) == 0:
            return None, None
        d_min, d_max = float(np.min(pos)), float(np.max(pos))
    else:
        d_min, d_max = float(np.min(clean)), float(np.max(clean))

    if d_min == d_max:
        if d_min == 0.0:
            d_min, d_max = -1.0, 1.0
        else:
            pad   = abs(d_min) * 0.1
            d_min -= pad
            d_max += pad

    return d_min, d_max


# ─────────────────────────────────────────────────────────────────────────────
#  COSMOLOGICAL SCALE FACTORS  (EAGLE / VirgoDB convention)
# ─────────────────────────────────────────────────────────────────────────────

def apply_cosmological_factors(
        h:     float,
        h_exp: float,
        a:     float,
        a_exp: float,
) -> float:
    """
    Return the dimensionless cosmological correction factor h^h_exp × a^a_exp.

    EAGLE/VirgoDB datasets store co-moving values.  Each dataset carries two
    attributes that encode the exponents needed to recover physical values:

        h-scale-exponent    (e.g.  2  for Density)
        aexp-scale-exponent (e.g. -3  for Density)

    So for Density:
        ρ_physical = raw × CGSConversionFactor × h^2 × a^-3

    For Gadget-4 native files these attributes are absent (callers supply
    h_exp=0, a_exp=0), so the function returns 1.0 — a no-op.

    Parameters
    ----------
    h     : HubbleParam  (dimensionless, e.g. 0.7)
    h_exp : h-scale-exponent from dataset attrs
    a     : scale factor = Time from Header  (= 1/(1+z))
    a_exp : aexp-scale-exponent from dataset attrs

    Returns
    -------
    float  — multiply raw CGS value by this to get physical CGS value
    """
    if h_exp == 0.0 and a_exp == 0.0:
        return 1.0

    factor = 1.0
    if h_exp != 0.0:
        if h <= 0:
            raise ValueError(f"HubbleParam must be > 0, got {h}")
        factor *= h ** h_exp
    if a_exp != 0.0:
        if a <= 0:
            raise ValueError(f"Scale factor a must be > 0, got {a}")
        factor *= a ** a_exp

    return factor


# ─────────────────────────────────────────────────────────────────────────────
#  RUN COSMOLOGY AND SCALING EXPONENTS
# ─────────────────────────────────────────────────────────────────────────────

# Exponent attribute names: EAGLE / VirgoDB first, then Gadget-4.
_H_EXP_KEYS = ("h-scale-exponent", "h_scaling")
_A_EXP_KEYS = ("aexp-scale-exponent", "a_scaling")


def _first_float(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


def scaling_exponents(attrs: dict) -> tuple[float, float]:
    """(h_exp, a_exp) for a dataset: physical = raw × to_cgs × h^h_exp × a^a_exp."""
    return (_first_float(attrs, _H_EXP_KEYS, 0.0),
            _first_float(attrs, _A_EXP_KEYS, 0.0))


def run_cosmology(header: dict, params: dict | None = None) -> dict:
    """
    Decide whether the run is cosmological and which a, h, Ω apply.

    In a non-cosmological run Header/Time is a time, not a scale factor, so
    ``a`` is 1.0 there.  Gadget-4 keeps HubbleParam, Omega0 and
    ComovingIntegrationOn in Parameters; EAGLE keeps them in Header and
    RuntimePars (merged into params by VirtualSnapshotSet).

    Returns dict: is_cosmological, source, a, h, h_known, omega0,
    omega_lambda, omega_baryon, time_code.
    """
    params = params or {}
    time_code = _first_float(header, ("Time",), 0.0)

    flag = _first_float(params, ("ComovingIntegrationOn",))
    if flag is None:
        flag = _first_float(header, ("ComovingIntegrationOn",))
    omega0 = _first_float(header, ("Omega0",), None)
    if omega0 is None:
        omega0 = _first_float(params, ("Omega0",), None)
    if flag is not None:
        is_cosmo = int(flag) == 1
        source = "ComovingIntegrationOn"
    else:
        # No flag (Gadget-2 style files): a positive Ω0 with 0 < Time ≤ 1 means a scale factor.
        is_cosmo = bool(omega0 and omega0 > 0 and 0.0 < time_code <= 1.0 + 1e-9)
        source = "inferred from Header (no ComovingIntegrationOn)"

    h = _first_float(header, ("HubbleParam",))
    if h is None:
        h = _first_float(params, ("HubbleParam",))
    h_known = h is not None and h > 0
    if not h_known:
        h = 1.0

    if is_cosmo:
        a = _first_float(header, ("ExpansionFactor", "Time"), 1.0)
    else:
        a = 1.0

    def _omega(keys):
        v = _first_float(header, keys)
        return v if v is not None else _first_float(params, keys)

    return dict(
        is_cosmological = is_cosmo,
        source          = source,
        a               = a,
        h               = h,
        h_known         = h_known,
        omega0          = omega0,
        omega_lambda    = _omega(("OmegaLambda",)),
        omega_baryon    = _omega(("OmegaBaryon",)),
        time_code       = time_code,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET READER
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_attr(v):
    """
    Type-aware coercion of one HDF5 attribute value.

    Checks actual Python/numpy type — never blindly calls float().
      - int / float / np.number  → float
      - bytes / np.bytes_        → decode to str
      - np.ndarray               → keep as-is
      - anything else            → str()
    """
    if isinstance(v, (int, float, np.number)):
        return float(v)
    elif isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    elif isinstance(v, np.ndarray):
        return v
    else:
        return str(v)


def coerce_attrs(attrs_proxy) -> dict:
    """
    Convert an h5py AttributeManager proxy to a plain Python dict.

    Scalar numerics cast to float.  Bytes (including np.bytes_, which is a
    subclass of bytes) decoded to str.  ndarrays kept as-is.  Any other
    type — e.g. Python list, which is non-standard in Gadget/EAGLE HDF5 —
    falls through to str() for safety.

    This is the single canonical implementation; io_manager and
    density_explorer import it under the local alias _coerce_attrs.
    """
    return {k: _coerce_attr(v) for k, v in attrs_proxy.items()}


# Known alternative CGS-factor attribute names across simulation codes
# Gadget-4 native : to_cgs
# EAGLE / VirgoDB : CGSConversionFactor
_CGS_FACTOR_KEYS = ("to_cgs", "CGSConversionFactor")


def read_dataset_attrs(hdf5_group, dataset_name: str) -> dict:
    """
    Return the coerced attribute dict for one HDF5 dataset.

    No array data is read.  O(1) I/O: opens the dataset object only, reads
    only the compact attrs metadata, never calls ds[:] or ds[0:N].

    Parameters
    ----------
    hdf5_group   : open h5py Group
    dataset_name : str

    Returns
    -------
    dict  keyed by attribute name, values coerced via _coerce_attr:
          numeric → float, bytes → str, ndarray → ndarray, other → str.

    Use cases
    ---------
    * GadgetCodeUnits.from_hdf5 — reads cosmological exponents and CGS
      conversion factors without materialising the particle array.
    * Pipeline Stage 2 unit resolution — reads h-scale-exponent and
      aexp-scale-exponent per dataset without a memory budget.
    """
    ds = hdf5_group[dataset_name]
    return {k: _coerce_attr(ds.attrs[k]) for k in ds.attrs}

def _read_dataset_full(
        hdf5_group,
        dataset_name:  str,
        header:        dict | None = None,
        max_particles: int | None  = None,
) -> dict:
    """
    INTERNAL: Read one Gadget-4 or EAGLE/VirgoDB HDF5 dataset together
    with its unit-conversion attributes.

    .. warning::
        You MUST supply a ``max_particles`` budget.  Callers that need
        unbounded reads must use ``VirtualSnapshotSet.extract_array``
        which enforces the global memory ceiling (MAX_HIST_PARTICLES) and
        performs phase-continuous striding across distributed chunks.

    Supports both attribute naming conventions:
      Gadget-4 native : to_cgs
      EAGLE / VirgoDB : CGSConversionFactor + h-scale-exponent + aexp-scale-exponent

    Parameters
    ----------
    hdf5_group    : open h5py Group
    dataset_name  : str
    header        : dict of Header attributes (needed for HubbleParam and Time/a).
                    If None, h and a both default to 1.0 (= no cosmological correction).
    max_particles : int | None
                    REQUIRED.  Raises ValueError if None.

    Returns
    -------
    raw            : ndarray float64  — code-unit values (length ≤ max_particles)
    to_cgs         : float            — CGSConversionFactor attribute only
    cosmo_factor   : float            — h^h_exp × a^a_exp
    physical_factor: float            — to_cgs × cosmo_factor  (apply this to raw)
    cgs            : astropy Quantity — raw × physical_factor, with unit attached
    attrs          : dict             — all attributes, strings kept as str
    shape          : tuple
    n_particles    : int
    cgs_unit_str   : str
    h_exp          : float            — h-scale-exponent (0 if absent)
    a_exp          : float            — aexp-scale-exponent (0 if absent)
    h              : float            — HubbleParam used
    a              : float            — scale factor used
    """
    if max_particles is None:
        raise ValueError(
            "_read_dataset_full called without a max_particles budget. "
            "For full-array reads use VirtualSnapshotSet.extract_array. "
            f"Unconstrained read of {dataset_name!r} may exhaust RAM on large snapshots."
        )

    header = header or {}

    ds    = hdf5_group[dataset_name]
    raw   = ds[:max_particles].astype(np.float64)
    attrs = {k: _coerce_attr(ds.attrs[k]) for k in ds.attrs}

    # ── CGS base factor ───────────────────────────────────────────────────────
    to_cgs: float = 1.0
    for key in _CGS_FACTOR_KEYS:
        raw_val = attrs.get(key, None)
        if raw_val is not None:
            try:
                to_cgs = float(raw_val)
                break
            except (TypeError, ValueError):
                to_cgs = 1.0

    # ── Cosmological exponents (EAGLE or Gadget-4 names; absent → 0 → no-op) ─
    h_exp, a_exp = scaling_exponents(attrs)

    # ── a and h for this run (a = 1 in non-cosmological runs) ────────────────
    cosmo = run_cosmology(header)
    h, a = cosmo["h"], cosmo["a"]

    # ── Combined physical conversion factor ───────────────────────────────────
    cosmo_factor    = apply_cosmological_factors(h, h_exp, a, a_exp)
    physical_factor = to_cgs * cosmo_factor

    cgs_unit = _CGS_UNITS.get(dataset_name, u.dimensionless_unscaled)
    cgs      = (raw * physical_factor) * cgs_unit

    return dict(
        raw             = raw,
        to_cgs          = to_cgs,
        cosmo_factor    = cosmo_factor,
        physical_factor = physical_factor,
        cgs             = cgs,
        attrs           = attrs,
        shape           = raw.shape,
        n_particles     = raw.shape[0],
        cgs_unit_str    = cgs_unit.to_string(),
        h_exp           = h_exp,
        a_exp           = a_exp,
        h               = h,
        a               = a,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GADGET CODE UNITS  (fallback — used when to_cgs is absent)
# ─────────────────────────────────────────────────────────────────────────────

class GadgetCodeUnits:
    """
    Derives the full Gadget-4 unit system from header / parameter dicts.
    For per-dataset attribute reading use read_dataset_attrs().
    Use this for global labelling, computing G_code, etc.
    """

    def __init__(
        self,
        unit_length_cm:    float = 3.085678e21,   # 1 kpc
        unit_mass_g:       float = 1.989e43,      # 1e10 M☉
        unit_velocity_cms: float = 1.0e5,         # 1 km/s
        hubble_param:      float = 0.7,
    ):
        self.unit_length   = unit_length_cm    * u.cm
        self.unit_mass     = unit_mass_g       * u.g
        self.unit_velocity = unit_velocity_cms * (u.cm / u.s)
        self.hubble_param  = hubble_param
        self.unit_time     = (self.unit_length / self.unit_velocity).to(u.s)
        self.unit_density  = (self.unit_mass / self.unit_length**3).to(u.g / u.cm**3)
        self.unit_energy   = (self.unit_mass * self.unit_velocity**2).to(u.erg)

        G_cgs          = G_newton.to(u.cm**3 / (u.g * u.s**2)).value
        self.G_code    = (G_cgs * unit_mass_g
                          * (unit_length_cm / unit_velocity_cms)**2
                          / unit_length_cm**3)

    @classmethod
    def from_hdf5(cls, header: dict, params: dict | None = None) -> "GadgetCodeUnits":
        merged = dict(header)
        if params:
            merged.update(params)
        return cls(
            unit_length_cm    = float(merged.get("UnitLength_in_cm",          3.085678e21)),
            unit_mass_g       = float(merged.get("UnitMass_in_g",             1.989e43)),
            unit_velocity_cms = float(merged.get("UnitVelocity_in_cm_per_s",  1.0e5)),
            hubble_param      = float(merged.get("HubbleParam",               0.7)),
        )

    def __repr__(self) -> str:
        return (
            f"GadgetCodeUnits("
            f"L={self.unit_length.to(u.kpc):.3g}, "
            f"M={self.unit_mass.to(u.Msun):.3g}, "
            f"V={self.unit_velocity.to(u.km/u.s):.3g})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE BINNING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_bin_count(
        data:  np.ndarray,
        n_min: int = 10,
        n_max: int = 256,
        log:   bool = False,
) -> tuple[int, str]:
    """
    Compute an adaptive histogram bin count from the data distribution.

    Algorithm
    ---------
    1. Strip non-finite values; cast to float64 to prevent float32 jitter.
    2. Freedman–Diaconis (FD) rule — outlier-resistant:
           W = 2 · IQR · n^(−1/3)
           k = ⌈ (max − min) / W ⌉
    3. If IQR == 0 (degenerate / uniform data) → Rice Rule fallback:
           k = ⌈ 2 · n^(1/3) ⌉
    4. Clamp result to [n_min, n_max].

    log=True applies the rule to log10 of the positive values: the variable a
    log-spaced histogram actually bins. FD on the linear values of data that
    span decades gives a width set by the dense low end, so k hits n_max and
    the log bins in the sparse tail are mostly empty.

    Returns
    -------
    (k, heuristic)
        k         : int  — recommended bin count, clamped to [n_min, n_max]
        heuristic : str  — "FD" | "Rice"
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if log:
        arr = np.log10(arr[arr > 0])
    n   = len(arr)

    if n < 2:
        k = max(n_min, min(n_max, int(np.ceil(2.0 * max(1, n) ** (1.0 / 3.0)))))
        return k, "Rice"

    q25, q75   = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr        = q75 - q25
    data_range = float(np.max(arr) - np.min(arr))

    if iqr == 0.0 or data_range == 0.0:
        # Rice Rule — safe fallback for uniform / zero-variance data
        k         = int(np.ceil(2.0 * n ** (1.0 / 3.0)))
        heuristic = "Rice"
    else:
        # Freedman–Diaconis
        w         = 2.0 * iqr * (n ** (-1.0 / 3.0))
        k         = int(np.ceil(data_range / w))
        heuristic = "FD"
        # Capped by Rice, about what FD gives for Gaussian data. FD's width
        # comes from the bulk, so long tails (a velocity component of 3,562
        # EAGLE black holes) gave 70–86 bins, most of the tail ones empty.
        # From n ≈ 125k Rice exceeds every MAX_BINS, so only small sets change.
        k_rice = int(np.ceil(2.0 * n ** (1.0 / 3.0)))
        if k > k_rice:
            k, heuristic = k_rice, "FD, capped at Rice"

    return max(n_min, min(n_max, k)), heuristic


def integer_bin_edges(lo: int, hi: int, k: int, log: bool = False) -> np.ndarray:
    """
    Bin edges on half-integers for whole-number data from lo to hi.

    Bins narrower than the spacing of the values alternate full and empty
    (a picket fence): group numbers, seed counts, group lengths. Here every
    bin holds at least one whole number and no value sits on an edge.

    Linear: equal bins of ceil(span / k) integers, at most k of them.
    Log (lo >= 1): k log-spaced edges moved down to the nearest half-integer,
    duplicates dropped, so the narrow low bins become one integer each.
    """
    lo, hi = int(lo), int(hi)
    if log and lo >= 1:
        raw   = np.logspace(np.log10(lo - 0.5), np.log10(hi + 0.5), max(1, k) + 1)
        inner = np.floor(raw[1:-1]) + 0.5
        inner = inner[(inner > lo - 0.5) & (inner < hi + 0.5)]
        return np.unique(np.concatenate([[lo - 0.5], inner, [hi + 0.5]]))
    span   = hi - lo + 1                            # whole numbers covered
    width  = max(1, int(np.ceil(span / max(1, k))))
    n_bins = int(np.ceil(span / width))
    return lo - 0.5 + width * np.arange(n_bins + 1, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  LATEX → HTML CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

# Ordered substitution table for latex_to_html().
# Priority matters: more-specific patterns must appear BEFORE general ones.
# Each entry is (compiled_regex, replacement_string).
_LATEX_SUBS: list[tuple] = []

def _build_subs() -> None:
    """Compile the substitution table once at import time."""
    global _LATEX_SUBS

    rules: list[tuple[str, str]] = [
        # ── 1. Strip outer $...$ delimiters ────────────────────────────────
        (r"\$([^$]*)\$",                            r"\1"),

        # ── 2. \mathrm{...} → inner text ───────────────────────────────────
        (r"\\mathrm\{([^}]*)\}",                    r"\1"),
        (r"\\text\{([^}]*)\}",                      r"\1"),

        # ── 3. \frac{A}{B} → A / B ─────────────────────────────────────────
        (r"\\frac\{([^}]*)\}\{([^}]*)\}",           r"\1 / \2"),

        # ── 4. Named astrophysical symbols (BEFORE generic _{ rule) ────────
        # M_{\odot} and M_\odot must match here so the generic _{} rule
        # cannot swallow the entire {\\odot} block first.
        (r"M_\{\\odot\}",                           r"M<sub>&#9737;</sub>"),
        (r"M_\\odot",                               r"M<sub>&#9737;</sub>"),
        (r"\\odot",                                 r"&#9737;"),
        # Common Greek
        (r"\\rho",                                  r"&rho;"),
        (r"\\sigma",                                r"&sigma;"),
        (r"\\alpha",                                r"&alpha;"),
        (r"\\beta",                                 r"&beta;"),
        (r"\\gamma",                                r"&gamma;"),
        (r"\\lambda",                               r"&lambda;"),
        (r"\\mu",                                   r"&mu;"),
        (r"\\nu",                                   r"&nu;"),
        (r"\\pi",                                   r"&pi;"),
        (r"\\tau",                                  r"&tau;"),
        (r"\\phi",                                  r"&phi;"),
        (r"\\chi",                                  r"&chi;"),
        (r"\\omega",                                r"&omega;"),
        (r"\\Omega",                                r"&Omega;"),
        # Operators
        (r"\\times",                                r"&times;"),
        (r"\\cdot",                                 r"&middot;"),

        # ── 5. Super/subscript with braces: ^{...} and _{...} ──────────────
        (r"\^\{([^}]*)\}",                          r"<sup>\1</sup>"),
        (r"_\{([^}]*)\}",                           r"<sub>\1</sub>"),

        # ── 6. Super/subscript single char: ^X and _X ───────────────────────
        (r"\^(-?[0-9a-zA-Z])",                      r"<sup>\1</sup>"),
        # Only a lone symbol such as x_0: not inside identifiers or filenames
        # (a_scaling, phase_space_305_r.npy stay as written).
        (r"(?<![0-9a-zA-Z][0-9a-zA-Z])_(-?[0-9a-zA-Z])(?![0-9a-zA-Z_])",
                                                    r"<sub>\1</sub>"),

        # ── 7. Plain Python scientific notation: 1.5e+10 → 1.5×10^10 ───────
        # Applied AFTER symbol substitutions so LaTeX \times is already
        # converted and we do not double-convert.
        (r"([-+]?[0-9]+\.?[0-9]*)(?:[eE]([+-]?[0-9]+))",
         r"\1&times;10<sup>\2</sup>"),

        # LaTeX thin space from astropy ("g\,cm^{-3}")
        (r"\\,",                                   r"&thinsp;"),

        # ── 8. Stray backslash sequences (catch-all — must be last) ─────────
        (r"\\[a-zA-Z]+",                            r""),
    ]


    _LATEX_SUBS = [(_re.compile(pat), repl) for pat, repl in rules]

_build_subs()


def latex_to_html(text: str) -> str:
    """
    Convert a LaTeX-flavoured label string to Qt-compatible rich-text HTML.

    Scope
    ─────
    Strictly targets the subset of LaTeX generated by:
      • astropy.units.to_string('latex_inline')
      • Our internal plugin label strings
      • Python's default float repr (e.g. 1.5e+12)

    Does NOT attempt to implement a full LaTeX compiler.  Complex or
    deeply nested expressions that fall outside the known pattern set will
    pass through the try/except fallback and be returned unchanged.

    Parameters
    ──────────
    text : str — raw label / info / status string

    Returns
    ───────
    str — HTML-safe string suitable for Qt RichText QLabel.setText().
         Returns the original string unmodified if any regex step raises.
    """
    if not text:
        return text
    try:
        result = text
        for pattern, replacement in _LATEX_SUBS:
            result = pattern.sub(replacement, result)
        return result
    except Exception:
        return text  # safe fallback — never crash the UI thread