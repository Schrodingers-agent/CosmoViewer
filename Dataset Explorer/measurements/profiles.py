"""
profiles.py — radial profiles about a centre.

radial_profile() bins particles in spherical shells about a Centre and gives,
per shell:

  r_lo, r_hi, r_mean   shell edges and the mass-weighted mean radius
  n, mass              particles and mass in the shell
  density              mass / shell volume, Poisson error density / √n
  mass_enclosed        M(< r_hi), counting every particle inside r_hi
  v_circ               √(G M(< r_hi) / r_hi), r_hi physical (× a in the
                       comoving frame), in km/s
  sigma_r, sigma_theta, sigma_phi
                       mass-weighted velocity dispersions in spherical
                       components, about the shell's own mean velocity;
                       error σ / √(2 (n − 1)) (Gaussian approximation)
  sigma_t              √(σ_θ² + σ_φ²)
  beta                 1 − (σ_θ² + σ_φ²) / (2 σ_r²), error propagated from
                       the σ errors
  resolved             False inside r_resolved (below)

surface_density() bins the projection along one axis in annuli: Σ(R) with
error Σ / √n.

Bins are log-spaced (default) or hold equal particle counts. Velocities are
taken relative to the centre's velocity when it has one, otherwise relative
to the mass-weighted mean velocity of the particles inside the outermost bin.
In cosmological runs they are peculiar: the Hubble flow is not added, which
the provenance records.

r_resolved is the larger of 2.8 ε (Gadget's spline softening is Newtonian
beyond 2.8 times the Plummer-equivalent ε) when a softening is given, and,
when rho_crit is given, the Power et al. (2003) radius where
κ(r) = (√200 / 8) · N(<r) / ln N(<r) · (ρ̄(<r) / ρ_crit)^(-1/2) reaches 1.
Both are quoted from the literature from memory, not re-derived here.
"""
from __future__ import annotations

import numpy as np
from astropy import units as u

from .centre import Centre, minimum_image
from .models import G_KPC_KMS2_MSUN
from .results import Provenance, Result
from .snapshot_view import ParticleSet

VERSION = "1"
SPLINE_NEWTONIAN = 2.8


def _bin_edges(r_sorted, bins, n_bins, r_min, r_max, min_per_bin):
    n = len(r_sorted)
    if isinstance(bins, (list, tuple, np.ndarray)):
        return np.asarray(bins, dtype=float)
    lo = r_min if r_min is not None else r_sorted[min(max(min_per_bin, 1), n - 1)]
    hi = r_max if r_max is not None else r_sorted[-1] * (1 + 1e-12)
    if lo <= 0:
        lo = r_sorted[r_sorted > 0][0]
    if bins == "log":
        return np.geomspace(lo, hi, n_bins + 1)
    if bins == "equal_count":
        inside = r_sorted[(r_sorted >= lo) & (r_sorted <= hi)]
        k = np.linspace(0, len(inside) - 1, n_bins + 1).round().astype(int)
        edges = inside[k]
        edges[0], edges[-1] = lo, hi
        return np.unique(edges)
    raise ValueError("bins must be 'log', 'equal_count' or an array of edges")


def _spherical(d, v):
    """Velocity components (v_r, v_θ, v_φ) for displacements d."""
    x, y, z = d.T
    r = np.sqrt(x * x + y * y + z * z)
    R = np.sqrt(x * x + y * y)
    with np.errstate(invalid="ignore", divide="ignore"):
        st, ct = R / r, z / r
        sp, cp = np.where(R > 0, y / R, 0.0), np.where(R > 0, x / R, 1.0)
    vx, vy, vz = v.T
    vr = vx * st * cp + vy * st * sp + vz * ct
    vt = vx * ct * cp + vy * ct * sp - vz * st
    vp = -vx * sp + vy * cp
    return vr, vt, vp


def _wstd(x, w):
    m = np.average(x, weights=w)
    return np.sqrt(np.average((x - m) ** 2, weights=w))


def power_radius(r_sorted, m_sorted, rho_crit, kappa=1.0):
    """Smallest radius beyond which κ(r) ≥ kappa (Power et al. 2003; from memory)."""
    n_in = np.arange(1, len(r_sorted) + 1)
    m_in = np.cumsum(m_sorted)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_bar = m_in / (4.0 / 3.0 * np.pi * r_sorted ** 3)
        k = np.sqrt(200.0) / 8.0 * n_in / np.log(n_in) * np.sqrt(rho_crit / rho_bar)
    ok = np.isfinite(k) & (k >= kappa) & (n_in > 1)
    bad = np.flatnonzero(~ok)
    if len(bad) == 0:
        return 0.0
    if bad[-1] == len(r_sorted) - 1:
        return float(r_sorted[-1])
    return float(r_sorted[bad[-1] + 1])


def radial_profile(p: ParticleSet, centre: Centre, *, bins="log", n_bins: int = 30,
                   r_min: float | None = None, r_max: float | None = None,
                   min_per_bin: int = 10, softening: float | None = None,
                   rho_crit: float | None = None) -> Result:
    """Spherical profile of a ParticleSet about centre. Radii in kpc (the
    ParticleSet's frame), softening in kpc, rho_crit in Msun / kpc³."""
    d = minimum_image(p.pos - centre.pos, p.box)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    order = np.argsort(r, kind="stable")
    r_s, m_s, d_s = r[order], p.mass[order], d[order]
    edges = _bin_edges(r_s, bins, n_bins, r_min, r_max, min_per_bin)
    nb = len(edges) - 1
    idx = np.searchsorted(r_s, edges, side="left")
    lo_i, hi_i = idx[:-1], idx[1:]

    # Radii are in the ParticleSet's frame; v_circ needs the physical radius (comoving × a).
    a_phys = float(p.time.get("a", 1.0)) if p.frame == "comoving" else 1.0
    have_v = p.vel is not None and np.isfinite(p.vel).all()
    if have_v:
        v_s = p.vel[order]
        if centre.vel is not None:
            v0 = np.asarray(centre.vel, float)
            v_ref = "centre velocity"
        else:
            sel = slice(0, hi_i[-1])
            v0 = np.average(v_s[sel], axis=0, weights=m_s[sel])
            v_ref = "mass-weighted mean velocity inside the outermost bin"
        vr, vt, vp = _spherical(d_s, v_s - v0)

    cm = np.concatenate([[0.0], np.cumsum(m_s)])
    out = {k: np.full(nb, np.nan) for k in (
        "r_mean", "mass", "density", "density_err", "mass_enclosed", "v_circ",
        "sigma_r", "sigma_theta", "sigma_phi", "sigma_t", "sigma_r_err",
        "sigma_t_err", "beta", "beta_err")}
    out["n"] = (hi_i - lo_i).astype(np.int64)
    for b in range(nb):
        a, z = lo_i[b], hi_i[b]
        n = z - a
        vol = 4.0 / 3.0 * np.pi * (edges[b + 1] ** 3 - edges[b] ** 3)
        out["mass_enclosed"][b] = cm[z]
        out["v_circ"][b] = np.sqrt(G_KPC_KMS2_MSUN * cm[z] / (edges[b + 1] * a_phys))
        if n == 0:
            out["mass"][b] = 0.0
            out["density"][b] = 0.0
            continue
        w = m_s[a:z]
        out["mass"][b] = w.sum()
        out["r_mean"][b] = np.average(r_s[a:z], weights=w)
        out["density"][b] = w.sum() / vol
        out["density_err"][b] = out["density"][b] / np.sqrt(n)
        if have_v and n >= 2:
            sr, st, sp = _wstd(vr[a:z], w), _wstd(vt[a:z], w), _wstd(vp[a:z], w)
            out["sigma_r"][b], out["sigma_theta"][b], out["sigma_phi"][b] = sr, st, sp
            out["sigma_t"][b] = np.sqrt(st ** 2 + sp ** 2)
            f = 1.0 / np.sqrt(2.0 * (n - 1))
            out["sigma_r_err"][b] = sr * f
            out["sigma_t_err"][b] = out["sigma_t"][b] * f
            q = (st ** 2 + sp ** 2) / (2.0 * sr ** 2)
            out["beta"][b] = 1.0 - q
            # δq/q = 2 √((δσ_t/σ_t)² + (δσ_r/σ_r)²) = 2 √2 f
            out["beta_err"][b] = q * 2.0 * np.sqrt(2.0) * f

    r_res = 0.0
    why = []
    if softening:
        r_res = max(r_res, SPLINE_NEWTONIAN * softening)
        why.append(f"2.8 ε = {SPLINE_NEWTONIAN * softening:.4g} kpc")
    if rho_crit:
        rp = power_radius(r_s, m_s, rho_crit)
        r_res = max(r_res, rp)
        why.append(f"Power et al. (2003) radius {rp:.4g} kpc")

    kpc, msun, kms = u.kpc, u.Msun, u.km / u.s
    values = {
        "r_lo": edges[:-1] * kpc, "r_hi": edges[1:] * kpc, "r_mean": out["r_mean"] * kpc,
        "n": out["n"], "mass": out["mass"] * msun,
        "density": out["density"] * msun / kpc ** 3,
        "density_err": out["density_err"] * msun / kpc ** 3,
        "mass_enclosed": out["mass_enclosed"] * msun, "v_circ": out["v_circ"] * kms,
        "sigma_r": out["sigma_r"] * kms, "sigma_theta": out["sigma_theta"] * kms,
        "sigma_phi": out["sigma_phi"] * kms, "sigma_t": out["sigma_t"] * kms,
        "sigma_r_err": out["sigma_r_err"] * kms, "sigma_t_err": out["sigma_t_err"] * kms,
        "beta": out["beta"], "beta_err": out["beta_err"],
        "resolved": edges[:-1] >= r_res,
        "r_resolved": r_res * kpc,
    }
    notes = list(p.notes)
    if have_v:
        notes.append(f"velocities relative to the {v_ref}")
    else:
        notes.append("no velocities: dispersions and β not computed")
    if why:
        notes.append("resolved beyond " + " and ".join(why))
    notes.append("density errors are Poisson; dispersion errors assume Gaussian velocities")
    prov = Provenance(
        measurement="radial_profile", version=VERSION, files=list(p.files),
        particle_types=list(p.particle_types), n_used=len(p), n_total=p.n_total,
        frame=p.frame, time=dict(p.time), centre=centre.to_dict(),
        parameters={"bins": bins if isinstance(bins, str) else "custom", "n_bins": int(nb),
                    "r_min_kpc": float(edges[0]), "r_max_kpc": float(edges[-1]),
                    "softening_kpc": softening, "rho_crit_msun_kpc3": rho_crit},
        notes=notes)
    return Result(values=values, provenance=prov)


def surface_density(p: ParticleSet, centre: Centre, *, axis: int = 2, bins="log",
                    n_bins: int = 30, R_min: float | None = None, R_max: float | None = None,
                    min_per_bin: int = 10) -> Result:
    """Σ(R) of the projection along axis (0 = x, 1 = y, 2 = z) in annuli."""
    d = minimum_image(p.pos - centre.pos, p.box)
    keep = [i for i in range(3) if i != axis]
    R = np.sqrt(d[:, keep[0]] ** 2 + d[:, keep[1]] ** 2)
    order = np.argsort(R, kind="stable")
    R_s, m_s = R[order], p.mass[order]
    edges = _bin_edges(R_s, bins, n_bins, R_min, R_max, min_per_bin)
    idx = np.searchsorted(R_s, edges, side="left")
    n = np.diff(idx)
    cm = np.concatenate([[0.0], np.cumsum(m_s)])
    mass = cm[idx[1:]] - cm[idx[:-1]]
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    sigma = mass / area
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(n > 0, sigma / np.sqrt(n), np.nan)
    values = {"R_lo": edges[:-1] * u.kpc, "R_hi": edges[1:] * u.kpc, "n": n,
              "mass": mass * u.Msun, "surface_density": sigma * u.Msun / u.kpc ** 2,
              "surface_density_err": err * u.Msun / u.kpc ** 2}
    prov = Provenance(
        measurement="surface_density", version=VERSION, files=list(p.files),
        particle_types=list(p.particle_types), n_used=len(p), n_total=p.n_total,
        frame=p.frame, time=dict(p.time), centre=centre.to_dict(),
        parameters={"axis": "xyz"[axis], "bins": bins if isinstance(bins, str) else "custom",
                    "n_bins": int(len(edges) - 1)},
        notes=list(p.notes) + ["errors are Poisson"])
    return Result(values=values, provenance=prov)
