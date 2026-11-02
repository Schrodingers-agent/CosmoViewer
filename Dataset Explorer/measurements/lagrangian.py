"""
lagrangian.py — Lagrangian radii.

r_f is the radius about the centre that encloses a fraction f of the mass of
the particle set. It is the smallest particle radius at which the cumulative
mass reaches f × M_total.

Two error estimates are given, and they should agree:
  * sampling: the enclosed fraction at a fixed radius is binomial, so
    σ_f = sqrt(f(1-f)/N_eff) with N_eff = (Σm)²/Σm², propagated through the
    local slope dr/df of the measured profile;
  * bootstrap: particles resampled with replacement, centre held fixed.
Neither includes the uncertainty of the centre itself.
"""
from __future__ import annotations

import numpy as np
from astropy import units as u

from .centre import Centre, minimum_image
from .results import Provenance, Result
from .snapshot_view import ParticleSet

VERSION = "1"
DEFAULT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)


def _radii_from_sorted(r_sorted, m_sorted, fractions):
    cm = np.cumsum(m_sorted)
    target = np.asarray(fractions) * cm[-1]
    i = np.searchsorted(cm, target * (1 - 1e-12), side="left")
    return r_sorted[np.minimum(i, len(r_sorted) - 1)]


def lagrangian_radii(p: ParticleSet, centre: Centre, fractions=DEFAULT_FRACTIONS, *,
                     n_bootstrap: int = 0, seed: int | None = 0) -> Result:
    fractions = np.asarray(fractions, dtype=float)
    if np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError("fractions must lie strictly between 0 and 1")
    d = minimum_image(p.pos - centre.pos, p.box)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    order = np.argsort(r, kind="stable")
    r_s, m_s = r[order], p.mass[order]
    radii = _radii_from_sorted(r_s, m_s, fractions)

    # Sampling error through the local slope of r(f)
    n_eff = m_s.sum() ** 2 / np.sum(m_s ** 2)
    sig_f = np.sqrt(fractions * (1 - fractions) / n_eff)
    lo = _radii_from_sorted(r_s, m_s, np.clip(fractions - sig_f, 1e-9, 1 - 1e-9))
    hi = _radii_from_sorted(r_s, m_s, np.clip(fractions + sig_f, 1e-9, 1 - 1e-9))
    err_sampling = 0.5 * (hi - lo)

    values = {
        "fraction": fractions,
        "radius": radii * u.kpc,
        "radius_err_sampling": err_sampling * u.kpc,
        "mass_total": m_s.sum() * u.Msun,
    }
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        boot = np.empty((n_bootstrap, len(fractions)))
        n = len(r)
        for b in range(n_bootstrap):
            j = np.sort(rng.integers(0, n, n))      # r_s is sorted, so sorted indices stay sorted
            boot[b] = _radii_from_sorted(r_s[j], m_s[j], fractions)
        values["radius_err_bootstrap"] = boot.std(axis=0, ddof=1) * u.kpc

    prov = Provenance(
        measurement="lagrangian_radii", version=VERSION, files=list(p.files),
        particle_types=list(p.particle_types), n_used=len(p), n_total=p.n_total,
        frame=p.frame, time=dict(p.time), centre=centre.to_dict(),
        parameters={"fractions": fractions.tolist(), "n_bootstrap": n_bootstrap,
                    "periodic_box_kpc": p.box},
        notes=list(p.notes) + ["errors exclude centre uncertainty"])
    return Result(values=values, provenance=prov)
