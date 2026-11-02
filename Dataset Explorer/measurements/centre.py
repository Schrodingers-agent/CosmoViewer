"""
centre.py — where is the centre of this system?

Methods (all periodic-aware through the minimum image; all mass-weighted except
density, which weights by local density):

  com               centre of mass. In a periodic box the circular mean of
                    Bai & Breen (2008) picks the reference point, so a system
                    straddling the boundary is not averaged to the middle of
                    the box; the exact mean displacement from it follows.
  shrinking_sphere  iterative centre of mass in a sphere that shrinks by a
                    fixed fraction each step (the scheme of Power et al. 2003).
  density           density-weighted centre of Casertano & Hut (1985), with
                    each particle's density from its k nearest neighbours.
  potential_min     position of the particle with the lowest stored potential.
  particle          position of one chosen particle (e.g. a central IMBH).

Literature methods quoted from memory; the tests in test_lagrangian_radii_and_centres.py
check each against systems whose centre is known.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .snapshot_view import ParticleSet

METHODS = ("com", "shrinking_sphere", "density", "potential_min", "particle")


@dataclass
class Centre:
    pos: np.ndarray            # (3,) kpc, same frame as the particles
    method: str
    n_used: int
    parameters: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    vel: np.ndarray | None = None   # (3,) km/s, mass-weighted, when computed

    def to_dict(self) -> dict:
        d = {"method": self.method, "pos_kpc": [float(x) for x in self.pos],
             "n_used": int(self.n_used), "parameters": dict(self.parameters)}
        d.update({k: v for k, v in self.info.items()})
        if self.vel is not None:
            d["vel_kms"] = [float(x) for x in self.vel]
        return d


def minimum_image(dx: np.ndarray, box: float) -> np.ndarray:
    """Displacements wrapped into [-box/2, box/2) when box > 0."""
    if box and box > 0:
        return dx - box * np.round(dx / box)
    return dx


def _wrap(x: np.ndarray, box: float) -> np.ndarray:
    return np.mod(x, box) if box and box > 0 else x


def _com(pos: np.ndarray, mass: np.ndarray, box: float) -> np.ndarray:
    if not (box and box > 0):
        return np.average(pos, axis=0, weights=mass)
    theta = 2.0 * np.pi * pos / box
    xi = np.average(np.cos(theta), axis=0, weights=mass)
    zeta = np.average(np.sin(theta), axis=0, weights=mass)
    return box * (np.arctan2(-zeta, -xi) + np.pi) / (2.0 * np.pi)


def _com_about(pos, mass, ref, box):
    """Centre of mass computed in coordinates relative to ref (exact, no circular mean)."""
    d = minimum_image(pos - ref, box)
    return _wrap(ref + np.average(d, axis=0, weights=mass), box)


def _shrinking_sphere(pos, mass, box, start=None, r_start=None, shrink=0.025, n_stop=None):
    n = len(mass)
    c = _com(pos, mass, box) if start is None else np.asarray(start, float)
    d = minimum_image(pos - c, box)
    r2 = np.einsum("ij,ij->i", d, d)
    r = float(np.sqrt(r2.max())) if r_start is None else float(r_start)
    if n_stop is None:
        # 1000 as in Power et al. (2003), but never more than a tenth of the system.
        # Measured on Plummer spheres (test_lagrangian_radii_and_centres): centre rms error is
        # 0.05 a at N = 1e4 and 1e5 with this choice, 0.15 a with a 100-particle stop.
        n_stop = max(20, min(1000, n // 10))
    p_s, m_s = pos, mass          # the particles still inside the sphere
    it = 0
    while True:
        keep = r2 <= r * r
        if keep.sum() < n_stop:
            break
        p_s, m_s = p_s[keep], m_s[keep]
        c = _wrap(c + np.average(minimum_image(p_s - c, box), axis=0, weights=m_s), box)
        d = minimum_image(p_s - c, box)
        r2 = np.einsum("ij,ij->i", d, d)
        r *= (1.0 - shrink)
        it += 1
        if it > 10_000:
            break
    return c, {"iterations": it, "final_radius_kpc": r, "n_final": int(len(m_s)),
               "n_stop": int(n_stop), "shrink": shrink}


def local_density(pos, mass, box=0.0, k=6):
    """Casertano & Hut (1985) density estimate: the mass of the k-1 nearest
    neighbours (excluding the particle and the k-th) over the volume of the
    sphere reaching the k-th neighbour. Units: Msun / kpc^3."""
    from scipy.spatial import cKDTree
    p = _wrap(pos, box) if box and box > 0 else pos
    tree = cKDTree(p, boxsize=box if box and box > 0 else None)
    dist, nbr = tree.query(p, k=k + 1)
    r_k = dist[:, k]
    m_inner = mass[nbr[:, 1:k]].sum(axis=1)
    with np.errstate(divide="ignore"):
        return m_inner / (4.0 / 3.0 * np.pi * r_k ** 3)


def _density_centre(pos, mass, box, k=6):
    rho = local_density(pos, mass, box, k)
    ok = np.isfinite(rho)
    ref = _com(pos[ok], mass[ok] * rho[ok], box)
    c = _com_about(pos[ok], rho[ok], ref, box)
    # A density-weighted mean can land between two dense clumps (a merger,
    # a binary cluster). Compare the density near the centre with the peak.
    d = minimum_image(pos[ok] - c, box)
    near = np.argsort(np.einsum("ij,ij->i", d, d))[:k]
    ratio = float(np.median(rho[ok][near]) / np.max(rho[ok]))
    info = {"k": k, "n_finite_density": int(ok.sum()), "density_at_centre_over_peak": ratio}
    if ratio < 0.01:
        info["warning"] = ("density-weighted centre lies where the density is "
                           f"{ratio:.1e} of the peak: the system may have several "
                           "dense components; use shrinking_sphere or potential_min")
    return c, info


def centre_velocity(p: ParticleSet, centre_pos: np.ndarray, radius: float) -> tuple:
    """Mass-weighted mean velocity of particles within radius of the centre."""
    if p.vel is None:
        return None, 0
    d = minimum_image(p.pos - centre_pos, p.box)
    sel = np.einsum("ij,ij->i", d, d) <= radius * radius
    if not sel.any():
        return None, 0
    return np.average(p.vel[sel], axis=0, weights=p.mass[sel]), int(sel.sum())


def find_centre(p: ParticleSet, method: str = "shrinking_sphere", *,
                potential: np.ndarray | None = None, particle_id=None,
                velocity_radius: float | None = None, **kw) -> Centre:
    """Centre of a ParticleSet by one of METHODS. Extra keywords go to the method:
    shrinking_sphere: start, r_start, shrink (0.025), n_stop; density: k (6)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    box = p.box
    info: dict = {}
    if method == "com":
        # circular mean as reference, then the exact mean of displacements from it
        c = _com_about(p.pos, p.mass, _com(p.pos, p.mass, box), box) if box and box > 0 \
            else _com(p.pos, p.mass, box)
        n = len(p)
    elif method == "shrinking_sphere":
        c, info = _shrinking_sphere(p.pos, p.mass, box, **kw)
        n = len(p)
    elif method == "density":
        c, info = _density_centre(p.pos, p.mass, box, kw.get("k", 6))
        n = len(p)
    elif method == "potential_min":
        if potential is None or len(potential) != len(p):
            raise ValueError("potential_min needs a potential array matching the particles")
        i = int(np.nanargmin(potential))
        c, n = p.pos[i].copy(), len(p)
        info = {"index": i}
    else:
        if p.ids is None or particle_id is None:
            raise ValueError("particle needs particle_id and a ParticleSet with ids")
        hit = np.flatnonzero(p.ids == particle_id)
        if len(hit) != 1:
            raise ValueError(f"particle_id {particle_id} matches {len(hit)} particles")
        c, n = p.pos[hit[0]].copy(), 1
        info = {"particle_id": int(particle_id)}
    out = Centre(pos=np.asarray(c, float), method=method, n_used=n,
                 parameters={k: v for k, v in kw.items() if np.isscalar(v)}, info=info)
    if velocity_radius is not None:
        out.vel, nv = centre_velocity(p, out.pos, velocity_radius)
        out.info["velocity_radius_kpc"] = velocity_radius
        out.info["velocity_n"] = nv
    return out
