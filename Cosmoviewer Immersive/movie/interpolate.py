"""
In-between movie frames: particle positions between two snapshots.

A run with few snapshots (DM-L50 has three) makes a jumpy evolving movie.
Between snapshots A and B this places every particle drawn in A on a cubic
Hermite curve that passes through its position in A and in B with the
velocity it has in each snapshot, so motion is smooth and continuous in
direction across snapshots.

* Particles are matched by ParticleID. A particle of A that is not in B's
  display sample is looked up in B's file(s) at full resolution; one that is
  not in B at all (accreted, removed) is dropped from the in-between frames.
* In a periodic box B's position is first moved to the image nearest A's.
* The curve's parameter is the snapshot time variable (the scale factor a in
  a comoving run, the time otherwise). Stored velocities are converted to
  d(stored position)/d(time variable) with ``dx_dtime_factor``; for Gadget-4
  comoving runs the stored velocity is u = v_pec / sqrt(a) (Velocities
  a_scaling = 0.5), so dx/da = u a^(s-2) / H(a) with s = 0.5.
* When snapshots are far apart compared with how fast particles turn (an
  orbit or more between them), velocity tangents are much longer than the
  straight line between the two positions and the curve would overshoot
  wildly; tangents are then shortened to MAX_TANGENT times that line. The
  number of particles affected is reported.
* At fraction 0 and 1 the curve gives A's and B's positions exactly; the
  movie uses the snapshots themselves at those points and calls this only in
  between.

In-between frames are pictures for movies. Nothing here is used for any
quantity reported to the user, and every such frame is stamped "interpolated".
"""

from dataclasses import replace
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from model.tracking import periodic_mean

HUBBLE_PER_S = 3.2407789e-18          # 100 km/s/Mpc in 1/s (times h)
MAX_TANGENT = 3.0                     # longest tangent, in units of the chord between the snapshots


def dx_dtime_factor(header: Dict, params: Dict, vel_a_scaling: Optional[float]) -> Optional[float]:
    """
    Factor f with d(stored position)/d(Time) = f * (stored velocity), Time being the
    header's Time (a for comoving runs). None if the run's cosmology is incomplete.
    """
    comoving = params.get('ComovingIntegrationOn', header.get('ComovingIntegrationOn'))
    box = float(header.get('BoxSize', 0) or 0)
    if comoving is None:
        comoving = 1 if box > 0 else 0
    if int(comoving) == 0:
        # Non-comoving: stored velocity is dx/dt in code units (a_scaling 0).
        return 1.0
    a = float(header.get('Time', 0) or 0)
    if a <= 0:
        return None
    om = params.get('Omega0', header.get('Omega0'))
    ol = params.get('OmegaLambda', header.get('OmegaLambda'))
    if om is None or ol is None:
        return None
    ul = float(params.get('UnitLength_in_cm', 3.085678e21))
    uv = float(params.get('UnitVelocity_in_cm_per_s', 1e5))
    h0 = HUBBLE_PER_S * ul / uv                    # H0 in code units (lengths per h)
    om, ol = float(om), float(ol)
    hubble = h0 * np.sqrt(om / a ** 3 + (1.0 - om - ol) / a ** 2 + ol)
    s = 0.5 if vel_a_scaling is None else float(vel_a_scaling)   # Gadget convention if unlabelled
    # v_pec = u a^s = a dx/dt and da/dt = a H  =>  dx/da = u a^(s-2) / H
    return float(a ** (s - 2.0) / hubble)


_VEL_A_SCALING: Dict[str, Optional[float]] = {}


def velocity_a_scaling(filepath: str) -> Optional[float]:
    """The a exponent of the stored velocities (Gadget-4 and EAGLE names), None if unlabelled."""
    if filepath not in _VEL_A_SCALING:
        val = None
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                for t in range(6):
                    g = f.get(f'PartType{t}')
                    if g is None:
                        continue
                    for name in ('Velocities', 'Velocity'):
                        if name in g:
                            a = g[name].attrs
                            for key in ('a_scaling', 'aexp-scale-exponent'):
                                if key in a:
                                    val = float(np.asarray(a[key]).ravel()[0])
                            break
                    if val is not None:
                        break
        except (OSError, ValueError):
            pass
        _VEL_A_SCALING[filepath] = val
    return _VEL_A_SCALING[filepath]


def hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, f: float) -> np.ndarray:
    """Cubic Hermite curve at f in [0, 1]; m0, m1 are tangents already multiplied by the interval."""
    f2, f3 = f * f, f * f * f
    return ((2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * m0
            + (-2 * f3 + 3 * f2) * p1 + (f3 - f2) * m1)


def _min_image(p0: np.ndarray, p1: np.ndarray, box: float) -> np.ndarray:
    if box <= 0:
        return p1
    d = p1 - p0
    return p0 + d - box * np.round(d / box)


# Lookup of (pos, vel, found) for sorted unique IDs in a snapshot's files, at full resolution.
Locator = Callable[[Sequence[str], np.ndarray], Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]


def _default_locator(files, ids):
    from snapshot_io.id_lookup import locate_ids
    loc = locate_ids(files, ids, with_velocities=True)
    return loc.pos, loc.vel, loc.found


class PairInterpolator:
    """Positions of scene A's particles at any fraction between scene A and scene B."""

    LAYERS = (('pos', 'ids', 'vel'), ('lit_pos', 'lit_ids', 'lit_vel'),
              ('highlight_pos', 'highlight_ids', 'highlight_vel'))

    def __init__(self, a, b, locator: Locator = _default_locator):
        self.a, self.b = a, b
        self.box = float(a.box_period or 0.0)
        self.dt = float(b.time) - float(a.time)
        self.dropped = 0
        self.looked_up = 0
        self.limited = 0                 # particles whose tangents were shortened (see MAX_TANGENT)
        self.n_curves = 0
        self.tangents = a.dxdt is not None and b.dxdt is not None
        # Every particle B knows about, by ID: its drawn layers first.
        b_ids, b_pos, b_vel = [], [], []
        for p, i, v in self.LAYERS:
            pos, ids = getattr(b, p), getattr(b, i)
            if pos is None or ids is None or len(pos) == 0:
                continue
            vel = getattr(b, v)
            b_ids.append(np.asarray(ids, dtype=np.uint64))
            b_pos.append(np.asarray(pos, dtype=np.float64))
            b_vel.append(np.full((len(pos), 3), np.nan) if vel is None else np.asarray(vel, dtype=np.float64))
        if b_ids:
            ids = np.concatenate(b_ids)
            uniq, first = np.unique(ids, return_index=True)
            self._b_ids = uniq
            self._b_pos = np.vstack(b_pos)[first]
            self._b_vel = np.vstack(b_vel)[first]
        else:
            self._b_ids = np.empty(0, dtype=np.uint64)
            self._b_pos = self._b_vel = np.empty((0, 3))
        self._layers = {}
        missing = []
        for p, i, v in self.LAYERS:
            pos, ids = getattr(a, p), getattr(a, i)
            if pos is None or ids is None or len(pos) == 0:
                continue
            ids = np.asarray(ids, dtype=np.uint64)
            slot = np.searchsorted(self._b_ids, ids)
            slot[slot >= len(self._b_ids)] = 0
            hit = (self._b_ids[slot] == ids) if len(self._b_ids) else np.zeros(len(ids), dtype=bool)
            self._layers[p] = (ids, slot, hit)
            missing.append(ids[~hit])
        extra = np.unique(np.concatenate(missing)) if missing else np.empty(0, dtype=np.uint64)
        if len(extra) and b.files:
            pos, vel, found = locator(b.files, extra)
            self.looked_up = int(found.sum())
            self._x_ids = extra[found]
            self._x_pos = np.asarray(pos, dtype=np.float64)[found]
            self._x_vel = (np.full((int(found.sum()), 3), np.nan) if vel is None
                           else np.asarray(vel, dtype=np.float64)[found])
        else:
            self._x_ids = np.empty(0, dtype=np.uint64)
            self._x_pos = self._x_vel = np.empty((0, 3))
        self._curves = {}
        for p, i, v in self.LAYERS:
            if p in self._layers:
                self._curves[p] = self._curve(p, v)

    def _curve(self, p: str, v: str):
        ids, slot, hit = self._layers[p]
        n = len(ids)
        p1 = np.full((n, 3), np.nan)
        v1 = np.full((n, 3), np.nan)
        if hit.any():
            p1[hit] = self._b_pos[slot[hit]]
            v1[hit] = self._b_vel[slot[hit]]
        if (~hit).any() and len(self._x_ids):
            k = np.searchsorted(self._x_ids, ids[~hit])
            k[k >= len(self._x_ids)] = 0
            ok = self._x_ids[k] == ids[~hit]
            rows = np.nonzero(~hit)[0][ok]
            p1[rows] = self._x_pos[k[ok]]
            v1[rows] = self._x_vel[k[ok]]
        keep = np.isfinite(p1).all(axis=1)
        self.dropped += int(n - keep.sum())
        p0 = np.asarray(getattr(self.a, p), dtype=np.float64)[keep]
        p1 = _min_image(p0, p1[keep], self.box)
        va = getattr(self.a, v)
        v0 = None if va is None else np.asarray(va, dtype=np.float64)[keep]
        v1 = v1[keep]
        if self.tangents and v0 is not None and np.isfinite(v1).all():
            m0 = v0 * (self.a.dxdt * self.dt)
            m1 = v1 * (self.b.dxdt * self.dt)
            # Snapshots far apart compared with the orbital time (CollidingGalaxies: 000 -> 100)
            # give tangents much longer than the chord, and the curve would fling particles far
            # out. Shorten such tangents to MAX_TANGENT chords: the path is unknown there anyway.
            chord = np.linalg.norm(p1 - p0, axis=1)
            cap = MAX_TANGENT * chord + 1e-12
            lim = np.zeros(len(p0), dtype=bool)
            for m in (m0, m1):
                n = np.linalg.norm(m, axis=1)
                over = n > cap
                m[over] *= (cap[over] / n[over])[:, None]
                lim |= over
            self.limited += int(lim.sum())
        else:
            # No usable velocities: tangents of a straight line (linear interpolation).
            m0 = m1 = p1 - p0
        self.n_curves += len(p0)
        return keep, p0, p1, m0, m1

    def scene_at(self, f: float):
        """Scene A with its particles moved to fraction f (0 < f < 1) of the way to B."""
        a = self.a
        changes = {}
        for p, i, v in self.LAYERS:
            if p not in self._curves:
                continue
            keep, p0, p1, m0, m1 = self._curves[p]
            x = hermite(p0, p1, m0, m1, f)
            if self.box > 0:
                x = np.mod(x, self.box)
            changes[p] = x
            changes[i] = np.asarray(getattr(a, i))[keep]
            changes[v] = None                    # velocities in between are not defined here
            for extra in self._aligned(p):
                arr = getattr(a, extra)
                if arr is not None:
                    changes[extra] = np.asarray(arr)[keep]
        t = float(a.time) + f * self.dt
        hl = changes.get('highlight_pos', a.highlight_pos)
        centre = periodic_mean(hl, self.box) if hl is not None and len(hl) else None
        if centre is None and a.centre is not None and self.b.centre is not None:
            c0 = np.asarray(a.centre, dtype=np.float64)
            c1 = _min_image(c0, np.asarray(self.b.centre, dtype=np.float64), self.box)
            centre = c0 + f * (c1 - c0)
        return replace(a, centre=centre, time=t, label=interpolated_label(a, t), interpolated=True, **changes)

    @staticmethod
    def _aligned(p: str):
        return {'pos': ('colors', 'mass', 'hsml'),
                'lit_pos': ('lit_colors', 'lit_types', 'lit_values', 'lit_mass', 'lit_hsml'),
                'highlight_pos': ('highlight_colors',)}[p]


def interpolated_label(scene, t: float) -> str:
    if scene.redshift is not None and t > 0:
        return f"a = {t:.4f}   z = {1.0 / t - 1.0:.3f}   (interpolated)"
    return f"t = {t:.4g} (code units, interpolated)"
