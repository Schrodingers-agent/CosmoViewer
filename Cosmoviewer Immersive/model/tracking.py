"""
Following particles across snapshots: centres, trajectories and export.

The behaviour follows GadgetViewer (selection.F90, movie.f90):
  * a selection is a set of ParticleIDs and stays selected when the snapshot
    changes;
  * the view can be recentred on the mean position of the selected particles,
    using the first selected particle as the reference for the periodic wrap;
  * an evolving movie smooths the path of the centre over the last N frames.

Positions here come from snapshot_io.id_lookup at full resolution, never from
the decimated display sample.  All positions are in the code units stored in
the file (comoving and per h for Gadget-4 cosmological runs); the export also
writes physical values when the file carries the a_scaling/h_scaling attributes.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from snapshot_io.id_lookup import LocatedParticles, physical_factor, scale_factor


def periodic_mean(pos: np.ndarray, box: float = 0.0) -> Optional[np.ndarray]:
    """
    Mean of ``pos`` (rows with NaN ignored). In a periodic box every point is
    first moved to the image nearest the first point, as GadgetViewer does in
    selection_recentre_view, and the mean is wrapped back into [0, box). This
    is well defined for a clump smaller than half the box.
    """
    p = np.asarray(pos, dtype=np.float64)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) == 0:
        return None
    if box > 0:
        d = p - p[0]
        p = p[0] + d - box * np.round(d / box)
        return np.mod(p.mean(axis=0), box)
    return p.mean(axis=0)


def sphere_wraps(centre, radius: float, box: float = 0.0) -> List[str]:
    """Axes ('x', 'y', 'z') along which a sphere crosses a face of a periodic box [0, box)."""
    if box <= 0:
        return []
    c = np.mod(np.asarray(centre, dtype=np.float64), box)
    return [a for a, ck in zip('xyz', c) if ck - radius < 0 or ck + radius > box]


def unwrap_for_display(pos: np.ndarray, box: float = 0.0) -> np.ndarray:
    """
    Positions of one set moved, axis by axis, to the images nearest the set's
    circular-mean centre, so a clump cut by a face of a periodic box is drawn in
    one piece (a sphere about y = 4.7 pokes out of the y = 0 face instead of
    showing a cap at y = 50). An axis is moved only if the moved set spans less
    than half the box along it; a set that fills the box is left as stored.
    For drawing only: membership and the stored positions do not change.
    """
    p = np.array(pos, dtype=np.float64, copy=True)
    if box <= 0 or len(p) == 0:
        return p
    for k in range(p.shape[1]):
        col = p[:, k]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        ang = col[ok] * (2.0 * np.pi / box)
        c = np.mod(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * box / (2.0 * np.pi), box)
        d = col[ok] - c
        d -= box * np.round(d / box)
        if d.max() - d.min() < 0.5 * box:
            col[ok] = c + d
    return p


def unwrap_series(pos: np.ndarray, box: float) -> np.ndarray:
    """
    Make a time series of positions continuous across periodic boundaries.

    ``pos`` has shape (n_snap, ..., 3); NaN marks a missing value. Each step is
    replaced by its minimum-image displacement from the previous finite value,
    which is correct as long as nothing moves more than half the box between
    two snapshots.
    """
    out = np.array(pos, dtype=np.float64, copy=True)
    if box <= 0 or len(out) < 2:
        return out
    last = out[0].copy()
    for i in range(1, len(out)):
        cur = out[i]
        ok = np.isfinite(cur) & np.isfinite(last)
        d = np.where(ok, cur - last, 0.0)
        d -= box * np.round(d / box)
        out[i] = np.where(ok, last + d, cur)
        last = np.where(np.isfinite(out[i]), out[i], last)
    return out


def smooth_centres(centres: List[Optional[np.ndarray]], i: int, n_smooth: int, box: float = 0.0):
    """
    GadgetViewer's movie centre: the mean of the raw centres of the last
    ``n_smooth`` frames (including frame ``i``) for which a centre exists.
    In a periodic box the centres are unwrapped relative to frame ``i`` first,
    so a path crossing the boundary is not averaged to the middle of the box.
    """
    cur = centres[i]
    if cur is None or n_smooth <= 1:
        return cur
    picked = []
    j = i
    while j >= 0 and len(picked) < n_smooth:
        if centres[j] is not None:
            picked.append(np.asarray(centres[j], dtype=np.float64))
        j -= 1
    p = np.array(picked)
    if box > 0:
        d = p - cur
        p = cur + d - box * np.round(d / box)
        return np.mod(p.mean(axis=0), box)
    return p.mean(axis=0)


@dataclass
class Trajectories:
    """Positions of a fixed list of particles across snapshots."""
    ids: np.ndarray                     # (n,) uint64
    times: np.ndarray                   # (n_snap,) header Time
    redshifts: np.ndarray               # (n_snap,) NaN if absent
    files: List[str]
    pos: np.ndarray                     # (n_snap, n, 3) code units as stored, NaN if missing
    vel: Optional[np.ndarray]           # (n_snap, n, 3) as stored
    types: np.ndarray                   # (n_snap, n) int8, -1 if missing
    mass: np.ndarray                    # (n_snap, n)
    box: float
    scaling: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pos_factor: Optional[np.ndarray] = None   # (n_snap,) stored -> physical code units
    vel_factor: Optional[np.ndarray] = None
    units: Dict[str, float] = field(default_factory=dict)
    # what the exported set is (name, describe, warning and the fields of selection_sets.py)
    population: Optional[dict] = None

    @property
    def n_snap(self) -> int:
        return len(self.times)

    def unwrapped(self) -> np.ndarray:
        return unwrap_series(self.pos, self.box)

    def centre_path(self) -> np.ndarray:
        """Per-snapshot centre (periodic mean) of the found particles, (n_snap, 3)."""
        out = np.full((self.n_snap, 3), np.nan)
        for i in range(self.n_snap):
            c = periodic_mean(self.pos[i], self.box)
            if c is not None:
                out[i] = c
        return unwrap_series(out, self.box)


def build_trajectories(located: Sequence[LocatedParticles], files: Sequence[str]) -> Trajectories:
    """Stack per-snapshot lookups of the same ID list (snapshot order as given)."""
    if not located:
        raise ValueError("no snapshots")
    ids = located[0].ids
    for loc in located[1:]:
        if len(loc.ids) != len(ids) or not np.array_equal(loc.ids, ids):
            raise ValueError("lookups are for different ID lists")
    n_snap = len(located)
    times = np.array([float(l.header.get('Time', np.nan)) for l in located])
    reds = np.array([float(l.header.get('Redshift', np.nan)) for l in located])
    pos = np.stack([l.pos for l in located])
    vel = np.stack([l.vel for l in located]) if all(l.vel is not None for l in located) else None
    types = np.stack([l.types for l in located])
    mass = np.stack([l.mass for l in located])
    box = float(located[0].header.get('BoxSize', 0.0) or 0.0)
    scaling = located[0].scaling
    params = located[0].parameters
    h = float(params.get('HubbleParam', located[0].header.get('HubbleParam', 1.0)) or 1.0)
    pf = [physical_factor(scaling.get('Coordinates', {}), scale_factor(l.header, l.parameters), h)
          for l in located]
    vf = [physical_factor(scaling.get('Velocities', {}), scale_factor(l.header, l.parameters), h)
          for l in located]
    units = {k: float(params[k]) for k in ('UnitLength_in_cm', 'UnitMass_in_g',
                                           'UnitVelocity_in_cm_per_s', 'HubbleParam') if k in params}
    return Trajectories(ids, times, reds, [str(f) for f in files], pos, vel, types, mass, box, scaling,
                        None if any(x is None for x in pf) else np.array(pf),
                        None if any(x is None for x in vf) else np.array(vf), units)


def _metadata(tr: Trajectories) -> dict:
    meta = {
        'description': "Particle trajectories exported by CosmoViewer Immersive. "
                       "pos/vel are the values stored in the snapshot (code units; comoving "
                       "and per h in Gadget-4 cosmological runs). pos_physical = pos * a**a_scaling "
                       "* h**h_scaling (code length units without h), likewise vel_physical; "
                       "absent when the file has no scaling attributes. pos_unwrapped removes "
                       "periodic jumps between consecutive snapshots.",
        'box_size': tr.box,
        'scaling_attributes': tr.scaling,
        'units': tr.units,
        'files': tr.files,
    }
    if tr.population:
        meta['population'] = tr.population      # which particles these are, and how they were chosen
    return meta


def export_npz(tr: Trajectories, path: str) -> str:
    out = dict(ids=tr.ids, times=tr.times, redshifts=tr.redshifts, pos=tr.pos,
               pos_unwrapped=tr.unwrapped(), types=tr.types, mass=tr.mass,
               centre=tr.centre_path(), metadata=json.dumps(_metadata(tr)))
    if tr.vel is not None:
        out['vel'] = tr.vel
    if tr.pos_factor is not None:
        out['pos_physical'] = tr.unwrapped() * tr.pos_factor[:, None, None]
    if tr.vel is not None and tr.vel_factor is not None:
        out['vel_physical'] = tr.vel * tr.vel_factor[:, None, None]
    np.savez_compressed(path, **out)
    return path


def export_csv(tr: Trajectories, path: str) -> str:
    """Long format: one row per (snapshot, particle) that was found."""
    unw = tr.unwrapped()
    with open(path, 'w', newline='') as fh:
        fh.write("# " + json.dumps(_metadata(tr)) + "\n")
        w = csv.writer(fh)
        cols = ['snapshot', 'file', 'time', 'redshift', 'id', 'type', 'x', 'y', 'z',
                'x_unwrapped', 'y_unwrapped', 'z_unwrapped', 'mass']
        if tr.vel is not None:
            cols += ['vx', 'vy', 'vz']
        w.writerow(cols)
        for i in range(tr.n_snap):
            name = Path(tr.files[i]).name if i < len(tr.files) else ''
            for k in np.nonzero(tr.types[i] >= 0)[0]:
                row = [i, name, f"{tr.times[i]:.8g}", f"{tr.redshifts[i]:.8g}", int(tr.ids[k]),
                       int(tr.types[i, k]), *(f"{x:.9g}" for x in tr.pos[i, k]),
                       *(f"{x:.9g}" for x in unw[i, k]), f"{tr.mass[i, k]:.9g}"]
                if tr.vel is not None:
                    row += [f"{x:.9g}" for x in tr.vel[i, k]]
                w.writerow(row)
    return path


def parse_id_text(text: str) -> np.ndarray:
    """'12, 40-45 99' -> IDs. Ranges are inclusive. Raises ValueError on bad input."""
    def whole(s: str, tok: str) -> int:
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"'{tok}' is not a particle ID: use whole numbers and ranges "
                             f"such as 5000-5010") from None
    ids: List[int] = []
    for tok in text.replace(',', ' ').split():
        if '-' in tok[1:]:
            a, b = tok.split('-', 1)
            lo, hi = whole(a, tok), whole(b, tok)
            if hi < lo:
                raise ValueError(f"bad range {tok}")
            if hi - lo > 50_000_000:
                raise ValueError(f"range {tok} too large")
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(whole(tok, tok))
    if any(i < 0 for i in ids):
        raise ValueError("IDs must be non-negative")
    return np.asarray(ids, dtype=np.uint64)


def read_id_file(path: str) -> np.ndarray:
    """IDs from a text file (any whitespace or commas; '#' starts a comment), a .npy array,
    or a CosmoViewer selection-set file (.hdf5, selection_sets.py)."""
    if str(path).endswith(('.hdf5', '.h5')):
        from selection_sets import read_set
        return read_set(path).ids
    if str(path).endswith('.npy'):
        return np.asarray(np.load(path), dtype=np.uint64).ravel()
    with open(path) as fh:
        text = " ".join(line.split('#', 1)[0] for line in fh)
    return parse_id_text(text)


def write_id_file(path: str, ids, header: str = "") -> str:
    ids = np.unique(np.asarray(list(ids) if isinstance(ids, (set, frozenset)) else ids, dtype=np.uint64))
    with open(path, 'w') as fh:
        if header:
            fh.write(f"# {header}\n")
        for i in ids:
            fh.write(f"{int(i)}\n")
    return path
