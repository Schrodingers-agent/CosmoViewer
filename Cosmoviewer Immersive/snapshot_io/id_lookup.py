"""
Find particles by ParticleID in a snapshot, at full resolution.

GadgetViewer follows particles between snapshots by ID: a selection is a list
of IDs, and on every snapshot the IDs are matched against the particles read
(selection.F90, selection_apply). Here the display sample is decimated
independently in every snapshot, so a particle selected in one snapshot is
usually missing from the next snapshot's sample. This
module therefore reads the ParticleIDs column of every particle type straight
from the snapshot file(s), in chunks, and returns the requested particles with
their positions, velocities, masses and types, whatever the display sampling.

Only the ID column is read in full; the other columns are read only for chunks
that contain a requested particle. Memory is bounded by the chunk size plus
the number of requested IDs.

The type is looked up in every snapshot rather than assumed, because a gas
particle can become a star (CollidingGalaxiesSFR) and keeps its ID.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

CHUNK_ROWS = 1 << 22                 # 4M rows: 32 MB of uint64 IDs per read
FANCY_READ_MAX = 512                 # fewer hits than this in a chunk: read rows individually

VELOCITY_NAMES = ('Velocities', 'Velocity')
MASS_NAMES = ('Masses', 'Mass')

# Gadget-4 and EAGLE names for the cosmological scaling attributes of a dataset.
_A_KEYS = ('a_scaling', 'aexp-scale-exponent')
_H_KEYS = ('h_scaling', 'h-scale-exponent')
_CGS_KEYS = ('to_cgs', 'CGSConversionFactor')


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


def _first(group, names):
    for n in names:
        if n in group and isinstance(group[n], h5py.Dataset):
            return group[n]
    return None


def _scaling(dataset) -> Dict[str, float]:
    out = {}
    for key, names in (('a_scaling', _A_KEYS), ('h_scaling', _H_KEYS), ('to_cgs', _CGS_KEYS)):
        for n in names:
            if n in dataset.attrs:
                out[key] = float(_scalar(dataset.attrs[n]))
                break
    return out


@dataclass
class LocatedParticles:
    """Requested particles in one snapshot, aligned with ``ids`` (sorted, unique)."""
    ids: np.ndarray                    # (n,) uint64, sorted
    found: np.ndarray                  # (n,) bool
    types: np.ndarray                  # (n,) int8, -1 if not found
    pos: np.ndarray                    # (n, 3) as stored (float32 in Gadget-4), NaN if not found
    vel: Optional[np.ndarray]          # (n, 3) as stored, NaN if not found
    mass: np.ndarray                   # (n,) code units, NaN if not found
    duplicates: int = 0                # extra rows whose ID was already matched
    header: Dict = field(default_factory=dict)
    parameters: Dict = field(default_factory=dict)
    scaling: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 'Coordinates'/'Velocities'

    @property
    def n_found(self) -> int:
        return int(self.found.sum())

    def subset(self, mask: np.ndarray) -> 'LocatedParticles':
        return LocatedParticles(self.ids[mask], self.found[mask], self.types[mask], self.pos[mask],
                                None if self.vel is None else self.vel[mask], self.mass[mask],
                                self.duplicates, self.header, self.parameters, self.scaling)


def _attrs(group) -> Dict:
    return {k: _scalar(v) for k, v in group.attrs.items()}


def unique_ids(ids) -> np.ndarray:
    return np.unique(np.asarray(list(ids) if isinstance(ids, (set, frozenset)) else ids,
                                dtype=np.uint64))


def locate_ids(files: Sequence[str], ids, with_velocities: bool = True,
               chunk_rows: int = CHUNK_ROWS, use_index: bool = True) -> LocatedParticles:
    """
    Look up ``ids`` in every particle type of the snapshot made of ``files``.

    IDs that are not in the snapshot are reported as not found. If an ID occurs
    more than once (non-unique IDs), the first occurrence is used and the extra
    rows are counted in ``duplicates``; GadgetViewer's README warns of the same
    limitation.

    With ``use_index`` the snapshot's cached ID index (snapshot_io/id_index.py) finds the
    rows by binary search when the snapshot has one or can get one; the result is the same.
    """
    files = [str(f) for f in files]
    targets = unique_ids(ids)
    n = len(targets)
    found = np.zeros(n, dtype=bool)
    types = np.full(n, -1, dtype=np.int8)
    duplicates = 0
    scaling: Dict[str, Dict[str, float]] = {}

    with h5py.File(files[0], 'r') as f:
        header = _attrs(f['Header']) if 'Header' in f else {}
        params = {}
        for name in ('Parameters', 'Units'):
            if name in f:
                params.update(_attrs(f[name]))
        # Keep the stored precision (float32 in Gadget-4; float64 if the run wrote doubles),
        # so a cache over many snapshots is no larger than it needs to be.
        dtype = np.float32
        for t in range(6):
            g = f.get(f'PartType{t}')
            if g is not None and 'Coordinates' in g:
                dtype = np.promote_types(g['Coordinates'].dtype, np.float32)
                break
    pos = np.full((n, 3), np.nan, dtype=dtype)
    vel = np.full((n, 3), np.nan, dtype=dtype) if with_velocities else None
    mass = np.full(n, np.nan, dtype=dtype)
    mass_table = np.atleast_1d(header.get('MassTable', np.zeros(6)))

    if n == 0:
        return LocatedParticles(targets, found, types, pos, vel, mass, 0, header, params, scaling)

    hits = None                          # from the index: piece, type and row of every found ID
    if use_index:
        from snapshot_io.id_index import open_index
        index = open_index(files)
        if index is not None:
            ok, h_piece, h_type, h_row, duplicates = index.lookup(targets)
            hits = (np.flatnonzero(ok), h_piece, h_type, h_row)

    for piece, fname in enumerate(files):
        with h5py.File(fname, 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is None or 'ParticleIDs' not in g or 'Coordinates' not in g:
                    continue
                cd = g['Coordinates']
                vd = _first(g, VELOCITY_NAMES) if with_velocities else None
                md = _first(g, MASS_NAMES)
                if 'Coordinates' not in scaling:
                    scaling['Coordinates'] = _scaling(cd)
                if vd is not None and 'Velocities' not in scaling:
                    scaling['Velocities'] = _scaling(vd)
                m_table = float(mass_table[t]) if t < len(mass_table) else 0.0
                n_rows = g['ParticleIDs'].shape[0]
                if hits is not None:
                    here = (hits[1] == piece) & (hits[2] == t)
                    h_rows, h_slots = hits[3][here], hits[0][here]
                    order = np.argsort(h_rows)
                    h_rows, h_slots = h_rows[order], h_slots[order]
                for start in range(0, n_rows, chunk_rows):
                    stop = min(start + chunk_rows, n_rows)
                    if hits is not None:     # rows already known: only read the chunks that hold some
                        a, b = np.searchsorted(h_rows, [start, stop])
                        if a == b:
                            continue
                        rows, slots = h_rows[a:b] - start, h_slots[a:b]
                        pos[slots], types[slots], found[slots] = _read_located(
                            cd, vd, md, m_table, start, stop, rows, vel, slots, mass), t, True
                        continue
                    pid = g['ParticleIDs'][start:stop].astype(np.uint64)
                    slot = np.searchsorted(targets, pid)
                    slot[slot == n] = 0
                    hit = targets[slot] == pid
                    if not hit.any():
                        continue
                    rows = np.nonzero(hit)[0]
                    slots = slot[rows]
                    # First occurrence wins: drop rows whose ID is already matched,
                    # and repeated IDs within this chunk.
                    _, first = np.unique(slots, return_index=True)
                    keep = np.zeros(len(rows), dtype=bool)
                    keep[first] = True
                    keep &= ~found[slots]
                    duplicates += int(len(rows) - keep.sum())
                    rows, slots = rows[keep], slots[keep]
                    if len(rows) == 0:
                        continue

                    pos[slots] = _read_located(cd, vd, md, m_table, start, stop, rows, vel, slots, mass)
                    types[slots] = t
                    found[slots] = True

    return LocatedParticles(targets, found, types, pos, vel, mass, duplicates, header, params, scaling)


def _read_located(cd, vd, md, m_table, start, stop, rows, vel, slots, mass):
    """Positions of the increasing chunk rows ``rows`` (chunk start..stop); fills vel and mass."""
    def read(ds):
        if len(rows) <= FANCY_READ_MAX:
            return ds[start + rows]           # rows are increasing
        return ds[start:stop][rows]

    if vd is not None:
        vel[slots] = read(vd)
    mass[slots] = read(md) if md is not None else m_table
    return read(cd)


def select_in_sphere(files: Sequence[str], centre, radius: float, box: float = 0.0,
                     ptypes: Sequence[int] = range(6), chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
    """
    IDs of all particles (of ``ptypes``) within ``radius`` of ``centre``, read at
    full resolution. Distances use the minimum image when ``box`` > 0.
    """
    centre = np.asarray(centre, dtype=np.float64)
    out: List[np.ndarray] = []
    for fname in files:
        with h5py.File(str(fname), 'r') as f:
            for t in ptypes:
                g = f.get(f'PartType{t}')
                if g is None or 'ParticleIDs' not in g or 'Coordinates' not in g:
                    continue
                n_rows = g['Coordinates'].shape[0]
                for start in range(0, n_rows, chunk_rows):
                    stop = min(start + chunk_rows, n_rows)
                    d = g['Coordinates'][start:stop].astype(np.float64) - centre
                    if box > 0:
                        d -= box * np.round(d / box)
                    inside = np.einsum('ij,ij->i', d, d) <= radius * radius
                    if inside.any():
                        out.append(g['ParticleIDs'][start:stop][inside].astype(np.uint64))
    return np.concatenate(out) if out else np.empty(0, dtype=np.uint64)


def scale_factor(header: Dict, parameters: Dict) -> float:
    """a for a comoving run, 1 for a run without comoving integration."""
    comoving = parameters.get('ComovingIntegrationOn', header.get('ComovingIntegrationOn'))
    if comoving is not None and int(comoving) == 0:
        return 1.0
    if comoving is None and float(header.get('BoxSize', 0) or 0) == 0:
        return 1.0                     # isolated run without the flag: Time is code time, not a
    return float(header.get('Time', 1.0))


def physical_factor(scaling: Dict[str, float], a: float, h: float) -> Optional[float]:
    """
    Factor converting a stored value to physical code units:
    value * a**a_scaling * h**h_scaling (the Gadget-4 / EAGLE convention).
    None if the file does not carry the attributes.
    """
    if 'a_scaling' not in scaling or 'h_scaling' not in scaling:
        return None
    return float(a ** scaling['a_scaling'] * h ** scaling['h_scaling'])
