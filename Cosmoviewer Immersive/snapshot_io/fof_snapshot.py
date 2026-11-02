"""
Run FoF on a Gadget-4 HDF5 snapshot at full resolution.

The interactive view only holds a decimated subset of particles, so FoF must
not run on it: decimation changes both the particle count used for the mean
separation and which particles can link.  This module reads the primary
link type(s) straight from the snapshot file(s), reading only Coordinates,
ParticleIDs and (if present) Masses, and returns group membership keyed by
ParticleID so any view, decimated or not, can look it up.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

from .fof import (GADGET4_DEFAULT_B, GADGET4_DEFAULT_MIN_LEN, UNGROUPED, FoFResult, Progress,
                  attach_secondaries, find_groups, mean_separation_from_box,
                  mean_separation_gadget4)


def snapshot_file_list(path: str) -> List[str]:
    """All files of the snapshot that ``path`` belongs to, in order.

    Gadget writes multi-file snapshots as ``<base>.<i>.hdf5``; a single-file
    snapshot is returned as-is.  Missing pieces raise instead of being skipped,
    because silently dropping a file would drop particles.
    """
    with h5py.File(path, 'r') as f:
        nfiles = int(np.atleast_1d(f['Header'].attrs.get('NumFilesPerSnapshot', 1))[0])
    if nfiles <= 1:
        return [path]
    m = re.match(r'^(.*)\.(\d+)\.(hdf5|h5)$', path)
    if not m:
        raise ValueError(f"{path} says NumFilesPerSnapshot={nfiles} but is not named <base>.<i>.hdf5")
    files = [f"{m.group(1)}.{i}.{m.group(3)}" for i in range(nfiles)]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"snapshot is missing {len(missing)} of {nfiles} files, e.g. {missing[0]}")
    return files


@dataclass
class SnapshotParticles:
    positions: np.ndarray
    ids: np.ndarray
    masses: np.ndarray
    header: Dict
    parameters: Dict
    types: np.ndarray = field(default=None)


def _attrs(group) -> Dict:
    return {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v) for k, v in group.attrs.items()}


_READ_ROWS = 1 << 22  # rows per read, so progress moves and no dataset is held twice


def read_particles(path: str, ptypes: Sequence[int], progress: Progress = None) -> SnapshotParticles:
    """Concatenate Coordinates, ParticleIDs and masses of ``ptypes`` across files.

    Arrays are allocated once at full size and filled in blocks, so the peak is
    one copy of each (positions as float64, masses as float64).
    """
    files = snapshot_file_list(path)
    with h5py.File(files[0], 'r') as f:
        header = _attrs(f['Header'])
        params = _attrs(f['Parameters']) if 'Parameters' in f else {}
    mass_table = np.atleast_1d(header.get('MassTable', np.zeros(6)))
    pieces, id_types = [], []
    for fname in files:
        with h5py.File(fname, 'r') as f:
            for t in ptypes:
                g = f.get(f'PartType{t}')
                if g is None or 'Coordinates' not in g:
                    continue
                pieces.append((fname, t, int(g['Coordinates'].shape[0])))
                id_types.append(g['ParticleIDs'].dtype)
    expected = np.atleast_1d(header.get('NumPart_Total', np.zeros(6))).astype(np.int64)
    high = np.atleast_1d(header.get('NumPart_Total_HighWord', np.zeros(len(expected)))).astype(np.int64)
    expected = expected + (high << 32)
    n_read = sum(m for _, _, m in pieces)
    n_expected = int(sum(expected[t] for t in ptypes if t < len(expected)))
    if n_read != n_expected:
        raise ValueError(f"read {n_read} particles of types {list(ptypes)} but header says {n_expected}")
    if not pieces:
        z = np.zeros(0)
        return SnapshotParticles(np.zeros((0, 3)), z.astype(np.int64), z, header, params, z.astype(np.int8))
    pos = np.empty((n_read, 3), dtype=np.float64)
    ids = np.empty(n_read, dtype=np.result_type(*id_types))
    mass = np.empty(n_read, dtype=np.float64)
    types = np.empty(n_read, dtype=np.int8)
    k = 0
    for fname, t, m in pieces:
        with h5py.File(fname, 'r') as f:
            g = f[f'PartType{t}']
            for s0 in range(0, m, _READ_ROWS):
                e0 = min(s0 + _READ_ROWS, m)
                dst = slice(k + s0, k + e0)
                pos[dst] = g['Coordinates'][s0:e0]
                ids[dst] = g['ParticleIDs'][s0:e0]
                if 'Masses' in g:
                    mass[dst] = g['Masses'][s0:e0]
                else:
                    mass[dst] = float(mass_table[t]) if t < len(mass_table) else 0.0
                if progress is not None:
                    progress((k + e0) / n_read, f"Reading particles ({k + e0:,} of {n_read:,})")
            types[k:k + m] = t
        k += m
    return SnapshotParticles(pos, ids, mass, header, params, types)


def linking_length_for(prim: SnapshotParticles, b: float, convention: str = 'gadget4') -> float:
    """b times the mean primary separation, using Gadget-4's convention by default.

    'gadget4' needs Omega0/OmegaBaryon and units from the Parameters group and
    non-zero masses; 'box' uses (BoxSize^3 / N)^(1/3) and needs a box size.
    """
    h, p = prim.header, prim.parameters
    if convention == 'gadget4':
        has_gas = int(np.atleast_1d(h.get('NumPart_Total', np.zeros(6)))[0]) > 0
        d = mean_separation_gadget4(prim.masses, float(p['Omega0']), float(p.get('OmegaBaryon', 0.0)), has_gas,
                                    float(p['UnitLength_in_cm']), float(p['UnitMass_in_g']),
                                    float(p['UnitVelocity_in_cm_per_s']))
    elif convention == 'box':
        d = mean_separation_from_box(len(prim.positions), float(h['BoxSize']))
    else:
        raise ValueError(f"unknown linking-length convention {convention!r}")
    return b * d


@dataclass
class SnapshotFoF:
    """FoF groups for one snapshot, keyed by particle ID."""
    result: FoFResult
    ids: np.ndarray            # IDs of every particle that took part (primary and secondary)
    group_index: np.ndarray    # group of each entry in ids (UNGROUPED = -1)
    boxsize: Optional[float]
    convention: str
    ids_sorted: bool = False   # ids ascending (see sorted_by_id), so lookups need no sort

    def sorted_by_id(self) -> 'SnapshotFoF':
        """The same membership with ids ascending (stable, so a repeated ID keeps its first entry first)."""
        if self.ids_sorted:
            return self
        order = np.argsort(self.ids, kind='stable')
        return SnapshotFoF(self.result, self.ids[order], self.group_index[order], self.boxsize,
                           self.convention, ids_sorted=True)

    def lookup(self, query_ids: np.ndarray) -> np.ndarray:
        """Group index for arbitrary particle IDs; IDs not in the run get UNGROUPED."""
        if self.ids_sorted:
            sid, gidx = self.ids, self.group_index
        else:
            order = np.argsort(self.ids, kind='stable')
            sid, gidx = self.ids[order], self.group_index[order]
        q = np.asarray(query_ids)
        pos = np.clip(np.searchsorted(sid, q), 0, max(len(sid) - 1, 0))
        out = np.full(len(q), UNGROUPED, dtype=np.int64)
        if len(sid):
            hit = sid[pos] == q
            out[hit] = gidx[pos[hit]]
        return out

# Peak memory per linked particle: positions, IDs, masses, grid sort keys and
# per-cell arrays.  Measured 107 B/particle (process peak less imports) on a
# uniform 8M-particle box, the worst case for the per-cell arrays; 217 B before
# the grid linker's memory was cut.  The FoF process is also stopped if it
# passes its memory limit (snapshot_io/fof_process.py).
FOF_BYTES_PER_PARTICLE = 120


def primary_count(path: str, ptypes: Sequence[int]) -> int:
    """Total particles of ``ptypes`` in the snapshot, from the header only."""
    with h5py.File(path, 'r') as f:
        h = f['Header'].attrs
        tot = np.atleast_1d(h.get('NumPart_Total', np.zeros(6))).astype(np.int64)
        high = np.atleast_1d(h.get('NumPart_Total_HighWord', np.zeros(len(tot)))).astype(np.int64)
    tot = tot + (high << 32)
    return int(sum(tot[t] for t in ptypes if t < len(tot)))


def physical_memory_bytes() -> Optional[int]:
    try:
        return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        return None


def check_fof_fits_in_memory(path: str, ptypes: Sequence[int], max_fraction: float = 0.5) -> None:
    """Refuse a full-resolution FoF whose estimated peak exceeds max_fraction of RAM."""
    n = primary_count(path, ptypes)
    ram = physical_memory_bytes()
    need = n * FOF_BYTES_PER_PARTICLE
    if ram and need > max_fraction * ram:
        raise MemoryError(f"FoF on {n:,} particles of types {list(ptypes)} needs about {need / 2**30:.1f} GB, "
                          f"more than {max_fraction:.0%} of this machine's {ram / 2**30:.1f} GB. "
                          f"Link fewer particle types, or run FoF on a machine with more memory.")


def _stage(progress: Progress, lo: float, hi: float) -> Progress:
    """progress for one stage, mapped onto [lo, hi] of the whole run."""
    if progress is None:
        return None
    return lambda f, text: progress(lo + (hi - lo) * f, text)


def run_snapshot_fof(path: str, b: float = GADGET4_DEFAULT_B, min_group_size: int = GADGET4_DEFAULT_MIN_LEN,
                     primary_types: Sequence[int] = (1,), secondary_types: Sequence[int] = (),
                     convention: str = 'gadget4', periodic: Optional[bool] = None,
                     linking_length: Optional[float] = None,
                     max_memory_fraction: Optional[float] = 0.5,
                     progress: Progress = None) -> SnapshotFoF:
    """FoF on a snapshot with Gadget-4's defaults (b=0.2, N_min=32, DM links).

    periodic=None treats the box as periodic when the header has BoxSize > 0.
    linking_length, if given, is used as-is (code units) and b is ignored; this
    is required for isolated runs (BoxSize 0), where no mean density exists.
    max_memory_fraction=None disables the memory check.
    progress, if given, is called as progress(fraction, text) while it runs.
    The result's ids are sorted (SnapshotFoF.sorted_by_id).
    """
    if max_memory_fraction is not None:
        check_fof_fits_in_memory(path, primary_types, max_memory_fraction)
    prim = read_particles(path, primary_types, progress=_stage(progress, 0.0, 0.25))
    if len(prim.positions) == 0:
        raise ValueError(f"no particles of primary type(s) {list(primary_types)} in {path}")
    box = float(prim.header.get('BoxSize', 0.0) or 0.0)
    if periodic is None:
        periodic = box > 0
    boxsize = box if periodic else None
    if linking_length is not None:
        link = float(linking_length)
    elif not periodic:
        raise ValueError("isolated (non-periodic) run: give an explicit linking length")
    elif convention == 'gadget4' and not {'Omega0', 'UnitLength_in_cm'} <= set(prim.parameters):
        raise ValueError("snapshot has no cosmology/units in its Parameters group; use convention='box'")
    else:
        link = linking_length_for(prim, b, convention)
    prim.masses = prim.types = None    # not needed past the linking length; frees memory for linking
    res = find_groups(prim.positions, link, boxsize=boxsize, min_group_size=min_group_size, ids=prim.ids,
                      progress=_stage(progress, 0.25, 0.9), overwrite_positions=True)
    if progress is not None:
        progress(0.9, "Numbering groups")
    ids, gidx = prim.ids, res.group_index
    if secondary_types:
        sec = read_particles(path, secondary_types, progress=_stage(progress, 0.9, 0.95))
        if progress is not None:
            progress(0.95, "Attaching other particle types")
        sg = attach_secondaries(prim.positions, res.group_index, sec.positions, link, boxsize)
        ids, gidx = np.concatenate([ids, sec.ids]), np.concatenate([gidx, sg])
    del prim
    out = SnapshotFoF(res, ids, gidx, boxsize, convention).sorted_by_id()
    if progress is not None:
        progress(1.0, "Done")
    return out


PARTICLE_TYPE_LABELS = {0: "Gas", 1: "Dark matter", 2: "Disk", 3: "Bulge", 4: "Stars", 5: "Black holes"}


def fof_defaults(path: str) -> Dict:
    """Suggested FoF settings for a snapshot, from its header and parameters.

    Periodic (cosmological) box: link dark matter with b = 0.2, as Gadget-4 does.
    Isolated run (BoxSize 0): there is no mean density, so b is meaningless and a
    fixed linking length is needed.  Dark-matter haloes of isolated galaxies are
    extended and overlap, so the default links the baryonic types instead and
    suggests 10x the largest gravitational softening of those types.  On the
    CollidingGalaxiesSFR ICs this separates the two galaxies for any length
    from 0.5 to 10 kpc/h; the suggestion (1 kpc/h) sits inside that range.
    """
    with h5py.File(snapshot_file_list(path)[0], 'r') as f:
        header = _attrs(f['Header'])
        params = _attrs(f['Parameters']) if 'Parameters' in f else {}
    counts = np.atleast_1d(header.get('NumPart_Total', np.zeros(6))).astype(np.int64)
    present = [t for t in range(len(counts)) if counts[t] > 0]
    periodic = float(header.get('BoxSize', 0.0) or 0.0) > 0
    if periodic:
        types = [1] if 1 in present else present
    else:
        types = [t for t in present if t not in (1, 5)] or present
    soft = []
    for t in types:
        cls = params.get(f'SofteningClassOfPartType{t}')
        val = params.get(f'SofteningComovingClass{int(cls)}') if cls is not None else None
        if val is not None and float(val) > 0:
            soft.append(float(val))
    suggested = 10.0 * max(soft) if soft else None
    length_cm = float(params.get('UnitLength_in_cm', 0.0) or 0.0)
    def _fmt(factor, name):
        return name if abs(factor - 1) < 1e-3 else f"{factor:g} {name}"
    unit = _fmt(length_cm / 3.085678e21, "kpc") if length_cm else "code units"
    if length_cm >= 3.085678e24 * 0.999:
        unit = _fmt(length_cm / 3.085678e24, "Mpc")
    if float(params.get('HubbleParam', 1.0) or 1.0) != 1.0 or params.get('ComovingIntegrationOn', 0):
        unit += "/h"
    return {'periodic': periodic, 'present_types': present, 'link_types': types,
            'suggested_length': suggested, 'length_unit': unit}
