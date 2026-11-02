"""
Particle properties beyond positions: listing and reading any dataset in a snapshot.

GadgetViewer colours, selects and plots by "any quantity that can be read from
the snapshot". The display loader reads only Coordinates, velocities, masses
and IDs, so this module reads the other datasets on demand:

- ``list_properties`` lists every per-particle dataset (scalars, vector
  components and magnitudes, EAGLE sub-groups such as ElementAbundance/Oxygen)
  plus a few derived quantities (Masses from MassTable, Temperature from
  InternalEnergy, speed), and halo membership stored per particle (EAGLE's
  GroupNumber / SubGroupNumber) as 'Group (snapshot)' and 'Subhalo (snapshot)'.
- ``read_sample_values`` gives one property for exactly the rows held in the
  display sample or a LOD view set, from a read-once cache of the whole
  dataset (snapshot_io.property_cache). Rows are checked against ParticleIDs;
  if the check fails (for example because the loader's sampling changed),
  values are matched by ID instead, so a property can never be attached to
  the wrong particle.
- ``select_by_range`` returns the IDs of every particle in the full snapshot
  whose property lies in [lo, hi], read in chunks.

Values are returned as stored in the file (code units, comoving and per h where
the run is cosmological). ``PropertyInfo.scaling`` carries the Gadget-4 /
EAGLE attributes needed to convert them, and ``unit_label`` says what they are.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

CHUNK_ROWS = 1 << 22

# Physical constants (cgs) for derived temperature.
BOLTZMANN = 1.380649e-16
PROTON_MASS = 1.67262192e-24
HYDROGEN_MASS_FRACTION = 0.76          # primordial X, as in Gadget's cooling module
GAMMA = 5.0 / 3.0

_A_KEYS = ('a_scaling', 'aexp-scale-exponent')
_H_KEYS = ('h_scaling', 'h-scale-exponent')
_CGS_KEYS = ('to_cgs', 'CGSConversionFactor')

# Always handled by the loader or meaningless as a colour.
_SKIP = {'ParticleIDs'}
MASS_NAMES = ('Masses', 'Mass')

# Halo membership written into the snapshot by the halo finder (EAGLE): GroupNumber is the
# FoF group (1 = first), SubGroupNumber the subhalo within it (0 = the main subhalo), and
# 2**30 means "none" in both. 'Group (snapshot)' is GroupNumber with 0 for none, like the
# catalogue's groups. 'Subhalo (snapshot)' must tell subhaloes of different groups apart,
# so it is group * SUBHALO_STRIDE + SubGroupNumber (group 12, subhalo 3 -> 1200003), 0 for none.
SNAP_GROUP = 'Group (snapshot)'
SNAP_SUBHALO = 'Subhalo (snapshot)'
NO_HALO = 1 << 30
SUBHALO_STRIDE = 100_000


def snapshot_group(gn: np.ndarray) -> np.ndarray:
    gn = np.asarray(gn, dtype=np.int64)
    return np.where((gn > 0) & (gn < NO_HALO), gn, 0)


def snapshot_subhalo(gn: np.ndarray, sg: np.ndarray) -> np.ndarray:
    g = snapshot_group(gn)
    sg = np.asarray(sg, dtype=np.int64)
    ok = (g > 0) & (sg >= 0) & (sg < min(NO_HALO, SUBHALO_STRIDE))
    return np.where(ok, g * SUBHALO_STRIDE + sg, 0)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


def _scaling(ds) -> Dict[str, float]:
    out = {}
    for key, names in (('a_scaling', _A_KEYS), ('h_scaling', _H_KEYS), ('to_cgs', _CGS_KEYS)):
        for n in names:
            if n in ds.attrs:
                out[key] = float(_scalar(ds.attrs[n]))
                break
    return out


def _masstable_scaling(f, g) -> Dict[str, float]:
    """
    Unit attributes for masses that come from Header/MassTable, which has none of its own.

    Gadget-4 writes a Masses dataset with a_scaling 0 and h_scaling -1 in a comoving run, both 0
    otherwise (src/io/snap_io.cc init_field "Masses"; src/io/io.cc write_dataset_attributes), and
    in MassTable the same masses. A file written that way is recognised by its Coordinates, which
    carry Gadget-4's length_scaling next to a_scaling and h_scaling; their h_scaling is -1 or 0
    in the same cases as the masses'. Any other file gets no attributes: they are never guessed.
    """
    c = g.get('Coordinates')
    if c is None or not all(k in c.attrs for k in ('a_scaling', 'h_scaling', 'length_scaling')):
        return {}
    out = {'a_scaling': 0.0, 'h_scaling': float(_scalar(c.attrs['h_scaling']))}
    for grp in ('Parameters', 'Header'):
        unit = f[grp].attrs.get('UnitMass_in_g') if grp in f else None
        if unit is not None:
            out['to_cgs'] = float(_scalar(unit))
            break
    return out


@dataclass
class PropertyInfo:
    """One plottable particle property."""
    name: str                          # key shown to the user, e.g. 'Density', 'Velocities |v|'
    dataset: str                       # HDF5 path inside PartTypeN ('' for derived)
    component: Optional[int] = None    # column of a vector dataset; -1 = magnitude
    types: Tuple[int, ...] = ()        # particle types that have it
    integer: bool = False              # integer data (group indices, flags): random colours make sense
    derived: str = ''                  # '', 'masstable', 'temperature', 'snap_group', 'snap_subhalo'
    scaling: Dict[str, float] = field(default_factory=dict)

    @property
    def unit_label(self) -> str:
        s = self.scaling
        if self.derived == 'temperature':
            return 'K'
        if self.derived in ('snap_group', 'snap_subhalo'):
            return 'index (0 = none)'
        if not s:
            return 'as stored (no unit attributes)'
        parts = ['code units']
        a, h = s.get('a_scaling', 0.0), s.get('h_scaling', 0.0)
        if a or h:
            fac = []
            if a:
                fac.append(f"a^{a:g}")
            if h:
                fac.append(f"h^{h:g}")
            parts.append("× " + " ".join(fac) + " for physical")
        return ", ".join(parts)


def _vector_names(ncol: int) -> List[str]:
    return ['x', 'y', 'z'][:ncol] if ncol <= 3 else [str(i) for i in range(ncol)]


def list_properties(files: Sequence[str]) -> Dict[str, PropertyInfo]:
    """Every per-particle property in the snapshot (first file decides), keyed by name."""
    props: Dict[str, PropertyInfo] = {}
    files = [str(f) for f in files]

    def add(name, **kw):
        t = kw.pop('ptype')
        if name in props:
            props[name].types = tuple(sorted(set(props[name].types) | {t}))
            if not props[name].scaling and kw.get('scaling'):
                props[name].scaling = kw['scaling']
        else:
            props[name] = PropertyInfo(name=name, types=(t,), **kw)

    with h5py.File(files[0], 'r') as f:
        header = dict(f['Header'].attrs) if 'Header' in f else {}
        table = np.atleast_1d(header.get('MassTable', np.zeros(6)))
        for t in range(6):
            g = f.get(f'PartType{t}')
            if g is None or 'Coordinates' not in g:
                continue
            n = g['Coordinates'].shape[0]
            items = []
            for k, obj in g.items():
                if isinstance(obj, h5py.Group):          # EAGLE ElementAbundance etc.
                    items += [(f"{k}/{kk}", o2) for kk, o2 in obj.items() if isinstance(o2, h5py.Dataset)]
                else:
                    items.append((k, obj))
            for path, ds in items:
                if path in _SKIP or path in MASS_NAMES or ds.shape[:1] != (n,) or ds.ndim > 2:
                    continue
                integer = ds.dtype.kind in 'iub'
                sc = _scaling(ds)
                if ds.ndim == 1:
                    add(path, dataset=path, integer=integer, scaling=sc, ptype=t)
                elif ds.shape[1] <= 9:
                    for c, lab in enumerate(_vector_names(ds.shape[1])):
                        add(f"{path} {lab}", dataset=path, component=c, integer=integer, scaling=sc, ptype=t)
                    if ds.shape[1] == 3 and path != 'Coordinates':
                        add(f"{path} |v|", dataset=path, component=-1, scaling=sc, ptype=t)
            # Masses: the dataset if present, else Header/MassTable (Gadget omits equal masses).
            md = next((g[k] for k in MASS_NAMES if k in g), None)
            if md is not None or (t < len(table) and float(table[t]) > 0):
                add('Masses', dataset='', derived='masstable', ptype=t,
                    scaling=_scaling(md) if md is not None else _masstable_scaling(f, g))
            if 'InternalEnergy' in g and 'Temperature' not in g:
                add('Temperature', dataset='InternalEnergy', derived='temperature', ptype=t)
            if 'GroupNumber' in g and g['GroupNumber'].shape[:1] == (n,):
                add(SNAP_GROUP, dataset='GroupNumber', derived='snap_group', integer=True, ptype=t)
                if 'SubGroupNumber' in g:
                    add(SNAP_SUBHALO, dataset='SubGroupNumber', derived='snap_subhalo', integer=True, ptype=t)
    return props


# ----------------------------------------------------------------------------- sample rows

def sample_rows(snap: dict) -> List[Tuple[str, int, np.ndarray]]:
    """
    (file, ptype, row indices) for the rows of the display sample, in sample order.

    Uses snap['rows'] if the loader provides it ({(piece, ptype): rows or (start, step)}),
    as the HDF5 loader and the LOD store always do. Otherwise it assumes a strided
    sample: every step-th particle in the order type 0..5, then piece 0..N-1. A
    snapshot dict sampled any other way fails the ParticleID check in
    read_sample_values and is matched by ID instead.
    """
    files = list(snap.get('files') or [snap['filepath']])
    if 'rows' in snap:
        out = []
        for (piece, t), r in sorted(snap['rows'].items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if isinstance(r, tuple):
                with h5py.File(files[piece], 'r') as f:
                    n = f[f'PartType{t}/Coordinates'].shape[0]
                r = np.arange(r[0], n, r[1])
            out.append((files[piece], t, np.asarray(r, dtype=np.int64)))
        return out
    step = int(snap.get('step', 1) or 1)
    counts = []
    for fn in files:
        with h5py.File(fn, 'r') as f:
            c = np.zeros(6, dtype=np.int64)
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    c[t] = g['Coordinates'].shape[0]
            counts.append(c)
    out, g0 = [], 0
    for t in range(6):
        for i, fn in enumerate(files):
            n = int(counts[i][t])
            if n:
                start = (-g0) % step
                if start < n:
                    out.append((fn, t, np.arange(start, n, step, dtype=np.int64)))
                g0 += n
    return out


def _read_rows(ds, rows: np.ndarray):
    """Rows of a dataset (in the given order); strided slice when evenly spaced."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) > 1 and np.any(np.diff(rows) <= 0):
        uniq, inv = np.unique(rows, return_inverse=True)
        return _read_rows(ds, uniq)[inv]
    if len(rows) == 0:
        return ds[0:0]
    if len(rows) > 1:
        d = np.diff(rows)
        if np.all(d == d[0]):
            return ds[int(rows[0]):int(rows[-1]) + 1:int(d[0])]
    elif len(rows) == 1:
        return ds[int(rows[0]):int(rows[0]) + 1]
    if len(rows) <= 512:
        return ds[np.asarray(rows)]
    # Irregular rows (a random sample): read contiguous slabs and index in memory;
    # h5py point selection is far slower than reading the slab.
    parts = []
    for s in range(int(rows[0]), int(rows[-1]) + 1, CHUNK_ROWS):
        e = min(s + CHUNK_ROWS, int(rows[-1]) + 1)
        r = rows[(rows >= s) & (rows < e)]
        if len(r):
            parts.append(ds[s:e][r - s])
    return np.concatenate(parts)


def _units(f) -> Dict[str, float]:
    out = {}
    for name in ('Parameters', 'Units', 'Header'):
        if name in f:
            for k in ('UnitVelocity_in_cm_per_s', 'UnitMass_in_g', 'UnitLength_in_cm', 'HubbleParam'):
                if k in f[name].attrs and k not in out:
                    out[k] = float(_scalar(f[name].attrs[k]))
    return out


def temperature_from_u(u: np.ndarray, xe: Optional[np.ndarray], u_to_cgs: float) -> np.ndarray:
    """
    T = (gamma - 1) u mu m_p / k_B with mu = 4 / (1 + 3X + 4X x_e).

    Without ElectronAbundance the gas is taken as fully ionised (x_e = 1 + 2y,
    mu ~ 0.59), the usual choice for hot gas; this is stated in the UI tooltip.
    """
    X = HYDROGEN_MASS_FRACTION
    if xe is None:
        mu = 4.0 / (8.0 - 5.0 * (1.0 - X))           # fully ionised H and He
    else:
        mu = 4.0 / (1.0 + 3.0 * X + 4.0 * X * np.asarray(xe, dtype=np.float64))
    return (GAMMA - 1.0) * np.asarray(u, dtype=np.float64) * u_to_cgs * mu * PROTON_MASS / BOLTZMANN


def _values_from_group(f, g, t: int, info: PropertyInfo, rows: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Values of ``info`` for ``rows`` (None = all) of type ``t``; None if the type lacks it."""
    def get(path):
        ds = g[path]
        return ds[:] if rows is None else _read_rows(ds, rows)

    if info.derived == 'masstable':
        if 'Masses' in g or 'Mass' in g:
            return get('Masses' if 'Masses' in g else 'Mass').astype(np.float64)
        table = np.atleast_1d(f['Header'].attrs.get('MassTable', np.zeros(6)))
        n = g['Coordinates'].shape[0] if rows is None else len(rows)
        return np.full(n, float(table[t]) if t < len(table) else np.nan)
    if info.derived == 'temperature':
        if 'InternalEnergy' not in g:
            return None
        sc = _scaling(g['InternalEnergy'])
        u_cgs = sc.get('to_cgs') or _units(f).get('UnitVelocity_in_cm_per_s', 1e5) ** 2
        xe = get('ElectronAbundance') if 'ElectronAbundance' in g else None
        return temperature_from_u(get('InternalEnergy'), xe, u_cgs)
    if info.derived in ('snap_group', 'snap_subhalo'):
        if 'GroupNumber' not in g or (info.derived == 'snap_subhalo' and 'SubGroupNumber' not in g):
            return None
        if info.derived == 'snap_group':
            return snapshot_group(get('GroupNumber'))
        return snapshot_subhalo(get('GroupNumber'), get('SubGroupNumber'))
    if info.dataset not in g:
        return None
    v = get(info.dataset)
    if info.component is None:
        return v.astype(np.float64)
    if info.component == -1:
        return np.sqrt(np.einsum('ij,ij->i', v.astype(np.float64), v.astype(np.float64)))
    return v[:, info.component].astype(np.float64)


def read_sample_values(snap: dict, info: PropertyInfo, progress=None, cancelled=None) -> np.ndarray:
    """
    ``info`` for every particle in the display sample, aligned with snap['c'].
    Particles whose type has no such dataset get NaN. Cached on the snapshot.

    Values come from the read-once cache (snapshot_io.property_cache), indexed by
    the set's file rows. The rows are checked first against the ParticleIDs held
    in the set (first, middle and last row of every file and type read); if they
    disagree, values are matched by ID instead, so a property can never be
    attached to the wrong particle.
    """
    from snapshot_io import property_cache
    cache = snap.setdefault('_props', {})
    if info.name in cache:
        return cache[info.name]
    out = None
    try:
        plan = sample_rows(snap)
        if sum(len(r) for _, _, r in plan) == len(snap['c']) and _rows_match_ids(snap, plan, info):
            out = property_cache.gather(snap, info, plan, progress, cancelled)
    except InterruptedError:
        raise
    except (ValueError, KeyError, OSError):
        out = None
    if out is None:
        out = _read_by_id(snap, info)
    cache[info.name] = out
    return out


def _rows_match_ids(snap: dict, plan, info: PropertyInfo) -> bool:
    """Spot check: the file's ParticleIDs at a few of the set's rows equal the set's IDs."""
    if 'id' not in snap:
        return True
    sid = np.asarray(snap['id'], dtype=np.uint64)
    pos = 0
    handles: Dict[str, h5py.File] = {}
    try:
        for fn, t, rows in plan:
            if t in info.types and len(rows):
                k = np.unique([0, len(rows) // 2, len(rows) - 1])
                f = handles.get(fn) or handles.setdefault(fn, h5py.File(fn, 'r'))
                g = f[f'PartType{t}']
                if 'ParticleIDs' in g:
                    got = _read_rows(g['ParticleIDs'], rows[k]).astype(np.uint64)
                    if not np.array_equal(got, sid[pos + k]):
                        return False
            pos += len(rows)
    finally:
        for f in handles.values():
            f.close()
    return True


def _read_by_id(snap: dict, info: PropertyInfo) -> np.ndarray:
    """Slow path: match the sample's (type, ID) pairs against the file, chunk by chunk."""
    files = list(snap.get('files') or [snap['filepath']])
    out = np.full(len(snap['c']), np.nan)
    sid = np.asarray(snap['id'], dtype=np.uint64)
    st = np.asarray(snap['t'])
    for t in np.unique(st):
        sel = np.nonzero(st == t)[0]
        order = np.argsort(sid[sel])
        keys = sid[sel][order]
        for fn in files:
            with h5py.File(fn, 'r') as f:
                g = f.get(f'PartType{int(t)}')
                if g is None or 'ParticleIDs' not in g:
                    continue
                n = g['ParticleIDs'].shape[0]
                for s in range(0, n, CHUNK_ROWS):
                    e = min(s + CHUNK_ROWS, n)
                    pid = g['ParticleIDs'][s:e].astype(np.uint64)
                    slot = np.searchsorted(keys, pid)
                    slot[slot == len(keys)] = 0
                    hit = keys[slot] == pid
                    if not hit.any():
                        continue
                    rows = np.nonzero(hit)[0] + s
                    vals = _values_from_group(f, g, int(t), info, rows)
                    if vals is not None:
                        out[sel[order[slot[hit]]]] = vals
    return out


def select_by_range(files: Sequence[str], info: PropertyInfo, lo: float, hi: float,
                    ptypes: Sequence[int] = range(6), chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
    """IDs of every particle of ``ptypes`` in the full snapshot with lo <= value <= hi."""
    from snapshot_io import property_cache
    files = [str(f) for f in files]
    out: List[np.ndarray] = []
    for t in sorted(set(ptypes) & set(info.types)):
        arr = property_cache.values_for_type(files, info, t)
        if arr is None:
            continue
        rows = []
        for s in range(0, len(arr), chunk_rows):
            v = np.asarray(arr[s:s + chunk_rows])
            rows.append(np.flatnonzero((v >= lo) & (v <= hi)) + s)
        rows = np.concatenate(rows) if rows else np.empty(0, np.int64)
        if len(rows):
            out.append(property_cache.ids_at(files, t, rows))
    return np.concatenate(out) if out else np.empty(0, dtype=np.uint64)


def read_full_values(files: Sequence[str], info: PropertyInfo, ptype: int) -> Optional[np.ndarray]:
    """All values of one type, concatenated over the files (snapshot order)."""
    from snapshot_io import property_cache
    arr = property_cache.values_for_type([str(f) for f in files], info, ptype)
    return None if arr is None else np.asarray(arr, dtype=np.float64)


def read_values_for_ids(files: Sequence[str], info: PropertyInfo, ids, types: Optional[np.ndarray] = None,
                        chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
    """
    ``info`` for the particles with ParticleIDs ``ids`` (full resolution), aligned
    with ``np.unique(ids)``; NaN where not found or the type lacks the property.
    If an ID occurs more than once, the first occurrence is used (as id_lookup).
    """
    keys = np.unique(np.asarray(ids, dtype=np.uint64))
    out = np.full(len(keys), np.nan)
    done = np.zeros(len(keys), dtype=bool)
    if len(keys) == 0:
        return out
    want_types = info.types if types is None else sorted(set(int(t) for t in np.unique(types) if t >= 0)
                                                         & set(info.types))
    for fn in files:
        with h5py.File(str(fn), 'r') as f:
            for t in want_types:
                g = f.get(f'PartType{t}')
                if g is None or 'ParticleIDs' not in g:
                    continue
                n = g['ParticleIDs'].shape[0]
                for s in range(0, n, chunk_rows):
                    e = min(s + chunk_rows, n)
                    pid = g['ParticleIDs'][s:e].astype(np.uint64)
                    slot = np.searchsorted(keys, pid)
                    slot[slot == len(keys)] = 0
                    hit = (keys[slot] == pid)
                    hit[hit] &= ~done[slot[hit]]
                    if not hit.any():
                        continue
                    rows = np.nonzero(hit)[0]
                    sl = slot[rows]
                    _, first = np.unique(sl, return_index=True)
                    first.sort()                         # keep rows increasing for the read
                    rows, sl = rows[first], sl[first]
                    vals = _values_from_group(f, g, t, info, rows + s)
                    if vals is not None:
                        out[sl] = vals
                    done[sl] = True
    return out
