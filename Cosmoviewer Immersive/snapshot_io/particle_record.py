"""
Everything a snapshot stores about one particle, and how it changed between snapshots.

This is the data behind the particle inspector. For a ParticleID it finds the
particle in a snapshot at full resolution (file, type and row, whatever the
display sample holds) and reads one row of every per-particle property the
snapshot lists (snapshot_io.properties.list_properties): stored datasets,
vector components and magnitudes, Masses (dataset or MassTable), the derived
Temperature and the snapshot's own halo membership. Gadget-4 catalogue
membership is added by the caller from the row (group_catalogue.membership).

The change between two records of the same particle is the "delta":

- ``value - value_before`` for every property both records have, the
  relative change, and the rate per unit time;
- positions differ by the minimum image in a periodic box, so a particle
  that crosses the box edge does not jump by a box length;
- a change of particle type (gas that became a star) is reported, and
  properties only one of the records has are listed, not differenced.

Values are as stored (code units; comoving and per h where the run is
cosmological). ``physical`` multiplies by a^a_scaling h^h_scaling from the
dataset's own attributes, so a delta in physical units includes the cosmic
expansion between the snapshots; the comoving delta does not. Time is the
code time for non-comoving runs, and for comoving runs the cosmic time from
the file's own Omega0, OmegaLambda and h (flat or curved, no radiation).

Pure numpy/h5py: no Qt, usable from scripts and tests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from snapshot_io.properties import PropertyInfo, _values_from_group, list_properties

CHUNK_ROWS = 1 << 22
SEC_PER_GYR = 3.15576e16
MPC_IN_KM = 3.0856775814913673e19
TYPE_NAMES = {0: 'gas', 1: 'DM', 2: 'disk', 3: 'bulge', 4: 'star', 5: 'BH'}


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


# ----------------------------------------------------------------------------- locating

def locate_rows(files: Sequence[str], ids, chunk_rows: int = CHUNK_ROWS
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Where each ID is stored: (ids sorted unique, file index, type, row in that
    file, global row within the type across files). -1 where not found. The
    first occurrence wins for repeated IDs, as in id_lookup.locate_ids; the
    scan stops once every ID is found, so later duplicates are not looked for.
    ``global_row`` is only valid for files read up to that point, which always
    includes the file that holds the particle.
    """
    keys = np.unique(np.asarray(ids, dtype=np.uint64))
    n = len(keys)
    fidx = np.full(n, -1, np.int64)
    ptype = np.full(n, -1, np.int64)
    row = np.full(n, -1, np.int64)
    grow = np.full(n, -1, np.int64)
    running = np.zeros(6, np.int64)
    for i, fn in enumerate(files):
        if n and (fidx >= 0).all():
            break                       # everything found: the remaining files need not be scanned
        with h5py.File(str(fn), 'r') as f:
            for t in range(6):
                if n and (fidx >= 0).all():
                    break
                g = f.get(f'PartType{t}')
                if g is None or 'ParticleIDs' not in g:
                    continue
                nrow = g['ParticleIDs'].shape[0]
                for s in range(0, nrow, chunk_rows) if n else ():
                    e = min(s + chunk_rows, nrow)
                    pid = g['ParticleIDs'][s:e].astype(np.uint64)
                    slot = np.searchsorted(keys, pid)
                    slot[slot == n] = 0
                    hit = keys[slot] == pid
                    hit[hit] &= fidx[slot[hit]] < 0
                    if not hit.any():
                        continue
                    r = np.nonzero(hit)[0]
                    sl = slot[r]
                    _, first = np.unique(sl, return_index=True)
                    r, sl = r[first], sl[first]
                    fidx[sl], ptype[sl], row[sl] = i, t, r + s
                    grow[sl] = r + s + running[t]
                running[t] += nrow
    return keys, fidx, ptype, row, grow


# ----------------------------------------------------------------------------- time

def _attrs(f, name) -> Dict:
    return {k: _scalar(v) for k, v in f[name].attrs.items()} if name in f else {}


def cosmic_time_gyr(a: float, omega_m: float, omega_l: float, h: float, n: int = 20000) -> float:
    """Age of the universe at scale factor ``a`` (Gyr), matter + Lambda + curvature, no radiation."""
    omega_k = 1.0 - omega_m - omega_l
    x = np.linspace(0.0, 1.0, n + 1)[1:] ** 2 * a          # dense near a = 0, where the integrand peaks
    E = np.sqrt(omega_m / x ** 3 + omega_k / x ** 2 + omega_l)
    integrand = 1.0 / (x * E)
    # t = int_0^a da' / (a' H(a')); the integrand ~ a'^(1/2) near 0, so start from x[0] analytically
    head = (2.0 / 3.0) * x[0] ** 1.5 / np.sqrt(omega_m) if omega_m > 0 else 0.0
    body = np.trapezoid(integrand, x) if hasattr(np, 'trapezoid') else np.trapz(integrand, x)
    hubble_time_gyr = MPC_IN_KM / (100.0 * h) / SEC_PER_GYR
    return float((head + body) * hubble_time_gyr)


@dataclass
class SnapshotClock:
    """When a snapshot is: code time, and for comoving runs a, z and cosmic time."""
    time: float
    comoving: bool
    a: float = 1.0
    h: float = 1.0
    box: float = 0.0
    time_gyr: Optional[float] = None          # comoving: cosmic time; else code time in Gyr if units known
    time_label: str = 't (code units)'

    @property
    def redshift(self) -> Optional[float]:
        return (1.0 / self.a - 1.0) if self.comoving and self.a > 0 else None


def snapshot_clock(path: str) -> SnapshotClock:
    with h5py.File(str(path), 'r') as f:
        hd, par = _attrs(f, 'Header'), _attrs(f, 'Parameters')
        units = {**_attrs(f, 'Units'), **par}
    flag = par.get('ComovingIntegrationOn', hd.get('ComovingIntegrationOn'))
    box = float(hd.get('BoxSize', par.get('BoxSize', 0.0)) or 0.0)
    comoving = bool(int(flag)) if flag is not None else box > 0
    time = float(hd.get('Time', 0.0))
    h = float(par.get('HubbleParam', hd.get('HubbleParam', 1.0)) or 1.0)
    if comoving:
        om = float(par.get('Omega0', hd.get('Omega0', 0.3)))
        ol = float(par.get('OmegaLambda', hd.get('OmegaLambda', 0.7)))
        return SnapshotClock(time, True, time, h, box, cosmic_time_gyr(time, om, ol, h), 'cosmic time (Gyr)')
    gyr = None
    L, V = units.get('UnitLength_in_cm'), units.get('UnitVelocity_in_cm_per_s')
    if L and V:
        gyr = time * float(L) / float(V) / SEC_PER_GYR
    return SnapshotClock(time, False, 1.0, h, box, gyr, 't (code units)')


# ----------------------------------------------------------------------------- records

# Gadget-4 writes no a/h scaling attributes for these because there is nothing to scale: mass fractions,
# an abundance ratio, and the formation time (code time, or the scale factor in a comoving run).
DIMENSIONLESS = ('Metallicity', 'ElectronAbundance', 'StellarFormationTime')


@dataclass
class ParticleRecord:
    pid: int
    found: bool
    ptype: int = -1
    file: str = ''
    row: int = -1                       # row in ``file``
    global_row: int = -1                # row within the type across the snapshot's files
    clock: Optional[SnapshotClock] = None
    values: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, PropertyInfo] = field(default_factory=dict)

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.ptype, f'type {self.ptype}') if self.found else 'not found'

    def physical(self, name: str) -> Optional[float]:
        """Value in physical code units, or None if the dataset has no a/h attributes."""
        v, inf = self.values.get(name), self.info.get(name)
        if v is None or inf is None or self.clock is None:
            return None
        s = inf.scaling
        if inf.derived in ('temperature', 'snap_group', 'snap_subhalo'):
            return v
        if 'a_scaling' not in s or 'h_scaling' not in s:
            return v if not s and inf.dataset.split('/')[-1] in DIMENSIONLESS else None
        a = self.clock.a if self.clock.comoving else 1.0
        return float(v * a ** s['a_scaling'] * self.clock.h ** s['h_scaling'])


def read_records(files: Sequence[str], ids, props: Optional[Dict[str, PropertyInfo]] = None
                 ) -> Dict[int, ParticleRecord]:
    """One record per ID: every property of the snapshot made of ``files``, read from one row each."""
    files = [str(f) for f in files]
    props = props if props is not None else list_properties(files)
    clock = snapshot_clock(files[0])
    keys, fidx, ptype, row, grow = locate_rows(files, ids)
    out: Dict[int, ParticleRecord] = {}
    for k, i, t, r, gr in zip(keys, fidx, ptype, row, grow):
        out[int(k)] = ParticleRecord(int(k), i >= 0, int(t), files[i] if i >= 0 else '', int(r), int(gr), clock)
    found = np.flatnonzero(fidx >= 0)
    for i in np.unique(fidx[found]):
        with h5py.File(files[i], 'r') as f:
            for t in np.unique(ptype[found][fidx[found] == i]):
                sel = found[(fidx[found] == i) & (ptype[found] == t)]
                order = sel[np.argsort(row[sel])]
                rows = row[order]
                g = f[f'PartType{int(t)}']
                for name, inf in props.items():
                    if int(t) not in inf.types:
                        continue
                    vals = _values_from_group(f, g, int(t), inf, rows)
                    if vals is None:
                        continue
                    for k, v in zip(keys[order], np.asarray(vals, dtype=np.float64)):
                        rec = out[int(k)]
                        rec.values[name] = float(v)
                        rec.info[name] = inf
    return out


# ----------------------------------------------------------------------------- deltas

@dataclass
class Change:
    name: str
    now: Optional[float]
    before: Optional[float]
    delta: Optional[float] = None
    relative: Optional[float] = None    # delta / |before|
    rate: Optional[float] = None        # delta per unit of ``clock`` time (code time or Gyr)


@dataclass
class RecordDelta:
    changes: List[Change]
    dt: Optional[float]                 # time between the records, in ``time_unit``
    time_unit: str
    type_change: Optional[Tuple[str, str]] = None
    displacement: Optional[float] = None  # |delta r|, minimum image, stored units
    only_now: List[str] = field(default_factory=list)
    only_before: List[str] = field(default_factory=list)


_COORD = ('Coordinates x', 'Coordinates y', 'Coordinates z')


def record_delta(now: ParticleRecord, before: ParticleRecord, physical: bool = False) -> RecordDelta:
    """How ``now`` differs from ``before`` (the same particle in an earlier or later snapshot)."""
    cn, cb = now.clock, before.clock
    if cn is not None and cb is not None:
        if cn.comoving and cn.time_gyr is not None and cb.time_gyr is not None:
            dt, unit = cn.time_gyr - cb.time_gyr, 'Gyr'
        else:
            dt, unit = cn.time - cb.time, 'code time'
    else:
        dt, unit = None, ''
    box = 0.0 if cn is None else cn.box
    if physical and cn is not None and cn.comoving:
        box = 0.0                       # physical coordinates of two epochs are not on one periodic grid
    get_n = now.physical if physical else now.values.get
    get_b = before.physical if physical else before.values.get
    changes = []
    names = [k for k in now.values if k in before.values]
    for name in names:
        a, b = get_n(name), get_b(name)
        c = Change(name, a, b)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            d = a - b
            if name in _COORD and box > 0:
                d -= box * np.round(d / box)
            c.delta = d
            c.relative = d / abs(b) if b != 0 else None
            c.rate = d / dt if dt else None
        changes.append(c)
    disp = None
    if all(n in now.values and n in before.values for n in _COORD):
        d = np.array([get_n(n) - get_b(n) if get_n(n) is not None and get_b(n) is not None else np.nan
                      for n in _COORD])
        if box > 0:
            d -= box * np.round(d / box)
        disp = float(np.sqrt(np.sum(d * d)))
    tchange = (before.type_name, now.type_name) if now.found and before.found and now.ptype != before.ptype else None
    return RecordDelta(changes, dt, unit, tchange, disp,
                       sorted(set(now.values) - set(before.values)),
                       sorted(set(before.values) - set(now.values)))


def history(records: Sequence[ParticleRecord], name: str, physical: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(time, value) of one property over records ordered in time; NaN where it is missing."""
    t, v = [], []
    for r in records:
        c = r.clock
        t.append(np.nan if c is None else (c.time_gyr if c.comoving and c.time_gyr is not None else c.time))
        x = (r.physical(name) if physical else r.values.get(name)) if r.found else None
        v.append(np.nan if x is None else x)
    return np.asarray(t, float), np.asarray(v, float)
