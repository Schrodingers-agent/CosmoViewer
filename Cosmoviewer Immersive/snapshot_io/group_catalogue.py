"""
Halo finder output attached to particles (GadgetViewer: File > Read groups).

GadgetViewer reads the halo catalogue that belongs to the open snapshot and
adds particle properties saying which group each particle is in (FoFGroupIndex,
SubGroupIndex: 1 = first group in the catalogue, 0 = in no group), then reads
the matching catalogue again whenever the snapshot changes. It does this for
Gadget-4 by offsets (read_gadget4_groups.F90): Gadget-4 writes the snapshot in
group order, so group i of type t is rows GroupOffsetType[i, t] ..
+ GroupLenType[i, t] of PartType t, counted across the snapshot's files.

This module does the same for Gadget-4 ``fof_tab_NNN`` and
``fof_subhalo_tab_NNN`` (single or multi-file), for any set of rows (the
display sample, or every particle). Offsets are int64 throughout.

Not implemented, for want of test data: old binary SubFind group_tab/sub_tab
and VELOCIraptor.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

_NUM_RE = re.compile(r'_(\d{3,})(?!\d)')


@dataclass
class GroupCatalogue:
    files: List[str]
    group_offset: np.ndarray                  # (Ngroups, 6) int64
    group_len: np.ndarray                     # (Ngroups, 6) int64
    sub_offset: Optional[np.ndarray] = None   # (Nsubhalos, 6) int64
    sub_len: Optional[np.ndarray] = None
    header: Dict = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.group_len)

    @property
    def n_subhalos(self) -> int:
        return 0 if self.sub_len is None else len(self.sub_len)


def catalogue_files(path: str) -> List[str]:
    """All files of the catalogue ``path`` (``x.N.hdf5`` pieces are gathered and ordered)."""
    p = Path(path)
    m = re.match(r'^(?P<base>.+)\.(?P<i>\d+)\.(?P<ext>hdf5|h5)$', p.name)
    if not m:
        return [str(p)]
    with h5py.File(p, 'r') as f:
        n = int(np.atleast_1d(f['Header'].attrs.get('NumFiles', 1))[0])
    files = [p.parent / f"{m.group('base')}.{i}.{m.group('ext')}" for i in range(n)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"catalogue is split over {n} files; missing {missing[0]}")
    return [str(f) for f in files]


def _six(a: np.ndarray) -> np.ndarray:
    """Pad per-type columns to six: Gadget-4 built with NTYPES < 6 writes fewer columns."""
    a = np.atleast_2d(np.asarray(a, dtype=np.int64))
    if a.shape[1] >= 6:
        return a[:, :6]
    return np.hstack([a, np.zeros((a.shape[0], 6 - a.shape[1]), dtype=np.int64)])


def read_gadget4_catalogue(path: str) -> GroupCatalogue:
    """Read GroupOffsetType/GroupLenType (and the Subhalo equivalents if present)."""
    files = catalogue_files(path)
    go, gl, so, sl = [], [], [], []
    header = {}
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if not header and 'Header' in f:
                header = {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v)
                          for k, v in f['Header'].attrs.items()}
            if 'Group' in f and 'GroupLenType' in f['Group']:
                gl.append(_six(f['Group/GroupLenType'][:]))
                if 'GroupOffsetType' not in f['Group']:
                    raise ValueError(f"{Path(fn).name}: Group/GroupOffsetType missing; "
                                     "cannot map groups onto particles")
                go.append(_six(f['Group/GroupOffsetType'][:]))
            if 'Subhalo' in f and 'SubhaloLenType' in f['Subhalo'] and 'SubhaloOffsetType' in f['Subhalo']:
                sl.append(_six(f['Subhalo/SubhaloLenType'][:]))
                so.append(_six(f['Subhalo/SubhaloOffsetType'][:]))
    if not gl:
        if 'Ngroups_Total' in header:          # a valid catalogue with no groups
            z = np.zeros((0, 6), dtype=np.int64)
            return GroupCatalogue(files, z, z, None, None, header)
        raise ValueError(f"{Path(path).name} is not a Gadget-4 group catalogue (no Group/GroupLenType)")
    cat = GroupCatalogue(files, np.vstack(go), np.vstack(gl),
                         np.vstack(so) if so else None, np.vstack(sl) if sl else None, header)
    for name, off, ln in (('Group', cat.group_offset, cat.group_len),
                          ('Subhalo', cat.sub_offset, cat.sub_len)):
        if off is None or len(off) == 0:
            continue
        # Membership by offsets needs non-overlapping, ascending ranges per type.
        for t in range(6):
            o, n = off[:, t], ln[:, t]
            nz = n > 0
            if np.any(np.diff(o[nz]) < 0) or np.any(o[nz][1:] < (o[nz] + n[nz])[:-1]):
                raise ValueError(f"{name} offsets of type {t} are not ascending and disjoint")
    return cat


def read_catalogue_fields(path: str, subhalo: bool = False,
                          names: Optional[Sequence[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    """
    Per-group (or, with ``subhalo``, per-subhalo) datasets of a Gadget-4 catalogue,
    joined over its files in order and kept as stored, with the catalogue's Header
    and Parameters attributes. ``names`` limits which datasets are read (all of the
    Group or Subhalo group by default). Read-only; used by the halo table.

    Gadget-4 writes these datasets without unit attributes, so their units follow
    from Parameters (UnitLength_in_cm, UnitMass_in_g) and the run's conventions
    (comoving and per h in cosmological runs); the caller labels them.
    """
    files = catalogue_files(path)
    part = 'Subhalo' if subhalo else 'Group'
    pieces: Dict[str, List[np.ndarray]] = {}
    header, params = {}, {}
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if not header and 'Header' in f:
                header = {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v)
                          for k, v in f['Header'].attrs.items()}
            if not params and 'Parameters' in f:
                params = {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v)
                          for k, v in f['Parameters'].attrs.items()}
            g = f.get(part)
            if g is None:
                continue
            for k, ds in g.items():
                if isinstance(ds, h5py.Dataset) and (names is None or k in names) and ds.shape:
                    pieces.setdefault(k, []).append(ds[...])
    return {k: np.concatenate(v) for k, v in pieces.items()}, header, params


def membership(offset: np.ndarray, length: np.ndarray, ptype: int, rows: np.ndarray) -> np.ndarray:
    """
    1-based index of the group containing each row of type ``ptype`` (0 = none).
    ``rows`` are global row numbers within the snapshot's PartType ``ptype``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(len(rows), dtype=np.int64)
    if len(offset) == 0:
        return out
    o, n = offset[:, ptype], length[:, ptype]
    keep = np.nonzero(n > 0)[0]
    if len(keep) == 0:
        return out
    o, n = o[keep], n[keep]
    k = np.searchsorted(o, rows, side='right') - 1
    ok = (k >= 0)
    ok[ok] &= rows[ok] < (o[k[ok]] + n[k[ok]])
    out[ok] = keep[k[ok]] + 1
    return out


def global_rows(snap_files: Sequence[str], plan: List[Tuple[str, int, np.ndarray]]) -> List[Tuple[int, np.ndarray]]:
    """Turn per-file rows from properties.sample_rows into global rows per type."""
    base: Dict[Tuple[str, int], int] = {}
    running = np.zeros(6, dtype=np.int64)
    for fn in snap_files:
        with h5py.File(fn, 'r') as f:
            for t in range(6):
                base[(str(fn), t)] = int(running[t])
                g = f.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    running[t] += g['Coordinates'].shape[0]
    return [(t, rows + base[(str(fn), t)]) for fn, t, rows in plan]


def sample_membership(snap: dict, cat: GroupCatalogue) -> Dict[str, np.ndarray]:
    """{'FoFGroupIndex': ..., 'SubGroupIndex': ...} aligned with the display sample."""
    from snapshot_io.properties import sample_rows
    files = list(snap.get('files') or [snap['filepath']])
    plan = sample_rows(snap)
    n = len(snap['c'])
    out = {'FoFGroupIndex': np.zeros(n, dtype=np.int64)}
    if cat.sub_offset is not None:
        out['SubGroupIndex'] = np.zeros(n, dtype=np.int64)
    pos = 0
    for t, rows in global_rows(files, plan):
        out['FoFGroupIndex'][pos:pos + len(rows)] = membership(cat.group_offset, cat.group_len, t, rows)
        if cat.sub_offset is not None:
            out['SubGroupIndex'][pos:pos + len(rows)] = membership(cat.sub_offset, cat.sub_len, t, rows)
        pos += len(rows)
    if pos != n:
        raise ValueError(f"sample rows ({pos}) do not match the display sample ({n})")
    return out


def group_ids_full(snap_files: Sequence[str], cat: GroupCatalogue, group: int,
                   subhalo: bool = False) -> np.ndarray:
    """ParticleIDs of every member of 1-based group (or subhalo) ``group``, all types."""
    off, ln = (cat.sub_offset, cat.sub_len) if subhalo else (cat.group_offset, cat.group_len)
    if off is None or not (1 <= group <= len(off)):
        raise ValueError(f"{'subhalo' if subhalo else 'group'} {group} is not in the catalogue")
    out = []
    running = np.zeros(6, dtype=np.int64)
    for fn in snap_files:
        with h5py.File(str(fn), 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is None or 'ParticleIDs' not in g:
                    continue
                nrow = g['ParticleIDs'].shape[0]
                lo = int(off[group - 1, t]) - int(running[t])
                hi = lo + int(ln[group - 1, t])
                a, b = max(lo, 0), min(hi, nrow)
                if b > a:
                    out.append(g['ParticleIDs'][a:b].astype(np.uint64))
                running[t] += nrow
    return np.concatenate(out) if out else np.empty(0, dtype=np.uint64)


def renumber_path(path: str, num: str) -> Optional[str]:
    """
    ``path`` (a file picked for one snapshot) with that snapshot's number replaced by ``num``.

    The number is the last ``_NNN`` (at least three digits) in the file name, and
    only that number is replaced, wherever it occurs (``groups_003/fof_tab_003.hdf5``).
    Other numbers in the path stay: ``run_512/pot_003.hdf5`` must not become
    ``run_007/pot_007.hdf5``. None if the file name has no such number.
    """
    found = list(_NUM_RE.finditer(Path(path).name))
    if not found:
        return None
    old = found[-1].group(1)
    s = str(path)
    for mm in reversed(list(_NUM_RE.finditer(s))):
        if mm.group(1) == old:
            s = s[:mm.start(1)] + num.zfill(len(old)) + s[mm.end(1):]
    return s


def find_catalogue_for(snapshot_path: str, template: Optional[str] = None) -> Optional[str]:
    """
    The catalogue for a snapshot. With ``template`` (a catalogue the user picked for
    another snapshot) the snapshot number in it is replaced, as GadgetViewer does
    (numbers padded to at least three digits and preceded by an underscore).
    Otherwise Gadget-4's default names are tried next to the snapshot and in
    groups_NNN/.
    """
    sp = Path(snapshot_path)
    m = _NUM_RE.search(sp.name)
    if not m:
        return None
    num = m.group(1)
    cands: List[Path] = []
    if template:
        s = renumber_path(template, num)
        if s is not None:
            cands.append(Path(s))
    for d in (sp.parent, sp.parent / f"groups_{num}"):
        for stem in ('fof_subhalo_tab', 'fof_tab'):
            cands += [d / f"{stem}_{num}.hdf5", d / f"{stem}_{num}.0.hdf5"]
    for c in cands:
        if c.exists():
            return str(c)
    return None
