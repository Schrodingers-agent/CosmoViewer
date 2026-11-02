"""
Structures: halo-finder groups as a table, linked to the 3D view and to analysis.

Mixed into CosmoViewerController. Contract: selecting a group always gives its complete
membership as a population of kind "group" (ParticleSelectionSet.set_population
with a definition, defined_at, complete=True and n_expected = the group length).

What it gives a researcher, on the way from a structure seen in 3D to a
defensible measurement of it:

- Structures > Halo table lists the groups of the snapshot on screen: from the
  Gadget-4 catalogue attached to it (fof_tab / fof_subhalo_tab; groups, and
  subhaloes when there are any), or from FoF run in this window on that
  snapshot. The table follows the snapshot on screen and never shows one
  snapshot's groups as another's: on a snapshot without FoF it says which
  snapshots have FoF groups, as the FoF window does.
- Selecting a row makes the active set that group's complete membership, read
  at full resolution (catalogue groups by Gadget-4 offsets, FoF groups from the
  full-resolution FoF result), and centres the 3D view on it. When the snapshot
  file lacks some of a catalogue group's members (a cut snapshot), the set says
  it is incomplete, and how many members it has of how many.
- Check against catalogue measures the selected group from its members (members
  found, summed member mass, periodic centre of mass) and compares with the
  catalogue's GroupLen, GroupMass and GroupPos. Measure all groups does the same
  for every group in one pass over the snapshot (and, for FoF groups, fills the
  mass and centre columns, which the FoF result itself does not hold).
- The particle inspector offers "Select its catalogue group" and "Select its FoF
  group" through the same code (select_group_population).

Units. Every number is as stored in the files, in the run's code units; the
labels come from the run's Parameters (UnitLength_in_cm, UnitMass_in_g) and the
snapshot's dataset attributes (a_scaling, h_scaling), with the Gadget convention
(comoving, per h) for a cosmological run whose datasets carry none. Gadget-4
writes catalogue datasets without unit attributes, so the snapshot's are used.

Measurement. Gadget-4 (fof.cc) takes a group's centre of mass relative to one of
its particles, each particle at its nearest periodic image, in double precision,
and writes it as float32. Here the same is done in float64 from the float32
positions stored in the snapshot, so a measured centre can differ from GroupPos
by the rounding of the stored positions and of GroupPos itself: about one float32
ulp at the box size (3.8e-6 Mpc/h in a 50 Mpc/h box). Agreement is judged within
4 such ulp, the tolerance of the Explorer's own check of GroupPos
(Dataset Explorer/measurements/README.md), and masses within 2 ulp of GroupMass.
GroupVel is not shown. It is not in the stored velocity unit: on DM-L50 at
a = 0.25 and 0.5 it equals the mean stored Velocities x a^1.5, that is a times
the mean peculiar velocity (Gadget's internal velocity), to float32 precision;
the catalogue itself says nothing about it, and a third velocity convention in
the window would invite mistakes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDockWidget

from controller.console import log
from controller.thread_hold import hold
from snapshot_io.group_catalogue import group_ids_full, membership, read_catalogue_fields

CHUNK_ROWS = 1 << 22                 # rows read at a time in a pass over a snapshot
SYNC_LOOKUP_MAX = 4_000_000          # up to this many particles a selected group is looked up at once;
                                     # bigger snapshots are looked up by the tracking worker, off the GUI thread
FOLLOW_MS = 400                      # how often the table checks that it shows the snapshot on screen
POS_ULP_TOL = 4.0                    # centre of mass against GroupPos, in float32 ulp at the box size
MASS_ULP_TOL = 2.0                   # summed mass against GroupMass, in float32 ulp of GroupMass
MASS_NAMES = ('Masses', 'Mass')
_SUPERSCRIPT = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
_TYPE_NAMES = {0: "gas", 1: "DM", 2: "disk", 3: "bulge", 4: "stars", 5: "BH"}


class MeasureCancelled(Exception):
    """A pass over the snapshot was cancelled; nothing was changed."""


# ----------------------------------------------------------------------------- measuring groups

@dataclass
class GroupMeasure:
    """Per-group results of a pass over member particles; entry i is group (or subhalo) i + 1."""
    count: np.ndarray                    # members found
    mass: np.ndarray                     # summed member mass, code units as stored
    com: np.ndarray                      # (n, 3) periodic centre of mass, code units as stored; NaN if none found
    amax: np.ndarray                     # largest |coordinate| among the members (isolated runs only, else 0)


class GroupAccumulator:
    """
    Member count, mass and periodic centre of mass of many groups, from members
    given in any order and in pieces (one chunk of a file at a time), so memory
    stays bounded by the chunk. Each group's first member seen is its reference
    point; every member counts at its nearest periodic image of that point, as
    Gadget-4 does (fof.cc), which is right for any group smaller than half the box.
    """

    def __init__(self, n_groups: int, box: float):
        self.n, self.box = int(n_groups), float(box or 0.0)
        self.count = np.zeros(self.n, dtype=np.int64)
        self.mass = np.zeros(self.n)
        self.mdx = np.zeros((self.n, 3))
        self.ref = np.full((self.n, 3), np.nan)
        self.amax = np.zeros(self.n)

    def add(self, groups: np.ndarray, pos: np.ndarray, mass) -> None:
        """Members ``pos`` (k, 3) of 0-based ``groups`` (k,) with masses ``mass`` (k,) or one value."""
        g = np.asarray(groups, dtype=np.int64)
        if not len(g):
            return
        pos = np.asarray(pos, dtype=np.float64)
        m = np.broadcast_to(np.asarray(mass, dtype=np.float64), (len(g),))
        u, first = np.unique(g, return_index=True)
        new = np.isnan(self.ref[u, 0])
        self.ref[u[new]] = pos[first[new]]
        d = pos - self.ref[g]
        if self.box > 0:
            d -= self.box * np.round(d / self.box)
        for k in range(3):
            self.mdx[:, k] += np.bincount(g, weights=m * d[:, k], minlength=self.n)
        self.mass += np.bincount(g, weights=m, minlength=self.n)
        self.count += np.bincount(g, minlength=self.n)
        if self.box <= 0:
            np.maximum.at(self.amax, g, np.abs(pos).max(axis=1))

    def result(self) -> GroupMeasure:
        with np.errstate(invalid='ignore', divide='ignore'):
            com = self.ref + self.mdx / self.mass[:, None]
        if self.box > 0:
            com = np.mod(com, self.box)
        com[self.count == 0] = np.nan
        return GroupMeasure(self.count.copy(), self.mass.copy(), com, self.amax.copy())


def _first(group, names):
    return next((group[n] for n in names if n in group and isinstance(group[n], h5py.Dataset)), None)


def _header(files: Sequence[str]) -> Tuple[Dict, Dict]:
    """Header and Parameters (with Units) attributes of a snapshot's first file."""
    def attrs(g):
        return {k: (v.item() if isinstance(v, np.ndarray) and v.size == 1 else v) for k, v in g.attrs.items()}
    with h5py.File(str(files[0]), 'r') as f:
        header = attrs(f['Header']) if 'Header' in f else {}
        params = {}
        for name in ('Parameters', 'Units'):
            if name in f:
                params.update(attrs(f[name]))
    return header, params


def _mass_table(header: Dict) -> np.ndarray:
    mt = np.zeros(6)
    got = np.atleast_1d(np.asarray(header.get('MassTable', np.zeros(6)), dtype=np.float64))[:6]
    mt[:len(got)] = got
    return mt


def _progress(progress, done: int, total: int, cancelled) -> None:
    if cancelled is not None and cancelled():
        raise MeasureCancelled()
    if progress is not None:
        progress(done / max(total, 1))


def _rows_by_type(files: Sequence[str]) -> np.ndarray:
    n = np.zeros(6, dtype=np.int64)
    for fn in files:
        with h5py.File(str(fn), 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    n[t] += g['Coordinates'].shape[0]
    return n


def measure_catalogue_groups(files: Sequence[str], cat, subhalo: bool = False, groups: Optional[Sequence[int]] = None,
                             box: Optional[float] = None, progress: Optional[Callable[[float], None]] = None,
                             cancelled: Optional[Callable[[], bool]] = None,
                             chunk_rows: int = CHUNK_ROWS) -> GroupMeasure:
    """
    Members found, summed mass and periodic centre of mass of catalogue groups (or
    subhaloes), from the snapshot's own particles at full resolution. Members are
    the rows Gadget-4's offsets give (GroupOffsetType, GroupLenType), counted
    across the snapshot's files; ``groups`` (1-based) limits the pass to those
    groups' rows, otherwise one pass covers every group. A group whose rows lie
    beyond the end of the file (a cut snapshot) is found with fewer members.
    """
    off, ln = (cat.sub_offset, cat.sub_len) if subhalo else (cat.group_offset, cat.group_len)
    if off is None:
        raise ValueError("the catalogue has no subhaloes")
    files = [str(f) for f in files]
    header, params = _header(files)
    box = float(header.get('BoxSize', 0.0) or 0.0) if box is None else float(box)
    n = len(off)
    acc = GroupAccumulator(n, box)
    if groups is None:
        ranges = {}
        for t in range(6):
            has = ln[:, t] > 0
            if has.any():
                ranges[t] = [(int(off[has, t].min()), int((off[has, t] + ln[has, t]).max()))]
    else:
        gs = np.unique(np.asarray(list(groups), dtype=np.int64))
        bad = gs[(gs < 1) | (gs > n)]
        if len(bad):
            raise ValueError(f"{'subhalo' if subhalo else 'group'} {int(bad[0])} is not in the catalogue "
                             f"(it has 1 to {n:,})")
        ranges = {t: [(int(off[g - 1, t]), int(off[g - 1, t] + ln[g - 1, t])) for g in gs if ln[g - 1, t] > 0]
                  for t in range(6)}
    total = sum(b - a for rs in ranges.values() for a, b in rs)
    mt = _mass_table(header)
    done = 0
    running = np.zeros(6, dtype=np.int64)
    for fn in files:
        with h5py.File(fn, 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is None or 'Coordinates' not in g:
                    continue
                nrow = g['Coordinates'].shape[0]
                md = _first(g, MASS_NAMES)
                for a, b in ranges.get(t, []):
                    lo, hi = max(a - int(running[t]), 0), min(b - int(running[t]), nrow)
                    for s in range(lo, hi, chunk_rows):
                        e = min(s + chunk_rows, hi)
                        grp = membership(off, ln, t, np.arange(s, e, dtype=np.int64) + running[t]) - 1
                        keep = grp >= 0
                        if keep.any():
                            pos = g['Coordinates'][s:e][keep]
                            m = md[s:e][keep] if md is not None else mt[t]
                            acc.add(grp[keep], pos, m)
                        done += e - s
                        _progress(progress, done, total, cancelled)
                running[t] += nrow
    return acc.result()


def measure_fof_groups(files: Sequence[str], fof, n_groups: int, box: Optional[float] = None,
                       progress: Optional[Callable[[float], None]] = None,
                       cancelled: Optional[Callable[[], bool]] = None,
                       chunk_rows: int = CHUNK_ROWS) -> GroupMeasure:
    """
    The same for the groups of a FoF result (snapshot_io.fof_snapshot.SnapshotFoF),
    whose members are known by ParticleID: one pass over the snapshot's ID columns
    finds them, and positions and masses are read only for chunks that hold some.
    """
    files = [str(f) for f in files]
    header, _ = _header(files)
    box = float(header.get('BoxSize', 0.0) or 0.0) if box is None else float(box)
    sf = fof if getattr(fof, 'ids_sorted', False) else fof.sorted_by_id()
    gix = np.asarray(sf.group_index, dtype=np.int64)
    keep = gix >= 0
    sid, gix = np.asarray(sf.ids)[keep].astype(np.uint64), gix[keep]
    acc = GroupAccumulator(n_groups, box)
    mt = _mass_table(header)
    total = int(_rows_by_type(files).sum())
    done = 0
    for fn in files:
        with h5py.File(fn, 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is None or 'Coordinates' not in g or 'ParticleIDs' not in g:
                    continue
                nrow = g['ParticleIDs'].shape[0]
                md = _first(g, MASS_NAMES)
                for s in range(0, nrow, chunk_rows):
                    e = min(s + chunk_rows, nrow)
                    if len(sid):
                        pid = g['ParticleIDs'][s:e].astype(np.uint64)
                        j = np.minimum(np.searchsorted(sid, pid), len(sid) - 1)
                        hit = sid[j] == pid
                        if hit.any():
                            pos = g['Coordinates'][s:e][hit]
                            m = md[s:e][hit] if md is not None else mt[t]
                            acc.add(gix[j[hit]], pos, m)
                    done += e - s
                    _progress(progress, done, total, cancelled)
    return acc.result()


def measure_located(loc, header: Dict, box: float) -> Tuple[int, float, np.ndarray]:
    """(members found, summed mass, periodic centre of mass) of one group's located particles
    (snapshot_io.id_lookup.LocatedParticles). Masses of MassTable types are taken from the
    header in double precision, as the pass over the file does."""
    ok = np.asarray(loc.found, dtype=bool)
    types = np.asarray(loc.types)[ok].astype(np.int64)
    mt = _mass_table(header)
    m = np.where(mt[types] > 0, mt[types], np.asarray(loc.mass, dtype=np.float64)[ok])
    acc = GroupAccumulator(1, box)
    acc.add(np.zeros(int(ok.sum()), dtype=np.int64), np.asarray(loc.pos)[ok], m)
    r = acc.result()
    return int(r.count[0]), float(r.mass[0]), r.com[0]


# ----------------------------------------------------------------------------- comparing with the catalogue

@dataclass
class CatalogueCheck:
    """Measured against catalogue values for some groups (arrays aligned with ``groups``)."""
    groups: np.ndarray                   # 1-based numbers
    subhalo: bool
    n_cat: np.ndarray
    n_found: np.ndarray
    mass_cat: np.ndarray                 # the catalogue's float32 values, as float64
    mass: np.ndarray
    mass_ulp: np.ndarray                 # |difference| in float32 ulp of the catalogue value
    pos_field: str                       # 'GroupPos' or 'SubhaloCM' ('' if the catalogue has none)
    pos_cat: np.ndarray                  # (k, 3)
    com: np.ndarray                      # (k, 3)
    dpos: np.ndarray                     # (k, 3) nearest periodic image of com - pos_cat
    ulp_size: np.ndarray                 # (k,) one float32 ulp of a stored coordinate here
    pos_ulp: np.ndarray                  # (k,) largest |dpos| component in those ulp

    @property
    def complete(self) -> np.ndarray:
        return self.n_found == self.n_cat

    @property
    def mass_ok(self) -> np.ndarray:
        return self.complete & (self.mass_ulp <= MASS_ULP_TOL)

    @property
    def pos_ok(self) -> np.ndarray:
        return self.complete & (self.pos_ulp <= POS_ULP_TOL) if self.pos_field else np.zeros(len(self.groups), bool)


def compare_with_catalogue(measure: GroupMeasure, fields: Dict[str, np.ndarray], cat, subhalo: bool, box: float,
                           groups: Optional[Sequence[int]] = None) -> CatalogueCheck:
    """
    Compare measured groups with the catalogue: members found against the length
    (GroupLen, or SubhaloLen), summed mass against GroupMass (SubhaloMass), and the
    periodic centre of mass against GroupPos. For subhaloes the centre of mass is
    compared with SubhaloCM, because SubhaloPos is not a centre of mass in Gadget-4
    (it is the position of the particle with the minimum potential).
    """
    n_all = cat.n_subhalos if subhalo else cat.n_groups
    gs = np.arange(1, n_all + 1) if groups is None else np.asarray(list(groups), dtype=np.int64)
    i = gs - 1
    p = 'Subhalo' if subhalo else 'Group'
    lens = cat.sub_len if subhalo else cat.group_len
    n_cat = np.asarray(fields[f'{p}Len'], dtype=np.int64)[i] if f'{p}Len' in fields else lens.sum(axis=1)[i]
    mass_cat = np.asarray(fields.get(f'{p}Mass', np.full(n_all, np.nan)), dtype=np.float32)[i]
    with np.errstate(invalid='ignore', divide='ignore'):
        mass_ulp = np.abs(measure.mass[i] - mass_cat.astype(np.float64)) / np.spacing(np.abs(mass_cat))
    pos_field = 'SubhaloCM' if subhalo else 'GroupPos'
    if pos_field in fields:
        pc32 = np.asarray(fields[pos_field], dtype=np.float32)[i]
        pc = pc32.astype(np.float64)
        d = measure.com[i] - pc
        if box > 0:
            d -= box * np.round(d / box)
            size = np.full(len(i), float(np.spacing(np.float32(box))))
        else:
            size = np.spacing(np.maximum(np.abs(pc32).max(axis=1), measure.amax[i].astype(np.float32)))
            size = size.astype(np.float64)
        with np.errstate(invalid='ignore'):
            pos_ulp = np.abs(d).max(axis=1) / size
    else:
        pos_field, pc = '', np.full((len(i), 3), np.nan)
        d, size, pos_ulp = pc.copy(), np.full(len(i), np.nan), np.full(len(i), np.nan)
    return CatalogueCheck(gs, subhalo, n_cat, measure.count[i], mass_cat.astype(np.float64), measure.mass[i],
                          mass_ulp, pos_field, pc, measure.com[i], d, size, pos_ulp)


def check_summary(check: CatalogueCheck, catalogue: str, snapshot: str, length_unit: str) -> str:
    """One paragraph for all checked groups: what agrees, the worst cases, which groups differ."""
    what = "subhaloes" if check.subhalo else "groups"
    n = len(check.groups)
    if n == 0:
        return f"{catalogue} has no {what} to check."

    def worst(values, ok):
        v = np.where(ok, values, -np.inf)
        k = int(np.argmax(v))
        return float(values[k]), int(check.groups[k])

    def listed(mask):
        g = check.groups[mask]
        return ", ".join(f"{int(x):,}" for x in g[:5]) + (f" and {len(g) - 5:,} more" if len(g) > 5 else "")

    lines = [f"Checked all {n:,} {what} of {catalogue} against the particles of {snapshot}:"]
    full = check.complete
    if full.all():
        lines.append(f"members found = catalogue length for all {n:,}.")
    else:
        lines.append(f"<b>{int((~full).sum()):,} {what} have fewer members in the file than the catalogue says</b> "
                     f"({listed(~full)}); their mass and centre are not compared.")
    if full.any():
        ok = check.mass_ok
        if ok[full].all():
            m, g = worst(check.mass_ulp, full)
            lines.append(f"Summed member mass = catalogue mass within float32 precision for all {int(full.sum()):,} "
                         f"(worst {m:.2g} ulp, {what[:-2] if check.subhalo else 'group'} {g:,}).")
        else:
            bad = full & ~ok
            lines.append(f"<b>Mass differs by more than {MASS_ULP_TOL:g} float32 ulp for {int(bad.sum()):,}</b> "
                         f"({listed(bad)}).")
        if check.pos_field:
            ok = check.pos_ok
            unit = f"{float(np.nanmax(check.ulp_size)):.2g} {length_unit}"
            if ok[full].all():
                p, g = worst(check.pos_ulp, full)
                lines.append(f"Centre of mass = {check.pos_field} within {POS_ULP_TOL:g} float32 ulp for all "
                             f"{int(full.sum()):,} (1 ulp = {unit}; worst {p:.2g} ulp, "
                             f"{'subhalo' if check.subhalo else 'group'} {g:,}).")
            else:
                bad = full & ~ok
                lines.append(f"<b>Centre of mass differs from {check.pos_field} by more than {POS_ULP_TOL:g} ulp "
                             f"for {int(bad.sum()):,}</b> ({listed(bad)}).")
        else:
            lines.append("The catalogue stores no centre of mass for these to compare with.")
    return " ".join(lines)


def check_table(check: CatalogueCheck, k: int, mass_unit: str, length_unit: str) -> str:
    """HTML table for group ``check.groups[k]``: catalogue, measured, difference, verdict."""
    what = "Subhalo" if check.subhalo else "Group"
    p = what
    rows = []
    nc, nf = int(check.n_cat[k]), int(check.n_found[k])
    rows.append(("Members", f"{nc:,} ({p}Len)", f"{nf:,} found", f"{nf - nc:+,}",
                 "agrees" if nf == nc else "<b>members missing from the file</b>"))
    complete = nf == nc
    mc, mm = float(check.mass_cat[k]), float(check.mass[k])
    rows.append((f"Mass [{mass_unit}]", f"{mc:.7g} ({p}Mass)", f"{mm:.7g}", f"{mm - mc:+.2g} ({check.mass_ulp[k]:.2g} ulp)",
                 ("agrees within float32 precision" if check.mass_ulp[k] <= MASS_ULP_TOL else "<b>differs</b>")
                 if complete else "not compared"))
    if check.pos_field:
        pc, cm, d = check.pos_cat[k], check.com[k], check.dpos[k]
        fmt = lambda v: ", ".join(f"{x:.6f}" for x in v)
        rows.append((f"Centre [{length_unit}]", f"({fmt(pc)}) ({check.pos_field})", f"({fmt(cm)}) centre of mass",
                     f"({', '.join(f'{x:+.1e}' for x in d)}) ({check.pos_ulp[k]:.2g} ulp)",
                     ("agrees within float32 precision" if check.pos_ulp[k] <= POS_ULP_TOL else "<b>differs</b>")
                     if complete else "not compared"))
    head = "".join(f"<th align='left'>{h}</th>" for h in ("", "Catalogue", "Measured", "Difference", ""))
    body = "".join("<tr>" + "".join(f"<td>{c}&nbsp;&nbsp;</td>" for c in r) + "</tr>" for r in rows)
    note = (f"1 ulp = {float(check.ulp_size[k]):.2g} {length_unit} for positions here; agreement means within "
            f"{POS_ULP_TOL:g} ulp (positions) and {MASS_ULP_TOL:g} ulp (mass) of the stored float32 values."
            if check.pos_field else "")
    return (f"<b>{what} {int(check.groups[k]):,} measured from its member particles</b>"
            f"<table cellspacing='0'><tr>{head}</tr>{body}</table><span style='color:#888'>{note}</span>")


# ----------------------------------------------------------------------------- units as stored

@dataclass
class StoredUnits:
    """How numbers stored in this run's files are labelled."""
    length: str = "code units"           # e.g. "Mpc/h comoving"
    mass: str = "code units"             # e.g. "10¹⁰ M☉/h"
    comoving: bool = False
    sentence: str = ""


def _named(value: float, names) -> Optional[str]:
    for name, size in names:
        if abs(value / size - 1.0) < 1e-3:
            return name
    return None


def stored_units(files: Sequence[str]) -> StoredUnits:
    """Unit labels of stored lengths and masses, from Parameters and the datasets' a/h attributes."""
    header, params = _header(files)
    coord, masses = {}, {}
    with h5py.File(str(files[0]), 'r') as f:
        for t in range(6):
            g = f.get(f'PartType{t}')
            if g is None:
                continue
            if not coord and 'Coordinates' in g:
                coord = dict(g['Coordinates'].attrs)
            md = _first(g, MASS_NAMES)
            if not masses and md is not None:
                masses = dict(md.attrs)
    flag = params.get('ComovingIntegrationOn', header.get('ComovingIntegrationOn'))
    comoving = bool(int(flag)) if flag is not None else float(header.get('BoxSize', 0) or 0) > 0

    def h_exp(attrs):
        for k in ('h_scaling', 'h-scale-exponent'):
            if k in attrs:
                return float(np.asarray(attrs[k]).ravel()[0])
        return -1.0 if comoving else 0.0           # Gadget's convention when a dataset is unlabelled

    per_h = lambda e: "/h" if e == -1 else (f" h{str(int(e)).translate(_SUPERSCRIPT)}" if e else "")
    ul = float(params.get('UnitLength_in_cm', 0) or 0)
    um = float(params.get('UnitMass_in_g', 0) or 0)
    length = _named(ul, (('Gpc', 3.085678e27), ('Mpc', 3.085678e24), ('kpc', 3.085678e21), ('pc', 3.085678e18))) \
        if ul else None
    if length is None and ul:
        length = f"× {ul:.4g} cm"
    if um:
        msun = um / 1.989e33
        e = np.log10(msun)
        mass = "M☉" if abs(e) < 1e-6 else (f"10{str(int(round(e))).translate(_SUPERSCRIPT)} M☉"
                                           if abs(e - round(e)) < 1e-6 else f"{msun:.4g} M☉")
    else:
        mass = None
    out = StoredUnits(comoving=comoving)
    if length:
        out.length = length + per_h(h_exp(coord)) + (" (comoving)" if comoving else "")
    if mass:
        out.mass = mass + per_h(h_exp(masses))
    out.sentence = (f"Values as stored: lengths in {out.length}, masses in {out.mass}"
                    + (" (cosmological run: positions are comoving, and h is the run's HubbleParam = "
                       f"{float(params.get('HubbleParam', 1.0)):g})." if comoving else "."))
    return out


# ----------------------------------------------------------------------------- what the table shows

@dataclass
class StructureSource:
    """One list of groups for one snapshot: a catalogue's groups or subhaloes, or a FoF result."""
    key: tuple
    kind: str                            # 'catalogue' or 'fof'
    subhalo: bool
    label: str
    snap: dict                           # the snapshot the groups belong to (identity, not a copy)
    path: str = ""                       # catalogue file
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    fof: object = None                   # the FoF result the list was made from
    note: str = ""
    warn: str = ""


class _MeasureJob(QThread):
    """Runs one measurement pass off the GUI thread."""
    done = pyqtSignal(object, object, str)   # key, GroupMeasure (or None), error ('' on success)
    progress = pyqtSignal(float)

    def __init__(self, key, fn):
        super().__init__()
        self.key, self.fn, self._cancel = key, fn, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.done.emit(self.key, self.fn(self.progress.emit, lambda: self._cancel), '')
        except MeasureCancelled:
            self.done.emit(self.key, None, 'cancelled')
        except Exception as e:  # reported in the panel and the console, never swallowed
            self.done.emit(self.key, None, f"{type(e).__name__}: {e}")


class StructuresMixin:

    def _init_structures(self):
        """The Structures menu's halo table: a dock that follows the snapshot on screen."""
        from views.structures_panel import StructuresPanel
        self.structures_panel = p = StructuresPanel()
        dock = QDockWidget("Halo table", self.view)
        dock.setObjectName("HaloTableDock")
        dock.setWidget(p)
        dock.setStyleSheet("QDockWidget { color: #cccccc; } QDockWidget::title { background: #1a1a1a; padding: 4px; }")
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.view.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.hide()
        self.structures_dock = dock
        self._structures_source: Optional[StructureSource] = None
        self._structures_sources: Dict[tuple, StructureSource] = {}
        self._structures_want: Optional[tuple] = None        # the kind of source last chosen
        self._structures_selected: Optional[Tuple[tuple, int]] = None
        self._structures_follow_key = None
        self._structures_set_key = None
        self._structures_pending = None                      # a selected group waiting for its lookup
        self._structures_fields: Dict[tuple, tuple] = {}
        self._structures_units: Dict[tuple, StoredUnits] = {}
        self._fof_measures: Dict[int, Tuple[object, GroupMeasure]] = {}
        self._fof_sorted: Dict[int, Tuple[object, object]] = {}
        self._structures_cause = (None, None, "")      # (set, version, why it is incomplete)
        self._catalogue_measures: Dict[tuple, GroupMeasure] = {}
        self._structures_job: Optional[_MeasureJob] = None
        self._structures_old_jobs: List[_MeasureJob] = []
        self.structures_last_check: Optional[CatalogueCheck] = None
        self._structures_fullscreen_hidden = False

        act = getattr(self.view, 'act_halo_table', None)
        if act is not None:
            act.toggled.connect(self._show_halo_table)
            dock.toggleViewAction().toggled.connect(act.setChecked)
        dock.visibilityChanged.connect(lambda on: self._structures_follow(force=True) if on else None)
        if hasattr(self.view, 'act_fullscreen'):
            self.view.act_fullscreen.toggled.connect(self._structures_full_screen)
        self.view.frame_changed.connect(lambda _: self._structures_follow())
        p.source_chosen.connect(self._on_structure_source_chosen)
        p.group_chosen.connect(self._on_structure_chosen)
        p.analyse_requested.connect(self._structures_analyse)
        p.check_requested.connect(self._check_selected_structure)
        p.measure_all_requested.connect(self._measure_all_structures)
        p.cancel_requested.connect(self._cancel_structures_job)
        self._structures_timer = QTimer()
        self._structures_timer.setInterval(FOLLOW_MS)
        self._structures_timer.timeout.connect(self._structures_poll)
        self._structures_timer.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_structures)

    # ------------------------------------------------------------------ showing the dock

    def _show_halo_table(self, on: bool = True):
        self.structures_dock.setVisible(bool(on))
        if on:
            self.structures_dock.raise_()
            self._structures_follow(force=True)

    def _halo_table_open(self) -> bool:
        """The table was opened (and not closed), whether or not the main window is on screen yet."""
        return not self.structures_dock.isHidden()

    def _structures_full_screen(self, on: bool):
        """Full screen shows only the 3D view: the table steps aside and comes back afterwards."""
        dock = self.structures_dock
        if on and not dock.isHidden():
            self._structures_fullscreen_hidden = True
            dock.hide()
        elif not on and self._structures_fullscreen_hidden:
            self._structures_fullscreen_hidden = False
            dock.show()

    def _shutdown_structures(self):
        """On quit (and in tests): stop a measurement pass and wait for its thread."""
        self._structures_timer.stop()
        for job in [self._structures_job] + self._structures_old_jobs:
            if job is not None and job.isRunning():
                job.cancel()
                job.wait(10000)

    # ------------------------------------------------------------------ following the snapshot on screen

    def _structures_poll(self):
        try:
            self._structures_follow()
            self._structures_finish_pending()
        except RuntimeError:              # the window was deleted (Qt object gone): stop following
            self._structures_timer.stop()
        except Exception as e:  # the table must never break the main window
            log(f"Halo table: {type(e).__name__}: {e}", 'error')

    def _structures_state(self):
        snaps = self.model.snapshots or []
        snap = self.model.get_snapshot(self.current_snap_idx)
        return (id(snaps), len(snaps), self.current_snap_idx, id(snap), bool(self.loading), bool(self.is_playing),
                self._catalogue_mode, getattr(self, '_features_gen', 0),
                tuple(id(s.get('fof')) for s in snaps))

    def _structures_follow(self, force: bool = False):
        """Show the groups of the snapshot on screen, if the table is open and anything changed."""
        if not self._halo_table_open():
            self._structures_follow_key = None
            return
        key = self._structures_state()
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        set_key = (sp.active_set_idx, s_set.version)
        if not force and key == self._structures_follow_key:
            if set_key != self._structures_set_key:          # the active set changed elsewhere: say what it is
                self._structures_set_key = set_key
                self._show_structure_selection()
            return
        self._structures_follow_key, self._structures_set_key = key, set_key
        self._refresh_structures()

    def _structure_choices(self, snap: dict) -> Tuple[List[StructureSource], List[str]]:
        """The group lists that belong to ``snap``, and notes on what is missing and why."""
        name = Path(snap['filepath']).name
        out, notes = [], []
        if self._catalogue_mode:
            path, cat = self._catalogue_for(snap)
            if path is None:
                notes.append(f"Halo catalogue: {cat}.")
            else:
                cname = Path(path).name
                out.append(self._catalogue_source(snap, path, False, f"Catalogue groups · {cname}"))
                if cat.n_subhalos:
                    out.append(self._catalogue_source(snap, path, True, f"Catalogue subhaloes · {cname}"))
        else:
            notes.append(f"No halo catalogue is attached for {name} (Structures ▸ Read halo catalogue...).")
        if snap.get('fof') is not None:
            out.append(self._fof_source(snap))
        else:
            others = [Path(s['filepath']).name for s in self.model.snapshots or [] if s.get('fof') is not None]
            if others:
                listed = ", ".join(others[:3]) + (f" and {len(others) - 3} more" if len(others) > 3 else "")
                notes.append(f"FoF groups were found in {listed}, not in {name} (on screen); they are not listed "
                             f"here. Run FoF on this snapshot (Structures ▸ Find FoF groups...), or go back to "
                             f"that snapshot to use its groups.")
            else:
                notes.append(f"FoF has not been run on {name} (Structures ▸ Find FoF groups...).")
        return out, notes

    def _refresh_structures(self):
        p = self.structures_panel
        snap = self.model.get_snapshot(self.current_snap_idx)
        self._structures_source = None
        live = [s.get('fof') for s in self.model.snapshots or [] if s.get('fof') is not None]
        self._fof_measures = {k: v for k, v in self._fof_measures.items() if any(v[0] is f for f in live)}
        if snap is None or self.loading or self.is_playing:
            p.set_sources([], None)
            p.clear_table()
            p.set_actions(None, *self._analyse_state())
            p.set_note("Loading snapshots..." if self.loading else
                       "Playing snapshots: the table lists the groups of the snapshot on screen when playback stops."
                       if self.is_playing and snap is not None else
                       "Load snapshots first (File ▸ Read simulation snapshot...).")
            return
        choices, notes = self._structure_choices(snap)
        if not choices:
            p.set_sources([], None)
            p.clear_table()
            p.set_actions(None, *self._analyse_state())
            p.set_note(f"No groups to list for {Path(snap['filepath']).name} (on screen). " + " ".join(notes),
                       warn=True)
            self._show_structure_selection()
            return
        src = next((c for c in choices if self._structures_want and c.key[:1] + c.key[2:] == self._structures_want),
                   choices[0])
        self._show_structure_source(src, choices, notes)

    def _show_structure_source(self, src: StructureSource, choices: List[StructureSource], notes: List[str]):
        p = self.structures_panel
        self._structures_source = src
        p.set_sources([(c.label, c.key) for c in choices], src.key)
        columns, decimals = self._structure_columns(src)
        p.set_table(columns, len(columns[0].values), decimals)
        p.set_note(" ".join([src.note] + ([src.warn] if src.warn else []) + notes), warn=bool(src.warn))
        p.set_actions(src.kind, *self._analyse_state(), can_check=True)
        sel = self._structures_selected
        p.show_group(sel[1] if sel is not None and sel[0] == src.key else None)
        self._show_structure_selection()

    def _on_structure_source_chosen(self, key):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if snap is None:
            return
        choices, notes = self._structure_choices(snap)
        src = next((c for c in choices if c.key == key), None)
        if src is None:
            self._refresh_structures()
            return
        self._structures_want = src.key[:1] + src.key[2:]
        self._show_structure_source(src, choices, notes)

    # ------------------------------------------------------------------ the lists

    def _catalogue_fields(self, path: str, subhalo: bool) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
        p = Path(path)
        st = p.stat()
        key = (str(p.resolve()), subhalo, st.st_mtime_ns, st.st_size)
        if key not in self._structures_fields:
            if len(self._structures_fields) >= 6:          # a few catalogues; each is small
                self._structures_fields.pop(next(iter(self._structures_fields)))
            self._structures_fields[key] = read_catalogue_fields(path, subhalo)
        return self._structures_fields[key]

    def _stored_units(self, snap: dict) -> StoredUnits:
        key = tuple(self._snap_files(snap))
        if key not in self._structures_units:
            try:
                self._structures_units[key] = stored_units(list(key))
            except Exception as e:  # labels say "code units" rather than guess
                log(f"Halo table: could not read the units of {Path(key[0]).name}: {e}", 'warning')
                self._structures_units[key] = StoredUnits(sentence="Values as stored, in code units.")
        return self._structures_units[key]

    def _catalogue_source(self, snap: dict, path: str, subhalo: bool, label: str) -> StructureSource:
        fields, cheader, _ = self._catalogue_fields(path, subhalo)
        name, cname = Path(snap['filepath']).name, Path(path).name
        what = "subhaloes" if subhalo else "groups"
        _, cat = self._catalogue_for(snap)
        n = cat.n_subhalos if subhalo else cat.n_groups
        src = StructureSource(('catalogue', str(path), subhalo), 'catalogue', subhalo, label, snap, str(path), fields)
        units = self._stored_units(snap)
        src.note = (f"{n:,} {what} of {name} (on screen), from {cname}. {units.sentence} Selecting a row puts all "
                    f"members of that {'subhalo' if subhalo else 'group'} into the active set.")
        hdr = snap.get('header') or {}
        warn = []
        for k, label_k in (('Time', 'time'), ('BoxSize', 'box size')):
            a, b = cheader.get(k), hdr.get(k)
            if a is not None and b is not None and not np.isclose(float(a), float(b), rtol=1e-6, atol=0):
                warn.append(f"{cname} has {label_k} {float(a):.6g} but {name} has {float(b):.6g}: "
                            f"this catalogue may belong to another snapshot.")
        src.warn = " ".join(warn)
        return src

    def _fof_source(self, snap: dict) -> StructureSource:
        fof = snap['fof']
        d = self._fof_definition(snap, 1)
        params = []
        if d.get('b') is not None:
            params.append(f"b = {d['b']:g}")
        elif d.get('linking_length') is not None:
            params.append(f"l = {d['linking_length']:.4g}")
        if d.get('min_size') is not None:
            params.append(f"N_min = {d['min_size']}")
        if d.get('link_types'):
            params.append("+".join(_TYPE_NAMES.get(int(t), f"type {t}") for t in d['link_types']))
        text = ", ".join(params)
        src = StructureSource(('fof', id(fof)), 'fof', False, f"FoF on this snapshot · {text}", snap, fof=fof)
        units = self._stored_units(snap)
        n = len(snap.get('fof_sizes') or {})
        src.note = (f"{n:,} FoF groups found in this window on {Path(snap['filepath']).name} (on screen) with "
                    f"{text}. {units.sentence} Mass and centre are measured from the members at full resolution "
                    f"when a group is selected or with Measure all groups; — means not measured yet.")
        return src

    def _structure_columns(self, src: StructureSource):
        snap = src.snap
        units = self._stored_units(snap)
        box = float((snap.get('header') or {}).get('BoxSize', 0.0) or 0.0)
        span = box if box > 0 else (float(np.max(np.abs([self.model.g_min, self.model.g_max])))
                                    if self.model.g_max is not None else 1.0)
        decimals = int(np.clip(6 - np.ceil(np.log10(max(span, 1.0))), 0, 6))
        from views.structures_panel import Column
        L, M = units.length, units.mass
        if src.kind == 'fof':
            sizes = snap.get('fof_sizes') or {}
            n = len(sizes)
            numbers = np.arange(1, n + 1, dtype=np.int64)
            counts = np.array([sizes.get(int(g), 0) for g in numbers], dtype=np.int64)
            got = self._fof_measures.get(id(snap['fof']))
            meas = got[1] if got is not None and got[0] is snap['fof'] else None
            mass = meas.mass.copy() if meas is not None else np.full(n, np.nan)
            com = meas.com.copy() if meas is not None else np.full((n, 3), np.nan)
            if meas is not None:
                mass[meas.count == 0] = np.nan
            return [Column('group', "FoF\ngroup", "1 = the largest group (the FoF window's numbering)", numbers, 'int'),
                    Column('n', "N\nparticles", "Member particles, all linked types, from the FoF result", counts,
                           'int'),
                    Column('mass', f"Mass\n{M}", "Summed mass of every member, measured from the snapshot", mass),
                    Column('x', f"Centre x\n{L}", "Periodic centre of mass of every member, measured", com[:, 0],
                           'pos'),
                    Column('y', f"Centre y\n{L}", "Periodic centre of mass of every member, measured", com[:, 1],
                           'pos'),
                    Column('z', f"Centre z\n{L}", "Periodic centre of mass of every member, measured", com[:, 2],
                           'pos')], decimals
        f = src.fields
        _, cat = self._catalogue_for(snap)
        p = 'Subhalo' if src.subhalo else 'Group'
        lens = cat.sub_len if src.subhalo else cat.group_len
        n = len(lens)
        numbers = np.arange(1, n + 1, dtype=np.int64)
        counts = np.asarray(f[f'{p}Len'], dtype=np.int64) if f'{p}Len' in f else lens.sum(axis=1)
        mass = np.asarray(f.get(f'{p}Mass', np.full(n, np.nan)), dtype=np.float64)
        pos = np.asarray(f.get(f'{p}Pos', np.full((n, 3), np.nan)), dtype=np.float64)
        if src.subhalo:
            where = "SubhaloPos: in Gadget-4 the position of the member with the minimum gravitational potential"
            cols = [Column('group', "Subhalo", "1 = first subhalo in the catalogue", numbers, 'int')]
            if 'SubhaloGroupNr' in f:
                cols.append(Column('parent', "In\ngroup", "SubhaloGroupNr + 1: the group holding this subhalo "
                                                           "(1 = first group)",
                                   np.asarray(f['SubhaloGroupNr'], dtype=np.int64) + 1, 'int'))
            if 'SubhaloRankInGr' in f:
                cols.append(Column('rank', "Rank in\ngroup", "SubhaloRankInGr: 0 = the main subhalo of its group",
                                   np.asarray(f['SubhaloRankInGr'], dtype=np.int64), 'int'))
            cols += [Column('n', "N\nparticles", "SubhaloLen: member particles, all types", counts, 'int'),
                     Column('mass', f"Mass\n{M}", "SubhaloMass: summed mass of the members, as stored", mass)]
        else:
            where = "GroupPos: in Gadget-4 FoF catalogues the centre of mass of the members"
            cols = [Column('group', "Group", "1 = first group in the catalogue (as the Catalogue criterion counts)",
                           numbers, 'int'),
                    Column('n', "N\nparticles", "GroupLen: member particles, all types", counts, 'int'),
                    Column('mass', f"Mass\n{M}", "GroupMass: summed mass of the members, as stored", mass)]
        for k, axis in enumerate("xyz"):
            cols.append(Column(axis, f"Centre {axis}\n{L}", where, pos[:, k], 'pos'))
        return cols, decimals

    # ------------------------------------------------------------------ selecting a group

    def _group_population(self, snap: dict, kind: str, group: int, subhalo: bool = False):
        """(ids, definition, complete, n_expected, cause) of a group of ``snap``; ValueError with a message
        if the group cannot be selected here. ``cause`` explains an incomplete membership."""
        name = Path(snap['filepath']).name
        group = int(group)
        if kind == 'fof':
            ids = self._fof_members(snap, group)          # ValueError when FoF did not run here
            n_expected = int((snap.get('fof_sizes') or {}).get(group, len(ids)))
            return ids, self._fof_definition(snap, group), True, n_expected, ""
        if not self._catalogue_mode:
            raise ValueError(f"No halo catalogue is attached for {name} (Structures ▸ Read halo catalogue...).")
        path, cat = self._catalogue_for(snap)
        if path is None:
            raise ValueError(f"No halo catalogue for {name}: {cat}")
        what = 'subhalo' if subhalo else 'group'
        n_all = cat.n_subhalos if subhalo else cat.n_groups
        if not 1 <= group <= n_all:
            raise ValueError(f"{Path(path).name} has no {what} {group:,}: it has {what}s 1 to {n_all:,}.")
        ids = np.unique(group_ids_full(self._snap_files(snap), cat, group, subhalo).astype(np.uint64))
        n_expected = int((cat.sub_len if subhalo else cat.group_len)[group - 1].sum())
        definition = {'finder': 'catalogue', 'group': group, 'subhalo': bool(subhalo),
                      'catalogue_file': Path(path).name}
        cause = ""
        if len(ids) < n_expected:
            cause = (f"Only {len(ids):,} of the {n_expected:,} members of {Path(path).name} {what} {group:,} are "
                     f"in {name}: the file holds fewer particles than the catalogue describes (for example a cut "
                     f"snapshot), so the set holds those {len(ids):,} and is marked incomplete.")
        return ids, definition, len(ids) == n_expected, n_expected, cause

    def select_group_population(self, snap: dict, kind: str, group: int, subhalo: bool = False):
        """
        Make the active set the complete membership of a group of ``snap`` (the
        snapshot on screen): a catalogue group or subhalo, or a FoF group. The set
        becomes a population of kind "group" defined at ``snap``, and the 3D view is
        centred on it. Returns the set; raises ValueError (with a message for the
        reader) when the group cannot be selected, and then leaves the set as it was.
        """
        ids, definition, complete, n_expected, cause = self._group_population(snap, kind, group, subhalo)
        name = Path(snap['filepath']).name
        if len(ids) == 0:
            raise ValueError(f"None of the {n_expected:,} members of {'subhalo' if subhalo else 'group'} "
                             f"{group:,} are in {name}; the set was left as it was.")
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        if not s_set.enabled:
            s_set.enabled = True
        # When members are missing from the file, complete is False and the set says why
        # (cause is non-empty exactly then).
        s_set.set_population(ids, 'group', definition, self._defined_at(snap), complete=complete,
                             n_expected=n_expected,
                             incomplete_reason="missing_in_file" if cause else None)
        self._structures_cause = (s_set.set_id, s_set.version, cause)
        log(f"Set {s_set.set_id + 1}: {s_set.describe()}" + (f" ({cause})" if cause else ""))
        key = ('catalogue', str(self._catalogue_for(snap)[0]), bool(subhalo)) if kind == 'catalogue' \
            else ('fof', id(snap['fof']))
        self._structures_selected = (key, int(group))
        sp._refresh_ui()
        self._structures_pending = None
        n_total = int(snap.get('n_total', 0) or len(snap.get('id', ())))
        if n_total <= SYNC_LOOKUP_MAX:
            self._centre_on_group(snap, s_set, kind, group)
        else:                                            # the tracking worker looks it up; centred when it lands
            self._structures_pending = (snap, s_set.set_id, s_set.version, kind, int(group))
        self._update_frame()
        if self._halo_table_open():
            src = self._structures_source
            if src is not None and src.key == key:
                self.structures_panel.show_group(int(group))
            self._show_structure_selection()
        return s_set

    def _centre_on_group(self, snap: dict, s_set, kind: str, group: int):
        """Full-resolution positions of the set here (cached for tracking too), then the usual recentre."""
        idx = self.current_snap_idx
        loc = self._tracked(idx, s_set, compute=True)
        self._recentre_camera()
        if kind == 'fof' and loc is not None:
            header, _ = _header(self._snap_files(snap))
            box = float(header.get('BoxSize', 0.0) or 0.0)
            count, mass, com = measure_located(loc, header, box)
            self._store_fof_measure(snap, group, count, mass, com)

    def _structures_finish_pending(self):
        pend = self._structures_pending
        if pend is None:
            return
        snap, set_id, version, kind, group = pend
        s_set = self.model.tracker.get_set(set_id)
        if s_set.version != version or self.model.get_snapshot(self.current_snap_idx) is not snap:
            self._structures_pending = None              # the set or the snapshot changed since
            return
        if self._tracked(self.current_snap_idx, s_set) is not None:
            self._structures_pending = None
            self._centre_on_group(snap, s_set, kind, group)
            self._show_structure_selection()

    def _store_fof_measure(self, snap: dict, group: int, count: int, mass: float, com: np.ndarray):
        fof = snap['fof']
        n = len(snap.get('fof_sizes') or {})
        got = self._fof_measures.get(id(fof))
        if got is None or got[0] is not fof:
            got = (fof, GroupMeasure(np.zeros(n, np.int64), np.full(n, np.nan), np.full((n, 3), np.nan),
                                     np.zeros(n)))
            self._fof_measures[id(fof)] = got
        m = got[1]
        m.count[group - 1], m.mass[group - 1], m.com[group - 1] = count, mass, com
        self._show_fof_measures(snap)

    def _show_fof_measures(self, snap: dict):
        src = self._structures_source
        if src is None or src.kind != 'fof' or src.snap is not snap or not self._halo_table_open():
            return
        got = self._fof_measures.get(id(src.fof))
        if got is None or got[0] is not src.fof:
            return
        m = got[1]
        mass = np.where(m.count > 0, m.mass, np.nan)
        p = self.structures_panel
        p.set_values('mass', mass)
        for k, axis in enumerate("xyz"):
            p.set_values(axis, m.com[:, k])

    def _source_is_current(self, src: Optional[StructureSource], snap: Optional[dict]) -> bool:
        """True when ``src`` still describes the snapshot on screen: the same snapshot, and the same FoF
        result or catalogue file (FoF may have been run again, or another catalogue attached, since)."""
        if src is None or snap is None or src.snap is not snap:
            return False
        if src.kind == 'fof':
            return snap.get('fof') is src.fof
        return bool(self._catalogue_mode) and str(self._catalogue_for(snap)[0]) == src.path

    def _on_structure_chosen(self, group: int):
        src = self._structures_source
        snap = self.model.get_snapshot(self.current_snap_idx)
        p = self.structures_panel
        if not self._source_is_current(src, snap):
            self._refresh_structures()                   # never apply a list made for another snapshot
            p.set_note("The list was out of date and has been refreshed; choose the group again.", warn=True)
            return
        try:
            self.select_group_population(snap, src.kind, group, src.subhalo)
        except ValueError as e:
            p.set_selection_text(str(e), warn=True)

    def _show_structure_selection(self):
        """Under the table: what the active set now is, and why it is incomplete if it is."""
        p = self.structures_panel
        s_set = self.model.tracker.get_set(self.view.selection_panel.active_set_idx)
        sel = self._structures_selected
        src = self._structures_source
        if not s_set.selected_ids:
            p.set_selection_text(f"Selecting a row puts all members of that group into Set {s_set.set_id + 1} "
                                 f"(the active set) and centres the view on them.")
        else:
            text = f"Set {s_set.set_id + 1} ({s_set.name}): {s_set.describe()}"
            sid, ver, why = self._structures_cause
            cause = why if (sid, ver) == (s_set.set_id, s_set.version) else ""
            if self._structures_pending is not None:
                text += " · finding its members in the snapshot to centre the view..."
            p.set_selection_text(text + (f"\n{cause}" if cause else ""), warn=bool(cause))
        p.set_actions(src.kind if src is not None else None, *self._analyse_state(),
                      can_check=src is not None and sel is not None and sel[0] == src.key)

    # ------------------------------------------------------------------ analyse and check

    def _analyse_state(self) -> Tuple[str, bool]:
        fn = getattr(self, '_analyse_active_set', None)
        s_set = self.model.tracker.get_set(self.view.selection_panel.active_set_idx)
        if not callable(fn):
            return "Opens the active set in the Analyse view, which is not part of this build yet.", False
        if not s_set.selected_ids:
            return "Select a group first; Analyse opens the active set in the Analyse view.", False
        return (f"Open Set {s_set.set_id + 1} in the Analyse view: the same snapshot, with exactly these "
                f"particles", True)

    def _structures_analyse(self):
        fn = getattr(self, '_analyse_active_set', None)
        if callable(fn):
            fn()

    def _check_selected_structure(self):
        """Measure the selected catalogue group from its members and compare with the catalogue."""
        p = self.structures_panel
        src, sel = self._structures_source, self._structures_selected
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not self._source_is_current(src, snap) or src.kind != 'catalogue' or sel is None or sel[0] != src.key:
            p.set_check_text("Select a catalogue group in the table first.")
            return
        group = sel[1]
        _, cat = self._catalogue_for(snap)
        box = float((snap.get('header') or {}).get('BoxSize', 0.0) or 0.0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            meas = measure_catalogue_groups(self._snap_files(snap), cat, src.subhalo, [group], box)
        except Exception as e:  # shown where the result would be
            p.set_check_text(f"Could not measure {'subhalo' if src.subhalo else 'group'} {group:,}: "
                             f"{type(e).__name__}: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        check = compare_with_catalogue(meas, src.fields, cat, src.subhalo, box, [group])
        self.structures_last_check = check
        units = self._stored_units(snap)
        p.set_check_text(check_table(check, 0, units.mass, units.length))
        log(f"Halo table: {'subhalo' if src.subhalo else 'group'} {group:,} of {Path(src.path).name}: members "
            f"{int(check.n_found[0]):,} of {int(check.n_cat[0]):,}, mass {check.mass_ulp[0]:.2g} ulp, centre "
            f"{check.pos_ulp[0]:.2g} ulp from the catalogue")

    def _measure_all_structures(self):
        """Measure every group of the list shown in one background pass over the snapshot."""
        p = self.structures_panel
        src = self._structures_source
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not self._source_is_current(src, snap) or self._structures_job is not None:
            return
        files = self._snap_files(snap)
        box = float((snap.get('header') or {}).get('BoxSize', 0.0) or 0.0)
        if src.kind == 'catalogue':
            _, cat = self._catalogue_for(snap)
            sub = src.subhalo
            fn = lambda progress, cancelled: measure_catalogue_groups(files, cat, sub, None, box, progress, cancelled)
            key = src.key + (tuple(files),)
        else:
            fof, n = src.fof, len(snap.get('fof_sizes') or {})
            fn = lambda progress, cancelled: measure_fof_groups(files, fof, n, box, progress, cancelled)
            key = src.key + (tuple(int(snap['fof_sizes'][g]) for g in range(1, n + 1)),)
        job = _MeasureJob(key, fn)
        job.progress.connect(lambda f: p.set_busy(f"Measuring every group from its members... {100 * f:.0f}%"))
        job.done.connect(lambda k, result, err, j=job, s=src: self._on_structures_measured(j, s, result, err))
        self._structures_job = job
        p.set_busy("Measuring every group from its members... 0%")
        hold(job).start()

    def _cancel_structures_job(self):
        if self._structures_job is not None:
            self._structures_job.cancel()
            self.structures_panel.set_busy("Cancelling...")

    def _on_structures_measured(self, job: _MeasureJob, src: StructureSource, result: Optional[GroupMeasure],
                                err: str):
        job.wait()                       # run() is past its last emit: let the thread end before it is released
        job.fn = None
        self._structures_old_jobs = [j for j in self._structures_old_jobs if j.isRunning()] + [job]
        self._structures_job = None
        p = self.structures_panel
        p.set_busy(None)
        if err:
            if err == 'cancelled':
                p.set_check_text("Measurement cancelled; nothing was changed.")
            else:
                log(f"Halo table: measuring all groups failed: {err}", 'error')
                p.set_check_text(f"Measuring all groups failed, nothing was changed: {err}")
            return
        snap = src.snap
        if src.kind == 'fof':
            self._fof_measures[id(src.fof)] = (src.fof, result)
            self._show_fof_measures(snap)
            n = len(result.count)
            want = np.asarray(job.key[-1], dtype=np.int64)    # the sizes of the result that was measured
            bad = np.flatnonzero(result.count != want)
            text = (f"Measured all {n:,} FoF groups from their members." if not len(bad) else
                    f"Measured all {n:,} FoF groups; <b>{len(bad):,} have a different number of members in the "
                    f"file than in the FoF result</b> (first: group {int(bad[0]) + 1:,}).")
            if self._structures_source is not None and self._structures_source.key == src.key:
                p.set_check_text(text)
            return
        _, cat = self._catalogue_for(snap)
        box = float((snap.get('header') or {}).get('BoxSize', 0.0) or 0.0)
        if len(self._catalogue_measures) >= 8:
            self._catalogue_measures.pop(next(iter(self._catalogue_measures)))
        self._catalogue_measures[job.key] = result
        check = compare_with_catalogue(result, src.fields, cat, src.subhalo, box)
        self.structures_last_check = check
        units = self._stored_units(snap)
        text = check_summary(check, Path(src.path).name, Path(snap['filepath']).name, units.length)
        log("Halo table: " + text.replace("<b>", "").replace("</b>", ""))
        if self._structures_source is not None and self._structures_source.key == src.key:
            p.set_check_text(text)

    # ------------------------------------------------------------------ for the particle inspector

    def _fof_group_of(self, snap: Optional[dict], pid: int) -> Optional[int]:
        """The FoF group (1 = first, 0 = none) of particle ``pid`` in the FoF result of ``snap``;
        None when FoF has not been run on ``snap``."""
        fof = snap.get('fof') if snap else None
        if fof is None:
            return None
        if not getattr(fof, 'ids_sorted', False):           # FoF results are sorted by ID; others once here
            got = self._fof_sorted.get(id(fof))
            if got is None or got[0] is not fof:
                self._fof_sorted = {k: v for k, v in self._fof_sorted.items()
                                    if any(s.get('fof') is v[0] for s in self.model.snapshots or [])}
                got = self._fof_sorted[id(fof)] = (fof, fof.sorted_by_id())
            fof = got[1]
        q = np.asarray([pid]).astype(np.asarray(fof.ids).dtype)
        g = int(fof.lookup(q)[0])
        return g + 1 if g >= 0 else 0
