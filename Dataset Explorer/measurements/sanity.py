"""
sanity.py — is this snapshot fit to measure?

One pass over every file of a snapshot, reading every particle (never a
sample), and one table of checks, each PASS, WARN, FAIL or INFO:

  counts       header NumPart_Total (with HighWord) = sum of NumPart_ThisFile
               = the length of every dataset of that type
  finite       no NaN or inf in any floating-point dataset
  positive     Masses, Density, SmoothingLength and InternalEnergy are > 0
               (≥ 0 for InternalEnergy, which is 0 in cold gas)
  ids          no particle ID appears twice, across all types together
  positions    no two particles share exactly the same position
  box          in a periodic run every coordinate lies in [0, BoxSize);
               a coordinate exactly equal to BoxSize is a WARN, anything
               outside [0, BoxSize] a FAIL
  velocities   no speed above c; speeds above 10 × the 99.9th percentile
               of the type are a WARN (a heuristic, reported as such)
  mass         total mass per type; MassTable and a Masses dataset both
               set for one type is a WARN; in a cosmological box the total
               is compared with Ω₀ ρ_crit V (WARN beyond 1 %)
  softening    physical softening per type from the parameters (INFO)

Checks read the raw stored numbers (code units). Only the mass budget and
softening convert units, through SnapshotView, so they agree with every
other measurement. Datasets are read one file and one block at a time, so
memory stays bounded except for the ID and position checks, which need every
value at once (8 and 24 bytes per particle), and the velocity check, which
keeps one speed per particle of a type for its percentile (8 bytes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np

from .results import Provenance
from .snapshot_view import KPC_CM, MSUN_G, SnapshotView

VERSION = "1"
STATUSES = ("PASS", "WARN", "FAIL", "INFO")
BLOCK = 4_000_000                 # rows read at once per dataset
C_KMS = 299_792.458
VELOCITY_OUTLIER_FACTOR = 10.0    # × the 99.9th percentile of |v| for the type
MASS_BUDGET_TOLERANCE = 0.01      # cosmological total mass vs Ω₀ ρ_crit V
GADGET_G = 6.6738e-8               # cgs, Gadget-4 src/data/constants.h
GADGET_HUBBLE = 3.2407789e-18      # h / s, same file
MAX_LISTED = 5                    # example indices / IDs listed per finding

_POSITIVE = {"Masses": "> 0", "Mass": "> 0", "Density": "> 0",
             "SmoothingLength": "> 0", "InternalEnergy": "≥ 0"}


@dataclass
class Check:
    check: str
    scope: str             # "PartType1", "all types", "header", …
    status: str
    detail: str
    count: int = 0         # particles affected, when that applies
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"check": self.check, "scope": self.scope, "status": self.status,
                "count": int(self.count), "detail": self.detail,
                "examples": [int(x) if isinstance(x, (int, np.integer)) else x
                             for x in self.examples]}


@dataclass
class SanityReport:
    checks: list
    provenance: Provenance

    @property
    def worst(self) -> str:
        for s in ("FAIL", "WARN"):
            if any(c.status == s for c in self.checks):
                return s
        return "PASS"

    def counts(self) -> dict:
        return {s: sum(c.status == s for c in self.checks) for s in STATUSES}

    def by_status(self, status: str) -> list:
        return [c for c in self.checks if c.status == status]

    def find(self, check: str, scope: str | None = None) -> list:
        return [c for c in self.checks if c.check == check and (scope is None or c.scope == scope)]

    def summary(self) -> str:
        n = self.counts()
        return (f"Sanity {self.worst}: {n['FAIL']} fail, {n['WARN']} warn, "
                f"{n['PASS']} pass · all {self.provenance.n_total:,} particles read")

    def to_text(self) -> str:
        w = max(len(c.check) for c in self.checks)
        s = max(len(c.scope) for c in self.checks)
        lines = [self.summary(), f"files: {', '.join(self.provenance.files)}", ""]
        for c in self.checks:
            lines.append(f"{c.status:<4}  {c.check:<{w}}  {c.scope:<{s}}  {c.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {"summary": self.summary(), "worst": self.worst,
                "checks": [c.to_dict() for c in self.checks],
                "provenance": asdict(self.provenance)}


# ── Reading helpers ────────────────────────────────────────────────────────

def _blocks(files, group, name):
    """Yield (global_offset, block) for one dataset over every file."""
    offset = 0
    for path in files:
        with h5py.File(path, "r", rdcc_nbytes=0) as f:
            g = f.get(group)
            if g is None or name not in g:
                continue
            ds = g[name]
            for i in range(0, ds.shape[0], BLOCK):
                yield offset + i, ds[i:i + BLOCK]
            offset += ds.shape[0]


def _concat(files, group, name):
    parts = [b for _, b in _blocks(files, group, name)]
    return np.concatenate(parts) if parts else None


def _datasets(files, group):
    """Union of dataset names (and their dtype/shape) of a group over all files."""
    out = {}
    for path in files:
        with h5py.File(path, "r") as f:
            g = f.get(group)
            if g is None:
                continue
            for k, ds in g.items():
                if isinstance(ds, h5py.Dataset):
                    out.setdefault(k, (ds.dtype, ds.shape[1:]))
    return out


def _lengths(files, group, name):
    total = 0
    for path in files:
        with h5py.File(path, "r") as f:
            g = f.get(group)
            if g is not None and name in g:
                total += g[name].shape[0]
    return total


def _config(files) -> dict | None:
    with h5py.File(files[0], "r") as f:
        if "Config" not in f:
            return None
        return dict(f["Config"].attrs)


def _examples(idx, ids=None):
    idx = np.asarray(idx)[:MAX_LISTED]
    if ids is None:
        return [f"index {int(i)}" for i in idx]
    return [f"index {int(i)} (ID {int(ids[i])})" for i in idx]


# ── Checks ─────────────────────────────────────────────────────────────────

def _check_counts(view, files, types, out):
    hdr = view.header
    low = np.asarray(hdr.get("NumPart_Total", [0] * 6), dtype=np.uint64)
    high = np.asarray(hdr.get("NumPart_Total_HighWord", [0] * len(low)), dtype=np.uint64)
    per_file = np.zeros(len(low), dtype=np.uint64)
    for path in files:
        with h5py.File(path, "r") as f:
            this = np.asarray(f["Header"].attrs.get("NumPart_ThisFile", [0] * len(low)), np.uint64)
            per_file[:len(this)] += this
    for t in range(len(low)):
        total = int((high[t] << np.uint64(32)) | low[t]) if t < len(high) else int(low[t])
        grp = f"PartType{t}"
        dsets = _datasets(files, grp)
        if total == 0 and not dsets:
            continue
        bad = []
        if int(per_file[t]) != total:
            bad.append(f"sum of NumPart_ThisFile = {int(per_file[t]):,}")
        for name in sorted(dsets):
            n = _lengths(files, grp, name)
            if n != total:
                bad.append(f"{name} has {n:,} rows")
        if bad:
            out.append(Check("counts", grp, "FAIL",
                             f"header says {total:,}; " + "; ".join(bad)))
        else:
            out.append(Check("counts", grp, "PASS",
                             f"{total:,} particles; header, files and {len(dsets)} datasets agree"))


def _check_finite_and_sign(files, types, out):
    for t in types:
        grp = f"PartType{t}"
        bad_finite, n_checked, sign_rows = [], 0, []
        for name, (dtype, _) in sorted(_datasets(files, grp).items()):
            if not np.issubdtype(dtype, np.floating):
                continue
            n_checked += 1
            n_bad, first = 0, []
            rule = _POSITIVE.get(name)
            n_neg, first_neg = 0, []
            for off, b in _blocks(files, grp, name):
                fin = np.isfinite(b)
                row_bad = ~fin.all(axis=tuple(range(1, b.ndim))) if b.ndim > 1 else ~fin
                k = int(row_bad.sum())
                if k:
                    n_bad += k
                    if len(first) < MAX_LISTED:
                        first.extend((off + np.flatnonzero(row_bad))[:MAX_LISTED - len(first)])
                if rule:
                    neg = (b < 0) if rule == "≥ 0" else (b <= 0)
                    neg &= fin
                    k = int(neg.sum())
                    if k:
                        n_neg += k
                        if len(first_neg) < MAX_LISTED:
                            first_neg.extend((off + np.flatnonzero(neg))[:MAX_LISTED - len(first_neg)])
            if n_bad:
                bad_finite.append((name, n_bad, first))
            if rule:
                if n_neg:
                    sign_rows.append(Check("positive", grp, "FAIL",
                                           f"{name}: {n_neg:,} values break {rule}",
                                           n_neg, _examples(first_neg)))
                else:
                    sign_rows.append(Check("positive", grp, "PASS", f"{name} {rule} everywhere"))
        if bad_finite:
            for name, n_bad, first in bad_finite:
                out.append(Check("finite", grp, "FAIL", f"{name}: {n_bad:,} rows with NaN or inf",
                                 n_bad, _examples(first)))
        else:
            out.append(Check("finite", grp, "PASS",
                             f"no NaN or inf in {n_checked} floating-point datasets"))
        out.extend(sign_rows)


def _id_name(files, t):
    d = _datasets(files, f"PartType{t}")
    return next((n for n in ("ParticleIDs", "ParticleID") if n in d), None)


def _check_ids(files, types, out):
    parts, owners = [], []
    for t in types:
        name = _id_name(files, t)
        if name is None:
            out.append(Check("ids", f"PartType{t}", "WARN", "no ParticleIDs dataset"))
            continue
        a = _concat(files, f"PartType{t}", name)
        parts.append(a.astype(np.uint64, copy=False))
        owners.append(np.full(len(a), t, dtype=np.int8))
    if not parts:
        return
    ids = np.concatenate(parts)
    owner = np.concatenate(owners)
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    dup = np.flatnonzero(s[1:] == s[:-1])
    if len(dup) == 0:
        out.append(Check("ids", "all types", "PASS", f"all {len(ids):,} IDs unique"))
        return
    dup_ids = np.unique(s[dup])
    n_part = int(np.isin(ids, dup_ids).sum())
    ex = []
    for v in dup_ids[:MAX_LISTED]:
        where = order[np.searchsorted(s, v):np.searchsorted(s, v, side="right")]
        ex.append(f"ID {int(v)} × {len(where)} (types {sorted(set(int(owner[i]) for i in where))})")
    out.append(Check("ids", "all types", "FAIL",
                     f"{len(dup_ids):,} IDs appear more than once ({n_part:,} particles)",
                     n_part, ex))


def _check_positions(files, types, out):
    pos = []
    for t in types:
        a = _concat(files, f"PartType{t}", "Coordinates")
        if a is not None:
            pos.append(np.asarray(a, dtype=np.float64))
    if not pos:
        return
    p = np.concatenate(pos)
    order = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
    s = p[order]
    same = np.flatnonzero(np.all(s[1:] == s[:-1], axis=1))
    if len(same) == 0:
        out.append(Check("positions", "all types", "PASS",
                         f"no two of {len(p):,} particles share a position"))
        return
    n = len(np.unique(np.concatenate([order[same], order[same + 1]])))
    out.append(Check("positions", "all types", "WARN",
                     f"{n:,} particles share an exact position with another "
                     "(a softened force stays finite, but initial conditions may be broken)",
                     n, _examples(np.sort(order[same]))))


def _periodic(view, cfg) -> tuple:
    box = float(view.header.get("BoxSize", 0.0) or 0.0)
    if box <= 0:
        return False, box, "BoxSize is 0: not a periodic box"
    if cfg is not None:
        if "PERIODIC" in cfg:
            return True, box, "PERIODIC set in Config"
        return False, box, "BoxSize > 0 but PERIODIC not in Config"
    if view.is_cosmological:
        return True, box, "no Config group; cosmological run assumed periodic"
    return False, box, "no Config group and not cosmological; box not checked"


def _check_box(view, files, types, cfg, out):
    periodic, box, why = _periodic(view, cfg)
    if not periodic:
        out.append(Check("box", "all types", "INFO", why))
        return
    for t in types:
        grp = f"PartType{t}"
        n_edge = n_out = 0
        ex_edge, ex_out = [], []
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for off, b in _blocks(files, grp, "Coordinates"):
            b = np.asarray(b, dtype=np.float64)
            lo = np.minimum(lo, b.min(axis=0))
            hi = np.maximum(hi, b.max(axis=0))
            outside = ((b < 0) | (b > box)).any(axis=1)
            edge = (b == box).any(axis=1) & ~outside
            k = int(outside.sum())
            if k:
                n_out += k
                ex_out.extend((off + np.flatnonzero(outside))[:MAX_LISTED])
            k = int(edge.sum())
            if k:
                n_edge += k
                ex_edge.extend((off + np.flatnonzero(edge))[:MAX_LISTED])
        rng = (f"range x [{lo[0]:.6g}, {hi[0]:.6g}], y [{lo[1]:.6g}, {hi[1]:.6g}], "
               f"z [{lo[2]:.6g}, {hi[2]:.6g}] of BoxSize {box:g}")
        ids = None
        if n_edge or n_out:
            name = _id_name(files, t)
            ids = _concat(files, grp, name) if name else None
        if n_out:
            out.append(Check("box", grp, "FAIL",
                             f"{n_out:,} particles outside [0, BoxSize]; {rng}",
                             n_out, _examples(ex_out[:MAX_LISTED], ids)))
        if n_edge:
            out.append(Check("box", grp, "WARN",
                             f"{n_edge:,} particles exactly on the upper edge (coordinate = BoxSize, "
                             "which is the same point as 0); wrap them before binning",
                             n_edge, _examples(ex_edge[:MAX_LISTED], ids)))
        if not (n_out or n_edge):
            out.append(Check("box", grp, "PASS", f"every coordinate in [0, BoxSize); {why}"))


def _check_velocities(view, files, types, out):
    for t in types:
        grp = f"PartType{t}"
        vname = next((n for n in ("Velocities", "Velocity") if n in _datasets(files, grp)), None)
        if vname is None:
            continue
        attrs = view.attrs(t, vname)
        factor = view.conversion(attrs, "physical") / 1.0e5       # raw → km/s
        speed = np.concatenate([np.sqrt(np.einsum("ij,ij->i", b, b))
                                for _, b in _blocks(files, grp, vname)
                                for b in [np.asarray(b, dtype=np.float64)]]) * factor
        if not len(speed):
            continue
        fin = np.isfinite(speed)
        sp = speed[fin]
        if not len(sp):
            continue
        p999 = float(np.percentile(sp, 99.9))
        vmax = float(sp.max())
        fast = np.flatnonzero(fin & (speed > C_KMS))
        if len(fast):
            out.append(Check("velocities", grp, "FAIL",
                             f"{len(fast):,} speeds above c (max {vmax:.4g} km/s)",
                             len(fast), _examples(fast)))
            continue
        limit = VELOCITY_OUTLIER_FACTOR * p999
        odd = np.flatnonzero(fin & (speed > limit)) if p999 > 0 else np.array([], int)
        note = "" if not view.is_cosmological else " (peculiar)"
        if len(odd):
            out.append(Check("velocities", grp, "WARN",
                             f"{len(odd):,} speeds above {VELOCITY_OUTLIER_FACTOR:g} × the 99.9th "
                             f"percentile ({p999:.4g} km/s){note}; max {vmax:.4g} km/s. "
                             "Heuristic: check whether these are real",
                             len(odd), _examples(odd)))
        else:
            out.append(Check("velocities", grp, "PASS",
                             f"max speed {vmax:.4g} km/s{note}, 99.9th percentile {p999:.4g} km/s"))


def _check_mass(view, types, out):
    table = np.asarray(view.header.get("MassTable", [0.0] * 6), dtype=float)
    total = 0.0
    for t in types:
        grp = f"PartType{t}"
        has_ds = any(view.has(t, n) for n in ("Masses", "Mass"))
        if has_ds and t < len(table) and table[t] > 0:
            out.append(Check("mass", grp, "WARN",
                             f"MassTable[{t}] = {table[t]:g} and a Masses dataset are both set; "
                             "the dataset is used"))
        try:
            notes = []
            m = view._masses(t, "physical", view.count(t), None, notes)
        except ValueError as exc:
            out.append(Check("mass", grp, "FAIL", str(exc)))
            continue
        mt = float(m.sum())
        total += mt
        src = "Masses dataset" if has_ds else "MassTable"
        out.append(Check("mass", grp, "INFO",
                         f"{mt:.6e} Msun in {view.count(t):,} particles ({src})"
                         + (f"; {notes[0]}" if notes else "")))
    if view.is_cosmological and view.box_size("comoving") > 0:
        om = view.cosmo.get("omega0")
        if om:
            # Gadget-4's own constants (src/data/constants.h: GRAVITY 6.6738e-8,
            # HUBBLE 3.2407789e-18 h/s), so the check matches how the ICs were made;
            # astropy's G (6.6743e-8) alone would shift the ratio by 7.5e-5.
            rho_crit = 3.0 * (GADGET_HUBBLE * view.h) ** 2 / (8.0 * np.pi * GADGET_G)
            v_cm3 = (view.box_size("comoving") * KPC_CM) ** 3          # comoving volume
            expect = om * rho_crit * v_cm3 / MSUN_G
            ratio = total / expect
            st = "PASS" if abs(ratio - 1) <= MASS_BUDGET_TOLERANCE else "WARN"
            out.append(Check("mass", "all types", st,
                             f"total {total:.6e} Msun = {ratio:.5f} × Ω₀ ρ_crit V "
                             f"(Ω₀ = {om:g}, comoving box, Gadget-4 G and H₀)"))
            return
    out.append(Check("mass", "all types", "INFO", f"total {total:.6e} Msun"))


def _check_softening(view, types, out):
    any_found = False
    for t in types:
        eps, how = view.softening(t)
        if eps is None:
            continue
        any_found = True
        out.append(Check("softening", f"PartType{t}", "INFO",
                         f"ε = {eps:.4g} kpc physical ({how}); "
                         "Gadget's spline is Newtonian beyond 2.8 ε (Plummer-equivalent ε)"))
    if not any_found:
        out.append(Check("softening", "all types", "INFO", "no softening parameters in the file"))


# ── Entry point ────────────────────────────────────────────────────────────

def sanity_report(snapshot, types=None) -> SanityReport:
    """Run every check on a snapshot (path, VirtualSnapshotSet or SnapshotView)."""
    view = snapshot if isinstance(snapshot, SnapshotView) else SnapshotView(snapshot)
    files = view.files
    types = view.types_present() if types is None else [int(t) for t in types]
    cfg = _config(files)
    out: list = []
    _check_counts(view, files, types, out)
    _check_finite_and_sign(files, types, out)
    _check_ids(files, types, out)
    _check_positions(files, types, out)
    _check_box(view, files, types, cfg, out)
    _check_velocities(view, files, types, out)
    _check_mass(view, types, out)
    _check_softening(view, types, out)
    n = sum(view.count(t) for t in types)
    prov = Provenance(measurement="sanity_report", version=VERSION, files=list(files),
                      particle_types=types, n_used=n, n_total=n, frame="raw (code units)",
                      time=view.time_info(),
                      parameters={"velocity_outlier_factor": VELOCITY_OUTLIER_FACTOR,
                                  "mass_budget_tolerance": MASS_BUDGET_TOLERANCE})
    return SanityReport(out, prov)
