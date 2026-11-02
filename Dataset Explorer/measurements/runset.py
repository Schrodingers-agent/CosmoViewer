"""
runset.py — one measurement over many snapshots and runs (the on-disk
results cache is in cache.py).

A Run is a folder of snapshots plus the parameters Gadget-4 writes into each
snapshot's Parameters group. A RunSet is a list of runs. RunSet.apply() calls
a measurement on every (run, snapshot) and returns a long-format ResultTable:
one row per (run, snapshot, quantity, element), so ensemble statistics are
plain group-by operations on it.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u

from .snapshot_view import SnapshotView

# snapshot_012.hdf5, snap_028_z000p000.0.hdf5, snapshot_005.3.hdf5
_SNAP_RE = re.compile(
    r"^(?P<stem>(?:snapshot|snap)_(?P<num>\d{3,})(?:_z\d+p\d+)?)(?:\.(?P<chunk>\d+))?\.(?:hdf5|h5)$")


_SNAPDIR_RE = re.compile(r"^snapdir_\d{3,}$")


def discover_snapshots(folder: str) -> list:
    """[(number, path to chunk 0 or the single file)] sorted by number.
    Catalogues (fof_tab_*, fof_subhalo_tab_*) and backups (bak-*) are not snapshots.

    Gadget-4 writes a snapshot in several files into its own folder snapdir_NNN/ (the
    NumFilesPerSnapshot > 1 layout), so those folders are searched too, and a folder that
    is itself a snapdir_NNN stands for the output folder above it."""
    folder = os.path.normpath(folder)
    if _SNAPDIR_RE.match(os.path.basename(folder)):
        folder = os.path.dirname(folder)
    out = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    places = [(folder, names)]
    for name in sorted(names):
        sub = os.path.join(folder, name)
        if _SNAPDIR_RE.match(name) and os.path.isdir(sub):
            try:
                places.append((sub, os.listdir(sub)))
            except OSError:
                continue
    for where, listing in places:
        for name in listing:
            m = _SNAP_RE.match(name)
            if not m:
                continue
            chunk = m.group("chunk")
            if chunk not in (None, "0"):
                continue
            out.setdefault(int(m.group("num")), os.path.join(where, name))
    return sorted(out.items())


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


@dataclass
class Run:
    name: str
    folder: str
    snapshots: list                      # [(number, path)]
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_folder(cls, folder: str, name: str | None = None, extra_parameters: dict | None = None):
        snaps = discover_snapshots(folder)
        params = {}
        if snaps:
            params = {k: _jsonable(v) for k, v in SnapshotView(snaps[0][1]).params.items()}
        if extra_parameters:
            params.update(extra_parameters)
        return cls(name=name or os.path.basename(os.path.normpath(folder)),
                   folder=folder, snapshots=snaps, parameters=params)


class RunSet:
    def __init__(self, runs: list):
        self.runs = list(runs)

    @classmethod
    def from_folder(cls, folder: str, parameters_csv: str | None = None) -> "RunSet":
        """A folder of snapshots is one run; otherwise each subfolder (or its
        output/ subfolder) that holds snapshots is a run. parameters_csv, if
        given, has a 'run' column and one column per extra parameter."""
        extra = {}
        if parameters_csv:
            with open(parameters_csv, newline="") as fh:
                for row in csv.DictReader(fh):
                    name = row.pop("run")
                    extra[name] = {k: _number(v) for k, v in row.items()}
        if discover_snapshots(folder):
            name = os.path.basename(os.path.normpath(folder))
            return cls([Run.from_folder(folder, name, extra.get(name))])
        runs = []
        for sub in sorted(os.listdir(folder)):
            path = os.path.join(folder, sub)
            if not os.path.isdir(path):
                continue
            for cand in (path, os.path.join(path, "output")):
                if discover_snapshots(cand):
                    runs.append(Run.from_folder(cand, sub, extra.get(sub)))
                    break
        return cls(runs)

    def varying_parameters(self) -> dict:
        """{parameter: [value per run]} for parameters that differ between runs."""
        keys = set().union(*(r.parameters.keys() for r in self.runs)) if self.runs else set()
        out = {}
        for k in sorted(keys):
            vals = [r.parameters.get(k) for r in self.runs]
            if len({repr(v) for v in vals}) > 1:
                out[k] = vals
        return out

    def apply(self, measure, snapshots=None, on_error: str = "record", cache=None,
              cache_name: str | None = None, cache_args: dict | None = None,
              **kwargs) -> "ResultTable":
        """measure(SnapshotView, **kwargs) -> Result, for every run and snapshot.
        snapshots: None for all, or a list of snapshot numbers to keep.
        on_error='record' keeps going and lists failures; 'raise' stops.
        cache: a ResultCache; then cache_name and cache_args (everything the
        result depends on besides the files) form the key."""
        rows, failures = [], []
        for run in self.runs:
            for num, path in run.snapshots:
                if snapshots is not None and num not in snapshots:
                    continue
                try:
                    if cache is not None:
                        view = SnapshotView(path)
                        res = cache.get_or_compute(
                            cache_name or getattr(measure, "__name__", "measure"), view.files,
                            {**(cache_args or {}), **kwargs},
                            lambda: measure(view, **kwargs))
                    else:
                        res = measure(SnapshotView(path), **kwargs)
                except Exception as exc:          # noqa: BLE001 - recorded, not swallowed
                    if on_error == "raise":
                        raise
                    failures.append({"run": run.name, "snapshot": num, "error": repr(exc)})
                    continue
                t = res.provenance.time
                base = {"run": run.name, "snapshot": num, "time_code": t.get("code"),
                        "a": t.get("a"), "time_Gyr": t.get("Gyr", t.get("age_Gyr")),
                        "is_science": res.is_science}
                for q, v in res.values.items():
                    unit = v.unit.to_string() if isinstance(v, u.Quantity) else ""
                    arr = np.atleast_1d(np.asarray(v.value if isinstance(v, u.Quantity) else v))
                    for i, x in enumerate(arr.ravel()):
                        rows.append({**base, "quantity": q, "index": i,
                                     "value": float(x), "unit": unit})
        return ResultTable(rows, failures)


def _number(s):
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s else f
    except (TypeError, ValueError):
        return s


class ResultTable:
    COLUMNS = ("run", "snapshot", "time_code", "a", "time_Gyr", "quantity",
               "index", "value", "unit", "is_science")

    def __init__(self, rows: list, failures: list | None = None):
        self.rows = rows
        self.failures = failures or []

    def __len__(self):
        return len(self.rows)

    def series(self, quantity: str, index: int = 0) -> dict:
        """{run: (snapshot numbers, time_code, values)} for one quantity element."""
        out = {}
        for r in self.rows:
            if r["quantity"] == quantity and r["index"] == index:
                out.setdefault(r["run"], []).append((r["snapshot"], r["time_code"], r["value"]))
        return {k: tuple(np.array(c) for c in zip(*sorted(v))) for k, v in out.items()}

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
