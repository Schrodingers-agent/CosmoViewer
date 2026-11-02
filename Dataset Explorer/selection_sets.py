"""
Selection-set files shared by Immersive Mode, the Dataset Explorer and scripts.

A selection set is a named list of ParticleIDs with a colour (GadgetViewer's
"Select particles" sets). Saving one as a file lets the other app, or a
script, use exactly the same particles: Immersive Mode's "Send to Explorer"
writes one, and the Explorer lists every set next to the open snapshot under
"Sets" and can restrict its analyses to it.

Format (HDF5, one set per file):

    /ParticleIDs                uint64, sorted, unique
    attrs: format = "cosmoviewer-selection-set", version = 2, name, colour
           ("#rrggbb"), created (ISO time), made_in, source_snapshot (file
           name), description (how the set was made), n
    version 2 adds what the set *means* (a "population", see below):
           kind, definition (JSON), defined_at (JSON), complete
           ("yes" / "no" / "unknown"), n_expected (-1 when unknown),
           incomplete_reason ("" when unknown or not incomplete)

Version-1 files are still read: they become kind "ids" with completeness unknown.
Version-2 files written before incomplete_reason existed read it as unknown.

Populations
-----------
A set is always stored as a fixed list of ParticleIDs (so tracking, files and
scripts work the same for every kind), but it also records how it was made,
so its meaning can never change silently when it moves between views or
snapshots:

    kind        "ids"     an explicit list of particles (ID list or file, picks)
                "group"   the members of one halo-finder group: definition has
                          finder ("fof" or "catalogue"), group (1-based),
                          subhalo (bool), and catalogue_file (or stored_in
                          "snapshot" for membership stored per particle, as
                          EAGLE does) or FoF b, linking_length, min_size,
                          link_types, convention
                "region"  particles inside a region at one snapshot: shape
                          ("sphere"), centre [x, y, z], radius, units, periodic,
                          wraps (axes it crosses the box along), types, scope ("full" or "sample")
                "query"   particles that matched criteria at one snapshot:
                          criteria = list of {"kind": "property" | "region" |
                          "group" | "ids" | "bin" | "sanity" | "top", ...}, all
                          ANDed. "property" has field, lo, hi (None = open),
                          units, scope; "top" is a particle chosen from a list
                          of the highest values: field, n (how many were
                          listed), rank (1 = highest), scope
    defined_at  the snapshot the definition was evaluated on: file, index,
                time and redshift where known. A population used at another
                snapshot is "the particles of <definition> at <defined_at>,
                traced by ID", never the definition re-evaluated there.
    complete    True when the definition was evaluated on every particle of
                the snapshot and the set holds everything it describes; False
                when the set may hold only part of what the definition
                describes; None when unknown.
    incomplete_reason
                why a set is incomplete (only with complete False):
                "display_sample" (a criterion was evaluated on the display
                sample only), "missing_in_file" (members the definition names,
                such as a catalogue group's, are not in the snapshot files),
                or None (not known).
    n_expected  how many particles the definition has (a group's length), when
                known; with complete False it says how much is missing.

describe() turns these into one line for the interface, and
completeness_warning() into the sentence an analysis must show when a set is
known to be incomplete.

Where: ``<snapshot folder>/cosmoviewer_sets/<name>.hdf5``. If that folder
cannot be written (read-only simulation output), ``~/.cosmoviewer/sets/<key>``
is used instead, where key identifies the snapshot folder; both places are
searched when listing. Nothing in the simulation output itself is changed.

This file is kept identical in both apps ("Cosmoviewer Immersive" and
"Dataset Explorer"), like cli_import.py; a test checks that.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np

FORMAT = "cosmoviewer-selection-set"
VERSION = 2
KINDS = ("ids", "group", "region", "query")
INCOMPLETE_REASONS = ("display_sample", "missing_in_file")
SETS_FOLDER = "cosmoviewer_sets"
_HOME = Path(os.environ.get("COSMOVIEWER_HOME", Path.home() / ".cosmoviewer"))
FALLBACK_ROOT = _HOME / "sets"
INSTANCES_DIR = _HOME / "explorer_instances"


@dataclass
class SelectionSet:
    ids: np.ndarray
    name: str
    colour: str = "#ffffff"
    created: str = ""
    made_in: str = ""
    source_snapshot: str = ""
    description: str = ""
    path: str = ""
    # what the set means (version 2); see "Populations" above
    kind: str = "ids"
    definition: dict = field(default_factory=dict)
    defined_at: dict = field(default_factory=dict)
    complete: Optional[bool] = None
    n_expected: Optional[int] = None
    incomplete_reason: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.ids)

    def describe(self) -> str:
        return describe(self.kind, self.definition, self.defined_at, self.complete, self.n, self.n_expected,
                        self.incomplete_reason)

    def completeness_warning(self) -> Optional[str]:
        return completeness_warning(self.kind, self.complete, self.n, self.n_expected, self.incomplete_reason)


# ----------------------------------------------------------------------------- describing a population

def _num(v) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


_TYPE_NAMES = {0: "gas", 1: "DM", 2: "disk", 3: "bulge", 4: "stars", 5: "BH"}


def _types_text(types) -> str:
    return "+".join(_TYPE_NAMES.get(int(t), f"type {t}") for t in types) if types else ""


def _bound(v) -> Optional[float]:
    """A finite range bound, or None for an open one (None, +-inf or NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _range_text(c: dict) -> str:
    """'lo <= field <= hi unit', leaving out an open bound; a unit of several words in brackets."""
    u = str(c.get("units") or "")
    unit = "" if not u else f" [{u}]" if " " in u else f" {u}"
    lo, hi, name = _bound(c.get("lo")), _bound(c.get("hi")), c.get("field", "?")
    if lo is None and hi is None:
        return f"any value of {name}"
    if lo is None:
        return f"{name} <= {_num(hi)}{unit}"
    if hi is None:
        return f"{name} >= {_num(lo)}{unit}"
    return f"{_num(lo)} <= {name} <= {_num(hi)}{unit}"


def _top_text(c: dict) -> str:
    """A particle chosen from a list of the highest values (Immersive Mode's "List top")."""
    name, rank = c.get("field", "?"), c.get("rank")
    among = " of the drawn particles" if c.get("scope") == "sample" else ""
    if rank is None:
        return f"the {c.get('n', '?')} highest {name}{among}"
    which = "highest" if int(rank) == 1 else f"{_ordinal(rank)} highest"
    return f"{which} {name}{among}" + (f" (from a list of {c['n']})" if c.get("n") else "")


def _definition_text(kind: str, d: dict) -> str:
    d = d or {}
    if kind == "group":
        what = "subhalo" if d.get("subhalo") else "group"
        if d.get("finder") == "catalogue":
            src = (f" of {d['catalogue_file']}" if d.get("catalogue_file")
                   else " (membership stored in the snapshot)" if d.get("stored_in") == "snapshot"
                   else " of the catalogue")
            return f"catalogue {what} {d.get('group', '?')}{src}"
        params = []
        if d.get("b") is not None:
            params.append(f"b = {_num(d['b'])}")
        elif d.get("linking_length") is not None:
            params.append(f"l = {_num(d['linking_length'])}" + (f" {d['units']}" if d.get("units") else ""))
        if d.get("min_size") is not None:
            params.append(f"N_min = {d['min_size']}")
        if d.get("link_types"):
            params.append(_types_text(d["link_types"]))
        return f"FoF group {d.get('group', '?')}" + (f" ({', '.join(params)})" if params else "")
    if kind == "region":
        c = d.get("centre")
        about = f" about ({', '.join(_num(x) for x in c)})" if c is not None else ""
        unit = f" {d['units']}" if d.get("units") else ""
        wraps = f", wraps {''.join(d['wraps'])} (periodic)" if d.get("wraps") else ""
        return f"sphere r = {_num(d.get('radius', '?'))}{unit}{about}{wraps}"
    if kind == "query":
        parts = []
        for c in d.get("criteria", []):
            k = c.get("kind")
            if k == "property":
                parts.append(_range_text(c))
            elif k == "top":
                parts.append(_top_text(c))
            elif k in ("region", "group"):
                parts.append(_definition_text(k, c))
            elif k == "bin":
                unit = f" {c['units']}" if c.get("units") else ""
                parts.append(f"{c.get('field', '?')} in [{_num(c.get('lo'))}, {_num(c.get('hi'))}]{unit}")
            elif k == "sanity":
                parts.append(f"sanity check '{c.get('check', '?')}'")
            elif k == "ids":
                parts.append("listed IDs")
            else:
                parts.append(str(c.get("text", k)))
        return " and ".join(parts) if parts else "criteria"
    # ids
    if d.get("edited_from"):
        return f"particle IDs, edited ({d.get('operation', 'changed')}) from: {d['edited_from']}"
    if d.get("source") == "file" and d.get("file"):
        return f"particle IDs from {Path(str(d['file'])).name}"
    if d.get("source") == "pick":
        return "picked particles"
    return "particle IDs"


_INCOMPLETE_TEXT = {"display_sample": "display sample only", "missing_in_file": "members missing from the file"}


def describe(kind: str, definition: Optional[dict], defined_at: Optional[dict], complete: Optional[bool],
             n: int, n_expected: Optional[int] = None, incomplete_reason: Optional[str] = None) -> str:
    """One line saying what a population is, e.g.
    "FoF group 1 (b = 0.2, N_min = 32, DM) at snapshot_002.hdf5 · complete · 34,120 particles"."""
    text = _definition_text(kind or "ids", definition or {})
    at = (defined_at or {}).get("file")
    if at and kind != "ids":
        text += f" at {at}"
    if complete is True:
        status = "complete"
    elif complete is False:
        status = (_INCOMPLETE_TEXT.get(incomplete_reason, "incomplete")
                  + (f": {n:,} of {n_expected:,}" if n_expected else ""))
    else:
        status = ""
    count = f"{n:,} particle{'' if n == 1 else 's'}" if not (complete is False and n_expected) else ""
    return " · ".join(x for x in (text, status, count) if x)


def completeness_warning(kind: str, complete: Optional[bool], n: int, n_expected: Optional[int] = None,
                         incomplete_reason: Optional[str] = None) -> Optional[str]:
    """The sentence an analysis must show for a set known to be incomplete, else None.
    It says why when that is known (incomplete_reason)."""
    if complete is not False:
        return None
    what = {"group": "group", "region": "region", "query": "selection"}.get(kind, "selection")
    if incomplete_reason == "display_sample":
        of = f" {n:,} of its {n_expected:,} particles" if n_expected else f" {n:,} particles"
        return (f"This set was taken from the display sample, not the whole snapshot: it holds{of}. "
                f"Results describe these particles only, not the whole {what}.")
    if incomplete_reason == "missing_in_file":
        held = (f"only {n:,} of this {what}'s {n_expected:,} members" if n_expected
                else f"only {n:,} of this {what}'s members")
        return (f"The snapshot files hold {held}; results describe these particles only, "
                f"not the whole {what}.")
    of = f"{n:,} of its {n_expected:,} particles" if n_expected else f"{n:,} particles"
    return (f"This set is incomplete: it holds {of}; results describe these particles only, "
            f"not the whole {what}.")


def check_incomplete_reason(complete: Optional[bool], reason: Optional[str]) -> Optional[str]:
    """``reason`` itself when it is valid: None, or one of INCOMPLETE_REASONS for a set that is
    incomplete (complete False). Anything else is refused, so a set never says it is complete and
    incomplete at once."""
    if reason is None:
        return None
    if reason not in INCOMPLETE_REASONS:
        raise ValueError(f"unknown incomplete_reason {reason!r}; expected one of {', '.join(INCOMPLETE_REASONS)}")
    if complete is not False:
        raise ValueError("incomplete_reason applies only to a set that is incomplete (complete=False)")
    return reason


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return s[:80] or "set"


def _folder_key(folder: Path) -> str:
    folder = Path(folder).resolve()
    return f"{_safe(folder.name)}-{hashlib.sha1(str(folder).encode()).hexdigest()[:10]}"


def candidate_dirs(snapshot_path: str) -> List[Path]:
    """Folders searched for the sets of the run that ``snapshot_path`` belongs to."""
    folder = Path(snapshot_path).resolve().parent
    return [folder / SETS_FOLDER, FALLBACK_ROOT / _folder_key(folder)]


def _writable_dir(snapshot_path: str) -> Path:
    for d in candidate_dirs(snapshot_path):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("")
            probe.unlink()
            return d
        except OSError:
            continue
    raise OSError(f"no writable folder for sets of {snapshot_path}")


def _complete_text(complete: Optional[bool]) -> str:
    return "yes" if complete is True else "no" if complete is False else "unknown"


def write_set(snapshot_path: str, ids, name: str, colour: str = "#ffffff", made_in: str = "",
              description: str = "", path: Optional[str] = None, kind: str = "ids",
              definition: Optional[dict] = None, defined_at: Optional[dict] = None,
              complete: Optional[bool] = None, n_expected: Optional[int] = None,
              incomplete_reason: Optional[str] = None) -> str:
    """Write a set for the run of ``snapshot_path``; returns the file written. Same name replaces it.

    kind, definition, defined_at, complete, n_expected and incomplete_reason say what the set
    means (see the module docstring); description defaults to describe() of them."""
    if kind not in KINDS:
        raise ValueError(f"unknown set kind {kind!r}; expected one of {', '.join(KINDS)}")
    check_incomplete_reason(complete, incomplete_reason)
    ids = np.unique(np.asarray(list(ids) if isinstance(ids, (set, frozenset)) else ids, dtype=np.uint64))
    target = Path(path) if path else _writable_dir(snapshot_path) / f"{_safe(name)}.hdf5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not description:
        description = describe(kind, definition, defined_at, complete, len(ids), n_expected, incomplete_reason)
    tmp = target.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("ParticleIDs", data=ids, compression="gzip" if len(ids) > 100_000 else None)
        f.attrs.update(dict(format=FORMAT, version=VERSION, name=name, colour=colour,
                            created=_dt.datetime.now().isoformat(timespec="seconds"),
                            made_in=made_in, source_snapshot=Path(snapshot_path).name,
                            description=description, n=len(ids),
                            kind=kind, definition=json.dumps(definition or {}, default=_json_default),
                            defined_at=json.dumps(defined_at or {}, default=_json_default),
                            complete=_complete_text(complete),
                            n_expected=-1 if n_expected is None else int(n_expected),
                            incomplete_reason=incomplete_reason or ""))
    os.replace(tmp, target)             # readers never see a half-written file
    return str(target)


def _json_default(v):
    """numpy scalars and arrays in definitions (centres, group numbers) as plain JSON."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"{type(v).__name__} is not JSON serialisable")


def is_set_file(path: str) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("format", "") == FORMAT and "ParticleIDs" in f
    except (OSError, KeyError):
        return False


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _json_attr(a, key) -> dict:
    try:
        v = json.loads(_text(a.get(key, "{}")) or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def read_set(path: str) -> SelectionSet:
    """Read a set file of either version; version 1 files become kind "ids", completeness unknown."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format", "") != FORMAT:
            raise ValueError(f"{Path(path).name} is not a CosmoViewer selection set")
        a = f.attrs
        kind = _text(a.get("kind", "ids")) or "ids"
        complete = {"yes": True, "no": False}.get(_text(a.get("complete", "unknown")))
        n_exp = int(a.get("n_expected", -1)) if "n_expected" in a else -1
        reason = _text(a.get("incomplete_reason", "")) or None
        if reason not in INCOMPLETE_REASONS or complete is not False:
            reason = None                   # older files, or a reason this version does not know
        return SelectionSet(np.asarray(f["ParticleIDs"][:], dtype=np.uint64), _text(a.get("name", Path(path).stem)),
                            _text(a.get("colour", "#ffffff")), _text(a.get("created", "")),
                            _text(a.get("made_in", "")), _text(a.get("source_snapshot", "")),
                            _text(a.get("description", "")), str(path),
                            kind=kind if kind in KINDS else "ids",
                            definition=_json_attr(a, "definition"), defined_at=_json_attr(a, "defined_at"),
                            complete=complete, n_expected=None if n_exp < 0 else n_exp,
                            incomplete_reason=reason)


def list_sets(snapshot_path: str) -> List[str]:
    """Set files of the run, newest first."""
    out = []
    for d in candidate_dirs(snapshot_path):
        if d.is_dir():
            out += [p for p in d.glob("*.hdf5") if is_set_file(str(p))]
    return [str(p) for p in sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)]


# ----------------------------------------------------------------------------- open Explorer windows

def register_instance(pid: int, snapshot_files: Sequence[str]) -> Optional[Path]:
    """The Explorer records which snapshot it has open, so Immersive Mode can send it sets."""
    try:
        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        p = INSTANCES_DIR / f"{pid}.json"
        p.write_text(json.dumps({"pid": pid, "files": [str(Path(f).resolve()) for f in snapshot_files]}))
        return p
    except OSError:
        return None


def unregister_instance(pid: int):
    try:
        (INSTANCES_DIR / f"{pid}.json").unlink()
    except OSError:
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def explorer_for(snapshot_path: str) -> Optional[int]:
    """PID of a running Explorer that has a snapshot of this run's folder open, if any."""
    folder = Path(snapshot_path).resolve().parent
    if not INSTANCES_DIR.is_dir():
        return None
    for p in INSTANCES_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        pid = int(d.get("pid", -1))
        if not _alive(pid):
            try:
                p.unlink()               # left behind by an Explorer that crashed
            except OSError:
                pass
            continue
        if any(Path(f).parent == folder for f in d.get("files", [])):
            return pid
    return None
