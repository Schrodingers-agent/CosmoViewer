"""
cli_import.py: turn what the user typed into the files to open.

Shared by CosmoViewer Immersive and the Dataset Explorer (the two copies are
identical; change both). Used for paths given at launch

    python main.py ~/runs/DM-L50-N128/output
    python density_explorer.py "snap_028_z000p000.*.hdf5" --dataset PartType0/Density

and for the Immersive console's `load` command.

A source is one of:
  * a folder: every snapshot in it (the app's own discovery decides which),
  * a file: one snapshot. Naming one piece of a multi-file snapshot
    (<base>.<i>.hdf5) opens all its pieces in that folder,
  * a glob pattern such as "snapshot_0*.hdf5", expanded here so it also
    works when the shell did not expand it (quoted, or on Windows),
  * a URL-style source "scheme:..." (agora:, sxs:, http(s):). Nothing is
    downloaded yet: each scheme reports that it is not available. A later
    importer registers a handler with register_source(); the handler fetches
    or streams the data to local files and returns their paths, so the apps
    open them like any other file.

Nothing here imports Qt, so `--list` can describe a source without a window.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

SNAPSHOT_SUFFIXES = (".hdf5", ".h5")

# scheme -> handler(spec) -> list of local paths. None = known but not built yet.
SOURCE_HANDLERS: Dict[str, Optional[Callable[[str], List[Path]]]] = {
    "agora": None,
    "sxs": None,
    "http": None,
    "https": None,
}

_PIECE_RE = re.compile(r"^(?P<base>.+)\.(?P<piece>\d+)\.(?P<ext>hdf5|h5)$")

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]+):(?!\\)")   # not C:\ on Windows


def register_source(scheme: str, handler: Callable[[str], List[Path]]) -> None:
    """Make 'scheme:...' sources importable. handler returns local file paths."""
    SOURCE_HANDLERS[scheme.lower()] = handler


@dataclass
class ImportPlan:
    folders: List[Path] = field(default_factory=list)
    files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.folders and not self.files

    def describe(self) -> str:
        lines = [f"folder: {p}" for p in self.folders]
        lines += [f"file:   {p}" for p in self.files]
        lines += [f"error:  {e}" for e in self.errors]
        return "\n".join(lines) or "nothing to open"


def _scheme(spec: str) -> Optional[str]:
    m = _SCHEME_RE.match(spec)
    if not m or len(m.group(1)) == 1:          # a drive letter, not a scheme
        return None
    return m.group(1).lower()


def resolve_sources(specs: List[str]) -> ImportPlan:
    """Sort what was typed into folders and files; report what cannot be opened."""
    plan = ImportPlan()
    seen = set()

    def add(kind: List[Path], p: Path):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            kind.append(p)

    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        scheme = _scheme(spec)
        if scheme is not None:
            if scheme not in SOURCE_HANDLERS:
                plan.errors.append(f"{spec}: unknown source type '{scheme}:'")
                continue
            handler = SOURCE_HANDLERS[scheme]
            if handler is None:
                plan.errors.append(f"{spec}: importing from '{scheme}:' is not available yet")
                continue
            try:
                for p in handler(spec):
                    add(plan.files, Path(p))
            except Exception as exc:
                plan.errors.append(f"{spec}: {type(exc).__name__}: {exc}")
            continue

        path = Path(os.path.expandvars(spec)).expanduser()
        if path.is_dir():
            add(plan.folders, path)
        elif path.is_file():
            for p in _with_sibling_pieces(path):
                add(plan.files, p)
        elif glob.has_magic(str(path)):
            matches = sorted(Path(m) for m in glob.glob(str(path)))
            snaps = [m for m in matches if m.is_file()]
            if not snaps:
                plan.errors.append(f"{spec}: no files match")
            for m in snaps:
                add(plan.files, m)
        else:
            plan.errors.append(f"{spec}: no such file or folder")

    return plan


def _with_sibling_pieces(path: Path) -> List[Path]:
    """<base>.<i>.hdf5 → every <base>.<j>.hdf5 beside it, in piece order."""
    m = _PIECE_RE.match(path.name)
    if not m:
        return [path]
    pieces = []
    for q in path.parent.iterdir():
        mq = _PIECE_RE.match(q.name)
        if mq and mq.group("base") == m.group("base") and mq.group("ext") == m.group("ext"):
            pieces.append((int(mq.group("piece")), q))
    return [q for _, q in sorted(pieces)] or [path]


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Arguments both apps accept. Each app adds its own with add_argument()."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("sources", nargs="*", metavar="SOURCE",
                   help="snapshot file, folder, glob pattern, or scheme:... source")
    p.add_argument("--list", action="store_true",
                   help="print what would be opened, then exit without a window")
    return p


def parse_args(parser: argparse.ArgumentParser, argv: List[str]):
    """Parse our arguments and leave the rest (Qt's own, such as -platform) alone."""
    args, rest = parser.parse_known_args(argv[1:])
    unknown = [a for a in rest if a.startswith("--")]
    if unknown:
        parser.error(f"unrecognised arguments: {' '.join(unknown)}")
    return args, [argv[0]] + rest
