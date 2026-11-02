"""
Additional per-particle arrays from separate files (GadgetViewer: File >
Auxiliary data > Read additional data).

As in GadgetViewer, an array applies to one particle type, holds one value per
particle of that type in snapshot order, and may be split over files named like
Gadget snapshots (``name.0.hdf5``, ``name.1.hdf5`` ...). Formats: HDF5 (a named
dataset) and ASCII (a column, after a number of header lines). Fortran binary
records are not supported (deferred: no test data).

When the snapshot changes, the snapshot number in the file name is replaced
(``_NNN``, at least three digits), so the matching file is read for every
snapshot.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

_PIECE_RE = re.compile(r'^(?P<base>.+)\.(?P<i>\d+)\.(?P<ext>[^.]+)$')
_NUM_RE = re.compile(r'_(\d{3,})(?!\d)')


@dataclass
class AuxSpec:
    name: str                 # property name shown in the app
    path: str                 # file chosen for the snapshot it was set up on
    ptype: int
    fmt: str = 'hdf5'         # 'hdf5' or 'ascii'
    dataset: str = ''         # HDF5 dataset
    column: int = 0           # ASCII column (0-based)
    header_lines: int = 0
    integer: bool = False


def _pieces(path: str) -> List[Path]:
    """All pieces of a split file; a lone file is its own list."""
    p = Path(path)
    m = _PIECE_RE.match(p.name)
    if not m:
        return [p]
    out, i = [], 0
    while True:
        q = p.parent / f"{m.group('base')}.{i}.{m.group('ext')}"
        if not q.exists():
            break
        out.append(q)
        i += 1
    if not out:
        raise FileNotFoundError(f"no pieces found for {p.name} (expected {m.group('base')}.0.{m.group('ext')})")
    return out


def path_for_snapshot(spec_path: str, snapshot_path: str) -> str:
    """``spec_path`` with its snapshot number replaced by the snapshot's (group_catalogue.renumber_path)."""
    m = _NUM_RE.search(Path(snapshot_path).name)
    if not m:
        return spec_path
    from snapshot_io.group_catalogue import renumber_path
    s = renumber_path(spec_path, m.group(1))
    return spec_path if s is None else s


def read_aux_array(spec: AuxSpec, path: Optional[str] = None) -> np.ndarray:
    """All values (every particle of spec.ptype), concatenated over the pieces."""
    parts = []
    for p in _pieces(path or spec.path):
        if spec.fmt == 'hdf5':
            with h5py.File(p, 'r') as f:
                if spec.dataset not in f:
                    raise KeyError(f"{p.name} has no dataset '{spec.dataset}'")
                v = f[spec.dataset][:]
            if v.ndim != 1:
                raise ValueError(f"{p.name}:{spec.dataset} is not one value per particle (shape {v.shape})")
        elif spec.fmt == 'ascii':
            v = np.loadtxt(p, skiprows=spec.header_lines, usecols=(spec.column,), ndmin=1,
                           dtype=np.int64 if spec.integer else np.float64)
        else:
            raise ValueError(f"unsupported format {spec.fmt!r}")
        parts.append(np.asarray(v))
    out = np.concatenate(parts) if parts else np.empty(0)
    return out.astype(np.int64) if spec.integer else out.astype(np.float64)


def attach_to_sample(snap: dict, spec: AuxSpec, values: np.ndarray) -> np.ndarray:
    """
    Values for the display sample (NaN for other types). The array must have
    exactly one value per particle of spec.ptype in the snapshot; otherwise
    nothing is attached, because values would land on the wrong particles.
    """
    from snapshot_io.properties import sample_rows
    from snapshot_io.group_catalogue import global_rows
    files = list(snap.get('files') or [snap['filepath']])
    n_type = 0
    for fn in files:
        with h5py.File(fn, 'r') as f:
            g = f.get(f'PartType{spec.ptype}')
            if g is not None and 'Coordinates' in g:
                n_type += g['Coordinates'].shape[0]
    if len(values) != n_type:
        raise ValueError(f"{spec.name}: {len(values):,} values but the snapshot has {n_type:,} "
                         f"particles of type {spec.ptype}")
    out = np.full(len(snap['c']), np.nan)
    pos = 0
    for t, rows in global_rows(files, sample_rows(snap)):
        if t == spec.ptype:
            out[pos:pos + len(rows)] = values[rows]
        pos += len(rows)
    return out
