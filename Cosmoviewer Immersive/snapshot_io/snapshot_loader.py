"""
Unified Multi-Format Snapshot Loader (HDF5 + Gadget Binary Type 1 & Type 2).

HDF5 snapshots (Gadget-2/3/4, EAGLE, SWIFT) are read here with h5py: only
Coordinates, velocities, masses and ParticleIDs, decimated at read time.
A multi-file snapshot (<base>.<i>.hdf5, NumFilesPerSnapshot > 1) is one
snapshot, read piece by piece in parallel. Gadget binary snapshots (SnapFormat
1 and 2) are read whole by cosmo_io_engine.GadgetBinaryReader.
"""

import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

from pygadgetviewer_path import ensure_on_path, missing_message

ensure_on_path()
try:
    from cosmo_io_engine import SnapshotHarvester, GadgetBinaryReader, GadgetHDF5Reader
    HAS_PYGADGETVIEWER = True
    IMPORT_ERROR = ""
except ImportError:
    try:
        from pygadgetviewer import SnapshotHarvester, GadgetBinaryReader, GadgetHDF5Reader  # type: ignore[import-not-found]
        HAS_PYGADGETVIEWER = True
        IMPORT_ERROR = ""
    except ImportError as e:
        HAS_PYGADGETVIEWER = False
        IMPORT_ERROR = f"{missing_message()}\n({e})"
        print(f"WARNING: {IMPORT_ERROR}")

import h5py
from PyQt6.QtCore import QThread, pyqtSignal

from lod.store import id_uniforms, row_uniforms

LOD_SEED = 0                  # the LOD store's default seed: small and big snapshots sample alike

HDF5_SUFFIXES = ('.hdf5', '.h5')
BIG_SNAPSHOT = 8_000_000      # particles; from here the display sample comes from the LOD store
_PIECE_RE = re.compile(r'^(?P<base>.+)\.(?P<piece>\d+)\.(?P<ext>hdf5|h5)$')

# Gadget names first, then EAGLE / OWLS names.
VELOCITY_NAMES = ('Velocities', 'Velocity')
MASS_NAMES = ('Masses', 'Mass')


def _is_hdf5(path) -> bool:
    return str(path).endswith(HDF5_SUFFIXES)


def _scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.size == 1 else value


def _read_hdf5_parameters(filepath: str) -> dict:
    """Gadget-4 stores the unit system in the Parameters group, EAGLE in Units."""
    out = {}
    try:
        with h5py.File(filepath, 'r') as f:
            for name in ('Parameters', 'Units'):
                if name in f:
                    out.update({k: _scalar(v) for k, v in f[name].attrs.items()})
    except OSError:
        pass
    return out


def _mass_table_value(header_attrs: dict, ptype: int) -> float:
    table = header_attrs.get('MassTable')
    if table is None:
        return 0.0
    table = np.atleast_1d(table)
    return float(table[ptype]) if ptype < len(table) else 0.0


def _num_files(header: dict) -> int:
    return int(np.atleast_1d(header.get('NumFilesPerSnapshot', 1))[0])


def _first(group, names):
    for n in names:
        if n in group and isinstance(group[n], h5py.Dataset):
            return group[n]
    return None


# ----------------------------------------------------------------------------
# Grouping files into snapshots
# ----------------------------------------------------------------------------

def group_snapshot_files(files: List[Path]) -> Tuple[List[List[Path]], List[str]]:
    """
    Group the files of multi-file HDF5 snapshots together.

    Files named <base>.<i>.hdf5 whose header says NumFilesPerSnapshot > 1 are
    one snapshot. Everything else is a snapshot of its own. Returns the
    groups (pieces ordered by i) and the error messages for incomplete sets;
    an incomplete set is not loaded, since dropping a file drops particles.
    """
    groups: Dict[tuple, Dict[int, Path]] = {}
    expected: Dict[tuple, int] = {}
    order: List[tuple] = []
    for f in files:
        p = Path(f)
        m = _PIECE_RE.match(p.name)
        key = (str(p),)
        if m and _is_hdf5(p):
            try:
                with h5py.File(p, 'r') as h:
                    nfiles = _num_files(dict(h['Header'].attrs)) if 'Header' in h else 1
            except OSError:
                nfiles = 1          # unreadable: let the loader report it on it's own
            if nfiles > 1:
                key = (str(p.parent), m.group('base'), m.group('ext'))
                expected[key] = nfiles
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][int(m.group('piece')) if len(key) == 3 else 0] = p

    result, errors = [], []
    for key in order:
        pieces = groups[key]
        if len(key) == 3:
            missing = [i for i in range(expected[key]) if i not in pieces]
            if missing:
                name = f"{key[1]}.*.{key[2]}"
                errors.append(f"{name}: missing {len(missing)} of {expected[key]} files "
                              f"(e.g. piece {missing[0]}); not loaded")
                continue
        result.append([pieces[i] for i in sorted(pieces)])
    return result, errors


# ----------------------------------------------------------------------------
# HDF5 reading
# ----------------------------------------------------------------------------

def plan_hdf5_snapshot(files: List[Path], max_particles: int) -> dict:
    """
    Decide which rows to read from every piece.

    Keeps every particle when the snapshot has at most max_particles. Otherwise
    keeps the particles whose u (a hash of the ParticleID, lod/store.py) is below
    max_particles / N_total: about max_particles of them, a uniform sample, and the
    same particles in every snapshot of a run however the files order them (a
    stride over rows picked different particles each snapshot, so movies flickered).
    Each (piece, type) gets ('all',) or ('hash', fraction, piece).
    """
    counts = []         # counts[i][t] = rows of type t in piece i
    for f in files:
        with h5py.File(f, 'r') as h:
            n = np.zeros(6, dtype=np.int64)
            for t in range(6):
                g = h.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    n[t] = g['Coordinates'].shape[0]
            counts.append(n)
    counts = np.array(counts).reshape(len(files), 6)
    n_total = int(counts.sum())
    fraction = max_particles / n_total if n_total > max_particles else 1.0
    pieces = [{t: ('all',) if fraction >= 1.0 else ('hash', fraction, i)
               for t in range(6) if counts[i, t]} for i in range(len(files))]
    return {'files': [str(f) for f in files], 'pieces': pieces, 'n_total': n_total,
            'step': 1.0 / fraction, 'fraction': fraction, 'counts': counts}


def read_hdf5_piece(args: Tuple[str, Dict[int, Tuple[int, int]], dict]) -> dict:
    """Read the planned rows of one file. Returns {ptype: {'c','v','m','id'}} or {'error'}."""
    filepath, plan, header = args
    try:
        out = {}
        with h5py.File(filepath, 'r') as f:
            for t, spec in sorted(plan.items()):
                g = f[f'PartType{t}']
                n_rows = g['Coordinates'].shape[0]
                idd = g['ParticleIDs'] if 'ParticleIDs' in g else None
                if spec[0] == 'all':
                    sel = np.s_[:]
                    rows = np.arange(n_rows, dtype=np.int64)
                else:
                    _, fraction, piece = spec  # type: ignore[misc]  # spec is ('hash', float, int) here
                    u, all_ids = row_uniforms(g, LOD_SEED, piece, t, 0, n_rows)
                    rows = np.flatnonzero(u < fraction)
                    sel = rows
                    if len(rows) == 0:
                        continue
                c = _take(g['Coordinates'], sel)
                vd = _first(g, VELOCITY_NAMES)
                v = _take(vd, sel) if vd is not None else np.zeros_like(c)
                md = _first(g, MASS_NAMES)
                if md is not None:
                    m = _take(md, sel)
                else:
                    # Gadget omits the mass array when every particle of a type has
                    # the same mass; the value is then in Header/MassTable. Everyone who
                    # reads Gadget files learns this exactly once.
                    m_table = _mass_table_value(header, t)
                    m = np.full(len(c), m_table if m_table > 0 else 1.0, dtype=np.float32)
                ids = _take(idd, sel) if idd is not None else rows.astype(np.uint64)
                out[t] = {'c': c, 'v': v, 'm': m, 'id': ids, 'rows': rows}
        return {'data': out, 'filepath': filepath}
    except Exception as e:
        return {'error': f"{type(e).__name__}: {e}", 'filepath': filepath}


def _take(dataset, sel):
    """Rows of a dataset: a slice, or sorted row indices (read whole and indexed: these are
    small snapshots, and a full read beats a fancy read of a large share of the rows)."""
    if isinstance(sel, slice):
        return dataset[sel]
    return dataset[:][sel]


def _field_scaling(filepath: str) -> dict:
    """Cosmological scaling attributes of Coordinates (Gadget-4 and EAGLE names)."""
    keys = ('a_scaling', 'h_scaling', 'to_cgs', 'aexp-scale-exponent', 'h-scale-exponent',
            'CGSConversionFactor')
    try:
        with h5py.File(filepath, 'r') as f:
            for t in range(6):
                g = f.get(f'PartType{t}')
                if g is not None and 'Coordinates' in g:
                    a = g['Coordinates'].attrs
                    return {k: _scalar(a[k]) for k in keys if k in a}
    except OSError:
        pass
    return {}


def assemble_hdf5_snapshot(plan: dict, piece_results: List[dict], idx: int) -> dict:
    """Join piece results (ordered like plan['files']) into one snapshot dict."""
    first = plan['files'][0]
    with h5py.File(first, 'r') as f:
        header = {k: _scalar(v) if k not in ('MassTable', 'NumPart_ThisFile', 'NumPart_Total',
                                             'NumPart_Total_HighWord') else v
                  for k, v in f['Header'].attrs.items()}
    c, v, t, m, ids = [], [], [], [], []
    for ptype in range(6):
        for res in piece_results:
            d = res['data'].get(ptype)
            if d is None or len(d['c']) == 0:
                continue
            c.append(d['c'])
            v.append(d['v'])
            m.append(d['m'])
            ids.append(d['id'])
            t.append(np.full(len(d['c']), ptype, dtype=np.uint8))
    if not c:
        return {'error': "no particles with Coordinates found", 'filepath': first}

    unit_source = dict(header)
    unit_source.update(_read_hdf5_parameters(first))
    rows = {(i, t): d['rows'] for i, res in enumerate(piece_results)
            for t, d in res['data'].items() if len(d['rows'])}
    return {
        'rows': rows,
        'c': np.vstack(c),
        'v': np.vstack(v),
        't': np.hstack(t),
        'm': np.hstack(m),
        'id': np.hstack(ids),
        'time': float(header.get('Time', idx * 1.0)),
        'header': header,
        'unit_source': unit_source,
        'idx': idx,
        'filepath': first,
        'files': list(plan['files']),
        'n_total': plan['n_total'],
        'step': plan['step'],
        'coord_scaling': _field_scaling(first),
    }


def snapshot_from_store(files: List[str], idx: int, max_particles: int,
                        progress=None, cancelled=lambda: False) -> Optional[dict]:
    """
    A big snapshot's display sample, taken from its LOD store (lod/store.py).

    Reading every step-th row of a compressed multi-gigabyte snapshot
    decompresses all of it anyway (32 s for the 850M-particle EAGLE snapshot),
    so big snapshots build the LOD store once instead (cached on disk, opened
    in about a second afterwards) and take a uniform random sample from it.
    """
    from lod.store import build_store
    from lod.view import base_sample
    files = [str(f) for f in files]
    store = build_store(files, progress=progress, cancelled=cancelled)
    if store is None:
        return None
    view = base_sample(store, max_particles)
    with h5py.File(files[0], 'r') as f:
        header = {k: _scalar(v) if k not in ('MassTable', 'NumPart_ThisFile', 'NumPart_Total',
                                             'NumPart_Total_HighWord') else v
                  for k, v in f['Header'].attrs.items()}
    unit_source = dict(header)
    unit_source.update(_read_hdf5_parameters(files[0]))
    n = len(view['t'])
    view.update({
        'time': float(header.get('Time', idx * 1.0)),
        'header': header,
        'unit_source': unit_source,
        'idx': idx,
        'filepath': files[0],
        'files': files,
        'n_total': store.layout['n_total'],
        'step': store.layout['n_total'] / max(n, 1),     # average; the sample is random, see 'rows'
        'coord_scaling': _field_scaling(files[0]),
    })
    return view


def load_hdf5_snapshot(files: List[Path], idx: int, max_particles: int) -> dict:
    """Serial version: plan, read every piece, assemble."""
    try:
        files = [str(f) for f in files]
        plan = plan_hdf5_snapshot(files, max_particles)
        with h5py.File(files[0], 'r') as f:
            header = dict(f['Header'].attrs) if 'Header' in f else {}
        results = [read_hdf5_piece((fp, pl, header)) for fp, pl in zip(plan['files'], plan['pieces'])]
        for r in results:
            if 'error' in r:
                return {'error': f"{Path(r['filepath']).name}: {r['error']}", 'filepath': files[0]}
        return assemble_hdf5_snapshot(plan, results, idx)
    except Exception as e:
        return {'error': f"{type(e).__name__}: {e}", 'filepath': str(files[0])}


# ----------------------------------------------------------------------------
# Single-file entry point (binary via cosmo_io_engine, HDF5 via h5py)
# ----------------------------------------------------------------------------

def load_single_snapshot_multiformat(args: Tuple[str, int, int]) -> dict:
    """Returns the snapshot dict, or {'error': message, 'filepath': ...} on failure."""
    filepath, idx, max_particles = args
    filepath_str = str(filepath)

    if _is_hdf5(filepath_str):
        return load_hdf5_snapshot([filepath_str], idx, max_particles)

    if not HAS_PYGADGETVIEWER:
        return {'error': IMPORT_ERROR, 'filepath': filepath_str}

    try:
        c_list, v_list, t_list, m_list, id_list = [], [], [], [], []
        header_attrs = {}

        harvester = SnapshotHarvester(filepath_str)
        raw_data = harvester.harvest()

        if hasattr(harvester.reader, 'header'):
            hdr = harvester.reader.header
            if isinstance(hdr, dict):
                header_attrs = hdr
            else:
                # The binary header's own values, under the HDF5 names the rest of the app reads.
                header_attrs = {
                    'Time': float(hdr.time),
                    'Redshift': float(hdr.redshift),
                    'BoxSize': float(hdr.boxsize),
                    'HubbleParam': float(hdr.hubble_param),
                    'Omega0': float(hdr.omega0),
                    'OmegaLambda': float(hdr.omega_lambda),
                    'MassTable': np.asarray(hdr.massarr, dtype=np.float64),
                    'NumPart_Total': np.asarray(hdr.total_particles, dtype=np.uint64),
                    'NumFilesPerSnapshot': int(hdr.num_files),
                }

        for ptype in range(6):
            if ptype in raw_data and raw_data[ptype]:
                sp_data = raw_data[ptype]
                if 'Coordinates' in sp_data:
                    c = sp_data['Coordinates']
                    v = sp_data.get('Velocities', np.zeros_like(c))
                    t = np.full(len(c), ptype, dtype=np.uint8)
                    m = sp_data.get('Masses')
                    if m is None:
                        m_table = _mass_table_value(header_attrs, ptype)
                        m = (np.full(len(c), m_table, dtype=np.float32) if m_table > 0
                             else np.ones(len(c), dtype=np.float32))
                    p_id = sp_data.get('ParticleIDs', np.arange(len(c), dtype=np.uint64))

                    c_list.append(c)
                    v_list.append(v)
                    t_list.append(t)
                    m_list.append(m)
                    id_list.append(p_id)

        if not c_list:
            return {'error': "no particles with Coordinates found", 'filepath': filepath_str}

        c_all = np.vstack(c_list)
        v_all = np.vstack(v_list)
        t_all = np.hstack(t_list)
        m_all = np.hstack(m_list)
        id_all = np.hstack(id_list)

        if len(c_all) > max_particles:      # same rule as HDF5: u of the ParticleID
            keep = id_uniforms(id_all, LOD_SEED) < max_particles / len(c_all)
            c_all = c_all[keep]
            v_all = v_all[keep]
            t_all = t_all[keep]
            m_all = m_all[keep]
            id_all = id_all[keep]

        return {
            'c': c_all,
            'v': v_all,
            't': t_all,
            'm': m_all,
            'id': id_all,
            'time': float(header_attrs.get('Time', idx * 1.0)),
            'header': header_attrs,
            'unit_source': dict(header_attrs),
            'idx': idx,
            'filepath': filepath_str,
            'files': [filepath_str],
        }

    except Exception as e:
        print(f"Error loading snapshot {filepath}: {e}")
        return {'error': f"{type(e).__name__}: {e}", 'filepath': filepath_str}


class AsyncSnapshotLoader(QThread):
    """Loads the display samples of a list of snapshot files off the UI thread.

    Small HDF5 snapshots are read piece by piece in a process pool, big ones
    (BIG_SNAPSHOT particles or more) through the LOD store, binary ones whole.
    ``finished`` carries the snapshots sorted by Time and renumbered from 0, the
    GadgetUnits of the first one loaded, and its folder; files that failed are
    listed in ``errors`` and left out. Cancelling emits nothing.
    """
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list, object, object)

    def __init__(self, files: List[Path], max_particles: int = 150_000):
        super().__init__()
        self.files = files
        self.max_particles = max_particles
        self._cancel = False
        self.errors: List[str] = []
        self.n_requested = 0        # snapshots (not files) the loader tried

    def cancel(self):
        self._cancel = True

    def run(self):
        snapshots = []
        units = None
        output_dir = None

        sets, group_errors = group_snapshot_files(self.files)
        self.errors.extend(group_errors)
        self.n_requested = len(sets) + len(group_errors)

        # One task per HDF5 piece, one per binary file. Big HDF5 snapshots go
        # through the LOD store instead (one at a time, it uses every core).
        plans, tasks, big = {}, [], []
        for i, files in enumerate(sets):
            if _is_hdf5(files[0]):
                try:
                    plan = plan_hdf5_snapshot(files, self.max_particles)
                    if plan['n_total'] >= BIG_SNAPSHOT:
                        big.append((i, files))
                        continue
                    with h5py.File(files[0], 'r') as f:
                        header = dict(f['Header'].attrs) if 'Header' in f else {}
                except Exception as e:
                    self.errors.append(f"{Path(files[0]).name}: {type(e).__name__}: {e}")
                    continue
                plans[i] = {'plan': plan, 'results': [None] * len(files), 'failed': None}
                for j, (fp, pl) in enumerate(zip(plan['files'], plan['pieces'])):
                    tasks.append((i, j, read_hdf5_piece, (fp, pl, header)))
            else:
                tasks.append((i, None, load_single_snapshot_multiformat,
                              (str(files[0]), i, self.max_particles)))

        def _accept(result):
            nonlocal units, output_dir
            if 'error' in result:
                self.errors.append(f"{Path(result['filepath']).name}: {result['error']}")
                return
            snapshots.append(result)
            if units is None:
                from model.units import GadgetUnits
                units = GadgetUnits(result.get('unit_source'))
            if output_dir is None:
                output_dir = Path(result['filepath']).parent

        for i, files in big:
            if self._cancel:
                return
            name = Path(files[0]).name
            try:
                snap = snapshot_from_store(
                    files, i, self.max_particles, cancelled=lambda: self._cancel,
                    progress=lambda k, n, name=name: self.progress.emit(k, n, f"Indexing {name}: {k}/{n}"))
            except Exception as e:
                snap = {'error': f"{type(e).__name__}: {e}", 'filepath': str(files[0])}
            if snap is None:
                return
            _accept(snap)

        if not tasks:
            snapshots.sort(key=lambda s: s['time'])
            for i, snap in enumerate(snapshots):
                snap['idx'] = i
            self.finished.emit(snapshots, units, output_dir)
            return

        n_workers = max(1, min(mp.cpu_count(), len(tasks)))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fn, a): (i, j) for i, j, fn, a in tasks}
            for k, future in enumerate(as_completed(futures)):
                if self._cancel:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return
                i, j = futures[future]
                result = future.result()
                if j is None:
                    _accept(result)
                else:
                    entry = plans[i]
                    if 'error' in result and entry['failed'] is None:
                        entry['failed'] = result
                    entry['results'][j] = result
                    if all(r is not None for r in entry['results']):
                        if entry['failed'] is not None:
                            bad = entry['failed']
                            _accept({'error': f"{Path(bad['filepath']).name}: {bad['error']}",
                                     'filepath': entry['plan']['files'][0]})
                        else:
                            _accept(assemble_hdf5_snapshot(entry['plan'], entry['results'], i))
                        entry['results'] = []      # free the arrays
                self.progress.emit(k + 1, len(tasks), f"Read {k + 1}/{len(tasks)} files")

        snapshots.sort(key=lambda s: s['time'])
        for i, snap in enumerate(snapshots):
            snap['idx'] = i

        self.finished.emit(snapshots, units, output_dir)
