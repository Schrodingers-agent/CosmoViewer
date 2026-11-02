"""
FoF on an isolated (BoxSize 0) run, e.g. two colliding galaxies.

Builds a small synthetic Gadget-4-style HDF5 file: two compact discs (PartType2)
inside two extended, overlapping dark-matter haloes (PartType1).  The defaults
must link the baryonic type with a fixed length and find exactly two groups.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snapshot_io.fof_finder import FoFGroupsEngine  # noqa: E402
from snapshot_io.fof_snapshot import fof_defaults  # noqa: E402


def _write_two_galaxies(path):
    rng = np.random.default_rng(0)
    centres = np.array([[-40.0, 0, 0], [40.0, 0, 0]])
    disc = np.vstack([c + rng.normal(scale=[3, 3, 0.3], size=(2000, 3)) for c in centres])
    halo = np.vstack([c + rng.normal(scale=60, size=(3000, 3)) for c in centres])
    with h5py.File(path, 'w') as f:
        h = f.create_group('Header')
        h.attrs['BoxSize'] = 0.0
        h.attrs['NumPart_Total'] = np.array([0, len(halo), len(disc), 0, 0, 0], dtype=np.uint64)
        h.attrs['NumPart_ThisFile'] = h.attrs['NumPart_Total']
        h.attrs['MassTable'] = np.array([0, 1e-3, 1e-4, 0, 0, 0])
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['Time'] = 0.0
        p = f.create_group('Parameters')
        p.attrs['UnitLength_in_cm'] = 3.085678e21
        p.attrs['HubbleParam'] = 1.0
        p.attrs['ComovingIntegrationOn'] = 0
        p.attrs['SofteningComovingClass0'] = 0.1
        p.attrs['SofteningComovingClass1'] = 0.3
        p.attrs['SofteningClassOfPartType1'] = 1
        p.attrs['SofteningClassOfPartType2'] = 0
        for t, pos in ((1, halo), (2, disc)):
            g = f.create_group(f'PartType{t}')
            g['Coordinates'] = pos.astype(np.float32)
            g['ParticleIDs'] = np.arange(len(pos), dtype=np.uint32) + (0 if t == 1 else 10 ** 6)
    return disc, halo


def test_isolated_defaults_find_two_galaxies(tmp_path):
    path = str(tmp_path / 'snapshot_000.hdf5')
    disc, _ = _write_two_galaxies(path)
    d = fof_defaults(path)
    assert d['periodic'] is False
    assert d['link_types'] == [2]
    assert d['suggested_length'] == 1.0 and d['length_unit'] == 'kpc'

    ids = np.arange(len(disc), dtype=np.uint32) + 10 ** 6
    ui, n, sizes, _ = FoFGroupsEngine.detect_groups_in_snapshot(
        path, ids, r_link=d['suggested_length'], primary_types=d['link_types'])
    assert n == 2 and sum(sizes.values()) > 0.95 * len(disc)
    # each galaxy's particles land in one group
    assert len(np.unique(ui[:2000][ui[:2000] > 0])) == 1
    assert len(np.unique(ui[2000:][ui[2000:] > 0])) == 1


def test_isolated_run_without_length_is_refused(tmp_path):
    path = str(tmp_path / 'snapshot_000.hdf5')
    _write_two_galaxies(path)
    try:
        FoFGroupsEngine.detect_groups_in_snapshot(path, np.arange(3), primary_types=[2])
    except ValueError as e:
        assert 'linking length' in str(e)
    else:
        raise AssertionError("isolated run without a linking length must raise")


def test_memory_guard_refuses_oversized_run(tmp_path, monkeypatch):
    import snapshot_io.fof_snapshot as fs
    path = str(tmp_path / 'snapshot_000.hdf5')
    _write_two_galaxies(path)
    monkeypatch.setattr(fs, 'physical_memory_bytes', lambda: 1000)
    try:
        fs.run_snapshot_fof(path, linking_length=1.0, primary_types=(2,))
    except MemoryError as e:
        assert 'needs about' in str(e)
    else:
        raise AssertionError("oversized FoF must be refused")
    fof = fs.run_snapshot_fof(path, linking_length=1.0, primary_types=(2,), max_memory_fraction=None)
    assert fof.result.n_groups == 2
