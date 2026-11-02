"""
Regression tests for the final bug sweep of Immersive Mode (2026-10-27).

Each test failed on the code before the sweep and passes after it:

1. Background threads were let go too early: a tracking lookup still running after
   wait(5000), and finished jobs whose last signals were still queued.
   controller/thread_hold.py keeps each one until QThread.finished is delivered.
2. A loader replaced by Rescan (or a new folder) could still finish and install its
   snapshots over the new scan's.
3. Masses from Header/MassTable had no unit attributes, so the inspector showed no
   physical mass for a dark-matter-only Gadget-4 run.
4. fof_tab_NNN catalogues were classified as unknown files.
5. The catalogue dialog offered VELOCIraptor and binary SubFind as if they could be read.
6. A movie's worker thread and the window filled the catalogue cache at the same time,
   unlocked, so a catalogue could be read twice and the caches changed mid-iteration.
7. Gadget binary readers: SnapFormat 1 read 0 particles of every type, SnapFormat 2 was not
   recognised (its first record is 8 bytes, not 16) and split MASS over every type, IDs were
   always 32-bit, and big-endian Type 2 files were refused.

Synthetic data only.
"""
import gc
import struct
import sys
import threading
import time
import weakref
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_analysis_host import app, host, _wait  # noqa: E402,F401


# ── 1. thread lifetimes ──────────────────────────────────────────────────────

def test_finished_thread_is_held_until_its_finished_signal_is_delivered(app):
    from PyQt6.QtCore import QThread
    from controller.thread_hold import hold

    class Job(QThread):
        def run(self):
            pass

    job = hold(Job())
    job.start()
    job.wait()
    ref = weakref.ref(job)
    del job
    gc.collect()
    assert ref() is not None              # QThread.finished is still queued: not freed
    for _ in range(5):
        app.processEvents()
    gc.collect()
    assert ref() is None                  # delivered: released


def test_replaced_tracking_lookup_neither_blocks_nor_leaks_its_result(app, host, monkeypatch):
    import controller.tracking_controller as tc
    from controller import thread_hold
    gate, calls = threading.Event(), []
    real = tc.locate_ids

    def slow(files, ids):
        calls.append(1)
        if len(calls) == 1:
            gate.wait(20)                 # the first lookup hangs until released
        return real(files, ids)

    monkeypatch.setattr(tc, "locate_ids", slow)
    s = host.model.tracker.get_set(0)
    s.set_population(np.arange(1, 51, dtype=np.uint64), 'ids')
    host._start_tracking()
    assert _wait(app, lambda: calls, 10)
    old = host._track_worker
    s.set_population(np.arange(1, 41, dtype=np.uint64), 'ids')
    t0 = time.time()
    host._start_tracking()                # replaces the hung lookup
    assert time.time() - t0 < 2.0         # used to wait up to 5 s, then drop it while it ran
    assert old in thread_hold._HELD and old.isRunning()
    gate.set()
    assert _wait(app, lambda: old.isFinished() and host._track_worker.isFinished(), 60)
    for _ in range(5):
        app.processEvents()
    for (i, set_id), (version, loc) in host._track_cache.items():
        assert version == s.version and len(loc.ids) == 40   # nothing of the replaced lookup


def test_rescan_drops_what_the_replaced_loader_read(app, host, monkeypatch):
    got = []
    monkeypatch.setattr(type(host), "_on_load_complete", lambda self, *a: got.append(a))
    files = [Path(p) for p in host.test_paths]
    host._start_loader(files)
    first = host.model._loader_thread
    host._start_loader(files[:1])         # a Rescan with one file left
    assert _wait(app, lambda: first.isFinished() and host.model._loader_thread.isFinished(), 120)
    for _ in range(10):
        app.processEvents()
    assert len(got) == 1 and len(got[0][0]) == 1     # only the second loader's one snapshot


# ── 3. MassTable masses ──────────────────────────────────────────────────────

def _g4(path, cosmological):
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs.update({"MassTable": np.array([0, 0.5, 0, 0, 0, 0.]), "Time": 0.5,
                        "NumPart_ThisFile": np.array([0, 4, 0, 0, 0, 0], dtype=np.uint64)})
        p = f.create_group("Parameters")
        p.attrs.update({"UnitMass_in_g": 1.989e43, "HubbleParam": 0.7,
                        "ComovingIntegrationOn": np.int32(int(cosmological))})
        g = f.create_group("PartType1")
        c = g.create_dataset("Coordinates", data=np.zeros((4, 3)))
        c.attrs.update({"a_scaling": 1.0 if cosmological else 0.0, "h_scaling": -1.0 if cosmological else 0.0,
                        "length_scaling": 1.0, "to_cgs": 3.085678e24})
        g.create_dataset("ParticleIDs", data=np.arange(1, 5, dtype=np.uint64))
    return str(path)


def test_masstable_masses_have_gadget4_units(tmp_path):
    from snapshot_io.properties import list_properties
    cos = list_properties([_g4(tmp_path / "c.hdf5", True)])['Masses']
    assert cos.scaling == {'a_scaling': 0.0, 'h_scaling': -1.0, 'to_cgs': 1.989e43}
    iso = list_properties([_g4(tmp_path / "i.hdf5", False)])['Masses']
    assert iso.scaling == {'a_scaling': 0.0, 'h_scaling': 0.0, 'to_cgs': 1.989e43}


def test_masstable_units_are_not_guessed_for_other_writers(tmp_path):
    from snapshot_io.properties import list_properties
    path = _g4(tmp_path / "x.hdf5", True)
    with h5py.File(path, "r+") as f:
        del f["PartType1/Coordinates"].attrs["length_scaling"]     # not Gadget-4's attribute set
    assert list_properties([path])['Masses'].scaling == {}


def test_inspector_shows_physical_masstable_mass(tmp_path):
    from snapshot_io.particle_record import read_records
    path = _g4(tmp_path / "c.hdf5", True)
    rec = read_records([path], [2])[2]
    assert rec.physical('Masses') == pytest.approx(0.5 / 0.7)


# ── 4. catalogue files in the folder listing ────────────────────────────────

def test_fof_tab_files_are_catalogues(tmp_path):
    from gadget_file_discovery import FileDiscoveryEngine
    for name in ("snapshot_002.hdf5", "fof_tab_002.hdf5", "fof_subhalo_tab_003.1.hdf5"):
        (tmp_path / name).write_bytes(b"")
    m = FileDiscoveryEngine(validate_snapshots=False).discover(tmp_path)
    assert sorted((r.filepath.name, r.index) for r in m.fof_catalogs) == \
        [("fof_subhalo_tab_003.1.hdf5", 3), ("fof_tab_002.hdf5", 2)]
    assert not m.unknown_files


# ── 5. catalogue formats offered ─────────────────────────────────────────────

def test_catalogue_dialog_offers_only_readable_formats(app):
    from views.group_catalogue_dialog import GroupCatalogueDialog
    dlg = GroupCatalogueDialog()
    model = dlg.combo_fmt.model()
    enabled = {dlg.combo_fmt.itemData(i): model.item(i).isEnabled() for i in range(dlg.combo_fmt.count())}
    # Unreadable formats are not offered at all, not even greyed out.
    assert enabled == {'GADGET4': True}
    dlg.deleteLater()


# ── 6. catalogue cache shared with a movie's thread ─────────────────────────

def test_catalogue_is_read_once_when_two_threads_ask(app, host, monkeypatch):
    import controller.features_controller as fc
    reads = []

    def slow_read(path):
        reads.append(path)
        time.sleep(0.3)
        return object()

    monkeypatch.setattr(fc, "find_catalogue_for", lambda f, t=None: "cat.hdf5")
    monkeypatch.setattr(fc, "read_gadget4_catalogue", slow_read)
    host._catalogue_mode = 'auto'
    snap = host.model.snapshots[0]
    out = []
    t = threading.Thread(target=lambda: out.append(host._catalogue_for(snap)))
    t.start()
    out.append(host._catalogue_for(snap))
    t.join()
    assert len(reads) == 1 and out[0] is out[1]


# ── 7. Gadget binary snapshots ──────────────────────────────────────────────

def _header(npart, massarr, nfiles, e):
    nall = list(npart) if nfiles == 1 else [2 * n for n in npart]
    return struct.pack(e + "6I 6d d d i i 6I i i d d d d i i 6I i 60x",
                       *npart, *massarr, 0.5, 1.0, 0, 0, *nall, 0, nfiles, 50.0, 0.3, 0.7, 0.7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0)


def _write_binary(path, fmt, npart, massarr, arrays, e="<", nfiles=1):
    """arrays: [(name, ndarray)] in file order, HEAD excluded."""
    def rec(data):
        return struct.pack(e + "I", len(data)) + data + struct.pack(e + "I", len(data))

    out = b""
    for name, data in [("HEAD", _header(npart, massarr, nfiles, e))] + \
            [(n, a.astype(a.dtype.newbyteorder(e)).tobytes()) for n, a in arrays]:
        if fmt == 2:
            out += rec(name.ljust(4).encode() + struct.pack(e + "I", len(data) + 8))
        out += rec(data)
    Path(path).write_bytes(out)


def _particles(npart, seed=0, id_dtype=np.uint32, double=False):
    rng = np.random.default_rng(seed)
    n = sum(npart)
    ft = np.float64 if double else np.float32
    return (rng.random((n, 3)).astype(ft), rng.random((n, 3)).astype(ft),
            (np.arange(n) + 1 + (2 ** 40 if id_dtype == np.uint64 else 0)).astype(id_dtype))


@pytest.mark.parametrize("fmt,e,id_dtype,double", [
    (1, "<", np.uint32, False), (1, ">", np.uint64, True), (2, "<", np.uint64, False), (2, ">", np.uint32, False)])
def test_gadget_binary_formats(tmp_path, fmt, e, id_dtype, double):
    from cosmo_io_engine import GadgetBinaryReader, SnapshotHarvester
    npart = [3, 5, 0, 0, 2, 0]
    massarr = [0.0, 0.25, 0.0, 0.0, 0.0, 0.0]          # gas and stars have their own masses
    pos, vel, ids = _particles(npart, id_dtype=id_dtype, double=double)
    mass = np.array([1, 2, 3, 7, 8], dtype=np.float32)  # 3 gas, then 2 stars
    u = np.array([10, 11, 12], dtype=np.float32)
    rho = np.array([20, 21, 22], dtype=np.float32)
    path = tmp_path / "snapshot_000"
    _write_binary(path, fmt, npart, massarr,
                  [("POS", pos), ("VEL", vel), ("ID", ids), ("MASS", mass), ("U", u), ("RHO", rho)], e=e)
    assert SnapshotHarvester(str(path)).format_type == f"GADGET_BINARY_TYPE{fmt}"
    data = GadgetBinaryReader(str(path)).read_particles()
    assert np.array_equal(data[0]['Coordinates'], pos[:3]) and np.array_equal(data[1]['Coordinates'], pos[3:8])
    assert np.array_equal(data[4]['Velocities'], vel[8:])
    assert data[1]['ParticleIDs'].dtype.itemsize == np.dtype(id_dtype).itemsize
    assert np.array_equal(data[4]['ParticleIDs'], ids[8:])
    assert np.array_equal(data[0]['Masses'], mass[:3]) and np.array_equal(data[4]['Masses'], mass[3:])
    assert np.all(data[1]['Masses'] == 0.25)
    assert np.array_equal(data[0]['InternalEnergy'], u) and np.array_equal(data[0]['Density'], rho)


def test_gadget_binary_multifile_uses_each_files_counts(tmp_path):
    from cosmo_io_engine import GadgetBinaryReader
    counts = ([2, 3, 0, 0, 0, 0], [1, 4, 0, 0, 0, 0])
    expect = {0: [], 1: []}
    for i, npart in enumerate(counts):
        pos, vel, ids = _particles(npart, seed=i)
        ids = ids + 100 * i
        _write_binary(tmp_path / f"snapshot_000.{i}", 2, npart, [0, 0.5, 0, 0, 0, 0],
                      [("POS", pos), ("VEL", vel), ("ID", ids),
                       ("MASS", np.full(npart[0], 3.0, np.float32))], nfiles=2)
        expect[0].append(ids[:npart[0]])
        expect[1].append(ids[npart[0]:])
    data = GadgetBinaryReader(str(tmp_path / "snapshot_000.0")).read_particles()
    assert np.array_equal(data[0]['ParticleIDs'], np.concatenate(expect[0]))
    assert np.array_equal(data[1]['ParticleIDs'], np.concatenate(expect[1]))
    assert len(data[0]['Masses']) == 3 and np.all(data[0]['Masses'] == 3.0)


def test_gadget_binary_multifile_with_a_missing_piece_is_refused(tmp_path):
    from cosmo_io_engine import GadgetBinaryReader
    npart = [0, 2, 0, 0, 0, 0]
    pos, vel, ids = _particles(npart)
    _write_binary(tmp_path / "snapshot_000.0", 1, npart, [0, 1, 0, 0, 0, 0],
                  [("POS", pos), ("VEL", vel), ("ID", ids)], nfiles=2)
    with pytest.raises(ValueError, match="missing"):
        GadgetBinaryReader(str(tmp_path / "snapshot_000.0")).read_particles()


def test_binary_snapshot_header_reaches_the_app(tmp_path):
    from snapshot_io.snapshot_loader import load_single_snapshot_multiformat
    npart = [0, 4, 0, 0, 0, 0]
    pos, vel, ids = _particles(npart)
    path = tmp_path / "snapshot_000"
    _write_binary(path, 1, npart, [0, 0.25, 0, 0, 0, 0], [("POS", pos), ("VEL", vel), ("ID", ids)])
    snap = load_single_snapshot_multiformat((str(path), 0, 1000))
    assert 'error' not in snap, snap.get('error')
    assert len(snap['c']) == 4 and np.all(snap['m'] == 0.25)
    hdr = snap['header']
    assert hdr['BoxSize'] == 50.0 and hdr['HubbleParam'] == 0.7 and list(hdr['MassTable']) == [0, 0.25, 0, 0, 0, 0]
