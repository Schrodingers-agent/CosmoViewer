"""
FoF on big snapshots: separate process, memory guard, progress and cancel.

The FoF run happens in a child process (snapshot_io/fof_process.py), so the
viewer survives a run that runs out of memory or crashes.  These tests check
that the child's groups are exactly the in-process groups, that a memory
overrun, a kill by the system, a failure and a cancel are each reported, that
progress reaches the window, and that two crowded neighbouring cells no
longer need memory growing with the product of their counts.
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from snapshot_io import fof, fof_process  # noqa: E402
from snapshot_io.fof_finder import FoFGroupsEngine  # noqa: E402
from snapshot_io.fof_snapshot import run_snapshot_fof  # noqa: E402

BOX = 20.0


def _write_snapshot(folder, n=60_000, seed=5, nfiles=1):
    rng = np.random.default_rng(seed)
    centres = rng.random((30, 3)) * BOX
    c = (centres[rng.integers(0, 30, n)] + rng.normal(0, 0.4, (n, 3))) % BOX
    ids = rng.permutation(n).astype(np.uint64) + 1
    parts = np.array_split(np.arange(n), nfiles)
    for i, sl in enumerate(parts):
        name = "snapshot_000.hdf5" if nfiles == 1 else f"snapshot_000.{i}.hdf5"
        with h5py.File(Path(folder) / name, "w") as f:
            h = f.create_group("Header")
            h.attrs.update({"BoxSize": BOX, "Time": 1.0, "Redshift": 0.0, "NumFilesPerSnapshot": nfiles,
                            "MassTable": np.array([0.0, 1.0, 0, 0, 0, 0]),
                            "NumPart_ThisFile": np.array([0, len(sl), 0, 0, 0, 0]),
                            "NumPart_Total": np.array([0, n, 0, 0, 0, 0])})
            p = f.create_group("Parameters")
            p.attrs.update({"UnitLength_in_cm": 3.085678e24, "UnitMass_in_g": 1.989e43, "Omega0": 0.3,
                            "HubbleParam": 0.7, "UnitVelocity_in_cm_per_s": 1e5})
            g = f.create_group("PartType1")
            g.create_dataset("Coordinates", data=c[sl])
            g.create_dataset("Velocities", data=np.zeros((len(sl), 3), np.float32))
            g.create_dataset("ParticleIDs", data=ids[sl])
    first = "snapshot_000.hdf5" if nfiles == 1 else "snapshot_000.0.hdf5"
    return str(Path(folder) / first), c, ids


def _children_of_this_process():
    if not sys.platform.startswith("linux"):
        return []
    me = str(os.getpid())
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if fields[1] == me and fields[0] != "Z":
                out.append(int(d.name))
    return out


def test_process_gives_the_same_groups_as_in_process(tmp_path):
    path, _, ids = _write_snapshot(tmp_path, nfiles=3)
    seen = []
    shown = ids[::7]
    a = FoFGroupsEngine.detect_groups_in_snapshot(path, shown, min_group_size=20, separate_process=True,
                                                  progress=lambda f, t: seen.append((f, t)))
    b = FoFGroupsEngine.detect_groups_in_snapshot(path, shown, min_group_size=20)
    assert a[1] == b[1] > 10
    np.testing.assert_array_equal(a[0], b[0])
    assert a[2] == b[2]
    np.testing.assert_array_equal(a[3].lookup(ids), b[3].lookup(ids))
    assert a[3].result.linking_length == b[3].result.linking_length
    np.testing.assert_array_equal(a[3].result.group_len, b[3].result.group_len)
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any(t.startswith("Reading") for _, t in seen) and any(t.startswith("Linking") for _, t in seen)


def test_in_process_result_matches_the_original_unsorted_lookup(tmp_path):
    path, _, ids = _write_snapshot(tmp_path)
    fof_run = run_snapshot_fof(path, min_group_size=20)
    assert fof_run.ids_sorted and np.all(np.diff(fof_run.ids.astype(np.int64)) > 0)
    # the same membership kept in an arbitrary order answers lookups identically
    perm = np.random.default_rng(0).permutation(len(fof_run.ids))
    unsorted = type(fof_run)(fof_run.result, fof_run.ids[perm], fof_run.group_index[perm],
                             fof_run.boxsize, fof_run.convention)
    q = np.concatenate([ids[::3], [10 ** 9]])
    np.testing.assert_array_equal(fof_run.lookup(q), unsorted.lookup(q))


def test_memory_limit_stops_the_process_with_a_clear_message(tmp_path):
    path, _, _ = _write_snapshot(tmp_path)
    with pytest.raises(fof_process.FoFProcessError, match="memory"):
        fof_process.run_fof_process(path, {"min_group_size": 20}, memory_limit=1 << 20)
    assert _children_of_this_process() == []


def test_a_process_killed_by_the_system_is_reported(tmp_path, monkeypatch):
    path, _, _ = _write_snapshot(tmp_path)
    killed = []

    def kill_once(pid):   # what the system does when the computer runs out of memory
        if not killed:
            killed.append(pid)
            os.kill(pid, 9)
        return None
    monkeypatch.setattr(fof_process, "_footprint", kill_once)
    with pytest.raises(fof_process.FoFProcessError, match="ran out of memory"):
        fof_process.run_fof_process(path, {"min_group_size": 20})


def test_failure_and_cancel_are_reported(tmp_path):
    path, _, _ = _write_snapshot(tmp_path)
    with pytest.raises(fof_process.FoFProcessError, match="ValueError: no particles of primary type"):
        fof_process.run_fof_process(path, {"primary_types": [4], "max_memory_fraction": None})
    t = time.time()
    with pytest.raises(fof_process.FoFCancelled):
        fof_process.run_fof_process(path, {"min_group_size": 20}, cancelled=lambda: time.time() - t > 0.3)
    assert _children_of_this_process() == []


def test_crowded_neighbouring_cells_need_little_memory():
    # two dense clumps in neighbouring cells, just too far apart to link: before, testing them
    # took memory growing with the product of their counts (8,000 each: 6 GB; 20,000: ~40 GB)
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.01, (20_000, 3))
    pos = np.concatenate([a, a + [1.02, 0, 0]])
    tracemalloc.start()
    labels = fof._link_labels_grid(pos, 1.0, None)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(np.unique(labels)) == 2
    assert peak < 64 << 20
    joined = np.concatenate([a, a + [0.99, 0, 0]])   # one pair within l: one group
    assert len(np.unique(fof._link_labels_grid(joined, 1.0, None))) == 1


@pytest.mark.parametrize("boxsize", [None, 10.0])
def test_crowded_cell_test_is_exact(boxsize):
    rng = np.random.default_rng(int(bool(boxsize)))
    pts = [rng.uniform(0, 10, (3000, 3))]
    for _ in range(8):
        pts.append(rng.uniform(0, 10, 3) + rng.normal(0, 0.05, (3000, 3)))
    pos = np.concatenate(pts)
    if boxsize:
        pos %= boxsize
    l = 0.04
    reference = fof._link_labels_kdtree(pos, l, boxsize, 100_000)
    for heavy in (1, 1 << 18):    # 1: every remaining cell pair goes through the k-d tree test
        for in_place in (False, True):
            p = pos.copy()
            labels = fof._link_labels_grid(p, l, boxsize, heavy=heavy, reorder_in_place=in_place)
            np.testing.assert_array_equal(p, pos)    # positions restored
            _, ref_inv = np.unique(reference, return_inverse=True)
            _, inv = np.unique(labels, return_inverse=True)
            assert len(np.unique(np.stack([ref_inv, inv]), axis=1).T) == ref_inv.max() + 1 == inv.max() + 1


# --- through the window ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    import lod.store
    monkeypatch.setattr(lod.store, "CACHE_DIR", tmp_path / "lod_cache")
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
    c._shutdown_fof()
    c._shutdown_lod()
    c.view.close()


def _wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


SETTINGS = {'b_linking': 0.2, 'min_size': 20, 'r_link': None, 'link_types': [1]}


def test_window_shows_progress_and_gets_the_groups(app, ctl, tmp_path):
    path, _, _ = _write_snapshot(tmp_path)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots)
    snap = ctl.model.snapshots[0]
    ctl._open_fof_dialog()
    dlg = ctl.fof_dialog
    values = []
    dlg.progress.valueChanged.connect(values.append)
    ctl._run_fof_algorithm(SETTINGS)
    assert dlg.progress.isVisible() and dlg.btn_cancel.isVisible()
    assert _wait(app, lambda: 'FoFGroupIndex' in snap and not ctl._fof_worker.isRunning())
    expected = FoFGroupsEngine.detect_groups_in_snapshot(path, snap['id'], min_group_size=20)
    np.testing.assert_array_equal(snap['FoFGroupIndex'], expected[0])
    assert values and max(values) == 1000
    assert not dlg.progress.isVisible() and dlg.btn_run.isEnabled()
    assert dlg.lbl_status.text().startswith("Detected")


def test_window_reports_failure_and_cancel(app, ctl, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    _write_snapshot(tmp_path)
    ctl._load_directory(tmp_path)
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots)
    ctl._open_fof_dialog()
    monkeypatch.setattr(fof_process, "MEMORY_FRACTION", 1e-6)   # any run passes the limit
    ctl._run_fof_algorithm(SETTINGS)
    assert _wait(app, lambda: errors and not ctl._fof_worker.isRunning())
    assert "memory" in errors[0] and ctl.fof_dialog.lbl_status.text() == "FoF failed."
    monkeypatch.setattr(fof_process, "MEMORY_FRACTION", 0.6)
    ctl._run_fof_algorithm(SETTINGS)
    ctl._cancel_fof()
    assert _wait(app, lambda: not ctl._fof_worker.isRunning())
    app.processEvents()
    assert ctl.fof_dialog.lbl_status.text() == "FoF cancelled." and ctl.fof_dialog.btn_run.isEnabled()
    assert 'FoFGroupIndex' not in ctl.model.snapshots[0]
