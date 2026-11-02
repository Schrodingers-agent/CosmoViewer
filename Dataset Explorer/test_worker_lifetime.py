"""
A replaced analysis worker must stay referenced until it has finished.

    QT_QPA_PLATFORM=offscreen python3 -m pytest -q test_worker_lifetime.py

When a new computation starts while one is running (another leaf, tab, centre
or set), CosmoExplorer._spawn_worker cancels the running ComputeWorker and
drops its reference; the worker is then held only by its own finished-signal
lambda, a reference cycle, so Python's garbage collector can free it while its
thread runs or while its queued signals wait. That crashed the process (SIGSEGV
in PyQtSlot::call, delivering a queued signal to the freed slot): in 2 of 6
runs of test_real_population_handoff.py, and in 3 of 4 runs of
verification/repro_orphaned_worker.py, which forces the collection.
_PrefixWorker already avoids this (density_explorer.py keeps it in
self._prefix_workers until it finishes).

While A's run() executes, its frame holds A, so the collector cannot free it
then (a repro run on main saw A "still alive" for that reason); the danger
starts once A has finished while its queued signals still wait. The check
therefore waits for A to finish without processing events, then collects:
while A's finished signal is still queued, A must still be referenced. (Once
that signal has been delivered nothing of A is queued, and letting it go is
correct; the lead's first version of this check missed that case and failed
whenever A finished during the switch.) It runs in a subprocess on a small synthetic
snapshot and exits right after looking, before any queued signal is
delivered, so it cannot crash the suite. Fixed by keeping every started
worker in CosmoExplorer._live_workers until its finished signal is delivered.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

CHILD = textwrap.dedent("""
    import gc, os, sys, weakref
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import density_explorer, selection_sets
    from measurements.models import plummer, write_gadget4_snapshot
    from measurements.snapshot_view import MSUN_G
    home = Path(sys.argv[2])
    selection_sets.FALLBACK_ROOT = home / "sets"
    selection_sets.INSTANCES_DIR = home / "explorer_instances"
    n = 20000
    import time, compute_worker
    original_run = compute_worker.ComputeWorker.run
    def slow_run(self):               # A must still be computing when B replaces it, on any machine
        time.sleep(0.5)
        original_run(self)
    compute_worker.ComputeWorker.run = slow_run
    pos, vel, m = plummer(n, 1e6, 0.003, seed=3)
    snap = home / "output" / "snapshot_000.hdf5"
    snap.parent.mkdir(parents=True)
    write_gadget4_snapshot(snap, {1: dict(pos=pos + 2.0, vel=vel, mass=None, ids=np.arange(n))},
                           mass_table=[0, 1e6 / n / (1.989e43 / MSUN_G), 0, 0, 0, 0])
    delivered = set()                 # workers whose finished signal has reached the window
    original = density_explorer.CosmoExplorer._on_worker_finished
    def spy(self, worker):
        delivered.add(id(worker))
        return original(self, worker)
    density_explorer.CosmoExplorer._on_worker_finished = spy
    win = density_explorer.CosmoExplorer()
    assert win.open_path(str(snap))
    assert win.sidebar.select_leaf("Analyses", "Profile, all types")    # worker A
    a = win._active_worker
    win.plot_panel._toolbar._switch_view("Matrix")                     # worker B replaces A
    assert a is not None and win._active_worker is not a
    a.wait(60000)                     # A's run() has returned; no events are processed from here on
    pending = id(a) not in delivered  # its finished signal (the last one) is still queued
    ref = weakref.ref(a)
    del a
    gc.collect()
    print("pending" if pending else "delivered", "alive" if ref() is not None else "freed", flush=True)
    os._exit(0)                       # before any queued signal of A is delivered
""")


def test_replaced_worker_stays_referenced_until_it_finishes(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path / "home"))
    run = subprocess.run([sys.executable, "-c", CHILD, str(HERE), str(tmp_path)], capture_output=True,
                         text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert run.returncode == 0, run.stderr[-2000:]
    # The danger is a worker freed while its signals still wait. One whose finished signal has
    # already been delivered may be let go: nothing of it is queued any more.
    assert "pending freed" not in run.stdout, run.stdout + run.stderr[-500:]
    if "delivered" in run.stdout:
        pytest.skip("A finished and its signals were delivered before the check; the case was not exercised")
    assert "pending alive" in run.stdout, run.stdout + run.stderr[-500:]
