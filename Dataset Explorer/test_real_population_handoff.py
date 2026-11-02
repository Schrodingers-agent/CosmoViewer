"""
Populations handed to the Dataset Explorer, on real data.

    COSMOVIEWER_DM_OUTPUT=<...>/DM-L50-N128/output \\
    COSMOVIEWER_SFR_OUTPUT=<...>/CollidingGalaxiesSFR/output \\
        QT_QPA_PLATFORM=offscreen python3 -m pytest -q test_real_population_handoff.py

Complements test_selection_sets.test_real_dm_halo_set_matches_headless (a
version-1 set file) with version-2 populations handed over the way a host
(Immersive Mode's Analyse view) does it, CosmoExplorer.activate_population():

  * a complete FoF group, catalogue group 1 of DM-L50-N128 fof_tab_002
    (34,120 particles, which CosmoViewer's FoF reproduces exactly, see
    Immersive tests/test_real_fof_vs_catalogue.py): Lagrangian radii, the
    radial profile and the energy budget equal the headless measurements on
    the catalogue membership; the notes say what the set is; nothing warns;
  * the same group known to be incomplete (a quarter of its members, about
    what a 500k display sample holds): every analysis made from it carries
    completeness_warning() in its exported notes, and the user can still see
    the warning once the analysis has finished (status or info line);
  * a host's population stays active until someone changes it.

Three earlier faults are covered: energy analyses that lost the set's notes,
a host's population dropped when a file appeared in the run folder, and the
warning vanishing from the status once an analysis finished.
Snapshots are copied to a temporary folder and ~/.cosmoviewer is redirected
there, so nothing is written next to the original output.
Skipped without COSMOVIEWER_DM_OUTPUT; the phase-diagram case needs
COSMOVIEWER_SFR_OUTPUT instead.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DM = os.environ.get("COSMOVIEWER_DM_OUTPUT", "")
SFR = os.environ.get("COSMOVIEWER_SFR_OUTPUT", "")
DM_NAMES = [f"snapshot_00{i}.hdf5" for i in range(3)] + ["fof_tab_002.hdf5"]
HAVE_DM = bool(DM) and all((Path(DM) / n).exists() for n in DM_NAMES)
HAVE_SFR = bool(SFR) and (Path(SFR) / "snapshot_100.hdf5").exists()
needs_dm = pytest.mark.skipif(not HAVE_DM, reason="set COSMOVIEWER_DM_OUTPUT to DM-L50-N128/output")
needs_sfr = pytest.mark.skipif(not HAVE_SFR, reason="set COSMOVIEWER_SFR_OUTPUT to CollidingGalaxiesSFR/output")

# what Immersive Mode records for FoF group 1 of snapshot_002 (Immersive test_real_fof_vs_catalogue.py)
FOF_DEF = {"finder": "fof", "group": 1, "subhalo": False, "min_size": 32, "link_types": [1],
           "convention": "gadget4", "b": 0.2, "linking_length_derived": 0.078125}
AT = {"file": "snapshot_002.hdf5", "index": 2, "time": 1.0, "redshift": 0.0}
WARNING = "WARNING: "              # the note restrict_to_set() adds; its wording may change

# ----------------------------------------------------------------------------- fixtures and helpers

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    """~/.cosmoviewer of these tests: set fallbacks, live populations, Explorer instances, results cache."""
    import measurements.cache
    import selection_sets
    h = tmp_path_factory.mktemp("cosmoviewer_home")
    mp = pytest.MonkeyPatch()
    mp.setattr(selection_sets, "FALLBACK_ROOT", h / "sets")
    mp.setattr(selection_sets, "INSTANCES_DIR", h / "explorer_instances")
    mp.setattr(measurements.cache, "DEFAULT_DIR", h / "results_cache")
    yield h
    mp.undo()


@pytest.fixture(scope="module")
def dm(tmp_path_factory, home):
    run = tmp_path_factory.mktemp("dm_l50") / "output"
    run.mkdir()
    for n in DM_NAMES:
        shutil.copyfile(Path(DM) / n, run / n)
    with h5py.File(run / "fof_tab_002.hdf5", "r") as f:
        off, ln = int(f["Group/GroupOffsetType"][0, 1]), int(f["Group/GroupLenType"][0, 1])
    with h5py.File(run / "snapshot_002.hdf5", "r") as f:
        members = np.unique(f["PartType1/ParticleIDs"][off:off + ln].astype(np.uint64))
    assert len(members) == 34_120
    return {"run": run, "snap": run / "snapshot_002.hdf5", "members": members}


def _idle(app, win, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        w = win._active_worker
        if w is None or not w.isRunning():
            app.processEvents()
            if win._active_worker is None or not win._active_worker.isRunning():
                return True
        time.sleep(0.02)
    return False


@pytest.fixture
def window(qapp):
    import density_explorer
    wins = []

    def make(snapshot):
        w = density_explorer.CosmoExplorer()
        w.show()
        wins.append(w)
        assert w.open_path(str(snapshot))
        return w
    yield make
    # Destroy the windows here, deterministically: an expected failure keeps the test's frames (and so
    # the window) alive in its report, and a window deleted later, at an arbitrary moment of the next
    # test, crashed the process in 2 of 4 runs before this was added.
    from PyQt6.QtCore import QCoreApplication, QEvent
    for w in wins:
        _idle(qapp, w)
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qapp.processEvents()
    wins.clear()


def _population(ids, complete, n_expected, name="FoF group 1", definition=FOF_DEF, at=AT):
    import selection_sets
    pop = selection_sets.SelectionSet(np.asarray(ids, dtype=np.uint64), name, "#ffaa00",
                                      made_in="CosmoViewer Immersive", kind="group", definition=dict(definition),
                                      defined_at=dict(at), complete=complete, n_expected=n_expected)
    if complete is False and "incomplete_reason" in getattr(selection_sets.SelectionSet, "__dataclass_fields__", {}):
        pop.incomplete_reason = "display_sample"       # set files that say why
    return pop


def _show(app, win, leaf, view):
    """Open an analysis leaf in a view, as a user does; what the window then shows.

    Each step waits for the analysis it started: replacing a running worker could crash (a replaced
    ComputeWorker is held only by a reference cycle, and the garbage collector can free it while
    its signals are pending: SIGSEGV), which crashed this file in 2 of 6 runs before the waits."""
    assert win.sidebar.select_leaf("Analyses", leaf), win.sidebar.available_leaves()
    assert _idle(app, win)
    win.plot_panel._toolbar._switch_view(view)
    assert _idle(app, win)
    pp = win.plot_panel
    notes = (pp._export_columns or pp._export_text or ([],))[0] if view == "Matrix" else []
    return {"status": pp.lbl_status.text(), "info": pp.lbl_info.text(), "notes": list(notes)}


def _cells(win):
    t = win.plot_panel._matrix_view._table
    labels = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    return labels, [[t.item(i, j).text() for j in range(t.columnCount())] for i in range(t.rowCount())]


def _shown(values):
    from plot_panel import _format_cell
    return [_format_cell(float(v), "Auto") for v in values]


def _headless(snapshot, ids):
    from measurements import SnapshotView, find_centre, lagrangian_radii
    view = SnapshotView(str(snapshot))
    p = view.particles([1], velocities=True, ids=True)
    return view, p.subset(np.isin(p.ids.astype(np.uint64), ids)), find_centre, lagrangian_radii


# ----------------------------------------------------------------------------- a complete group

@needs_dm
def test_complete_group_gives_the_headless_numbers(qapp, dm, window):
    from data_utils import latex_to_html
    from measurements.energy import energy_budget
    from measurements.profiles import radial_profile
    win = window(dm["snap"])
    assert win.activate_population(_population(dm["members"], True, 34_120))
    sel = win._global_context["selection_set"]
    assert (sel["kind"], sel["complete"], sel["n_expected"], sel["warning"]) == ("group", True, 34_120, None)
    assert not Path(sel["path"]).is_relative_to(dm["run"])            # nothing written next to the run

    view, p, find_centre, lagrangian_radii = _headless(dm["snap"], dm["members"])
    assert len(p) == p.n_total == 34_120

    # Lagrangian radii and the mass
    shown = _show(qapp, win, "Lagr. radii, all types", "Matrix")
    ref = lagrangian_radii(p, find_centre(p, "shrinking_sphere"))
    assert [row[1] for row in _cells(win)[1]] == _shown(ref["radius"].value)
    assert latex_to_html(f"M = {float(ref['mass_total'].value):.4e} Msun") in shown["info"], shown["info"]
    assert "All 34,120 particles" in shown["status"] and "set 'FoF group 1'" in shown["status"]
    assert any("is: FoF group 1 (b = 0.2, N_min = 32, DM) at snapshot_002.hdf5 · complete · 34,120 particles"
               in n for n in shown["notes"]), shown["notes"]
    assert not any("WARNING" in n for n in shown["notes"])

    # radial profile, computed as the leaf does (centre velocity inside r10, softening in this frame)
    shown = _show(qapp, win, "Profile, PartType1", "Matrix")
    c = find_centre(p, "shrinking_sphere")
    r10 = float(lagrangian_radii(p, c, [0.10])["radius"].value[0])
    c = find_centre(p, "shrinking_sphere", velocity_radius=r10)
    prof = radial_profile(p, c, n_bins=30, softening=view.softening(1)[0] or None)
    labels, cells = _cells(win)
    assert labels[4] == "density [Msun/kpc3]" and labels[6] == "M(<r_hi) [Msun]"
    assert [row[4] for row in cells] == _shown(prof["density"].value)
    assert [row[6] for row in cells] == _shown(prof["mass_enclosed"].value)
    assert [row[8] for row in cells] == _shown(prof["sigma_r"].value)
    assert "All 34,120 particles" in shown["status"]

    # energy budget: the same particles, the same mass
    shown = _show(qapp, win, "Energy, all types", "Budget")
    eb = energy_budget(view, [1], ids=dm["members"], ids_label="set 'FoF group 1':")
    row = {c[0]: c for c in _cells(win)[1]}
    assert row["total mass"][1] == _shown([eb["mass"].value])[0] == _shown([ref["mass_total"].value])[0]
    assert row["K, kinetic"][1] == _shown([eb["K"].value])[0]
    assert row["W, potential"][1] == _shown([eb["W"].value])[0]
    print(f"\ncomplete group 1: M = {float(ref['mass_total'].value):.6e} Msun, K = {float(eb['K'].value):.6e}, "
          f"W = {float(eb['W'].value):.6e} ({eb.provenance.parameters.get('potential')}), Q = {float(eb['Q']):.4f}")


# ----------------------------------------------------------------------------- the same group, incomplete

def _quarter(dm):
    return dm["members"][::4]                      # 8,530 of 34,120: about what a 500k display sample holds


@needs_dm
@pytest.mark.parametrize("leaf", [
    "Lagr. radii, all types",
    "Profile, PartType1",
    "Energy, all types",
    "Energy over run",
])
def test_incomplete_group_warns_in_the_notes_of_every_analysis(qapp, dm, window, leaf):
    win = window(dm["snap"])
    ids = _quarter(dm)
    assert win.activate_population(_population(ids, False, 34_120))
    assert win._global_context["selection_set"]["warning"]
    shown = _show(qapp, win, leaf, "Matrix")
    assert "Error" not in shown["status"], shown["status"]
    assert any(WARNING in n for n in shown["notes"]), shown["notes"]
    assert any(f"{len(ids):,} of its 34,120" in n for n in shown["notes"]), shown["notes"]


@needs_dm
@pytest.mark.parametrize("leaf,view", [
    ("Lagr. radii, all types", "Profile"),          # "⚠ incomplete set" in the status
    ("Profile, PartType1", "Density"),
    ("Lagr. radii, all types", "Matrix"),
])
def test_incomplete_group_warning_is_visible_once_the_analysis_has_finished(qapp, dm, window, leaf, view):
    win = window(dm["snap"])
    assert win.activate_population(_population(_quarter(dm), False, 34_120))
    assert "display sample" in win.plot_panel.lbl_status.text()      # true before the analysis runs
    shown = _show(qapp, win, leaf, view)
    assert "All 8,530 particles" in shown["status"], shown["status"]  # the computation is right ...
    seen = shown["status"] + " " + shown["info"]
    assert "display sample" in seen or "incomplete" in seen, seen     # ... and so must be what is said about it


@needs_dm
def test_host_population_stays_active_when_a_file_appears_in_the_run_folder(qapp, dm, window):
    win = window(dm["snap"])
    assert win.activate_population(_population(_quarter(dm), False, 34_120))
    path = win._global_context["selection_set"]["path"]
    (dm["run"] / "notes_by_the_researcher.txt").write_text("an export next to the run\n")
    try:
        end = time.time() + 3
        while time.time() < end:                                     # the folder watcher fires
            qapp.processEvents()
            time.sleep(0.02)
        win._refresh_sets(announce=True)                             # what it calls
        sel = win._global_context["selection_set"]
        assert sel is not None and sel["path"] == path, (
            f"active set after a file appeared: {sel}; status still says: {win.plot_panel.lbl_status.text()}")
    finally:
        (dm["run"] / "notes_by_the_researcher.txt").unlink()


# ----------------------------------------------------------------------------- gas: the phase diagram

@needs_sfr
def test_incomplete_gas_population_warns_in_the_phase_diagram(qapp, tmp_path, home, window):
    run = tmp_path / "output"
    run.mkdir()
    shutil.copyfile(Path(SFR) / "snapshot_100.hdf5", run / "snapshot_100.hdf5")
    with h5py.File(run / "snapshot_100.hdf5", "r") as f:
        gas = np.unique(f["PartType0/ParticleIDs"][:].astype(np.uint64))
    ids = gas[::4]
    win = window(run / "snapshot_100.hdf5")
    region = {"shape": "sphere", "centre": [0.0, 0.0, 0.0], "radius": 20.0, "units": "kpc/h", "periodic": False,
              "types": [0], "scope": "sample"}
    pop = _population(ids, False, len(gas), name="gas in the sample", definition=region,
                      at={"file": "snapshot_100.hdf5", "index": 0, "time": 1.203125})
    pop.kind = "region"
    assert win.activate_population(pop)
    shown = _show(qapp, win, "Phase diagram, gas", "Matrix")
    assert "Error" not in shown["status"], shown["status"]
    assert any(WARNING in n for n in shown["notes"]), shown["notes"]
