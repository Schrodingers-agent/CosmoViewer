"""
The Analyse view: the Dataset Explorer as a tab of Immersive Mode's window, in one process
(controller/analysis_host.py).

Checks that the tab sits between the 3D view and the console and starts the Explorer only
when first shown; that it analyses the snapshot shown in 3D and follows it (while shown,
keeping the analysis and its tab; when shown again after changes; after playback, never
during it); that the active set reaches the Explorer with its meaning and an incomplete set's
warning is shown; that Analyse ▸, Use every particle, Show in 3D and a population proposed
by the Explorer do what they say; that a failed open keeps the previous result and says why;
and that the hosted Explorer neither registers as a separate Explorer window nor lets new set
files replace the population. Synthetic Plummer snapshots, headless.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# the Explorer's results cache, away from ~/.cosmoviewer (read when measurements is first imported)
os.environ.setdefault("COSMOVIEWER_RESULTS_CACHE", str(Path(tempfile.gettempdir()) / "cv-test-results-cache"))

import selection_sets  # noqa: E402
from controller.analysis_host import EXPLORER_DIR  # noqa: E402

if str(EXPLORER_DIR) not in sys.path:
    sys.path.append(str(EXPLORER_DIR))        # as the host does, after Immersive's folder
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402

N = 2000
PROFILE = ("Analyses", "Profile, PartType1")


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def _write_run(folder: Path) -> list:
    """Three snapshots of one Plummer sphere of N dark-matter particles, moving by 0.1 per snapshot;
    distinct Time values, because the loader orders snapshots by time."""
    code_mass = 1.989e43 / MSUN_G
    paths = []
    for i in range(3):
        pos, vel, _ = plummer(N, 1e6, 0.003, seed=i)
        p = folder / f"snapshot_{i:03d}.hdf5"
        write_gadget4_snapshot(p, {1: dict(pos=pos + 2.0 + 0.1 * i, vel=vel, mass=None, ids=np.arange(1, N + 1))},
                               time=0.5 + 0.25 * i, mass_table=[0, 1e6 / N / code_mass, 0, 0, 0, 0])
        paths.append(p)
    return paths


def _wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _settled(c):
    """Nothing pending: no sync scheduled or running, no Explorer computation."""
    ex = c._explorer
    w = None if ex is None else ex._active_worker
    return not c._analysis_timer.isActive() and not c._analysis_busy and (w is None or not w.isRunning())


def _settle(app, c, timeout=60):
    end = time.time() + timeout
    while time.time() < end:            # settled twice in a row: a finished worker's signals delivered
        if _wait(app, lambda: _settled(c), max(0.1, end - time.time())):
            app.processEvents()
            if _settled(c):
                return True
    return False


def _opened(c) -> str:
    return Path(c._explorer.open_snapshot_file()).name if c._explorer is not None else ""


def _population(c):
    return c._explorer._global_context.get("selection_set")


@pytest.fixture
def host(app, tmp_path, monkeypatch):
    """A controller with the three-snapshot run loaded; sets and instance records kept in tmp_path."""
    from PyQt6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(selection_sets, "FALLBACK_ROOT", tmp_path / "home" / "sets")
    monkeypatch.setattr(selection_sets, "INSTANCES_DIR", tmp_path / "home" / "explorer_instances")
    run = tmp_path / "run"
    run.mkdir()
    paths = _write_run(run)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c._load_directory(run)
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 120)
    app.processEvents()
    c.test_paths = paths
    yield c
    c._shutdown_analysis_host()
    c._cancel_record_job()
    for j in c._old_record_jobs:
        j.wait(5000)
    c._reset_tracking()
    c._shutdown_lod()
    c.view.close()


def _show_analyse(app, c):
    c.view.tab_widget.setCurrentWidget(c.view.analyse_tab)
    assert _settle(app, c)


def _set_population(c, ids, kind="ids", definition=None, complete=None, n_expected=None):
    """Make the active set these particles, as the 3D view's tools do, and refresh as they do."""
    s = c.model.tracker.get_set(c.view.selection_panel.active_set_idx)
    s.set_population(np.asarray(ids, np.uint64), kind, definition, c._defined_at(c.model.snapshots[c.current_snap_idx]),
                     complete=complete, n_expected=n_expected)
    c._on_selection_changed()
    return s


# --------------------------------------------------------------------------- the tab

def test_tab_sits_between_3d_and_console_and_explorer_waits(app, host):
    c = host
    tw = c.view.tab_widget
    assert tw.count() == 3 and tw.widget(1) is c.view.analyse_tab and tw.tabText(1) == "Analyse"
    assert tw.widget(0) is c.view.explore_tab and tw.widget(2) is c.view.telemetry_view
    c.view.set_telemetry_badge(2)                        # the console tab is still found by widget
    assert tw.tabText(2).endswith("(2)")
    assert c._explorer is None and c.view.analyse_frame.layout().count() == 0
    c.view.slide_time.setValue(1)                        # 3D work does not start the Explorer
    c._update_frame()
    app.processEvents()
    assert c._explorer is None
    sp = c.view.selection_panel
    assert sp.btn_send.text() == "Analyse ▸" and sp.btn_send.toolTip() == "Analyse this set in the Analyse tab"


def test_showing_the_tab_opens_the_3d_snapshot_in_this_window(app, host):
    import pyqtgraph as pg
    c = host
    before = {k: pg.getConfigOption(k) for k in ("antialias", "background", "foreground")}
    c.view.slide_time.setValue(2)
    _show_analyse(app, c)
    ex = c._explorer
    assert ex is not None and ex._hosted
    assert _opened(c) == "snapshot_002.hdf5"
    # hosted: its interface is in the tab with its style, it is not a separate Explorer window
    import density_explorer
    assert ex.plot_panel.window() is c.view and c.view.analyse_frame.styleSheet() == density_explorer.STYLESHEET
    assert not ex.sidebar.btn_open.isVisibleTo(ex.sidebar)
    assert selection_sets.explorer_for(str(c.test_paths[2])) is None
    # Immersive's pyqtgraph options are as before the Explorer was imported
    assert {k: pg.getConfigOption(k) for k in before} == before
    # its options are in the Analysis menu, before the old separate-window path
    texts = [a.text() for a in c._analysis_menu.actions()]
    assert texts[0] == "Analyse the active set" and texts[-1] == "Open in a separate Explorer window"
    assert "Physical" in texts and "Comoving" in texts and any(t.startswith("Centre:") for t in texts)
    bar = c.view.analyse_bar
    assert bar.lbl_main.text() == "Analysing snapshot_002.hdf5 (snapshot 3 of 3) · Population: every particle"
    assert not bar.btn_every.isEnabled()


def test_follows_the_3d_snapshot_while_shown_keeping_the_analysis(app, host):
    c = host
    _show_analyse(app, c)
    ex = c._explorer
    assert _opened(c) == "snapshot_000.hdf5"
    assert ex.sidebar.select_leaf(*PROFILE)
    assert _settle(app, c)
    ex.plot_panel._toolbar._switch_view("Enclosed mass")
    assert _settle(app, c)
    c.view.slide_time.setValue(1)                        # until it follows, the bar says it is behind
    assert c.view.analyse_bar.lbl_warning.text() == ("The 3D view shows snapshot_001.hdf5; this analysis is of "
                                                     "snapshot_000.hdf5. Following…")
    assert _wait(app, lambda: _opened(c) == "snapshot_001.hdf5") and _settle(app, c)
    assert c.view.analyse_bar.lbl_warning.text() == ""
    assert ex.shown_leaf() == PROFILE and ex.plot_panel.current_tab() == "Enclosed mass"
    assert "All 2,000 particles" in ex.plot_panel.lbl_status.text()
    assert "snapshot 2 of 3" in c.view.analyse_bar.lbl_main.text()


def test_follows_when_shown_again_after_changes_while_hidden(app, host):
    c = host
    _show_analyse(app, c)
    c.view.tab_widget.setCurrentWidget(c.view.explore_tab)
    c.view.slide_time.setValue(2)
    _wait(app, lambda: False, 0.5)                       # well past the follow delay
    assert _opened(c) == "snapshot_000.hdf5"             # nothing re-opened while hidden
    _show_analyse(app, c)
    assert _opened(c) == "snapshot_002.hdf5"


def test_playback_is_followed_only_once_it_stops(app, host):
    c = host
    _show_analyse(app, c)
    ex = c._explorer
    calls = []
    follow = ex.follow_snapshot
    ex.follow_snapshot = lambda path: (calls.append((Path(path).name, c.is_playing)), follow(path))[1]
    c.view.btn_play.click()                              # play ...
    assert c.is_playing
    c.timer.stop()                                       # ... and step by hand, so the test decides the frames
    c._next_frame()
    c._next_frame()
    _wait(app, lambda: False, 0.6)
    assert c.current_snap_idx == 2 and _opened(c) == "snapshot_000.hdf5" and calls == []
    assert "It follows when playback stops." in c.view.analyse_bar.lbl_warning.text()
    c.view.btn_play.click()                              # stop
    assert not c.is_playing
    assert _wait(app, lambda: _opened(c) == "snapshot_002.hdf5") and _settle(app, c)
    assert calls == [("snapshot_002.hdf5", False)]
    assert c.view.analyse_bar.lbl_warning.text() == ""


# --------------------------------------------------------------------------- the population

def test_the_active_sets_meaning_reaches_the_analysis_and_its_warning_is_shown(app, host):
    c = host
    c.view.slide_time.setValue(2)
    s = _set_population(c, np.arange(1, 501), "group", {"finder": "fof", "group": 1, "b": 0.2, "min_size": 32},
                        complete=False, n_expected=800)
    _show_analyse(app, c)
    pop = _population(c)
    assert pop["kind"] == "group" and pop["complete"] is False and pop["n_expected"] == 800
    assert pop["definition"]["group"] == 1 and pop["defined_at"]["file"] == "snapshot_002.hdf5"
    np.testing.assert_array_equal(pop["ids"], np.arange(1, 501, dtype=np.uint64))
    bar = c.view.analyse_bar
    assert bar.lbl_main.text() == (f"Analysing snapshot_002.hdf5 (snapshot 3 of 3) · Population: "
                                   f"{s.name}: {s.describe()}")
    assert bar.lbl_warning.text() == s.completeness_warning() and bar.btn_every.isEnabled()
    # an unchanged set is not handed over again; a new version of it is, while the tab is shown
    handed = []
    activate = c._explorer.activate_population
    c._explorer.activate_population = lambda p: (handed.append(p), activate(p))[1]
    c._update_frame()
    assert _settle(app, c) and handed == []
    _set_population(c, np.arange(1, 801), "group", {"finder": "fof", "group": 1}, complete=True, n_expected=800)
    assert _wait(app, lambda: len(handed) == 1 and _population(c)["complete"] is True) and _settle(app, c)
    assert bar.lbl_warning.text() == "" and len(_population(c)["ids"]) == 800
    # a set defined at another snapshot is analysed by its IDs, and the bar says so
    c.view.slide_time.setValue(1)
    assert _wait(app, lambda: _opened(c) == "snapshot_001.hdf5") and _settle(app, c)
    assert _population(c)["kind"] == "group"
    assert bar.lbl_note.text() == "Defined at snapshot_002.hdf5; here its particles are found by their IDs."


def test_with_a_set_active_a_field_view_says_it_shows_the_whole_type(app, host):
    """Field views are not restricted to the set. With a set active, the
    bar's main line says so in full while a field view is shown, and drops it for an analysis."""
    c = host
    c.view.slide_time.setValue(2)
    s = _set_population(c, np.arange(1, 501))
    _show_analyse(app, c)
    ex, bar = c._explorer, c.view.analyse_bar
    base = f"Analysing snapshot_002.hdf5 (snapshot 3 of 3) · Population: {s.name}: {s.describe()}"
    assert bar.lbl_main.text() == base                          # an analysis is shown
    field = next(leaf for leaf in ex.sidebar.available_leaves()
                 if not leaf.startswith(("Analyses/", "Sets/")))
    group, name = field.split("/", 1)
    assert ex.sidebar.select_leaf(group, name) and _settle(app, c)
    assert bar.lbl_main.text() == f"{base} · Field views show all of {group}, not this set"
    assert ex.sidebar.select_leaf(*PROFILE) and _settle(app, c)
    assert bar.lbl_main.text() == base
    # with every particle there is nothing to contradict
    c.view.selection_panel.chk_enable.setChecked(False)
    assert _wait(app, lambda: _population(c) is None) and _settle(app, c)
    assert ex.sidebar.select_leaf(group, name) and _settle(app, c)
    assert bar.lbl_main.text().endswith("Population: every particle")


def test_switched_off_or_empty_set_means_every_particle(app, host):
    c = host
    s = _set_population(c, np.arange(1, 301))
    _show_analyse(app, c)
    assert len(_population(c)["ids"]) == 300
    c.view.selection_panel.chk_enable.setChecked(False)
    assert _wait(app, lambda: _population(c) is None) and _settle(app, c)
    assert c.view.analyse_bar.lbl_main.text().endswith("Population: every particle")
    assert not s.enabled


def test_analyse_button_shows_the_tab_with_the_set_and_every_particle_holds(app, host):
    c = host
    sp = c.view.selection_panel
    sp.btn_send.click()                                  # empty set: refused, with a message
    assert c.view.tab_widget.currentWidget() is c.view.explore_tab and c._explorer is None
    assert "is empty" in sp.lbl_status.text()
    _set_population(c, np.arange(1, 301))
    sp.btn_send.click()                                  # one action: the tab, the set, and a result
    assert c.view.tab_widget.currentWidget() is c.view.analyse_tab
    assert _wait(app, lambda: _population(c) is not None) and _settle(app, c)
    np.testing.assert_array_equal(_population(c)["ids"], np.arange(1, 301, dtype=np.uint64))
    ex = c._explorer
    first = next(leaf for leaf in ex.sidebar.available_leaves() if leaf.startswith("Analyses/"))
    assert ex.shown_leaf() == tuple(first.split("/", 1)) and "All 300 particles" in ex.plot_panel.lbl_status.text()
    bar = c.view.analyse_bar
    bar.btn_every.click()                                # Use every particle ...
    assert _population(c) is None and _settle(app, c)
    assert "All 2,000 particles" in ex.plot_panel.lbl_status.text()
    c._update_frame()                                    # ... holds while the set is unchanged
    assert _settle(app, c) and _population(c) is None
    assert "Analyse ▸ in the selection panel hands it over" in bar.lbl_note.text()
    assert ex.sidebar.select_leaf(*PROFILE) and _settle(app, c)
    sp.btn_send.click()                                  # Analyse ▸ hands it over again, keeping the analysis
    assert _wait(app, lambda: _population(c) is not None) and _settle(app, c)
    assert ex.shown_leaf() == PROFILE and "All 300 particles" in ex.plot_panel.lbl_status.text()
    c.view.tab_widget.setCurrentWidget(c.view.explore_tab)
    c._analysis_menu.actions()[0].trigger()              # Analysis ▸ Analyse the active set: the same
    assert c.view.tab_widget.currentWidget() is c.view.analyse_tab and _settle(app, c)


def test_show_in_3d_recentres_on_the_set(app, host):
    c = host
    c.view.slide_time.setValue(1)
    ids = np.arange(100, 400)
    _set_population(c, ids)
    _show_analyse(app, c)
    c.view.selection_panel.chk_enable.setChecked(False)  # switched off: Show in 3D switches it on
    with h5py.File(c.test_paths[1], "r") as f:
        pos = f["PartType1/Coordinates"][:][np.isin(f["PartType1/ParticleIDs"][:], ids)]
    for show in (c._explorer.show_in_3d_requested.emit, c.view.analyse_bar.btn_show_3d.click):
        c.view.set_camera_centre((0.0, 0.0, 0.0))
        c.view.tab_widget.setCurrentWidget(c.view.analyse_tab)
        show()
        app.processEvents()
        assert c.view.tab_widget.currentWidget() is c.view.explore_tab
        assert c.model.tracker.get_set(0).enabled
        np.testing.assert_allclose(c.view.camera_state()["center"], pos.mean(axis=0), rtol=1e-5, atol=1e-5)


def _outline(c):
    """The grey box outline's size and its corner (where the item's transform puts (0, 0, 0))."""
    from PyQt6.QtGui import QVector3D
    b = c.view.box_item
    o = b.transform().map(QVector3D(0, 0, 0))
    return np.array([float(x) for x in b.size()] + [o.x(), o.y(), o.z()])


def test_show_in_3d_frames_the_set_but_leaves_the_box_outline_alone(app, host):
    """Regression: Show in 3D and Recentre View used to resize the simulation box outline to the
    set's extent, and no later selection, set switch or Enable untick put it back."""
    c = host
    sp = c.view.selection_panel
    box = _outline(c)
    g_size = c.model.g_max - c.model.g_min
    np.testing.assert_allclose(box, np.concatenate([g_size, c.model.g_min]), rtol=1e-5, atol=1e-6)
    ids = np.arange(100, 400)
    _set_population(c, ids)
    _show_analyse(app, c)
    c.view.analyse_bar.btn_show_3d.click()
    app.processEvents()
    with h5py.File(c.test_paths[c.current_snap_idx], "r") as f:
        pos = f["PartType1/Coordinates"][:][np.isin(f["PartType1/ParticleIDs"][:], ids)]
    np.testing.assert_allclose(c.view.camera_state()["center"], pos.mean(axis=0), rtol=1e-5, atol=1e-5)
    assert c.view.camera_state()["distance"] == pytest.approx(1.5 * np.max(np.ptp(pos, axis=0)), rel=1e-4)
    np.testing.assert_array_equal(_outline(c), box)          # framed, and the outline is still the box
    for step in (lambda: _set_population(c, np.arange(1, 50)),       # a new selection
                 lambda: sp.btn_recentre.click(),                     # Recentre View on it
                 lambda: sp.combo_sets.setCurrentIndex(1),            # another set
                 lambda: sp.combo_sets.setCurrentIndex(0),
                 lambda: sp.chk_enable.setChecked(False),             # switched off
                 lambda: c.view.slide_time.setValue(2)):              # another snapshot
        step()
        app.processEvents()
        np.testing.assert_array_equal(_outline(c), box)


def test_a_population_proposed_by_the_analysis_becomes_the_active_set(app, host):
    c = host
    _set_population(c, np.arange(1, 50))
    _show_analyse(app, c)
    s = c.model.tracker.get_set(c.view.selection_panel.active_set_idx)
    v0 = s.version
    ids = np.arange(1000, 1100, dtype=np.uint64)
    definition = {"criteria": [{"kind": "bin", "field": "Velocities", "lo": 1.0, "hi": 2.0, "units": "km/s"}]}
    c._explorer.population_proposed.emit(selection_sets.SelectionSet(
        ids, "Fast tail", "#ff0000", made_in="test", kind="query", definition=definition,
        defined_at={"file": "snapshot_000.hdf5", "index": 0}, complete=True, n_expected=100))
    assert s.version > v0 and s.name == "Fast tail" and s.kind == "query" and s.complete is True
    assert s.definition == definition and s.defined_at["file"] == "snapshot_000.hdf5" and s.n_expected == 100
    np.testing.assert_array_equal(np.sort(s.id_array()), ids)
    assert "Set 1: Fast tail" in c.view.selection_panel.combo_sets.itemText(0)
    assert _wait(app, lambda: (_population(c) or {}).get("name") == "Fast tail") and _settle(app, c)
    assert _population(c)["kind"] == "query" and len(_population(c)["ids"]) == 100
    v1 = s.version                                       # a proposal of unknown meaning is refused whole
    c._explorer.population_proposed.emit(selection_sets.SelectionSet(np.arange(1, 5, dtype=np.uint64), "odd",
                                                                      kind="cluster"))
    assert s.version == v1 and s.name == "Fast tail" and len(s.selected_ids) == 100
    assert "unknown kind 'cluster'" in c.view.analyse_bar.lbl_problem.text()


# --------------------------------------------------------------------------- failures and the old path

def test_a_failed_open_keeps_the_previous_result_and_says_why(app, host):
    c = host
    c.view.slide_time.setValue(2)
    _show_analyse(app, c)
    ex = c._explorer
    assert ex.sidebar.select_leaf(*PROFILE) and _settle(app, c)
    plugin, vset = ex._plugin, ex._vset

    def broken(path):
        ex.plot_panel.set_status("Failed to open: the disk is unplugged")
        return False

    ex.open_path = broken
    c.view.slide_time.setValue(1)
    assert _wait(app, lambda: "Could not open" in c.view.analyse_bar.lbl_problem.text()) and _settle(app, c)
    assert _opened(c) == "snapshot_002.hdf5" and ex._plugin is plugin and ex._vset is vset
    bar = c.view.analyse_bar
    assert bar.lbl_problem.text() == ("Could not open snapshot_001.hdf5 for analysis: "
                                      "Failed to open: the disk is unplugged")
    assert bar.lbl_warning.text() == "The 3D view shows snapshot_001.hdf5; this analysis is of snapshot_002.hdf5."
    del ex.open_path                                     # fixed; showing the tab again retries
    c.view.tab_widget.setCurrentWidget(c.view.explore_tab)
    _show_analyse(app, c)
    assert _opened(c) == "snapshot_001.hdf5" and ex.shown_leaf() == PROFILE
    assert bar.lbl_problem.text() == ""


def test_a_view_that_cannot_start_says_why_and_is_retried(app, host, monkeypatch):
    import controller.analysis_host as ah
    c = host
    real = ah.import_explorer

    def missing():
        raise ImportError("the Dataset Explorer is not at /nowhere")

    monkeypatch.setattr(ah, "import_explorer", missing)
    c.view.tab_widget.setCurrentWidget(c.view.analyse_tab)
    app.processEvents()
    assert c._explorer is None
    assert c.view.analyse_bar.lbl_problem.text() == ("The Analyse view could not start: "
                                                     "ImportError: the Dataset Explorer is not at /nowhere")
    monkeypatch.setattr(ah, "import_explorer", real)
    c.view.tab_widget.setCurrentWidget(c.view.explore_tab)
    _show_analyse(app, c)
    assert _opened(c) == "snapshot_000.hdf5" and c.view.analyse_bar.lbl_problem.text() == ""


def test_a_snapshot_the_analysis_cannot_read_is_explained(app, host, monkeypatch, tmp_path):
    c = host
    _show_analyse(app, c)
    binary = tmp_path / "snapshot_001"                   # e.g. Gadget binary format, which Immersive reads
    binary.write_bytes(b"\x00\x01\x00\x00" * 64)
    monkeypatch.setattr(c.model, "get_snapshot", lambda i: {'filepath': str(binary), 'files': [str(binary)]})
    c._sync_analysis_now()
    assert _opened(c) == "snapshot_000.hdf5"
    assert c.view.analyse_bar.lbl_problem.text() == ("Could not open snapshot_001 for analysis: the Analyse view "
                                                     "reads HDF5 snapshots only, and this one is not HDF5.")


def test_a_set_chosen_in_the_analysis_tree_is_what_the_bar_states(app, host):
    c = host
    _set_population(c, np.arange(1, 101))
    _show_analyse(app, c)
    assert "Population: Selection Set 1" in c.view.analyse_bar.lbl_main.text()
    assert c._explorer.sidebar.select_leaf("Sets", "All particles")
    assert _population(c) is None
    assert c.view.analyse_bar.lbl_main.text().endswith("Population: every particle")


def test_new_set_files_in_the_run_do_not_replace_the_hosts_population(app, host):
    c = host
    _set_population(c, np.arange(1, 101))
    _show_analyse(app, c)
    ex = c._explorer
    live = _population(c)["path"]
    other = selection_sets.write_set(str(c.test_paths[0]), [1, 2, 3], "other set")
    ex._refresh_sets(announce=True)                      # what the Explorer's folder watcher calls
    assert _population(c)["path"] == live and len(_population(c)["ids"]) == 100
    assert other in ex._set_files.values() and ex._active_leaf() is not None


def test_open_in_a_separate_explorer_window_still_launches_one(app, host, monkeypatch):
    import controller.sets_controller as scm
    c = host
    _set_population(c, [7, 8, 9])
    _show_analyse(app, c)                                # a hosted Explorer has this run open ...
    launched = []
    monkeypatch.setattr(scm.subprocess, "Popen", lambda args, **kw: launched.append(args) or object())
    c._analysis_separate.trigger()
    args = launched[0]                                   # ... and is not mistaken for one
    assert Path(args[2]).name == "snapshot_000.hdf5" and args[3] == "--set"
    assert "Opening the Explorer" in c.view.selection_panel.lbl_status.text()
