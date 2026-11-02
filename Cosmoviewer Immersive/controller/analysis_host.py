"""
The Analyse view: the Dataset Explorer shown inside Immersive Mode's window.

Mixed into CosmoViewerController. The contract between the two applications:

- the Explorer's interface is hosted as a tab of this window, in this process,
  and follows the snapshot shown in the 3D view;
- the active selection set is handed over as a population with
  ``CosmoExplorer.activate_population(s_set.as_selection_set(...))``;
- ``CosmoExplorer.show_in_3d_requested`` recentres and highlights the active set
  here; ``CosmoExplorer.population_proposed`` (a selection_sets.SelectionSet)
  becomes the active set here;
- the Explorer stays runnable on its own.

How it behaves:

- The Explorer is imported and created when the Analyse tab is first shown, or
  Analyse ▸ first used, so a session that never analyses pays nothing for it.
- Analyse shows the snapshot on screen in 3D. While the tab is shown, a new
  snapshot is opened there with the same analysis and tab; never during
  playback or while the slider is held (it follows when they stop). While the
  tab is hidden, it follows when the tab is shown again.
- The population is the active set. It is handed over when the tab is shown, or
  when the set or its version changes while the tab is shown. An empty or
  switched-off set means every particle. "Use every particle" holds until the
  set changes or Analyse ▸ hands it over again. Analyse ▸ with no analysis on
  show opens the first one, as the Explorer does for a set given at launch.
- The bar above the view reads what the Explorer actually uses, so it states
  the snapshot and population of the result on screen, not what was intended.
"""

import sys
from pathlib import Path

import h5py
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from controller.console import log
from controller.sets_controller import EXPLORER_SCRIPT
from selection_sets import KINDS

EXPLORER_DIR = EXPLORER_SCRIPT.parent
_PG_OPTIONS = ("antialias", "background", "foreground")


def import_explorer():
    """The Dataset Explorer's main module, imported into this process from the sibling folder.

    The folder goes on sys.path after Immersive's own, so where both apps have a module of the
    same name, Immersive's is used; the only such modules are selection_sets.py and
    cli_import.py, which are kept byte-identical. Importing the Explorer sets pyqtgraph's
    global options (antialiasing, a lighter axis colour, its background). Immersive's are
    put back at once, so Immersive's plots draw exactly as before: its large scatter plots
    take antialiasing from the global option, and it would slow them down. The Explorer's
    plots set their own background and antialiased curves; their axis text takes this
    window's grey."""
    if "density_explorer" not in sys.modules:
        if not (EXPLORER_DIR / "density_explorer.py").exists():
            raise ImportError(f"the Dataset Explorer is not at {EXPLORER_DIR}")
        if str(EXPLORER_DIR) not in sys.path:
            sys.path.append(str(EXPLORER_DIR))
        host_options = {k: pg.getConfigOption(k) for k in _PG_OPTIONS}
        try:
            import density_explorer  # noqa: F401
        finally:
            pg.setConfigOptions(**host_options)
    return sys.modules["density_explorer"]


def _files_of(snap: dict) -> set:
    """The files of a loaded snapshot, resolved (several for a multi-file snapshot)."""
    return {Path(p).resolve() for p in (snap.get('files') or [snap['filepath']])}


class AnalysisHostMixin:

    ANALYSIS_FOLLOW_MS = 250    # a burst of snapshot steps (arrow keys) re-opens once, after it

    def _init_analysis_host(self):
        """The Analyse tab, its bar and the Analysis menu; the Explorer itself comes when needed."""
        v = self.view
        self._explorer = None
        self._analysis_handed = None      # what the Explorer was last given: ('set', index, version) or ('all',)
        self._analysis_problems = {}      # 'start', 'open', 'population': what went wrong, for the bar
        self._analysis_failed_file = None   # not retried until the snapshot changes or the tab is shown again
        self._analysis_show_result = False  # Analyse ▸ asked for a result: open an analysis if none is shown
        self._analysis_busy = False
        self._analysis_timer = QTimer(v)
        self._analysis_timer.setSingleShot(True)
        self._analysis_timer.timeout.connect(self._sync_analysis_now)

        v.add_analyse_tab()
        v.analyse_bar.btn_show_3d.clicked.connect(lambda: self._show_population_in_3d())
        v.analyse_bar.btn_every.clicked.connect(lambda: self._analyse_every_particle())
        v.tab_widget.currentChanged.connect(self._on_analysis_tab_changed)
        v.play_toggled.connect(lambda: self._schedule_analysis_sync(0))        # after _toggle_play
        v.slide_time.sliderReleased.connect(lambda: self._schedule_analysis_sync(0))
        v.selection_panel.analyse_requested.connect(lambda: self._analyse_active_set())
        v.selection_panel.combo_sets.currentIndexChanged.connect(lambda _i: self._sync_analysis_view())

        menu = self._analysis_menu = v.add_analysis_menu()
        menu.addAction("Analyse the active set", lambda: self._analyse_active_set())
        menu.addSeparator()
        # the Explorer's options (units, physical or comoving, centre) go here once it has started
        self._analysis_separate = menu.addAction("Open in a separate Explorer window")
        self._analysis_separate.triggered.connect(lambda: self._send_set_to_explorer())
        self._update_analyse_bar()

    # ------------------------------------------------------------------ creating the view

    def _ensure_explorer(self) -> bool:
        """Start the Explorer the first time the Analyse view is needed. False, with the reason
        in the bar and the console, when it cannot start."""
        if self._explorer is not None:
            return True
        try:
            dx = import_explorer()
            ex = dx.CosmoExplorer()
            ex.set_hosted(True)
        except Exception as e:          # reported in words, never swallowed
            self._analysis_problems['start'] = f"The Analyse view could not start: {type(e).__name__}: {e}"
            log(self._analysis_problems['start'], 'error')
            self._update_analyse_bar()
            return False
        self._analysis_problems.pop('start', None)
        ex.setParent(self.view, Qt.WindowType.Window)   # never shown; lives and dies with this window
        self.view.host_analysis_view(ex.takeCentralWidget(), dx.STYLESHEET)
        for top in ex.menuBar().actions():               # its Options menu, into the Analysis menu
            if top.menu() is not None:
                for act in top.menu().actions():
                    self._analysis_menu.insertAction(self._analysis_separate, act)
                self._analysis_menu.insertSeparator(self._analysis_separate)
        ex.show_in_3d_requested.connect(self._show_population_in_3d)
        ex.population_proposed.connect(self._adopt_population)
        # a set or "All particles" chosen in its own tree changes what it uses: the bar says so
        ex.sidebar.leaf_clicked.connect(lambda *_: self._update_analyse_bar())
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_analysis_host)
        self._explorer = ex
        log("Analyse view started: the Dataset Explorer, in this window")
        return True

    def _shutdown_analysis_host(self):
        """On quit: stop the Analyse view's background work and wait for it, so that no thread
        outlives its window."""
        ex = self._explorer
        if ex is None:
            return
        app = QApplication.instance()
        try:
            app.aboutToQuit.disconnect(self._shutdown_analysis_host)
        except (AttributeError, TypeError, RuntimeError):
            pass                                    # not connected, or called from aboutToQuit itself
        workers = [w for w in [ex._active_worker, *getattr(ex, '_prefix_workers', ())] if w is not None]
        workers += [w._lod_worker for w in workers if getattr(w, '_lod_worker', None) is not None]
        if getattr(ex, '_lod_worker', None) is not None:     # the histogram's click-a-bin index build
            workers.append(ex._lod_worker)
        for w in workers:
            if hasattr(w, 'cancel'):
                w.cancel()
        for w in workers:
            w.wait(10000)

    # ------------------------------------------------------------------ keeping it in step

    def _analyse_tab_shown(self) -> bool:
        return self.view.tab_widget.currentWidget() is self.view.analyse_tab

    def _on_analysis_tab_changed(self, index: int):
        if self.view.tab_widget.widget(index) is not self.view.analyse_tab:
            return
        self._analysis_failed_file = None          # showing the tab again retries a snapshot that failed
        if self._ensure_explorer():
            self._schedule_analysis_sync(0)

    def _sync_analysis_view(self):
        """After a 3D frame update or a change of active set: when the Analyse tab is shown, say at
        once whether it is out of step, and bring it in step shortly. The work is left to the event
        loop, never done inside the frame update, because opening a snapshot processes events."""
        if self._explorer is not None and self._analyse_tab_shown():
            self._update_analyse_bar()
            self._schedule_analysis_sync(self.ANALYSIS_FOLLOW_MS)

    def _schedule_analysis_sync(self, delay_ms: int):
        if self._explorer is not None and self._analyse_tab_shown():
            self._analysis_timer.start(delay_ms)

    def _sync_analysis_now(self):
        """Same snapshot as the 3D view, and the active set as the population."""
        ex = self._explorer
        if ex is None or not self._analyse_tab_shown():
            return
        if self._analysis_busy:                     # an open in progress processed events: come back after it
            self._schedule_analysis_sync(self.ANALYSIS_FOLLOW_MS)
            return
        self._analysis_busy = True
        try:
            self._follow_3d_snapshot(ex)
            self._hand_over_population(ex)
            if self._analysis_show_result and ex.open_snapshot_file():
                self._analysis_show_result = False
                self._show_first_analysis(ex)
        finally:
            self._analysis_busy = False
            self._update_analyse_bar()

    def _is_open_in_analysis(self, snap: dict) -> bool:
        opened = self._explorer.open_snapshot_file() if self._explorer is not None else ""
        return bool(opened) and Path(opened).resolve() in _files_of(snap)

    def _loaded_index(self, path: str):
        """Index of the loaded snapshot that has this file, or None. The lookup is built once per
        run, so the bar can be refreshed on every frame of playback through hundreds of snapshots."""
        snaps = self.model.snapshots or []
        key = (id(snaps), len(snaps), snaps[0].get('filepath') if snaps else None,
               snaps[-1].get('filepath') if snaps else None)
        if getattr(self, '_loaded_index_key', None) != key:
            self._loaded_index_map = {f: i for i, s in enumerate(snaps) if s.get('filepath') for f in _files_of(s)}
            self._loaded_index_key = key
        return self._loaded_index_map.get(Path(path).resolve())

    def _follow_3d_snapshot(self, ex):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or not snap.get('filepath') or self._is_open_in_analysis(snap):
            return
        if self.is_playing or self.view.slide_time.isSliderDown():
            return                                  # followed when playback stops or the slider is let go
        path = str(snap['filepath'])
        if path == self._analysis_failed_file:
            return                                  # the bar says why; retried on the next change
        name = Path(path).name
        if Path(path).is_file() and not h5py.is_hdf5(path):   # e.g. a Gadget binary snapshot
            self._analysis_failed_file = path
            self._analysis_problems['open'] = (f"Could not open {name} for analysis: the Analyse view reads HDF5 "
                                               f"snapshots only, and this one is not HDF5.")
            log(self._analysis_problems['open'], 'warning')
            return
        self.view.analyse_bar.show_state(f"Opening {name}…")
        if ex.follow_snapshot(path):
            self._analysis_failed_file = None
            self._analysis_problems.pop('open', None)
            log(f"Analyse view: {name}")
        else:
            self._analysis_failed_file = path
            self._analysis_problems['open'] = f"Could not open {name} for analysis: {ex.plot_panel.lbl_status.text()}"
            log(self._analysis_problems['open'], 'error')

    def _wanted_population(self):
        """The active set and its key; (None, ('all',)) when it is empty or switched off: every particle."""
        idx = self.view.selection_panel.active_set_idx
        s_set = self.model.tracker.get_set(idx)
        if not s_set.enabled or not s_set.selected_ids:
            return None, ('all',)
        return s_set, ('set', idx, s_set.version)

    def _hand_over_population(self, ex):
        """Give the Explorer the active set as its population, if it changed since it was last given."""
        if not ex.open_snapshot_file():
            return                                  # handed over once a snapshot is open
        s_set, key = self._wanted_population()
        if key == self._analysis_handed:
            return
        try:
            ok = ex.activate_population(None if s_set is None
                                        else s_set.as_selection_set(self._set_colour_hex(s_set)))
            why = ex.plot_panel.lbl_status.text()
        except Exception as e:                      # e.g. the set could not be written for the Explorer
            ok, why = False, f"{type(e).__name__}: {e}"
        if ok:
            self._analysis_handed = key
            self._analysis_problems.pop('population', None)
        else:
            what = "every particle" if s_set is None else s_set.name
            self._analysis_problems['population'] = f"Could not hand {what} to the Analyse view: {why}"
            log(self._analysis_problems['population'], 'error')

    @staticmethod
    def _show_first_analysis(ex):
        """Analyse ▸ with nothing on show yet: open the first analysis, as the Explorer does for a
        set given at launch (--set), so the set's result appears without a second click. Once
        the researcher has chosen an analysis, that one is kept."""
        if ex.shown_leaf() is None:
            first = next((leaf for leaf in ex.sidebar.available_leaves() if leaf.startswith("Analyses/")), None)
            if first:
                ex.sidebar.select_leaf(*first.split("/", 1))

    def _update_analyse_bar(self):
        """Say what the Analyse view analyses, from what the Explorer actually uses."""
        bar = self.view.analyse_bar
        ex = self._explorer
        snaps = self.model.snapshots or []
        problem = " ".join(self._analysis_problems.values())
        opened = ex.open_snapshot_file() if ex is not None else ""
        bar.btn_show_3d.setEnabled(bool(snaps))
        if not opened:
            bar.btn_every.setEnabled(False)
            bar.show_state("Open a run to analyse it (File ▸ Read simulation snapshot…). The Analyse view "
                           "shows the snapshot on screen in 3D." if not snaps else
                           "The Analyse view opens the snapshot shown in 3D.", problem=problem)
            return
        name = Path(opened).name
        k = self._loaded_index(opened)
        where = f"snapshot {k + 1} of {len(snaps)}" if k is not None else "not one of the snapshots loaded in 3D"
        pop = ex._global_context.get("selection_set")
        main = f"Analysing {name} ({where}) · Population: " + (
            f"{pop['name']}: {pop.get('describe', '')}" if pop else "every particle")
        # Field views (histogram, line, matrix of one dataset) are not restricted to the set,
        # so with a set active the bar says so in full, on its main line, instead of claiming
        # the set as the population of a whole-type plot.
        leaf = ex.shown_leaf()
        if pop and leaf is not None and leaf[0] not in ("Analyses", "Sets"):
            main += f" · Field views show all of {leaf[0]}, not this set"
        notes, warnings = [], []
        if pop and pop.get('kind', 'ids') != 'ids':
            defined = (pop.get('defined_at') or {}).get('file')
            if defined and defined != name:
                notes.append(f"Defined at {defined}; here its particles are found by their IDs.")
        wanted, _ = self._wanted_population()
        if pop is None and wanted is not None:
            notes.append(f"The active set ({wanted.name}) is not used here; Analyse ▸ in the selection "
                         f"panel hands it over.")
        if pop and pop.get('warning'):
            warnings.append(pop['warning'])
        shown = self.model.get_snapshot(self.current_snap_idx)
        if shown and not self._is_open_in_analysis(shown):
            then = (" It follows when playback stops." if self.is_playing else
                    " It follows when the slider is released." if self.view.slide_time.isSliderDown() else
                    "" if self._analysis_failed_file else " Following…")
            warnings.append(f"The 3D view shows {Path(shown['filepath']).name}; this analysis is of {name}.{then}")
        bar.btn_every.setEnabled(pop is not None)
        bar.show_state(main, " ".join(notes), " ".join(warnings), problem)

    # ------------------------------------------------------------------ actions

    def _enable_active_set(self) -> bool:
        """Switch the active set on if it was off, through its checkbox, so that the panel, tracking
        and the 3D view follow as when the researcher ticks it. True if it was off."""
        sp = self.view.selection_panel
        if self.model.tracker.get_set(sp.active_set_idx).enabled:
            return False
        sp.chk_enable.setChecked(True)
        return True

    def _analyse_active_set(self) -> bool:
        """Analyse ▸ (selection panel, Analysis menu, halo table): show the Analyse tab with the
        active set as its population. False, with a message, when the set is empty."""
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        if not s_set.selected_ids:
            msg = f"Set {s_set.set_id + 1} is empty: select particles first, then Analyse ▸."
            sp.lbl_status.setText(msg)
            log(msg, 'warning')
            return False
        self._enable_active_set()
        self._analysis_handed = None                # hand it over even after "Use every particle"
        self._analysis_show_result = True
        self.view.tab_widget.setCurrentWidget(self.view.analyse_tab)
        if self._ensure_explorer():
            self._schedule_analysis_sync(0)
        return True

    def _analyse_every_particle(self):
        """Use every particle: until the active set changes or Analyse ▸ hands it over again."""
        ex = self._explorer
        if ex is None or not ex.open_snapshot_file():
            return
        ex.activate_population(None)
        self._analysis_handed = self._wanted_population()[1]
        self._analysis_problems.pop('population', None)
        self._update_analyse_bar()

    def _show_population_in_3d(self):
        """Show in 3D: the 3D view, centred on the active set (switched on if it was off), using its
        full-resolution positions where tracking has found them (_recentre_camera)."""
        self.view.tab_widget.setCurrentWidget(self.view.explore_tab)
        self._enable_active_set()
        self._recentre_camera()

    def _adopt_population(self, pop):
        """population_proposed: particles chosen in the Analyse view become the active set, with what
        they mean (kind, definition, defining snapshot, completeness)."""
        if getattr(pop, 'kind', None) not in KINDS:     # refused whole, never half applied
            self._analysis_problems['population'] = (f"Could not select the proposed particles: unknown kind "
                                                     f"{getattr(pop, 'kind', None)!r}.")
            log(self._analysis_problems['population'], 'error')
            self._update_analyse_bar()
            return
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        s_set.set_population(pop.ids, pop.kind, pop.definition, pop.defined_at, pop.complete, pop.n_expected)
        if getattr(pop, 'name', ''):
            s_set.name = pop.name
        log(f"Set {s_set.set_id + 1} ({s_set.name}), chosen in the Analyse view: {s_set.describe()}")
        if not self._enable_active_set():
            self._on_selection_changed()            # panel, tracking and the 3D view
        self._schedule_analysis_sync(0)             # and the Analyse view takes it as its population
