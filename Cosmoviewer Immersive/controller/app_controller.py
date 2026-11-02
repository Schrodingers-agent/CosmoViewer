"""
CosmoViewerController owns the model (SimulationModel) and the main window
(CosmoViewerWindow) and connects the two.

This file keeps the core of the application: finding and loading snapshots
(FileDiscoveryEngine, then AsyncSnapshotLoader in a background thread), playback
and the time slider, redrawing a frame, the run's unit labels, and the FoF group
finder, which FoFWorker runs at full resolution in a separate process so that
running out of memory costs the FoF run and leaves the viewer alone. The rest
arrives through the mixins on the class, one per controller/*.py file, each
started by its own _init_* call in __init__: tracking, colouring and catalogues
(features), picking, the particle inspector, selection-set files, level of
detail, view tools, the halo table (structures) and the Analyse tab. The dialogs
and the property plot window follow GadgetViewer's behaviour.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from model.simulation_model import SimulationModel
from views.main_window import CosmoViewerWindow
from views.fof_dialog import FoFGroupFinderDialog
from views.property_plot_window import PropertyPlotWindow
from views.group_catalogue_dialog import GroupCatalogueDialog
from snapshot_io.fof_finder import FoFGroupsEngine
from snapshot_io.fof_snapshot import fof_defaults
from snapshot_io.fof_process import FoFCancelled, FoFProcessError
from snapshot_io.halo_catalog import HaloCatalogLoader
from snapshot_io.group_catalogue import find_catalogue_for
from controller.features_controller import CAT_GROUP
from gadget_file_discovery import FileDiscoveryEngine, FileManifest, SnapshotInspector
from snapshot_io.snapshot_loader import AsyncSnapshotLoader
from controller.console import log
from controller.console_commands import attach_console, describe_snapshot
from controller.thread_hold import hold
from controller.tracking_controller import TrackingMixin
from controller.features_controller import FeaturesMixin
from controller.lod_controller import LODMixin
from controller.picking_controller import PickingMixin
from controller.inspector_controller import InspectorMixin
from controller.sets_controller import SetsMixin
from controller.view_tools_controller import ViewToolsMixin, is_cosmological, time_text
from controller.analysis_host import AnalysisHostMixin
from controller.structures_controller import StructuresMixin
from lod.budget import base_sample_size

_KPC_CM = 3.085678e21
_MSUN_G = 1.989e33
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _power_text(x: float) -> str:
    """1e10 -> "10¹⁰", 4.3e10 -> "4.3×10¹⁰", 250 -> "250": a unit's size, readable."""
    if not np.isfinite(x) or x <= 0:
        return f"{x:g}"
    e = int(np.floor(np.log10(x)))
    m = float(f"{x / 10.0 ** e:.3g}")
    if m >= 10.0:                       # 9.9996e9 rounds to 10×10⁹: write 10¹⁰
        m, e = m / 10.0, e + 1
    if -2 <= e <= 3:
        return f"{x:.3g}"
    p = "10" + str(e).translate(_SUPERSCRIPT)
    return p if m == 1.0 else f"{m:g}×{p}"


class FoFWorker(QThread):
    """Runs full-resolution FoF off the GUI thread, in its own process (snapshot_io/fof_process.py),
    so running out of memory or a crash stops only that process, never the viewer."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(float, str)
    cancelled = pyqtSignal()

    def __init__(self, filepath: str, display_ids: np.ndarray, settings: dict):
        super().__init__()
        self.filepath, self.display_ids, self.settings = filepath, display_ids, settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            s = self.settings
            self.done.emit(FoFGroupsEngine.detect_groups_in_snapshot(
                self.filepath, self.display_ids, b_linking=s['b_linking'], min_group_size=s['min_size'],
                r_link=s['r_link'], primary_types=s['link_types'], separate_process=True,
                progress=self.progress.emit, cancelled=lambda: self._cancel))
        except FoFCancelled:
            self.cancelled.emit()
        except FoFProcessError as e:  # the process's own message says what happened
            self.failed.emit(str(e))
        except Exception as e:  # reported to the user, never swallowed
            self.failed.emit(f"{type(e).__name__}: {e}")


class CosmoViewerController(TrackingMixin, FeaturesMixin, PickingMixin, InspectorMixin, SetsMixin, LODMixin,
                            ViewToolsMixin, StructuresMixin, AnalysisHostMixin):
    """
    Application Controller for CosmoViewer Immersive.
    Wires signals, drives the FoF group finder, property plots and rendering.
    """

    TYPE_COLORS = np.array([
        [0.0, 0.8, 1.0, 0.4],   # 0 Gas
        [0.8, 0.1, 0.1, 0.3],   # 1 Dark Matter
        [1.0, 1.0, 0.9, 0.8],   # 2 Disk Stars
        [1.0, 0.9, 0.5, 0.8],   # 3 Bulge Stars
        [0.5, 0.0, 1.0, 1.0],   # 4 Star Particles
        [0.2, 0.2, 0.2, 0.2],   # 5 Black holes (boundary particles in old Gadget-2 runs)
    ])

    def __init__(self):
        self.model = SimulationModel()
        self.view = CosmoViewerWindow(self.model.tracker)

        # Dialogs
        self.fof_dialog: Optional[FoFGroupFinderDialog] = None
        self.plot_window: Optional[PropertyPlotWindow] = None
        self.group_dialog: Optional[GroupCatalogueDialog] = None

        self.current_snap_idx: int = 0
        self.is_playing: bool = False
        self.color_mode: str = 'component'
        self.loading: bool = False

        # Random FoF Group Color Palette
        self.fof_colors_palette: Optional[np.ndarray] = None
        self._fof_worker: Optional[FoFWorker] = None

        self._connect_signals()
        self._init_tracking()
        self._init_features()
        self._init_picking()
        self._init_inspector()
        self._init_sets()
        self._init_lod()
        self._init_view_tools()
        self._init_structures()
        self._init_analysis_host()
        self.console = attach_console(self)

        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        self.timer.setInterval(80)

    def show(self):
        self.view.show()

    # ── Command-line and console import ──────────────────────────────────────

    def load_sources(self, plan) -> bool:
        """Open a cli_import.ImportPlan: folders are discovered, files used as given.

        Returns False (and logs why) when nothing could be started."""
        for err in plan.errors:
            log(f"Cannot open {err}", 'error')
        if self.loading:
            log("A load is already running; try again when it finishes.", 'warning')
            return False
        files = []
        for folder in plan.folders:
            engine = FileDiscoveryEngine(scan_subfolders=False, include_backups=False,
                                         validate_snapshots=True)
            manifest = engine.discover(folder)
            self.model.manifest = manifest
            found = manifest.get_snapshot_files()
            if not found:
                log(f"No snapshots found in {folder}", 'warning')
            files.extend(found)
        files.extend(plan.files)
        if not files:
            if plan.empty and not plan.errors:
                log("Nothing to open.", 'warning')
            self.view.set_status_text("NO SNAPSHOTS FOUND")
            return False
        if plan.files and not plan.folders:
            self.model.manifest = None        # Rescan then asks for a folder
        log(f"Opening {len(files)} snapshot file(s)", 'info')
        self.current_snap_idx = 0
        self._start_loader(files)
        return True

    def start_from_command_line(self, plan, commands=()):
        """Open launch-time sources once the window is up, then run console commands."""
        pending = list(commands)

        def run_commands():
            if self.loading:
                QTimer.singleShot(200, run_commands)
                return
            for line in pending:
                self.console.run(line)

        def start():
            if not plan.empty or plan.errors:
                self.load_sources(plan)
            if pending:
                QTimer.singleShot(200, run_commands)

        QTimer.singleShot(0, start)

    def _connect_signals(self):
        v = self.view
        v.directory_open_requested.connect(self._open_directory_dialog)
        v.rescan_requested.connect(self._rescan_directory)
        v.color_mode_set.connect(self._set_color_mode)
        v.play_toggled.connect(self._toggle_play)
        v.frame_changed.connect(self._on_frame_slider_changed)
        v.z_slider_changed.connect(self._update_frame)
        v.layer_panel.layer_changed.connect(self._update_frame)
        v.selection_panel.selection_changed.connect(self._on_selection_changed)
        v.selection_panel.recentre_requested.connect(self._recentre_camera)
        v.selection_panel.box_centre_requested.connect(self._centre_on_box)
        v.selection_panel.draw_selected_only_toggled.connect(lambda _on: self._update_frame())
        self._radius_timer = QTimer(v)      # keeps the radius centre label and marker current
        self._radius_timer.setInterval(200)
        self._radius_timer.timeout.connect(self._update_radius_marker)
        self._radius_timer.start()

        # Gadgetviewer menu action connections
        v.fof_dialog_requested.connect(self._open_fof_dialog)
        v.plot_window_requested.connect(self._open_plot_window)
        v.group_dialog_requested.connect(self._open_group_dialog)

    def _open_fof_dialog(self):
        if not self.fof_dialog:
            self.fof_dialog = FoFGroupFinderDialog(self.view)
            self.fof_dialog.fof_run_requested.connect(self._run_fof_algorithm)
            self.fof_dialog.halo_highlight_requested.connect(self._highlight_fof_halo)
            self.fof_dialog.recentre_halo_requested.connect(self._recentre_on_fof_halo)
            self.fof_dialog.color_by_fof_requested.connect(lambda: self._set_color_mode('fof_group'))
            self.fof_dialog.fof_cancel_requested.connect(self._cancel_fof)
            self._fof_configured_for = None
        snap = self.model.get_snapshot(self.current_snap_idx)
        run_dir = str(Path(snap['filepath']).parent) if snap and snap.get('filepath') else None
        if run_dir and run_dir != self._fof_configured_for:
            try:
                self.fof_dialog.configure_for_snapshot(fof_defaults(snap['filepath']))
                self._fof_configured_for = run_dir
            except Exception as e:
                self.fof_dialog.lbl_status.setText(f"Could not read FoF defaults: {e}")
        self.fof_dialog.show()
        self.fof_dialog.raise_()

    def _open_plot_window(self):
        if not self.plot_window:
            self.plot_window = PropertyPlotWindow(self.view)
            self.plot_window.plot_requested.connect(self._render_property_plot)
            self._last_prop_names = None
            self._refresh_property_list()
        self.plot_window.show()
        self.plot_window.raise_()

    def _open_group_dialog(self):
        if not self.group_dialog:
            self.group_dialog = GroupCatalogueDialog(self.view)
            self.group_dialog.catalog_load_requested.connect(self._load_group_catalog)
        self.group_dialog.show()
        self.group_dialog.raise_()

    def _run_fof_algorithm(self, settings: dict):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or 'c' not in snap:
            QMessageBox.warning(self.view, "FoF Error", "No snapshot particle data loaded.")
            return
        if self._fof_worker and self._fof_worker.isRunning():
            return
        if 'id' not in snap or not snap.get('filepath'):
            QMessageBox.warning(self.view, "FoF Error", "Snapshot has no particle IDs or source file.")
            return

        # FoF reads the full-resolution particles from the file; snap['c'] is only the display sample.
        worker = FoFWorker(snap['filepath'], snap['id'], settings)
        worker.done.connect(lambda result, s=snap, st=dict(settings): self._on_fof_done(s, result, st))
        worker.failed.connect(self._on_fof_failed)
        worker.cancelled.connect(self._on_fof_cancelled)
        if self.fof_dialog:
            worker.progress.connect(self.fof_dialog.set_progress)
        if self._fof_worker is None:
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._shutdown_fof)
        self._fof_worker = worker
        if self.fof_dialog:
            self.fof_dialog.set_running(True)
        hold(worker).start()

    def _cancel_fof(self):
        if self._fof_worker and self._fof_worker.isRunning():
            self._fof_worker.cancel()
            if self.fof_dialog:
                self.fof_dialog.btn_cancel.setEnabled(False)
                self.fof_dialog.lbl_status.setText("Cancelling FoF...")

    def _shutdown_fof(self):
        """On quit: stop a running FoF process and wait for its thread."""
        if self._fof_worker and self._fof_worker.isRunning():
            self._fof_worker.cancel()
            self._fof_worker.wait(5000)

    def _on_fof_cancelled(self):
        if self.fof_dialog:
            self.fof_dialog.set_running(False)
            self.fof_dialog.lbl_status.setText("FoF cancelled.")

    def _on_fof_failed(self, message: str):
        if self.fof_dialog:
            self.fof_dialog.set_running(False)
            self.fof_dialog.lbl_status.setText("FoF failed.")
        QMessageBox.critical(self.view, "FoF Error", message)

    def _on_fof_done(self, snap: dict, result, settings: Optional[dict] = None):
        group_ids, n_groups, group_sizes, fof = result
        snap['FoFGroupIndex'] = group_ids
        snap['fof_sizes'] = group_sizes
        snap['fof_linking_length'] = fof.result.linking_length
        snap['fof'] = fof                   # full-resolution lookup, to colour level-of-detail particles
        snap['fof_settings'] = dict(settings or {})   # what produced these groups (b, N_min, linked types)

        # Random distinct colours for groups, one per group of the snapshot with the most groups:
        # the palette is shared by every snapshot with FoF, and colouring wraps group numbers
        # beyond its length, which drew those groups in the grey of "in no group". A local
        # generator draws rows in order, so a group keeps its colour when the palette grows,
        # and the global numpy random state is left alone.
        n_max = max([n_groups] + [len(s.get('fof_sizes') or {}) for s in self.model.snapshots or []])
        palette = np.random.default_rng(42).random((n_max + 1, 4))
        palette[:, 3] = 0.8
        palette[0] = [0.2, 0.2, 0.2, 0.2]  # Unbound background particles
        self.fof_colors_palette = palette

        if self.fof_dialog:
            self.fof_dialog.set_running(False)
            self.fof_dialog.update_results(n_groups, group_sizes, fof.result.linking_length,
                                           int(len(fof.ids)), where=Path(snap['filepath']).name)
        self._fof_shown_snap = snap

        self._update_frame()

    # FoF results belong to the snapshot they were computed for (snap['fof'], the full-resolution
    # membership by ParticleID; snap['FoFGroupIndex'] maps only the display sample onto it).

    @staticmethod
    def _fof_members(snap: dict, grp_id: int) -> np.ndarray:
        """Every member of FoF group ``grp_id`` (1-based, as in the FoF window) of ``snap``.

        Taken from the full-resolution result, not from the display sample, which holds only
        some members of a group when the snapshot is bigger than the particle budget.
        Raises ValueError, with a message for the user, when there is no such group here."""
        name = Path(snap['filepath']).name if snap.get('filepath') else 'this snapshot'
        fof = snap.get('fof')
        if fof is None:
            raise ValueError(f"FoF has not been run on the snapshot shown ({name}); run it there "
                             f"(Structures > Find FoF groups...) or untick the FoF criterion.")
        n_groups = len(snap.get('fof_sizes') or {})
        if not 1 <= int(grp_id) <= n_groups:
            raise ValueError(f"FoF group {grp_id} does not exist in {name}: it has groups 1 to {n_groups}.")
        ids = np.asarray(fof.ids)[np.asarray(fof.group_index) == int(grp_id) - 1]
        return np.unique(ids.astype(np.uint64))

    def _fof_unavailable_here(self, snap: dict) -> bool:
        """True, and the FoF window says so, when the snapshot shown has no FoF result."""
        if snap and 'fof' in snap:
            return False
        if self.fof_dialog:
            self._sync_fof_dialog()          # says whose groups the window holds, if any
            if not any('fof' in s for s in self.model.snapshots or []):
                self.fof_dialog.lbl_status.setText("FoF has not been run on this snapshot yet.")
        return True

    def _sync_fof_dialog(self):
        """The FoF window lists the groups of the snapshot on screen, or says whose groups it holds."""
        dlg = self.fof_dialog
        if dlg is None or (self._fof_worker is not None and self._fof_worker.isRunning()):
            return
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            return
        shown = getattr(self, '_fof_shown_snap', None)
        if shown is not None and not any(s is shown for s in self.model.snapshots or []):
            shown = self._fof_shown_snap = None          # a new run was loaded
            dlg.reset()
        if 'fof' in snap:
            if shown is not snap:
                dlg.update_results(len(snap['fof_sizes']), snap['fof_sizes'], snap.get('fof_linking_length'),
                                   int(len(snap['fof'].ids)), where=Path(snap['filepath']).name)
                self._fof_shown_snap = snap
            return
        with_fof = [Path(s['filepath']).name for s in self.model.snapshots or [] if 'fof' in s]
        if with_fof:
            dlg.show_groups_elsewhere(with_fof, Path(snap['filepath']).name)
            self._fof_shown_snap = None

    def _defined_at(self, snap: dict, index: Optional[int] = None) -> dict:
        """Which snapshot a population was defined at: file, index, time and redshift where known."""
        if index is None:
            index = next((i for i, s in enumerate(self.model.snapshots or []) if s is snap), None)
        out = {'file': Path(snap['filepath']).name if snap.get('filepath') else ''}
        if index is not None:
            out['index'] = int(index)
        if snap.get('time') is not None:
            out['time'] = float(snap['time'])
        z = (snap.get('header') or {}).get('Redshift')
        if z is not None:
            try:
                out['redshift'] = float(z)
            except (TypeError, ValueError):
                pass
        return out

    def _fof_definition(self, snap: dict, grp_id: int) -> dict:
        """The 'group' definition of FoF group grp_id of snap (selection_sets.py, kind "group")."""
        st = dict(snap.get('fof_settings') or {})
        length = snap.get('fof_linking_length')
        d = {'finder': 'fof', 'group': int(grp_id), 'subhalo': False,
             'min_size': st.get('min_size'), 'link_types': st.get('link_types'),
             'convention': getattr(snap.get('fof'), 'convention', None)}
        if st.get('r_link') is None and st.get('b_linking') is not None:
            d['b'] = float(st['b_linking'])                 # the researcher's choice...
            d['linking_length_derived'] = None if length is None else float(length)   # ...and what it gave
        else:
            d['linking_length'] = None if length is None else float(length)
        return d

    def _highlight_fof_halo(self, grp_id: int):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or self._fof_unavailable_here(snap):
            return
        try:
            ids = self._fof_members(snap, grp_id)
        except ValueError as e:
            if self.fof_dialog:
                self.fof_dialog.lbl_status.setText(str(e))
            return
        if len(ids):
            active_set_idx = self.view.selection_panel.active_set_idx
            s_set = self.model.tracker.get_set(active_set_idx)
            s_set.set_population(ids, 'group', self._fof_definition(snap, grp_id), self._defined_at(snap),
                                 complete=True, n_expected=int((snap.get('fof_sizes') or {}).get(grp_id, len(ids))))
            log(f"Set {s_set.set_id + 1}: FoF group {grp_id} of {Path(snap['filepath']).name}, "
                f"all {len(ids):,} members")
            self.view.selection_panel._refresh_ui()
            self._update_frame()

    def _recentre_on_fof_halo(self, grp_id: int):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or self._fof_unavailable_here(snap):
            return

        mask = (snap['FoFGroupIndex'] == grp_id)
        if not np.any(mask) and self.fof_dialog:
            self.fof_dialog.lbl_status.setText(
                f"None of FoF group {grp_id}'s members are in the display sample; Select halo (all members), "
                f"then Recentre View in the selection panel (it uses every member).")
        if np.any(mask):
            pts = snap['c'][mask].astype(np.float64)
            box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
            if box > 0:  # unwrap periodic images around the first member
                d = pts - pts[0]
                pts = pts[0] + d - box * np.round(d / box)
            center = np.mean(pts, axis=0)
            mins = np.min(pts, axis=0)
            maxs = np.max(pts, axis=0)
            size = np.maximum(maxs - mins, 1.0)
            max_dim = float(np.max(size))
            self.view.frame_camera(center, max_dim)    # the box outline stays the simulation box

    def _open_directory_dialog(self):
        if self.loading:
            return
        directory = QFileDialog.getExistingDirectory(
            self.view, "Select Simulation Output Directory", os.getcwd())
        if directory:
            self._load_directory(Path(directory))

    def _rescan_directory(self):
        if self.is_playing:
            self._toggle_play()

        directory = (self.model.manifest.root_dir if self.model.manifest else None)
        self._reset_tracking()
        self._reset_features()
        self._reset_lod()
        self.model.clear_all_data()

        self.view.clear_scatter_3d()
        self.view.clear_scatter_2d()
        self.view.clear_layer_panel()
        self.view.set_time_slider_max(0)
        self.view.set_play_enabled(False)
        self.current_snap_idx = 0

        if directory:
            self._load_directory(directory)
        else:
            self._open_directory_dialog()

    def _load_directory(self, directory_path: Path):
        self.loading = True
        self.view.set_progress_visible(True)
        self.view.set_progress(0)
        self.view.set_status_text("SCANNING DIRECTORY...")
        self.view.set_play_enabled(False)

        engine = FileDiscoveryEngine(scan_subfolders=False, include_backups=False, validate_snapshots=True)
        manifest = engine.discover(directory_path)
        self.model.manifest = manifest

        if not manifest.snapshots:
            self.view.set_status_text("NO SNAPSHOTS FOUND")
            self.loading = False
            self.view.set_progress_visible(False)
            return

        snapshot_files = manifest.get_snapshot_files()
        
        self._start_loader(snapshot_files)

    def _start_loader(self, snapshot_files):
        self.loading = True
        self.view.set_progress_visible(True)
        self.view.set_play_enabled(False)
        # --max-particles sets the on-screen budget; each snapshot keeps a base sample of that
        # size unless that would not fit in memory with this many snapshots (lod/budget.py).
        self.lod_budget = int(getattr(self, 'max_particles', None) or self.lod_budget)
        loader = AsyncSnapshotLoader(snapshot_files, base_sample_size(self.lod_budget, len(snapshot_files)))
        # A loader replaced by a Rescan or a new folder can still finish (it was past its last
        # cancel check); what it read belongs to the old scan and is dropped, not installed.
        loader.progress.connect(lambda *a, l=loader: self._on_progress(*a) if l is self.model._loader_thread else None)
        loader.finished.connect(lambda *a, l=loader: self._on_load_complete(*a) if l is self.model._loader_thread else None)
        self.model._loader_thread = loader
        hold(loader).start()

    def _on_progress(self, current: int, total: int, msg: str):
        if total > 0:
            self.view.set_progress(int((current / total) * 100))
        self.view.set_status_text(msg)

    def _on_load_complete(self, snapshots, units, output_dir):
        self.loading = False
        self.view.set_progress_visible(False)

        loader = self.model._loader_thread
        errors = loader.errors if loader is not None else []
        if errors:
            # Full messages go to the console; the pop-up only says where to look.
            for err in errors:
                log(f"Snapshot load failed: {err}", 'error')
            n_tried = max(getattr(loader, 'n_requested', 0), len(errors) + len(snapshots))
            box = QMessageBox(QMessageBox.Icon.Warning, "Snapshot Load Errors",
                              f"{len(errors)} of {n_tried} snapshots failed to load.\n\n"
                              f"First error: {errors[0][:300]}", parent=self.view)
            details = box.addButton("Show in the log", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Ok)
            box.buttonClicked.connect(lambda b: self.view.show_telemetry_tab() if b is details else None)
            box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            box.open()                      # not modal-blocking; loading carries on
            self._load_error_box = box
        if not snapshots:
            self.view.set_status_text("NO SNAPSHOTS LOADED")
            return

        self._reset_tracking()
        self._reset_features()
        self._reset_lod()
        self.model.snapshots = snapshots
        self.model.units = units
        self._slice_key = None
        # Gadget-4 catalogues next to the snapshots are attached automatically
        # (GadgetViewer asks via File > Read groups; here Structures > Read halo catalogue... still can).
        found = [p for p in (find_catalogue_for(s['filepath']) for s in snapshots) if p]
        self._catalogue_mode = 'auto' if found else None
        if found:
            log(f"Found {len(found)} Gadget-4 group catalogue(s) for {len(snapshots)} snapshot(s); "
                f"membership is available as '{CAT_GROUP}'")

        if self.model.snapshots:
            mins = [np.min(s['c'], axis=0) for s in self.model.snapshots]
            maxs = [np.max(s['c'], axis=0) for s in self.model.snapshots]
            self.model.g_min = np.min(mins, axis=0)
            self.model.g_max = np.max(maxs, axis=0)

            size = self.model.g_max - self.model.g_min
            center = self.model.g_min + size / 2
            max_dim = float(np.max(size))
            self.view.set_box_geometry(size, self.model.g_min, center, max_dim)

            for snap in self.model.snapshots:
                path_str = snap['filepath']
                self.model.particle_type_cache[path_str] = SnapshotInspector.inspect(Path(path_str))

        n_loaded = len(self.model.snapshots)
        self.view.set_time_slider_max(n_loaded - 1)
        self.view.set_play_enabled(True)
        self.current_snap_idx = 0

        units_text = self._run_units_text()
        self.view.set_loaded_info(f"{n_loaded} snapshots loaded | {units_text}")
        log(f"Loaded {n_loaded} snapshots | {units_text}")
        self._show_radius_unit()
        for snap in self.model.snapshots:
            log("  " + describe_snapshot(snap))

        particle_types = self.model.get_particle_types_at_index(0)
        if particle_types:
            self.view.rebuild_layer_panel(particle_types)

        self._update_frame()

    @staticmethod
    def _describe_units(units, cosmological: Optional[bool] = None) -> str:
        """
        The code units, as stored in the file, in words. A cosmological Gadget run stores
        lengths in comoving UnitLength/h and masses in UnitMass/h, and says so:
        "code units: L = 1 Mpc/h (comoving), M = 10¹⁰ M☉/h, V = 1 km/s". A run without
        comoving integration: "code units: L = 1 kpc, M = 10¹⁰ M☉, V = 1 km/s".
        ``cosmological`` None (not known) gives the units without a convention.
        """
        if units is None:
            return "units unknown"
        length_kpc = units.UnitLength_in_cm / _KPC_CM
        length = f"{length_kpc / 1000:.4g} Mpc" if length_kpc >= 1000 else f"{length_kpc:.4g} kpc"
        mass = f"{_power_text(units.UnitMass_in_g / _MSUN_G)} M☉"
        vel = f"{units.UnitVelocity_in_cm_per_s / 1e5:.4g} km/s"
        if cosmological:
            length += "/h (comoving)"
            mass += "/h"
        return f"code units: L = {length}, M = {mass}, V = {vel}"

    @staticmethod
    def _length_unit_name(units, cosmological: bool) -> str:
        """
        One length in the 3D view and in selections, written as the view's scale bar writes it:
        "cMpc/h" (comoving Mpc per h, a cosmological run), "kpc" (a run without comoving integration).
        """
        if units is None:
            return "code units"
        length_kpc = units.UnitLength_in_cm / _KPC_CM
        unit, size = next(((n, s) for n, s in (('Mpc', 1000.0), ('kpc', 1.0), ('pc', 1e-3))
                           if abs(length_kpc / s - 1.0) < 1e-4), (None, None))
        prefix = ""
        if unit is None:            # an unusual unit: say how big one code length is
            unit, size = ('Mpc', 1000.0) if length_kpc >= 1000 else ('kpc', 1.0)
            prefix = f"× {length_kpc / size:.4g} "
        return prefix + (f"c{unit}/h" if cosmological else unit)

    def _run_snapshot(self) -> Optional[dict]:
        snaps = self.model.snapshots or []
        return snaps[0] if snaps else None

    def _run_is_cosmological(self) -> Optional[bool]:
        """From the header and parameters of the run's first snapshot; None before a run is loaded."""
        s = self._run_snapshot()
        return None if s is None else is_cosmological(s.get('header') or {}, s.get('unit_source'))

    def _run_units_stated(self) -> bool:
        """False when the files do not give their units (GadgetUnits then assumes Gadget's defaults)."""
        s = self._run_snapshot()
        return s is not None and 'UnitLength_in_cm' in (s.get('unit_source') or {})

    def _run_units_text(self) -> str:
        """_describe_units for the loaded run, with its convention (header line, log, summary, console)."""
        text = self._describe_units(self.model.units, self._run_is_cosmological())
        if self.model.units is not None and self._run_snapshot() is not None and not self._run_units_stated():
            text += " (assumed: the files do not state their units)"
        return text

    def _run_length_unit(self) -> str:
        """The unit next to the radius criterion; "code units" when the files do not state it."""
        if not self._run_units_stated():
            return "code units"
        return self._length_unit_name(self.model.units, bool(self._run_is_cosmological()))

    def _show_radius_unit(self):
        """Set the radius criterion's unit, and say in its tooltip what it means."""
        lbl = self.view.selection_panel.lbl_radius_unit
        unit = self._run_length_unit()
        lbl.setText(unit)
        if unit == "code units":
            tip = "The snapshot files do not state their length unit"
        elif self._run_is_cosmological():
            words = {"cMpc/h": "comoving Mpc per h", "ckpc/h": "comoving kpc per h"}.get(unit, "comoving, per h")
            tip = f"{unit}: {words}, the length unit as stored in the snapshot"
        else:
            tip = f"{unit}: the length unit as stored in the snapshot"
        lbl.setToolTip(tip)

    def _toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.view.set_play_button_text("Play")
        else:
            self.timer.start()
            self.view.set_play_button_text("Pause")
        self.is_playing = not self.is_playing

    def _next_frame(self):
        idx = (self.current_snap_idx + 1) % len(self.model.snapshots)
        self.view.set_time_slider_value(idx)
        self.current_snap_idx = idx
        self._update_frame()

    def _on_frame_slider_changed(self, val: int):
        self.current_snap_idx = val
        self._update_frame()

    def _recentre_camera(self):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or 'c' not in snap:
            return
        
        active_set_idx = self.view.selection_panel.active_set_idx
        s_set = self.model.tracker.get_set(active_set_idx)
        
        if s_set.selected_ids and 'id' in snap:
            # Full-resolution positions of the set if looked up, else the displayed sample.
            loc = self._tracked(self.current_snap_idx, s_set)
            pts = loc.pos[loc.found].astype(np.float64) if loc is not None \
                else snap['c'][np.isin(snap['id'], list(s_set.selected_ids))].astype(np.float64)
            if len(pts):
                box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
                if box > 0:  # unwrap periodic images around the first member
                    d = pts - pts[0]
                    pts = pts[0] + d - box * np.round(d / box)
                center = np.mean(pts, axis=0)
                mins = np.min(pts, axis=0)
                maxs = np.max(pts, axis=0)
                size = np.maximum(maxs - mins, 1e-6 * max(1.0, float(np.max(np.abs(center)))))
                max_dim = float(np.max(size))
                self.view.frame_camera(center, max_dim)    # the box outline stays the simulation box
                self._followed_idx = self.current_snap_idx
                return

        size = self.model.g_max - self.model.g_min
        center = self.model.g_min + size / 2.0
        max_dim = float(np.max(size))
        self.view.set_box_geometry(size, self.model.g_min, center, max_dim)

    def _on_selection_changed(self):
        """A set changed (criteria applied, cleared, enabled or disabled): refresh and re-track."""
        self.view.selection_panel._refresh_ui()
        self._start_tracking()
        self._update_frame()

    def _update_frame(self):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            return

        particle_types = self.model.get_particle_types_at_index(self.current_snap_idx)
        if particle_types:
            self.view.rebuild_layer_panel(particle_types)
        self._refresh_property_list()

        # Points, colours and selected particles are built by _build_scene, which the
        # movie maker also uses, so a movie frame shows what the window shows.
        self._maybe_start_tracking()
        ui = self._ui_state()
        scene = self._build_scene(self.current_snap_idx, ui)
        c_vis = scene.pos
        n_lit = 0 if scene.lit_pos is None else len(scene.lit_pos)
        if len(c_vis) > 0 or n_lit or scene.highlight_pos is not None:
            self.view.update_scatter_3d(c_vis, scene.colors)
            self.view.update_scatter_lit(scene.lit_pos if n_lit else np.empty((0, 3)),
                                         scene.lit_colors if n_lit else np.empty((0, 4)))
            if scene.highlight_pos is not None:
                self.view.update_scatter_highlight(scene.highlight_pos, scene.highlight_colors)
            else:
                self.view.clear_scatter_highlight()
        else:
            self.view.clear_scatter_3d()
        self.view.apply_drawing(ui['drawing'], scene.lit_pos is not None)
        self._remember_pick_candidates(scene)
        self.view.update_trails(*self._lines_to_segments(scene.lines))
        self._follow_selection(self.current_snap_idx, scene)
        self._after_scene(scene)
        self._update_tracking_label()
        self._refresh_inspector()
        self._sync_fof_dialog()
        self._sync_analysis_view()      # the Analyse tab follows this snapshot and the active set

        n_sel = 0 if scene.highlight_pos is None else len(scene.highlight_pos)
        extra = self._features_status(snap)
        n_text = (f"N={len(c_vis) + n_lit:,}" if scene.lit_pos is None
                  else f"N={len(c_vis) + n_lit:,} (lit {n_lit:,})")
        self.view.set_status_text(f"{time_text(snap)} | {n_text}" + (f" | selected {n_sel:,}" if n_sel else "")
                                  + (f" | {extra}" if extra else "") + self._lod_status())
        self.view.set_colour_bar(self._ui_state()['colouring'])
        self._update_overlays(snap)
        if not self.is_playing:
            self._auto_update_plot()

        if not self.view.extras.panel_wanted:
            self._slice_key = None      # hidden 2D panel: nothing to draw until it is shown again
        elif self.view.panel2d_mode() != 0:
            self._slice_key = None
            self._render_panel2d()      # smoothed image of the 3D view (FeaturesMixin)
        elif self.model.g_max is not None and self.model.g_min is not None:
            self._update_z_slice(snap)

    SLICE_MAX_POINTS = 20_000

    def _update_z_slice(self, snap: dict):
        """
        Z-slice of the base sample. It does not depend on the level-of-detail view set,
        so a view-set swap (camera move) redraws nothing here; at most SLICE_MAX_POINTS
        points, a fixed random subset, so the slice is cheap and steady.
        """
        key = (self.current_snap_idx, id(snap), self.view.get_z_slider_value(),
               tuple(self.model.g_min), tuple(self.model.g_max))
        if key == getattr(self, '_slice_key', None):
            return
        self._slice_key = key
        total_z = float(self.model.g_max[2] - self.model.g_min[2])
        slice_z = (self.model.g_min[2] + (self.view.get_z_slider_value() / 100.0) * total_z)
        thk = total_z * 0.03
        rows = np.flatnonzero(np.abs(snap['c'][:, 2] - slice_z) < (thk / 2))
        if len(rows) > self.SLICE_MAX_POINTS:
            rows = np.sort(np.random.default_rng(0).choice(rows, self.SLICE_MAX_POINTS, replace=False))
        if len(rows):
            s_c = snap['c'][rows]
            # one brush for all points: a list of one brush per point is slow for large slices
            self.view.update_scatter_2d(s_c[:, 0], s_c[:, 1], pg.mkBrush(255, 255, 255, 150))
        else:
            self.view.clear_scatter_2d()
