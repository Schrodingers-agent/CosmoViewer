"""
Particle tracking across snapshots and movie making for the controller.

A mixin of CosmoViewerController, which owns ``self.model`` and ``self.view``.
It applies the selection panel's criteria, loads and saves ID files, exports
trajectories, builds the scene that both the window and movie frames draw
(``_build_scene``), and runs the movie maker in a worker thread.

Tracking follows GadgetViewer: a selection set is a list of ParticleIDs that
stays selected when the snapshot changes, and the view can follow it. Unlike
the display sample, the selected particles are found in every snapshot at full
resolution (snapshot_io.id_lookup), in a background thread, and cached per
(snapshot, set, set version).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

import selection_sets
from controller.console import log
from controller.thread_hold import hold
from controller.view_tools_controller import is_cosmological
from model.tracking import (build_trajectories, export_csv, export_npz, parse_id_text, periodic_mean,
                            read_id_file, sphere_wraps, unwrap_for_display, unwrap_series,
                            write_id_file)
from movie.maker import MovieSettings, Scene, make_movie
from movie.renderer import Camera
from snapshot_io.id_lookup import LocatedParticles, locate_ids, select_in_sphere
from snapshot_io.properties import SNAP_GROUP, SNAP_SUBHALO
from views.colouring_panel import TYPE_MODE

# Above this many (particle, snapshot) entries the background lookup asks first:
# about 28 bytes each in the cache (float32 position, velocity and mass).
TRACK_CACHE_WARN_ENTRIES = 20_000_000
KPC_CM = 3.085678e21


def code_length_label(snap: dict) -> str:
    """
    Short name of the unit positions are stored in, for the definitions of selections:
    "Mpc/h (comoving)" for a cosmological Gadget run with Mpc code lengths, "kpc" for a
    non-cosmological run in kpc. Read from the snapshot's header and parameters
    (UnitLength_in_cm; ComovingIntegrationOn, or else a Time that is the scale factor
    matching the Redshift). Cosmological Gadget and EAGLE lengths are per h as stored.
    "code units" when the file does not say: the length unit is never guessed.
    (The window's own unit line is app_controller._describe_units.)
    """
    src = dict(snap.get('header') or {})
    src.update(snap.get('unit_source') or {})

    def first(key):
        try:
            return float(np.atleast_1d(src[key])[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    length = first('UnitLength_in_cm')
    if length is None or not math.isfinite(length) or length <= 0:
        return "code units"
    kpc = length / KPC_CM
    if abs(kpc / 1000.0 - 1.0) < 1e-3:
        unit = "Mpc"
    elif abs(kpc - 1.0) < 1e-3:
        unit = "kpc"
    else:
        unit = f"× {kpc / 1000.0:.4g} Mpc" if kpc >= 1000.0 else f"× {kpc:.4g} kpc"
    comoving = first('ComovingIntegrationOn')
    cosmological = bool(comoving) if comoving is not None else is_cosmological(snap.get('header') or {})
    return f"{unit}/h (comoving)" if cosmological else unit


def _open_bound(v: float) -> Optional[float]:
    """A range bound for a definition: a number, or None for an open (infinite) bound."""
    return float(v) if math.isfinite(v) else None


class TrackWorker(QThread):
    """Looks up selected IDs in each snapshot file, nearest snapshots first."""
    located = pyqtSignal(int, int, int, object)     # snap_idx, set_id, version, LocatedParticles
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, int)

    def __init__(self, jobs: List[Tuple[int, List[str], int, int, np.ndarray]]):
        super().__init__()
        self.jobs = jobs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for k, (snap_idx, files, set_id, version, ids) in enumerate(self.jobs):
            if self._cancel:
                return
            try:
                self.located.emit(snap_idx, set_id, version, locate_ids(files, ids))
            except Exception as e:
                self.failed.emit(snap_idx, f"{Path(files[0]).name}: {type(e).__name__}: {e}")
            self.progress.emit(k + 1, len(self.jobs))


class MovieWorker(QThread):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(object)                        # MovieResult

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.done.emit(self.fn(lambda i, n, m: self.progress.emit(i, n, m), lambda: self._cancel))
        except Exception as e:                       # reported, never swallowed
            from movie.maker import MovieResult
            self.done.emit(MovieResult(False, f"{type(e).__name__}: {e}"))


class TrackingMixin:

    def _init_tracking(self):
        self._track_cache: Dict[Tuple[int, int], Tuple[int, LocatedParticles]] = {}
        self._track_worker: Optional[TrackWorker] = None
        self._followed_idx: Optional[int] = None
        self._tracking_started_for: Dict[int, int] = {}      # set_id -> version last looked up
        self.movie_dialog = None
        self._movie_worker: Optional[MovieWorker] = None
        sp = self.view.selection_panel
        sp.apply_requested.connect(self._apply_selection_criteria)
        sp.ids_load_requested.connect(self._load_ids)
        sp.ids_save_requested.connect(self._save_ids)
        sp.export_tracks_requested.connect(self._export_trajectories)
        sp.tracking_options_changed.connect(self._update_frame)
        self.view.movie_dialog_requested.connect(self._open_movie_dialog)

    def _stop_track_worker(self):
        """Cancel the lookup in flight without waiting for it. A lookup in a big file can take longer
        than any wait the window could afford; it was let go after 5 s while still running, so Qt
        could free it mid-read. hold() keeps it until it has ended, and _on_located ignores what it
        still sends because it is no longer the current worker."""
        if self._track_worker and self._track_worker.isRunning():
            self._track_worker.cancel()
        self._track_worker = None

    def _reset_tracking(self):
        self._stop_track_worker()
        self._track_cache.clear()
        self._followed_idx = None
        self._tracking_started_for = {}

    def _maybe_start_tracking(self):
        """Start lookups for sets changed by code that did not ask (e.g. FoF 'highlight halo')."""
        if not self.model.snapshots:
            return
        now = {s.set_id: s.version for s in self.model.tracker.sets if s.selected_ids}
        if any(self._tracking_started_for.get(k) != v for k, v in now.items()):
            self._start_tracking()

    # ------------------------------------------------------------------ lookups

    @staticmethod
    def _snap_files(snap: dict) -> List[str]:
        return list(snap.get('files') or [snap['filepath']])

    def _tracked(self, idx: int, s_set, compute: bool = False) -> Optional[LocatedParticles]:
        """Full-resolution lookup of a set in snapshot ``idx``, if cached (or computed)."""
        if not s_set.selected_ids:
            return None
        hit = self._track_cache.get((idx, s_set.set_id))
        if hit is not None and hit[0] == s_set.version:
            return hit[1]
        if not compute:
            return None
        snap = self.model.get_snapshot(idx)
        loc = locate_ids(self._snap_files(snap), s_set.id_array())
        self._track_cache[(idx, s_set.set_id)] = (s_set.version, loc)
        return loc

    def _start_tracking(self):
        """Look up every non-empty set in every loaded snapshot, current snapshot first."""
        self._stop_track_worker()
        # Drop lookups of old versions so memory follows the current selection.
        current = {s.set_id: s.version for s in self.model.tracker.sets}
        # list(): a movie's worker thread may add lookups meanwhile (_tracked with compute=True).
        self._track_cache = {k: v for k, v in list(self._track_cache.items()) if v[0] == current.get(k[1])}
        snaps = self.model.snapshots or []
        n = len(snaps)
        self._tracking_started_for = {s.set_id: s.version for s in self.model.tracker.sets if s.selected_ids}
        order = sorted(range(n), key=lambda i: (abs(i - self.current_snap_idx), i))
        jobs, entries = [], 0
        for s_set in self.model.tracker.sets:
            if not s_set.selected_ids:
                continue
            ids = s_set.id_array()
            entries += len(ids) * n
            for i in order:
                hit = self._track_cache.get((i, s_set.set_id))
                if hit is None or hit[0] != s_set.version:
                    jobs.append((i, self._snap_files(snaps[i]), s_set.set_id, s_set.version, ids))
        if not jobs:
            self._update_tracking_label()
            return
        if entries > TRACK_CACHE_WARN_ENTRIES:
            ans = QMessageBox.question(
                self.view, "Large selection",
                f"Tracking {entries:,} particle-snapshot entries needs about {entries * 28 / 1e9:.1f} GB "
                f"of memory. Track them in every snapshot anyway?\n\n"
                f"'No' tracks them in the current snapshot only.")
            if ans != QMessageBox.StandardButton.Yes:
                jobs = [j for j in jobs if j[0] == self.current_snap_idx]
        worker = TrackWorker(jobs)
        # Results of a worker that was replaced (new lookups, or a new run loaded: the snapshot
        # numbers then mean other files) are dropped.
        worker.located.connect(lambda *a, w=worker: self._on_located(*a) if w is self._track_worker else None)
        worker.failed.connect(lambda i, msg: log(f"Tracking lookup failed: {msg}", 'error'))
        worker.progress.connect(lambda k, total, w=worker: self._update_tracking_label(f"looking up {k}/{total}")
                                if w is self._track_worker else None)
        worker.finished.connect(lambda w=worker: self._update_tracking_label() if w is self._track_worker else None)
        self._track_worker = worker
        hold(worker).start()

    def _on_located(self, snap_idx: int, set_id: int, version: int, loc: LocatedParticles):
        s_set = self.model.tracker.get_set(set_id)
        if s_set.version != version:
            return
        self._track_cache[(snap_idx, set_id)] = (version, loc)
        if loc.duplicates:
            log(f"Set {set_id + 1}, snapshot {snap_idx}: {loc.duplicates} extra particles share a "
                f"selected ID (IDs are not unique); the first one is used", 'warning')
        if snap_idx == self.current_snap_idx or self.view.selection_panel.chk_trails.isChecked():
            self._update_frame()

    def _update_tracking_label(self, busy: str = ""):
        n = len(self.model.snapshots or [])
        parts = []
        for s_set in self.model.tracker.sets:
            if not s_set.selected_ids:
                continue
            done = sum(1 for i in range(n)
                       if self._track_cache.get((i, s_set.set_id), (None,))[0] == s_set.version)
            loc = self._tracked(self.current_snap_idx, s_set)
            here = f"{loc.n_found:,} of {len(loc.ids):,} found here" if loc is not None else "not looked up here yet"
            parts.append(f"Set {s_set.set_id + 1}: {here}; {done}/{n} snapshots")
        if busy:
            parts.append(busy)
        self.view.selection_panel.set_tracking_text("\n".join(parts))

    # ------------------------------------------------------------------ selection

    def _view_centre(self) -> np.ndarray:
        """Centre of a radius selection: the 3D view's orbit centre (it moves with pan, recentre
        and wheel zoom, which zooms toward the cursor), else the middle of the data."""
        cam = self.view.camera_state()
        if cam:
            return np.array(cam['center'], dtype=np.float64)
        return (self.model.g_min + self.model.g_max) / 2.0

    def _centre_on_box(self):
        """Put the view centre (the radius criterion's centre) in the middle of the box."""
        snap = self.model.get_snapshot(self.current_snap_idx)
        box = float((snap or {}).get('header', {}).get('BoxSize', 0.0) or 0.0)
        if box > 0:
            self.view.set_camera_centre(np.full(3, box / 2.0))
        elif self.model.g_min is not None and self.model.g_max is not None:
            self.view.set_camera_centre((self.model.g_min + self.model.g_max) / 2.0)

    def _update_radius_marker(self):
        """Label and 3D marker for where a radius selection would be measured from."""
        sp = self.view.selection_panel
        snap = self.model.get_snapshot(self.current_snap_idx) if self.model.snapshots else None
        on = bool(snap) and sp.chk_radius.isChecked() and self.model.g_min is not None
        key = (on, tuple(self._view_centre()) if on else None, sp.spin_radius.value(), self.current_snap_idx)
        if key == getattr(self, '_radius_marker_key', None):
            return
        self._radius_marker_key = key
        if not on:
            sp.lbl_radius_centre.setText("")
            self.view.update_radius_marker(None, 0.0, False)
            return
        centre, radius = self._view_centre(), sp.spin_radius.value()
        box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
        wraps = sphere_wraps(centre, radius, box)
        text = f"about the view centre ({', '.join(f'{x:.4g}' for x in centre)})"
        if wraps:
            text += f"; crosses the box edge in {', '.join(wraps)}, so it wraps (periodic)"
        sp.lbl_radius_centre.setText(text)
        self.view.update_radius_marker(centre, radius, True)

    @staticmethod
    def _sample_is_everything(snap: dict) -> bool:
        """The display sample holds every particle of the snapshot (it fitted in the budget), so a
        search of the sample is a search of the whole snapshot."""
        n_total = int(snap.get('n_total', 0) or 0)
        return n_total > 0 and len(snap['c']) >= n_total

    def _apply_selection_criteria(self):
        """Apply the checked criteria (AND) to the active set: new, add or subset.

        A new set records what it means (selection_sets.py). One criterion gives its own kind:
        a sphere is a "region", a FoF or catalogue group a "group", typed IDs are "ids"; a
        property range, or several criteria, give a "query" that lists them. The set is
        complete only when every criterion was evaluated on every particle of the snapshot:
        radius and property searches of the display sample are not (unless the sample holds
        every particle), and a catalogue group whose members are not all in the snapshot
        files is not either. Each criterion's definition is recorded as it was evaluated."""
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            return
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
        chosen: Optional[np.ndarray] = None
        criteria: List[dict] = []         # kind, definition, full (every particle?), n_expected, n_found
        everything = self._sample_is_everything(snap)

        def combine(ids) -> int:
            nonlocal chosen
            ids = np.unique(np.asarray(ids, dtype=np.uint64))
            chosen = ids if chosen is None else np.intersect1d(chosen, ids)
            return len(ids)

        def visible_types() -> List[int]:
            return [t for t in range(6) if self.view.layer_panel.layer_state.is_visible(t)]

        try:
            if sp.chk_ids.isChecked():
                n = combine(parse_id_text(sp.edit_ids.text()))
                criteria.append(dict(kind='ids', definition={'source': 'typed', 'n': n}, full=True))
            if sp.chk_radius.isChecked():
                centre, radius = self._view_centre(), sp.spin_radius.value()
                visible = visible_types()
                if sp.chk_full.isChecked():
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                    try:
                        combine(select_in_sphere(self._snap_files(snap), centre, radius, box, visible))
                    finally:
                        QApplication.restoreOverrideCursor()
                else:
                    d = snap['c'].astype(np.float64) - centre
                    if box > 0:
                        d -= box * np.round(d / box)
                    m = (np.einsum('ij,ij->i', d, d) <= radius * radius) & np.isin(snap['t'], visible)
                    combine(snap['id'][m])
                full = sp.chk_full.isChecked() or everything
                wraps = sphere_wraps(centre, radius, box)
                criteria.append(dict(kind='region', full=full, definition={
                    'shape': 'sphere', 'centre': [float(x) for x in centre], 'radius': float(radius),
                    'units': code_length_label(snap), 'periodic': box > 0, 'wraps': wraps,
                    'types': visible, 'scope': 'full' if full else 'sample'}))
                log(f"Radius selection: r={radius:g} about ({', '.join(f'{x:.4g}' for x in centre)}), "
                    f"{'full snapshot' if sp.chk_full.isChecked() else 'displayed sample'}"
                    + (f"; crosses the periodic boundary in {', '.join(wraps)}" if wraps else ""))
            if sp.chk_fof.isChecked():
                # every member, from the full-resolution FoF result of this snapshot; an error
                # (not a silently dropped criterion) when this snapshot has no such group
                g = int(sp.spin_fof_id.value())
                n = combine(self._fof_members(snap, g))
                criteria.append(dict(kind='group', definition=self._fof_definition(snap, g), full=True,
                                     n_expected=int((snap.get('fof_sizes') or {}).get(g, n)), n_found=n))
            if sp.chk_prop.isChecked():
                name, lo, hi = sp.property_range()
                visible = visible_types()
                info = self._file_properties(snap).get(name)
                combine(self._select_by_property(snap, name, lo, hi, sp.chk_full.isChecked(), visible))
                # only file properties can be searched in the file; attached ones (FoF and
                # catalogue groups, extra data) exist for the display sample only
                full = (sp.chk_full.isChecked() and info is not None) or everything
                criteria.append(dict(kind='property', full=full, definition={
                    'field': name, 'lo': _open_bound(lo), 'hi': _open_bound(hi),
                    'units': info.unit_label if info is not None else '', 'types': visible,
                    'scope': 'full' if full else 'sample'}))
                log(f"Property selection: {lo:g} <= {name} <= {hi:g}, "
                    f"{'full snapshot' if full else 'displayed sample'}")
                if sp.chk_full.isChecked() and not full:
                    log(f"{name} is known for the display sample only, so the sample was searched "
                        f"for it, not the full snapshot", 'warning')
            if sp.chk_cat.isChecked():
                g, subhalo = int(sp.spin_cat.value()), sp.combo_cat.currentText() == 'subhalo'
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    n = combine(self._select_catalogue_group(snap, g, subhalo))
                finally:
                    QApplication.restoreOverrideCursor()
                definition, n_expected = self._catalogue_definition(snap, g, subhalo)
                criteria.append(dict(kind='group', definition=definition, full=True,
                                     n_expected=n_expected, n_found=n))
        except ValueError as e:
            QMessageBox.warning(self.view, "Selection", f"Could not apply the criteria: {e}")
            return
        if chosen is None:
            return
        self._apply_ids_to_set(s_set, chosen, sp.get_selection_mode(),
                               self._criteria_meaning(snap, criteria, len(chosen)))

    def _catalogue_definition(self, snap: dict, group: int, subhalo: bool) -> Tuple[dict, Optional[int]]:
        """(definition, length) of catalogue group or subhalo ``group`` as _select_catalogue_group
        reads it: from the attached Gadget-4 catalogue (length = its GroupLen or SubhaloLen), or
        from membership stored in the snapshot (EAGLE GroupNumber; length unknown)."""
        d = {'finder': 'catalogue', 'group': int(group), 'subhalo': bool(subhalo)}
        if not self._catalogue_mode:
            d['stored_in'] = 'snapshot'
            d['field'] = SNAP_SUBHALO if subhalo else SNAP_GROUP
            return d, None
        path, cat = self._catalogue_for(snap)
        if path is None:
            return d, None
        d['catalogue_file'] = Path(path).name
        lengths = cat.sub_len if subhalo else cat.group_len
        if lengths is None or not 1 <= group <= len(lengths):
            return d, None
        return d, int(np.asarray(lengths[group - 1]).sum())

    def _criteria_meaning(self, snap: dict, criteria: List[dict], n_chosen: int) -> Optional[dict]:
        """What the criteria select (keyword arguments of ParticleSelectionSet.set_population)."""
        if not criteria:
            return None
        if len(criteria) == 1 and criteria[0]['kind'] in ('ids', 'region', 'group'):
            c = criteria[0]
            kind = c['kind']
            definition = {'source': 'typed'} if kind == 'ids' else dict(c['definition'])
            n_expected = c.get('n_expected')
        else:
            kind = 'query'
            definition = {'criteria': [{'kind': c['kind'], **c['definition']} for c in criteria]}
            n_expected = None
        missing = any(c.get('n_expected') is not None and c.get('n_found', 0) < c['n_expected']
                      for c in criteria)
        if not all(c['full'] for c in criteria):
            complete, reason = False, 'display_sample'
        elif missing:
            complete, reason = False, 'missing_in_file'
        else:
            complete, reason = True, None
            if n_expected is None and kind != 'ids':
                n_expected = n_chosen              # evaluated on every particle: this is its size
        return dict(kind=kind, definition=definition, defined_at=self._defined_at(snap, self.current_snap_idx),
                    complete=complete, n_expected=n_expected, incomplete_reason=reason)

    def _apply_ids_to_set(self, s_set, ids: np.ndarray, mode: str, meaning: Optional[dict] = None):
        """Put ``ids`` into the set: replace it ('new'), add to it or keep the overlap ('subset').

        A new set records ``meaning`` (keyword arguments of set_population); adding to an
        empty set is the same as a new one. Add and subset turn a set with a definition into
        plain IDs that remember what they were edited from (model/simulation_model.py)."""
        if mode == 'new' or (mode == 'add' and not s_set.selected_ids):
            if meaning:
                s_set.set_population(ids, **meaning)
            else:
                s_set.select_by_ids(ids)
        elif mode == 'add':
            s_set.add_ids(ids)
        else:
            s_set.intersect_ids(ids)
        log(f"Set {s_set.set_id + 1}: {len(s_set.selected_ids):,} particles selected: {s_set.describe()}")
        warning = s_set.completeness_warning()
        if warning:
            log(f"Set {s_set.set_id + 1}: {warning}", 'warning')
        self._followed_idx = self.current_snap_idx      # follow from the next snapshot change
        self._on_selection_changed()

    def _load_ids(self):
        f, _ = QFileDialog.getOpenFileName(self.view, "Load particle IDs", "",
                                           "ID lists (*.txt *.dat *.csv *.npy);;Selection sets (*.hdf5);;All files (*)")
        if not f:
            return
        self.load_ids_file(f)

    def load_ids_file(self, f: str) -> bool:
        """Load IDs into the active set (mode as chosen in the panel). A selection-set file keeps
        the meaning written in it (version 1 files: plain IDs, completeness unknown); any other
        ID list becomes plain IDs from that file, whose completeness is not known."""
        snap = self.model.get_snapshot(self.current_snap_idx)
        try:
            if str(f).endswith(('.hdf5', '.h5')) and selection_sets.is_set_file(f):
                st = selection_sets.read_set(f)
                ids = st.ids
                meaning = dict(kind=st.kind, definition=st.definition, defined_at=st.defined_at,
                               complete=st.complete, n_expected=st.n_expected,
                               incomplete_reason=st.incomplete_reason)
            else:
                ids = read_id_file(f)
                meaning = dict(kind='ids', definition={'source': 'file', 'file': str(f)},
                               defined_at=self._defined_at(snap, self.current_snap_idx) if snap else {})
        except Exception as e:
            QMessageBox.warning(self.view, "Load IDs", f"Could not read {Path(f).name}: {e}")
            return False
        sp = self.view.selection_panel
        self._apply_ids_to_set(self.model.tracker.get_set(sp.active_set_idx), ids, sp.get_selection_mode(), meaning)
        return True

    def _save_ids(self):
        s_set = self.model.tracker.get_set(self.view.selection_panel.active_set_idx)
        if not s_set.selected_ids:
            QMessageBox.information(self.view, "Save IDs", "The active set is empty.")
            return
        f, _ = QFileDialog.getSaveFileName(self.view, "Save particle IDs", f"{s_set.name}.txt",
                                           "Text, one ID per line (*.txt);;"
                                           "Selection set, with what it means (*.hdf5)")
        if f:
            self.save_ids_file(f, s_set)

    def save_ids_file(self, f: str, s_set) -> str:
        """Save a set's IDs: a selection-set file (.hdf5) keeps what the set means, so Load IDs
        restores it; a text list carries the description in its header line only."""
        snap = self.model.get_snapshot(self.current_snap_idx)
        here = Path(snap['filepath']).name if snap else '?'
        if str(f).endswith(('.hdf5', '.h5')):
            selection_sets.write_set(snap['filepath'] if snap else f, s_set.id_array(), s_set.name,
                                     self._set_colour_hex(s_set), made_in="CosmoViewer Immersive",
                                     description=s_set.describe(), path=f, **s_set.meaning())
        else:
            write_id_file(f, s_set.selected_ids, header=f"{s_set.name}: {s_set.describe()}; saved at {here}")
        log(f"Saved {len(s_set.selected_ids):,} IDs to {f}")
        return f

    # ------------------------------------------------------------------ trajectories

    def _ensure_located(self, s_set, title: str) -> Optional[List[LocatedParticles]]:
        """Lookups of a set in every loaded snapshot, computing missing ones with a progress dialog."""
        n = len(self.model.snapshots)
        missing = [i for i in range(n) if self._tracked(i, s_set) is None]
        if missing:
            dlg = QProgressDialog(title, "Cancel", 0, len(missing), self.view)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            for k, i in enumerate(missing):
                dlg.setValue(k)
                QApplication.processEvents()
                if dlg.wasCanceled():
                    return None
                self._tracked(i, s_set, compute=True)
            dlg.setValue(len(missing))
        return [self._tracked(i, s_set) for i in range(n)]

    def _export_trajectories(self):
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        if not s_set.selected_ids or not self.model.snapshots:
            QMessageBox.information(self.view, "Export trajectories", "Select particles in the active set first.")
            return
        f, filt = QFileDialog.getSaveFileName(
            self.view, "Export trajectories", f"{s_set.name.replace(' ', '_')}_trajectories.npz",
            "NumPy archive (*.npz);;CSV, one row per particle per snapshot (*.csv)")
        if not f:
            return
        located = self._ensure_located(s_set, "Finding the selected particles in every snapshot...")
        if located is None:
            return
        tr = build_trajectories(located, [s['filepath'] for s in self.model.snapshots])
        tr.population = self._population_record(s_set)
        (export_csv if f.lower().endswith('.csv') else export_npz)(tr, f)
        found = (tr.types >= 0).sum(axis=1)
        log(f"Exported trajectories of {len(tr.ids):,} particles over {tr.n_snap} snapshots to {f} "
            f"(found per snapshot: min {found.min():,}, max {found.max():,})")
        self._update_frame()

    @staticmethod
    def _population_record(s_set) -> dict:
        """The set's name and meaning for an export's metadata: its one-line description, the
        completeness warning (None when there is none) and the fields of selection_sets.py."""
        import json                          # numpy values in a definition become plain JSON
        rec = dict(name=s_set.name, describe=s_set.describe(), warning=s_set.completeness_warning(),
                   n=len(s_set.selected_ids), **s_set.meaning())
        return json.loads(json.dumps(rec, default=selection_sets._json_default))

    def _trail_lines(self, idx: int, ui: dict) -> List[Tuple[np.ndarray, Tuple[float, ...]]]:
        """Polylines up to snapshot ``idx`` from cached lookups only (never reads files)."""
        lines = []
        box = float(self.model.snapshots[0].get('header', {}).get('BoxSize', 0.0) or 0.0)
        for set_id, enabled, color, _ in ui['sets']:
            s_set = self.model.tracker.get_set(set_id)
            if not enabled or not s_set.selected_ids:
                continue
            locs = [self._tracked(i, s_set) for i in range(idx + 1)]
            if sum(l is not None for l in locs) < 2:
                continue
            n_ids = len(next(l for l in locs if l is not None).ids)
            pick = np.arange(min(ui['n_trails'], n_ids))
            pos = np.full((idx + 1, len(pick), 3), np.nan)
            cen = np.full((idx + 1, 3), np.nan)
            for i, l in enumerate(locs):
                if l is None:
                    continue
                pos[i] = l.pos[pick]
                c = periodic_mean(l.pos, box)
                if c is not None:
                    cen[i] = c
            rgba = tuple(c / 255.0 for c in color[:3])
            pos = unwrap_series(pos, box)
            for k in range(len(pick)):
                lines.append((pos[:, k], rgba + (0.45,)))
            lines.append((unwrap_series(cen, box), (1.0, 1.0, 1.0, 0.9)))
        return lines

    @staticmethod
    def _lines_to_segments(lines) -> Tuple[np.ndarray, np.ndarray]:
        segs, cols = [], []
        for pts, rgba in lines:
            pts = np.asarray(pts, dtype=np.float64)
            ok = np.all(np.isfinite(pts[:-1]), axis=1) & np.all(np.isfinite(pts[1:]), axis=1)
            if not ok.any():
                continue
            a, b = pts[:-1][ok], pts[1:][ok]
            seg = np.empty((2 * len(a), 3))
            seg[0::2], seg[1::2] = a, b
            segs.append(seg)
            cols.append(np.tile(rgba, (len(seg), 1)))
        if not segs:
            return np.empty((0, 3)), np.empty((0, 4))
        return np.vstack(segs), np.vstack(cols)

    # ------------------------------------------------------------------ scene for screen and movies

    def _ui_state(self) -> dict:
        """Everything the scene needs from the widgets, read on the GUI thread."""
        sp = self.view.selection_panel
        return {
            'visible': {t for t in range(6) if self.view.layer_panel.layer_state.is_visible(t)},
            'color_mode': self.color_mode,
            'draw_selected_only': sp.draw_selected_only,
            'trails': sp.chk_trails.isChecked(),
            'n_trails': sp.spin_trails.value(),
            'sets': [(s.set_id, s.enabled, s.color, s.version) for s in self.model.tracker.sets],
            # Follow selection and movies centre on this set only, not on every enabled set
            'active_set': int(sp.active_set_idx),
            # Colour by property (FeaturesMixin) and the camera for view-dependent colours.
            'colouring': self.view.colouring_panel.state(),
            'drawing': self.view.colouring_panel.drawing_state(),
            'camera': self.view.camera_state(),
        }

    @staticmethod
    def _time_label(snap: dict) -> str:
        hdr = snap.get('header', {})
        box = float(hdr.get('BoxSize', 0.0) or 0.0)
        z = hdr.get('Redshift')
        if box > 0 and z is not None:
            return f"a = {snap['time']:.4f}   z = {float(z):.3f}"
        return f"t = {snap['time']:.4g} (code units)"

    def _build_scene(self, idx: int, ui: dict, compute: bool = False, snap: Optional[dict] = None) -> Scene:
        """
        Points, colours and selected particles for snapshot ``idx``. With
        ``compute`` the selected particles are looked up in the file when not
        cached (movie worker); otherwise uncached sets fall back to the
        selected particles that happen to be in the display sample. ``snap``
        is a sample of snapshot ``idx`` read for a movie (lazy loading); by
        default the loaded one is used.
        """
        # On screen, the level-of-detail view set of the current camera (controller/lod_controller.py);
        # movies (compute=True) use the base sample, which does not depend on the camera.
        if snap is None:
            snap = self.model.get_snapshot(idx) if compute else self._display_snapshot(idx)
        box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
        mask = np.isin(snap['t'], list(ui['visible']))
        if mask.all():                      # every type shown: use the arrays as they are, no copies
            c_vis, v_vis, t_vis, m_vis = snap['c'], snap['v'], snap['t'], snap['m']
            id_vis = snap['id'] if 'id' in snap else np.arange(len(c_vis))
        else:
            c_vis, v_vis, t_vis = snap['c'][mask], snap['v'][mask], snap['t'][mask]
            m_vis = snap['m'][mask]
            id_vis = snap['id'][mask] if 'id' in snap else np.arange(len(c_vis))

        tracked, fallback, tracked_ids = [], [], []
        for set_id, enabled, color, _ in ui['sets']:
            s_set = self.model.tracker.get_set(set_id)
            if not enabled or not s_set.selected_ids:
                continue
            loc = self._tracked(idx, s_set, compute=compute)
            if loc is not None:
                tracked.append((loc, color))
                tracked_ids.append(s_set.set_id)
            else:
                fallback.append((s_set, color))

        sel_mask = None
        if ui['draw_selected_only']:
            if fallback:
                all_ids = set()
                for s_set, _ in fallback:
                    all_ids.update(s_set.selected_ids)
                sel_mask = np.isin(id_vis, list(all_ids))
            else:
                sel_mask = np.zeros(len(c_vis), dtype=bool)  # tracked sets are drawn as highlights
            c_vis, v_vis, t_vis, id_vis = c_vis[sel_mask], v_vis[sel_mask], t_vis[sel_mask], id_vis[sel_mask]
            m_vis = m_vis[sel_mask]

        # Colours are float32 throughout (what the GPU takes) and made only once it is known which.
        colors = None
        layers = self._scene_layers(snap, mask, sel_mask, t_vis, ui) if len(c_vis) else None
        lit = values = None
        if layers is not None:
            colors, lit, values = layers
        elif len(c_vis):
            if ui['color_mode'] == 'component':
                tc = getattr(self, '_type_colors32', None)
                if tc is None:
                    tc = self._type_colors32 = np.asarray(self.TYPE_COLORS, dtype=np.float32)
                colors = tc[t_vis]
            elif ui['color_mode'] == 'velocity':
                colors = np.empty((len(c_vis), 4), dtype=np.float32)
                speed = np.linalg.norm(v_vis, axis=1)
                speed_max = float(np.percentile(speed, 95)) or 1.0
                norm = np.clip(speed / speed_max, 0, 1)
                colors[:, 0] = norm
                colors[:, 1] = 0.5 - norm * 0.5
                colors[:, 2] = 1.0 - norm
                colors[:, 3] = 0.6
            elif ui['color_mode'] == 'fof_group' and 'FoFGroupIndex' in snap and self.fof_colors_palette is not None:
                grp_vis = snap['FoFGroupIndex'][mask]
                if sel_mask is not None:
                    grp_vis = grp_vis[sel_mask]
                palette_len = len(self.fof_colors_palette)
                colors = self.fof_colors_palette.astype(np.float32)[grp_vis % palette_len]
        if colors is None:
            colors = np.empty((len(c_vis), 4), dtype=np.float32)
            colors[:] = (1.0, 1.0, 1.0, 0.5)

        hl_pos, hl_col, hl_ids, hl_vel, centre_pts = [], [], [], [], []
        # The centre (Follow selection, movies, recentre on pick) is the active set's: the mean of
        # every enabled set used to pull the view off the set being followed. A ui without
        # 'active_set' keeps the old meaning, every enabled set.
        active = ui.get('active_set')
        centre_set = None if active is None else self.model.tracker.get_set(active).set_id
        for k, (loc, color) in enumerate(tracked):
            ok = loc.found & np.isin(loc.types, list(ui['visible']))
            if ok.any():
                p = unwrap_for_display(loc.pos[ok], box)    # one piece across a box face
                hl_pos.append(p)
                hl_col.append(np.tile([c / 255.0 for c in color], (len(p), 1)))
                hl_ids.append(loc.ids[ok])
                hl_vel.append(np.full((len(p), 3), np.nan) if loc.vel is None else loc.vel[ok])
                if centre_set is None or tracked_ids[k] == centre_set:
                    centre_pts.append(p)
        for s_set, color in fallback:
            s_mask = np.isin(id_vis, list(s_set.selected_ids))
            if s_mask.any():
                hl_pos.append(unwrap_for_display(c_vis[s_mask], box))
                hl_col.append(np.tile([c / 255.0 for c in color], (int(s_mask.sum()), 1)))
                hl_ids.append(id_vis[s_mask])
                hl_vel.append(v_vis[s_mask])
                if centre_set is None or s_set.set_id == centre_set:
                    centre_pts.append(c_vis[s_mask].astype(np.float64))

        centre = periodic_mean(np.vstack(centre_pts), box) if centre_pts else None
        lines = self._trail_lines(idx, ui) if ui['trails'] else []
        g_min, g_max = self.model.g_min, self.model.g_max

        # Colouring by a property: lit particles (with a value) are their own layer, drawn
        # over the rest. On screen the layers' fade is a shader setting; movie frames,
        # which have no shader, get it in the colours, and drop the unlit layer if hidden.
        drawing = ui.get('drawing') or {}
        lit_pos = lit_colors = lit_ids = lit_types = lit_values = lit_vel = lit_mass = None
        if lit is not None:
            lit_pos, lit_colors, lit_ids, lit_types = c_vis[lit], colors[lit], id_vis[lit], t_vis[lit]
            lit_values = None if values is None else values[lit]
            lit_vel, lit_mass = v_vis[lit], m_vis[lit]
            c_vis, colors, id_vis, t_vis = c_vis[~lit], colors[~lit], id_vis[~lit], t_vis[~lit]
            v_vis, m_vis = v_vis[~lit], m_vis[~lit]
        if compute and drawing:
            colors = np.array(colors, dtype=np.float32, copy=True)
            colors[:, 3] *= drawing['unlit' if lit is not None else 'lit']['fade']
            if lit is not None:
                lit_colors = np.array(lit_colors, dtype=np.float32, copy=True)
                lit_colors[:, 3] *= drawing['lit']['fade']
                if not drawing.get('show_unlit', True):
                    c_vis, colors = c_vis[:0], colors[:0]
                    id_vis, v_vis, m_vis = id_vis[:0], v_vis[:0], m_vis[:0]
        extra = self._movie_scene_extras(snap, ui) if compute else {}
        return Scene(pos=c_vis, colors=colors, ids=id_vis,
                     lit_pos=lit_pos, lit_colors=lit_colors, lit_ids=lit_ids, lit_types=lit_types,
                     lit_values=lit_values,
                     lit_solid=drawing.get('blend') == 'Solid',
                     highlight_pos=np.vstack(hl_pos) if hl_pos else None,
                     highlight_colors=np.vstack(hl_col) if hl_col else None,
                     centre=centre, label=self._time_label(snap),
                     box_min=g_min, box_size=None if g_min is None else g_max - g_min,
                     box_period=box, lines=lines,
                     vel=v_vis, lit_vel=lit_vel, mass=m_vis, lit_mass=lit_mass,
                     highlight_ids=np.concatenate(hl_ids) if hl_ids else None,
                     highlight_vel=np.vstack(hl_vel) if hl_vel else None, **extra)

    def _movie_scene_extras(self, snap: dict, ui: dict) -> dict:
        """Time, cosmology and overlay information a movie frame needs (movie/maker.py, Scene)."""
        from movie.interpolate import dx_dtime_factor, velocity_a_scaling
        from movie.overlays import length_info
        hdr = snap.get('header', {})
        params = snap.get('unit_source') or hdr
        box = float(hdr.get('BoxSize', 0.0) or 0.0)
        files = list(snap.get('files') or [snap['filepath']])
        n_total = int(snap.get('n_total', 0) or 0)
        cbar = None
        col = ui.get('colouring') or {}
        if col.get('property') not in (None, TYPE_MODE) and not col.get('random'):
            lo, hi = col.get('range', (0.0, 1.0))
            cbar = {'table': col.get('table', 'viridis'), 'lo': float(lo), 'hi': float(hi),
                    'log': bool(col.get('log')), 'title': col['property']}
        z = hdr.get('Redshift')
        return {
            'time': float(snap['time']),
            'redshift': float(z) if (box > 0 and z is not None) else None,
            'dxdt': dx_dtime_factor(hdr, params, velocity_a_scaling(files[0])),
            'files': files,
            'mass_scale': n_total / len(snap['c']) if n_total and len(snap['c']) else 1.0,
            'length': length_info(hdr, params, snap.get('coord_scaling') or {}),
            'colour_bar': cbar,
        }

    def _follow_selection(self, idx: int, scene: Scene):
        """GadgetViewer's 'follow selection': recentre once per snapshot change."""
        if not self.view.selection_panel.chk_follow.isChecked():
            return
        if idx != self._followed_idx and scene.centre is not None:
            self.view.set_camera_centre(scene.centre)
            self._followed_idx = idx

    # ------------------------------------------------------------------ movies

    def _open_movie_dialog(self):
        if not self.model.snapshots:
            QMessageBox.information(self.view, "Make movie", "Load snapshots first.")
            return
        if self.movie_dialog is None:
            from views.movie_dialog import MovieDialog
            self.movie_dialog = MovieDialog(self.view)
            self.movie_dialog.movie_requested.connect(self._make_movie)
            self.movie_dialog.keyframe_requested.connect(self._add_keyframe)
            self.movie_dialog.keyframe_view_requested.connect(self._show_keyframe)
            self.movie_dialog.set_particle_budget(int(getattr(self, 'lod_budget', 2_000_000) or 2_000_000))
        self.movie_dialog.set_snapshots([s['filepath'] for s in self.model.snapshots], self.current_snap_idx)
        self.movie_dialog.show()
        self.movie_dialog.raise_()

    def _make_movie(self, settings: MovieSettings):
        if self._movie_worker and self._movie_worker.isRunning():
            return
        err = settings.validate(len(self.model.snapshots))
        if err:
            QMessageBox.warning(self.view, "Make movie", err)
            return
        if self.is_playing:
            self._toggle_play()
        cam_state = self.view.camera_state()
        if not cam_state:
            g = self.model.g_max - self.model.g_min
            cam_state = {'center': tuple((self.model.g_min + g / 2)), 'distance': float(np.max(g)) * 1.5,
                         'elevation': 30.0, 'azimuth': 45.0, 'fov': 60.0}
        camera = Camera(**cam_state)
        ui = self._ui_state()
        names = [s['filepath'] for s in self.model.snapshots]
        sel = {f"set_{s.set_id + 1}": {'name': s.name, 'n_ids': len(s.selected_ids)}
               for s in self.model.tracker.sets if s.selected_ids and s.enabled}
        extra = {'colour_mode': ui['color_mode'], 'colouring': ui['colouring'], 'visible_types': sorted(ui['visible']),
                 'selection_sets': sel, 'trails': ui['trails'],
                 'positions': "code units as stored in the snapshot (comoving, per h for cosmological runs)",
                 'display_sample': "every snapshot's decimated display sample; selected particles at full resolution"}
        current = self.current_snap_idx

        loaded = list(self.model.snapshots)

        def scene_for(i):
            snap = read_movie_sample(loaded[i], i, settings.lazy_particles) if settings.lazy else None
            return self._build_scene(i, ui, compute=True, snap=snap)

        def run(progress, cancelled):
            return make_movie(settings, camera, current, scene_for, names, progress, cancelled, extra)

        if settings.lazy:
            extra['lazy'] = (f"snapshots read from disk one at a time while rendering, "
                             f"{settings.lazy_particles} particles each")
        n_steps = settings.frame_count(len(names))
        dlg = QProgressDialog("Making movie...", "Cancel", 0, n_steps, self.view)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        worker = MovieWorker(run)
        worker.progress.connect(lambda i, n, m: (dlg.setValue(i), dlg.setLabelText(f"Making movie: {m}")))
        dlg.canceled.connect(worker.cancel)
        worker.done.connect(lambda r: self._on_movie_done(r, dlg))
        self._movie_worker = worker
        if self.movie_dialog:
            self.movie_dialog.set_running(True, "Making movie...")
        where = f"frames to {settings.basename}_*.png" if settings.keep_frames else settings.output_file
        log(f"Movie: {settings.kind}, {settings.style}, {settings.width}x{settings.height}, "
            f"{n_steps} frames, {where}")
        hold(worker).start()

    def _on_movie_done(self, result, dlg):
        dlg.reset()
        if self.movie_dialog:
            self.movie_dialog.set_running(False, result.message)
        for s in result.skipped:
            log(f"Movie skipped {s}", 'warning')
        if not result.success and result.message == "Cancelled by user":
            log("Movie cancelled")           # the user's choice, not a failure: no warning box
            return
        if result.success:
            log(result.message + (f"; settings in {result.params_file}" if result.params_file else ""))
            if result.smooth_range:
                log(f"Movie colour range (surface density, code units): {result.smooth_range}")
            QMessageBox.information(self.view, "Make movie", result.message)
        else:
            log(f"Movie making failed: {result.message}", 'error')
            QMessageBox.warning(self.view, "Make movie", f"Movie making failed: {result.message}")

    # ------------------------------------------------------------------ keyframes

    def _add_keyframe(self):
        from movie.keyframes import Keyframe
        cam = self.view.camera_state()
        if not cam or not self.model.snapshots or self.movie_dialog is None:
            return
        idx = self.current_snap_idx
        name = f"{idx}: {Path(self.model.snapshots[idx]['filepath']).name}"
        self.movie_dialog.add_keyframe(Keyframe(cam, idx, name))

    def _show_keyframe(self, kf):
        """Put a keyframe's view and snapshot on screen."""
        if 0 <= kf.snapshot < len(self.model.snapshots):
            self.view.slide_time.setValue(kf.snapshot)
        self.view.set_camera_state(kf.camera)


def read_movie_sample(loaded: dict, idx: int, n: int) -> dict:
    """
    Snapshot ``idx`` with about ``n`` particles, read from disk for a movie frame
    (lazy loading: the movie holds a few of these at a time, not all snapshots).
    The loaded sample is used as it is when it is already complete or as large.
    The same reader as at load time is used, so the sample is drawn the same way.
    """
    from snapshot_io import snapshot_loader as sl
    n_total = int(loaded.get('n_total', 0) or 0)
    if len(loaded['c']) >= min(n, n_total or n):
        return loaded
    files = list(loaded.get('files') or [loaded['filepath']])
    if sl._is_hdf5(files[0]):
        if n_total >= sl.BIG_SNAPSHOT:
            snap = sl.snapshot_from_store(files, idx, n)
        else:
            snap = sl.load_hdf5_snapshot(files, idx, n)
    else:
        snap = sl.load_single_snapshot_multiformat((files[0], idx, n))
    if snap is None or 'error' in snap:
        raise OSError((snap or {}).get('error', 'could not be read'))
    return snap

