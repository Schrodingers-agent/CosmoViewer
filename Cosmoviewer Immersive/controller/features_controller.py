"""
GadgetViewer README features for the controller: colouring by any property,
halo catalogues attached to particles, additional data files, selection by
property and by catalogue group, and the Make plot window.

Mixed into CosmoViewerController (like TrackingMixin), which owns
``self.model`` and ``self.view``. ``_scene_colours`` is called from
``_build_scene``, which the movie maker runs in a worker thread, so it reads
settings only from the ``ui`` dict and never touches widgets.

Every per-particle array attached here is aligned with the snapshot's display
sample (snap['c']); file properties are read for exactly those rows
(snapshot_io.properties), catalogue membership by Gadget-4 offsets
(snapshot_io.group_catalogue), extra arrays by row (snapshot_io.aux_data).
"""

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QColorDialog, QFileDialog, QMessageBox

from controller.console import log
from controller.thread_hold import hold
from model.colour_tables import MISSING_RGBA, auto_range, map_values, random_colours
from snapshot_io.aux_data import AuxSpec, attach_to_sample, path_for_snapshot, read_aux_array
from snapshot_io.group_catalogue import (find_catalogue_for, group_ids_full, read_gadget4_catalogue,
                                         sample_membership)
from snapshot_io.properties import (SNAP_GROUP, SNAP_SUBHALO, SUBHALO_STRIDE, PropertyInfo, list_properties,
                                    read_sample_values, read_values_for_ids, select_by_range)
from snapshot_io.id_lookup import locate_ids
from views.colouring_panel import LAGRANGE, RGB_MODES, SCENE_MODES, TYPE_MODE

FOF_COMPUTED = 'FoF group (computed here)'
CAT_GROUP = 'Group (catalogue)'
CAT_SUBHALO = 'Subhalo (catalogue)'
KNN_K = 32
BACKGROUND_ROWS = 300_000     # a set at least this big reads file properties off the GUI thread...
BACKGROUND_TOTAL = 2_000_000  # ...as does any set of a snapshot this big (first read fills the cache)


DENSITY = 'Local density (display sample)'
LAGRANGE_POS = '_lagrangian_pos'
LAGRANGE_INDEX_MAX = 20_000_000   # IDs remembered with their first-snapshot position before starting afresh
HALF_CORES = max(1, (os.cpu_count() or 2) // 2)   # background work leaves the other half to the UI and LOD
PANEL2D_FULL_MAX = 2_000_000  # smoothed 2D images use every particle of snapshots up to this size
PANEL2D_ZOOM_PAD = 0.25       # a zoomed 2D image covers this much more on each side, for panning
PANEL2D_INLINE = 60_000       # smoothed images of fewer particles are drawn at once, bigger ones off-thread


class _ValuesJob(QThread):
    """Computes one colouring quantity for a set; the values land in the set's own cache."""
    done = pyqtSignal(str, str)              # name, error ('' on success)
    progress = pyqtSignal(str, int, int)

    def __init__(self, name: str, fn):
        super().__init__()
        self.name, self.fn = name, fn

    def run(self):
        try:
            self.fn(lambda i, n: self.progress.emit(self.name, i, n))
            self.done.emit(self.name, '')
        except Exception as e:  # reported in the console; the particles stay dim
            self.done.emit(self.name, f"{type(e).__name__}: {e}")


class _Panel2DJob(QThread):
    """Draws smoothed 2D images off the window's thread; ``fn`` returns the pictures."""
    done = pyqtSignal(object, object)        # pictures (or None), error text (or None)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as e:  # reported in the console; the old picture stays
            self.done.emit(None, f"{type(e).__name__}: {e}")


class FeaturesMixin:

    def _init_features(self):
        self._props_by_files: Dict[Tuple[str, ...], Dict[str, PropertyInfo]] = {}
        self._catalogue_mode: Optional[str] = None     # None (off), 'auto', or a template path
        self._aux_specs: List[AuxSpec] = []
        self._plot_cache: Dict[tuple, np.ndarray] = {}
        self._catalogues: Dict[tuple, tuple] = {}
        self._aux_full: Dict[tuple, np.ndarray] = {}
        self._aux_errors: List[str] = []
        # A movie's worker thread fills the catalogue and extra-data caches too (on first use of
        # a snapshot), so they are filled and changed under this lock, and each file is read once.
        self._features_lock = threading.RLock()
        self._features_gen = 0
        self._prop_jobs: Dict[tuple, object] = {}
        self._prop_progress: Dict[tuple, float] = {}
        self._done_jobs: List[QThread] = []      # finished jobs, released later: never inside their own signal
        self._auto_range_pending = False
        self._lagr_index: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        v, sp = self.view, self.view.selection_panel
        v.colouring_panel.colouring_changed.connect(self._on_colouring_changed)
        v.colouring_panel.auto_range_requested.connect(self._auto_colour_range)
        v.colouring_panel.drawing_changed.connect(self._on_drawing_changed)
        v.aux_data_requested.connect(self._open_aux_dialog)
        sp.colour_change_requested.connect(self._choose_set_colour)
        v.panel2d_changed.connect(self._update_frame)
        self._panel2d_timer = QTimer()
        self._panel2d_timer.setInterval(300)
        self._panel2d_timer.timeout.connect(self._poll_panel2d_camera)
        self._panel2d_timer.start()
        # Zooming or panning the smoothed image redraws the part on screen at screen resolution.
        self._panel2d_key = self._panel2d_region = self._panel2d_full = None
        self._panel2d_levels = None
        self._panel2d_job: Optional[_Panel2DJob] = None
        self._panel2d_pending = None            # (region,) asked for while a picture was being drawn
        self._panel2d_zoom_timer = QTimer()
        self._panel2d_zoom_timer.setSingleShot(True)
        self._panel2d_zoom_timer.setInterval(250)
        self._panel2d_zoom_timer.timeout.connect(self._on_panel2d_zoomed)
        v.plot_2d.getViewBox().sigRangeChangedManually.connect(lambda *_: self._panel2d_zoom_timer.start())
        v.btn_2d_render.clicked.connect(self._rerender_panel2d)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._finish_panel2d_job)

    def _reset_features(self):
        self._panel2d_full = self._panel2d_key = self._panel2d_region = None
        self._plot_cache.clear()
        with self._features_lock:
            self._catalogues.clear()
            self._aux_full.clear()
        self._features_gen += 1
        self._last_prop_names = None
        self._lagr_index = {}

    # ------------------------------------------------------------------ property registry

    def _file_properties(self, snap: dict) -> Dict[str, PropertyInfo]:
        key = tuple(snap.get('files') or [snap['filepath']])
        if key not in self._props_by_files:
            try:
                self._props_by_files[key] = list_properties(list(key)) if str(key[0]).endswith(('.hdf5', '.h5')) else {}
            except Exception as e:
                log(f"Could not list properties of {Path(key[0]).name}: {e}", 'warning')
                self._props_by_files[key] = {}
        return self._props_by_files[key]

    # Per-particle arrays are cached in snap['_props'], which is per view set when the LOD
    # controller hands _build_scene a shallow copy of the snapshot with other rows. Keys carry
    # a generation number that changes whenever catalogues or extra data are switched.

    def _catalogue_for(self, snap: dict):
        """(path, GroupCatalogue) or (None, note) for the snapshot's file, cached per file."""
        with self._features_lock:
            return self._catalogue_for_unlocked(snap)

    def _catalogue_for_unlocked(self, snap: dict):
        key = (snap['filepath'], self._catalogue_mode)
        if key not in self._catalogues:
            template = None if self._catalogue_mode == 'auto' else self._catalogue_mode
            path = find_catalogue_for(snap['filepath'], template)
            if path is None:
                self._catalogues[key] = (None, f"no catalogue found for {Path(snap['filepath']).name}")
            else:
                try:
                    self._catalogues[key] = (path, read_gadget4_catalogue(path))
                except Exception as e:
                    self._catalogues[key] = (None, f"{Path(path).name}: {type(e).__name__}: {e}")
        return self._catalogues[key]

    def _groups_of(self, snap: dict) -> Dict[str, np.ndarray]:
        """Catalogue membership aligned with this snapshot dict's rows."""
        with self._features_lock:
            return self._groups_of_unlocked(snap)

    def _groups_of_unlocked(self, snap: dict) -> Dict[str, np.ndarray]:
        if not self._catalogue_mode:
            return {}
        cache = snap.setdefault('_props', {})
        key = ('_groups', self._features_gen)
        if key not in cache:
            path, cat = self._catalogue_for(snap)
            out = {}
            if path is not None:
                try:
                    mem = sample_membership(snap, cat)
                    out[CAT_GROUP] = mem['FoFGroupIndex']
                    if 'SubGroupIndex' in mem:
                        out[CAT_SUBHALO] = mem['SubGroupIndex']
                except Exception as e:
                    self._catalogues[(snap['filepath'], self._catalogue_mode)] = (None, f"{Path(path).name}: {e}")
            cache[key] = out
        return cache[key]

    def _aux_of(self, snap: dict) -> Dict[str, np.ndarray]:
        with self._features_lock:
            return self._aux_of_unlocked(snap)

    def _aux_of_unlocked(self, snap: dict) -> Dict[str, np.ndarray]:
        cache = snap.setdefault('_props', {})
        out = {}
        for spec in self._aux_specs:
            key = ('_aux', spec.name, self._features_gen)
            if key not in cache:
                try:
                    path = path_for_snapshot(spec.path, snap['filepath'])
                    full = self._aux_full.get((spec.name, path))
                    if full is None:
                        full = read_aux_array(spec, path)
                        if len(self._aux_full) > 8:          # bound memory: keep a few snapshots
                            self._aux_full.pop(next(iter(self._aux_full)))
                        self._aux_full[(spec.name, path)] = full
                    cache[key] = attach_to_sample(snap, spec, full)
                except Exception as e:
                    cache[key] = None
                    self._aux_errors.append(f"{spec.name}: {type(e).__name__}: {e}")
            if cache[key] is not None:
                out[spec.name] = cache[key]
        return out

    def _attached(self, snap: dict) -> Dict[str, Tuple[np.ndarray, bool]]:
        """Arrays attached to the drawn rows: name -> (values, integer?)."""
        out: Dict[str, Tuple[np.ndarray, bool]] = {}
        if 'FoFGroupIndex' in snap:
            out[FOF_COMPUTED] = (snap['FoFGroupIndex'], True)
        for k, v in self._groups_of(snap).items():
            out[k] = (v, True)
        integer = {sp.name: sp.integer for sp in self._aux_specs}
        for k, v in self._aux_of(snap).items():
            out[k] = (v, integer[k])
        return out

    def _property_names(self, snap: dict) -> Tuple[List[str], Dict[str, bool], Dict[str, str]]:
        props = self._file_properties(snap)
        att = self._attached(snap)
        names = list(att) + sorted(props)
        integer = {k: v[1] for k, v in att.items()}
        integer.update({k: p.integer for k, p in props.items()})
        tips = {k: f"{k}: {p.unit_label}; types {', '.join(map(str, p.types))}" for k, p in props.items()}
        if 'Temperature' in props and props['Temperature'].derived:
            tips['Temperature'] = ("Derived from InternalEnergy: T = (γ-1) u μ m_p / k_B with γ = 5/3, X = 0.76,\n"
                                   "μ from ElectronAbundance if present (else fully ionised). Kelvin.")
        tips[FOF_COMPUTED] = "Groups found by Structures > Find FoF groups (display sample, 0 = none)"
        tips[CAT_GROUP] = "Gadget-4 catalogue group of each particle by offsets (1 = first group, 0 = none)"
        tips[CAT_SUBHALO] = "Gadget-4 SUBFIND subhalo of each particle by offsets (1 = first, 0 = none)"
        tips[SNAP_GROUP] = "FoF group stored in the snapshot (GroupNumber; 1 = first group, 0 = none)"
        tips[SNAP_SUBHALO] = (f"Subhalo stored in the snapshot: group × {SUBHALO_STRIDE:,} + SubGroupNumber\n"
                              f"(group 12, subhalo 3 = {12 * SUBHALO_STRIDE + 3}; subhalo 0 is the main one; 0 = none)")
        return names, integer, tips

    def _refresh_property_list(self):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            return
        names, integer, tips = self._property_names(snap)
        if tuple(names) == getattr(self, '_last_prop_names', None):
            return                              # unchanged: keep the widgets as they are
        self._last_prop_names = tuple(names)
        self.view.colouring_panel.set_properties(names, integer, tips)
        self.view.selection_panel.set_properties([n for n in names])
        if self.plot_window is not None:
            self.plot_window.set_properties(names + ['Distance from view centre', 'Local density (display sample)'])

    def _sample_values(self, snap: dict, name: str, wait: bool = False) -> Optional[np.ndarray]:
        """
        Values of a file or attached property for the whole set. On the GUI thread a
        large set's first read runs in a worker and this returns None until it is done
        (the frame is redrawn then), unless ``wait``.
        """
        att = self._attached(snap)
        if name in att:
            return np.asarray(att[name][0], dtype=np.float64)
        info = self._file_properties(snap).get(name)
        if info is None:
            return None
        if not wait and self._read_in_background(snap, info):
            return None                     # drawn dim until the job finishes and redraws
        job = self._prop_jobs.get(info.name)
        if job is not None:
            job.wait()                      # being read (perhaps for an earlier set): let it finish
        if info.name in snap.setdefault('_props', {}) or not self._on_gui_thread():
            return read_sample_values(snap, info)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return read_sample_values(snap, info)
        finally:
            QApplication.restoreOverrideCursor()

    @staticmethod
    def _on_gui_thread() -> bool:
        app = QApplication.instance()
        return app is not None and QThread.currentThread() is app.thread()

    @staticmethod
    def _big(snap: dict) -> bool:
        return len(snap['c']) >= BACKGROUND_ROWS or int(snap.get('n_total', 0) or 0) >= BACKGROUND_TOTAL

    def _read_in_background(self, snap: dict, info: PropertyInfo) -> bool:
        """
        True while a large set's values for ``info`` are being read in a worker. Only the
        GUI thread hands work off; the movie maker reads in place.
        """
        if info.name in snap.setdefault('_props', {}) or not self._big(snap) or not self._on_gui_thread():
            return False
        self._start_job(info.name, lambda prog: read_sample_values(snap, info, prog))
        return True

    def _start_job(self, name: str, fn):
        """
        At most one job per quantity. If one is still running for an earlier set (the view
        moved on), no second one starts: when it ends the frame is redrawn and the current
        set is read then, quickly, since the first read filled the property cache.
        """
        if name in self._prop_jobs:
            return
        self._done_jobs = [j for j in self._done_jobs if not j.isFinished()]
        job = _ValuesJob(name, fn)
        job.done.connect(self._on_property_job_done)
        job.progress.connect(self._on_property_job_progress)
        self._prop_jobs[name] = job
        job.start(QThread.Priority.LowPriority)

    def _on_property_job_progress(self, name: str, i: int, n: int):
        self._prop_progress[name] = i / max(n, 1)

    def _on_property_job_done(self, name: str, error: str):
        job = self._prop_jobs.pop(name, None)
        self._prop_progress.pop(name, None)
        if job is not None:
            job.wait()
            job.fn = None                   # let go of the set it read for
            self._done_jobs.append(job)     # deleting a QThread from its own signal's slot can crash Qt
        if error:
            log(f"Reading {name}: {error}", 'error')
            return
        if self._auto_range_pending and not self._prop_jobs:
            self._auto_range_pending = False
            self._auto_colour_range()
        self._update_frame()

    # ------------------------------------------------------------------ colours for the scene

    @staticmethod
    def _view_direction(cam: dict) -> Optional[np.ndarray]:
        """Unit vector from the view centre towards the camera (pyqtgraph's convention)."""
        if not cam:
            return None
        el, az = np.radians(cam['elevation']), np.radians(cam['azimuth'])
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def _scene_values(self, snap: dict, name: str, mask: np.ndarray, ui: dict,
                      wait: bool = False) -> Optional[np.ndarray]:
        """Values for the visible particles (snap rows where ``mask``)."""
        box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
        if name == 'Line-of-sight velocity':
            n = self._view_direction(ui.get('camera') or {})
            if n is None:
                return None
            v = snap['v'][mask].astype(np.float64)
            m = snap['m'][mask].astype(np.float64)
            vbar = (v * m[:, None]).sum(0) / max(m.sum(), 1e-300) if len(v) else np.zeros(3)
            return -(v - vbar) @ n          # positive = receding from the viewer
        if name == 'Distance from view centre':
            cam = ui.get('camera') or {}
            c = np.array(cam.get('center', (self.model.g_min + self.model.g_max) / 2.0), dtype=np.float64)
            d = snap['c'][mask].astype(np.float64) - c
            if box > 0:
                d -= box * np.round(d / box)
            return np.sqrt(np.einsum('ij,ij->i', d, d))
        if name == DENSITY:
            rho = self._knn_density(snap, box, wait)
            return None if rho is None else rho[mask]
        if name == LAGRANGE:
            got = self._lagrangian(snap, wait)
            return None if got is None else got[1][mask]
        vals = self._sample_values(snap, name, wait)
        return None if vals is None else vals[mask]

    def _knn_density(self, snap: dict, box: float, wait: bool = False) -> Optional[np.ndarray]:
        """
        rho = (mass of the k nearest particles) / (4/3 pi r_k^3), scaled by N_total / N_base,
        computed once on the uniform base sample (the loader's). A zoomed LOD view set,
        whose rows are not a uniform sample, takes the value of the nearest base-sample
        particle, so zooming neither changes nor invalidates the estimate. Large sets are
        computed in a worker (None until done). A visual guide only.
        """
        cache = snap.setdefault('_props', {})
        if DENSITY in cache:
            return cache[DENSITY]
        base = self.model.get_snapshot(snap['idx']) if 'idx' in snap else None
        if base is None or len(base['c']) == 0:
            base = snap
        fn = lambda prog=None: self._compute_density(snap, base, box)
        if not wait and (self._big(snap) or self._big(base)) and self._on_gui_thread():
            self._start_job(DENSITY, fn)
            return None
        job = self._prop_jobs.get(DENSITY)
        if job is not None:
            job.wait()
        return cache[DENSITY] if DENSITY in cache else fn()

    @staticmethod
    def _compute_density(snap: dict, base: dict, box: float) -> np.ndarray:
        from scipy.spatial import cKDTree
        bcache = base.setdefault('_props', {})
        n = len(base['c'])
        tree = bcache.get(('_knn_tree', n))
        if tree is None:
            bpos = base['c'].astype(np.float64)
            tree = cKDTree(np.mod(bpos, box), boxsize=box) if box > 0 else cKDTree(bpos)
            bcache[('_knn_tree', n)] = tree
        rho_b = bcache.get(('_knn_rho', n))
        if rho_b is None:
            k = min(KNN_K, max(n - 1, 1))
            dist, idx = tree.query(tree.data, k=k, workers=HALF_CORES)
            dist, idx = dist.reshape(n, -1), idx.reshape(n, -1)
            r = np.maximum(dist[:, -1], 1e-30)
            m = base['m'].astype(np.float64)
            scale = float(base.get('n_total', n)) / max(n, 1)
            rho_b = scale * m[idx].sum(axis=1) / (4.0 / 3.0 * np.pi * r ** 3)
            bcache[('_knn_rho', n)] = rho_b
        if snap is base:
            rho = rho_b
        else:
            pos = snap['c'].astype(np.float64)
            _, near = tree.query(np.mod(pos, box) if box > 0 else pos, k=1, workers=HALF_CORES)
            rho = rho_b[np.asarray(near).reshape(-1)]
        snap.setdefault('_props', {})[DENSITY] = rho
        return rho

    # ------------------------------------------------------------------ Lagrangian origin

    def _first_snapshot(self) -> Optional[dict]:
        snaps = self.model.snapshots or []
        return min(snaps, key=lambda s: float(s.get('time', 0.0))) if snaps else None

    def _lagrangian(self, snap: dict, wait: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        (rgb, distance) for every row of ``snap`` from its position in the earliest loaded
        snapshot, found by ParticleID at full resolution (NaN where the ID is not there).
        rgb is the first-snapshot position scaled to the box (red = x, green = y, blue = z);
        distance is from the centre of that box. Large sets are looked up in a worker.
        """
        cache = snap.setdefault('_props', {})
        if LAGRANGE_POS in cache:
            return cache[LAGRANGE_POS]
        first = self._first_snapshot()
        if first is None or 'id' not in snap:
            return None
        fn = lambda prog=None: self._compute_lagrangian(snap, first)
        if not wait and (self._big(snap) or self._big(first)) and self._on_gui_thread():
            self._start_job(LAGRANGE, fn)
            return None
        job = self._prop_jobs.get(LAGRANGE)
        if job is not None:
            job.wait()
        return cache[LAGRANGE_POS] if LAGRANGE_POS in cache else fn()

    def _compute_lagrangian(self, snap: dict, first: dict) -> Tuple[np.ndarray, np.ndarray]:
        files = [str(f) for f in (first.get('files') or [first['filepath']])]
        key = tuple(files)
        known = self._lagr_index.get(key)
        if known is None or len(known[0]) > LAGRANGE_INDEX_MAX:
            ids0, pos0 = np.asarray(first['id'], dtype=np.uint64), first['c'].astype(np.float64)   # free: already read
            ids0, k = np.unique(ids0, return_index=True)
            known = (ids0, pos0[k])
        ids = np.asarray(snap['id'], dtype=np.uint64)
        need = np.setdiff1d(np.unique(ids), known[0], assume_unique=True)
        if len(need):
            loc = locate_ids(files, need, with_velocities=False)
            pos = np.where(loc.found[:, None], loc.pos.astype(np.float64), np.nan)   # not found: remembered as NaN
            all_ids = np.concatenate([known[0], loc.ids])
            order = np.argsort(all_ids, kind='stable')
            known = (all_ids[order], np.concatenate([known[1], pos])[order])
        self._lagr_index[key] = known
        j = np.clip(np.searchsorted(known[0], ids), 0, max(len(known[0]) - 1, 0))
        hit = known[0][j] == ids if len(known[0]) else np.zeros(len(ids), dtype=bool)
        p0 = np.where(hit[:, None], known[1][j], np.nan) if len(known[0]) else np.full((len(ids), 3), np.nan)
        box = float(first.get('header', {}).get('BoxSize', 0.0) or 0.0)
        if box > 0:
            lo, hi = np.zeros(3), np.full(3, box)
            p0 = np.mod(p0, box)
        else:
            lo, hi = first['c'].min(axis=0).astype(np.float64), first['c'].max(axis=0).astype(np.float64)
        span = np.maximum(hi - lo, 1e-30)
        rgb = 0.2 + 0.8 * np.clip((p0 - lo) / span, 0.0, 1.0)      # never black on the black background
        d = p0 - (lo + hi) / 2.0
        if box > 0:
            d -= box * np.round(d / box)
        out = (rgb, np.sqrt(np.einsum('ij,ij->i', d, d)))
        snap.setdefault('_props', {})[LAGRANGE_POS] = out
        return out

    def _density_brightness(self, snap: dict, mask: np.ndarray, sel_mask: Optional[np.ndarray],
                            colours: np.ndarray, lit: np.ndarray) -> np.ndarray:
        """
        Splotch-style frames: the colour keeps its hue from the property and takes its
        brightness from the local density, log-scaled between the 1st and 99.5th
        percentile of the lit particles in this frame. Unchanged while the density
        is being computed.
        """
        box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
        rho = self._knn_density(snap, box)
        if rho is None:
            return colours
        r = rho[mask] if sel_mask is None else rho[mask][sel_mask]
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.log10(np.maximum(r, 1e-300))
        ok = lit & np.isfinite(lr)
        if not ok.any():
            return colours
        lo, hi = np.percentile(lr[ok], [1.0, 99.5])
        b = np.clip((lr - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        out = np.array(colours, dtype=np.float32, copy=True)
        out[:, :3] *= (0.06 + 0.94 * np.nan_to_num(b))[:, None]
        return out

    def _scene_colours(self, snap: dict, mask: np.ndarray, sel_mask: Optional[np.ndarray],
                       t_vis: np.ndarray, ui: dict) -> Optional[np.ndarray]:
        """Colours for a property; None means "use the type/legacy colouring"."""
        layers = self._scene_layers(snap, mask, sel_mask, t_vis, ui)
        return None if layers is None else layers[0]

    def _scene_layers(self, snap: dict, mask: np.ndarray, sel_mask: Optional[np.ndarray],
                      t_vis: np.ndarray, ui: dict) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        (colours, lit, values) for the drawn particles when colouring by a property; None for
        type colouring. ``lit`` marks particles that carry a value (non-zero for random
        integer colours, where 0 means "in no group"); the rest are the unlit layer.
        """
        col = ui.get('colouring') or {}
        name = col.get('property', TYPE_MODE)
        if name == TYPE_MODE:
            return None
        layers = self._property_layers(snap, mask, sel_mask, ui, col, name)
        if col.get('density_brightness') and layers[1].any():
            layers = (self._density_brightness(snap, mask, sel_mask, layers[0], layers[1]),) + layers[1:]
        return layers

    def _property_layers(self, snap: dict, mask: np.ndarray, sel_mask: Optional[np.ndarray], ui: dict,
                         col: dict, name: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if name == LAGRANGE:
            got = self._lagrangian(snap)
            n = int(mask.sum()) if sel_mask is None else int(sel_mask.sum())
            if got is None:
                return np.tile(MISSING_RGBA, (n, 1)), np.zeros(n, dtype=bool), None
            rgb, dist = got[0][mask], got[1][mask]
            if sel_mask is not None:
                rgb, dist = rgb[sel_mask], dist[sel_mask]
            lit = np.isfinite(dist)
            colours = np.tile(MISSING_RGBA, (n, 1)).astype(np.float32)
            colours[lit, :3] = rgb[lit]
            colours[lit, 3] = 0.9
            return colours, lit, dist
        vals = self._scene_values(snap, name, mask, ui)
        n = int(mask.sum()) if sel_mask is None else int(sel_mask.sum())
        if vals is None:                    # being read, or not in this snapshot
            return np.tile(MISSING_RGBA, (n, 1)), np.zeros(n, dtype=bool), None
        if sel_mask is not None:
            vals = vals[sel_mask]
        if col.get('random'):
            lit = np.isfinite(vals) & (vals != 0)
            return random_colours(vals, col.get('seed', 0)), lit, vals
        lit = np.isfinite(vals)
        lo, hi = col.get('range', (0.0, 1.0))
        try:
            return map_values(vals, lo, hi, col.get('log', False), col.get('table', 'viridis')), lit, vals
        except ValueError:                  # log scale with a non-positive range
            return map_values(vals, lo, hi, False, col.get('table', 'viridis')), lit, vals

    def _on_drawing_changed(self):
        """Point style or the unlit checkbox: restyle the layers, no colours to recompute."""
        cp = self.view.colouring_panel
        self.view.apply_drawing(cp.drawing_state(), cp.state()['property'] != TYPE_MODE)

    def _on_colouring_changed(self):
        st = self.view.colouring_panel.state()
        self.color_mode = 'component' if st['property'] == TYPE_MODE else f"property:{st['property']}"
        self._update_frame()

    def _auto_colour_range(self):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            return
        st = self.view.colouring_panel.state()
        if st['property'] == TYPE_MODE:
            return
        ui = self._ui_state()
        mask = np.isin(snap['t'], list(ui['visible']))
        vals = self._scene_values(snap, st['property'], mask, ui)
        if vals is None and self._prop_jobs:
            self._auto_range_pending = True
            self.view.colouring_panel.set_info(f"Reading {st['property']}; the range is set when it is ready.")
            return
        if vals is None or not np.isfinite(vals).any():
            self.view.colouring_panel.set_info(f"{st['property']} is not available for the drawn particles "
                                               f"in this snapshot.")
            return
        cp = self.view.colouring_panel
        n_have = int(np.isfinite(vals).sum())
        defined = (f"{st['property']}: defined for {n_have:,} of {len(vals):,} drawn particles"
                   + ("" if n_have == len(vals) else "; the rest are drawn dim") + ".")
        if st['random'] or st['property'] in RGB_MODES:
            cp.set_info(defined)                # random colours or RGB: no range to set
            return
        if st['property'] == 'Line-of-sight velocity':
            a = float(np.nanpercentile(np.abs(vals), 99.0)) or 1.0
            lo, hi = -a, a                      # symmetric for a diverging table
        else:
            lo, hi = auto_range(vals, st['log'])
        cp.edit_min.setText(f"{lo:.4g}")
        cp.edit_max.setText(f"{hi:.4g}")
        cp.set_info(defined)
        cp._changed()

    def _set_color_mode(self, mode: str):
        """View menu and FoF dialog shortcuts, routed through the colouring panel."""
        cp = self.view.colouring_panel
        target = {'component': TYPE_MODE, 'velocity': 'Velocities |v|', 'fof_group': FOF_COMPUTED}.get(mode, mode)
        if not cp.select_property(target):
            self.color_mode = mode               # legacy colouring (e.g. no velocities listed)
            self._update_frame()

    # ------------------------------------------------------------------ catalogues

    def _load_group_catalog(self, format_type: str, file_path: str):
        """Structures > Read halo catalogue: attach Gadget-4 catalogue membership to every snapshot."""
        if format_type != 'GADGET4':
            QMessageBox.information(
                self.view, "Read halo catalogue",
                "Only Gadget-4 HDF5 catalogues (fof_tab / fof_subhalo_tab) can be read so far.\n"
                "VELOCIraptor and binary SubFind readers are deferred until test files are available.")
            return
        try:
            cat = read_gadget4_catalogue(file_path)
        except Exception as e:
            QMessageBox.warning(self.view, "Read halo catalogue", f"Could not read {Path(file_path).name}:\n{e}")
            return
        self._enable_catalogues(file_path)
        log(f"Group catalogue {Path(file_path).name}: {cat.n_groups:,} groups, {cat.n_subhalos:,} subhaloes; "
            f"matching catalogues are read for every snapshot")
        self.view.colouring_panel.select_property(CAT_GROUP)

    def _enable_catalogues(self, mode: Optional[str]):
        self._catalogue_mode = mode
        with self._features_lock:
            self._catalogues.clear()
        self._features_gen += 1
        self._refresh_property_list()
        self._update_frame()

    def _select_catalogue_group(self, snap: dict, group: int, subhalo: bool) -> np.ndarray:
        if not self._catalogue_mode:
            # No catalogue: use the halo membership stored in the snapshot itself, if any.
            props = self._file_properties(snap)
            name = SNAP_SUBHALO if subhalo else SNAP_GROUP
            if name in props:
                files = list(snap.get('files') or [snap['filepath']])
                return select_by_range(files, props[name], group, group, list(props[name].types))
            raise ValueError("no group catalogue loaded (Structures > Read halo catalogue...), and the snapshot "
                             "stores no group membership")
        path, cat = self._catalogue_for(snap)
        if path is None:
            raise ValueError(cat)
        return group_ids_full(list(snap.get('files') or [snap['filepath']]), cat, group, subhalo)

    # ------------------------------------------------------------------ selection by property

    def _select_by_property(self, snap: dict, name: str, lo: float, hi: float, full: bool,
                            visible: List[int]) -> np.ndarray:
        info = self._file_properties(snap).get(name)
        if full and info is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                return select_by_range(list(snap.get('files') or [snap['filepath']]), info, lo, hi, visible)
            finally:
                QApplication.restoreOverrideCursor()
        vals = self._sample_values(snap, name, wait=True)
        if vals is None:
            raise ValueError(f"{name} is not available in this snapshot")
        m = (vals >= lo) & (vals <= hi) & np.isin(snap['t'], visible)
        return snap['id'][m]

    def _choose_set_colour(self):
        from PyQt6.QtGui import QColor
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        c = QColorDialog.getColor(QColor(*s_set.color[:3]), self.view, f"Colour of {s_set.name}")
        if c.isValid():
            s_set.color = (c.red(), c.green(), c.blue(), 255)
            sp._refresh_ui()
            self._update_frame()

    # ------------------------------------------------------------------ additional data

    def _open_aux_dialog(self):
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            QMessageBox.information(self.view, "Read additional data", "Load snapshots first.")
            return
        from views.aux_data_dialog import AuxDataDialog
        dlg = AuxDataDialog(self.view, [t for t in range(6) if np.any(snap['t'] == t)])
        if not dlg.exec():
            return
        spec = dlg.spec()
        path = path_for_snapshot(spec.path, snap['filepath'])
        if not Path(path).is_file():
            QMessageBox.warning(self.view, "Read additional data", f"There is no file {path}.")
            return
        try:
            vals = read_aux_array(spec, path)
            attach_to_sample(snap, spec, vals)      # validates the length before adding the spec
        except KeyError:
            QMessageBox.warning(self.view, "Read additional data",
                                f"{Path(path).name} has no dataset '{spec.dataset}'.")
            return
        except Exception as e:
            QMessageBox.warning(self.view, "Read additional data", str(e))
            return
        self._aux_specs = [s for s in self._aux_specs if s.name != spec.name] + [spec]
        with self._features_lock:
            self._aux_full = {k: v for k, v in self._aux_full.items() if k[0] != spec.name}
        self._features_gen += 1
        log(f"Additional data '{spec.name}' ({len(vals):,} values for type {spec.ptype}) from {spec.path}; "
            f"read again for every snapshot with the snapshot number replaced")
        self._refresh_property_list()
        self.view.colouring_panel.select_property(spec.name)

    # ------------------------------------------------------------------ status line for the frame

    def _features_status(self, snap: dict) -> str:
        notes = []
        if self._catalogue_mode:
            path, cat = self._catalogue_for(snap)
            notes.append(cat if path is None else
                         f"{Path(path).name}: {cat.n_groups:,} groups"
                         + (f", {cat.n_subhalos:,} subhaloes" if cat.n_subhalos else ""))
        if self._prop_jobs:
            notes.append("reading " + ", ".join(
                f"{k} {100 * self._prop_progress.get(k, 0):.0f}%" for k in sorted(self._prop_jobs)))
        while self._aux_errors:
            log(f"Additional data: {self._aux_errors.pop(0)}", 'warning')
        return " | ".join(notes)

    # ------------------------------------------------------------------ Make plot

    def _plot_values_for_set(self, snap_idx: int, s_set, name: str) -> Optional[np.ndarray]:
        """Full-resolution values of a selection set (by ID), cached per set version."""
        snap = self.model.get_snapshot(snap_idx)
        key = (snap_idx, s_set.set_id, s_set.version, name)
        if key in self._plot_cache:
            return self._plot_cache[key]
        info = self._file_properties(snap).get(name)
        if info is None:
            # Attached arrays exist for the sample only: use the set's members in the sample.
            vals = self._sample_values(snap, name, wait=True)
            out = None if vals is None else vals[np.isin(snap['id'], s_set.id_array())]
        else:
            loc = self._tracked(snap_idx, s_set)
            types = None if loc is None else loc.types
            out = read_values_for_ids(list(snap.get('files') or [snap['filepath']]), info, s_set.id_array(), types)
        self._plot_cache[key] = out
        return out

    PLOT_MAX_SAMPLE_POINTS = 30_000

    def _render_property_plot(self, p_type: str, x_prop: str, y_prop: str, log_x: bool, log_y: bool):
        """Sample in red, each enabled selection set in its colour (sets at full resolution)."""
        from views.property_plot_window import SAMPLE_RGBA
        pw = self.plot_window
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or pw is None:
            return
        ui = self._ui_state()
        vis = np.isin(snap['t'], list(ui['visible']))
        names = [x_prop] if p_type == 'histogram' else [x_prop, y_prop]
        sample = []
        for nm in names:
            v = self._scene_values(snap, nm, vis, ui)
            if v is None and self._prop_jobs:
                pw.show_message(f"Reading '{nm}'; the plot is drawn when it is ready.")
                return
            if v is None or not np.isfinite(v).any():
                pw.show_message(f"'{nm}' cannot be read for the drawn particles of this snapshot.")
                return
            sample.append(v)

        def tr(v, log):
            v = np.asarray(v, dtype=np.float64)
            if log:
                with np.errstate(divide='ignore', invalid='ignore'):
                    v = np.where(v > 0, np.log10(v), np.nan)
            return v

        sets = []
        for set_id, enabled, color, _ in ui['sets']:
            s_set = self.model.tracker.get_set(set_id)
            if not enabled or not s_set.selected_ids:
                continue
            vals = []
            for nm in names:
                if nm in SCENE_MODES:           # scene quantities exist for the sample only
                    v = self._scene_values(snap, nm, np.ones(len(snap['c']), dtype=bool), ui)
                    v = None if v is None else v[np.isin(snap['id'], s_set.id_array())]
                else:
                    v = self._plot_values_for_set(self.current_snap_idx, s_set, nm)
                vals.append(v)
            if all(v is not None for v in vals):
                sets.append((s_set.name, vals, tuple(color[:3]) + (200,)))

        label = Path(snap['filepath']).name
        if p_type == 'histogram':
            x = tr(sample[0], log_x)
            ok = np.isfinite(x)
            if not ok.any():
                pw.show_message(f"No positive values of '{x_prop}' for a log axis.")
                return
            # Bins span the sets too: a set is read at full resolution and can reach values the
            # sample never holds, and np.histogram silently drops values outside the edges.
            set_vals = []
            for nm, (sv,), rgba in sets:
                s = tr(sv, log_x)
                set_vals.append((nm, s[np.isfinite(s)], rgba))
            edges = np.histogram_bin_edges(np.concatenate([x[ok]] + [s for _, s, _ in set_vals]), bins=60)
            layers = [(f"sample ({ok.sum():,})", edges, np.histogram(x[ok], bins=edges)[0], SAMPLE_RGBA)]
            for nm, s, rgba in set_vals:
                layers.append((f"{nm} ({len(s):,})", edges, np.histogram(s, bins=edges)[0], rgba))
            pw.draw_histogram_layers(layers, x_prop, log_x,
                                     note=f"{label}: display sample of {len(snap['c']):,} of "
                                          f"{snap.get('n_total', len(snap['c'])):,} particles; selection sets at full resolution.")
            return
        x, y = tr(sample[0], log_x), tr(sample[1], log_y)
        ok = np.nonzero(np.isfinite(x) & np.isfinite(y))[0]
        if not len(ok) and not sets:
            # e.g. a gas property against a star property: no particle has both
            logs = [n for n, lg in ((x_prop, log_x), (y_prop, log_y)) if lg]
            pw.show_message(f"No drawn particle has values of both '{x_prop}' and '{y_prop}'"
                            + (f" (positive, for the log axis of {' and '.join(repr(n) for n in logs)})"
                               if logs else "") + ".")
            return
        shown = ok
        if len(ok) > self.PLOT_MAX_SAMPLE_POINTS:
            shown = np.sort(np.random.default_rng(0).choice(ok, self.PLOT_MAX_SAMPLE_POINTS, replace=False))
        layers = [(f"sample ({len(shown):,} of {len(ok):,})", x[shown], y[shown], SAMPLE_RGBA)]
        for nm, (sx, sy), rgba in sets:
            a, b = tr(sx, log_x), tr(sy, log_y)
            k = np.isfinite(a) & np.isfinite(b)
            layers.append((f"{nm} ({k.sum():,})", a[k], b[k], rgba))
        pw.draw_scatter_layers(layers, x_prop, y_prop, log_x, log_y,
                               note=f"{label}: points from the display sample ({len(snap['c']):,} of "
                                    f"{snap.get('n_total', len(snap['c'])):,} particles"
                                    + (f", {len(shown):,} drawn" if len(shown) < len(ok) else "")
                                    + "); selection sets at full resolution where the property is in the file.")

    def _auto_update_plot(self):
        pw = self.plot_window
        if pw is not None and pw.isVisible() and pw.chk_auto.isChecked() and pw.has_plotted:
            try:
                self._render_property_plot(*pw.request())
            except Exception as e:                       # a plot must never break the main view
                log(f"Make plot: {type(e).__name__}: {e}", 'error')

    # ------------------------------------------------------------------ smoothed images in the 2D panel

    def _smoothing_lengths(self, snap: dict, mask: np.ndarray, box: float,
                           wait: bool = False) -> Optional[np.ndarray]:
        """Gas: SmoothingLength from the file; others: half the 32nd-neighbour distance in the set."""
        from model.sph_projection import knn_smoothing_length
        key = ('_hsml', tuple(np.unique(snap['t'][mask]).tolist()))
        cache = snap.setdefault('_props', {})
        if key not in cache:
            h = np.full(int(mask.sum()), np.nan)
            if 'SmoothingLength' in self._file_properties(snap):
                hs = self._sample_values(snap, 'SmoothingLength', wait)
                if hs is None:
                    return None             # being read; the panel is drawn when it is ready
                h = hs[mask].copy()
            need = ~np.isfinite(h)
            if need.any():
                pos = snap['c'][mask][need]
                h[need] = knn_smoothing_length(pos, 32, box) if len(pos) > 1 else 0.0
            cache[key] = h
        return cache[key]

    def _panel2d_particles(self, snap: dict) -> Tuple[dict, bool]:
        """
        (particles, every particle?) for smoothed 2D images. Snapshots of up to
        PANEL2D_FULL_MAX particles are read whole once, so sparse gas is not drawn
        as separate blobs (the smoothing lengths in the file belong to every
        particle, not to a 1-in-10 sample); bigger ones use the display sample.
        """
        n_total = int(snap.get('n_total', 0) or 0)
        if not n_total or len(snap['c']) >= n_total:
            return snap, True
        if n_total > PANEL2D_FULL_MAX:
            return snap, False
        key = (snap['filepath'], n_total, self._features_gen)
        if self._panel2d_full is None or self._panel2d_full[0] != key:
            from controller.tracking_controller import read_movie_sample
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                full = read_movie_sample(snap, self.current_snap_idx, n_total)
            except Exception as e:
                log(f"2D panel: could not read every particle ({e}); using the display sample", 'warning')
                return snap, False
            finally:
                QApplication.restoreOverrideCursor()
            self._panel2d_full = (key, full)
        return self._panel2d_full[1], True

    def _panel2d_pixels(self, lo: int, hi: int) -> int:
        """Image width in device pixels of the 2D panel, within [lo, hi]."""
        vb = self.view.plot_2d.getViewBox()
        dpr = float(self.view.plot_2d.devicePixelRatioF() or 1.0)
        return int(np.clip(max(vb.width(), vb.height()) * dpr, lo, hi))

    def _render_panel2d(self, region: Optional[Tuple[float, float, float, float]] = None):
        """
        Smoothed image of the 3D view in the 2D panel. Without ``region`` the whole
        view is drawn when anything it depends on changed (and a zoom into the old
        picture is drawn again for the new settings, as long as the camera stayed);
        ``region`` = (x0, x1, y0, y1) redraws a zoom into the picture. Big sets are
        drawn off the window's thread; a request made meanwhile is drawn next.
        """
        from model.sph_projection import project
        if self._panel2d_job is not None:
            self._panel2d_pending = (region,)
            return
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap:
            return
        ui = self._ui_state()
        cam = ui.get('camera')
        if not cam:
            return
        col = ui['colouring']
        mode = self.view.panel2d_mode()
        key = (self.current_snap_idx, id(snap), mode, repr(col), tuple(sorted(ui['visible'])), self._features_gen)
        same_camera = cam == getattr(self, '_panel2d_cam', None)
        whole = region is None
        if whole:
            if same_camera and key == self._panel2d_key:
                return                      # nothing the picture depends on changed (e.g. only the LOD set)
            region = self._panel2d_region if same_camera else None
        elif not (same_camera and key == self._panel2d_key):
            return                          # a zoom into a picture that is out of date
        box = float(snap.get('header', {}).get('BoxSize', 0.0) or 0.0)
        snap, full = self._panel2d_particles(snap)
        mask = np.isin(snap['t'], list(ui['visible']))
        if not mask.any():
            return
        h = self._smoothing_lengths(snap, mask, box, wait=full)
        if h is None:
            return
        scale = 1.0 if full else float(snap.get('n_total', len(snap['c']))) / max(len(snap['c']), 1)
        m = snap['m'][mask].astype(np.float64) * scale
        pos = snap['c'][mask]
        q = None
        if mode == 2 and col['property'] != TYPE_MODE:
            q = self._scene_values(snap, col['property'], mask, ui, wait=full)
            if q is None:
                return
        source = "every particle" if full else "the display sample"
        npix_whole, npix_zoom = self._panel2d_pixels(400, 1200), self._panel2d_pixels(200, 1600)
        levels = self._panel2d_levels

        def draw(reg, npix, levels):
            if q is not None:
                img, width = project(pos, m, h, cam, npix, weights=q, box=box, region=reg)
                lo, hi = col['range']
                rgba = map_values(img.ravel(), lo, hi, col['log'], col['table'], alpha=1.0).reshape(img.shape + (4,))
                rgba[~np.isfinite(img)] = (0, 0, 0, 1)
                title = f"Smoothed mass-weighted {col['property']} ({lo:.3g} to {hi:.3g}{', log' if col['log'] else ''})"
            else:
                img, width = project(pos, m, h, cam, npix, box=box, region=reg)
                if reg is None:             # a zoom keeps the colour scale of the whole picture
                    pos_px = img[img > 0]
                    if len(pos_px) == 0:
                        return None, levels
                    lo, hi = np.percentile(pos_px, [1.0, 99.9])
                    levels = (lo, max(hi, lo * 10))
                lo, hi = levels
                rgba = map_values(img.ravel(), lo, hi, True, 'magma', alpha=1.0).reshape(img.shape + (4,))
                rgba[~(img > 0)] = (0, 0, 0, 1)
                title = f"Smoothed surface density, log ({lo:.3g} to {hi:.3g} code mass / length²)"
            return (rgba, width, f"{title} — picture from {source}"), levels

        def work():
            out = {'levels': levels}
            if whole:
                pic, out['levels'] = draw(None, npix_whole, levels)
                if pic is None:
                    return out
                out['whole'] = pic
            if region is not None:
                pic, _ = draw(region, npix_zoom, out['levels'])
                if pic is not None:
                    out['zoom'] = pic
            return out

        def show(out):
            self._panel2d_levels = out['levels']
            if 'whole' in out:
                self.view.show_image_2d(*out['whole'], reset_range=region is None)
                self._panel2d_cam, self._panel2d_key = cam, key
            if 'zoom' in out:
                rgba, width, title = out['zoom']
                x0, _, y0, _ = region
                self.view.show_image_2d(rgba, width, title, rect=(x0, y0, width, width * rgba.shape[0] / rgba.shape[1]),
                                        reset_range=False)
            self._panel2d_region = region if ('zoom' in out or 'whole' in out) else self._panel2d_region

        if len(pos) < PANEL2D_INLINE:
            show(work())
            return
        job = _Panel2DJob(work)

        def finished(out, err):
            job.wait()                      # run() is past its last emit; let the thread end before it goes
            self._panel2d_job = None
            if err:
                log(f"2D panel: {err}", 'warning')
            elif out is not None:
                show(out)                   # still right for its own camera; anything newer follows
            pending, self._panel2d_pending = self._panel2d_pending, None
            if pending is not None:
                self._render_panel2d(*pending)

        job.done.connect(finished)
        self._panel2d_job = job
        self.view.plot_2d.setTitle("Drawing the smoothed picture…")
        hold(job).start()

    def _on_panel2d_zoomed(self):
        """The smoothed image was zoomed or panned: draw what is on screen, at screen resolution."""
        if self.view.panel2d_mode() == 0 or not self.view.image_2d.isVisible():
            return
        (x0, x1), (y0, y1) = self.view.plot_2d.getViewBox().viewRange()
        px, py = PANEL2D_ZOOM_PAD * (x1 - x0), PANEL2D_ZOOM_PAD * (y1 - y0)
        self._render_panel2d(region=(x0 - px, x1 + px, y0 - py, y1 + py))

    def _finish_panel2d_job(self):
        """On quit: let a picture being drawn finish (a thread must not outlive the window)."""
        if self._panel2d_job is not None:
            self._panel2d_job.wait()

    def _rerender_panel2d(self):
        """Render button: draw the whole view again from scratch."""
        self._panel2d_key = self._panel2d_region = None
        if self.view.panel2d_mode() != 0:
            self._render_panel2d()

    def _poll_panel2d_camera(self):
        """Follow camera: re-render once the 3D view has stopped moving."""
        from PyQt6 import sip
        if sip.isdeleted(self.view):          # the window is gone; this timer is not its child
            self._panel2d_timer.stop()
            return
        if self.view.panel2d_mode() == 0 or not self.view.chk_2d_follow.isChecked():
            return
        cam = self.view.camera_state()
        if cam and cam == getattr(self, '_panel2d_last_poll', None) and cam != getattr(self, '_panel2d_cam', None):
            self._render_panel2d()
        self._panel2d_last_poll = cam
