"""
Level of detail for the 3D view, mixed into CosmoViewerController.

The snapshot dict keeps its base display sample (snap['c'], ...), which every
other part of the program uses unchanged. On top of it, the current snapshot
gets an LOD store (lod/store.py) and, whenever the camera settles, a view set:
the particles in view at the finest level that fits the particle budget,
read from the store or, when zoomed in past the store's resolution, from the
snapshot files. The view set is only drawn: FoF, tracking, plots and movies
read the files or the base sample as before.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QApplication

from controller.console import log
from controller.thread_hold import hold
from lod.budget import BUDGET_CHOICES, DEFAULT_BUDGET
from lod.frustum import camera_planes
from lod.store import build_store
from lod.view import eye_position, query_view, release_idle_pool
from movie.renderer import Camera, view_matrix

POLL_MS = 120               # camera is checked this often; a view is fetched once it stops moving
FRUSTUM_MARGIN = 0.15       # load a little beyond the screen edges so small pans need no reload


class _LODJob(QThread):
    done = pyqtSignal(object, object)        # token, result
    failed = pyqtSignal(object, str)
    progress = pyqtSignal(int, int)

    def __init__(self, token, fn):
        super().__init__()
        self.token, self.fn, self._cancel = token, fn, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(lambda i, n: self.progress.emit(i, n), lambda: self._cancel)
            if not self._cancel:
                self.done.emit(self.token, res)
        except Exception as e:  # reported in the console, the view keeps the base sample
            self.failed.emit(self.token, f"{type(e).__name__}: {e}")


class LODMixin:
    """
    Mixed into CosmoViewerController, whose attributes it uses: view, model,
    current_snap_idx, loading, is_playing and _update_frame.
    """

    def _init_lod(self, budget: Optional[int] = None):
        self.lod_budget = int(budget or DEFAULT_BUDGET)
        self.lod_enabled = True
        self._lod_store = None                # (store key, LODStore)
        self._lod_view = None                 # (snap idx, store key, camera key, ViewResult)
        self._lod_build: Optional[_LODJob] = None
        self._lod_query: Optional[_LODJob] = None
        self._lod_last_cam = None
        self._lod_stable = 0
        self._lod_force = False
        self._lod_reload = False              # next query makes a new set even if the old one serves
        self._lod_pending = None              # (token, ViewResult) that arrived while the camera moved
        self._lod_timer = QTimer()
        self._lod_timer.setInterval(POLL_MS)
        self._lod_timer.timeout.connect(self._lod_poll)
        self._lod_timer.start()
        self._lod_retired = []                # jobs still finishing: kept referenced until they end
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_lod)
        self._lod_menu()

    # ------------------------------------------------------------------ menu (View)

    def _lod_menu(self):
        bar = self.view.menuBar()
        view_menu = next((a.menu() for a in bar.actions() if a.text() == 'View' and a.menu()), None)
        if view_menu is None:
            view_menu = bar.addMenu('View')
        view_menu.addSeparator()
        sub = view_menu.addMenu('Particle budget')
        group = QActionGroup(self.view)
        choices = sorted(set(BUDGET_CHOICES) | {self.lod_budget})
        for n in choices:
            a = QAction(f"{n / 1e6:g} million on screen", self.view, checkable=True)
            a.setChecked(n == self.lod_budget)
            a.triggered.connect(lambda _=False, n=n: self._set_lod_budget(n))
            group.addAction(a)
            sub.addAction(a)
        self._lod_toggle = QAction('Level of detail (load more particles when zoomed in)', self.view, checkable=True)
        self._lod_toggle.setChecked(True)
        self._lod_toggle.toggled.connect(self._set_lod_enabled)
        view_menu.addAction(self._lod_toggle)
        resample = QAction('Resample view now', self.view)
        resample.setShortcut('Ctrl+R')
        resample.triggered.connect(self._lod_resample)
        view_menu.addAction(resample)

    def _set_lod_budget(self, n: int):
        self.lod_budget = int(n)
        log(f"Particle budget: {n:,} on screen")
        self._lod_resample()

    def _set_lod_enabled(self, on: bool):
        self.lod_enabled = bool(on)
        if not on:
            self._lod_view = self._lod_pending = None
            self._update_frame()
        else:
            self._lod_resample()

    def _lod_resample(self):
        self._lod_force = True
        self._lod_reload = True
        self._lod_poll()

    # ------------------------------------------------------------------ state

    def _retire(self, job):
        """
        Cancel a job without waiting for it: the window must never block on a read in flight.
        A QThread must not be garbage-collected while it still runs, so it stays referenced.
        """
        if job is not None and job.isRunning():
            job.cancel()
            self._lod_retired.append(job)
        self._lod_retired = [j for j in self._lod_retired if j.isRunning()]

    def _shutdown_lod(self):
        self._lod_timer.stop()
        self._reset_lod()
        for job in self._lod_retired:
            job.wait()

    def _reset_lod(self):
        self._retire(self._lod_build)
        self._retire(self._lod_query)
        self._lod_build = self._lod_query = None
        self._lod_store = None
        self._lod_view = None
        self._lod_pending = None
        self._lod_last_cam = None

    @staticmethod
    def _lod_interacting() -> bool:
        """A mouse button is held (a drag in progress): no new set is drawn until it is released."""
        return QApplication.mouseButtons() != Qt.MouseButton.NoButton

    def _lod_camera(self):
        st = self.view.camera_state()
        if not st:
            return None
        w3 = self.view.view_3d
        return Camera(**st), max(w3.width(), 1), max(w3.height(), 1)

    @staticmethod
    def _lod_cam_key(cam_tuple):
        cam, w, h = cam_tuple
        return (tuple(round(x, 6) for x in cam.center), round(cam.distance, 6), round(cam.elevation, 4),
                round(cam.azimuth, 4), round(cam.fov, 4), w, h)

    def _lod_poll(self):
        from PyQt6 import sip
        if sip.isdeleted(self.view):          # the window is gone; this timer is not its child
            self._lod_timer.stop()
            return
        if self._lod_query is None or not self._lod_query.isRunning():
            release_idle_pool()
        if not self.lod_enabled or self.loading or self.is_playing or not self.model.snapshots:
            return
        snap = self.model.get_snapshot(self.current_snap_idx)
        if snap is None or not snap.get('files') or not str(snap['filepath']).endswith(('.hdf5', '.h5')):
            return
        key = tuple(snap['files'])
        if self._lod_store is None or self._lod_store[0] != key:
            self._lod_start_build(key)
            return
        cam_tuple = self._lod_camera()
        if cam_tuple is None:
            return
        ck = self._lod_cam_key(cam_tuple)
        if ck != self._lod_last_cam:            # still moving: wait until it settles
            self._lod_last_cam = ck
            self._lod_stable = 0
            if not self._lod_force:
                return
        self._lod_stable += 1
        if self._lod_pending is not None:
            if self._lod_interacting():
                return
            token, res = self._lod_pending
            self._lod_pending = None
            if token[0] == self.current_snap_idx and token[1] == key and token[2] == ck:
                self._lod_apply(token, res)
                return
            if token[0] == self.current_snap_idx and token[1] == key:
                self._lod_start_query(self.current_snap_idx, key, ck, cam_tuple, candidate=res)
                return
        v = self._lod_view
        current = v is not None and v[0] == self.current_snap_idx and v[1] == key and v[2] == ck \
            and v[3].info.get('budget') == self.lod_budget
        if (current and not self._lod_force) or (self._lod_query is not None and self._lod_query.isRunning()):
            return
        self._lod_force = False
        self._lod_start_query(self.current_snap_idx, key, ck, cam_tuple)

    def _lod_start_build(self, key):
        if self._lod_build is not None and self._lod_build.isRunning():
            if self._lod_build.token == key:
                return
            self._retire(self._lod_build)
        files = list(key)
        job = _LODJob(key, lambda progress, cancelled: build_store(files, progress=progress, cancelled=cancelled))
        job.done.connect(self._lod_built)
        job.failed.connect(lambda token, msg: log(f"Level of detail unavailable for this snapshot: {msg}", 'warning'))
        job.progress.connect(lambda i, n: self.view.set_status_text(f"Indexing snapshot for level of detail: {i}/{n}"))
        self._lod_build = job
        hold(job).start()

    def _lod_built(self, key, store):
        if store is None:
            return
        self._lod_store = (key, store)
        src = "opened from cache" if store.info.get('from_cache') else "built"
        log(f"LOD store {src} in {store.build_seconds:.2f} s: {store.n_stored:,} of "
            f"{store.layout['n_total']:,} particles ({store.fraction:.3g}), "
            f"{len(store.cells.keys):,} cells, {len(store.blocks['start']):,} file blocks")
        self._lod_force = True
        self._lod_poll()

    def _lod_start_query(self, idx, key, ck, cam_tuple, candidate=None):
        store = self._lod_store[1]
        cam, w, h = cam_tuple
        planes = camera_planes(cam, w, h, margin=FRUSTUM_MARGIN)
        screen = camera_planes(cam, w, h, margin=0.0)
        budget = self.lod_budget
        v = self._lod_view
        previous = v[3] if v is not None and v[0] == idx and v[1] == key and not self._lod_reload else None
        if candidate is not None and not self._lod_reload:
            previous = candidate                    # a held set that may serve the camera where it stopped
        self._lod_reload = False
        self._lod_force = False

        eye = eye_position(view_matrix(cam))

        def run(progress, cancelled):
            return query_view(store, planes, budget, eye, cam.distance, cancelled,
                              previous=previous, screen_planes=screen)

        job = _LODJob((idx, key, ck), run)
        job.done.connect(self._lod_queried)
        job.failed.connect(lambda token, msg: log(f"Level of detail query failed: {msg}", 'warning'))
        self._lod_query = job
        hold(job).start()

    def _lod_queried(self, token, res):
        if res is None:
            return
        idx, key, ck = token
        if idx != self.current_snap_idx or self._lod_store is None or self._lod_store[0] != key:
            return                                  # snapshot changed meanwhile
        v = self._lod_view
        if v is not None and v[3] is res:           # same particles still serve: nothing to redraw
            self._lod_view = (idx, key, ck, res)
            return
        cam_tuple = self._lod_camera()
        if self._lod_interacting() or cam_tuple is None or self._lod_cam_key(cam_tuple) != ck:
            # Drawing a new set takes the window a moment; doing it while the camera moves is the
            # stutter that makes exploring feel slow. It is drawn once the camera stops.
            self._lod_pending = (token, res)
            return
        self._lod_apply(token, res)

    def _lod_apply(self, token, res):
        idx, key, ck = token
        self._lod_view = (idx, key, ck, res)
        self._update_frame()

    # ------------------------------------------------------------------ what the scene draws

    def _display_snapshot(self, idx: int) -> Optional[dict]:
        """The snapshot dict with the current view set in place of the base sample, if there is one."""
        snap = self.model.get_snapshot(idx)
        v = self._lod_view
        if snap is None or not self.lod_enabled or v is None or v[0] != idx or v[1] != tuple(snap.get('files') or ()):
            return snap
        res = v[3]
        d = dict(snap)
        d.update(res.data)
        d['_props'] = res.info.setdefault('_props', {})   # property values cached per view set, not the base
        if 'FoFGroupIndex' in snap:
            d['FoFGroupIndex'] = self._lod_fof_groups(snap, res)
        d['lod'] = res
        return d

    @staticmethod
    def _lod_fof_groups(snap: dict, res) -> np.ndarray:
        """FoF group (1-based, 0 = none) of each drawn particle, from the full-resolution FoF result."""
        memo = res.info.get('fof')
        if memo is not None and memo[0] is snap['FoFGroupIndex']:
            return memo[1]
        ids = res.data['id']
        fof = snap.get('fof')
        if fof is not None:
            groups = (np.asarray(fof.lookup(ids)) + 1).astype(np.int32)
        else:                                       # only the base sample's groups are known
            base_ids, base_g = snap['id'], snap['FoFGroupIndex']
            order = np.argsort(base_ids)
            pos = np.clip(np.searchsorted(base_ids[order], ids), 0, len(order) - 1)
            hit = base_ids[order][pos] == ids
            groups = np.where(hit, base_g[order][pos], 0).astype(np.int32)
        res.info['fof'] = (snap['FoFGroupIndex'], groups)
        return groups

    def _lod_status(self) -> str:
        """
        Status bar: how complete the drawn picture is, e.g. " | Detail level 7: 8.8% of particles
        drawn near the focus (~1.56M in view, 1.7 s)". "In view" is the snapshot's particles
        estimated to be in view, not the number drawn.
        """
        v = self._lod_view
        if not self.lod_enabled or v is None or v[0] != self.current_snap_idx:
            return ""
        r = v[3]
        pct = 100.0 * float(r.fraction)
        drawn = ("all particles" if pct >= 99.95 else
                 f"{pct:.0f}% of particles" if pct >= 10 else
                 f"{pct:.1f}% of particles" if pct >= 1 else f"{pct:.2g}% of particles")
        n = int(r.n_in_view)
        in_view = (f"{n / 1e9:.3g} billion" if n >= 1e9 else f"{n / 1e6:.3g}M" if n >= 1e6
                   else f"{n / 1e3:.3g}k" if n >= 1e4 else f"{n:,}")
        took = f"{r.seconds:.1f} s" if r.seconds >= 1 else f"{r.seconds * 1000:.0f} ms"
        return f" | Detail level {r.level}: {drawn} drawn near the focus (~{in_view} in view, {took})"
