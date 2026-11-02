"""
Picking lit particles: by clicking them in the 3D view, or from a list of the
particles with the highest values of the colouring property.

A picked particle goes into the active selection set by its ParticleID, so it
is followed through every snapshot by the existing tracking (full-resolution
lookups by ID, trails, "follow selection", trajectory export). The set says
how it was chosen (selection_sets.py): a click gives plain IDs ("picked
particles"); a row of the list gives "the n-th highest <property>" of the
snapshot the list was made for, complete only when the list came from every
particle rather than from the drawn ones.

How a click finds a particle, whatever the snapshot size: only what is drawn
can be clicked, and what is drawn is bounded by the display budget (the LOD
view set or the display sample), so the drawn lit particles are projected to
the screen with the 3D view's own matrices and the nearest one to the cursor,
within a few pixels, is taken (ties go to the one nearest the camera). That is
a fixed cost per click (tens of milliseconds for millions of points) and needs
no extra GPU pass. GPU colour-ID picking (rendering particle indices into an
off-screen buffer) would make a click O(1), but pyqtgraph does not expose
off-screen passes.

The list ("List top") is built from the full snapshot for file properties,
through the read-once property cache (snapshot_io.property_cache), and from
the drawn particles for scene quantities and attached data.
"""

from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from controller.console import log
from controller.thread_hold import hold
from views.colouring_panel import SCENE_MODES, TYPE_MODE

PICK_RADIUS_PX = 6.0
TOP_COUNT = 200
TYPE_NAMES = {0: 'gas', 1: 'DM', 2: 'disk', 3: 'bulge', 4: 'star', 5: 'BH'}


def _fmt(v: float) -> str:
    """Whole numbers (group and subhalo numbers) in full, everything else to 4 digits."""
    return str(int(v)) if np.isfinite(v) and float(v).is_integer() and abs(v) < 1e15 else f"{v:.4g}"


class _TopJob(QThread):
    done = pyqtSignal(object, str)      # (ids, types, values) or None, error

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), '')
        except Exception as e:  # reported in the panel and the console
            self.done.emit(None, f"{type(e).__name__}: {e}")


class PickingMixin:

    def _init_picking(self):
        self._pick_candidates = None          # (pos, ids, types) of what can be clicked
        self._top_list: List[Tuple[int, int, float]] = []
        self._top_job: Optional[_TopJob] = None
        self._recentre_on_pick = False
        sp = self.view.selection_panel
        self.view.pick_requested.connect(self._on_pick)
        sp.pick_list_requested.connect(self._list_top)
        sp.pick_list_chosen.connect(self._pick_from_list)

    # ------------------------------------------------------------------ what can be clicked

    def _remember_pick_candidates(self, scene):
        """Lit particles when colouring by a property, otherwise everything drawn."""
        if scene.lit_pos is not None:
            self._pick_candidates = (scene.lit_pos, scene.lit_ids, scene.lit_types, scene.lit_values)
        elif scene.ids is not None:
            self._pick_candidates = (scene.pos, scene.ids, None, None)
        else:
            self._pick_candidates = None

    def _after_scene(self, scene):
        """Centre on a particle chosen from the list once its position is known."""
        if self._recentre_on_pick and scene.centre is not None:
            self.view.set_camera_centre(scene.centre)
            self._recentre_on_pick = False

    @staticmethod
    def nearest_on_screen(x: np.ndarray, y: np.ndarray, depth: np.ndarray, front: np.ndarray,
                          cx: float, cy: float, radius: float) -> Optional[int]:
        """Index of the point nearest (cx, cy) within ``radius`` px; ties go to the nearest depth."""
        near = front & (np.abs(x - cx) <= radius) & (np.abs(y - cy) <= radius)   # cheap box test first
        cand = np.flatnonzero(near)
        if len(cand) == 0:
            return None
        d2_c = (x[cand] - cx) ** 2 + (y[cand] - cy) ** 2
        d2 = np.full(len(x), np.inf)
        d2[cand] = d2_c
        ok = np.isfinite(d2) & (d2 <= radius * radius)
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)
        best = d2[idx].min()
        close = idx[d2[idx] <= best + 1.0]          # within a pixel of the best: prefer the nearest
        return int(close[np.argmin(depth[close])])

    def _on_pick(self, cx: float, cy: float, add: bool):
        sp = self.view.selection_panel
        if not sp.chk_pick.isChecked() or self._pick_candidates is None:
            return
        pos, ids, types, values = self._pick_candidates
        if pos is None or len(pos) == 0 or ids is None:
            sp.lbl_pick.setText("Nothing lit to pick in this view.")
            return
        x, y, depth, front = self.view.screen_coords(pos)
        size = self.view.colouring_panel.drawing_state()['lit']['size']
        k = self.nearest_on_screen(x, y, depth, front, cx, cy, max(PICK_RADIUS_PX, size))
        if k is None:
            sp.lbl_pick.setText("No lit particle under the cursor.")
            return
        pid = int(ids[k])
        t = None if types is None else int(types[k])
        snap = self.model.get_snapshot(self.current_snap_idx)
        # a picked particle is exactly the one clicked: plain IDs, picked at this snapshot
        meaning = dict(kind='ids', definition={'source': 'pick'}, complete=True,
                       defined_at=self._defined_at(snap, self.current_snap_idx) if snap else {})
        self._pick_ids(np.array([pid], dtype=np.uint64), 'add' if add else 'new',
                       self._describe(pid, t, None if values is None else float(values[k])), meaning)

    def _describe(self, pid: int, t: Optional[int], value: Optional[float]) -> str:
        prop = self.view.colouring_panel.state()['property']
        txt = f"ID {pid}"
        if t is not None:
            txt += f" ({TYPE_NAMES.get(t, f'type {t}')})"
        if value is not None and prop != TYPE_MODE:
            txt += f", {prop} = {_fmt(value)}"
        return txt

    def _pick_ids(self, ids: np.ndarray, mode: str, text: str, meaning: Optional[dict] = None):
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        if not s_set.enabled:
            s_set.enabled = True
            sp._refresh_ui()
        self._apply_ids_to_set(s_set, ids, mode, meaning)   # starts the lookups in every snapshot
        sp.lbl_pick.setText(f"Picked {text} into {s_set.name}; it is followed through the snapshots by ID.")
        log(f"Picked {text} into set {s_set.set_id + 1} ({'added' if mode == 'add' else 'new'})")
        if len(ids) == 1:
            self.inspect_particle(int(ids[0]), text)     # the inspector shows what it is and how it changes

    # ------------------------------------------------------------------ list of the highest values

    def _list_top(self):
        sp = self.view.selection_panel
        snap = self.model.get_snapshot(self.current_snap_idx)
        col = self.view.colouring_panel.state()
        name = col['property']
        if not snap or name == TYPE_MODE:
            sp.lbl_pick.setText("Colour by a property first; the list shows its highest values.")
            return
        if self._top_job is not None and self._top_job.isRunning():
            return
        self._top_job = None                  # finished: safe to release here, outside its own signal
        visible = [t for t in range(6) if self.view.layer_panel.layer_state.is_visible(t)]
        info = self._file_properties(snap).get(name)
        if info is not None and name not in SCENE_MODES:
            from snapshot_io.property_cache import top_values
            files = [str(f) for f in (snap.get('files') or [snap['filepath']])]
            fn = lambda: top_values(files, info, visible, TOP_COUNT)
            where, scope = "full snapshot", 'full'
        else:
            ui = self._ui_state()
            dsnap = self._display_snapshot(self.current_snap_idx)
            vis = np.isin(dsnap['t'], visible)
            vals = self._scene_values(dsnap, name, vis, ui, wait=True)
            if vals is None:
                sp.lbl_pick.setText(f"{name} is not available for the drawn particles.")
                return
            ids, types = dsnap['id'][vis], dsnap['t'][vis]

            def fn():
                ok = np.flatnonzero(np.isfinite(vals))
                order = ok[np.argsort(-vals[ok], kind='stable')[:TOP_COUNT]]
                return ids[order], types[order], vals[order]
            where = "drawn particles"
            scope = 'full' if self._sample_is_everything(dsnap) else 'sample'
        # What a particle chosen from this list is, fixed now: the list belongs to this snapshot
        # even if another one is shown by the time a row is chosen.
        context = dict(field=name, scope=scope, types=visible,
                       units=info.unit_label if info is not None else '',
                       defined_at=self._defined_at(snap, self.current_snap_idx))
        sp.btn_top.setEnabled(False)
        sp.lbl_pick.setText(f"Finding the highest {name} ({where})...")
        self._top_job = _TopJob(fn)
        self._top_job.done.connect(lambda res, err, n=name, w=where, c=context: self._on_top_done(res, err, n, w, c))
        hold(self._top_job).start()

    def _on_top_done(self, res, err: str, name: str, where: str, context: Optional[dict] = None):
        sp = self.view.selection_panel
        sp.btn_top.setEnabled(True)
        if self._top_job is not None:
            self._top_job.wait()              # kept referenced until the next listing (see _list_top)
        if err or res is None:
            sp.lbl_pick.setText(f"Could not list {name}: {err}")
            log(f"List top {name}: {err}", 'error')
            return
        ids, types, vals = res
        self._top_list = [(int(i), int(t), float(v)) for i, t, v in zip(ids, types, vals)]
        self._top_context = dict(context or {}, n=len(self._top_list))
        sp.combo_top.clear()
        for r, (i, t, v) in enumerate(self._top_list, start=1):
            sp.combo_top.addItem(f"{r}. {name} {_fmt(v)}  ID {i} ({TYPE_NAMES.get(t, t)})")
        sp.lbl_pick.setText(f"Highest {len(self._top_list)} values of {name} ({where}). Choose one to pick it.")

    def _pick_from_list(self, row: int):
        if not (0 <= row < len(self._top_list)):
            return
        pid, t, v = self._top_list[row]
        self._recentre_on_pick = True
        mode = 'add' if self.view.selection_panel.get_selection_mode() == 'add' else 'new'
        self._pick_ids(np.array([pid], dtype=np.uint64), mode, self._describe(pid, t, v), self._top_meaning(row))

    def _top_meaning(self, row: int) -> Optional[dict]:
        """A particle chosen from the list is a query: "the rank-th highest <field>" at the
        snapshot the list was made for; complete only when the list came from every particle."""
        ctx = getattr(self, '_top_context', None)
        if not ctx:
            return None
        full = ctx.get('scope') == 'full'
        criterion = {'kind': 'top', 'field': ctx['field'], 'n': int(ctx.get('n', len(self._top_list))),
                     'rank': int(row) + 1, 'scope': ctx.get('scope', 'sample'), 'units': ctx.get('units', ''),
                     'types': list(ctx.get('types', []))}
        return dict(kind='query', definition={'criteria': [criterion]}, defined_at=dict(ctx.get('defined_at') or {}),
                    complete=full, n_expected=1 if full else None,
                    incomplete_reason=None if full else 'display_sample')
