"""
What to draw for a camera: the visible particles at the finest level of
detail that fits the particle budget.

Sampling rule. Every particle has a fixed random number u (lod/store.py). A
particle in leaf cell C is drawn when u < f(C), with

    f(p) = min(1, f_focus * w(d)),   w(d) = (D / max(d, D)) ** DEPTH_FALLOFF

where d is the particle's distance from the eye and D the camera distance
(eye to the point the view orbits). Everything up to the focus distance is
drawn at the same fraction f_focus, so the density there is faithful; beyond
the focus the fraction falls as 1/d^2, like the area it covers on screen.
Without this, a zoom into a box would still count every particle behind the
focus and could never get sharp. f_focus is the finest LOD level 2^(-k/2)
whose expected count fits the budget. When the focus is already at full
resolution and budget is left over, a second, wider falloff spends it on the
background, from the store only (ViewRule.wide, widen_focus).

Whether a cell is drawn from the store or from the files depends on f at the
cell's point nearest the eye, so every particle has exactly one source. Store
cells then sample at f of the cell centre.

Two sources, one rule:
  * cells with f(C) <= the store's fraction come from the LOD store
    (contiguous slices of memory-mapped arrays, milliseconds);
  * cells with f(C) above it are read from the snapshot files: only the
    blocks whose occupancy touches the view and whose nearest point needs
    more than the store holds, so the cost follows what is on screen.
"""

import time
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np

from lod.budget import background_workers, lower_priority
from lod.cells import cell_index, level_fraction
from lod.frustum import INSIDE, OUTSIDE, classify_boxes, points_inside
from lod.store import SUB, LODStore, _read_rows, make_view, row_uniforms

DEPTH_FALLOFF = 2.0
MAX_LEVEL = 80                   # 2^-40: far below one particle of any snapshot


@dataclass
class ViewRule:
    """The sampling rule of one view (see module docstring)."""
    f_focus: float
    eye: np.ndarray
    focus_distance: float
    origin: np.ndarray
    size: float
    depth: int
    falloff: float = DEPTH_FALLOFF
    wide: float = 0.0                  # background radius (0: none), see widen_focus
    wide_cap: float = 0.0              # most the background draws: the store's fraction

    def weight(self, d: np.ndarray) -> np.ndarray:
        if self.falloff == 0:
            return np.ones(len(d))
        return (self.focus_distance / np.maximum(d, self.focus_distance)) ** self.falloff

    def _f_at(self, pts: np.ndarray) -> np.ndarray:
        d = pts - self.eye
        d = np.sqrt(np.einsum('ij,ij->i', d, d))
        f = np.minimum(1.0, self.f_focus * self.weight(d))
        if self.wide > 0:
            f = np.maximum(f, np.minimum(self.wide_cap, (self.wide / np.maximum(d, self.wide)) ** self.falloff))
        return f

    def centres_of(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + (ijk + 0.5) * (self.size / (1 << self.depth))

    def f_centres(self, centres: np.ndarray) -> np.ndarray:
        """f at cell centres."""
        return self._f_at(centres)

    def f_centres_max(self, centres: np.ndarray) -> np.ndarray:
        """f at each cell's point nearest the eye: the most any of its particles needs."""
        h = 0.5 * self.size / (1 << self.depth)
        return self._f_at(np.clip(self.eye, centres - h, centres + h))

    def f_points(self, pos: np.ndarray) -> np.ndarray:
        return self._f_at(np.asarray(pos, dtype=np.float64))

    def from_file(self, pos: np.ndarray, store_fraction: float) -> np.ndarray:
        """Particles whose cell is drawn from the files rather than the store."""
        ijk = cell_index(pos, self.origin, self.size, self.depth)
        return self.f_centres_max(self.centres_of(ijk)) > store_fraction


@dataclass
class ViewResult:
    data: Dict                     # c, v, m, id, t, rows (snap layout)
    level: int                     # LOD level at the focus: fraction drawn there = 2^(-level/2)
    fraction: float                # that fraction
    n_in_view: int                 # snapshot particles estimated to be in view
    source: str                    # 'store', 'file' or 'store+file'
    seconds: float = 0.0
    blocks_read: int = 0
    rows_read: int = 0
    info: Dict = field(default_factory=dict)

    @property
    def n_drawn(self) -> int:
        return len(self.data['t'])


def eye_position(view_matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(view_matrix)[:3, 3]


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum('ij,ij->i', v, v))


def widen_focus(est: np.ndarray, dist: np.ndarray, focus: float, budget: int, falloff: float,
                slack: float = 0.0, base: Optional[np.ndarray] = None, cap: float = 1.0) -> float:
    """
    The largest background radius R >= ``focus`` whose expected count,
    sum(est * max(base, min(cap, (R / max(d, R))^falloff))), fits the budget (bisection in
    log R); ``base`` is the fraction already drawn (full resolution at the focus). ``slack``:
    how far beyond its ``dist`` a counted particle can lie (half a cell diagonal for centres).
    """
    base = np.zeros(len(dist)) if base is None else base

    def count(r):
        return float(np.sum(est * np.maximum(base, np.minimum(cap, (r / np.maximum(dist, r)) ** falloff))))
    far = float(dist.max()) if len(dist) else focus
    if far <= focus or count(far) <= budget:      # everything in view fits: no falloff at all
        return max(far, focus) + slack
    lo, hi = np.log(focus), np.log(far)           # count(e^lo) fits, count(e^hi) does not
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if count(np.exp(mid)) <= budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return float(np.exp(lo))


def choose_level(est: np.ndarray, w: np.ndarray, budget: int) -> int:
    """Finest level k whose expected count sum(est * min(1, 2^(-k/2) * w)) fits the budget (bisection)."""
    def fits(k):
        return float(np.sum(est * np.minimum(1.0, level_fraction(k) * w))) <= budget
    lo, hi = 0, MAX_LEVEL
    if fits(lo):
        return lo
    while hi - lo > 1:                   # fits(hi) holds, fits(lo) does not
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi


# A settled camera keeps the particles already drawn while they still serve it: the level within KEEP_LEVELS,
# the eye moved less than this fraction of the camera distance, and the screen lies inside
# the region that was loaded. Small rotations and pans then cost no reload and no redraw.
KEEP_EYE_SHIFT = 0.25
KEEP_LEVELS = 1                  # a set one level off (a factor sqrt 2 in density) still serves


def query_view(store: LODStore, planes: np.ndarray, budget: int, eye: np.ndarray, focus_distance: float,
               cancelled: Callable[[], bool] = lambda: False, allow_file: bool = True,
               falloff: float = DEPTH_FALLOFF, previous: Optional[ViewResult] = None,
               screen_planes: Optional[np.ndarray] = None) -> Optional[ViewResult]:
    """
    The view set for this camera. ``planes`` bound the region loaded (the screen plus a margin).
    Given the ``previous`` result and the bare ``screen_planes``, returns ``previous`` itself
    when it still serves this camera (see KEEP_EYE_SHIFT).
    """
    t0 = time.time()
    ct = store.cells
    s = store.fraction
    cells, partial = ct.visible_cells(planes, with_partial=True)
    rule = ViewRule(1.0, np.asarray(eye, dtype=np.float64), float(focus_distance), ct.origin, ct.size,
                    ct.depth, falloff)
    cen = ct.centres(cells)
    # Expected count per unit fraction: whole cells inside the view count at their centre; for
    # cells cut by the view, the stored particles actually inside it count at their own distance.
    full = ~partial
    est = [ct.counts[cells[full]] / s]
    dist = [_norm(cen[full] - rule.eye)]
    if partial.any():
        prow = _ranges_of(ct, cells[partial])
        ppos = np.asarray(store.arrays['pos'][prow], dtype=np.float64)
        ppos = ppos[points_inside(planes, ppos)]
        est.append(np.full(len(ppos), 1.0 / s))
        dist.append(_norm(ppos - rule.eye))
    est, dist = np.concatenate(est), np.concatenate(dist)
    w = rule.weight(dist)                               # f_focus = 1: these are the weights
    level = choose_level(est, w, budget)
    rule.f_focus = level_fraction(level)
    if level == 0 and falloff:
        # Full resolution at the focus and budget to spare: spend it on the scene behind the
        # focus, which otherwise thins as (D/d)^2 with a tiny D, so a deep zoom went dark while
        # most of the budget was unused. The background comes from the store only (at most its
        # fraction), so it costs no file reads.
        rule.wide_cap = min(s, 1.0)
        rule.wide = widen_focus(est, dist, rule.focus_distance, budget, falloff,
                                slack=0.8661 * ct.size / (1 << ct.depth), base=w, cap=rule.wide_cap)
    if previous is not None and screen_planes is not None and _still_serves(
            store, previous, level, budget, rule, screen_planes):
        return previous
    from_store = (rule.f_centres_max(cen) <= s * (1 + 1e-9)) | (s >= 1.0) | (not allow_file)
    parts = []
    for sel, cut in ((from_store & ~partial, False), (from_store & partial, True)):
        if not sel.any():
            continue
        r = store.below(cells[sel], np.minimum(rule.f_centres(cen[sel]), s))
        if cut and len(r):                  # cells cut by the view: drop their particles off it
            r = r[points_inside(planes, np.asarray(store.arrays['pos'][r]))]
        parts.append(r)
    rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    data = store.subset(rows)
    source, blocks, rows_read = 'store', 0, 0
    if not from_store.all():
        detail = read_detail(store, planes, rule, cancelled)
        if detail is None:
            return None
        data = _merge(data, detail.data)
        source = 'file' if from_store.sum() == 0 else 'store+file'
        blocks, rows_read = detail.blocks_read, detail.rows_read
    n_view = int(round(est.sum()))
    return ViewResult(data, level, rule.f_focus, n_view, source, time.time() - t0, blocks, rows_read,
                      {'rule': rule, 'planes': planes, 'budget': budget})


def _still_serves(store: LODStore, prev: ViewResult, level: int, budget: int, rule: ViewRule,
                  screen: np.ndarray) -> bool:
    """True if ``prev`` (same store) already holds what this camera would draw, up to small changes."""
    old_rule, old_planes = prev.info.get('rule'), prev.info.get('planes')
    if old_rule is None or old_planes is None or abs(prev.level - level) > KEEP_LEVELS \
            or prev.info.get('budget') != budget:
        return False
    if np.linalg.norm(rule.eye - old_rule.eye) > KEEP_EYE_SHIFT * old_rule.focus_distance:
        return False
    if abs(rule.focus_distance / old_rule.focus_distance - 1.0) > KEEP_EYE_SHIFT:
        return False
    ct = store.cells
    cells = ct.visible_cells(screen)
    if len(cells) == 0:
        return True
    h = 0.5 * ct.size / (1 << ct.depth)
    cen = ct.centres(cells)
    state = classify_boxes(old_planes, cen - h, cen + h)
    if (state == OUTSIDE).any():
        return False
    cut = state != INSIDE                # cells the old region cuts: test their stored particles
    if cut.any():
        pos = np.asarray(store.arrays['pos'][_ranges_of(ct, cells[cut])], dtype=np.float64)
        on_screen = points_inside(screen, pos)
        if (on_screen & ~points_inside(old_planes, pos)).any():
            return False
    return True


def _ranges_of(ct, cells):
    from lod.cells import _ranges
    return _ranges(ct.starts[cells], ct.counts[cells])


def _merge(a: dict, b: dict) -> dict:
    """Two display sets with disjoint particles, merged in (type, piece, row) order."""
    if len(b['t']) == 0:
        return a
    if len(a['t']) == 0:
        return b
    # Both are already in (type, piece, row) order: merge them by one search, not a sort.
    key = lambda d: (d['t'].astype(np.int64) << 48) | (_pieces(d).astype(np.int64) << 32) | _rows(d)
    ka, kb = key(a), key(b)
    at_b = np.searchsorted(ka, kb) + np.arange(len(kb))
    from_b = np.zeros(len(ka) + len(kb), dtype=bool)
    from_b[at_b] = True

    def merged(x, y):
        out = np.empty((len(from_b),) + x.shape[1:], dtype=np.result_type(x, y))
        out[~from_b], out[from_b] = x, y
        return out
    k = merged(ka, kb)
    return make_view(merged(a['c'], b['c']), merged(a['v'], b['v']), merged(a['m'], b['m']),
                     merged(a['id'], b['id']), (k >> 48).astype(np.uint8), ((k >> 32) & 0xFFFF).astype(np.uint16),
                     (k & 0xFFFFFFFF).astype(np.int64))


def _pieces(d):
    out = np.empty(len(d['t']), dtype=np.uint16)
    k = 0
    for (piece, t), r in sorted(d['rows'].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out[k:k + len(r)] = piece
        k += len(r)
    return out


def _rows(d):
    parts = [r for _, r in sorted(d['rows'].items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def candidate_blocks(store: LODStore, planes: np.ndarray, rule: Optional[ViewRule] = None) -> np.ndarray:
    """File blocks with an occupied sub-cell in view (and, given a rule, needing more than the store)."""
    b = store.blocks
    if len(b['start']) == 0:
        return np.empty(0, dtype=np.int64)
    state = classify_boxes(planes, b['lo'], b['hi'])
    cand = np.flatnonzero(state != OUTSIDE)
    if rule is not None and len(cand):
        # A block can hold file-drawn particles only if some cell touching it has its nearest
        # point within reach: the block's distance to the eye less one cell diagonal.
        nearest = np.clip(rule.eye, b['lo'][cand], b['hi'][cand])
        slack = rule.size / (1 << rule.depth) * 1.7321
        d = np.maximum(np.linalg.norm(nearest - rule.eye, axis=1) - slack, 0.0)
        f_max = rule.f_focus * ((rule.focus_distance / np.maximum(d, rule.focus_distance)) ** rule.falloff
                                if rule.falloff else np.ones(len(d)))
        cand = cand[np.minimum(f_max, 1.0) > store.fraction]
    if len(cand) == 0:
        return cand
    lo, hi = b['lo'][cand], b['hi'][cand]
    span = (hi - lo) / SUB
    ijk = np.stack(np.unravel_index(np.arange(SUB ** 3), (SUB, SUB, SUB), order='F'), axis=1)  # bit = i + SUB*(j + SUB*k)
    sub_lo = (lo[:, None, :] + ijk[None, :, :] * span[:, None, :]).reshape(-1, 3)
    sub_hi = sub_lo + np.repeat(span, SUB ** 3, axis=0)
    vis = (classify_boxes(planes, sub_lo, sub_hi) != OUTSIDE).reshape(len(cand), SUB ** 3)
    bits = b['bits'][cand]
    occ = ((bits[:, None] >> np.arange(SUB ** 3, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    return cand[(vis & occ).any(axis=1)]


DETAIL_PARALLEL_ROWS = 2_000_000   # read in worker processes above this many rows
_POOL = None


DETAIL_RUN_ROWS = 1 << 20          # one read of consecutive blocks: bounds a worker's memory (~70 MB)


def _pool():
    """Worker processes for detail reads, started once (starting them costs ~1 s on macOS)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        # spawn, not fork: the pool starts from a QThread, and forking a threaded process can deadlock
        _POOL = ProcessPoolExecutor(max_workers=background_workers(),
                                    mp_context=multiprocessing.get_context('spawn'),
                                    initializer=lower_priority)
    global _POOL_USED
    _POOL_USED = time.time()
    return _POOL


POOL_IDLE_SECONDS = 30.0           # workers hold ~150-200 MB each after a read (measured on the M3)
_POOL_USED = 0.0


def release_idle_pool(idle_seconds: float = POOL_IDLE_SECONDS) -> bool:
    """
    Stop the worker processes once no detail read has used them for ``idle_seconds``, to give
    their memory back; the next deep zoom starts them again (~1 s). Call only while no read runs.
    """
    global _POOL
    if _POOL is None or time.time() - _POOL_USED < idle_seconds:
        return False
    pool, _POOL = _POOL, None
    pool.shutdown(wait=False, cancel_futures=True)
    return True


def _empty_cols():
    return {'pos': np.empty((0, 3), np.float32), 'v': np.empty((0, 3), np.float32), 'm': np.empty(0, np.float32),
            'id': np.empty(0, np.uint64), 't': np.empty(0, np.uint8), 'piece': np.empty(0, np.uint16),
            'row': np.empty(0, np.uint32)}


def read_runs(args) -> dict:
    """
    Read runs of consecutive blocks [(piece, ptype, start, stop, [block starts])] and keep
    the particles the rule draws from the files. Runs in a worker process or in-thread.
    """
    files, runs, rule, planes, store_fraction, seed, mass_table = args
    out = {k: [] for k in ('pos', 'v', 'm', 'id', 't', 'piece', 'row')}
    handles: Dict[int, h5py.File] = {}
    try:
        for piece, ptype, s, e, bstarts in runs:
            if piece not in handles:
                handles[piece] = h5py.File(files[piece], 'r')
            g = handles[piece][f'PartType{ptype}']
            pos = g['Coordinates'][s:e]
            u, all_ids = row_uniforms(g, seed, piece, ptype, s, e)
            sel = u < rule.f_points(pos)
            sel[sel] = rule.from_file(pos[sel], store_fraction)
            sel[sel] = points_inside(planes, pos[sel])
            n = int(sel.sum())
            if n == 0:
                continue
            ids, v, m = _read_rows(g, sel, s, e, float(mass_table[ptype]), n, all_ids)
            out['pos'].append(pos[sel].astype(np.float32)); out['v'].append(v); out['m'].append(m)
            out['id'].append(ids); out['t'].append(np.full(n, ptype, dtype=np.uint8))
            out['piece'].append(np.full(n, piece, dtype=np.uint16))
            out['row'].append((s + np.nonzero(sel)[0]).astype(np.uint32))
    finally:
        for h in handles.values():
            h.close()
    if not out['t']:
        return _empty_cols()
    return {k: np.concatenate(v) for k, v in out.items()}


def read_detail(store: LODStore, planes: np.ndarray, rule: ViewRule,
                cancelled: Callable[[], bool] = lambda: False,
                parallel: Optional[bool] = None) -> Optional[ViewResult]:
    """Particles in view that the rule draws from the files (see module docstring)."""
    b = store.blocks
    cand = candidate_blocks(store, planes, rule)
    order = np.lexsort((b['start'][cand], b['piece'][cand], b['ptype'][cand]))
    cand = cand[order]
    # Consecutive blocks of one (type, piece) become one read of at most DETAIL_RUN_ROWS rows.
    runs: List[list] = []
    for k in cand:
        piece, ptype, s, e = int(b['piece'][k]), int(b['ptype'][k]), int(b['start'][k]), int(b['stop'][k])
        if runs:
            r = runs[-1]
            if r[0] == piece and r[1] == ptype and r[3] == s and e - r[2] <= DETAIL_RUN_ROWS:
                r[3] = e
                r[4].append(s)
                continue
        runs.append([piece, ptype, s, e, [s]])
    rows_read = int(sum(r[3] - r[2] for r in runs))
    common = (store.layout['files'], rule, planes, store.fraction, store.seed, store.layout['mass_table'])
    if parallel is None:
        parallel = rows_read > DETAIL_PARALLEL_ROWS
    if not parallel:
        if cancelled():
            return None
        cols = read_runs((common[0], runs) + common[1:])
    else:
        pool = _pool()
        n_tasks = 4 * pool._max_workers
        target = max(rows_read // n_tasks, 1)
        groups, cur, acc = [], [], 0
        for r in runs:                      # contiguous groups keep the (type, piece, row) order
            cur.append(r)
            acc += r[3] - r[2]
            if acc >= target:
                groups.append(cur)
                cur, acc = [], 0
        if cur:
            groups.append(cur)
        futs = [pool.submit(read_runs, (common[0], g) + common[1:]) for g in groups]
        parts = []
        for f in futs:
            while True:
                if cancelled():
                    for x in futs:
                        x.cancel()
                    return None
                try:
                    parts.append(f.result(timeout=0.1))
                    break
                except TimeoutError:
                    continue
        cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else _empty_cols()
    data = make_view(cols['pos'], cols['v'], cols['m'], cols['id'], cols['t'], cols['piece'], cols['row'])
    return ViewResult(data, 0, rule.f_focus, 0, 'file', blocks_read=len(cand), rows_read=rows_read)


def base_sample(store: LODStore, n: int) -> dict:
    """
    A uniform sample of n particles of the whole snapshot (the playback/overview sample): those
    with the n smallest u. u follows the ParticleID, so a run whose particles keep their IDs
    gets the same sample in every snapshot.
    """
    return store.subset(smallest_u_rows(np.asarray(store.arrays['u']), n))


def smallest_u_rows(u: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n smallest values of u (ties at the cut are all kept or all left out)."""
    if n >= len(u):
        return np.arange(len(u))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    cut = np.partition(u, n)[n]                 # the (n+1)-th smallest: everything below it is kept
    return np.flatnonzero(u < cut)
