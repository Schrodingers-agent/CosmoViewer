"""
Particles ordered for level of detail: sorted by octree leaf cell, shuffled
inside each cell.

Because each cell's particles are in random order, the first k of a cell are
a uniform random sample of that cell. Every level of detail is therefore a
set of contiguous slices, one per visible cell, and a finer level contains
every particle of a coarser one (no popping when zooming in).

The same fraction is taken from every visible cell, so the drawn particles
are a uniform sample of what is in view and their on-screen density stays
proportional to the true density. Cells are only used to skip what is off
screen and to read contiguous runs.
"""

from dataclasses import dataclass
from math import ceil, log2, sqrt
from typing import Tuple

import numpy as np

from lod.frustum import INSIDE, OUTSIDE, PARTIAL, classify_boxes

MAX_DEPTH = 7                # 128^3 leaf grid: cell table stays under ~40 MB
COARSE_DEPTH = 4             # 16^3 cells tested first, leaves only where partly visible
LEVELS_PER_OCTAVE = 2        # LOD level k keeps 2^(-k/2) of the stored particles


def level_fraction(level: int) -> float:
    return 2.0 ** (-level / LEVELS_PER_OCTAVE)


def level_for(n_visible: int, budget: int) -> int:
    """Finest level whose expected count fits the budget."""
    if n_visible <= budget or n_visible == 0:
        return 0
    return int(ceil(LEVELS_PER_OCTAVE * log2(n_visible / budget) - 1e-9))


def choose_depth(n: int, per_leaf: int = 32) -> int:
    if n <= per_leaf:
        return 1
    return int(min(MAX_DEPTH, max(1, ceil(log2(n / per_leaf) / 3))))


def _part1by2(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.uint64) & np.uint64(0x3FF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x030000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x0300F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x030C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x09249249)
    return x


def _compact1by2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) & np.uint64(0x09249249)
    x = (x | (x >> np.uint64(2))) & np.uint64(0x030C30C3)
    x = (x | (x >> np.uint64(4))) & np.uint64(0x0300F00F)
    x = (x | (x >> np.uint64(8))) & np.uint64(0x030000FF)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x3FF)
    return x


def morton(ijk: np.ndarray) -> np.ndarray:
    return _part1by2(ijk[:, 0]) | (_part1by2(ijk[:, 1]) << np.uint64(1)) | (_part1by2(ijk[:, 2]) << np.uint64(2))


def demorton(keys: np.ndarray) -> np.ndarray:
    k = keys.astype(np.uint64)
    return np.stack([_compact1by2(k), _compact1by2(k >> np.uint64(1)), _compact1by2(k >> np.uint64(2))],
                    axis=1).astype(np.int64)


def cell_index(pos: np.ndarray, origin: np.ndarray, size: float, depth: int) -> np.ndarray:
    n = 1 << depth
    ijk = np.floor((np.asarray(pos, dtype=np.float64) - origin) * (n / size)).astype(np.int64)
    np.clip(ijk, 0, n - 1, out=ijk)
    return ijk


def lod_order(pos: np.ndarray, origin, size: float, depth: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Permutation that sorts by leaf cell then randomly within the cell, and the sorted keys."""
    keys = morton(cell_index(pos, np.asarray(origin, dtype=np.float64), size, depth))
    rnd = np.random.default_rng(seed).integers(0, 1 << 30, size=len(keys), dtype=np.uint64)
    composite = (keys << np.uint64(30)) | rnd           # 3*7 + 30 bits fit in 64
    perm = np.argsort(composite, kind='stable')
    return perm, keys[perm]


@dataclass
class CellTable:
    """Leaf cells of a cube (origin, size) at ``depth``; rows of each cell are contiguous."""
    origin: np.ndarray
    size: float
    depth: int
    keys: np.ndarray          # occupied leaf Morton keys, increasing
    starts: np.ndarray        # first row of each cell
    counts: np.ndarray        # rows in each cell
    jitter: np.ndarray        # per-cell offset in [0, 1) for unbiased rounding

    @classmethod
    def from_sorted_keys(cls, sorted_keys: np.ndarray, origin, size: float, depth: int, seed: int = 0):
        keys, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
        jitter = np.random.default_rng(seed + 1).random(len(keys))
        return cls(np.asarray(origin, dtype=np.float64), float(size), int(depth), keys,
                   starts.astype(np.int64), counts.astype(np.int64), jitter)

    @property
    def n_rows(self) -> int:
        return int(self.counts.sum())

    def _boxes(self, keys: np.ndarray, depth: int) -> Tuple[np.ndarray, np.ndarray]:
        w = self.size / (1 << depth)
        lo = self.origin + demorton(keys) * w
        return lo, lo + w

    def visible_cells(self, planes: np.ndarray, with_partial: bool = False):
        """
        Indices of occupied leaf cells that intersect the frustum (coarse test first).
        With ``with_partial`` also a mask of the cells only partly inside.
        """
        cells, partial = self._visible(planes)
        return (cells, partial) if with_partial else cells

    def _visible(self, planes: np.ndarray):
        if len(self.keys) == 0:
            return np.empty(0, dtype=np.int64), np.zeros(0, dtype=bool)
        cd = min(COARSE_DEPTH, self.depth)
        shift = np.uint64(3 * (self.depth - cd))
        coarse = self.keys >> shift
        ckeys, cstart = np.unique(coarse, return_index=True)
        cend = np.append(cstart[1:], len(self.keys))
        state = classify_boxes(planes, *self._boxes(ckeys, cd))
        parts, flags = [], []
        inside = state == INSIDE
        if inside.any():
            r = _ranges(cstart[inside], cend[inside] - cstart[inside])
            parts.append(r)
            flags.append(np.zeros(len(r), dtype=bool))
        partial = state == PARTIAL
        if partial.any():
            cand = _ranges(cstart[partial], cend[partial] - cstart[partial])
            st = classify_boxes(planes, *self._boxes(self.keys[cand], self.depth))
            keep = st != OUTSIDE
            parts.append(cand[keep])
            flags.append(st[keep] == PARTIAL)
        if not parts:
            return np.empty(0, dtype=np.int64), np.zeros(0, dtype=bool)
        cells, part = np.concatenate(parts), np.concatenate(flags)
        order = np.argsort(cells, kind='stable')
        return cells[order], part[order]

    def take(self, cells: np.ndarray, fraction) -> np.ndarray:
        """
        Rows of the first count * fraction particles of each cell (nested in fraction).
        The count is rounded down after adding the cell's fixed jitter, so it equals
        count * fraction on average instead of always rounding the same way.
        ``fraction`` is one number or one per cell.
        """
        cnt = self.counts[cells]
        frac = np.minimum(np.asarray(fraction, dtype=np.float64), 1.0)
        n = np.minimum(np.floor(cnt * frac + self.jitter[cells]).astype(np.int64), cnt)
        return _ranges(self.starts[cells], n)

    def centres(self, cells: np.ndarray) -> np.ndarray:
        """Cell centres (cached for all occupied cells on first use)."""
        if getattr(self, '_centres', None) is None:
            w = self.size / (1 << self.depth)
            self._centres = self.origin + (demorton(self.keys) + 0.5) * w
        return self._centres[cells]


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenation of arange(s, s + l) for every (s, l), vectorised."""
    lengths = np.asarray(lengths, dtype=np.int64)
    keep = lengths > 0
    starts, lengths = np.asarray(starts, dtype=np.int64)[keep], lengths[keep]
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offs = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
    return offs + np.arange(total, dtype=np.int64)
