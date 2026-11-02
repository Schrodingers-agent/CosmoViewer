"""
Box and sphere queries on an array of positions.

Despite the names, no octree is built; the name is aspirational.
cosmo_io_engine.OctreeSpatialIndex is a linear scan with a box mask, and the
fallback here does the same. query_radius
ignores the box size, so it does not apply the minimum image. Nothing in the
application calls this module at present; the FoF and selection code use
scipy's cKDTree or the chunked file scans in snapshot_io.
"""

import sys
from pathlib import Path
import numpy as np

from pygadgetviewer_path import ensure_on_path

ensure_on_path()
try:
    from cosmo_io_engine import OctreeSpatialIndex
    HAS_OCTREE = True
except ImportError:
    try:
        from pygadgetviewer import OctreeSpatialIndex
        HAS_OCTREE = True
    except ImportError:
        HAS_OCTREE = False


class OctreeSpatialFilter:
    """Box and radius queries over ``positions`` (see the module docstring)."""

    def __init__(self, positions: np.ndarray, boxsize: float = 100.0):
        self.positions = positions
        self.boxsize = boxsize
        if HAS_OCTREE:
            self.index = OctreeSpatialIndex(positions, boxsize)
        else:
            self.index = None

    def query_box(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> np.ndarray:
        if self.index:
            return self.index.query_box(xmin, xmax, ymin, ymax, zmin, zmax)
        mask = (self.positions[:, 0] >= xmin) & (self.positions[:, 0] <= xmax) & \
               (self.positions[:, 1] >= ymin) & (self.positions[:, 1] <= ymax) & \
               (self.positions[:, 2] >= zmin) & (self.positions[:, 2] <= zmax)
        return np.where(mask)[0]

    def query_radius(self, center: np.ndarray, radius: float) -> np.ndarray:
        dist_sq = np.sum((self.positions - center) ** 2, axis=1)
        return np.where(dist_sq <= radius ** 2)[0]
