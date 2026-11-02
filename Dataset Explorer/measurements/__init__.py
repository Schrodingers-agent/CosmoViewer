"""
measurements — the Explorer's headless measurement layer.

Pure Python: no Qt, no plotting. Every function here takes particle data and
returns a Result with astropy units and a record of what it was computed from,
so the same call gives the same numbers in the window, a script or a test.

    from measurements import SnapshotView, find_centre, lagrangian_radii
    snap = SnapshotView("snapshot_000.hdf5")
    p = snap.particles([1])
    c = find_centre(p, method="shrinking_sphere")
    r = lagrangian_radii(p, c)
"""

from .results import Provenance, Result
from .snapshot_view import ParticleSet, SnapshotView
from .centre import Centre, find_centre, minimum_image
from .lagrangian import DEFAULT_FRACTIONS, lagrangian_radii
from .runset import Run, RunSet, discover_snapshots

__all__ = [
    "Provenance", "Result", "SnapshotView", "ParticleSet",
    "Centre", "find_centre", "minimum_image",
    "DEFAULT_FRACTIONS", "lagrangian_radii",
    "Run", "RunSet", "discover_snapshots",
]
