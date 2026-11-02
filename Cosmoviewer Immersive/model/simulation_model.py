"""
The Immersive model: loaded snapshots, the file manifest, units, per-type
visibility and the six particle selection sets.

The selection sets are GadgetViewer's: lists of ParticleIDs that stay selected
across snapshots. Finding them in each snapshot file is done elsewhere
(snapshot_io.id_lookup, driven by controller/tracking_controller.py); a set's
``version`` lets those lookups be cached until the set changes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

from gadget_file_discovery import FileManifest, ParticleTypeInfo, SnapshotInspector
from snapshot_io.snapshot_loader import AsyncSnapshotLoader
from model.units import GadgetUnits
import selection_sets


class LayerState:
    """Tracks per-type visibility state."""

    def __init__(self):
        self._visibility: Dict[int, bool] = {i: True for i in range(6)}

    def set_visible(self, type_id: int, visible: bool):
        self._visibility[type_id] = visible

    def is_visible(self, type_id: int) -> bool:
        return self._visibility.get(type_id, True)

    def get_drawable_types(self, existing_types: List[int]) -> List[int]:
        return [t for t in existing_types if self.is_visible(t)]


class ParticleSelectionSet:
    """One of up to 6 particle selection sets: a list of ParticleIDs plus what it means.

    The IDs are what tracking, plots, files and scripts use. kind, definition,
    defined_at, complete and n_expected record how the set was made (a
    "population", see selection_sets.py), so its meaning survives moving between
    views and snapshots: a FoF group stays "group g of snapshot k", a set drawn from
    the display sample says it may be incomplete. Changing the membership by hand
    (add, subset) turns it into plain IDs that remember what they were edited from.
    """

    DEFAULT_COLORS = [
        (255, 0, 0, 255),       # Set 1: Red
        (0, 255, 0, 255),       # Set 2: Green
        (255, 255, 0, 255),     # Set 3: Yellow
        (255, 0, 255, 255),     # Set 4: Magenta
        (0, 255, 255, 255),     # Set 5: Cyan
        (255, 128, 0, 255),     # Set 6: Orange
    ]

    def __init__(self, set_id: int, name: str = ""):
        self.set_id = set_id
        self.name = name or f"Selection Set {set_id + 1}"
        self.color = self.DEFAULT_COLORS[set_id % 6]
        self.selected_ids: Set[int] = set()
        self.enabled = True
        # Bumped on every change, so per-snapshot lookups of these IDs can be cached.
        self.version = 0
        self._reset_meaning()

    # ------------------------------------------------------------------ meaning of the set

    def _reset_meaning(self, kind: str = 'ids', definition: Optional[dict] = None,
                       defined_at: Optional[dict] = None, complete: Optional[bool] = None,
                       n_expected: Optional[int] = None, incomplete_reason: Optional[str] = None):
        if kind not in selection_sets.KINDS:
            raise ValueError(f"unknown set kind {kind!r}")
        reason = selection_sets.check_incomplete_reason(complete, incomplete_reason)
        self.kind = kind
        self.definition = dict(definition or {})
        self.defined_at = dict(defined_at or {})
        self.complete = complete
        self.n_expected = None if n_expected is None else int(n_expected)
        self.incomplete_reason = reason

    def set_population(self, particle_ids: np.ndarray, kind: str = 'ids', definition: Optional[dict] = None,
                       defined_at: Optional[dict] = None, complete: Optional[bool] = None,
                       n_expected: Optional[int] = None, incomplete_reason: Optional[str] = None):
        """Replace the set with these particles and record what they are (see selection_sets.py).

        incomplete_reason ("display_sample" or "missing_in_file") says why a set with
        complete=False is incomplete; it is refused for a set that is not incomplete."""
        self._reset_meaning(kind, definition, defined_at, complete, n_expected, incomplete_reason)
        self.selected_ids = set(int(i) for i in np.asarray(particle_ids).ravel())
        self.version += 1

    def describe(self) -> str:
        """One line for the interface, e.g. "FoF group 1 (...) at snapshot_002.hdf5 · complete · 34,120 particles"."""
        return selection_sets.describe(self.kind, self.definition, self.defined_at, self.complete,
                                       len(self.selected_ids), self.n_expected, self.incomplete_reason)

    def completeness_warning(self) -> Optional[str]:
        return selection_sets.completeness_warning(self.kind, self.complete, len(self.selected_ids),
                                                   self.n_expected, self.incomplete_reason)

    def meaning(self) -> dict:
        """What the set means, as keyword arguments of set_population (sessions, exports)."""
        return {'kind': self.kind, 'definition': dict(self.definition), 'defined_at': dict(self.defined_at),
                'complete': self.complete, 'n_expected': self.n_expected,
                'incomplete_reason': self.incomplete_reason}

    def as_selection_set(self, colour: str = '#ffffff') -> 'selection_sets.SelectionSet':
        """The set as the shared (file / Explorer) object, without writing anything."""
        return selection_sets.SelectionSet(self.id_array(), self.name, colour, made_in="CosmoViewer Immersive",
                                           source_snapshot=self.defined_at.get('file', ''),
                                           description=self.describe(), **self.meaning())

    def _edited(self, operation: str):
        """Membership changed by hand: the set is now plain IDs that remember what they came from.
        A set known to be incomplete stays incomplete, for the same reason: adding or removing
        particles by hand does not bring back the members it lacks."""
        if self.kind != 'ids' or self.definition:
            before = self.describe()
            incomplete = self.complete is False
            reason = self.incomplete_reason if incomplete else None
            self._reset_meaning('ids', {'edited_from': before, 'operation': operation},
                                complete=False if incomplete else None, incomplete_reason=reason)

    # ------------------------------------------------------------------ membership

    def select_by_ids(self, particle_ids: np.ndarray):
        """Replace the set with these particles, as plain IDs (use set_population to say what they are)."""
        self.set_population(particle_ids)

    def add_ids(self, particle_ids: np.ndarray):
        self._edited('add')
        self.selected_ids.update(int(i) for i in np.asarray(particle_ids).ravel())
        self.version += 1

    def intersect_ids(self, particle_ids: np.ndarray):
        self._edited('subset')
        self.selected_ids.intersection_update(int(i) for i in np.asarray(particle_ids).ravel())
        self.version += 1

    def select_by_radius(self, coords: np.ndarray, ids: np.ndarray, center: np.ndarray, radius: float):
        dist = np.linalg.norm(coords - center, axis=1)
        mask = dist <= radius
        self.select_by_ids(ids[mask])

    def id_array(self) -> np.ndarray:
        """Selected IDs, sorted, as uint64."""
        return np.fromiter(self.selected_ids, dtype=np.uint64, count=len(self.selected_ids)) \
            if self.selected_ids else np.empty(0, dtype=np.uint64)

    def clear(self):
        self.selected_ids.clear()
        self._reset_meaning()
        self.version += 1


class ParticleTracker:
    """The selection sets (six by default, as in GadgetViewer)."""

    def __init__(self, max_sets: int = 6):
        self.sets: List[ParticleSelectionSet] = [
            ParticleSelectionSet(i) for i in range(max_sets)
        ]

    def get_set(self, set_id: int) -> ParticleSelectionSet:
        return self.sets[set_id % len(self.sets)]

    def get_selection_mask(self, current_ids: np.ndarray, set_id: int) -> np.ndarray:
        s_set = self.get_set(set_id)
        if not s_set.enabled or not s_set.selected_ids:
            return np.zeros(len(current_ids), dtype=bool)
        return np.isin(current_ids, list(s_set.selected_ids))


class SimulationModel:
    """
    Model: owns simulation snapshots, manifest, physical units, and particle selection tracker.
    """

    def __init__(self):
        self.snapshots: List[dict] = []
        self.manifest: Optional[FileManifest] = None
        self.particle_type_cache: Dict[str, List[ParticleTypeInfo]] = {}
        self.g_min: Optional[np.ndarray] = None
        self.g_max: Optional[np.ndarray] = None
        self.units: Optional[GadgetUnits] = None
        self.tracker: ParticleTracker = ParticleTracker()

        self._loader_thread: Optional[AsyncSnapshotLoader] = None

    def clear_all_data(self):
        """Full data reset."""
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.cancel()    # not waited for: it is held until it ends (controller/thread_hold.py)
        self._loader_thread = None

        self.snapshots = []
        self.manifest = None
        self.particle_type_cache = {}
        self.g_min = None
        self.g_max = None
        self.units = None
        for s_set in self.tracker.sets:
            s_set.clear()

    def get_snapshot(self, idx: int) -> Optional[dict]:
        if 0 <= idx < len(self.snapshots):
            return self.snapshots[idx]
        return None

    def get_particle_types_at_index(self, idx: int) -> List[ParticleTypeInfo]:
        snap = self.get_snapshot(idx)
        if snap is None:
            return []
        return self.particle_type_cache.get(snap['filepath'], [])
