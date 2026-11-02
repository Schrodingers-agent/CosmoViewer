"""
Automatic Friends-of-Friends (FoF) Group Finding Engine for CosmoViewer.

Thin adapter between the scientific core (snapshot_io/fof.py,
snapshot_io/fof_snapshot.py) and the Immersive Mode UI, which numbers groups
from 1 with 0 meaning "not in a group".
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .fof import GADGET4_DEFAULT_B, GADGET4_DEFAULT_MIN_LEN, find_groups, mean_separation_from_box
from .fof_process import run_fof_process
from .fof_snapshot import SnapshotFoF, run_snapshot_fof


def _to_ui_numbering(group_index: np.ndarray, group_len: np.ndarray) -> Tuple[np.ndarray, int, Dict[int, int]]:
    ui = (np.asarray(group_index) + 1).astype(np.int32)  # UNGROUPED (-1) -> 0
    sizes = {g + 1: int(n) for g, n in enumerate(group_len)}
    return ui, len(group_len), sizes


class FoFGroupsEngine:
    """Interface to run FoF halo detection on simulation snapshots."""

    @staticmethod
    def detect_groups(positions: np.ndarray,
                      boxsize: Optional[float] = None,
                      b_linking: float = GADGET4_DEFAULT_B,
                      r_link: Optional[float] = None,
                      min_group_size: int = GADGET4_DEFAULT_MIN_LEN) -> Tuple[np.ndarray, int, Dict[int, int]]:
        """
        FoF on the given positions exactly as passed (no file access).
        Returns (group_ids, n_groups, group_sizes) with group_ids 1-based, 0 = none.

        b_linking is applied to (BoxSize^3 / N)^(1/3) of these positions, so it is
        only meaningful when ``positions`` is the complete primary particle set.
        """
        if r_link is None:
            if not boxsize:
                raise ValueError("no box size: pass an explicit linking length r_link")
            r_link = b_linking * mean_separation_from_box(len(positions), boxsize)
        res = find_groups(positions, r_link, boxsize=boxsize or None, min_group_size=min_group_size)
        return _to_ui_numbering(res.group_index, res.group_len)

    @staticmethod
    def detect_groups_in_snapshot(filepath: str,
                                  display_ids: np.ndarray,
                                  b_linking: float = GADGET4_DEFAULT_B,
                                  min_group_size: int = GADGET4_DEFAULT_MIN_LEN,
                                  r_link: Optional[float] = None,
                                  primary_types: Sequence[int] = (1,),
                                  separate_process: bool = False,
                                  progress: Optional[Callable[[float, str], None]] = None,
                                  cancelled: Optional[Callable[[], bool]] = None,
                                  ) -> Tuple[np.ndarray, int, Dict[int, int], SnapshotFoF]:
        """
        Full-resolution FoF on the snapshot file, mapped onto the displayed particles.

        Links ``primary_types`` (default PartType1, Gadget-4's default) read
        directly from the file, independent of how the view was decimated.
        group_sizes are the true member counts, not the number currently
        displayed.  Displayed particles of other types are reported as 0.
        With r_link=None the length is b times the mean separation: Gadget-4's
        mass-based convention for dark matter alone, (V/N)^(1/3) otherwise.
        separate_process runs it in its own process (snapshot_io/fof_process.py), so
        running out of memory stops only that process; cancelled() is polled there.
        """
        types = tuple(sorted(primary_types))
        convention = 'gadget4' if types == (1,) else 'box'
        kwargs = dict(b=b_linking, min_group_size=min_group_size, linking_length=r_link,
                      primary_types=list(types), convention=convention)
        if separate_process:
            fof = run_fof_process(filepath, kwargs, progress=progress, cancelled=cancelled)
        else:
            fof = run_snapshot_fof(filepath, progress=progress, **kwargs)
        ui, n, sizes = _to_ui_numbering(fof.lookup(display_ids), fof.result.group_len)
        return ui, n, sizes, fof
