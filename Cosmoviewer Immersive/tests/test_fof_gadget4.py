"""
End-to-end FoF check against real Gadget-4 output (HDF5 -> FoF -> membership).

Point COSMOVIEWER_G4_OUTPUT at a Gadget-4 output folder holding snapshot_NNN.hdf5
and fof_tab_NNN.hdf5 (or fof_subhalo_tab_NNN.hdf5) written with FOF on, e.g.
gadget4/examples/DM-L50-N128/output.  Skipped when the variable is unset.

Pass criterion: same number of groups, and every particle whose membership
differs is explained by a link within float32 rounding of the linking length.
Gadget-4 links on its internal integer positions; the snapshot stores float32
coordinates (spacing ~4e-6 at x=50), so a pair separated by l +/- 1 ulp can
legitimately go either way.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snapshot_io.fof import UNGROUPED  # noqa: E402
from snapshot_io.fof_snapshot import read_particles, run_snapshot_fof  # noqa: E402
from validate_gadget4 import reference_membership  # noqa: E402

OUT = os.environ.get('COSMOVIEWER_G4_OUTPUT')


def _pairs():
    if not OUT:
        return []
    found = []
    for snap in sorted(glob.glob(os.path.join(OUT, 'snapshot_*.hdf5'))):
        num = Path(snap).stem.split('_')[-1]
        for cat in (f'fof_tab_{num}.hdf5', f'fof_subhalo_tab_{num}.hdf5'):
            if os.path.exists(os.path.join(OUT, cat)):
                found.append((snap, os.path.join(OUT, cat)))
    return found


@pytest.mark.skipif(not OUT, reason="set COSMOVIEWER_G4_OUTPUT to a Gadget-4 output folder")
@pytest.mark.parametrize("snap,cat", _pairs())
def test_membership_matches_gadget4(snap, cat):
    prim = read_particles(snap, (1,))
    ref, ref_len, _, _, _ = reference_membership(cat, len(prim.ids))
    fof = run_snapshot_fof(snap)  # Gadget-4 defaults: b = 0.2, N_min = 32, DM links
    ours = fof.lookup(prim.ids)
    assert fof.result.n_groups == len(ref_len)
    assert fof.result.group_len.sum() == np.count_nonzero(ours >= 0)

    tol = 4 * float(np.spacing(np.float32(fof.boxsize or np.abs(prim.positions).max())))
    link = fof.result.linking_length
    box = fof.boxsize
    for g in np.unique(ref[(ref >= 0) & (ours != ref_major(ref, ours))]):
        mem = np.nonzero(ref == g)[0]
        main = np.bincount(ours[mem][ours[mem] >= 0]).argmax() if np.any(ours[mem] >= 0) else None
        inside = ours[mem] == main if main is not None else np.zeros(len(mem), bool)
        a, b = prim.positions[mem[inside]].astype(float), prim.positions[mem[~inside]].astype(float)
        d = a[:, None] - b[None]
        if box:
            d -= box * np.round(d / box)
        gap = np.sqrt((d ** 2).sum(-1)).min()
        assert link < gap <= link + tol, f"group {g}: unexplained split, gap {gap} vs l {link}"


def ref_major(ref, ours):
    """Map each particle's reference group to the group of ours holding most of it."""
    out = np.full(len(ref), UNGROUPED, dtype=np.int64)
    sel = (ref >= 0) & (ours >= 0)
    pairs, cnt = np.unique(np.stack([ref[sel], ours[sel]], 1), axis=0, return_counts=True)
    best = {}
    for (rg, og), c in zip(pairs, cnt):
        if c > best.get(rg, (0, -1))[1]:
            best[rg] = (og, c)
    lut = np.full(ref.max() + 1, UNGROUPED, dtype=np.int64)
    for rg, (og, _) in best.items():
        lut[rg] = og
    out[ref >= 0] = lut[ref[ref >= 0]]
    return out
