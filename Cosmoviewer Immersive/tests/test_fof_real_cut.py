"""
FoF on real Gadget-4 output, run on every test run (no environment variable needed).

Data: the tracked fixture _dev_tools/inspector_check/dm_cut, cut from the
Gadget-4 example DM-L50-N128 at z = 0:
  * snapshot_002.hdf5: the first 400,000 PartType1 rows of snapshot_002, with
    the full header and parameters (BoxSize 50 Mpc/h, MassTable, Omega0 0.308,
    OmegaBaryon 0.0482, no gas). Gadget-4 writes particles in FoF group order,
    so these rows are catalogue groups 1-29 complete, then the first 5,937 of
    group 30's 6,886 rows.
  * fof_tab_002.hdf5: Gadget-4's own FoF catalogue for the full snapshot
    (2,774 groups, b = 0.2, N_min = 32, DM links).

Catalogue group g (0-based) is rows GroupOffsetType[g, 1] .. + GroupLenType[g, 1].
FoF on the cut must reproduce groups 1-29 exactly, member by member; group 30 is
truncated, so only its leftover rows are checked loosely. Linking every particle
of the cut only to particles inside the cut cannot add members to a complete
group, and cannot remove any, because each complete group's links all lie within
its own rows.

The full-snapshot comparison (all 2,774 groups) is test_real_fof_vs_catalogue.py,
which needs COSMOVIEWER_G4_OUTPUT.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from snapshot_io.fof_snapshot import run_snapshot_fof  # noqa: E402

DM_CUT = ROOT.parent / "_dev_tools" / "inspector_check" / "dm_cut"
N_COMPLETE = 29
L_LINK = 0.078125      # 0.2 x 50 / 128: Gadget-4's mean separation with Omega0 (no gas in this run)


@pytest.fixture(scope="module")
def cut():
    if not (DM_CUT / "snapshot_002.hdf5").exists():
        pytest.fail("tracked test fixture _dev_tools/inspector_check/dm_cut is missing: "
                    "restore it with git checkout -- _dev_tools/inspector_check/dm_cut")
    with h5py.File(DM_CUT / "fof_tab_002.hdf5", "r") as f:
        off = f["Group/GroupOffsetType"][:, 1].astype(np.int64)
        ln = f["Group/GroupLenType"][:, 1].astype(np.int64)
    with h5py.File(DM_CUT / "snapshot_002.hdf5", "r") as f:
        ids = f["PartType1/ParticleIDs"][:]
        omega_b = float(f["Parameters"].attrs["OmegaBaryon"])
    fof = run_snapshot_fof(str(DM_CUT / "snapshot_002.hdf5"), b=0.2, min_group_size=32,
                           primary_types=(1,), convention="gadget4")
    return {"off": off, "len": ln, "ids": ids, "omega_b": omega_b, "fof": fof}


def test_fixture_is_what_this_file_says(cut):
    off, ln, ids = cut["off"], cut["len"], cut["ids"]
    assert len(ids) == 400_000 and len(ln) == 2_774
    np.testing.assert_array_equal(off[1:N_COMPLETE + 1], np.cumsum(ln[:N_COMPLETE]))   # contiguous, group order
    assert off[N_COMPLETE] + ln[N_COMPLETE] > len(ids) > off[N_COMPLETE]               # group 30 is truncated
    assert cut["omega_b"] > 0      # so the test below shows Omega0, not Omega0 - OmegaBaryon, sets the length


def test_linking_length_is_gadget4s(cut):
    assert cut["fof"].result.linking_length == pytest.approx(L_LINK, rel=1e-7)


def test_complete_groups_match_the_catalogue_member_by_member(cut):
    off, ln, ids, fof = cut["off"], cut["len"], cut["ids"], cut["fof"]
    gl = np.asarray(fof.result.group_len)
    np.testing.assert_array_equal(gl[:N_COMPLETE], ln[:N_COMPLETE])       # sizes, rank by rank
    ours = fof.lookup(ids.astype(fof.ids.dtype))                            # our group of each row
    seen = set()
    for g in range(N_COMPLETE):
        rows = ours[off[g]:off[g] + ln[g]]
        assert np.all(rows == rows[0]) and rows[0] >= 0, f"catalogue group {g + 1} is split or ungrouped"
        assert rows[0] not in seen, f"catalogue group {g + 1} merged with another"
        seen.add(int(rows[0]))
        assert np.count_nonzero(ours == rows[0]) == ln[g], f"catalogue group {g + 1} gained members"


def test_truncated_group_leaves_only_its_own_rows(cut):
    off, ids, fof = cut["off"], cut["ids"], cut["fof"]
    ours = fof.lookup(ids.astype(fof.ids.dtype))
    tail = ours[off[N_COMPLETE]:]
    assert fof.result.n_groups == N_COMPLETE + 1
    assert np.all(tail[tail >= 0] == N_COMPLETE)                 # the leftover rows form one group
    assert np.count_nonzero(ours < 0) == np.count_nonzero(tail < 0)   # nobody else is ungrouped
