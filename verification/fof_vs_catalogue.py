"""
Compare CosmoViewer's FoF (the code Immersive Mode runs) with Gadget-4's own
FoF catalogue, exactly: number of groups, sizes, and membership by ParticleID.

    python3 verification/fof_vs_catalogue.py <snapshot_NNN.hdf5> <fof_tab_NNN.hdf5> [--b 0.2] [--nmin 32]

Gadget-4 writes snapshots in group order, so catalogue group g (0-based) is
PartType1 rows GroupOffsetType[g, 1] .. + GroupLenType[g, 1].

Also reports the linking length under both readings of Gadget-4's mean
separation (rho_link from Omega0, or from Omega0 - OmegaBaryon), because a
DM-only run can still carry OmegaBaryon > 0 in its parameters.
Prints one JSON object. Reads only; writes nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

IMM = Path(__file__).resolve().parents[1] / "Cosmoviewer Immersive"
sys.path.insert(0, str(IMM))
from snapshot_io.fof import mean_separation_gadget4  # noqa: E402
from snapshot_io.fof_snapshot import read_particles, run_snapshot_fof  # noqa: E402


def catalogue(cat_path):
    with h5py.File(cat_path, "r") as f:
        off = f["Group/GroupOffsetType"][:, 1].astype(np.int64)
        ln = f["Group/GroupLenType"][:, 1].astype(np.int64)
        glen = f["Group/GroupLen"][:].astype(np.int64)
        hdr = dict(f["Header"].attrs)
    return off, ln, glen, hdr


def members_by_catalogue(snap_ids, off, ln, g):
    return np.unique(snap_ids[off[g]:off[g] + ln[g]].astype(np.uint64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("catalogue")
    ap.add_argument("--b", type=float, default=0.2)
    ap.add_argument("--nmin", type=int, default=32)
    a = ap.parse_args()
    out = {"snapshot": a.snapshot, "catalogue": a.catalogue, "b": a.b, "min_len": a.nmin}

    prim = read_particles(a.snapshot, (1,))
    snap_ids = prim.ids
    p = prim.parameters
    omega_b_reading = {}
    for label, has_gas in (("Omega0", False), ("Omega0-OmegaBaryon", True)):
        d = mean_separation_gadget4(prim.masses, float(p["Omega0"]), float(p.get("OmegaBaryon", 0.0)), has_gas,
                                    float(p["UnitLength_in_cm"]), float(p["UnitMass_in_g"]),
                                    float(p["UnitVelocity_in_cm_per_s"]))
        omega_b_reading[label] = a.b * d
    out["linking_length_if_rho_from"] = omega_b_reading
    out["box_over_N_cuberoot_times_b"] = a.b * float(prim.header["BoxSize"]) / round(len(snap_ids) ** (1 / 3))
    out["OmegaBaryon_in_parameters"] = float(p.get("OmegaBaryon", 0.0))
    out["n_gas_in_header"] = int(np.atleast_1d(prim.header["NumPart_Total"])[0])

    off, ln, glen, hdr = catalogue(a.catalogue)
    out["catalogue_groups"] = int(len(ln))
    out["catalogue_Ngroups_Total"] = int(hdr.get("Ngroups_Total", -1))
    out["catalogue_Nids_Total"] = int(hdr.get("Nids_Total", -1))
    out["catalogue_GroupLen_equals_GroupLenType1"] = bool(np.array_equal(glen, ln))
    out["catalogue_sizes_descending"] = bool(np.all(np.diff(ln) <= 0))
    out["catalogue_offsets_contiguous"] = bool(np.array_equal(off[1:], off[:-1] + ln[:-1]) and off[0] == 0)

    t0 = time.perf_counter()
    fof = run_snapshot_fof(a.snapshot, b=a.b, min_group_size=a.nmin, primary_types=(1,), convention="gadget4")
    out["fof_seconds"] = round(time.perf_counter() - t0, 2)
    out["fof_linking_length"] = fof.result.linking_length
    out["fof_convention"] = fof.convention
    gl = np.asarray(fof.result.group_len)
    out["fof_groups"] = int(len(gl))
    out["fof_particles_in_groups"] = int(gl.sum())
    out["catalogue_particles_in_groups"] = int(ln.sum())
    out["largest20_fof"] = gl[:20].tolist()
    out["largest20_catalogue"] = ln[:20].tolist()
    n = min(len(gl), len(ln))
    out["sizes_identical_all_groups"] = bool(len(gl) == len(ln) and np.array_equal(gl, ln))
    out["n_groups_with_different_size_rank_by_rank"] = int(np.count_nonzero(gl[:n] != ln[:n]) + abs(len(gl) - len(ln)))

    # membership of every group, by ParticleID
    ids_sorted, gidx_sorted = fof.ids, fof.group_index          # ids ascending (sorted_by_id)
    ours_for_snap_rows = fof.lookup(snap_ids)                  # our group of each snapshot row
    ref = np.full(len(snap_ids), -1, dtype=np.int64)
    for g in range(len(ln)):
        ref[off[g]:off[g] + ln[g]] = g
    out["particles_with_same_group_number"] = int(np.count_nonzero(ref == ours_for_snap_rows))
    out["particles_with_different_group_number"] = int(np.count_nonzero(ref != ours_for_snap_rows))
    g1_cat = members_by_catalogue(snap_ids, off, ln, 0)
    g1_ours = np.unique(ids_sorted[gidx_sorted == 0].astype(np.uint64))
    out["group1_catalogue_n"] = int(len(g1_cat))
    out["group1_fof_n"] = int(len(g1_ours))
    out["group1_membership_identical"] = bool(np.array_equal(g1_cat, g1_ours))
    out["group1_only_in_catalogue"] = int(len(np.setdiff1d(g1_cat, g1_ours)))
    out["group1_only_in_fof"] = int(len(np.setdiff1d(g1_ours, g1_cat)))
    # groups whose ID sets differ (compare as sets, group by group, by rank)
    differ = []
    for g in range(n):
        a_ = members_by_catalogue(snap_ids, off, ln, g)
        b_ = np.unique(ids_sorted[gidx_sorted == g].astype(np.uint64)) if g < 50 else None
        if b_ is not None and not np.array_equal(a_, b_):
            differ.append(g + 1)
    out["first50_groups_with_different_membership_1based"] = differ
    out["duplicate_ids_in_snapshot"] = int(len(snap_ids) - len(np.unique(snap_ids)))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
