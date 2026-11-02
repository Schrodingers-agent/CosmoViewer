"""
Compare CosmoViewer FoF with Gadget-4's own FoF catalogue for one snapshot.

    python tests/validate_gadget4.py <snapshot_XXX.hdf5> <fof_tab_XXX.hdf5>

Gadget-4 (with FOF on) writes snapshots in group order: group g occupies
snapshot indices GroupOffsetType[g, t] .. + GroupLenType[g, t] for each type t.
That gives the reference membership of every particle without needing IDs in
the catalogue.  The script prints a JSON summary of the comparison.
"""

import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snapshot_io.fof import UNGROUPED, group_centres_of_mass  # noqa: E402
from snapshot_io.fof_snapshot import read_particles, run_snapshot_fof  # noqa: E402


def reference_membership(cat_path, n_part, ptype=1):
    with h5py.File(cat_path, 'r') as f:
        off = f['Group/GroupOffsetType'][:, ptype].astype(np.int64)
        ln = f['Group/GroupLenType'][:, ptype].astype(np.int64)
        gpos = f['Group/GroupPos'][:]
        gmass = f['Group/GroupMass'][:]
        hdr = dict(f['Header'].attrs)
    ref = np.full(n_part, UNGROUPED, dtype=np.int64)
    for g, (o, l) in enumerate(zip(off, ln)):
        ref[o:o + l] = g
    return ref, ln, gpos, gmass, hdr


def compare(snap_path, cat_path, b=0.2, min_len=32):
    t0 = time.perf_counter()
    prim = read_particles(snap_path, (1,))
    t_read = time.perf_counter() - t0
    ref, ref_len, ref_pos, ref_mass, cat_hdr = reference_membership(cat_path, len(prim.ids))

    t0 = time.perf_counter()
    fof = run_snapshot_fof(snap_path, b=b, min_group_size=min_len)
    t_fof = time.perf_counter() - t0
    ours = fof.lookup(prim.ids)  # aligned with snapshot order

    # Group-by-group: for each reference group, which of our groups holds its particles?
    n_ref = len(ref_len)
    exact = 0
    # vectorised: a reference group matches exactly when all its members share one
    # of our groups and that group has exactly the same length
    grouped = ref >= 0
    pairs = np.stack([ref[grouped], ours[grouped]], 1)
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
    our_len = np.bincount(ours[ours >= 0], minlength=fof.result.n_groups)
    for (rg, og), c in zip(uniq, cnt):
        if og >= 0 and c == ref_len[rg] and our_len[og] == ref_len[rg]:
            exact += 1
    n_exact_particles = sum(ref_len[rg] for (rg, og), c in zip(uniq, cnt)
                            if og >= 0 and c == ref_len[rg] and our_len[og] == ref_len[rg])
    disagree = int(np.count_nonzero((ref >= 0) != (ours >= 0)))

    # Derived quantity: centre of mass of our groups vs Gadget GroupPos (match by largest overlap)
    cm = group_centres_of_mass(prim.positions, ours, fof.result.n_groups, prim.masses, fof.boxsize)
    best = {}
    for (rg, og), c in zip(uniq, cnt):
        if og >= 0 and c > best.get(rg, (None, 0))[1]:
            best[rg] = (og, c)
    d = np.array([cm[best[g][0]] - ref_pos[g] for g in range(n_ref) if g in best])
    d -= fof.boxsize * np.round(d / fof.boxsize)
    sep = np.sqrt((d ** 2).sum(1))
    our_mass = np.bincount(ours[ours >= 0], weights=prim.masses[ours >= 0], minlength=fof.result.n_groups)
    mrel = np.array([our_mass[best[g][0]] / ref_mass[g] - 1 for g in range(n_ref) if g in best])

    return {
        'snapshot': Path(snap_path).name,
        'redshift': float(cat_hdr['Redshift']),
        'n_primary': int(len(prim.ids)),
        'linking_length': fof.result.linking_length,
        'gadget_groups': int(n_ref),
        'our_groups': int(fof.result.n_groups),
        'groups_identical_membership': int(exact),
        'particles_in_identical_groups': int(n_exact_particles),
        'gadget_grouped_particles': int(ref_len.sum()),
        'our_grouped_particles': int(np.count_nonzero(ours >= 0)),
        'particles_grouped_in_one_but_not_other': disagree,
        'largest_5_gadget': ref_len[:5].tolist(),
        'largest_5_ours': fof.result.group_len[:5].tolist(),
        'centre_offset_max': float(sep.max()),
        'centre_offset_median': float(np.median(sep)),
        'mass_rel_diff_max': float(np.abs(mrel).max()),
        'ids_unique': bool(len(np.unique(prim.ids)) == len(prim.ids)),
        'read_seconds': round(t_read, 2),
        'fof_seconds_incl_read': round(t_fof, 2),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


if __name__ == '__main__':
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))
