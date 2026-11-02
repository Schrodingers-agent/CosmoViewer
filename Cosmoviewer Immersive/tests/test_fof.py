"""
Tests for the abstract FoF algorithm (snapshot_io/fof.py).

These do not touch HDF5 or Qt; see test_fof_gadget4.py for real-output checks.
Run from the "Cosmoviewer Immersive" folder:  python -m pytest tests -q
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snapshot_io.fof import (UNGROUPED, attach_secondaries, find_groups, group_centres_of_mass,
                             mean_separation_from_box, mean_separation_gadget4)


# ---------------------------------------------------------------------------
# Independent reference: O(N^2) distances + breadth-first search.
# ---------------------------------------------------------------------------
def brute_force_groups(pos, link, boxsize=None, min_size=1):
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    d = pos[:, None, :] - pos[None, :, :]
    if boxsize:
        d -= boxsize * np.round(d / boxsize)
    adj = (d ** 2).sum(-1) <= link ** 2
    seen = np.zeros(n, bool)
    groups = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for m in np.nonzero(adj[k] & ~seen)[0]:
                seen[m] = True
                stack.append(m)
        if len(comp) >= min_size:
            groups.append(frozenset(comp))
    return set(groups)


def as_sets(result):
    out = {}
    for i, g in enumerate(result.group_index):
        if g != UNGROUPED:
            out.setdefault(g, set()).add(i)
    return {frozenset(s) for s in out.values()}


# ---------------------------------------------------------------------------
# Analytic configurations
# ---------------------------------------------------------------------------
def test_empty_input():
    r = find_groups(np.zeros((0, 3)), 1.0, boxsize=10.0, min_group_size=1)
    assert r.n_groups == 0 and len(r.group_index) == 0


def test_single_particle():
    r = find_groups([[1.0, 1.0, 1.0]], 0.5, min_group_size=1)
    assert r.n_groups == 1 and list(r.group_len) == [1]
    assert find_groups([[1.0, 1.0, 1.0]], 0.5, min_group_size=2).n_groups == 0


def test_isolated_particles_do_not_link():
    pos = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], float)
    r = find_groups(pos, 1.0, min_group_size=2)
    assert r.n_groups == 0
    assert np.all(r.group_index == UNGROUPED)


def test_simple_group():
    pos = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [5, 5, 5]], float)
    r = find_groups(pos, 0.6, min_group_size=2)
    assert as_sets(r) == {frozenset({0, 1, 2})}
    assert r.group_index[3] == UNGROUPED


def test_transitive_chain():
    # Chain spacing 0.9 < l = 1 but end-to-end length 9 >> l: one group.
    pos = np.zeros((11, 3))
    pos[:, 0] = 0.9 * np.arange(11)
    r = find_groups(pos, 1.0, min_group_size=2)
    assert r.n_groups == 1 and r.group_len[0] == 11


def test_dense_clumps_bridged_by_single_particle():
    # two dense clumps whose only connection is one particle in between:
    # exercises the grid linker's cell pruning (most cell pairs pre-joined)
    rng = np.random.default_rng(5)
    a = rng.normal(scale=0.02, size=(3000, 3)) + [4.0, 5, 5]
    b = rng.normal(scale=0.02, size=(3000, 3)) + [6.0, 5, 5]
    chain = np.zeros((19, 3)) + [5, 5, 5]
    chain[:, 0] = np.linspace(4.1, 5.9, 19)
    for method in ("grid", "kdtree"):
        r = find_groups(np.vstack([a, b, chain]), 0.11, boxsize=10.0, min_group_size=2, method=method)
        assert r.n_groups == 1 and r.group_len[0] == 6019
        r = find_groups(np.vstack([a, b, chain[:9], chain[10:]]), 0.09, boxsize=10.0, min_group_size=2, method=method)
        assert r.n_groups == 2


@pytest.mark.parametrize("box", [1.0, 3.0])
def test_small_box_falls_back_and_agrees(box):
    rng = np.random.default_rng(2)
    pos = rng.random((300, 3)) * box
    link = 0.3 * box
    ref = brute_force_groups(pos, link, box, 1)
    assert as_sets(find_groups(pos, link, boxsize=box, min_group_size=1)) == ref


def test_chain_order_does_not_matter():
    pos = np.zeros((50, 3))
    pos[:, 0] = 0.9 * np.arange(50)
    perm = np.random.default_rng(1).permutation(50)
    r = find_groups(pos[perm], 1.0, min_group_size=2, chunk_size=7)
    assert r.n_groups == 1 and r.group_len[0] == 50


def test_multiple_disconnected_groups_are_numbered_by_size():
    a = np.zeros((3, 3)); a[:, 0] = [0, 0.5, 1.0]
    b = np.zeros((5, 3)); b[:, 0] = [10, 10.5, 11, 11.5, 12]
    c = np.zeros((2, 3)); c[:, 0] = [20, 20.5]
    r = find_groups(np.vstack([a, b, c]), 0.6, min_group_size=2)
    assert list(r.group_len) == [5, 3, 2]
    assert np.all(r.group_index[3:8] == 0) and np.all(r.group_index[0:3] == 1)
    assert np.all(r.group_index[8:10] == 2)


def test_min_group_size_threshold():
    a = np.zeros((4, 3)); a[:, 0] = 0.5 * np.arange(4)
    b = np.zeros((3, 3)); b[:, 0] = 10 + 0.5 * np.arange(3)
    r = find_groups(np.vstack([a, b]), 0.6, min_group_size=4)
    assert list(r.group_len) == [4]
    assert np.all(r.group_index[4:] == UNGROUPED)


@pytest.mark.parametrize("sep,linked", [(1.0 - 1e-9, True), (1.0, True), (1.0 + 1e-9, False)])
def test_linking_threshold_is_inclusive(sep, linked):
    pos = np.array([[0, 0, 0], [sep, 0, 0]], float)
    r = find_groups(pos, 1.0, min_group_size=2)
    assert (r.n_groups == 1) is linked


def test_periodic_link_across_boundary():
    pos = np.array([[0.1, 5, 5], [9.95, 5, 5]], float)  # separation 0.15 through the face
    assert find_groups(pos, 0.2, boxsize=10.0, min_group_size=2).n_groups == 1
    assert find_groups(pos, 0.2, boxsize=None, min_group_size=2).n_groups == 0


def test_periodic_corner_and_out_of_range_coordinates():
    # eight particles around the box corner, some given as -x or as x + L
    eps = 0.05
    pts = np.array([[sx * eps, sy * eps, sz * eps] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pts[0] += 10.0  # same physical point as pts[0] - 10
    r = find_groups(pts, 0.11, boxsize=10.0, min_group_size=2)
    assert r.n_groups == 1 and r.group_len[0] == 8


def test_coordinate_exactly_at_boxsize_is_accepted():
    pos = np.array([[10.0, 5, 5], [0.05, 5, 5]], dtype=np.float32)
    r = find_groups(pos, 0.1, boxsize=10.0, min_group_size=2)
    assert r.n_groups == 1


def test_duplicate_positions_link():
    pos = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], float)
    r = find_groups(pos, 1e-6, min_group_size=3)
    assert r.n_groups == 1 and r.group_len[0] == 3


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        find_groups(np.zeros((2, 3)), 0.0)
    with pytest.raises(ValueError):
        find_groups(np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError):
        find_groups(np.array([[np.nan, 0, 0]]), 1.0)
    with pytest.raises(ValueError):
        find_groups(np.zeros((2, 3)), 6.0, boxsize=10.0)


def test_every_particle_counted_once():
    rng = np.random.default_rng(3)
    pos = rng.random((5000, 3)) * 10
    r = find_groups(pos, 0.2 * mean_separation_from_box(5000, 10.0) * 3, boxsize=10.0, min_group_size=5)
    assert r.group_len.sum() == np.count_nonzero(r.group_index >= 0)
    assert np.array_equal(np.bincount(r.group_index[r.group_index >= 0], minlength=r.n_groups), r.group_len)
    assert np.all(np.diff(r.group_len) <= 0)


# ---------------------------------------------------------------------------
# Randomised comparison against the brute-force reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("method", ["grid", "kdtree"])
def test_matches_brute_force_on_random_clustered_sets(seed, periodic, method):
    rng = np.random.default_rng(seed)
    box = 10.0
    centres = rng.random((6, 3)) * box
    if periodic:
        centres[0] = [0.05, 0.05, 9.95]  # force a cluster across the corner
    pts = [c + rng.normal(scale=0.3, size=(rng.integers(5, 60), 3)) for c in centres]
    pts.append(rng.random((150, 3)) * box)
    pos = np.vstack(pts)
    boxsize = box if periodic else None
    if periodic:
        pos = np.mod(pos, box)
    link = 0.35
    for min_size in (1, 2, 8):
        ref = brute_force_groups(pos, link, boxsize, min_size)
        got = find_groups(pos, link, boxsize=boxsize, min_group_size=min_size, chunk_size=97, method=method)
        assert as_sets(got) == ref


# ---------------------------------------------------------------------------
# Linking-length conventions
# ---------------------------------------------------------------------------
def test_mean_separation_box():
    assert mean_separation_from_box(128 ** 3, 50.0) == pytest.approx(50.0 / 128)


def test_gadget4_mean_separation_equals_box_value_for_dm_only_box():
    # DM-L50-N128 example: equal-mass DM filling Omega0 * rho_crit in a 50 Mpc/h box
    box, n, omega0 = 50.0, 128 ** 3, 0.308
    rho_crit = 27.7536627  # 1e10 Msun/h per (Mpc/h)^3
    m = omega0 * rho_crit * box ** 3 / n
    d = mean_separation_gadget4(np.full(10, m), omega0, 0.0482, has_gas=False,
                                unit_length_cm=3.085678e24, unit_mass_g=1.989e43, unit_velocity_cms=1e5)
    assert d == pytest.approx(box / 128, rel=1e-3)


# ---------------------------------------------------------------------------
# Secondary attachment and derived quantities
# ---------------------------------------------------------------------------
def test_attach_secondaries_to_nearest_primary():
    prim = np.array([[0, 0, 0], [0.1, 0, 0], [5, 5, 5], [5.1, 5, 5]], float)
    gidx = np.array([0, 0, 1, 1])
    sec = np.array([[0.05, 0.01, 0], [5.2, 5, 5], [9, 9, 9]], float)
    out = attach_secondaries(prim, gidx, sec, linking_length=0.2)
    assert list(out) == [0, 1, UNGROUPED]


def test_centre_of_mass_across_periodic_boundary():
    pos = np.array([[9.9, 5, 5], [0.1, 5, 5]])
    cm = group_centres_of_mass(pos, np.array([0, 0]), 1, boxsize=10.0)
    assert np.allclose(cm[0], [0.0, 5, 5]) or np.allclose(cm[0], [10.0, 5, 5])


# ---------------------------------------------------------------------------
# Performance (recorded, generous bound so it does not flake)
# ---------------------------------------------------------------------------
def test_performance_one_million_uniform_particles():
    rng = np.random.default_rng(0)
    n, box = 1_000_000, 100.0
    pos = rng.random((n, 3)) * box
    t0 = time.perf_counter()
    r = find_groups(pos, 0.2 * mean_separation_from_box(n, box), boxsize=box, min_group_size=2)
    dt = time.perf_counter() - t0
    print(f"\n1e6 uniform particles: {dt:.2f} s, {r.n_groups} groups")
    assert dt < 120                       # a guard against gross slowdowns, not a benchmark
    # the result must still be a valid partition: groups numbered by size, counts consistent
    assert np.array_equal(np.bincount(r.group_index[r.group_index >= 0], minlength=r.n_groups), r.group_len)
    assert np.all(np.diff(r.group_len) <= 0) and (r.n_groups == 0 or r.group_len[-1] >= 2)
    # agrees with the independent pair-enumerating linker on the same data
    sub, l_sub = pos[:200_000], 0.2 * mean_separation_from_box(200_000, box)
    ref = find_groups(sub, l_sub, boxsize=box, min_group_size=2, method="kdtree")
    got = find_groups(sub, l_sub, boxsize=box, min_group_size=2)
    assert got.n_groups > 1000 and as_sets(got) == as_sets(ref)
