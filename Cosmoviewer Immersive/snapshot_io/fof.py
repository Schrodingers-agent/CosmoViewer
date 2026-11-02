"""
Friends-of-Friends (FoF) group finder: the scientific core used by Immersive Mode.

Definition (Davis et al. 1985; the convention Gadget-4 uses):
  * Two primary particles are "friends" if their separation is <= the linking
    length ``l_link``.  Separations use the minimum-image convention when the
    box is periodic.
  * A group is a connected component of the friendship graph, so linking is
    transitive: A-B and B-C puts A, B and C in one group whatever |A-C| is.
  * Groups with fewer than ``min_group_size`` members are discarded.
  * ``l_link = b * <d>``, with ``b`` the dimensionless linking parameter
    (Gadget-4 FOF_LINKLENGTH, default 0.2) and ``<d>`` the mean interparticle
    separation of the primary (linking) particle type(s).

Gadget-4 conventions reproduced here (src/fof/fof.cc):
  * Only primary link types take part in linking (FOF_PRIMARY_LINK_TYPES,
    a bit mask; the default 2 = 1 << 1 selects PartType1 only).
  * ``<d> = (m_mean / rho_link)**(1/3)`` where ``m_mean`` is the mean mass of the
    primary particles and ``rho_link = Omega_dm * rho_crit`` when the run has gas
    (Omega_dm = Omega0 - OmegaBaryon) and ``Omega0 * rho_crit`` otherwise.  For an
    equal-mass dark-matter-only box this equals ``(BoxSize**3 / N)**(1/3)``.
  * FOF_GROUP_MIN_LEN defaults to 32.
  * Secondary types (FOF_SECONDARY_LINK_TYPES, default none) never link; each
    is attached to the group of its nearest primary particle, searching out to
    a few linking lengths (see ``attach_secondaries``).
  * Groups are numbered from 0 by decreasing length.

This module has no Qt dependency so it can be tested and reused directly.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GADGET4_DEFAULT_B = 0.2
GADGET4_DEFAULT_MIN_LEN = 32

# Physical constants as used by Gadget-4 (src/data/constants.h)
GRAVITY_CGS = 6.6738e-8
HUBBLE_CGS = 3.2407789e-18  # 100 km/s/Mpc in 1/s (i.e. h = 1)

UNGROUPED = -1

# progress(fraction in [0, 1], short description); optional everywhere
Progress = Optional[Callable[[float, str], None]]


@dataclass
class FoFResult:
    """Group membership for the particles passed to ``find_groups``.

    group_index[i] is the 0-based group number of particle i (0 = largest
    group) or UNGROUPED (-1).  group_len[g] is the member count of group g.
    """
    group_index: np.ndarray
    group_len: np.ndarray
    linking_length: float

    @property
    def n_groups(self) -> int:
        return len(self.group_len)


def mean_separation_from_box(n_particles: int, boxsize: float) -> float:
    """Mean interparticle separation (V/N)^(1/3) for N particles filling a cube."""
    if n_particles <= 0 or boxsize is None or boxsize <= 0:
        raise ValueError("need N > 0 and a positive box size")
    return (boxsize ** 3 / n_particles) ** (1.0 / 3.0)


def mean_separation_gadget4(primary_masses: np.ndarray, omega0: float, omega_baryon: float,
                            has_gas: bool, unit_length_cm: float, unit_mass_g: float,
                            unit_velocity_cms: float) -> float:
    """Gadget-4 mean separation (m_mean / rho_link)^(1/3), in comoving code units."""
    m = np.asarray(primary_masses, dtype=np.float64)
    if m.size == 0:
        raise ValueError("no primary particles")
    unit_time_s = unit_length_cm / unit_velocity_cms
    g_code = GRAVITY_CGS * unit_mass_g * unit_time_s ** 2 / unit_length_cm ** 3
    hubble_code = HUBBLE_CGS * unit_time_s
    rho_crit = 3.0 * hubble_code ** 2 / (8.0 * np.pi * g_code)
    omega = (omega0 - omega_baryon) if has_gas else omega0
    if omega <= 0:
        raise ValueError(f"non-positive linking density parameter Omega={omega}")
    return float((m.sum() / m.size / (omega * rho_crit)) ** (1.0 / 3.0))


def _wrap(positions: np.ndarray, boxsize: Optional[float], in_place: bool = False) -> np.ndarray:
    """float64 positions inside [0, boxsize); with in_place, a float64 input is wrapped without a copy."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions contain NaN or infinite values")
    if boxsize:
        if in_place and pos.flags.writeable:
            np.mod(pos, boxsize, out=pos)
        else:
            pos = np.mod(pos, boxsize)
        pos[pos >= boxsize] = 0.0  # np.mod can round up to exactly boxsize
    return pos


def _link_labels_kdtree(pos: np.ndarray, linking_length: float, boxsize: Optional[float],
                        chunk_size: int) -> np.ndarray:
    """Reference linker: enumerate every pair within the linking length.

    Simple and exact but slow and memory-hungry in dense haloes, where the
    number of pairs grows with the square of the local density.  Kept as an
    independent cross-check of the grid linker.
    """
    n = len(pos)
    tree = cKDTree(pos, boxsize=boxsize if boxsize else None)
    labels = np.arange(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sub = cKDTree(pos[start:stop], boxsize=boxsize if boxsize else None)
        rec = sub.sparse_distance_matrix(tree, linking_length, output_type='ndarray')
        i = rec['i'].astype(np.int64) + start
        j = rec['j'].astype(np.int64)
        keep = i < j
        a, b = labels[i[keep]], labels[j[keep]]
        keep = a != b
        if not np.any(keep):
            continue
        labels = _merge(labels, a[keep], b[keep], n)
    return labels


def _merge(labels: np.ndarray, a: np.ndarray, b: np.ndarray, n_nodes: int) -> np.ndarray:
    """Relabel so that nodes a[k] and b[k] (and everything already joined to them) share a label."""
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n_nodes, n_nodes))
    _, comp = connected_components(graph, directed=False)
    return comp[labels].astype(np.int64)


def _sq_sep(x: np.ndarray, y: np.ndarray, boxsize: Optional[float]) -> np.ndarray:
    """Squared separation of x[k] and y[k], minimum image when periodic (the linkers' one test)."""
    dd = x - y
    if boxsize:
        dd -= boxsize * np.round(dd / boxsize)
    return np.einsum('ij,ij->i', dd, dd)


def _cells_touch(pa: np.ndarray, pb: np.ndarray, linking_length: float, boxsize: Optional[float],
                 chunk: int = 1 << 15) -> bool:
    """Is some pair (one point of pa, one of pb) within the linking length?

    Same answer as testing every pair with _sq_sep, without the len(pa) x len(pb)
    arrays that would need: a k-d tree on pb finds, for each point of pa, its
    nearest point of pb within slightly more than the linking length, and that
    pair is tested exactly.  Only if every such nearest pair misses by rounding
    are all candidates of those points tested, so the result is exact.
    """
    l2 = linking_length ** 2
    reach = linking_length * (1 + 1e-9)
    tree = cKDTree(pb, boxsize=boxsize if boxsize else None)
    for s0 in range(0, len(pa), chunk):
        q = pa[s0:s0 + chunk]
        d, j = tree.query(q, k=1, distance_upper_bound=reach)
        near = np.flatnonzero(np.isfinite(d))
        if len(near) == 0:
            continue
        if np.any(_sq_sep(q[near], pb[j[near]], boxsize) <= l2):
            return True
        for i in near:
            cand = np.asarray(tree.query_ball_point(q[i], reach), dtype=np.int64)
            if len(cand) and np.any(_sq_sep(np.repeat(q[i:i + 1], len(cand), axis=0), pb[cand], boxsize) <= l2):
                return True
    return False


def _link_labels_grid(pos: np.ndarray, linking_length: float, boxsize: Optional[float],
                      batch: int = 2_000_000, quick: int = 4, progress: Progress = None,
                      reorder_in_place: bool = False, heavy: int = 1 << 18) -> Optional[np.ndarray]:
    """Grid linker: exact, with Gadget-style pruning of already-linked regions.

    Space is cut into cubic cells of side s <= l/sqrt(3), so every pair inside
    one cell is within l and the cell's particles are linked outright.  Cells
    are then joined across each neighbour offset (nearest offsets first), and
    a pair of cells is only examined while the two are still in different
    groups.  Inside haloes nearly all cell pairs are already joined after the
    first few offsets, which is what keeps time and memory bounded; the
    remaining pairs are tested particle by particle, first on a few particles
    per cell and then exhaustively, so the result is exact.  A cell pair with
    more than ``heavy`` particle pairs (two crowded cells, as in a galaxy disc)
    is tested with a k-d tree instead of all pairs at once (_cells_touch),
    which keeps memory bounded however crowded the cells are.

    Memory: besides ``pos``, about 16 bytes per particle and 100 per occupied
    cell.  With reorder_in_place, ``pos`` is sorted by cell in place while
    linking (one column at a time) and restored before returning, instead of
    holding a sorted copy (24 bytes per particle).
    Returns None when the box is too small for the grid (caller falls back).
    """
    n = len(pos)
    l2 = linking_length ** 2
    if boxsize:
        ncell = int(np.ceil(boxsize * np.sqrt(3.0) / linking_length))
        s = boxsize / ncell
        dims = np.array([ncell] * 3, dtype=np.int64)
        lo = np.zeros(3)
    else:
        s = linking_length / np.sqrt(3.0) * (1 - 1e-9)
        lo = pos.min(axis=0)
        dims = ((pos.max(axis=0) - lo) // s).astype(np.int64) + 1
    reach = int(np.ceil(linking_length / s))
    if boxsize and dims[0] < 2 * reach + 1:
        return None
    if np.prod(dims.astype(np.float64)) > 2.0 ** 62:
        return None

    key = np.zeros(n, dtype=np.int64)
    for k in range(3):  # one axis at a time: same keys as the (N, 3) form, a third of the temporaries
        c = np.floor((pos[:, k] - lo[k]) / s).astype(np.int64)
        np.clip(c, 0, dims[k] - 1, out=c)
        key *= dims[k]
        key += c
        del c
    order = np.argsort(key, kind='stable')
    skey = key[order]
    del key
    first = np.concatenate([[0], np.flatnonzero(skey[1:] != skey[:-1]) + 1]) if n else np.zeros(0, np.int64)
    ukey = skey[first]
    del skey
    start = first
    count = np.diff(np.append(first, n))
    ncells = len(ukey)
    if reorder_in_place:
        for k in range(3):
            pos[:, k] = pos[order, k]
        spos = pos
    else:
        spos = pos[order]
    cdt = np.int32 if dims.max() < 2 ** 31 else np.int64
    ucell = np.stack([ukey // (dims[1] * dims[2]), (ukey // dims[2]) % dims[1], ukey % dims[2]], axis=1).astype(cdt)
    clab = np.arange(ncells, dtype=np.int64)

    offsets = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                d = (dx, dy, dz)
                if d <= (0, 0, 0):
                    continue  # each unordered cell pair once
                gap = s * np.sqrt(sum(max(abs(c) - 1, 0) ** 2 for c in d))
                if gap <= linking_length:
                    offsets.append((gap, d))
    offsets.sort()

    def report(done):
        if progress is not None:
            progress(min(done / len(offsets), 1.0), "Linking particles")

    def linked(a, b, cap, step):
        """For cell pairs (a[k], b[k]): is some particle pair within l?  Tests up to cap x cap particles."""
        na = np.minimum(count[a], cap)
        nb = np.minimum(count[b], cap)
        work = na * nb
        hit = np.zeros(len(a), dtype=bool)
        big = np.flatnonzero(work > heavy)
        for i, k in enumerate(big):
            pa = spos[start[a[k]]:start[a[k]] + na[k]]
            pb = spos[start[b[k]]:start[b[k]] + nb[k]]
            hit[k] = _cells_touch(pa, pb, linking_length, boxsize)
            if i % 64 == 63:
                report(step + i / len(big))
        work[big] = 0
        ends = np.cumsum(work)
        lo_k = 0
        while lo_k < len(a):
            hi_k = int(np.searchsorted(ends, (ends[lo_k - 1] if lo_k else 0) + batch, side='right'))
            hi_k = max(hi_k, lo_k + 1)
            w = work[lo_k:hi_k]
            pair = np.repeat(np.arange(lo_k, hi_k), w)
            k = np.arange(len(pair)) - np.repeat(np.cumsum(w) - w, w)
            ia = start[a[pair]] + k // nb[pair]
            ib = start[b[pair]] + k % nb[pair]
            close = _sq_sep(spos[ia], spos[ib], boxsize) <= l2
            hit[np.unique(pair[close])] = True
            lo_k = hi_k
        return hit

    for step, (_, d) in enumerate(offsets):
        report(step)
        nkey = np.zeros(ncells, dtype=np.int64)
        valid = np.ones(ncells, dtype=bool)
        for k in range(3):
            c = ucell[:, k].astype(np.int64) + d[k]
            if boxsize:
                c %= dims[k]
            else:
                valid &= (c >= 0) & (c < dims[k])
            nkey *= dims[k]
            nkey += c
            del c
        idx = np.searchsorted(ukey, nkey)
        idx[idx >= ncells] = 0
        occ = valid & (ukey[idx] == nkey)
        del nkey, valid
        a = np.nonzero(occ)[0]
        b = idx[occ]
        del idx, occ
        for cap in (quick, None):
            todo = clab[a] != clab[b]
            if not np.any(todo):
                break
            a, b = a[todo], b[todo]
            if cap is None:
                cap = int(count.max())
            hit = linked(a, b, cap, step)
            if np.any(hit):
                clab = _merge(clab, clab[a[hit]], clab[b[hit]], ncells)
            if cap >= count.max():
                break
    report(len(offsets))

    del ucell
    labels = np.empty(n, dtype=np.int64)
    labels[order] = np.repeat(clab, count)
    if reorder_in_place:
        tmp = np.empty(n, dtype=pos.dtype)
        for k in range(3):
            tmp[order] = pos[:, k]
            pos[:, k] = tmp
    return labels


def _link_labels(pos: np.ndarray, linking_length: float, boxsize: Optional[float],
                 chunk_size: int, method: str = 'grid', progress: Progress = None,
                 reorder_in_place: bool = False) -> np.ndarray:
    """Connected-component label (arbitrary values) for every particle."""
    if method == 'grid':
        labels = _link_labels_grid(pos, linking_length, boxsize, progress=progress,
                                   reorder_in_place=reorder_in_place)
        if labels is not None:
            return labels
    elif method != 'kdtree':
        raise ValueError(f"unknown FoF method {method!r}")
    return _link_labels_kdtree(pos, linking_length, boxsize, chunk_size)


def _number_groups(labels: np.ndarray, min_group_size: int, tie_key: Optional[np.ndarray]):
    """Turn raw component labels into 0-based group numbers, largest first."""
    n = len(labels)
    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    keep = np.nonzero(counts >= min_group_size)[0]
    if tie_key is None:
        tie_key = np.arange(n)
    # smallest tie_key (e.g. particle ID) inside each component breaks size ties
    min_key = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(min_key, inverse, np.asarray(tie_key, dtype=np.int64))
    order = keep[np.lexsort((min_key[keep], -counts[keep]))]
    number = np.full(len(uniq), UNGROUPED, dtype=np.int64)
    number[order] = np.arange(len(order))
    return number[inverse], counts[order].astype(np.int64)


def find_groups(positions: np.ndarray, linking_length: float, boxsize: Optional[float] = None,
                min_group_size: int = GADGET4_DEFAULT_MIN_LEN, ids: Optional[np.ndarray] = None,
                chunk_size: int = 1_000_000, method: str = 'grid', progress: Progress = None,
                overwrite_positions: bool = False) -> FoFResult:
    """Run FoF on primary-particle positions with an explicit linking length.

    positions      (N, 3) comoving coordinates in the same units as linking_length.
    boxsize        periodic box side; None or 0 means open (non-periodic) boundaries.
    ids            optional particle IDs used only to order equal-size groups.
    method         'grid' (default, bounded memory) or 'kdtree' (pair enumeration,
                   used as an independent cross-check in the tests).
    progress       optional callable(fraction, text), called as linking proceeds.
    overwrite_positions  saves memory on big runs: a float64 ``positions`` array is
                   wrapped into the box in place and used as the linker's work
                   array (its order is restored before returning).
    """
    if linking_length is None or not np.isfinite(linking_length) or linking_length <= 0:
        raise ValueError(f"linking length must be positive, got {linking_length}")
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")
    if boxsize and linking_length >= 0.5 * boxsize:
        raise ValueError("linking length must be smaller than half the box size")
    pos = _wrap(positions, boxsize, in_place=overwrite_positions)
    n = len(pos)
    if n == 0:
        return FoFResult(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), float(linking_length))
    in_place = overwrite_positions and pos.flags.c_contiguous and pos.flags.writeable
    labels = _link_labels(pos, float(linking_length), boxsize, max(1, int(chunk_size)), method,
                          progress=progress, reorder_in_place=in_place)
    group_index, group_len = _number_groups(labels, min_group_size, ids)
    return FoFResult(group_index, group_len, float(linking_length))


def attach_secondaries(primary_positions: np.ndarray, primary_group_index: np.ndarray,
                       secondary_positions: np.ndarray, linking_length: float,
                       boxsize: Optional[float] = None, max_distance_factor: float = 4.0) -> np.ndarray:
    """Give each secondary particle the group of its nearest primary particle.

    Mirrors Gadget-4 fof_nearest.cc, which stops searching beyond roughly four
    linking lengths; secondaries with no primary that close stay UNGROUPED.
    """
    sec = _wrap(secondary_positions, boxsize)
    out = np.full(len(sec), UNGROUPED, dtype=np.int64)
    if len(sec) == 0 or len(primary_positions) == 0:
        return out
    tree = cKDTree(_wrap(primary_positions, boxsize), boxsize=boxsize if boxsize else None)
    dist, idx = tree.query(sec, k=1, distance_upper_bound=max_distance_factor * linking_length)
    found = np.isfinite(dist)
    out[found] = np.asarray(primary_group_index)[idx[found]]
    return out


def group_centres_of_mass(positions: np.ndarray, group_index: np.ndarray, n_groups: int,
                          masses: Optional[np.ndarray] = None,
                          boxsize: Optional[float] = None) -> np.ndarray:
    """Mass-weighted centre of each group, unwrapping periodic images.

    Each member is placed at its nearest image relative to the group's first
    member before averaging, so groups straddling the box edge are handled.
    This assumes every member lies within half a box length of that first
    member, which holds for any FoF group much smaller than the box.
    """
    pos = np.asarray(positions, dtype=np.float64)
    gi = np.asarray(group_index)
    sel = gi >= 0
    pos, gi = pos[sel], gi[sel]
    w = np.ones(len(pos)) if masses is None else np.asarray(masses, dtype=np.float64)[sel]
    first = np.full(n_groups, -1, dtype=np.int64)
    order = np.arange(len(gi))[::-1]
    first[gi[order]] = order  # last write wins -> first member of each group
    ref = pos[first[gi]]
    d = pos - ref
    if boxsize:
        d -= boxsize * np.round(d / boxsize)
    wsum = np.bincount(gi, weights=w, minlength=n_groups)
    cm = np.empty((n_groups, 3))
    for k in range(3):
        cm[:, k] = np.bincount(gi, weights=w * d[:, k], minlength=n_groups) / wsum
    cm += pos[first]
    if boxsize:
        cm = np.mod(cm, boxsize)
    return cm
