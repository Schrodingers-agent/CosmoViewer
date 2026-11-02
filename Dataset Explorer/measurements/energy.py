"""
energy.py — energy, momentum and virial ratio of a snapshot, and their drift
over a run.

energy_budget() gives, for the chosen particle types:

  K        kinetic energy ½ Σ m v²  (peculiar velocities in cosmological runs,
           which is what Gadget-4's energy.txt reports)
  U        thermal energy Σ m u of the gas
  W        potential energy, with its method and error (below)
  E        K + U + W
  Q        virial ratio K / |W|   (0.5 in equilibrium)
  P, L     total linear momentum and angular momentum about the origin
  com_pos, com_vel   centre of mass and its velocity
  per type K, U and mass

Potential energy W = −½ Σᵢ Σⱼ≠ᵢ G mᵢ mⱼ g(rᵢⱼ), with Gadget's spline kernel
g (Plummer-equivalent ε, Newtonian beyond 2.8 ε) and pair softening
max(εᵢ, εⱼ). Methods:

  stored   ½ Σ m Φ from a Potential dataset, when the snapshot has one
  direct   every pair, exact (N ≤ MAX_DIRECT: ~10 s at 30,000)
  pairs    an unbiased Monte Carlo estimate from n_pairs random pairs, with
           its standard error. Particles are ordered by ID before sampling,
           so while a run keeps the same particles the same pairs are used in
           every snapshot, and the drift ΔW is far more precise than W itself
  none     W not computed

"auto" picks stored, then direct, then pairs. Every particle is used in all
of them; for "pairs" the error comes from sampling pairs, and it is reported.
Periodic boxes get no W: the potential there needs Ewald summation.

The kernel is the one in Springel (2005, Gadget-2 paper), quoted from memory
and checked numerically (continuity at u = ½ and 1, −1/ε at r = 0).
"""
from __future__ import annotations

import numpy as np
from astropy import units as u

from .models import G_KPC_KMS2_MSUN
from .results import Provenance, Result
from .snapshot_view import MSUN_G, SnapshotView

VERSION = "1"
MAX_DIRECT = 30_000
DEFAULT_PAIRS = 4_000_000
SPLINE_H = 2.8          # h = 2.8 ε
_BLOCK = 2_000_000      # pair evaluations per block


def spline_kernel(r, eps):
    """g(r) ≥ 0 with φ = −G m g(r): Gadget's cubic-spline softened 1/r.
    eps may be an array (pair softening); eps = 0 gives 1/r."""
    r = np.asarray(r, dtype=float)
    eps = np.broadcast_to(np.asarray(eps, dtype=float), r.shape)
    out = np.empty_like(r)
    newton = (eps <= 0) | (r >= SPLINE_H * eps)
    with np.errstate(divide="ignore"):
        out[newton] = 1.0 / r[newton]
    soft = ~newton
    if soft.any():
        h = SPLINE_H * eps[soft]
        q = r[soft] / h
        inner = q < 0.5
        w = np.empty_like(q)
        qi = q[inner]
        w[inner] = 16 / 3 * qi ** 2 - 48 / 5 * qi ** 4 + 32 / 5 * qi ** 5 - 14 / 5
        qo = q[~inner]
        w[~inner] = (1 / (15 * qo) + 32 / 3 * qo ** 2 - 16 * qo ** 3 + 48 / 5 * qo ** 4
                     - 32 / 15 * qo ** 5 - 16 / 5)
        out[soft] = -w / h
    return out


def _w_direct(pos, mass, eps):
    n = len(mass)
    total = 0.0
    rows = max(1, _BLOCK // max(n, 1))
    for i in range(0, n, rows):
        j0 = i + 1
        if j0 >= n:
            break
        a = pos[i:i + rows]
        d = a[:, None, :] - pos[None, j0:, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        e = np.maximum(eps[i:i + rows, None], eps[None, j0:])
        g = spline_kernel(r, e)
        k = np.arange(len(a))[:, None] + i
        jj = np.arange(j0, n)[None, :]
        g = np.where(jj > k, g, 0.0)
        total += np.einsum("i,ij,j->", mass[i:i + rows], g, mass[j0:])
    return -G_KPC_KMS2_MSUN * total, 0.0


def _w_pairs(pos, mass, eps, n_pairs, seed):
    n = len(mass)
    rng = np.random.default_rng(seed)
    acc, acc2, done = 0.0, 0.0, 0
    while done < n_pairs:
        k = min(_BLOCK, n_pairs - done)
        i = rng.integers(0, n, k)
        j = rng.integers(0, n - 1, k)
        j += j >= i                       # j ≠ i, uniform over the others
        d = pos[i] - pos[j]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        x = mass[i] * mass[j] * spline_kernel(r, np.maximum(eps[i], eps[j]))
        acc += x.sum()
        acc2 += np.dot(x, x)
        done += k
    mean = acc / done
    var = max(acc2 / done - mean ** 2, 0.0)
    n_pair_total = 0.5 * n * (n - 1)
    w = -G_KPC_KMS2_MSUN * n_pair_total * mean
    err = G_KPC_KMS2_MSUN * n_pair_total * np.sqrt(var / done)
    return w, err


def potential_energy(pos, mass, eps=None, method="auto", n_pairs=DEFAULT_PAIRS, seed=0):
    """(W, error, method) for particles in kpc and Msun; W in Msun (km/s)²."""
    pos = np.asarray(pos, float)
    mass = np.asarray(mass, float)
    eps = np.zeros(len(mass)) if eps is None else np.broadcast_to(np.asarray(eps, float), mass.shape)
    if method == "auto":
        method = "direct" if len(mass) <= MAX_DIRECT else "pairs"
    if method == "direct":
        w, e = _w_direct(pos, mass, eps)
    elif method == "pairs":
        w, e = _w_pairs(pos, mass, eps, int(n_pairs), seed)
    else:
        raise ValueError("method must be auto, direct or pairs")
    return w, e, method


def _thermal(view, t):
    if not view.has(t, "InternalEnergy"):
        return None
    u_cgs = view.field_cgs(t, "InternalEnergy", "physical")       # erg / g
    return u_cgs / 1.0e10                                           # (km/s)²


def energy_budget(snapshot, types=None, *, potential: str = "auto",
                  n_pairs: int = DEFAULT_PAIRS, seed: int = 0, ids=None,
                  ids_label: str = "") -> Result:
    """Energy budget of one snapshot. snapshot: path, VirtualSnapshotSet or SnapshotView.
    ids: keep only these particle IDs (a selection set); the result then
    describes that set, and W is the set's own potential energy."""
    view = snapshot if isinstance(snapshot, SnapshotView) else SnapshotView(snapshot)
    types = view.types_present() if types is None else [int(t) for t in types]
    p = view.particles(types, frame="physical", velocities=True, ids=True)
    notes = list(p.notes)
    if ids is not None:
        keep = np.isin(np.asarray(p.ids).astype(np.uint64), np.asarray(ids, dtype=np.uint64))
        if not keep.any():
            raise ValueError("none of the selected IDs are in these particle types")
        file_index = np.flatnonzero(keep)
        p = p.subset(keep)
        notes.append(f"restricted to {ids_label or 'a set of'} {len(p):,} particles")
    else:
        file_index = None
    if np.isnan(p.vel).any():
        raise ValueError("some particles have no velocities")

    # identical particle order in every snapshot: sort by ID across all types, so a
    # gas particle that becomes a star (same ID) keeps its place
    order = np.argsort(p.ids, kind="stable") if p.ids is not None and (p.ids >= 0).all() \
        else np.arange(len(p))
    pos, vel, m, pt = p.pos[order], p.vel[order], p.mass[order], p.ptype[order]

    v2 = np.einsum("ij,ij->i", vel, vel)
    values = {}
    per_K, per_U, per_M = [], [], []
    U_total = 0.0
    for t in types:
        s = pt == t
        per_K.append(0.5 * np.dot(m[s], v2[s]))
        per_M.append(m[s].sum())
        uu = _thermal(view, t)
        if uu is not None and file_index is not None:
            off = sum(view.count(x) for x in types[:types.index(t)])
            uu = uu[file_index[p.ptype == t] - off]
        # p is in file order, like the InternalEnergy dataset
        per_U.append(float(np.dot(p.mass[p.ptype == t], uu)) if uu is not None else 0.0)
    K = float(np.sum(per_K))
    U = float(np.sum(per_U))

    P = (m[:, None] * vel).sum(axis=0)
    L = (m[:, None] * np.cross(pos, vel)).sum(axis=0)
    M = m.sum()
    com = (m[:, None] * pos).sum(axis=0) / M
    vcom = P / M

    eps = np.zeros(len(m))
    soft_notes = []
    for t in types:
        e, how = view.softening(t)
        if e is not None:
            eps[pt == t] = e
            soft_notes.append(f"PartType{t} ε = {e:.4g} kpc ({how})")
    if not soft_notes:
        soft_notes.append("no softening parameters: unsoftened 1/r")

    stored = all(view.has(t, "Potential") for t in types)
    if p.box > 0:
        W, W_err, method = np.nan, np.nan, "none"
        notes.append("periodic box: W needs Ewald summation and is not computed")
    elif potential == "none":
        W, W_err, method = np.nan, np.nan, "none"
    elif stored and potential in ("auto", "stored"):
        phi = np.concatenate([view.field_cgs(t, "Potential", "physical") for t in types]) / 1.0e10
        if file_index is not None:
            # a selection: the stored Φ is the whole snapshot's potential at each member
            phi = phi[file_index]
            notes.append("W from the stored Potential: ½ Σ m Φ of the set's members in the "
                         "whole snapshot's field, not the set's own potential energy")
        W, W_err, method = 0.5 * float(np.dot(p.mass, phi)), 0.0, "stored"
    else:
        if potential == "stored":
            raise ValueError("no Potential dataset in this snapshot")
        W, W_err, method = potential_energy(pos, m, eps, potential, n_pairs, seed)
        if file_index is not None:
            notes.append("W is the potential energy of the set alone")
        notes.extend(soft_notes)
        if method == "pairs":
            notes.append(f"W from {n_pairs:,} random pairs (seed {seed}): a Monte Carlo "
                         "estimate with the quoted standard error")
    if view.is_cosmological:
        notes.append("cosmological run: K is peculiar; E is not conserved "
                     "(Layzer–Irvine is not implemented)")

    E = K + U + W
    eunit = u.Msun * (u.km / u.s) ** 2
    values.update({
        "K": K * eunit, "U": U * eunit, "W": W * eunit, "W_err": W_err * eunit,
        "E": E * eunit, "Q": K / abs(W) if np.isfinite(W) and W != 0 else np.nan,
        "Q_err": K * W_err / W ** 2 if np.isfinite(W) and W != 0 else np.nan,
        "P": P * u.Msun * u.km / u.s, "L": L * u.Msun * u.kpc * u.km / u.s,
        "mass": M * u.Msun, "com_pos": com * u.kpc, "com_vel": vcom * u.km / u.s,
        "K_type": np.array(per_K) * eunit, "U_type": np.array(per_U) * eunit,
        "mass_type": np.array(per_M) * u.Msun, "types": np.array(types),
    })
    prov = Provenance(
        measurement="energy_budget", version=VERSION, files=list(view.files),
        particle_types=types, n_used=len(p), n_total=p.n_total, frame="physical",
        time=view.time_info(),
        parameters={"potential": method, "n_pairs": n_pairs if method == "pairs" else None,
                    "seed": seed if method == "pairs" else None},
        notes=notes)
    return Result(values=values, provenance=prov)


def code_energy_unit(view: SnapshotView) -> float:
    """Msun (km/s)² per Gadget code energy unit (UnitMass · UnitVelocity², per h
    in cosmological runs), for comparison with energy.txt."""
    m = view._unit("UnitMass_in_g", 1.989e43) / MSUN_G
    v = view._unit("UnitVelocity_in_cm_per_s", 1e5) / 1e5
    return m * v ** 2 / (view.h if view.is_cosmological else 1.0)


def read_energy_txt(path: str) -> dict:
    """Gadget-4 energy.txt. Columns (verified on CollidingGalaxiesSFR snapshot_000
    against the snapshot, all to the file's 8 digits): time; total thermal,
    potential and kinetic; then thermal, potential and kinetic per type; then
    mass per type. Values in code units."""
    a = np.atleast_2d(np.loadtxt(path))
    nt = (a.shape[1] - 4) // 4
    if 4 + 4 * nt != a.shape[1]:
        raise ValueError(f"{path}: {a.shape[1]} columns do not fit 4 + 4 × NTYPES")
    per = a[:, 4:4 + 3 * nt].reshape(len(a), nt, 3)
    return {"time": a[:, 0], "thermal": a[:, 1], "potential": a[:, 2], "kinetic": a[:, 3],
            "thermal_type": per[:, :, 0], "potential_type": per[:, :, 1],
            "kinetic_type": per[:, :, 2], "mass_type": a[:, 4 + 3 * nt:], "ntypes": nt}


def conservation(run, snapshots=None, types=None, cache=None, **kw):
    """energy_budget over a Run (or RunSet), plus drifts relative to the first
    snapshot: dE/|E0|, |ΔL| / |L0| and the centre-of-mass shift.
    cache: a ResultCache, so a second call reads results back.
    Returns (ResultTable, drift dict of arrays per run)."""
    from .runset import RunSet
    rs = run if isinstance(run, RunSet) else RunSet([run])
    args = {"types": types, "version": VERSION,
            **{k: (np.asarray(v).tolist() if k == "ids" and v is not None else v)
               for k, v in kw.items()}}
    table = rs.apply(lambda v: energy_budget(v, types, **kw), snapshots=snapshots,
                     cache=cache, cache_name="energy_budget", cache_args=args)
    drift = {}
    for name in {r["run"] for r in table.rows}:
        s, t, E = table.series("E")[name]
        W = table.series("W")[name][2]
        Werr = table.series("W_err")[name][2]
        L = np.column_stack([table.series("L", i)[name][2] for i in range(3)])
        com = np.column_stack([table.series("com_pos", i)[name][2] for i in range(3)])
        with np.errstate(invalid="ignore", divide="ignore"):
            drift[name] = {
                "snapshot": s, "time_code": t, "E": E, "dE_rel": (E - E[0]) / abs(E[0]),
                "W_err_rel_E0": Werr / abs(E[0]),
                "Q": table.series("Q")[name][2], "Q_err": table.series("Q_err")[name][2],
                "dL_rel": np.linalg.norm(L - L[0], axis=1) / max(np.linalg.norm(L[0]), 1e-300),
                "com_shift_kpc": np.linalg.norm(com - com[0], axis=1)}
    return table, drift
