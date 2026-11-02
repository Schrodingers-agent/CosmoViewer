"""
models.py — synthetic test systems with known answers.

plummer() samples an isotropic Plummer sphere (Aarseth, Hénon & Wielen 1974):
    M(<r) = M r³ / (r² + a²)^{3/2}
    r_f   = a / sqrt(f^{-2/3} - 1)                 (Lagrangian radius)
    speeds from g(q) = q² (1 - q²)^{7/2}, v = q v_esc(r),
    v_esc = sqrt(2 G M) (r² + a²)^{-1/4}
write_gadget4_snapshot() writes particles as a Gadget-4 style HDF5 file with
the same Header / Parameters / unit attributes as the real example outputs,
so tests exercise the real reading path.

These generators are for tests only; their formulas are textbook results
quoted from memory and are checked numerically in test_lagrangian_radii_and_centres.py.
"""
from __future__ import annotations

import h5py
import numpy as np

G_KPC_KMS2_MSUN = 4.300917270e-6     # G in kpc (km/s)^2 / Msun


def plummer_lagrangian_radius(f, a=1.0):
    f = np.asarray(f, dtype=float)
    return a / np.sqrt(f ** (-2.0 / 3.0) - 1.0)


def _isotropic(rng, n):
    cos_t = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    return np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])


def plummer(n, total_mass=1.0e5, a=0.001, seed=0, r_max_factor=None):
    """(pos kpc, vel km/s, mass Msun) for an isotropic Plummer sphere centred at 0.
    r_max_factor, if given, rejects radii beyond r_max_factor × a (the
    truncation then changes the Lagrangian radii, so tests leave it None)."""
    rng = np.random.default_rng(seed)
    r = np.empty(0)
    while len(r) < n:
        x = rng.uniform(0, 1, 2 * n)
        x = x[x > 0]
        rr = a / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
        if r_max_factor:
            rr = rr[rr <= r_max_factor * a]
        r = np.concatenate([r, rr])
    r = r[:n]
    pos = r[:, None] * _isotropic(rng, n)
    q = np.empty(0)
    while len(q) < n:
        x1, x2 = rng.uniform(0, 1, 2 * n), rng.uniform(0, 0.1, 2 * n)
        q = np.concatenate([q, x1[x2 < x1 ** 2 * (1 - x1 ** 2) ** 3.5]])
    q = q[:n]
    v_esc = np.sqrt(2.0 * G_KPC_KMS2_MSUN * total_mass) * (r ** 2 + a ** 2) ** -0.25
    vel = (q * v_esc)[:, None] * _isotropic(rng, n)
    mass = np.full(n, total_mass / n)
    return pos, vel, mass


def plummer_density(r, total_mass=1.0, a=1.0):
    return 3.0 * total_mass / (4.0 * np.pi * a ** 3) * (1.0 + (np.asarray(r) / a) ** 2) ** -2.5


def plummer_mass(r, total_mass=1.0, a=1.0):
    r = np.asarray(r, dtype=float)
    return total_mass * r ** 3 / (r ** 2 + a ** 2) ** 1.5


def plummer_sigma(r, total_mass=1.0, a=1.0):
    """1-D velocity dispersion of the isotropic Plummer sphere: σ² = G M / (6 √(r² + a²))."""
    return np.sqrt(G_KPC_KMS2_MSUN * total_mass / (6.0 * np.sqrt(np.asarray(r) ** 2 + a ** 2)))


def hernquist(n, total_mass=1.0e5, a=0.001, seed=0):
    """(pos kpc, mass Msun) for a Hernquist (1990) sphere, M(<r) = M r² / (r + a)².
    Inverting the cumulative mass: r = a √f / (1 − √f). Positions only."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0, 1, n)
    s = np.sqrt(f)
    r = a * s / (1.0 - s)
    return r[:, None] * _isotropic(rng, n), np.full(n, total_mass / n)


def hernquist_mass(r, total_mass=1.0, a=1.0):
    r = np.asarray(r, dtype=float)
    return total_mass * r ** 2 / (r + a) ** 2


def anisotropic_velocities(pos, sigma_r, sigma_t, seed=0):
    """Gaussian velocities with dispersion sigma_r along r̂ and sigma_t in each of
    the two tangential directions, so β = 1 − σ_t² / σ_r² exactly."""
    rng = np.random.default_rng(seed)
    n = len(pos)
    r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    ref = np.where(np.abs(r_hat[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(r_hat, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(r_hat, e1)
    return (rng.normal(0, sigma_r, (n, 1)) * r_hat + rng.normal(0, sigma_t, (n, 1)) * e1
            + rng.normal(0, sigma_t, (n, 1)) * e2)


def write_gadget4_snapshot(path, parts: dict, time=0.0, box=0.0, *, cosmological=False,
                           hubble=1.0, omega0=0.0, omega_lambda=0.0,
                           unit_length_cm=3.085678e21, unit_mass_g=1.989e43,
                           unit_velocity_cms=1.0e5, mass_table=None, extra_params=None, config=None):
    """parts: {ptype: dict(pos=(N,3) code units, vel=(N,3), mass=(N,) or None, ids=(N,))}.
    mass None means the type's mass comes from mass_table[ptype]. An optional
    "fields" entry {name: array} adds raw datasets without unit attributes.
    config: {name: value} written as the Config group (compile options)."""
    npart = np.zeros(6, dtype=np.uint64)
    for t, d in parts.items():
        npart[t] = len(d["pos"])
    mt = np.zeros(6) if mass_table is None else np.asarray(mass_table, float)
    h_len = -1.0 if cosmological else 0.0
    a_len = 1.0 if cosmological else 0.0
    a_vel = 0.5 if cosmological else 0.0
    with h5py.File(path, "w") as f:
        hd = f.create_group("Header")
        hd.attrs["BoxSize"] = float(box)
        hd.attrs["MassTable"] = mt
        hd.attrs["NumFilesPerSnapshot"] = np.int32(1)
        hd.attrs["NumPart_ThisFile"] = npart
        hd.attrs["NumPart_Total"] = npart
        hd.attrs["Time"] = float(time)
        hd.attrs["Redshift"] = float(1.0 / time - 1.0) if cosmological and time > 0 else 0.0
        pa = f.create_group("Parameters")
        pa.attrs["BoxSize"] = float(box)
        pa.attrs["ComovingIntegrationOn"] = np.int32(1 if cosmological else 0)
        pa.attrs["HubbleParam"] = float(hubble)
        pa.attrs["Omega0"] = float(omega0)
        pa.attrs["OmegaLambda"] = float(omega_lambda)
        pa.attrs["UnitLength_in_cm"] = float(unit_length_cm)
        pa.attrs["UnitMass_in_g"] = float(unit_mass_g)
        pa.attrs["UnitVelocity_in_cm_per_s"] = float(unit_velocity_cms)
        for k, v in (extra_params or {}).items():
            pa.attrs[k] = v
        if config is not None:
            cg = f.create_group("Config")
            for k, v in config.items():
                cg.attrs[k] = v
        for t, d in parts.items():
            g = f.create_group(f"PartType{t}")
            ds = g.create_dataset("Coordinates", data=np.asarray(d["pos"], np.float64))
            ds.attrs.update({"a_scaling": a_len, "h_scaling": h_len, "length_scaling": 1.0,
                             "mass_scaling": 0.0, "velocity_scaling": 0.0, "to_cgs": unit_length_cm})
            ds = g.create_dataset("Velocities", data=np.asarray(d["vel"], np.float64))
            ds.attrs.update({"a_scaling": a_vel, "h_scaling": 0.0, "length_scaling": 0.0,
                             "mass_scaling": 0.0, "velocity_scaling": 1.0, "to_cgs": unit_velocity_cms})
            if d.get("mass") is not None:
                ds = g.create_dataset("Masses", data=np.asarray(d["mass"], np.float64))
                ds.attrs.update({"a_scaling": 0.0, "h_scaling": -1.0 if cosmological else 0.0,
                                 "length_scaling": 0.0, "mass_scaling": 1.0,
                                 "velocity_scaling": 0.0, "to_cgs": unit_mass_g})
            g.create_dataset("ParticleIDs", data=np.asarray(d["ids"], np.uint64))
            for name, arr in d.get("fields", {}).items():
                g.create_dataset(name, data=np.asarray(arr))
    return path
