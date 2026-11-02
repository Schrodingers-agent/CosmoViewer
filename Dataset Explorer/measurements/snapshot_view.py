"""
snapshot_view.py — one snapshot, typed access, explicit units.

SnapshotView wraps io_manager.VirtualSnapshotSet (multi-file, any size) and
converts with the same functions the plugin pipeline uses
(data_utils.run_cosmology, scaling_exponents, apply_cosmological_factors), so
a value here equals the value the Histogram tab shows for the same field and
coordinate mode.

Units
-----
Every value is converted from the file's own unit attributes:
    physical = raw × to_cgs × h^h_scaling × a^a_scaling
("comoving" drops the a factor, exactly like the Options ▸ Comoving menu).
Gadget-4 names (to_cgs, h_scaling, a_scaling) and EAGLE names
(CGSConversionFactor, h-scale-exponent, aexp-scale-exponent) are both read.

Particle sets use fixed working units: kpc, km/s, Msun. Velocities are the
stored peculiar velocities, converted to physical km/s in both frames; the
Hubble flow is not added. (Gadget's stored w = v/√a is not a velocity in any
frame, which has never stopped anyone from plotting it, so it is converted in
Comoving mode too; passed on raw, it would put profiles and dispersions in
units of √a km/s.)

Masses: from the Masses (or Mass) dataset when present, otherwise from
Header/MassTable. MassTable carries no attributes, so its conversion is taken
from another type's Masses dataset in the same file when one exists, else from
Parameters/UnitMass_in_g with the Gadget convention h^-1 in cosmological runs.
Which source was used is recorded in the provenance notes.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import h5py
import numpy as np
from astropy import units as u

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_utils import (  # noqa: E402
    apply_cosmological_factors,
    run_cosmology,
    scaling_exponents,
)
from io_manager import VirtualSnapshotSet  # noqa: E402

KPC_CM = (1 * u.kpc).to(u.cm).value
KMS_CMS = 1.0e5
MSUN_G = (1 * u.Msun).to(u.g).value

POS_UNIT = u.kpc
VEL_UNIT = u.km / u.s
MASS_UNIT = u.Msun

_MASS_NAMES = ("Masses", "Mass")
_VEL_NAMES = ("Velocities", "Velocity")
_ID_NAMES = ("ParticleIDs", "ParticleID")


@dataclass
class ParticleSet:
    """Particles of one or more types from one snapshot, in kpc, km/s and Msun."""
    pos: np.ndarray                  # (N, 3) kpc
    mass: np.ndarray                 # (N,) Msun
    vel: np.ndarray | None = None    # (N, 3) km/s, peculiar
    ids: np.ndarray | None = None    # (N,) as stored (integer)
    ptype: np.ndarray | None = None  # (N,) int8
    box: float = 0.0                 # periodic box side in kpc, same frame as pos; 0 = not periodic
    frame: str = "physical"
    n_total: int = 0                 # particles of these types in the snapshot
    files: list = field(default_factory=list)
    particle_types: list = field(default_factory=list)
    time: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.mass)

    @property
    def periodic(self) -> bool:
        return self.box > 0

    def subset(self, mask_or_index) -> "ParticleSet":
        """A ParticleSet of selected particles. n_total becomes the selection size,
        so a measurement of the whole selection is a science result."""
        sel = mask_or_index
        sub = ParticleSet(
            pos=self.pos[sel], mass=self.mass[sel],
            vel=None if self.vel is None else self.vel[sel],
            ids=None if self.ids is None else self.ids[sel],
            ptype=None if self.ptype is None else self.ptype[sel],
            box=self.box, frame=self.frame, files=list(self.files),
            particle_types=list(self.particle_types), time=dict(self.time),
            notes=list(self.notes))
        sub.n_total = len(sub.mass)
        return sub


class SnapshotView:
    """Typed, unit-explicit access to one (possibly multi-file) snapshot."""

    def __init__(self, path_or_vset):
        # Duck-typed: a reloaded io_manager (tests do this) gives a different class.
        self.vset = (path_or_vset if hasattr(path_or_vset, "extract_array")
                     else VirtualSnapshotSet(str(path_or_vset)))
        self.header = self.vset.global_header
        self.params = self.vset.global_params
        self.cosmo = run_cosmology(self.header, self.params)
        self.files = list(self.vset.files)

    # ── Run facts ──────────────────────────────────────────────────────────
    @property
    def is_cosmological(self) -> bool:
        return bool(self.cosmo["is_cosmological"])

    @property
    def a(self) -> float:
        return float(self.cosmo["a"])

    @property
    def h(self) -> float:
        return float(self.cosmo["h"])

    def count(self, ptype: int) -> int:
        return int(self.vset.particle_counts.get(f"PartType{ptype}", 0))

    def types_present(self) -> list:
        return [t for t in range(6) if self.count(t) > 0]

    def _unit(self, key: str, default: float) -> float:
        v = self.params.get(key, self.header.get(key, default))
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) and v > 0 else default

    def box_size(self, frame: str = "physical") -> float:
        """Box side in kpc (0 for a non-periodic run). BoxSize is in code length,
        comoving and per h in cosmological runs."""
        box = float(self.header.get("BoxSize", 0.0) or 0.0)
        if box <= 0:
            return 0.0
        length_cm = self._unit("UnitLength_in_cm", 3.085678e21)
        if self.is_cosmological:
            factor = apply_cosmological_factors(self.h, -1.0, self.a if frame == "physical" else 1.0,
                                                1.0 if frame == "physical" else 0.0)
        else:
            factor = 1.0
        return box * length_cm * factor / KPC_CM

    def time_info(self) -> dict:
        """Snapshot time: code value, plus a and z (cosmological) or Gyr (isolated)."""
        t = float(self.cosmo["time_code"])
        info = {"code": t, "cosmological": self.is_cosmological}
        if self.is_cosmological:
            info["a"] = t
            info["z"] = 1.0 / t - 1.0 if t > 0 else float("inf")
            om = self.cosmo.get("omega0")
            ol = self.cosmo.get("omega_lambda")
            if om is not None and ol is not None and self.cosmo.get("h_known"):
                from astropy.cosmology import LambdaCDM
                cos = LambdaCDM(H0=100.0 * self.h, Om0=om, Ode0=ol)
                info["age_Gyr"] = float(cos.age(info["z"]).to(u.Gyr).value)
        else:
            # Gadget-4 writes h_scaling = 0 on every dataset when ComovingIntegrationOn = 0
            # (src/io/io.cc), so an isolated run's code units carry no h, and the time unit
            # is UnitLength / UnitVelocity (src/main/begrun.cc), as the header bar and the
            # formation-time plugin use it.
            unit_t = self._unit("UnitLength_in_cm", 3.085678e21) / self._unit("UnitVelocity_in_cm_per_s", 1e5)
            info["Gyr"] = t * unit_t / (1 * u.Gyr).to(u.s).value
        return info

    def softening(self, ptype: int) -> tuple:
        """(ε in kpc, physical frame, how) for a type from the Gadget-4 softening
        parameters, or (None, reason). Gadget-4: ε_phys = min(ε_com · a, ε_maxphys)
        in cosmological runs (code lengths per h), ε_com otherwise. ε is the
        Plummer-equivalent softening; the spline kernel is Newtonian beyond 2.8 ε."""
        p = self.params
        cls = p.get(f"SofteningClassOfPartType{ptype}")
        if cls is None:
            return None, "no softening parameters in the file"
        cls = int(cls)
        com = p.get(f"SofteningComovingClass{cls}")
        maxp = p.get(f"SofteningMaxPhysClass{cls}")
        if com is None:
            return None, f"SofteningComovingClass{cls} missing"
        com = float(com)
        length = self._unit("UnitLength_in_cm", 3.085678e21)
        if self.is_cosmological:
            eps_code = min(com * self.a, float(maxp) if maxp is not None else np.inf)
            return eps_code * length / self.h / KPC_CM, f"class {cls}: min({com:g}·a, {maxp}) code units, /h"
        return com * length / KPC_CM, f"class {cls}: {com:g} code units"

    # ── Fields ─────────────────────────────────────────────────────────────
    def has(self, ptype: int, name: str) -> bool:
        return bool(self.vset.get_dtype(f"PartType{ptype}/{name}"))

    def attrs(self, ptype: int, name: str) -> dict:
        return self.vset.get_attributes(f"PartType{ptype}/{name}")

    def conversion(self, attrs: dict, frame: str = "physical") -> float:
        """to_cgs × h^h_exp × a^a_exp, exactly as plugin Stage 2 computes it."""
        to_cgs, cosmo = self._conversion_parts(attrs, frame)
        return to_cgs * cosmo

    def _conversion_parts(self, attrs: dict, frame: str) -> tuple:
        to_cgs_raw = attrs.get("to_cgs") or attrs.get("CGSConversionFactor", 1.0)
        try:
            to_cgs = float(to_cgs_raw)
        except (TypeError, ValueError):
            to_cgs = 1.0
        h_exp, a_exp = scaling_exponents(attrs)
        if frame == "comoving":
            return to_cgs, apply_cosmological_factors(self.h, h_exp, 1.0, 0.0)
        return to_cgs, apply_cosmological_factors(self.h, h_exp, self.a, a_exp)

    def field_cgs(self, ptype: int, name: str, frame: str = "physical",
                  max_particles: int | None = None) -> np.ndarray:
        """Full array (or strided preview) of one dataset in CGS, as plugin Stage 2 gives it."""
        path = f"PartType{ptype}/{name}"
        raw = self.vset.extract_array(path, max_particles=max_particles).data
        to_cgs, cosmo = self._conversion_parts(self.vset.get_attributes(path), frame)
        return raw * to_cgs * cosmo          # same operation order as Stage 2: bit-identical

    def _first_present(self, ptype, names):
        for n in names:
            if self.has(ptype, n):
                return n
        return None

    def _read_ids(self, ptype: int, stride: int) -> np.ndarray | None:
        name = self._first_present(ptype, _ID_NAMES)
        if name is None:
            return None
        parts = []
        offset = 0
        for path in self.files:
            with h5py.File(path, "r") as f:
                g = f.get(f"PartType{ptype}")
                if g is None or name not in g:
                    continue
                ds = g[name]
                parts.append(ds[offset::stride])
                n = len(ds)
                offset = (stride - (n - offset) % stride) % stride if stride > 1 else 0
        return np.concatenate(parts) if parts else None

    def _masses(self, ptype: int, frame: str, n: int, mp, notes: list) -> np.ndarray:
        name = self._first_present(ptype, _MASS_NAMES)
        if name is not None:
            return self.field_cgs(ptype, name, frame, mp) / MSUN_G
        table = np.asarray(self.header.get("MassTable", [0.0] * 6), dtype=float)
        m_code = float(table[ptype]) if ptype < len(table) else 0.0
        if m_code <= 0:
            raise ValueError(f"PartType{ptype} has no mass dataset and MassTable[{ptype}] = 0")
        donor = next((t for t in self.types_present()
                      if self._first_present(t, _MASS_NAMES) is not None), None)
        if donor is not None:
            attrs = self.attrs(donor, self._first_present(donor, _MASS_NAMES))
            factor = self.conversion(attrs, frame)
            notes.append(f"PartType{ptype} masses from MassTable, converted with "
                         f"PartType{donor} mass attributes")
        else:
            attrs = {"to_cgs": self._unit("UnitMass_in_g", 1.989e43),
                     "h_scaling": -1.0 if self.is_cosmological else 0.0, "a_scaling": 0.0}
            factor = self.conversion(attrs, frame)
            notes.append(f"PartType{ptype} masses from MassTable, converted with "
                         f"UnitMass_in_g{' and h^-1 (Gadget convention)' if self.is_cosmological else ''}")
        return np.full(n, m_code * factor / MSUN_G)

    def particles(self, ptypes=None, frame: str = "physical", velocities: bool = True,
                  ids: bool = True, max_particles: int | None = None) -> ParticleSet:
        """Positions, masses and (optionally) velocities and IDs of the given types.

        max_particles=None reads every particle (science). A number gives a
        strided preview per type; the ParticleSet then records n_used < n_total.
        """
        if frame not in ("physical", "comoving"):
            raise ValueError("frame must be 'physical' or 'comoving'")
        ptypes = self.types_present() if ptypes is None else [int(t) for t in ptypes]
        pos_l, vel_l, m_l, id_l, t_l, notes = [], [], [], [], [], []
        n_total = 0
        for t in ptypes:
            n_t = self.count(t)
            if n_t == 0:
                continue
            n_total += n_t
            stride = 1 if (max_particles is None or max_particles >= n_t) else max(1, n_t // max_particles)
            mp = None if stride == 1 else max_particles
            pos = self.field_cgs(t, "Coordinates", frame, mp) / KPC_CM
            pos_l.append(pos)
            n = len(pos)
            m_l.append(self._masses(t, frame, n, mp, notes))
            if velocities:
                vn = self._first_present(t, _VEL_NAMES)
                vel_l.append(self.field_cgs(t, vn, "physical", mp) / KMS_CMS if vn
                             else np.full((n, 3), np.nan))
            if ids:
                i = self._read_ids(t, stride)
                id_l.append(i if i is not None else np.full(n, -1, dtype=np.int64))
            t_l.append(np.full(n, t, dtype=np.int8))
        if not pos_l:
            raise ValueError(f"No particles of types {ptypes} in this snapshot")
        if self.is_cosmological and velocities:
            notes.append("velocities are physical peculiar velocities (no Hubble flow)"
                         + (", also in the comoving frame" if frame == "comoving" else ""))
        return ParticleSet(
            pos=np.concatenate(pos_l), mass=np.concatenate(m_l),
            vel=np.concatenate(vel_l) if velocities else None,
            ids=np.concatenate(id_l) if ids else None,
            ptype=np.concatenate(t_l), box=self.box_size(frame), frame=frame,
            n_total=n_total, files=self.files, particle_types=ptypes,
            time=self.time_info(), notes=notes)
