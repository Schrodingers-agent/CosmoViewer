"""
phase.py — property-vs-property histograms (the counterpart of GadgetViewer's "Make plot").

property_histogram2d() bins every particle of one type by two fields, weighted
by mass or by count, from the full arrays (never a display sample). The
weights are accounted for exactly: in range + outside the range + dropped
(non-finite, or ≤ 0 on a log axis) = the total, and the result says how much
fell in each.

Fields are dataset names as stored (in CGS after unit conversion, exactly as
the Explorer's Histogram tab gives them), "Name[x]" / "Name[y]" / "Name[z]" for
one component of a vector, "Name|·|" for its magnitude, or "Temperature",
derived from InternalEnergy as T = (γ − 1) μ m_p u / k_B with μ from
ElectronAbundance when stored (the Explorer's InternalEnergy plugin formula).

Immersive Mode's Make plot draws the display sample and each selection set at
full resolution; here the whole type is used and a set (by IDs) restricts it.
"""
from __future__ import annotations

import numpy as np

from .results import Provenance, Result
from .snapshot_view import SnapshotView

VERSION = "1"
_COMP = {"[x]": 0, "[y]": 1, "[z]": 2}


def field_values(view: SnapshotView, ptype: int, name: str, frame: str = "physical"):
    """(values in CGS, label) for one field of one type, every particle."""
    if name == "Temperature":
        from astro_units import BOLTZMANN_CGS, GAMMA_MONOATOMIC, MU_IONISED_PRIMORDIAL, PROTON_MASS_CGS
        from internal_energy_plugin import mean_molecular_weight
        u = view.field_cgs(ptype, "InternalEnergy", frame)
        if view.has(ptype, "ElectronAbundance"):
            ne = view.vset.extract_array(f"PartType{ptype}/ElectronAbundance").data
            xh = (view.vset.extract_array(f"PartType{ptype}/ElementAbundance/Hydrogen").data
                  if view.has(ptype, "ElementAbundance/Hydrogen") else 0.76)
            mu, how = mean_molecular_weight(np.asarray(ne, float), xh), "μ from ElectronAbundance"
        else:
            mu, how = MU_IONISED_PRIMORDIAL, "μ = 0.588 assumed (fully ionised)"
        return (GAMMA_MONOATOMIC - 1.0) * mu * PROTON_MASS_CGS * u / BOLTZMANN_CGS, f"T [K] ({how})"
    for suffix, k in _COMP.items():
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            return view.field_cgs(ptype, base, frame)[:, k], f"{name} [cgs]"
    if name.endswith("|·|"):
        v = view.field_cgs(ptype, name[:-3], frame)
        return np.sqrt(np.einsum("ij,ij->i", v, v)), f"{name} [cgs]"
    a = view.field_cgs(ptype, name, frame)
    if a.ndim > 1:
        raise ValueError(f"{name} is a vector: use {name}[x], {name}[y], {name}[z] or {name}|·|")
    return a, f"{name} [cgs]"


def property_histogram2d(snapshot, ptype: int, x: str, y: str, *, weight: str = "mass",
                         bins: int = 100, log_x: bool = True, log_y: bool = True,
                         x_range=None, y_range=None, frame: str = "physical",
                         ids=None, ids_label: str = "") -> Result:
    view = snapshot if isinstance(snapshot, SnapshotView) else SnapshotView(snapshot)
    p = view.particles([ptype], frame=frame, velocities=False, ids=ids is not None)
    xv, xlab = field_values(view, ptype, x, frame)
    yv, ylab = field_values(view, ptype, y, frame)
    w = p.mass if weight == "mass" else np.ones(len(p.mass))
    notes = list(p.notes)
    keep = np.ones(len(w), dtype=bool)
    if ids is not None:
        keep = np.isin(np.asarray(p.ids).astype(np.uint64), np.asarray(ids, dtype=np.uint64))
        notes.append(f"restricted to {ids_label or 'a set:'} {int(keep.sum()):,} particles")
    xv, yv, w = xv[keep], yv[keep], w[keep]
    total = float(w.sum())

    def tr(v, log):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, np.log10(v), np.nan) if log else v.astype(float)

    tx, ty = tr(xv, log_x), tr(yv, log_y)
    ok = np.isfinite(tx) & np.isfinite(ty)
    if not ok.any():
        raise ValueError("no particle has finite (and, on log axes, positive) values of both fields")
    xr = x_range or (float(tx[ok].min()), float(tx[ok].max()))
    yr = y_range or (float(ty[ok].min()), float(ty[ok].max()))
    xe = np.linspace(xr[0], xr[1], bins + 1)
    ye = np.linspace(yr[0], yr[1], bins + 1)
    grid, _, _ = np.histogram2d(tx[ok], ty[ok], bins=[xe, ye], weights=w[ok])
    # histogram2d includes the last edge, so "inside" is the closed range
    within = ok & (tx >= xe[0]) & (tx <= xe[-1]) & (ty >= ye[0]) & (ty <= ye[-1])
    inside = float(w[within].sum())
    dropped = float(w[~ok].sum())
    unit = "Msun" if weight == "mass" else "particles"
    values = {"x_edges": xe, "y_edges": ye, "grid": grid.T,      # grid[iy, ix]
              "total": total, "in_range": inside, "out_of_range": float(w[ok & ~within].sum()),
              "dropped": dropped, "n_dropped": int((~ok).sum())}
    notes.append(f"weights in {unit}; axes are log10 of the field where log is set")
    prov = Provenance(
        measurement="property_histogram2d", version=VERSION, files=list(view.files),
        particle_types=[ptype], n_used=int(keep.sum()), n_total=int(keep.sum()), frame=frame,
        time=view.time_info(),
        parameters={"x": x, "y": y, "x_label": xlab, "y_label": ylab, "weight": weight,
                    "bins": bins, "log_x": log_x, "log_y": log_y,
                    "x_range": list(xr), "y_range": list(yr)},
        notes=notes)
    return Result(values=values, provenance=prov)
