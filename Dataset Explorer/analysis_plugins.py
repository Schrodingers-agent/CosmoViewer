"""
analysis_plugins.py — tree leaves under "Analyses".

An analysis leaf behaves like a dataset leaf: click it, read it in the tabs,
export it. Underneath it does not read one HDF5 dataset; it calls the
headless measurement layer (measurements/) on the open snapshot, so the
window and a script give the same numbers.

The class subclasses BasePlugin only to fit the existing tree / worker /
renderer contract. It overrides compute() (as DiagnosticPlugin does),
declares REQUIRES = [] so the worker reads nothing itself, and sets
COMPUTES_MATRIX so the Matrix tab gets its table from compute() too.

Science results need every particle, so an analysis never samples. A
selection too large to hold in memory is refused with a message instead.
"""
from __future__ import annotations

import os

import numpy as np

from plugin_base import BasePlugin, CurveSpec, MarkerSpec, PlotRequest

# Particles one analysis may load at once. Positions and masses in float64
# plus a sort are about 50 bytes per particle: 50M particles ≈ 2.5 GB.
MAX_ANALYSIS_PARTICLES = 50_000_000

CENTRE_LABELS = {
    "shrinking_sphere": "shrinking sphere",
    "density": "density-weighted (Casertano & Hut)",
    "com": "centre of mass",
}
DEFAULT_CENTRE = "shrinking_sphere"
PROFILE_POINTS = 2000          # points drawn on the cumulative-mass curve


class AnalysisPlugin(BasePlugin):
    GROUP = "Analyses"
    REQUIRES: list = []
    COMPUTES_MATRIX = True
    PTYPES: tuple | None = None          # None = every type present

    @classmethod
    def available(cls, vset) -> bool:
        """Whether the open snapshot can supply this analysis."""
        types = cls._types(vset)
        return bool(types) and all(
            vset.get_dtype(f"PartType{t}/Coordinates") for t in types)

    @classmethod
    def _types(cls, vset) -> list:
        present = [t for t in range(6) if vset.particle_counts.get(f"PartType{t}", 0) > 0]
        if cls.PTYPES is None:
            return present
        return [t for t in cls.PTYPES if t in present]

    def can_render_from_cache(self, tab: str) -> bool:
        return False        # compute() keeps its own result cache


class LagrangianRadiiAnalysis(AnalysisPlugin):
    TABS = ["Profile"]
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "linear"

    def __init__(self) -> None:
        super().__init__()
        self._result_key = None
        self._result = None

    def _measure(self, vset, centre_method: str, frame: str, selection=None):
        from measurements import SnapshotView, find_centre, lagrangian_radii
        from measurements.centre import minimum_image

        key = (tuple(vset.files), self.PTYPES, centre_method, frame,
               None if selection is None else (selection["path"], selection.get("mtime")))
        if key == self._result_key:
            return self._result
        view = SnapshotView(vset)
        types = self._types(vset)
        n = sum(view.count(t) for t in types)
        if n > MAX_ANALYSIS_PARTICLES:
            raise MemoryError(
                f"{n:,} particles: Lagrangian radii need every particle, and more than "
                f"{MAX_ANALYSIS_PARTICLES:,} will not fit in memory here. "
                "Choose one particle type; a streaming version is planned.")
        p = view.particles(types, frame=frame, velocities=False, ids=selection is not None)
        if selection is not None:
            p = restrict_to_set(p, selection)
        c = find_centre(p, centre_method)
        res = lagrangian_radii(p, c)
        d = minimum_image(p.pos - c.pos, p.box)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        order = np.argsort(r, kind="stable")
        r_s = r[order]
        f_s = np.cumsum(p.mass[order]) / p.mass.sum()
        pick = np.unique(np.linspace(0, len(r_s) - 1, PROFILE_POINTS).astype(int))
        self._result_key = key
        self._result = (res, c, r_s[pick], f_s[pick])
        return self._result

    def compute(self, data, attrs, header, params, tab, scale_x, scale_y, window, context=None):
        context = context or {}
        vset = params.get("_vset")
        if vset is None:
            return self.error_request("No snapshot open.")
        frame = "comoving" if context.get("coordinates") == "Comoving" else "physical"
        method = context.get("centre", DEFAULT_CENTRE)
        try:
            res, c, r_curve, f_curve = self._measure(vset, method, frame, context.get("selection_set"))
        except (MemoryError, ValueError) as exc:
            return self.error_request(str(exc))

        sel = context.get("selection_set")
        prov = res.provenance
        radii = res["radius"].value
        err = res["radius_err_sampling"].value
        fr = res["fraction"]
        warn = c.info.get("warning", "")
        notes = list(prov.notes) + ([warn] if warn else [])
        title = f"Lagrangian radii, {self.DISPLAY_NAME.split(', ', 1)[-1]}"
        if sel is not None:
            title += f", set '{sel['name']}'"
        status = (prov.status_line() + set_suffix(sel)
                  + (" · ⚠ centre warning, see info bar" if warn else ""))
        info = (f"Centre ({CENTRE_LABELS.get(method, method)}): "
                f"({c.pos[0]:.4g}, {c.pos[1]:.4g}, {c.pos[2]:.4g}) kpc {frame}"
                + (f"  |  ⚠ {warn}" if warn else "")
                + f"  |  M = {res['mass_total'].value:.4e} Msun")

        if tab == "Matrix":
            from compute_worker import _MatrixRequest
            table = np.column_stack([fr, radii, err])
            header_notes = [f"# {title}; {prov.status_line()}",
                            f"# centre: {CENTRE_LABELS.get(method, method)} at "
                            f"{', '.join(f'{x:.8e}' for x in c.pos)} kpc ({frame})",
                            f"# files: {', '.join(prov.files)}",
                            "# errors are binomial sampling errors; they exclude centre uncertainty"]
            header_notes += [f"# note: {n}" for n in notes]
            return _MatrixRequest(
                {f"{self.GROUP}/{self.DATASET}": table}, self,
                labels=["mass fraction", "radius [kpc]", "sampling error [kpc]"],
                notes=header_notes, info=info, status=status)

        markers = [MarkerSpec(pos=float(rv), axis="x", label=f"{100 * fv:g}%",
                              color="#ffb703") for fv, rv in zip(fr, radii)]
        return PlotRequest(
            title=title, x_label=f"r [kpc, {frame}]", y_label="enclosed mass fraction",
            curves=[CurveSpec(x=r_curve, y=f_curve, color="#00b4d8",
                              label="enclosed mass fraction")],
            markers=markers, log_x=(scale_x == "log"), log_y=(scale_y == "log"),
            status=status, info=info, tab=tab, n_points=prov.n_used,
            dim_label="D0", unit_dimension="dimensionless")


def set_suffix(sel) -> str:
    """The end of an analysis status line: which set it used, and "⚠ incomplete set" when that
    set is known to hold only part of what it describes (its warning is in the notes)."""
    if sel is None:
        return ""
    return f" · set '{sel['name']}'" + (" · ⚠ incomplete set" if sel.get("complete") is False else "")


def restrict_to_set(p, selection: dict):
    """The particles of ``p`` whose IDs are in the selection set; the result describes the whole set
    (n_total = particles of the set found in these types), so it is still a science result."""
    ids = np.asarray(selection["ids"], dtype=np.uint64)
    keep = np.isin(np.asarray(p.ids).astype(np.uint64), ids)
    if not keep.any():
        raise ValueError(f"None of the {len(ids):,} particles of set '{selection['name']}' "
                         "are of the particle types this analysis uses.")
    sub = p.subset(keep)
    sub.notes.append(f"restricted to set '{selection['name']}': {int(keep.sum()):,} of its "
                     f"{len(ids):,} IDs are in PartType {', '.join(map(str, p.particle_types))} here "
                     f"({os.path.basename(selection.get('path') or '') or 'not saved'})")
    sub.notes.extend(set_notes(selection))
    return sub


def set_notes(selection) -> list:
    """What a set is and, when it is known to be incomplete, its warning: the notes every analysis
    of a set carries. restrict_to_set() adds them; analyses that pass the set's IDs to the
    measurement themselves (energy budget, energy over run, phase diagram) add them too."""
    if not selection:
        return []
    out = []
    if selection.get("describe"):
        out.append(f"set '{selection['name']}' is: {selection['describe']}")
    if selection.get("warning"):
        out.append(f"WARNING: {selection['warning']}")
    return out


def _make(label: str, ptypes) -> type:
    name = "LagrangianRadii_" + ("all" if ptypes is None else "_".join(map(str, ptypes)))
    return type(name, (LagrangianRadiiAnalysis,), {
        "DATASET": label, "DISPLAY_NAME": label, "PTYPES": ptypes,
        "TREE_CONTRIBUTIONS": {"Analyses": [label]}})


ANALYSES: dict = {}
for _pt in (None, 0, 1, 2, 3, 4, 5):
    # short enough for the sidebar; the plot title and tooltip give the full name
    _label = "Lagr. radii, all types" if _pt is None else f"Lagr. radii, PartType{_pt}"
    ANALYSES[("Analyses", _label)] = _make(_label, None if _pt is None else (_pt,))


def available_analysis_leaves(vset) -> list:
    """Analysis leaves the open snapshot can supply (the others stay greyed out)."""
    return [leaf for (_, leaf), cls in ANALYSES.items() if cls.available(vset)]
