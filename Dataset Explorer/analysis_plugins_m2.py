"""
analysis_plugins_m2.py — the second set of leaves under "Analyses":

  Sanity report            one pass over every particle, PASS/WARN/FAIL per check
  Profile, <types>         density, enclosed mass, dispersions and anisotropy
  Energy, all types        K, U, W, E, virial ratio, momentum and angular momentum
  Energy over run          energy drift, virial ratio and angular momentum over the
                           snapshots in the open file's folder (cached)
  Phase diagram, gas       mass-weighted ρ–T histogram of every gas particle

Same contract as analysis_plugins.py: each leaf calls the headless measurement
layer on the open snapshot, so the window and a script give the same numbers;
an analysis never samples particles; the Options ▸ Centre choice and the active
selection set (context["selection_set"]) apply where a centre or particle set
is involved. The sanity report checks the file itself, so a set does not
restrict it.

Text tables (the sanity report, the energy budget) go to the Matrix view as
`cells`; numeric tables as arrays with column labels, as in analysis_plugins.py.
"""
from __future__ import annotations

import numpy as np

from analysis_plugins import (CENTRE_LABELS, DEFAULT_CENTRE, MAX_ANALYSIS_PARTICLES,
                              AnalysisPlugin, restrict_to_set, set_notes, set_suffix)
from plugin_base import CurveSpec, ImageSpec, PlotRequest, RegionSpec

ORANGE, TEAL, GREY = "#ffb703", "#00b4d8", "#8d99ae"


def _matrix(plugin, table, labels, notes, info, status, cells=None):
    from compute_worker import _MatrixRequest
    req = _MatrixRequest({f"{plugin.GROUP}/{plugin.DATASET}": table}, plugin,
                         labels=labels, notes=notes, info=info, status=status)
    if cells is not None:
        try:
            req.cells = cells
        except AttributeError:       # an older _MatrixRequest without the slot
            pass
    return req


def _with_script(req, kind, plugin, vset, types, frame="physical", centre=DEFAULT_CENTRE,
                 sel=None, **kw):
    """Attach the Python that reproduces this result (Export ▸ .py)."""
    from measurements.scripts import script_for
    try:
        req.script = script_for(kind, title=f"{plugin.GROUP}/{plugin.DATASET}",
                                snapshot=vset.files[0], types=types, frame=frame,
                                centre=centre, selection=None if sel is None else sel["path"], **kw)
    except AttributeError:
        pass
    return req


def _set_suffix(sel):
    return set_suffix(sel)


def _check_size(n, what):
    if n > MAX_ANALYSIS_PARTICLES:
        raise MemoryError(
            f"{n:,} particles: {what} need every particle, and more than "
            f"{MAX_ANALYSIS_PARTICLES:,} will not fit in memory here. Choose one particle type.")


# ── Sanity report ──────────────────────────────────────────────────────────

class SanityReportAnalysis(AnalysisPlugin):
    DATASET = DISPLAY_NAME = "Sanity report"
    TREE_CONTRIBUTIONS = {"Analyses": ["Sanity report"]}
    TABS = ["Report"]
    PTYPES = None

    def __init__(self) -> None:
        super().__init__()
        self._key = self._report = None

    def compute(self, data, attrs, header, params, tab, scale_x, scale_y, window, context=None):
        from measurements.sanity import sanity_report
        vset = params.get("_vset")
        if vset is None:
            return self.error_request("No snapshot open.")
        key = tuple(vset.files)
        if key != self._key:
            try:
                self._report = sanity_report(vset)
            except (MemoryError, ValueError, OSError) as exc:
                return self.error_request(str(exc))
            self._key = key
        r = self._report
        cells = [[c.status, c.check, c.scope, f"{c.count:,}" if c.count else "",
                  c.detail + (f" (e.g. {', '.join(map(str, c.examples))})" if c.examples else "")]
                 for c in r.checks]
        counts = np.array([c.count for c in r.checks], dtype=float)
        notes = [f"# {r.summary()}", f"# files: {', '.join(r.provenance.files)}",
                 "# columns: status, check, scope, particles affected, detail"]
        status = r.summary() + ("" if context is None or context.get("selection_set") is None
                                else " · checks the whole file, not the set")
        n = r.counts()
        info = (f"{n['FAIL']} FAIL · {n['WARN']} WARN · {n['PASS']} PASS · {n['INFO']} INFO  |  "
                "raw stored values, every particle")
        req = _matrix(self, counts, ["status", "check", "scope", "particles", "detail"],
                      notes, info, status, cells)
        return _with_script(req, "sanity", self, vset, r.provenance.particle_types)


# ── Radial profiles ────────────────────────────────────────────────────────

class RadialProfileAnalysis(AnalysisPlugin):
    TABS = ["Density", "Enclosed mass", "Dispersion", "Anisotropy"]
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"
    N_BINS = 30

    def __init__(self) -> None:
        super().__init__()
        self._key = self._result = None

    def _measure(self, vset, method, frame, sel):
        from measurements import SnapshotView, find_centre, lagrangian_radii
        from measurements.profiles import radial_profile
        key = (tuple(vset.files), self.PTYPES, method, frame,
               None if sel is None else (sel["path"], sel.get("mtime")))
        if key == self._key:
            return self._result
        view = SnapshotView(vset)
        types = self._types(vset)
        _check_size(sum(view.count(t) for t in types), "profiles")
        p = view.particles(types, frame=frame, velocities=True, ids=sel is not None)
        if sel is not None:
            p = restrict_to_set(p, sel)
        if p.vel is not None and np.isnan(p.vel).any():
            p.vel = None
        c = find_centre(p, method)
        # centre velocity: mass-weighted mean inside the radius holding 10 % of the mass
        r10 = float(lagrangian_radii(p, c, [0.10])["radius"].value[0])
        c = find_centre(p, method, velocity_radius=r10) if p.vel is not None else c
        soft = [view.softening(t)[0] for t in types]
        soft = max((s for s in soft if s), default=None)
        if frame == "comoving" and soft:
            soft = soft / view.a            # resolution limit in the same frame as the radii
        res = radial_profile(p, c, n_bins=self.N_BINS, softening=soft)
        self._key, self._result = key, (res, c)
        return self._result

    def compute(self, data, attrs, header, params, tab, scale_x, scale_y, window, context=None):
        context = context or {}
        vset = params.get("_vset")
        if vset is None:
            return self.error_request("No snapshot open.")
        frame = "comoving" if context.get("coordinates") == "Comoving" else "physical"
        method = context.get("centre", DEFAULT_CENTRE)
        sel = context.get("selection_set")
        try:
            res, c = self._measure(vset, method, frame, sel)
        except (MemoryError, ValueError) as exc:
            return self.error_request(str(exc))
        prov = res.provenance
        what = self.DISPLAY_NAME.split(", ", 1)[-1]
        title = f"Radial profile, {what}" + ("" if sel is None else f", set '{sel['name']}'")
        warn = c.info.get("warning", "")
        status = prov.status_line() + _set_suffix(sel) + (" · ⚠ centre warning" if warn else "")
        r_res = float(res["r_resolved"].value)
        info = (f"Centre ({CENTRE_LABELS.get(method, method)}): "
                f"({c.pos[0]:.4g}, {c.pos[1]:.4g}, {c.pos[2]:.4g}) kpc {frame}"
                + (f"  |  resolved beyond {r_res:.3g} kpc (2.8 ε)" if r_res > 0 else "")
                + (f"  |  ⚠ {warn}" if warn else ""))
        r_lo, r_hi = res["r_lo"].value, res["r_hi"].value
        r_mid = np.where(np.isfinite(res["r_mean"].value), res["r_mean"].value,
                         np.sqrt(r_lo * r_hi))
        cols = [
            ("r_lo [kpc]", r_lo), ("r_hi [kpc]", r_hi), ("r_mean [kpc]", res["r_mean"].value),
            ("N", res["n"].astype(float)), ("density [Msun/kpc3]", res["density"].value),
            ("density err", res["density_err"].value),
            ("M(<r_hi) [Msun]", res["mass_enclosed"].value), ("v_circ [km/s]", res["v_circ"].value),
            ("sigma_r [km/s]", res["sigma_r"].value), ("sigma_r err", res["sigma_r_err"].value),
            ("sigma_t [km/s]", res["sigma_t"].value), ("sigma_t err", res["sigma_t_err"].value),
            ("beta", res["beta"]), ("beta err", res["beta_err"]),
            ("resolved", res["resolved"].astype(float))]
        if tab == "Matrix":
            notes = [f"# {title}; {prov.status_line()}",
                     f"# centre: {CENTRE_LABELS.get(method, method)} at "
                     f"{', '.join(f'{x:.8e}' for x in c.pos)} kpc ({frame})",
                     f"# files: {', '.join(prov.files)}"] + [f"# note: {n}" for n in prov.notes]
            return _with_script(
                _matrix(self, np.column_stack([v for _, v in cols]), [k for k, _ in cols],
                        notes, info, status),
                "profile", self, vset, self._types(vset), frame, method, sel, n_bins=self.N_BINS)

        regions = [RegionSpec(x_min=float(r_lo[0]), x_max=r_res, color="#8d99ae30")] \
            if r_res > r_lo[0] else []

        def band(y, e, color, label):
            return [CurveSpec(x=r_mid, y=y, color=color, label=label),
                    CurveSpec(x=r_mid, y=y + e, color=color, width=0.8, dash=True),
                    CurveSpec(x=r_mid, y=y - e, color=color, width=0.8, dash=True)]

        if tab == "Enclosed mass":
            curves = [CurveSpec(x=r_hi, y=res["mass_enclosed"].value, color=TEAL,
                                label="M(<r)")]
            ylab, dim = "M(<r) [Msun]", "mass"
        elif tab == "Dispersion":
            curves = (band(res["sigma_r"].value, res["sigma_r_err"].value, TEAL, "σ_r")
                      + band(res["sigma_t"].value / np.sqrt(2), res["sigma_t_err"].value / np.sqrt(2),
                             ORANGE, "σ_t / √2")
                      + [CurveSpec(x=r_hi, y=res["v_circ"].value, color=GREY, label="v_circ",
                                   dash=True)])
            ylab, dim = "velocity [km/s]", "velocity"
        elif tab == "Anisotropy":
            curves = band(res["beta"], res["beta_err"], TEAL, "β")
            ylab, dim = "β = 1 − (σ_θ² + σ_φ²) / 2σ_r²", "dimensionless"
            scale_y = "linear"
        else:
            curves = band(res["density"].value, res["density_err"].value, TEAL, "ρ(r)")
            ylab, dim = "ρ [Msun / kpc³]", "density"
        req = PlotRequest(
            title=title, x_label=f"r [kpc, {frame}]", y_label=ylab, curves=curves,
            regions=regions, log_x=(scale_x == "log"), log_y=(scale_y == "log"),
            status=status, info=info, tab=tab, n_points=prov.n_used, dim_label="D0",
            unit_dimension=dim)
        return _with_script(req, "profile", self, vset, self._types(vset), frame, method, sel,
                            n_bins=self.N_BINS)


# ── Energy budget ──────────────────────────────────────────────────────────

class EnergyBudgetAnalysis(AnalysisPlugin):
    DATASET = DISPLAY_NAME = "Energy, all types"
    TREE_CONTRIBUTIONS = {"Analyses": ["Energy, all types"]}
    TABS = ["Budget"]
    PTYPES = None

    def __init__(self) -> None:
        super().__init__()
        self._key = self._result = None

    def compute(self, data, attrs, header, params, tab, scale_x, scale_y, window, context=None):
        from measurements import SnapshotView
        from measurements.energy import energy_budget
        context = context or {}
        vset = params.get("_vset")
        if vset is None:
            return self.error_request("No snapshot open.")
        sel = context.get("selection_set")
        key = (tuple(vset.files), None if sel is None else (sel["path"], sel.get("mtime")))
        if key != self._key:
            view = SnapshotView(vset)
            try:
                _check_size(sum(view.count(t) for t in self._types(vset)), "energies")
                self._result = energy_budget(
                    view, self._types(vset), ids=None if sel is None else sel["ids"],
                    ids_label="" if sel is None else f"set '{sel['name']}':")
            except (MemoryError, ValueError) as exc:
                return self.error_request(str(exc))
            self._key = key
        r = self._result
        prov = r.provenance

        def q(name):
            v = r[name]
            return np.asarray(getattr(v, "value", v), dtype=float)

        e_unit = "Msun (km/s)^2"
        w_err = float(q("W_err"))
        rows = [("K, kinetic", float(q("K")), 0.0, e_unit),
                ("U, thermal", float(q("U")), 0.0, e_unit),
                ("W, potential", float(q("W")), w_err, e_unit),
                ("E = K + U + W", float(q("E")), w_err, e_unit),
                ("Q = K / |W|", float(q("Q")), float(q("Q_err")), ""),
                ("|P|", float(np.linalg.norm(q("P"))), 0.0, "Msun km/s"),
                ("|L| about origin", float(np.linalg.norm(q("L"))), 0.0, "Msun kpc km/s"),
                ("total mass", float(q("mass")), 0.0, "Msun")]
        rows += [(f"centre of mass {a}", float(x), 0.0, "kpc") for a, x in zip("xyz", q("com_pos"))]
        rows += [(f"centre-of-mass velocity {a}", float(x), 0.0, "km/s")
                 for a, x in zip("xyz", q("com_vel"))]
        for t, k, uu in zip(r["types"], q("K_type"), q("U_type")):
            rows.append((f"K, PartType{int(t)}", float(k), 0.0, e_unit))
            if uu:
                rows.append((f"U, PartType{int(t)}", float(uu), 0.0, e_unit))
        from plot_panel import _format_cell
        cells = [[name, _format_cell(v, "Auto"), _format_cell(e, "Auto") if e else "", unit]
                 for name, v, e, unit in rows]
        method = prov.parameters.get("potential")
        status = prov.status_line() + _set_suffix(sel) + f" · W: {method}"
        info = (f"Q = {float(q('Q')):.4f} ± {float(q('Q_err')):.4f}  |  W by {method}"
                if np.isfinite(q("Q")) else f"W not computed ({method})")
        notes = [f"# Energy budget; {prov.status_line()}", f"# files: {', '.join(prov.files)}",
                 "# columns: quantity, value, error, unit"] + [f"# note: {n}" for n in prov.notes] + \
            [f"# note: {n}" for n in set_notes(sel)]
        table = np.array([[v, e] for _, v, e, _ in rows], dtype=float)
        req = _matrix(self, table, ["quantity", "value", "error", "unit"], notes, info,
                      status, cells)
        return _with_script(req, "energy", self, vset, self._types(vset), sel=sel)


# ── Energy over a run ──────────────────────────────────────────────────────

MAX_RUN_SNAPSHOTS = 60      # more are thinned to this many, evenly spaced, first and last kept


def _run_snapshots(vset):
    import os
    from measurements.runset import discover_snapshots
    return discover_snapshots(os.path.dirname(os.path.abspath(vset.files[0])))


def pick_snapshots(numbers, limit=MAX_RUN_SNAPSHOTS):
    numbers = list(numbers)
    if len(numbers) <= limit:
        return numbers
    k = np.unique(np.linspace(0, len(numbers) - 1, limit).round().astype(int))
    return [numbers[i] for i in k]


class EnergyOverRunAnalysis(AnalysisPlugin):
    DATASET = DISPLAY_NAME = "Energy over run"
    TREE_CONTRIBUTIONS = {"Analyses": ["Energy over run"]}
    TABS = ["Energy drift", "Virial ratio", "Angular momentum"]
    DEFAULT_SCALE_X = "linear"
    DEFAULT_SCALE_Y = "linear"
    PTYPES = None

    def __init__(self) -> None:
        super().__init__()
        self._key = self._result = None

    @classmethod
    def available(cls, vset) -> bool:
        return super().available(vset) and len(_run_snapshots(vset)) >= 2

    def _measure(self, vset, sel):
        import os
        from measurements.cache import ResultCache
        from measurements.energy import conservation
        from measurements.runset import Run
        snaps = _run_snapshots(vset)
        nums = pick_snapshots([n for n, _ in snaps])
        key = (tuple(p for _, p in snaps), tuple(nums),
               None if sel is None else (sel["path"], sel.get("mtime")))
        if key == self._key:
            return self._result
        folder = os.path.dirname(os.path.abspath(vset.files[0]))
        run = Run(name=os.path.basename(folder), folder=folder, snapshots=snaps)
        kw = {} if sel is None else {"ids": sel["ids"], "ids_label": f"set '{sel['name']}':"}
        table, drift = conservation(run, snapshots=set(nums), types=self._types(vset),
                                    cache=ResultCache(), **kw)
        if not drift:
            raise ValueError("no snapshot of this run could be measured: "
                             + "; ".join(f["error"] for f in table.failures[:3]))
        self._key, self._result = key, (table, next(iter(drift.values())), len(snaps), nums)
        return self._result

    def compute(self, data, attrs, header, params, tab, scale_x, scale_y, window, context=None):
        context = context or {}
        vset = params.get("_vset")
        if vset is None:
            return self.error_request("No snapshot open.")
        sel = context.get("selection_set")
        try:
            table, d, n_all, nums = self._measure(vset, sel)
        except (MemoryError, ValueError) as exc:
            return self.error_request(str(exc))
        n_done = len(d["snapshot"])
        thinned = "" if n_all == len(nums) else f" (evenly spaced from {n_all:,})"
        status = (f"{n_done} snapshots{thinned} · every particle in each"
                  + _set_suffix(sel)
                  + (f" · {len(table.failures)} failed, see info bar" if table.failures else ""))
        t = d["time_code"]
        sci = all(r["is_science"] for r in table.rows)
        if np.isfinite(d["dE_rel"]).any():
            energy_note = (f"max |ΔE/E₀| = {np.nanmax(np.abs(d['dE_rel'])):.3e}  |  "
                           f"W sampling error per snapshot ≤ {np.nanmax(d['W_err_rel_E0']):.1e} of |E₀| "
                           "(while the particles stay the same, the same pairs are used and the drift "
                           "error is smaller)")
        else:   # no W: a periodic box (needs Ewald summation), so no E and no Q to draw
            energy_note = ("no potential energy W for this run (a periodic box needs Ewald "
                           "summation), so the energy drift and virial ratio cannot be drawn; "
                           "angular momentum is")
        info = (energy_note
                + ("  |  gas present: cooling and star formation change E physically"
                   if 0 in self._types(vset) else "")
                + ("" if sci else "  |  ⚠ some rows are previews")
                + ("" if not table.failures else "  |  failed: " + ", ".join(
                    str(f["snapshot"]) for f in table.failures)))
        title = "Energy over run" + ("" if sel is None else f", set '{sel['name']}'")
        if tab == "Matrix":
            cols = [("snapshot", d["snapshot"].astype(float)), ("time [code]", t),
                    ("E", d["E"]), ("dE/|E0|", d["dE_rel"]), ("W err/|E0|", d["W_err_rel_E0"]),
                    ("Q", d["Q"]), ("Q err", d["Q_err"]), ("|dL|/|L0|", d["dL_rel"]),
                    ("COM shift [kpc]", d["com_shift_kpc"])]
            notes = [f"# {title}; {status}", "# energies in Msun (km/s)^2",
                     f"# run folder: {table.rows[0]['run'] if table.rows else ''}"] + \
                [f"# note: {n}" for n in set_notes(sel)]
            return _matrix(self, np.column_stack([v for _, v in cols]), [k for k, _ in cols],
                           notes, info, status)
        if tab == "Virial ratio":
            curves = [CurveSpec(x=t, y=d["Q"], color=TEAL, label="Q = K/|W|"),
                      CurveSpec(x=t, y=d["Q"] + d["Q_err"], color=TEAL, width=0.8, dash=True),
                      CurveSpec(x=t, y=d["Q"] - d["Q_err"], color=TEAL, width=0.8, dash=True),
                      CurveSpec(x=t, y=np.full_like(t, 0.5), color=GREY, dash=True,
                                label="equilibrium 0.5")]
            ylab = "Q = K / |W|"
        elif tab == "Angular momentum":
            curves = [CurveSpec(x=t, y=d["dL_rel"], color=TEAL, label="|ΔL| / |L₀|")]
            ylab = "|L − L₀| / |L₀|"
        else:
            e = d["W_err_rel_E0"]
            curves = [CurveSpec(x=t, y=d["dE_rel"], color=TEAL, label="(E − E₀)/|E₀|"),
                      CurveSpec(x=t, y=d["dE_rel"] + e, color=TEAL, width=0.8, dash=True),
                      CurveSpec(x=t, y=d["dE_rel"] - e, color=TEAL, width=0.8, dash=True)]
            ylab = "(E − E₀) / |E₀|"
        return PlotRequest(
            title=title, x_label="time [code units]", y_label=ylab, curves=curves,
            log_x=False, log_y=(scale_y == "log" and tab == "Angular momentum"),
            status=status, info=info, tab=tab, n_points=n_done, dim_label="D0",
            unit_dimension="dimensionless")


# ── Phase diagram ──────────────────────────────────────────────────────────

class PhaseDiagramAnalysis(AnalysisPlugin):
    """Mass-weighted ρ–T diagram of the gas, every particle (GadgetViewer's
    Make plot of Density against Temperature, as a weighted 2-D histogram)."""
    DATASET = DISPLAY_NAME = "Phase diagram, gas"
    TREE_CONTRIBUTIONS = {"Analyses": ["Phase diagram, gas"]}
    TABS = ["Phase diagram"]
    PTYPES = (0,)
    BINS = 120

    def __init__(self) -> None:
        super().__init__()
        self._key = self._result = None

    @classmethod
    def available(cls, vset) -> bool:
        return (super().available(vset) and bool(vset.get_dtype("PartType0/Density"))
                and bool(vset.get_dtype("PartType0/InternalEnergy")))

    def compute(self, data, attrs, header, params, tab, scale_x, scale_y, window, context=None):
        from measurements import SnapshotView
        from measurements.phase import property_histogram2d
        context = context or {}
        vset = params.get("_vset")
        if vset is None:
            return self.error_request("No snapshot open.")
        frame = "comoving" if context.get("coordinates") == "Comoving" else "physical"
        sel = context.get("selection_set")
        key = (tuple(vset.files), frame, None if sel is None else (sel["path"], sel.get("mtime")))
        if key != self._key:
            try:
                view = SnapshotView(vset)
                _check_size(view.count(0), "the phase diagram")
                self._result = property_histogram2d(
                    view, 0, "Density", "Temperature", bins=self.BINS, frame=frame,
                    ids=None if sel is None else sel["ids"],
                    ids_label="" if sel is None else f"set '{sel['name']}':")
            except (MemoryError, ValueError) as exc:
                return self.error_request(str(exc))
            self._key = key
        r = self._result
        v, prov = r.values, r.provenance
        xe, ye, grid = v["x_edges"], v["y_edges"], v["grid"]
        title = "Phase diagram, gas" + ("" if sel is None else f", set '{sel['name']}'")
        status = (f"All {prov.n_total:,} gas particles · mass-weighted · {frame}" + _set_suffix(sel))
        info = (f"Σ bins = {v['in_range']:.6e} Msun of {v['total']:.6e}  |  outside the range "
                f"{v['out_of_range']:.3e}  |  dropped (T or ρ ≤ 0) {v['dropped']:.3e}  |  "
                f"{prov.parameters['y_label']}")
        if tab == "Matrix":
            iy, ix = np.nonzero(grid)
            table = np.column_stack([xe[ix], xe[ix + 1], ye[iy], ye[iy + 1], grid[iy, ix]])
            notes = [f"# {title}; {status}", f"# files: {', '.join(prov.files)}",
                     "# non-empty bins only; axes are log10 of rho [g/cm^3] and T [K]"] + \
                [f"# note: {n}" for n in prov.notes] + [f"# note: {n}" for n in set_notes(sel)]
            return _matrix(self, table, ["log rho lo", "log rho hi", "log T lo", "log T hi",
                                         "mass [Msun]"], notes, info, status)
        with np.errstate(divide="ignore"):
            img = np.log10(grid)
        floor = np.nanmin(img[np.isfinite(img)]) - 0.5 if np.isfinite(img).any() else 0.0
        img = np.where(np.isfinite(img), img, floor)
        return PlotRequest(
            title=title, x_label="log₁₀ ρ [g cm⁻³]", y_label="log₁₀ T [K]",
            images=[ImageSpec(data=img, rect=(xe[0], ye[0], xe[-1], ye[-1]),
                              label="log₁₀ mass per bin")],
            log_x=False, log_y=False, status=status, info=info, tab=tab,
            n_points=prov.n_used, dim_label="D0", unit_dimension="dimensionless")


def _profile(label, ptypes):
    name = "RadialProfile_" + ("all" if ptypes is None else "_".join(map(str, ptypes)))
    return type(name, (RadialProfileAnalysis,), {
        "DATASET": label, "DISPLAY_NAME": label, "PTYPES": ptypes,
        "TREE_CONTRIBUTIONS": {"Analyses": [label]}})


ANALYSES_M2: dict = {("Analyses", "Sanity report"): SanityReportAnalysis,
                     ("Analyses", "Energy, all types"): EnergyBudgetAnalysis,
                     ("Analyses", "Energy over run"): EnergyOverRunAnalysis,
                     ("Analyses", "Phase diagram, gas"): PhaseDiagramAnalysis}
for _pt in (None, 0, 1, 2, 3, 4, 5):
    _label = "Profile, all types" if _pt is None else f"Profile, PartType{_pt}"
    ANALYSES_M2[("Analyses", _label)] = _profile(_label, None if _pt is None else (_pt,))


def available_analysis_leaves_m2(vset) -> list:
    return [leaf for (_, leaf), cls in ANALYSES_M2.items() if cls.available(vset)]
