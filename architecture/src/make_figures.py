"""
Regenerates every figure in architecture/figures from the code and data in this repository.

    python architecture/src/make_figures.py

Diagrams are drawn with matplotlib so they come out as vector PDF (for a paper)
and PNG (for GitHub). Figures 4 and 5 run the application's own LOD hash and
FoF linker on the bundled test snapshot; figure 6 is built from the import
statements of the source tree; figure 7 plots the measurements recorded in
Cosmoviewer Immersive/PERFORMANCE.md.
"""

import ast
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMM = ROOT / "Cosmoviewer Immersive"
EXP = ROOT / "Dataset Explorer"
OUT = ROOT / "architecture" / "figures"
SAMPLE = ROOT / "_dev_tools" / "inspector_check" / "dm_cut"
sys.path.insert(0, str(IMM))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    "savefig.dpi": 220, "pdf.fonttype": 42,
})
INK, GREY, LIGHT = "#1b1b1b", "#8a8a8a", "#efefef"
ACC, ACC2, ACC3 = "#1f4e79", "#a23b2a", "#3c7a3b"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"Creator": "", "Producer": ""})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, title, body="", fc="white", ec=INK, lw=0.7, tsize=9, bsize=7.6, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=tsize, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.034 * 4.4 / ax.figure.get_figheight() - 0.006, body, ha="center", va="top", fontsize=bsize, color="#333",
                family="monospace" if mono else None, linespacing=1.25)


def arrow(ax, a, b, color=INK, lw=0.8, style="-|>", ls="-", rad=0.0, text=None, tpos=0.5, tsize=6.5, toff=(0, 0.008)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        x = a[0] + (b[0] - a[0]) * tpos + toff[0]
        y = a[1] + (b[1] - a[1]) * tpos + toff[1]
        ax.text(x, y, text, fontsize=tsize, ha="center", va="bottom", color=color, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ─── Fig. 1: layered architecture ────────────────────────────────────────────
def fig_architecture():
    fig, ax = canvas(7.2, 4.4)
    bands = [(0.03, "Storage"), (0.20, "I/O"), (0.42, "Core"), (0.66, "Control"), (0.84, "Interface")]
    for y, lab in bands:
        ax.text(0.012, y + 0.07, lab, rotation=90, va="center", ha="center", fontsize=8, color=GREY, style="italic")
    for y0 in (0.185, 0.405, 0.645, 0.825):
        ax.plot([0.03, 0.99], [y0, y0], color=LIGHT, lw=0.8, zorder=0)

    # storage
    box(ax, 0.04, 0.035, 0.30, 0.13, "Simulation output (read-only)",
        "snapshot_NNN[.i].hdf5 · snap_*.hdf5\nfof_tab / fof_subhalo_tab · energy.txt\nGadget binary SnapFormat 1/2", fc=LIGHT)
    box(ax, 0.36, 0.035, 0.30, 0.13, "~/.cosmoviewer",
        "lod_cache/  (memory-mapped .npy, 8 GB LRU)\nproperty_cache/ · results_cache/\nsets/ (fallback for read-only runs)", fc=LIGHT)
    box(ax, 0.68, 0.035, 0.30, 0.13, "<run>/cosmoviewer_sets/",
        "selection sets, HDF5 v2\n/ParticleIDs + population record\nshared by both applications", fc=LIGHT)

    # I/O
    box(ax, 0.04, 0.215, 0.44, 0.16, "snapshot_io  (Immersive)",
        "snapshot_loader · group_catalogue · properties\nid_lookup (chunked ID scan) · id_index (sorted IDs)\nproperty_cache · aux_data · gadget_file_discovery")
    box(ax, 0.50, 0.215, 0.23, 0.16, "io_manager  (Explorer)",
        "VirtualSnapshotSet\nmulti-file prefix sums\nread_particle(global row)")
    box(ax, 0.75, 0.215, 0.23, 0.16, "cosmo_io_engine",
        "Gadget binary readers\n(fallback; HDF5 goes\nthrough h5py directly)")

    # core
    box(ax, 0.04, 0.43, 0.22, 0.20, "lod/", "store: ID-hash sample,\nMorton-ordered cells\nview: f(d) rule, budget X\ncells · frustum · budget", fc="#eaf0f6", ec=ACC)
    box(ax, 0.28, 0.43, 0.22, 0.20, "FoF", "fof: grid linker, exact\nfof_snapshot: full-res read\nfof_process: own process,\nmemory guard", fc="#f6ece9", ec=ACC2)
    box(ax, 0.52, 0.43, 0.22, 0.20, "model/ + selection_sets", "SimulationModel\nParticleTracker (6 sets)\npopulation record\ntracking · units · SPH map", fc="#edf4ec", ec=ACC3)
    box(ax, 0.76, 0.43, 0.22, 0.20, "measurements/", "SnapshotView (explicit units)\ncentre · lagrangian · profiles\nenergy · phase · sanity\ncache · scripts · runset", fc="#edf4ec", ec=ACC3)

    # control
    box(ax, 0.04, 0.665, 0.62, 0.14, "CosmoViewerController  (controller/, one mixin per file)",
        "app · lod · tracking · features · picking · inspector · sets\nstructures · view_tools · analysis_host · console")
    box(ax, 0.68, 0.665, 0.30, 0.14, "CosmoExplorer pipeline",
        "plugin_registry → plugin_base stages\ncompute_worker (QThread) · lod_index")

    # interface
    box(ax, 0.04, 0.845, 0.62, 0.13, "Immersive window  (views/)",
        "Explore 3D (GL point sprites, soft_points) · 2D panel · side panel\nhalo table · inspector · movie dialog · Log & console")
    box(ax, 0.68, 0.845, 0.30, 0.13, "Analyse tab / standalone Explorer",
        "plot_panel: Histogram · Line · Matrix\nanalysis leaves · Export CSV / .py")

    for x in (0.19, 0.40):
        arrow(ax, (x, 0.165), (x, 0.215))
    arrow(ax, (0.615, 0.165), (0.615, 0.215))
    arrow(ax, (0.985, 0.12), (0.985, 0.50), rad=0.35, color=ACC3, ls="--")
    arrow(ax, (0.15, 0.375), (0.15, 0.43)); arrow(ax, (0.39, 0.375), (0.39, 0.43))
    arrow(ax, (0.62, 0.375), (0.85, 0.43), color=INK)
    arrow(ax, (0.30, 0.375), (0.60, 0.43), rad=-0.1)
    for x in (0.15, 0.39, 0.63):
        arrow(ax, (x, 0.63), (x, 0.665))
    arrow(ax, (0.87, 0.63), (0.87, 0.665))
    arrow(ax, (0.35, 0.805), (0.35, 0.845)); arrow(ax, (0.83, 0.805), (0.83, 0.845))
    arrow(ax, (0.66, 0.735), (0.68, 0.735), style="<|-|>", color=ACC3)
    ax.set_xlim(0, 1.03)
    fig.text(0.5, -0.05, "Fig. 1. CosmoViewer components by layer. Arrows point along the main read path; the dashed arrow is the\n"
             "selection-set file read by the Explorer; the double arrow is the in-process hand-off (analysis_host). Nothing in measurements/ imports Immersive's lod/.", ha="center", fontsize=8)
    save(fig, "fig01_architecture")


# ─── Fig. 2: two paths from one population ───────────────────────────────────
def fig_dataflow():
    fig, ax = canvas(7.2, 3.3)
    box(ax, 0.38, 0.80, 0.24, 0.16, "Population", "every particle in the files\nkeyed by ParticleID", fc=LIGHT)
    # representation path (left)
    box(ax, 0.02, 0.50, 0.25, 0.17, "Hash", "u = splitmix64(ID xor seed), top 24 bits\nsame u in every snapshot", fc="#eaf0f6", ec=ACC, mono=False)
    box(ax, 0.02, 0.24, 0.25, 0.17, "LOD store + view rule", "keep u < cap/N, cell order\ndraw u < f_focus·(D/d)²", fc="#eaf0f6", ec=ACC)
    box(ax, 0.02, 0.02, 0.25, 0.13, "Representation", "GPU points, 2D map, movies\n(sampled, smoothed, interpolated)", fc="#eaf0f6", ec=ACC)
    # measurement path (right)
    box(ax, 0.73, 0.50, 0.25, 0.17, "Full-resolution reads", "fof_snapshot · id_lookup/id_index\nproperties · SnapshotView", fc="#edf4ec", ec=ACC3)
    box(ax, 0.73, 0.24, 0.25, 0.17, "Selection", "IDs + kind, definition,\ndefined_at, complete, n_expected", fc="#edf4ec", ec=ACC3)
    box(ax, 0.73, 0.02, 0.25, 0.13, "Measurement", "profiles · radii · energies\nphase · sanity · export", fc="#edf4ec", ec=ACC3)
    arrow(ax, (0.40, 0.80), (0.20, 0.67), color=ACC, text="sampled for display", tpos=0.5, toff=(-0.04, 0.0))
    arrow(ax, (0.60, 0.80), (0.80, 0.67), color=ACC3, text="defined by ParticleID", tpos=0.5, toff=(0.05, 0.0))
    arrow(ax, (0.145, 0.50), (0.145, 0.41), color=ACC); arrow(ax, (0.145, 0.24), (0.145, 0.15), color=ACC)
    arrow(ax, (0.855, 0.50), (0.855, 0.41), color=ACC3); arrow(ax, (0.855, 0.24), (0.855, 0.15), color=ACC3)
    # the wall
    ax.plot([0.5, 0.5], [0.03, 0.70], color=ACC2, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.5, 0.40, "nothing drawn\nis measured\nunmarked", ha="center", va="center", fontsize=7.5, color=ACC2,
            bbox=dict(fc="white", ec="none"))
    arrow(ax, (0.27, 0.07), (0.73, 0.30), color=GREY, ls=":", text="pick or region on the sample → selection marked\n"
          "complete = False, reason \"display_sample\"", tpos=0.45, tsize=6, toff=(0, 0.02))
    arrow(ax, (0.73, 0.085), (0.27, 0.085), color=GREY, ls=":", text="Show in 3D (highlight only)", tpos=0.5, tsize=6, toff=(0, -0.045))
    fig.text(0.5, -0.07, "Fig. 2. One population, two paths. Rendering may sample; measurement reads every member by ID. A selection that\n"
             "did come from the sample carries that fact with it, and every analysis of it prints the warning.", ha="center", fontsize=8)
    save(fig, "fig02_two_paths")


# ─── Fig. 3: selection-set file schema ───────────────────────────────────────
def fig_schema():
    fig, ax = canvas(7.2, 3.4)
    rows = [
        ("/ParticleIDs", "uint64[n]", "sorted, unique; the set itself"),
        ("format, version", "str, int", '"cosmoviewer-selection-set", 2 (v1 still read)'),
        ("name, colour, n", "str, \"#rrggbb\", int", "as shown in both applications"),
        ("created, made_in, source_snapshot", "ISO time, str, file name", "provenance"),
        ("kind", "ids | group | region | query", "how the set was made"),
        ("definition", "JSON", "e.g. finder fof, group 1, b 0.2, min_size 32"),
        ("defined_at", "JSON", "file, index, time, redshift of evaluation"),
        ("complete", "yes | no | unknown", "every member present?"),
        ("n_expected", "int (−1 unknown)", "e.g. a group's length, 34,120"),
        ("incomplete_reason", "str (\"\" unknown)", "display_sample or missing_in_file; hand edits keep it"),
    ]
    x0, widths, y = 0.03, (0.30, 0.30, 0.36), 0.90
    heads = ("dataset / attribute", "type", "meaning")
    xs = [x0, x0 + widths[0], x0 + widths[0] + widths[1]]
    ax.add_patch(Rectangle((x0, y - 0.005), sum(widths), 0.07, fc=ACC, ec="none"))
    for xx, hh in zip(xs, heads):
        ax.text(xx + 0.008, y + 0.03, hh, color="white", fontsize=8, weight="bold", va="center")
    for i, r in enumerate(rows):
        yy = y - 0.075 * (i + 1)
        if i % 2 == 0:
            ax.add_patch(Rectangle((x0, yy - 0.005), sum(widths), 0.075, fc=LIGHT, ec="none"))
        for j, (xx, txt) in enumerate(zip(xs, r)):
            ax.text(xx + 0.008, yy + 0.032, txt, fontsize=7.4, va="center",
                    family="monospace" if j < 2 else None, color=INK if j else ACC)
        if i == 3:
            ax.plot([x0, x0 + sum(widths)], [yy - 0.005, yy - 0.005], color=ACC3, lw=1.0)
    ax.text(x0 + sum(widths) + 0.005, 0.47, "version 2:\npopulation\nrecord", fontsize=7, color=ACC3, va="center")
    fig.text(0.5, 0.01, "Fig. 3. The selection-set file (one HDF5 file per set, <run>/cosmoviewer_sets/<name>.hdf5). The IDs are\n"
             "the set; the attributes below the green rule say what it means and whether it is whole.", ha="center", fontsize=8)
    save(fig, "fig03_selection_schema")


# ─── Fig. 4: level-of-detail rule, on real code ──────────────────────────────
def fig_lod():
    from lod.store import id_uniforms
    from lod.view import DEPTH_FALLOFF
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.38))
    d = np.logspace(-1.3, 1.3, 400)
    for ff, c in ((1.0, ACC2), (0.25, ACC), (0.044, ACC3), (0.002, GREY)):
        f = np.minimum(1, ff * (1.0 / np.maximum(d, 1.0)) ** DEPTH_FALLOFF)
        axs[0].loglog(d, f, color=c, lw=1.1, label=fr"$f_{{\rm focus}}={ff:g}$")
    axs[0].axvline(1, color=GREY, lw=0.5, ls=":")
    axs[0].set_xlabel(r"distance from eye $d\,/\,D$"); axs[0].set_ylabel(r"drawn fraction $f$")
    axs[0].legend(fontsize=6.5, loc="lower left"); axs[0].set_title("(a) sampling rule", fontsize=8.5)

    k = np.arange(0, 41)
    axs[1].semilogy(k, 2.0 ** (-k / 2), "o", ms=2.2, color=ACC)
    axs[1].set_xlabel(r"level $k$"); axs[1].set_ylabel(r"kept fraction $2^{-k/2}$")
    axs[1].set_title("(b) nested levels, two per octave", fontsize=8.5)

    with h5py.File(SAMPLE / "snapshot_002.hdf5") as f:
        ids = f["PartType1/ParticleIDs"][:]
    u1 = id_uniforms(ids)
    perm = np.random.default_rng(3).permutation(len(ids))      # the "next snapshot": same IDs, new order
    u2 = id_uniforms(ids[perm])
    same = np.array_equal(u1[perm], u2)
    axs[2].hist(u1, bins=40, color=ACC, alpha=0.85, histtype="stepfilled", lw=0)
    axs[2].set_xlabel(r"$u$ (hash of ParticleID)"); axs[2].set_ylim(0, 13500); axs[2].set_ylabel("particles")
    axs[2].set_title("(c) hash on 400,000 real IDs", fontsize=8.5)
    axs[2].text(0.5, 0.93, "same u after reordering: " + ("yes" if same else "NO"), transform=axs[2].transAxes,
                ha="center", fontsize=6.5, color=ACC3 if same else ACC2)
    fig.text(0.5, -0.2, r"Fig. 4. Level of detail. (a) $f=\min(1,f_{\rm focus}(D/\max(d,D))^{2})$ from lod/view.py; (b) the store's nested levels;"
             "\n(c) u from lod/store.py on the DM-L50-N128 cut: uniform, and the same particle keeps its u when Gadget writes it elsewhere.",
             ha="center", fontsize=8)
    save(fig, "fig04_lod_rule")


# ─── Fig. 5: FoF on real data against Gadget-4 ──────────────────────────────
def fig_fof():
    from snapshot_io.fof import find_groups
    with h5py.File(SAMPLE / "snapshot_002.hdf5") as f:
        pos = f["PartType1/Coordinates"][:].astype(np.float64)
        ids = f["PartType1/ParticleIDs"][:]
        box_l = float(f["Header"].attrs["BoxSize"])
    with h5py.File(SAMPLE / "fof_tab_002.hdf5") as g:
        glen = g["Group/GroupLen"][:]
        off = g["Group/GroupOffsetType"][:, 1]
    n_full = 128 ** 3
    l_link = 0.2 * box_l / n_full ** (1 / 3)
    res = find_groups(pos, l_link, boxsize=box_l, min_group_size=32, ids=ids)
    # Gadget-4 stores particles grouped and in group order, so a group whose offset+len fits in
    # the cut is complete in it; compare those, matched by membership.
    gi = res.group_index
    pairs = []
    order_ids = np.argsort(ids)
    for j in range(len(glen)):
        if off[j] + glen[j] > len(ids):
            break
        # the cut keeps file order, so rows off..off+len are group j
        mem = gi[off[j]:off[j] + glen[j]]
        lab = np.bincount(mem[mem >= 0]).argmax() if (mem >= 0).any() else -1
        ours = res.group_len[lab] if lab >= 0 else 0
        pairs.append((glen[j], ours))
    pairs = np.array(pairs)
    exact = int((pairs[:, 0] == pairs[:, 1]).sum())

    fig = plt.figure(figsize=(7.2, 3.0))
    ax0 = fig.add_axes([0.02, 0.10, 0.40, 0.84])
    ax1 = fig.add_axes([0.52, 0.17, 0.20, 0.72])
    ax2 = fig.add_axes([0.79, 0.17, 0.20, 0.72])
    sl = (pos[:, 2] > 0) & (pos[:, 2] < box_l)
    ung = gi < 0
    ax0.scatter(pos[ung & sl, 0], pos[ung & sl, 1], s=0.05, c="#c8c8c8", lw=0, rasterized=True)
    rng = np.random.default_rng(1)
    pal = plt.get_cmap("tab20")(rng.permutation(20) / 20.0)
    ax0.scatter(pos[~ung, 0], pos[~ung, 1], s=0.08, c=pal[gi[~ung] % 20], lw=0, rasterized=True)
    ax0.set_aspect("equal"); ax0.set_xlim(0, box_l); ax0.set_ylim(0, box_l)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.text(0.02, 0.98, f"(a) {res.n_groups:,} FoF groups, b = 0.2, l = {l_link*1000:.0f} kpc/h", transform=ax0.transAxes,
             va="top", fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.8))
    ax1.loglog(pairs[:, 0], np.maximum(pairs[:, 1], 1), "o", ms=2.2, mfc="none", mec=ACC, mew=0.6)
    lim = [pairs[:, 0].min() / 1.5, pairs[:, 0].max() * 1.5]
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.plot(lim, lim, color=ACC2, lw=0.6)
    ax1.set_xlabel("GroupLen, Gadget-4"); ax1.set_ylabel("members, CosmoViewer")
    ax1.set_title(f"(b) {exact} of {len(pairs)} identical", fontsize=8.5)
    # grid-linker geometry
    ax2.set_xlim(0, 4); ax2.set_ylim(0, 4); ax2.set_aspect("equal")
    for t in range(5):
        ax2.plot([t, t], [0, 4], color=GREY, lw=0.4); ax2.plot([0, 4], [t, t], color=GREY, lw=0.4)
    ax2.add_patch(Rectangle((1, 1), 1, 1, fc="#f6ece9", ec=ACC2, lw=0.8))
    ax2.add_patch(plt.Circle((1.5, 1.5), np.sqrt(3) * 1.0 * 0.999, fill=False, ec=ACC, lw=0.7, ls="--"))
    ax2.annotate("", (2, 1.05), (1, 1.05), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax2.text(1.5, 0.7, r"$s \leq l/\sqrt{3}$", ha="center", fontsize=7.5)
    ax2.text(3.1, 2.75, r"$l$", color=ACC, fontsize=8)
    ax2.text(0.15, 3.65, "cell pairs joined\nnearest offsets first;\nlinked cells skipped", fontsize=6.3, va="top",
             bbox=dict(fc="white", ec="none", pad=0.5))
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("(c) grid linker", fontsize=8.5)
    fig.text(0.5, -0.04, "Fig. 5. snapshot_io/fof.py run on the bundled 400,000-particle cut of DM-L50-N128 (z = 0). (a) groups in projection;\n"
             "(b) each Gadget-4 group wholly inside the cut against the CosmoViewer group holding most of its members; (c) cell geometry.",
             ha="center", fontsize=8)
    save(fig, "fig05_fof_real_data")
    return exact, len(pairs), res.n_groups


# ─── Fig. 6: module dependency graph from the source ─────────────────────────
def fig_modules():
    pkgs = {"snapshot_io": 0, "lod": 1, "spatial": 1, "model": 2, "movie": 2, "controller": 3, "views": 4}
    nodes, edges = {}, set()
    for p in sorted(IMM.rglob("*.py")):
        rel = p.relative_to(IMM)
        if rel.parts[0] in ("tests", "tools") or p.name == "__init__.py":
            continue
        mod = ".".join(rel.with_suffix("").parts)
        nodes[mod] = pkgs.get(rel.parts[0], 0 if len(rel.parts) == 1 else 2)
    for p in sorted(IMM.rglob("*.py")):
        rel = p.relative_to(IMM)
        mod = ".".join(rel.with_suffix("").parts)
        if mod not in nodes:
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            targets = []
            if isinstance(n, ast.ImportFrom) and n.module:
                if n.level:
                    base = rel.parts[0] if len(rel.parts) > 1 else ""
                    targets.append(f"{base}.{n.module}" if base else n.module)
                else:
                    targets.append(n.module)
                    targets += [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                targets += [a.name for a in n.names]
            for t in targets:
                if t in nodes and t != mod:
                    edges.add((mod, t))
    nodes["main"] = 5
    cols = {}
    for m, c in nodes.items():
        cols.setdefault(c, []).append(m)
    pos = {}
    for c, ms in cols.items():
        ms.sort()
        for i, m in enumerate(ms):
            pos[m] = (c, 1 - (i + 0.5) / len(ms))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ups = []
    for a, b in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        if x2 == x1:
            col, rad, dx = GREY, 0.45, 0.0
        else:
            col, rad, dx = (ACC, 0.05, 0.05) if x2 < x1 else (ACC2, 0.05, -0.05)
            if x2 > x1:
                ups.append((a, b))
        ax.annotate("", (x2 + (dx if x2 != x1 else 0), y2), (x1 - (dx if x2 != x1 else 0), y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.4 if col != ACC2 else 0.7, color=col,
                                    alpha=0.45 if col != ACC2 else 0.9, shrinkA=4, shrinkB=4,
                                    mutation_scale=5, connectionstyle=f"arc3,rad={rad}"))
    for m, (x, y) in pos.items():
        ax.text(x, y, m.split(".")[-1], ha="center", va="center", fontsize=5.6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=GREY, lw=0.4))
    labels = ["I/O (snapshot_io, top level)", "LOD / spatial", "model / movie", "controller", "views", "entry"]
    for c, lab in enumerate(labels):
        ax.text(c, 1.03, lab, ha="center", fontsize=7, weight="bold")
    ax.set_xlim(-0.6, 5.6); ax.set_ylim(-0.02, 1.06); ax.axis("off")
    fig.text(0.5, 0.0, f"Fig. 6. Import graph of Cosmoviewer Immersive ({len(nodes)} modules, {len(edges)} edges), read from the source.\n"
             f"Blue edges point down the layers, grey ones stay within a layer, and the {len(ups)} red ones point up.",
             ha="center", fontsize=8)
    save(fig, "fig06_module_graph")
    print("upward imports:", sorted(ups))
    return len(nodes), len(edges)


# ─── Fig. 7: performance, from PERFORMANCE.md ────────────────────────────────
def fig_perf():
    zoom = np.array([1.5, 0.5, 0.2, 0.08, 0.03, 0.01, 0.004])
    drawn = np.array([1.46, 1.78, 1.56, 1.62, 1.56, 1.97, 1.99])     # native M3, 16M store, 1 Nov 2026
    frac = np.array([0.0020, 0.0078, 0.022, 0.044, 0.088, 0.35, 1.0])
    rows = np.array([0, 0, 48.8, 38.9, 33.1, 20.6, 19.2])
    qt = np.array([0.34, 0.37, 2.9, 2.0, 1.7, 1.1, 1.0])
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.62))
    axs[0].semilogx(zoom, drawn, "o-", color=ACC, ms=3, lw=0.9)
    axs[0].axhline(2.0, color=GREY, ls="--", lw=0.6); axs[0].text(1.0, 2.03, "budget X = 2M", fontsize=6.5, color=GREY)
    axs[0].invert_xaxis(); axs[0].set_ylim(0, 2.3)
    axs[0].set_xlabel("view size / box"); axs[0].set_ylabel("particles drawn (M)")
    ax0b = axs[0].twinx(); ax0b.loglog(zoom, frac, "s:", color=ACC2, ms=2.5, lw=0.7)
    ax0b.set_ylabel(r"$f_{\rm focus}$", color=ACC2); ax0b.tick_params(axis="y", colors=ACC2, labelsize=7)
    axs[0].set_title("(a) EAGLE, 850M particles", fontsize=8.5)
    axs[1].semilogx(zoom, qt, "o-", color=ACC3, ms=3, lw=0.9); axs[1].invert_xaxis()
    for z, r, q in zip(zoom, rows, qt):
        if r:
            axs[1].text(z, q + 0.12, f"{r:.0f}M", fontsize=5.8, ha="center", color=GREY)
    axs[1].set_xlabel("view size / box"); axs[1].set_ylabel("query time (s)"); axs[1].set_ylim(0, 3.5)
    axs[1].set_title("(b) query time, 8M store (VM)", fontsize=8.5)
    labs = ["peak, rapid\nzoom", "idle"]
    old, new = [3.51, 2.15], [3.95, 1.65]          # cold and warm runs, 1 Nov 2026
    xx = np.arange(2)
    axs[2].bar(xx - 0.18, old, 0.34, color=GREY, label="cold caches")
    axs[2].bar(xx + 0.18, new, 0.34, color=ACC, label="warm caches")
    axs[2].set_xticks(xx); axs[2].set_xticklabels(labs, fontsize=7); axs[2].set_ylabel("memory (GB)")
    axs[2].set_ylim(0, 5.2); axs[2].legend(fontsize=6.5, loc="upper right"); axs[2].set_title("(c) 16 GB MacBook, EAGLE", fontsize=8.5)
    fig.text(0.5, -0.2, "Fig. 7. Level of detail on EAGLE Ref-L0025N0752 (16 files, 58 GiB), from PERFORMANCE.md. (a) and (c) re-measured on the\n"
             "native M3 on 1 Nov 2026 (all processes, under heavy swap); (b) from the earlier VM run, with rows read from files marked.",
             ha="center", fontsize=8)
    save(fig, "fig07_performance")


# ─── Fig. 8: tracking a selection through time ───────────────────────────────
def fig_tracking():
    fig, ax = canvas(7.2, 2.6)
    for i, (lab, x) in enumerate((("snapshot k  (defined_at)", 0.03), ("snapshot j", 0.70))):
        box(ax, x, 0.30, 0.27, 0.55, lab, "", fc=LIGHT)
    rng = np.random.default_rng(5)
    a = rng.normal(0, 0.035, (90, 2)) + (0.165, 0.55)
    b = np.clip(rng.normal(0, 0.06, (90, 2)) * (1.0, 0.7) + (0.835, 0.55), (0.72, 0.34), (0.95, 0.78))
    ax.scatter(a[:, 0], a[:, 1], s=2, color=ACC3); ax.scatter(b[:, 0], b[:, 1], s=2, color=ACC3)
    box(ax, 0.355, 0.62, 0.29, 0.23, "IDs (uint64, sorted)", "the set never changes;\nonly where its members are")
    box(ax, 0.355, 0.30, 0.29, 0.26, "find by ID in snapshot j", "1M–200M particles: id_index,\nbinary search, 12 B/particle\nelse: chunked scan (id_lookup)\nfirst occurrence wins", fc="#edf4ec", ec=ACC3)
    arrow(ax, (0.30, 0.74), (0.355, 0.74)); arrow(ax, (0.50, 0.62), (0.50, 0.56))
    arrow(ax, (0.645, 0.43), (0.70, 0.43))
    ax.text(0.5, 0.17, "gas → star: the ID survives the type change, so the member is still found (as PartType4)\n"
            "IDs absent from snapshot j are reported as not found, and the panel says how many of n_expected were found", ha="center", fontsize=7, color="#333")
    fig.text(0.5, 0.01, "Fig. 8. Tracking answers \"where are these particles now\"; it does not re-run the definition at snapshot j.",
             ha="center", fontsize=8)
    save(fig, "fig08_tracking")


if __name__ == "__main__":
    fig_architecture(); fig_dataflow(); fig_schema(); fig_lod()
    exact, n, ng = fig_fof()
    nm, ne = fig_modules()
    fig_perf(); fig_tracking()
    print(f"FoF: {exact}/{n} catalogue groups identical in size; {ng} groups found")
    print(f"import graph: {nm} modules, {ne} edges")
