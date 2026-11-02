"""
Wires the "Log & console" tab to the app and registers the app's commands.
Generic commands (help, errors, jobs, ...) are in controller/console.py.
"""

from pathlib import Path

import numpy as np

from controller.console import CommandRegistry, ConsoleContext, LogBus, log


def attach_console(controller):
    view = controller.view
    tv = view.telemetry_view
    bus = LogBus.instance()

    for e in list(bus.history):               # output from before the window existed
        tv.append_entry(e.level, e.text, e.when)
    bus.entry.connect(tv.on_log_entry)       # bound method: disconnected when tv is deleted
    tv.error_count_changed.connect(view.set_telemetry_badge)

    registry = CommandRegistry()
    ctx = ConsoleContext(registry, controller, clear=tv.clear_log)
    tv.command_entered.connect(lambda line: registry.execute(line, ctx))
    registry.run = lambda line: registry.execute(line, ctx)     # for --run at launch

    def complete(text):
        matches = registry.complete(text.strip())
        if len(matches) == 1:
            tv.set_completion(matches[0] + " ")
        elif matches:
            log("  ".join(matches))
    tv.completion_requested.connect(complete)

    _register_app_commands(registry, controller)
    log("Console ready. Type 'help' for commands.")
    return registry


def describe_snapshot(snap: dict) -> str:
    files = snap.get('files') or [snap.get('filepath')]
    name = Path(files[0]).name if len(files) == 1 else f"{Path(files[0]).name} (+{len(files) - 1} files)"
    drawn = len(snap['c'])
    total = snap.get('n_total')
    # The sample is chosen by a hash of the ParticleID (snapshot_loader.plan_hdf5_snapshot), not
    # every step-th row, so 'step' is only the average ratio N_total / N_drawn.
    if not total:
        sampled = f"{drawn:,} particles"
    elif drawn >= total:
        sampled = f"all {total:,} particles"
    else:
        sampled = f"{drawn:,} of {total:,} particles (a uniform sample, about 1 in {total / max(drawn, 1):.3g})"
    types = ", ".join(f"type {t}: {n:,}" for t, n in zip(*np.unique(snap['t'], return_counts=True)))
    from controller.view_tools_controller import time_text      # as the status bar writes it
    return f"{name}: {time_text(snap)}; {sampled}; {types}"


def _register_app_commands(registry: CommandRegistry, controller):
    def cmd_info(ctx, args):
        snaps = controller.model.snapshots or []
        if not snaps:
            ctx.write("no snapshots loaded")
            return
        ctx.write(f"{len(snaps)} snapshots loaded | {controller._run_units_text()}")
        for s in snaps:
            ctx.write("  " + describe_snapshot(s))
        scaling = snaps[0].get('coord_scaling')
        if scaling:
            ctx.write("  Coordinates scaling attributes: " +
                      ", ".join(f"{k}={v}" for k, v in scaling.items()))

    def cmd_load(ctx, args):
        from cli_import import resolve_sources
        if not args:
            ctx.write("usage: load <folder|file|glob|scheme:...> ...", 'warning')
            return
        specs = args
        # An unquoted path with spaces arrives split; rejoin it when that names something.
        joined = " ".join(args)
        if len(args) > 1 and Path(joined).expanduser().exists():
            specs = [joined]
        plan = resolve_sources(specs)
        ctx.write(plan.describe())
        controller.load_sources(plan)

    def cmd_track(ctx, args):
        """track            summary of every selection set in every snapshot
           track export <file.npz|file.csv> [set]   trajectories of a set (default: active set)"""
        snaps = controller.model.snapshots or []
        sets = controller.model.tracker.sets
        if args and args[0] == 'export':
            if len(args) < 2:
                ctx.write("usage: track export <file.npz|file.csv> [set number]", 'warning')
                return
            from model.tracking import build_trajectories, export_csv, export_npz
            k = int(args[2]) - 1 if len(args) > 2 else controller.view.selection_panel.active_set_idx
            s_set = controller.model.tracker.get_set(k)
            if not s_set.selected_ids or not snaps:
                ctx.write(f"set {k + 1} is empty or nothing is loaded", 'warning')
                return
            located = [controller._tracked(i, s_set, compute=True) for i in range(len(snaps))]
            tr = build_trajectories(located, [s['filepath'] for s in snaps])
            path = str(Path(args[1]).expanduser())
            (export_csv if path.lower().endswith('.csv') else export_npz)(tr, path)
            ctx.write(f"wrote trajectories of {len(tr.ids):,} particles over {tr.n_snap} snapshots to {path}")
            return
        any_set = False
        for s_set in sets:
            if not s_set.selected_ids:
                continue
            any_set = True
            row = []
            for i in range(len(snaps)):
                loc = controller._tracked(i, s_set)
                if loc is None:
                    row.append("-")
                else:
                    by_type = np.bincount(loc.types[loc.found], minlength=6)
                    row.append(f"{loc.n_found:,}" + (" (" + ", ".join(f"type {t}: {n:,}" for t, n in
                                                                         enumerate(by_type) if n) + ")"))
            ctx.write(f"Set {s_set.set_id + 1} '{s_set.name}': {len(s_set.selected_ids):,} IDs")
            for i, r in enumerate(row):
                ctx.write(f"  {Path(snaps[i]['filepath']).name}: {r}")
        if not any_set:
            ctx.write("no particles selected")

    registry.register('info', cmd_info, "describe the loaded snapshots, sampling and units")
    registry.register('track', cmd_track, "selected particles found per snapshot; 'track export <file>'",
                      'track [export <file.npz|file.csv> [set]]')
    registry.register('load', cmd_load, "load snapshots from folders, files or glob patterns",
                      'load <folder|file|glob> ...')
