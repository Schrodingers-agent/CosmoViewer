"""
Particle inspector: what the snapshot stores about a picked particle, now and
how it changed.

A pick (click or List top, controller/picking_controller.py) also makes that
particle the inspected one. The inspector then reads the particle from every
loaded snapshot at full resolution, by ParticleID (snapshot_io/particle_record.py),
current snapshot first, in a background thread. The panel shows:

- every property in the current snapshot, as stored or in physical units;
- Δ and Δ% against the previous, next or first loaded snapshot (positions by
  the nearest periodic image), the time between them, the displacement, and
  a change of particle type (gas that became a star);
- the history of one chosen property over all loaded snapshots.

The design follows OVITO's data inspector (pick into a property table) and
ParaView's "Plot Selection Over Time" (one picked element's values at every
time step), with the per-snapshot change added. Gadget-4 catalogue membership
is added from the particle's row when catalogues are attached, and its group in
the FoF result of a snapshot FoF was run on (1 = first, 0 = in no group).
"Select its catalogue group" and "Select its FoF group" make that group's complete
membership the active set (StructuresMixin.select_group_population), so a group
seen through one particle never has to be retyped as a number.

Values never come from the display sample or a LOD view set, so the numbers
are the file's own.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog

from controller.console import log
from controller.thread_hold import hold
from snapshot_io.particle_record import (ParticleRecord, history, read_records, record_delta)
from snapshot_io.properties import SNAP_GROUP, SNAP_SUBHALO
from views.inspector_panel import fmt, fmt_rel

RECENT_MAX = 10
CAT_NAMES = ('Group (catalogue)', 'Subhalo (catalogue)')
FOF_NAME = 'FoF group (computed here)'          # as the colouring list calls it (features_controller)
# Group and subhalo numbers are labels, not quantities: a difference or a percentage of two
# of them means nothing (groups are renumbered at every snapshot), so Δ says what it was.
LABEL_NAMES = CAT_NAMES + (FOF_NAME, SNAP_GROUP, SNAP_SUBHALO, 'GroupNumber', 'SubGroupNumber')


class _RecordJob(QThread):
    """Reads one particle from the given snapshots, emitting each as it is read."""
    one = pyqtSignal(object, object, str)       # (files key, record or None, error)

    def __init__(self, pid: int, jobs: List[Tuple[tuple, dict]]):
        super().__init__()
        self.pid, self.jobs, self._cancel = pid, jobs, False

    def cancel(self):
        self._cancel = True

    def run(self):
        for key, props in self.jobs:
            if self._cancel:
                return
            try:
                rec = read_records(list(key), [self.pid], props)[self.pid]
                self.one.emit(key, rec, '')
            except Exception as e:  # shown in the panel and the console
                self.one.emit(key, None, f"{type(e).__name__}: {e}")


class InspectorMixin:

    def _init_inspector(self):
        self._inspect_pid: Optional[int] = None
        self._inspect_recent: List[Tuple[int, str]] = []
        self._records: Dict[Tuple[int, tuple], ParticleRecord] = {}
        self._record_errors: Dict[Tuple[int, tuple], str] = {}
        self._record_job: Optional[_RecordJob] = None
        self._old_record_jobs: List[_RecordJob] = []
        ip = self.view.inspector_panel
        ip.inspected_chosen.connect(lambda i: self.inspect_particle(self._inspect_recent[i][0])
                                    if 0 <= i < len(self._inspect_recent) else None)
        ip.options_changed.connect(self._refresh_inspector)
        ip.history_property_changed.connect(lambda _: self._refresh_inspector())
        ip.export_requested.connect(self._export_inspected)
        ip.clear_requested.connect(self._clear_inspector)
        ip.select_catalogue_group_requested.connect(lambda: self._select_inspected_group('catalogue'))
        ip.select_fof_group_requested.connect(lambda: self._select_inspected_group('fof'))

    # ------------------------------------------------------------------ choosing a particle

    def inspect_particle(self, pid: int, label: str = ""):
        """Make ``pid`` the inspected particle and read it from every loaded snapshot."""
        pid = int(pid)
        self._inspect_pid = pid
        self._inspect_recent = [(p, l) for p, l in self._inspect_recent if p != pid]
        self._inspect_recent.insert(0, (pid, label or f"ID {pid}"))
        del self._inspect_recent[RECENT_MAX:]
        self.view.inspector_panel.set_recent([l for _, l in self._inspect_recent], 0)
        if hasattr(self.view, 'inspector_section'):
            self.view.inspector_section.set_expanded(True)
        self._start_record_job()
        self._refresh_inspector()

    def _clear_inspector(self):
        self._cancel_record_job()
        self._inspect_pid = None
        self._inspect_recent = []
        self.view.inspector_panel.set_recent([], -1)
        self.view.inspector_panel.reset("Pick a particle (PARTICLE SELECTION > Pick lit particles) to inspect it.")
        self._update_group_actions(None)

    # ------------------------------------------------------------------ reading

    def _snap_key(self, snap: dict) -> tuple:
        return tuple(str(f) for f in (snap.get('files') or [snap['filepath']]))

    def _cancel_record_job(self):
        if self._record_job is not None and self._record_job.isRunning():
            self._record_job.cancel()
            self._old_record_jobs.append(self._record_job)   # kept referenced until it finishes
        self._record_job = None
        self._old_record_jobs = [j for j in self._old_record_jobs if j.isRunning()]

    def _start_record_job(self):
        self._cancel_record_job()
        pid = self._inspect_pid
        if pid is None or not self.model.snapshots:
            return
        n = len(self.model.snapshots)
        cur = min(max(self.current_snap_idx, 0), n - 1)
        order = sorted(range(n), key=lambda i: (abs(i - cur), i))   # current first, then neighbours
        jobs = []
        for i in order:
            snap = self.model.snapshots[i]
            key = self._snap_key(snap)
            if (pid, key) in self._records or not str(key[0]).endswith(('.hdf5', '.h5')):
                continue
            jobs.append((key, self._file_properties(snap)))
        if not jobs:
            return
        job = _RecordJob(pid, jobs)
        job.one.connect(lambda key, rec, err, p=pid: self._on_record(p, key, rec, err))
        self._record_job = job
        hold(job).start()

    def _on_record(self, pid: int, key: tuple, rec, err: str):
        if err:
            self._record_errors[(pid, key)] = err
            log(f"Inspector: ID {pid} in {Path(key[0]).name}: {err}", 'error')
        else:
            self._records[(pid, key)] = rec
        if pid == self._inspect_pid:
            self._refresh_inspector()

    def _record_at(self, idx: int) -> Optional[ParticleRecord]:
        if self._inspect_pid is None or not (0 <= idx < len(self.model.snapshots)):
            return None
        snap = self.model.snapshots[idx]
        rec = self._records.get((self._inspect_pid, self._snap_key(snap)))
        if rec is not None and rec.found and self._catalogue_mode:
            self._add_catalogue_membership(snap, rec)
        if rec is not None and rec.found:
            self._add_fof_membership(snap, rec)
        return rec

    def _add_fof_membership(self, snap: dict, rec: ParticleRecord):
        """The particle's group in the FoF result of this snapshot, kept current: FoF can be run
        again, and a record read earlier is reused for a reloaded snapshot that has no FoF result."""
        g = self._fof_group_of(snap, rec.pid)
        if g is None:
            rec.values.pop(FOF_NAME, None)
        else:
            rec.values[FOF_NAME] = float(g)

    def _add_catalogue_membership(self, snap: dict, rec: ParticleRecord):
        if CAT_NAMES[0] in rec.values:
            return
        try:
            from snapshot_io.group_catalogue import membership
            path, cat = self._catalogue_for(snap)
            if path is None:
                return
            rows = np.array([rec.global_row])
            rec.values[CAT_NAMES[0]] = float(membership(cat.group_offset, cat.group_len, rec.ptype, rows)[0])
            if cat.sub_offset is not None:
                rec.values[CAT_NAMES[1]] = float(membership(cat.sub_offset, cat.sub_len, rec.ptype, rows)[0])
        except Exception as e:  # membership is optional extra information
            log(f"Inspector: catalogue membership: {e}", 'warning')

    # ------------------------------------------------------------------ showing

    def _compare_index(self, mode: str) -> Optional[int]:
        cur, n = self.current_snap_idx, len(self.model.snapshots)
        j = {'previous snapshot': cur - 1, 'next snapshot': cur + 1, 'first snapshot': 0}[mode]
        return j if 0 <= j < n and j != cur else None

    def _refresh_inspector(self):
        ip = self.view.inspector_panel
        pid = self._inspect_pid
        if pid is None or not self.model.snapshots:
            return
        if self._record_job is None and any((pid, self._snap_key(s)) not in self._records
                                            and (pid, self._snap_key(s)) not in self._record_errors
                                            for s in self.model.snapshots):
            self._start_record_job()            # new snapshots were loaded since the pick
        cur = self.current_snap_idx
        now = self._record_at(cur)
        n_read = sum((pid, self._snap_key(s)) in self._records for s in self.model.snapshots)
        n = len(self.model.snapshots)
        self._update_group_actions(now)
        if now is None:
            err = self._record_errors.get((pid, self._snap_key(self.model.snapshots[cur])))
            ip.clear(f"ID {pid}: " + (f"could not be read ({err})" if err else "reading this snapshot..."))
            return
        if not now.found:
            ip.clear(f"ID {pid} is not in snapshot {cur} ({Path(now.file or self.model.snapshots[cur]['filepath']).name}).")
            ip.lbl_status.setText(f"Read in {n_read} of {n} snapshots.")
            self._show_history(now)
            return
        phys = ip.physical()
        c = now.clock
        when = f"t = {c.time:.6g}"
        if c.comoving:
            # the z = 0 snapshot of DM-L50 stores Redshift = 4.4e-16 (1/a - 1 in floating point)
            z = c.redshift if abs(c.redshift) >= 1e-8 else 0.0
            when = f"a = {c.a:.4g}, z = {z:.4g}, t = {c.time_gyr:.4g} Gyr"
        elif c.time_gyr is not None:
            when += f" ({c.time_gyr:.4g} Gyr)"
        head = f"ID {pid} · {now.type_name} · snapshot {cur} · {when}"

        j = self._compare_index(ip.compare_mode())
        before = self._record_at(j) if j is not None else None
        delta = record_delta(now, before, phys) if before is not None and before.found else None
        if delta is not None:
            head += f"\nΔ against snapshot {j}: Δt = {delta.dt:.4g} {delta.time_unit}"
            if delta.displacement is not None:
                head += f", |Δr| = {delta.displacement:.4g}" + (" (physical)" if phys else " (stored units)")
            if delta.type_change:
                head += f"\nType changed: {delta.type_change[0]} → {delta.type_change[1]}"
        elif j is None:
            head += "\nNo snapshot to compare with in that direction."
        elif before is None:
            head += f"\nReading snapshot {j} for the comparison..."
        else:
            head += f"\nNot present in snapshot {j}."
        ip.lbl_header.setText(head)

        changes = {ch.name: ch for ch in (delta.changes if delta else [])}
        rows = []
        for name in sorted(now.values, key=str.lower):
            val = now.physical(name) if phys else now.values[name]
            ch = changes.get(name)
            # A property the earlier snapshot did not have ("new", as StellarFormationTime when gas
            # becomes a star) is a change too, as "gone" is.
            if ip.only_changed() and delta is not None and ch is not None and not ch.delta:
                continue
            inf = now.info.get(name)
            tip = inf.unit_label if inf is not None else (
                "group found by FoF in this window on this snapshot (1 = first, 0 = none)" if name == FOF_NAME
                else "index from the catalogue (0 = none)")
            if phys and inf is not None and val is None:
                tip += "; no a/h attributes, so no physical value"
            if ch is not None and ch.rate is not None:
                tip += f"\nrate = {ch.rate:.6g} per {delta.time_unit}"
            if ch is not None and ch.before is not None:
                tip += f"\nvalue in snapshot {j} = {ch.before:.6g}"
            if ch is not None and name in LABEL_NAMES:
                same = ch.before is not None and val is not None and ch.before == val
                rows.append((name, fmt(val), "same" if same else f"was {fmt(ch.before)}", "–", tip))
                continue
            rows.append((name, fmt(val), fmt(ch.delta) if ch else ("new" if delta else "–"),
                         fmt_rel(ch.relative) if ch else "–", tip))
        if delta is not None and delta.only_before:
            for name in delta.only_before:
                rows.append((name, "–", "gone", "–", f"stored in snapshot {j} but not in this one"))
        ip.set_rows(rows)
        ip.set_history_choices(sorted(now.values, key=str.lower))
        ip.lbl_status.setText(f"Read in {n_read} of {n} snapshots."
                              + (" Values as stored (code units)." if not phys else " Physical code units."))
        self._show_history(now)

    def _show_history(self, now: ParticleRecord):
        ip = self.view.inspector_panel
        name = ip.history_property()
        recs = [self._record_at(i) for i in range(len(self.model.snapshots))]
        if not name or not any(r is not None for r in recs):
            ip.set_history(np.array([]), np.array([]), -1, '', '')
            return
        placeholder = ParticleRecord(self._inspect_pid, False)
        t, v = history([r if r is not None else placeholder for r in recs], name, ip.physical())
        clock = next((r.clock for r in recs if r is not None and r.clock is not None), None)
        xlabel = clock.time_label if clock is not None else 'time'
        # a group number is a label: no line joining group 18 to group 1 as if it passed through 2..17
        ip.set_history(t, v, self.current_snap_idx, xlabel, name, joined=name not in LABEL_NAMES)

    # ------------------------------------------------------------------ the particle's group

    def _inspected_groups(self, now: Optional[ParticleRecord]) -> Tuple[Optional[int], Optional[int], str, str]:
        """(catalogue group, FoF group, why not, why not) of the inspected particle in the snapshot on
        screen; a group is None when there is none to select, and the text says why."""
        snap = self.model.get_snapshot(self.current_snap_idx)
        if now is None or not now.found or snap is None:
            why = "Pick a particle first." if now is None else "The particle is not in this snapshot."
            return None, None, why, why
        cat = int(now.values.get(CAT_NAMES[0], 0) or 0)
        if not self._catalogue_mode:
            cat_tip, cat = "No halo catalogue is attached (Structures > Read halo catalogue...).", None
        elif cat <= 0:
            cat_tip, cat = "This particle is in no group of the catalogue.", None
        else:
            cat_tip = (f"Put every member of catalogue group {cat:,} into the active set, at full resolution, "
                       f"and centre the view on it")
        fof = self._fof_group_of(snap, now.pid)
        if fof is None:
            fof_tip = "FoF has not been run on this snapshot (Structures > Find FoF groups...)."
        elif fof <= 0:
            fof_tip, fof = "This particle is in no FoF group here (groups smaller than N_min are not kept).", None
        else:
            fof_tip = (f"Put every member of FoF group {fof:,} of this snapshot into the active set and centre "
                       f"the view on it")
        return cat, fof, cat_tip, fof_tip

    def _update_group_actions(self, now: Optional[ParticleRecord]):
        cat, fof, cat_tip, fof_tip = self._inspected_groups(now)
        self.view.inspector_panel.set_group_actions(cat, fof, cat_tip, fof_tip)

    def _select_inspected_group(self, kind: str):
        """'Select its catalogue / FoF group': the group's complete membership becomes the active set."""
        ip = self.view.inspector_panel
        snap = self.model.get_snapshot(self.current_snap_idx)
        cat, fof, cat_tip, fof_tip = self._inspected_groups(self._record_at(self.current_snap_idx))
        group = cat if kind == 'catalogue' else fof
        if group is None or snap is None:
            ip.lbl_status.setText(cat_tip if kind == 'catalogue' else fof_tip)
            return None
        try:
            s_set = self.select_group_population(snap, kind, group)
        except ValueError as e:
            ip.lbl_status.setText(str(e))
            return None
        ip.lbl_status.setText(f"Set {s_set.set_id + 1}: {s_set.describe()}")
        return s_set

    # ------------------------------------------------------------------ export

    def _export_inspected(self, path: str = ""):
        pid = self._inspect_pid
        if pid is None:
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self.view, "Export inspected particle",
                                                  f"particle_{pid}.csv", "CSV (*.csv)")
            if not path:
                return None
        recs = [self._record_at(i) for i in range(len(self.model.snapshots))]
        names = sorted({k for r in recs if r is not None for k in r.values}, key=str.lower)
        d_names = [n for n in names if n not in LABEL_NAMES]     # no differences of group numbers
        phys = self.view.inspector_panel.physical()
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow([f"# CosmoViewer particle inspector: ParticleID {pid}; values "
                        + ("in physical code units (a^a_scaling h^h_scaling applied)" if phys else "as stored (code units)")
                        + "; d_<name> = change since the previous loaded snapshot (positions by the nearest periodic image)"
                        + "; group numbers are labels and have no d_ column"])
            w.writerow(['snapshot', 'file', 'found', 'type', 'time', 'a', 'cosmic_time_gyr']
                       + names + [f"d_{n}" for n in d_names])
            prev = None
            for i, r in enumerate(recs):
                if r is None:
                    w.writerow([i, Path(self.model.snapshots[i]['filepath']).name, 'not read'])
                    continue
                c = r.clock
                vals = [(r.physical(n) if phys else r.values.get(n)) for n in names]
                d = {}
                if prev is not None and prev.found and r.found:
                    d = {ch.name: ch.delta for ch in record_delta(r, prev, phys).changes}
                w.writerow([i, Path(r.file or self.model.snapshots[i]['filepath']).name, int(r.found), r.type_name,
                            c.time, c.a if c.comoving else '', c.time_gyr if c.time_gyr is not None else '']
                           + ['' if v is None else repr(float(v)) for v in vals]
                           + ['' if d.get(n) is None else repr(float(d[n])) for n in d_names])
                prev = r
        log(f"Inspector: wrote {path}")
        self.view.inspector_panel.lbl_status.setText(f"Wrote {Path(path).name}")
        return path
