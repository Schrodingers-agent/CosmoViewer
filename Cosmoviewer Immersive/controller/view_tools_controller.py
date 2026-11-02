"""
View tools for the controller: the scale bar and time overlay, Jump to
coordinates, labelled points, the simulation summary, and saving and
restoring sessions.

A session file (JSON) holds what is needed to come back to the same picture:
the snapshot files, the snapshot shown, the camera, colouring and point style,
visible particle types, the selection sets (their ParticleIDs, names, colours),
the 2D panel, view options, labelled points, additional data files and the
catalogue setting. Opening it loads the snapshots if they are not the ones
already open, then applies the rest. It never stores particle data, so a
session is small unless the selection sets are large.

Version 2 also keeps what each set means (kind, definition, the snapshot it
was defined at, completeness; selection_sets.py) and the settings of every FoF
run (per snapshot file), so a reopened session still says which FoF run
defined a group. FoF is not run again on opening: its groups are recomputed
only when asked. Version 1 sessions still open; their sets are plain IDs
whose completeness is not known.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import QMessageBox

from controller.console import log

SESSION_FORMAT = 'cosmoviewer-session'
SESSION_VERSION = 2          # 2: what each set means, and FoF settings per snapshot
KPC_CM = 3.085678e21


def read_labels(path: str) -> Tuple[np.ndarray, List[str]]:
    """Points from a text file: ``x y z [name...]`` per line (commas or spaces); '#' comments."""
    pos, names = [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for n, line in enumerate(f, start=1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.replace(',', ' ').replace('\t', ' ').split()
            if len(parts) < 3:
                raise ValueError(f"line {n}: expected x y z [name]")
            try:
                xyz = [float(v) for v in parts[:3]]
            except ValueError:
                if not pos:                 # a header line such as "x y z name"
                    continue
                raise ValueError(f"line {n}: x, y and z must be numbers")
            pos.append(xyz)
            names.append(" ".join(parts[3:]))
    if not pos:
        raise ValueError("no points found")
    return np.array(pos, dtype=np.float64), names


def _plain(v):
    """``v`` with numpy numbers and arrays turned into plain JSON values (sessions are JSON)."""
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(f"{type(x).__name__} is not JSON serialisable")
    return json.loads(json.dumps(v, default=default))


def _meaning_from_session(d: dict) -> dict:
    """Keyword arguments of ParticleSelectionSet.set_population for a saved set. A set saved
    by a version-1 session has no meaning recorded: it is plain IDs of unknown completeness."""
    if 'kind' not in d:
        return {}
    complete = d.get('complete')
    if not (complete is None or isinstance(complete, bool)):
        raise ValueError(f"completeness {complete!r} is not yes, no or unknown")
    n_expected = d.get('n_expected')
    return dict(kind=str(d['kind']), definition=dict(d.get('definition') or {}),
                defined_at=dict(d.get('defined_at') or {}), complete=complete,
                n_expected=None if n_expected is None else int(n_expected),
                incomplete_reason=d.get('incomplete_reason'))


def is_cosmological(header: dict, params: Optional[dict] = None) -> bool:
    """
    A run with comoving integration, whose Time is the scale factor (Gadget/EAGLE cosmological runs).

    ComovingIntegrationOn decides when the file states it (Gadget-4 writes it in the Parameters
    group; pass those, or the snapshot's 'unit_source', as ``params``). Otherwise Time must be a
    scale factor and Redshift must match it: a run without comoving integration can have
    Time = 1 and Redshift = 0, which only the parameter tells apart.
    """
    flag = (params or {}).get('ComovingIntegrationOn', header.get('ComovingIntegrationOn'))
    if flag is not None:
        try:
            return int(np.atleast_1d(flag)[0]) != 0
        except (TypeError, ValueError, IndexError):
            pass
    a, z = header.get('Time'), header.get('Redshift')
    try:
        a, z = float(np.atleast_1d(a)[0]), float(np.atleast_1d(z)[0])
    except (TypeError, ValueError, IndexError):
        return False
    return 0 < a <= 1.0001 and abs(1.0 / a - 1.0 - z) <= 1e-3 * (1.0 + z)


def time_text(snap: dict) -> str:
    """
    When a snapshot is, as the status bar, the log and the summary show it:
    "a = 0.500 (z = 1.00)" in a cosmological run (Time is the scale factor),
    "t = 1.203 (code units)" otherwise.
    """
    header = snap.get('header') or {}
    t = float(snap.get('time', header.get('Time', 0.0)) or 0.0)
    if t > 0 and is_cosmological(header, snap.get('unit_source')):
        try:
            z = float(np.atleast_1d(header.get('Redshift'))[0])
        except (TypeError, ValueError, IndexError):
            z = 1.0 / t - 1.0
        z = 0.0 if abs(z) < 0.005 else z                      # never "-0.00"
        a_text = f"{t:.3f}" if t >= 0.01 else f"{t:.3g}"
        z_text = f"{z:.2f}" if z < 100 else f"{z:.1f}"
        return f"a = {a_text} (z = {z_text})"
    return f"t = {t:.4g} (code units)"


class ViewToolsMixin:

    def _init_view_tools(self):
        self._session_pending = None
        self._labels_source = None            # (path, positions, names) of the labelled points
        x = self.view.extras
        x.session_save_requested.connect(self.save_session)
        x.session_open_requested.connect(self.open_session)
        x.labels_requested.connect(self.load_labels)
        x.jump_requested.connect(self.jump_to)
        x.summary_requested.connect(self._show_summary)
        x.panel_toggled.connect(lambda on: on and self._update_frame())   # draw what was skipped while hidden

    # ------------------------------------------------------------------ overlay

    def _length_unit(self, snap: dict):
        """(kpc per code length, comoving?, per h?) or None when the length unit is unknown."""
        units = self.model.units
        L = getattr(units, 'UnitLength_in_cm', None) if units is not None else None
        if not L or not math.isfinite(float(L)) or float(L) <= 0:
            return None
        cosmo = is_cosmological(snap.get('header', {}), snap.get('unit_source'))
        # Gadget and EAGLE cosmological runs store lengths both comoving and per h.
        return float(L) / KPC_CM, cosmo, cosmo

    def _update_overlays(self, snap: dict):
        self.view.extras.set_overlay_info(self._time_label(snap), self._length_unit(snap))

    # ------------------------------------------------------------------ jump, labels, summary

    def jump_to(self, x: float, y: float, z: float, distance: float = float('nan')):
        v = getattr(self.view, 'view_3d', None)
        if v is None:
            return
        kw = {'pos': QVector3D(float(x), float(y), float(z))}
        if math.isfinite(distance) and distance > 0:
            kw['distance'] = float(distance)
        v.setCameraPosition(**kw)
        log(f"Jumped to ({x:.6g}, {y:.6g}, {z:.6g})" + (f", distance {distance:.6g}" if 'distance' in kw else ""))

    def load_labels(self, path: str) -> bool:
        try:
            pos, names = read_labels(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self.view, "Read labelled points", f"Could not read {Path(path).name}: {e}")
            return False
        self._labels_source = (str(path), pos, names)
        self.view.extras.set_labels(pos, names)
        log(f"Labelled points: {len(pos):,} from {Path(path).name}")
        return True

    def summary_text(self) -> str:
        """Plain-text description of the loaded run (Options > Simulation summary)."""
        from controller.console_commands import describe_snapshot
        snaps = self.model.snapshots or []
        if not snaps:
            return "No snapshots loaded."
        snap = self.model.get_snapshot(self.current_snap_idx) or snaps[0]
        files = snap.get('files') or [snap['filepath']]
        out = [f"Snapshots loaded: {len(snaps)}", f"Units: {self._run_units_text()}", "",
               f"Shown: {describe_snapshot(snap)}",
               f"Files: {len(files)} ({Path(files[0]).parent})"]
        header, groups = dict(snap.get('header', {})), {}
        try:
            import h5py
            with h5py.File(files[0], 'r') as f:
                for g in ('Header', 'Cosmology', 'Parameters', 'Units'):
                    if g in f:
                        groups[g] = {k: v for k, v in f[g].attrs.items()}
        except Exception as e:  # binary snapshots or unreadable files: the loader's header only
            groups = {}
            out.append(f"(File attributes not read: {type(e).__name__})")
        header.update(groups.get('Header', {}))

        def fmt(v):
            a = np.asarray(v)
            if a.dtype.kind in 'SO':
                return str(v.decode() if isinstance(v, bytes) else v)
            if a.size == 1:
                return f"{a.item():.6g}" if a.dtype.kind == 'f' else str(a.item())
            return "[" + ", ".join(f"{x:.6g}" if a.dtype.kind == 'f' else str(x) for x in a.ravel()) + "]"

        out += ["", "Header:"]
        keys = ['NumPart_Total', 'NumPart_Total_HighWord', 'MassTable', 'BoxSize', 'Time', 'Redshift',
                'NumFilesPerSnapshot', 'Omega0', 'OmegaBaryon', 'OmegaLambda', 'HubbleParam']
        for k in keys + sorted(k for k in header if k not in keys):
            if k in header:
                out.append(f"  {k} = {fmt(header[k])}")
        cos = groups.get('Cosmology') or {}
        par = groups.get('Parameters') or {}
        extra = {k: cos.get(k, par.get(k)) for k in ('Omega0', 'OmegaBaryon', 'OmegaLambda', 'HubbleParam',
                                                      'ComovingIntegrationOn', 'UnitLength_in_cm',
                                                      'UnitMass_in_g', 'UnitVelocity_in_cm_per_s')
                 if k not in header and (k in cos or k in par)}
        if extra:
            out.append("Cosmology and units:")
            out += [f"  {k} = {fmt(v)}" for k, v in extra.items()]
        out += ["", "Cosmological: yes (Time is the scale factor; lengths are comoving and per h as stored)"
                if is_cosmological(header, {**(snap.get('unit_source') or {}), **par}) else "Cosmological: no", ""]
        props = self._file_properties(snap)
        out.append(f"Per-particle properties in the file: {len(props)}")
        out.append("  " + ", ".join(sorted(props)))
        out.append("")
        out.append(f"Halo catalogue: {self._features_status(snap) or 'none attached'}")
        if self._aux_specs:
            out.append("Additional data: " + ", ".join(f"{s.name} (type {s.ptype}, {s.path})" for s in self._aux_specs))
        out += ["", "All snapshots:"] + ["  " + describe_snapshot(s) for s in snaps]
        return "\n".join(out)

    def _show_summary(self):
        from views.view_extras import TextDialog
        dlg = TextDialog(self.view, "Simulation summary", self.summary_text())
        dlg.setModal(False)
        dlg.show()
        self._summary_dialog = dlg

    # ------------------------------------------------------------------ sessions

    def session_state(self) -> dict:
        v, cp, sp = self.view, self.view.colouring_panel, self.view.selection_panel
        files = []
        for s in self.model.snapshots or []:
            files += [str(f) for f in (s.get('files') or [s['filepath']])]
        sets = [{'name': s.name, 'color': list(s.color), 'enabled': bool(s.enabled),
                 'ids': [int(i) for i in s.id_array()], **_plain(s.meaning())} for s in self.model.tracker.sets]
        labels = None
        if self._labels_source is not None:
            path, pos, names = self._labels_source
            labels = {'path': path, 'positions': np.asarray(pos).tolist(), 'names': list(names)}
        return {
            'format': SESSION_FORMAT, 'version': SESSION_VERSION,
            'files': files,
            'snapshot_index': int(self.current_snap_idx),
            'camera': v.camera_state(),
            'visible_types': {str(t): bool(v.layer_panel.layer_state.is_visible(t)) for t in range(6)},
            'colouring': dict(cp.state(), range=list(cp.state()['range']),
                              density_brightness=bool(getattr(cp, 'chk_density', None)
                                                      and cp.chk_density.isChecked())),
            'drawing': cp.drawing_state(),
            'sets': sets,
            'active_set': int(sp.active_set_idx),
            'selection': {'draw_selected_only': bool(sp.chk_draw_only.isChecked()),
                          'follow': bool(sp.chk_follow.isChecked()), 'trails': bool(sp.chk_trails.isChecked()),
                          'n_trails': int(sp.spin_trails.value())},
            'panel2d': {'mode': int(v.combo_2d.currentIndex()), 'follow': bool(v.chk_2d_follow.isChecked()),
                        'z_slider': int(v.slide_z.value())},
            'view': dict(v.extras.state(), stereo=bool(getattr(v, 'stereo_on', False)),
                         eye_separation=float(getattr(v, 'eye_separation', 0.05))),
            'labels': labels,
            'catalogue_mode': self._catalogue_mode,
            'aux_data': [asdict(s) for s in self._aux_specs],
            'fof_runs': self._fof_runs(),
        }

    def _fof_runs(self) -> List[dict]:
        """Settings of the FoF run of each snapshot that has one: the snapshot file, b or the
        linking length, N_min, linked types, the linking length it gave and the number of groups.
        Runs restored from a session and not repeated since are kept as they were."""
        runs = []
        restored = getattr(self, '_restored_fof_runs', {})
        for i, s in enumerate(self.model.snapshots or []):
            if 'fof' in s:
                runs.append(_plain({'file': str(s['filepath']), 'index': i,
                                    'settings': dict(s.get('fof_settings') or {}),
                                    'linking_length': s.get('fof_linking_length'),
                                    'n_groups': len(s.get('fof_sizes') or {})}))
            elif str(s['filepath']) in restored:
                runs.append(dict(restored[str(s['filepath'])], index=i))
        return runs

    def _restore_fof_runs(self, records: List[dict]):
        """FoF runs saved in a session. Their groups are not recomputed (that needs the FoF
        window), but each run is remembered for its snapshot file, so saving the session again
        keeps it, and the console says which runs there were."""
        self._restored_fof_runs = {}
        for r in records:
            f = str(r.get('file') or '')
            if not f:
                continue
            self._restored_fof_runs[f] = {k: v for k, v in r.items() if k != 'index'}
            st = r.get('settings') or {}
            if st.get('r_link') is None and st.get('b_linking') is not None:
                how = f"b = {st['b_linking']}"
            else:
                how = f"linking length {r.get('linking_length', st.get('r_link'))}"
            log(f"Session: FoF had been run on {Path(f).name} ({how}, N_min = {st.get('min_size')}, "
                f"linked types {st.get('link_types')}; {r.get('n_groups', '?')} groups). Its groups are "
                f"not recomputed: run FoF on that snapshot again to use them.")

    def save_session(self, path: str) -> bool:
        if not self.model.snapshots:
            QMessageBox.information(self.view, "Save session", "Load snapshots first.")
            return False
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.session_state(), f, indent=1)
        except (OSError, TypeError, ValueError) as e:
            QMessageBox.warning(self.view, "Save session", f"Could not write {Path(path).name}: {e}")
            return False
        log(f"Session saved to {path}")
        return True

    def open_session(self, path: str) -> bool:
        try:
            with open(path, encoding='utf-8') as f:
                st = json.load(f)
            if st.get('format') != SESSION_FORMAT:
                raise ValueError("not a CosmoViewer session file")
            if int(st.get('version', 0)) > SESSION_VERSION:
                raise ValueError(f"made by a newer CosmoViewer (version {st.get('version')})")
        except (OSError, ValueError) as e:
            QMessageBox.warning(self.view, "Open session", f"Could not open {Path(path).name}: {e}")
            return False
        files = [str(f) for f in st.get('files', [])]
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            QMessageBox.warning(self.view, "Open session",
                                f"{len(missing)} snapshot file(s) of this session are missing, "
                                f"for example\n{missing[0]}")
            return False
        loaded = []
        for s in self.model.snapshots or []:
            loaded += [str(f) for f in (s.get('files') or [s['filepath']])]
        self._session_pending = st
        if files and sorted(files) != sorted(loaded):
            if self.loading:
                QMessageBox.information(self.view, "Open session", "A load is running; try again when it finishes.")
                self._session_pending = None
                return False
            log(f"Session {Path(path).name}: loading {len(files)} snapshot file(s)")
            self.model.manifest = None
            self.current_snap_idx = 0
            self._start_loader([Path(f) for f in files])
        self._apply_pending_session()
        return True

    def _apply_pending_session(self):
        if self._session_pending is None:
            return
        if self.loading:
            QTimer.singleShot(200, self._apply_pending_session)
            return
        st, self._session_pending = self._session_pending, None
        try:
            self._apply_session(st)
        except Exception as e:  # a partly applied session is reported, never silent
            log(f"Session not fully applied: {type(e).__name__}: {e}", 'error')
            QMessageBox.warning(self.view, "Open session", f"The session was only partly applied: {e}")

    def _apply_session(self, st: dict):
        v, cp, sp = self.view, self.view.colouring_panel, self.view.selection_panel
        if not self.model.snapshots:
            return
        # Data first: snapshot, types, catalogue and extra arrays decide what can be coloured.
        idx = int(np.clip(st.get('snapshot_index', 0), 0, len(self.model.snapshots) - 1))
        self.current_snap_idx = idx
        v.set_time_slider_value(idx)
        for t, on in (st.get('visible_types') or {}).items():
            v.layer_panel.layer_state.set_visible(int(t), bool(on))
        for t, cb in v.layer_panel.checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(v.layer_panel.layer_state.is_visible(t))
            cb.blockSignals(False)
        if 'catalogue_mode' in st and st['catalogue_mode'] != self._catalogue_mode:
            self._enable_catalogues(st['catalogue_mode'])
        if st.get('aux_data'):
            from snapshot_io.aux_data import AuxSpec
            self._aux_specs = [AuxSpec(**d) for d in st['aux_data']]
            self._aux_full.clear()
            self._features_gen += 1
        self._last_prop_names = None
        self._refresh_property_list()

        # Colouring, set widget by widget with signals held, then one change.
        col = st.get('colouring') or {}
        name = col.get('property')
        if name and not cp.select_property(name):
            log(f"Session: '{name}' is not available in these snapshots; colouring by type", 'warning')
        widgets = [cp.combo_table, cp.chk_log, cp.chk_random, cp.edit_min, cp.edit_max]
        if hasattr(cp, 'chk_density'):
            widgets.append(cp.chk_density)
        for w in widgets:
            w.blockSignals(True)
        try:
            if col.get('table'):
                cp.combo_table.setCurrentText(col['table'])
            cp.chk_log.setChecked(bool(col.get('log', False)))
            cp.chk_random.setChecked(bool(col.get('random', cp.chk_random.isChecked())))
            if col.get('range'):
                cp.edit_min.setText(f"{float(col['range'][0]):.6g}")
                cp.edit_max.setText(f"{float(col['range'][1]):.6g}")
            cp.seed = int(col.get('seed', 0))
            if hasattr(cp, 'chk_density'):
                cp.chk_density.setChecked(bool(col.get('density_brightness', False)))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._auto_range_pending = False        # the saved range wins over a pending auto range
        dr = st.get('drawing') or {}
        if dr:
            cp.chk_unlit.setChecked(bool(dr.get('show_unlit', True)))
            if dr.get('blend'):
                cp.combo_blend.setCurrentText(dr['blend'])
            for layer in ('lit', 'unlit'):
                for key, scale in (('size', 1), ('blur', 100), ('fade', 100)):
                    if layer in dr and key in dr[layer]:
                        cp.sliders[(layer, key)].setValue(int(round(float(dr[layer][key]) * scale)))
        cp._changed()

        # Selection sets: IDs, names, colours and what they mean; tracking then follows them
        # through the snapshots. Sets of version-1 sessions are plain IDs of unknown completeness.
        for i, d in enumerate(st.get('sets') or []):
            if i >= len(self.model.tracker.sets):
                break
            s = self.model.tracker.get_set(i)
            s.name = str(d.get('name', s.name))
            if d.get('color'):
                s.color = tuple(int(c) for c in d['color'])
            s.enabled = bool(d.get('enabled', True))
            ids = np.asarray(d.get('ids', []), dtype=np.uint64)
            try:
                s.set_population(ids, **_meaning_from_session(d))
            except (TypeError, ValueError) as e:      # a meaning this version cannot read: say so
                log(f"Session: set {i + 1} ('{s.name}') is restored as plain IDs; "
                    f"what it meant could not be read ({e})", 'warning')
                s.select_by_ids(ids)
        self._restore_fof_runs(st.get('fof_runs') or [])
        sel = st.get('selection') or {}
        if 'active_set' in st:
            sp.active_set_idx = int(st['active_set'])
            sp.combo_sets.setCurrentIndex(sp.active_set_idx)
        sp.chk_draw_only.setChecked(bool(sel.get('draw_selected_only', False)))
        sp.chk_follow.setChecked(bool(sel.get('follow', True)))
        sp.chk_trails.setChecked(bool(sel.get('trails', False)))
        if 'n_trails' in sel:
            sp.spin_trails.setValue(int(sel['n_trails']))

        # View: options, 2D panel, labels, then the camera last so nothing moves it afterwards.
        vw = st.get('view') or {}
        if 'eye_separation' in vw and hasattr(v, 'slide_eye'):
            v.slide_eye.setValue(int(round(float(vw['eye_separation']) * 1000)))
        if hasattr(v, 'act_stereo'):
            v.act_stereo.setChecked(bool(vw.get('stereo', False)))
        v.extras.apply_state(vw)
        p2 = st.get('panel2d') or {}
        if 'z_slider' in p2:
            v.slide_z.setValue(int(p2['z_slider']))
        if 'follow' in p2:
            v.chk_2d_follow.setChecked(bool(p2['follow']))
        if 'mode' in p2:
            v.combo_2d.setCurrentIndex(int(p2['mode']))
        lab = st.get('labels')
        if lab and lab.get('positions'):
            self._labels_source = (lab.get('path', ''), np.asarray(lab['positions'], dtype=np.float64),
                                   list(lab.get('names', [])))
            v.extras.set_labels(self._labels_source[1], self._labels_source[2])
        cam = st.get('camera') or {}
        if cam and getattr(v, 'view_3d', None) is not None:
            v.view_3d.opts['fov'] = float(cam.get('fov', v.view_3d.opts['fov']))
            c = cam.get('center', (0, 0, 0))
            v.view_3d.setCameraPosition(pos=QVector3D(float(c[0]), float(c[1]), float(c[2])),
                                        distance=float(cam['distance']), elevation=float(cam['elevation']),
                                        azimuth=float(cam['azimuth']))
        self._followed_idx = self.current_snap_idx   # do not recentre on the sets straight away
        self._on_selection_changed()                 # redraw and look the sets up in every snapshot
        log(f"Session applied: snapshot {idx + 1} of {len(self.model.snapshots)}, "
            f"{sum(1 for d in st.get('sets') or [] if d.get('ids'))} selection set(s)")
