"""
Make-movie window, after GadgetViewer's movie window (movie.f90, movie_open).

Two tabs:
* Quick: a few plain choices, straight to one MP4 or GIF file.
* Custom: the full set (GadgetViewer's rotating and evolving movies, camera
  paths through keyframes, in-between frames, smoothed frames, overlays, PNG
  frames with optional MP4/GIF and a JSON record of every setting).
"""

import json
from pathlib import Path
from typing import List

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QRadioButton, QSpinBox,
    QSizePolicy, QTabWidget, QVBoxLayout, QWidget
)

from model.colour_tables import TABLES
from movie.keyframes import Keyframe
from movie.maker import MovieSettings, can_write_gif, can_write_mp4

QUICK_SIZES = (("640 x 480", 640, 480), ("1280 x 720 (HD)", 1280, 720),
               ("1920 x 1080 (Full HD)", 1920, 1080), ("1080 x 1080 (square)", 1080, 1080))
QUICK_FPS = {'.mp4': 25, '.gif': 15}
MAX_QUICK_IN_BETWEEN = 100
QUICK_SMOOTHED_PARTICLES = 500_000   # smoothed frames cost ~3 s each at this size (container, 640x480)


class MovieDialog(QDialog):
    movie_requested = pyqtSignal(object)       # MovieSettings
    keyframe_requested = pyqtSignal()          # add the current view as a keyframe
    keyframe_view_requested = pyqtSignal(object)   # Keyframe to show on screen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make a movie")
        self._n_snapshots = 1
        self._budget = 2_000_000
        self.keyframes: List[Keyframe] = []
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_quick(), "Quick")
        self.tabs.addTab(self._build_custom(), "Custom")
        layout.addWidget(self.tabs)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_make = QPushButton("Make movie")
        self.btn_make.clicked.connect(self._make)
        btns.addWidget(self.btn_make)
        layout.addLayout(btns)
        self.tabs.currentChanged.connect(self._fit_tab)
        self._update_enabled()
        self._fit_tab(0)

    def _fit_tab(self, index: int):
        """Size the window to the tab shown: Quick is short, Custom is tall."""
        for i in range(self.tabs.count()):
            pol = QSizePolicy.Policy.Preferred if i == index else QSizePolicy.Policy.Ignored
            self.tabs.widget(i).setSizePolicy(pol, pol)
        self.tabs.adjustSize()
        self.adjustSize()

    # ------------------------------------------------------------------ quick

    def _build_quick(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.quick_kind = QComboBox()
        self.quick_kind.addItems(["Turn around the current view", "Play through all snapshots"])
        f.addRow("Movie:", self.quick_kind)
        self.quick_seconds = QSpinBox()
        self.quick_seconds.setRange(1, 600)
        self.quick_seconds.setValue(8)
        self.quick_seconds.setSuffix(" s")
        f.addRow("Length:", self.quick_seconds)
        self.quick_size = QComboBox()
        self.quick_size.addItems([s[0] for s in QUICK_SIZES])
        self.quick_size.setCurrentIndex(1)
        f.addRow("Size:", self.quick_size)
        self.quick_look = QComboBox()
        self.quick_look.addItems(["Points, as on screen", "Smoothed (slower)"])
        f.addRow("Look:", self.quick_look)
        fmt = QHBoxLayout()
        self.quick_mp4 = QRadioButton("MP4")
        self.quick_gif = QRadioButton("GIF")
        group = QButtonGroup(w)
        group.addButton(self.quick_mp4)
        group.addButton(self.quick_gif)
        fmt.addWidget(self.quick_mp4)
        fmt.addWidget(self.quick_gif)
        fmt.addStretch()
        f.addRow("Format:", fmt)
        if not can_write_mp4():
            self.quick_mp4.setEnabled(False)
            self.quick_mp4.setToolTip("Neither ffmpeg nor OpenCV was found")
        if not can_write_gif():
            self.quick_gif.setEnabled(False)
        (self.quick_mp4 if self.quick_mp4.isEnabled() else self.quick_gif).setChecked(True)
        self.quick_mp4.toggled.connect(self._sync_quick_suffix)
        path = QHBoxLayout()
        self.quick_file = QLineEdit(str(Path.home() / "Movies" / "cosmoviewer" /
                                        ("movie.mp4" if self.quick_mp4.isChecked() else "movie.gif")))
        path.addWidget(self.quick_file)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_quick)
        path.addWidget(browse)
        f.addRow("Save as:", path)
        note = QLabel("Uses the current view, colours and selections. With selected particles, "
                      "the camera follows them. With few snapshots, smooth in-between frames are added.")
        note.setWordWrap(True)
        f.addRow(note)
        return w

    def _quick_suffix(self) -> str:
        return '.mp4' if self.quick_mp4.isChecked() else '.gif'

    def _sync_quick_suffix(self):
        p = Path(self.quick_file.text().strip() or "movie")
        self.quick_file.setText(str(p.with_suffix(self._quick_suffix())))

    def _browse_quick(self):
        suffix = self._quick_suffix()
        f, _ = QFileDialog.getSaveFileName(self, "Save movie as", self.quick_file.text(),
                                           f"{suffix[1:].upper()} (*{suffix})")
        if f:
            self.quick_file.setText(str(Path(f).with_suffix(suffix)))

    def quick_settings(self) -> MovieSettings:
        suffix = self._quick_suffix()
        fps = QUICK_FPS[suffix]
        _, w, h = QUICK_SIZES[self.quick_size.currentIndex()]
        n_target = self.quick_seconds.value() * fps
        s = MovieSettings(width=w, height=h, fps=fps, keep_frames=False,
                          output_file=str(Path(self.quick_file.text().strip()).expanduser().with_suffix(suffix)),
                          style='smoothed' if self.quick_look.currentIndex() == 1 else 'points',
                          scale_bar=True, colour_bar=True, draw_label=True, draw_box=True,
                          point_size=2 if w < 1600 else 3, lazy=True, lazy_particles=self._budget)
        s.highlight_size = s.point_size + 2
        if s.style == 'smoothed':
            s.lazy_particles = min(self._budget, QUICK_SMOOTHED_PARTICLES)
        n = self._n_snapshots
        if self.quick_kind.currentIndex() == 1 and n > 1:
            s.kind = 'evolving'
            s.first, s.last = 0, n - 1
            s.interp_frames = int(min(MAX_QUICK_IN_BETWEEN, max(0, round((n_target - n) / (n - 1)))))
            s.recentre = True
        else:
            s.kind = 'rotating'
            s.nframes = n_target
            s.angle = 360.0
        return s

    # ------------------------------------------------------------------ custom

    def _build_custom(self) -> QWidget:
        d = MovieSettings()
        tabs = QTabWidget()

        # Motion: GadgetViewer's two movie types and the camera path.
        motion = QWidget()
        layout = QVBoxLayout(motion)
        rot = QGroupBox()
        rf = QFormLayout(rot)
        self.radio_rotating = QRadioButton("Rotating movie (current snapshot)")
        self.radio_rotating.setChecked(True)
        self.radio_rotating.toggled.connect(self._update_enabled)
        rf.addRow(self.radio_rotating)
        self.spin_nframes = QSpinBox()
        self.spin_nframes.setRange(1, 100000)
        self.spin_nframes.setValue(d.nframes)
        rf.addRow("Number of frames to generate:", self.spin_nframes)
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-36000, 36000)
        self.spin_angle.setValue(d.angle)
        rf.addRow("Angle to rotate through (deg):", self.spin_angle)
        layout.addWidget(rot)

        evo = QGroupBox()
        ef = QFormLayout(evo)
        self.radio_evolving = QRadioButton("Evolving movie (one frame per snapshot)")
        self.radio_evolving.toggled.connect(self._update_enabled)
        ef.addRow(self.radio_evolving)
        self.combo_first = QComboBox()
        self.combo_last = QComboBox()
        ef.addRow("Use snapshots from:", self.combo_first)
        ef.addRow("to:", self.combo_last)
        self.spin_interp = QSpinBox()
        self.spin_interp.setRange(0, 1000)
        self.spin_interp.setValue(d.interp_frames)
        self.spin_interp.setToolTip("Frames between two snapshots, with particles moved along curves fitted to "
                                    "their positions and velocities in both. Stamped 'interpolated'.")
        ef.addRow("In-between frames per snapshot pair:", self.spin_interp)
        layout.addWidget(evo)

        path = QGroupBox()
        pf = QFormLayout(path)
        self.radio_path = QRadioButton("Camera path through keyframes")
        self.radio_path.toggled.connect(self._update_enabled)
        pf.addRow(self.radio_path)
        self.list_keys = QListWidget()
        self.list_keys.setMaximumHeight(110)
        self.list_keys.itemDoubleClicked.connect(lambda *_: self._show_selected_keyframe())
        pf.addRow(self.list_keys)
        kb = QHBoxLayout()
        self.btn_key_add = QPushButton("Add current view")
        self.btn_key_add.clicked.connect(self.keyframe_requested)
        self.btn_key_show = QPushButton("Show")
        self.btn_key_show.clicked.connect(self._show_selected_keyframe)
        self.btn_key_up = QPushButton("Up")
        self.btn_key_up.clicked.connect(lambda: self._move_keyframe(-1))
        self.btn_key_down = QPushButton("Down")
        self.btn_key_down.clicked.connect(lambda: self._move_keyframe(1))
        self.btn_key_remove = QPushButton("Remove")
        self.btn_key_remove.clicked.connect(self._remove_keyframe)
        self.btn_key_load = QPushButton("Load...")
        self.btn_key_load.setToolTip("Take the keyframes from an earlier movie's _movie.json")
        self.btn_key_load.clicked.connect(self._load_keyframes)
        for b in (self.btn_key_add, self.btn_key_show, self.btn_key_up, self.btn_key_down,
                  self.btn_key_remove, self.btn_key_load):
            kb.addWidget(b)
        pf.addRow(kb)
        self.spin_path_frames = QSpinBox()
        self.spin_path_frames.setRange(2, 100000)
        self.spin_path_frames.setValue(250)
        pf.addRow("Number of frames:", self.spin_path_frames)
        self.chk_ease = QCheckBox("Ease in and out")
        self.chk_ease.setChecked(d.ease)
        pf.addRow(self.chk_ease)
        self.chk_interp_path = QCheckBox("Move particles smoothly between snapshots")
        self.chk_interp_path.setChecked(d.interpolate)
        self.chk_interp_path.setToolTip("Off: each frame shows the nearest snapshot")
        pf.addRow(self.chk_interp_path)
        layout.addWidget(path)
        # The three radio buttons sit in different boxes, so they need a group to be exclusive.
        self.kind_group = QButtonGroup(motion)
        for r in (self.radio_rotating, self.radio_evolving, self.radio_path):
            self.kind_group.addButton(r)

        follow = QGroupBox()
        ff = QFormLayout(follow)
        self.chk_recentre = QCheckBox("Centre frames on selected particles")
        self.chk_recentre.setChecked(d.recentre)
        ff.addRow(self.chk_recentre)
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(0, 1000)
        self.spin_smooth.setValue(d.n_smooth)
        ff.addRow("Snapshots to smooth path over:", self.spin_smooth)
        self.chk_ignore = QCheckBox("Ignore unreadable snapshots")
        self.chk_ignore.setChecked(d.ignore_unreadable)
        ff.addRow(self.chk_ignore)
        layout.addWidget(follow)
        tabs.addTab(motion, "Motion")

        # Look: frame size, particles or smoothed, overlays.
        look = QWidget()
        lf = QFormLayout(look)
        size = QHBoxLayout()
        self.spin_w = QSpinBox()
        self.spin_w.setRange(100, 10000)
        self.spin_w.setValue(d.width)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(100, 10000)
        self.spin_h.setValue(d.height)
        size.addWidget(self.spin_w)
        size.addWidget(QLabel("by"))
        size.addWidget(self.spin_h)
        lf.addRow("Movie dimensions:", size)
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Points", "Smoothed (surface density)"])
        self.combo_style.currentIndexChanged.connect(self._update_enabled)
        lf.addRow("Frames:", self.combo_style)
        self.spin_point = QSpinBox()
        self.spin_point.setRange(1, 20)
        self.spin_point.setValue(d.point_size)
        lf.addRow("Particle size (pixels):", self.spin_point)
        self.combo_table = QComboBox()
        self.combo_table.addItems(list(TABLES))
        self.combo_table.setCurrentText(d.smooth_table)
        lf.addRow("Colour table (smoothed):", self.combo_table)
        self.chk_log = QCheckBox("Log scale")
        self.chk_log.setChecked(d.smooth_log)
        lf.addRow(self.chk_log)
        rng = QHBoxLayout()
        self.chk_auto_range = QCheckBox("From the first frame, then fixed")
        self.chk_auto_range.setChecked(True)
        self.chk_auto_range.toggled.connect(self._update_enabled)
        self.edit_lo = QLineEdit("1e-3")
        self.edit_hi = QLineEdit("1")
        rng.addWidget(self.chk_auto_range)
        rng.addWidget(self.edit_lo)
        rng.addWidget(QLabel("to"))
        rng.addWidget(self.edit_hi)
        lf.addRow("Colour range:", rng)
        self.chk_box = QCheckBox("Draw bounding box")
        self.chk_box.setChecked(d.draw_box)
        lf.addRow(self.chk_box)
        self.chk_label = QCheckBox("Label frames with time / redshift")
        self.chk_label.setChecked(d.draw_label)
        lf.addRow(self.chk_label)
        bar = QHBoxLayout()
        self.chk_scale = QCheckBox("Scale bar in")
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["comoving", "physical"])
        self.combo_scale.setToolTip("Comoving: lengths as stored (per h in cosmological runs). "
                                    "Physical: times a, divided by h. Runs without comoving "
                                    "integration have one kind of length.")
        bar.addWidget(self.chk_scale)
        bar.addWidget(self.combo_scale)
        bar.addWidget(QLabel("units"))
        bar.addStretch()
        lf.addRow(bar)
        self.chk_cbar = QCheckBox("Colour bar (smoothed frames, or colouring by a property)")
        lf.addRow(self.chk_cbar)
        tabs.addTab(look, "Look")

        # Output: frames, encoding, memory.
        out = QWidget()
        of = QFormLayout(out)
        path = QHBoxLayout()
        self.edit_base = QLineEdit(str(Path.home() / "Movies" / "cosmoviewer" / "frame"))
        path.addWidget(self.edit_base)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        path.addWidget(browse)
        of.addRow("Path and base name for frames:", path)
        enc = QHBoxLayout()
        self.chk_mp4 = QCheckBox("Also encode MP4")
        self.chk_gif = QCheckBox("GIF")
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(d.fps)
        enc.addWidget(self.chk_mp4)
        enc.addWidget(self.chk_gif)
        enc.addWidget(QLabel("at"))
        enc.addWidget(self.spin_fps)
        enc.addWidget(QLabel("fps"))
        if not can_write_mp4():
            self.chk_mp4.setEnabled(False)
            self.chk_mp4.setToolTip("Neither ffmpeg nor OpenCV was found; PNG frames are still written")
        if not can_write_gif():
            self.chk_gif.setEnabled(False)
        of.addRow(enc)
        lazy = QHBoxLayout()
        self.chk_lazy = QCheckBox("Read each snapshot from disk while rendering, with")
        self.chk_lazy.setChecked(True)
        self.chk_lazy.setToolTip("Frames get this many particles whatever was loaded, and only a few "
                                 "snapshots are in memory at a time. Off: use the loaded samples.")
        self.chk_lazy.toggled.connect(self._update_enabled)
        self.spin_lazy = QSpinBox()
        self.spin_lazy.setRange(1000, 100_000_000)
        self.spin_lazy.setSingleStep(500_000)
        self.spin_lazy.setValue(self._budget)
        lazy.addWidget(self.chk_lazy)
        lazy.addWidget(self.spin_lazy)
        lazy.addWidget(QLabel("particles"))
        of.addRow(lazy)
        tabs.addTab(out, "Output")
        return tabs

    # ------------------------------------------------------------------ keyframes

    def add_keyframe(self, kf: Keyframe):
        self.keyframes.append(kf)
        self._refresh_keys(len(self.keyframes) - 1)
        self.radio_path.setChecked(True)

    def _refresh_keys(self, select: int = -1):
        self.list_keys.clear()
        for i, k in enumerate(self.keyframes):
            c = k.camera
            self.list_keys.addItem(f"{i + 1}. snapshot {k.name or k.snapshot}, distance {c['distance']:.3g}, "
                                   f"elevation {c['elevation']:.0f}, azimuth {c['azimuth']:.0f}")
        if 0 <= select < len(self.keyframes):
            self.list_keys.setCurrentRow(select)
        self._update_enabled()

    def _show_selected_keyframe(self):
        r = self.list_keys.currentRow()
        if 0 <= r < len(self.keyframes):
            self.keyframe_view_requested.emit(self.keyframes[r])

    def _move_keyframe(self, step: int):
        r = self.list_keys.currentRow()
        j = r + step
        if 0 <= r < len(self.keyframes) and 0 <= j < len(self.keyframes):
            self.keyframes[r], self.keyframes[j] = self.keyframes[j], self.keyframes[r]
            self._refresh_keys(j)

    def _remove_keyframe(self):
        r = self.list_keys.currentRow()
        if 0 <= r < len(self.keyframes):
            del self.keyframes[r]
            self._refresh_keys(min(r, len(self.keyframes) - 1))

    def _load_keyframes(self):
        f, _ = QFileDialog.getOpenFileName(self, "Keyframes from a movie's settings", str(Path(self.edit_base.text()).parent),
                                           "Movie settings (*_movie.json)")
        if not f:
            return
        try:
            with open(f) as fh:
                keys = [Keyframe.from_dict(k) for k in json.load(fh)['settings'].get('keyframes', [])]
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, "Load keyframes", f"Could not read keyframes from {f}: {e}")
            return
        if not keys:
            QMessageBox.information(self, "Load keyframes", "That movie has no keyframes.")
            return
        bad = [k.snapshot for k in keys if k.snapshot >= self._n_snapshots]
        if bad:
            QMessageBox.warning(self, "Load keyframes", f"Keyframe snapshot {bad[0]} is not loaded now.")
        self.keyframes = keys
        self._refresh_keys(0)
        self.radio_path.setChecked(True)

    # ------------------------------------------------------------------ common

    def set_snapshots(self, names: List[str], current: int):
        self._n_snapshots = len(names)
        for combo, default in ((self.combo_first, 0), (self.combo_last, len(names) - 1)):
            old = combo.currentIndex()
            combo.clear()
            combo.addItems([f"{i}: {Path(n).name}" for i, n in enumerate(names)])
            combo.setCurrentIndex(old if 0 <= old < len(names) else default)
        self.radio_evolving.setEnabled(len(names) > 1)
        self._update_enabled()

    def set_particle_budget(self, n: int):
        self._budget = int(n)
        self.spin_lazy.setValue(int(n))

    def _browse(self):
        f, _ = QFileDialog.getSaveFileName(self, "Choose base name for output images", self.edit_base.text())
        if f:
            self.edit_base.setText(str(Path(f).with_suffix('')))

    def _update_enabled(self):
        rot = self.radio_rotating.isChecked()
        evo = self.radio_evolving.isChecked()
        pth = self.radio_path.isChecked()
        for w in (self.spin_nframes, self.spin_angle):
            w.setEnabled(rot)
        for w in (self.combo_first, self.combo_last, self.spin_interp):
            w.setEnabled(evo)
        for w in (self.chk_recentre, self.spin_smooth, self.chk_ignore):
            w.setEnabled(not rot)
        for w in (self.list_keys, self.btn_key_show, self.btn_key_up, self.btn_key_down, self.btn_key_remove,
                  self.spin_path_frames, self.chk_ease, self.chk_interp_path):
            w.setEnabled(pth)
        smooth = self.combo_style.currentIndex() == 1
        for w in (self.combo_table, self.chk_log, self.chk_auto_range):
            w.setEnabled(smooth)
        for w in (self.edit_lo, self.edit_hi):
            w.setEnabled(smooth and not self.chk_auto_range.isChecked())
        self.spin_point.setEnabled(not smooth)
        self.spin_lazy.setEnabled(self.chk_lazy.isChecked())

    def settings(self) -> MovieSettings:
        """Settings of the Custom tab."""
        kind = 'rotating' if self.radio_rotating.isChecked() else \
            'evolving' if self.radio_evolving.isChecked() else 'keyframes'
        rng = None
        if not self.chk_auto_range.isChecked():
            try:
                rng = (float(self.edit_lo.text()), float(self.edit_hi.text()))
            except ValueError:
                rng = (0.0, 0.0)                         # rejected by validate() with a message
        return MovieSettings(
            kind=kind,
            nframes=self.spin_path_frames.value() if kind == 'keyframes' else self.spin_nframes.value(),
            angle=self.spin_angle.value(),
            first=max(0, self.combo_first.currentIndex()), last=self.combo_last.currentIndex(),
            recentre=self.chk_recentre.isChecked(), n_smooth=self.spin_smooth.value(),
            ignore_unreadable=self.chk_ignore.isChecked(), interp_frames=self.spin_interp.value(),
            keyframes=[k.to_dict() for k in self.keyframes], interpolate=self.chk_interp_path.isChecked(),
            ease=self.chk_ease.isChecked(),
            width=self.spin_w.value(), height=self.spin_h.value(), basename=self.edit_base.text().strip(),
            point_size=self.spin_point.value(), highlight_size=self.spin_point.value() + 2,
            draw_box=self.chk_box.isChecked(), draw_label=self.chk_label.isChecked(),
            style='smoothed' if self.combo_style.currentIndex() == 1 else 'points',
            smooth_log=self.chk_log.isChecked(), smooth_table=self.combo_table.currentText(), smooth_range=rng,
            scale_bar=self.chk_scale.isChecked(), scale_units=self.combo_scale.currentText(),
            colour_bar=self.chk_cbar.isChecked(),
            lazy=self.chk_lazy.isChecked(), lazy_particles=self.spin_lazy.value(),
            encode_mp4=self.chk_mp4.isChecked(), encode_gif=self.chk_gif.isChecked(), fps=self.spin_fps.value())

    def current_settings(self) -> MovieSettings:
        return self.quick_settings() if self.tabs.currentIndex() == 0 else self.settings()

    def _make(self):
        self.movie_requested.emit(self.current_settings())

    def set_running(self, running: bool, text: str = ""):
        self.btn_make.setEnabled(not running)
        if text:
            self.lbl_status.setText(text)
