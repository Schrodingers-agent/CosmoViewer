"""
Movies: rotating and evolving movies after GadgetViewer's movie.f90, plus
camera paths through keyframes, in-between frames, smoothed frames and
burnt-in overlays.

* Rotating movie: ``nframes`` frames of the current snapshot, turning the
  camera through ``angle`` degrees about the view centre.
* Evolving movie: one frame per snapshot from ``first`` to ``last``,
  optionally recentred on the selected particles in every snapshot, with the
  centre smoothed over the last ``n_smooth`` frames. Unreadable snapshots can
  be skipped.

Differences from GadgetViewer, on purpose:
* frames are numbered consecutively (``<base>_00000.png``, ...) for both kinds,
  so they can be encoded directly; the snapshot of each frame is recorded in
  ``<base>_movie.json`` together with every setting and the camera, so a movie
  can be reproduced;
* the selected particles are found in every snapshot at full resolution (by
  ParticleID), so recentring does not depend on the display sample;
* in a periodic box the smoothed centre is averaged across the boundary
  correctly;
* the rotation turns about the scene's vertical (z) axis, like dragging in the
  3D view, rather than the screen's vertical axis;
* frames can optionally be encoded to MP4 or GIF with ffmpeg if it is installed;
  without ffmpeg, MP4 falls back to OpenCV (H.264 through AVFoundation on
  macOS) and GIF to Pillow.

Beyond GadgetViewer:
* Camera path (``kind='keyframes'``): the camera passes through stored views
  on a spline (movie/keyframes.py); the snapshot can change along the path.
* In-between frames: ``interp_frames`` per pair of snapshots in an evolving
  movie, and fractional snapshot positions on a camera path, are drawn with
  particles moved along curves fitted to both snapshots (movie/interpolate.py).
  They are labelled "interpolated" on the frame and in the JSON. Frames at a
  snapshot are drawn from the snapshot itself.
* ``style='smoothed'``: SPH-smoothed surface density frames with one colour
  range for the whole movie (movie/smoothed.py).
* Overlays: time or redshift, a scale bar (comoving or physical), a colour bar
  (movie/overlays.py).
* Memory: scenes are requested one snapshot at a time and at most a few are
  kept, so the caller can read snapshots lazily from disk (the controller
  does with ``lazy``).
* Quick movies (``keep_frames=False``): frames go to a temporary folder that is
  deleted after encoding the one .mp4 or .gif asked for.
"""

import json
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from math import floor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.tracking import smooth_centres
from movie.renderer import Camera, box_lines, render_frame

KINDS = ('rotating', 'evolving', 'keyframes')
STYLES = ('points', 'smoothed')


@dataclass
class MovieSettings:
    kind: str = 'rotating'               # 'rotating', 'evolving' or 'keyframes'
    nframes: int = 200                   # rotating and keyframes
    angle: float = 360.0                 # rotating, degrees
    first: int = 0                       # evolving, index into the loaded snapshots
    last: int = -1                       # evolving, inclusive; -1 = last loaded
    recentre: bool = True                # evolving, keyframes: centre frames on selected particles
    n_smooth: int = 10                   # snapshots to smooth the centre over
    ignore_unreadable: bool = True       # evolving, keyframes
    interp_frames: int = 0               # evolving: in-between frames per pair of snapshots
    keyframes: List[Dict] = field(default_factory=list)   # keyframes: Keyframe.to_dict() each
    interpolate: bool = True             # keyframes: move particles between snapshots
    ease: bool = True                    # keyframes: start and stop gently
    width: int = 640
    height: int = 480
    basename: str = './frame'
    point_size: int = 2
    highlight_size: int = 4
    draw_box: bool = True
    draw_label: bool = True
    style: str = 'points'                # 'points' or 'smoothed'
    smooth_log: bool = True
    smooth_table: str = 'magma'
    smooth_range: Optional[Tuple[float, float]] = None   # None: from the first frame, then fixed
    scale_bar: bool = False
    scale_units: str = 'comoving'        # 'comoving' or 'physical'
    colour_bar: bool = False
    lazy: bool = False                   # read each snapshot from disk while rendering
    lazy_particles: int = 2_000_000      # sample size of snapshots read while rendering
    encode_mp4: bool = False
    encode_gif: bool = False
    fps: int = 25
    keep_frames: bool = True             # False: frames go to a temporary folder (quick movies)
    output_file: str = ''                # quick movies: the .mp4 or .gif to write

    def validate(self, n_snapshots: int) -> Optional[str]:
        """Error message, or None. Limits as in GadgetViewer's movie window."""
        if self.kind not in KINDS:
            return f"unknown movie type {self.kind!r}"
        if self.style not in STYLES:
            return f"unknown frame style {self.style!r}"
        if self.kind in ('rotating', 'keyframes') and self.nframes < 1:
            return "Number of frames must be positive"
        if self.kind == 'evolving':
            last = n_snapshots - 1 if self.last < 0 else self.last
            if self.first < 0 or last >= n_snapshots:
                return f"Snapshots must be in the range 0-{n_snapshots - 1}"
            if self.first > last:
                return "First snapshot must be before last snapshot"
            if not 0 <= self.interp_frames <= 1000:
                return "In-between frames must be in the range 0-1000"
        if self.kind in ('evolving', 'keyframes') and not 0 <= self.n_smooth <= 1000:
            return "Number of frames to smooth path over must be in range 0-1000"
        if self.kind == 'keyframes':
            if len(self.keyframes) < 2:
                return "A camera path needs at least two keyframes"
            if self.nframes < 2:
                return "A camera path needs at least two frames"
            bad = [k['snapshot'] for k in self.keyframes if not 0 <= int(k['snapshot']) < n_snapshots]
            if bad:
                return f"Keyframe snapshot {bad[0]} is not loaded"
        if not (100 <= self.width <= 10000 and 100 <= self.height <= 10000):
            return "Image dimensions must be in range 100-10000"
        if self.keep_frames and not self.basename.strip():
            return "You need to specify a path to write the images to"
        if not self.keep_frames:
            if Path(self.output_file).suffix.lower() not in ('.mp4', '.gif'):
                return "The movie file must end in .mp4 or .gif"
        if self.fps < 1:
            return "Frame rate must be positive"
        if self.smooth_range is not None:
            lo, hi = self.smooth_range
            if not hi > lo or (self.smooth_log and lo <= 0):
                return "The colour range needs maximum > minimum (and minimum > 0 on a log scale)"
        if self.lazy and self.lazy_particles < 1000:
            return "Particles per snapshot must be at least 1000"
        return None

    def frame_count(self, n_snapshots: int) -> int:
        if self.kind in ('rotating', 'keyframes'):
            return self.nframes
        last = n_snapshots - 1 if self.last < 0 else self.last
        n = last - self.first + 1
        return n + max(n - 1, 0) * self.interp_frames


@dataclass
class Scene:
    """What to draw for one snapshot, in code units."""
    pos: np.ndarray
    colors: np.ndarray
    highlight_pos: Optional[np.ndarray] = None
    highlight_colors: Optional[np.ndarray] = None
    centre: Optional[np.ndarray] = None      # centre of the selected particles, if any
    label: str = ''
    box_min: Optional[np.ndarray] = None
    box_size: Optional[np.ndarray] = None
    box_period: float = 0.0                  # BoxSize if periodic
    lines: List[Tuple[np.ndarray, Sequence[float]]] = field(default_factory=list)
    # Colouring by a property: particles with a value ("lit") are a second layer drawn over pos.
    lit_pos: Optional[np.ndarray] = None
    lit_colors: Optional[np.ndarray] = None
    lit_solid: bool = False                  # nearest lit particle in front, instead of adding up
    ids: Optional[np.ndarray] = None         # ParticleIDs of pos / lit_pos (for picking on screen)
    lit_ids: Optional[np.ndarray] = None
    lit_types: Optional[np.ndarray] = None
    lit_values: Optional[np.ndarray] = None  # the colouring property's value of each lit particle
    # For in-between frames, smoothed frames and overlays (movie/interpolate.py, smoothed.py, overlays.py).
    vel: Optional[np.ndarray] = None         # stored velocities of pos / lit_pos
    lit_vel: Optional[np.ndarray] = None
    mass: Optional[np.ndarray] = None        # code masses of pos / lit_pos
    lit_mass: Optional[np.ndarray] = None
    hsml: Optional[np.ndarray] = None        # smoothing lengths, filled in by smoothed frames
    lit_hsml: Optional[np.ndarray] = None
    mass_scale: float = 1.0                  # N_total / N_sample
    highlight_ids: Optional[np.ndarray] = None
    highlight_vel: Optional[np.ndarray] = None
    time: float = 0.0                        # header Time (a for comoving runs)
    redshift: Optional[float] = None         # None for runs without comoving integration
    dxdt: Optional[float] = None             # d(stored position)/d(time) per stored velocity
    files: List[str] = field(default_factory=list)
    length: Optional[object] = None          # overlays.LengthInfo, for the scale bar
    colour_bar: Optional[Dict] = None        # {'table', 'lo', 'hi', 'log', 'title'} when colouring by a value
    interpolated: bool = False


@dataclass
class MovieResult:
    success: bool
    message: str
    frames: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    mp4: Optional[str] = None
    params_file: Optional[str] = None
    outputs: List[str] = field(default_factory=list)     # encoded .mp4 / .gif files
    smooth_range: Optional[List[float]] = None           # colour range of smoothed frames


def frame_path(basename: str, k: int) -> str:
    return f"{basename}_{k:05d}.png"


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg                      # optional
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _cv2():
    try:
        import cv2                                 # optional MP4 fallback
        return cv2
    except Exception:
        return None


def can_write_mp4() -> bool:
    return find_ffmpeg() is not None or _cv2() is not None


def _encode_mp4_cv2(cv2, base: str, fps: int, out_path: str) -> Optional[str]:
    """Encode <base>_%05d.png to H.264 MP4 with OpenCV, padding odd sizes with black like ffmpeg."""
    frames = sorted(Path(base).parent.glob(Path(base).name + '_[0-9][0-9][0-9][0-9][0-9].png'))
    if not frames:
        return "MP4 encoding failed: no frames were found"
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    H, W = h + h % 2, w + w % 2
    writer = None
    for cc in ('avc1', 'mp4v'):                    # H.264 first; MPEG-4 Part 2 if H.264 is unavailable
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), float(fps), (W, H))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        return "MP4 encoding failed: OpenCV could not open an MP4 writer"
    try:
        for f in frames:
            img = cv2.imread(str(f))
            if img is None or img.shape[:2] != (h, w):
                return f"MP4 encoding failed: frame {f.name} could not be read or has a different size"
            if (H, W) != (h, w):
                img = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            writer.write(img)
    finally:
        writer.release()
    if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
        return "MP4 encoding failed: OpenCV wrote no data"
    return None


def can_write_gif() -> bool:
    if find_ffmpeg():
        return True
    try:
        import PIL                                 # noqa: F401  optional fallback
        return True
    except ImportError:
        return False


def write_png(path: str, rgb: np.ndarray, label: str = '', bar=None, colour_bar=None) -> bool:
    """Write an RGB uint8 image with Qt (already a dependency), with the overlays burnt in."""
    from PyQt6.QtGui import QImage
    from movie.overlays import draw_overlays
    h, w = rgb.shape[:2]
    data = np.ascontiguousarray(rgb)
    img = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    if label or bar or colour_bar:
        draw_overlays(img, label, bar, colour_bar)
    return img.save(path, 'PNG')


def encode(base: str, fps: int, out_path: str) -> Optional[str]:
    """Encode <base>_%05d.png to an .mp4 or .gif. Returns an error message, or None."""
    exe = find_ffmpeg()
    gif = out_path.lower().endswith('.gif')
    if exe is None:
        if not gif:
            cv2 = _cv2()
            if cv2 is not None:
                return _encode_mp4_cv2(cv2, base, fps, out_path)
            return ("Neither ffmpeg nor OpenCV was found, so no MP4 was made; encode later with: "
                    f"ffmpeg -framerate {fps} -i '{base}_%05d.png' -pix_fmt yuv420p movie.mp4")
        try:
            from PIL import Image
        except ImportError:
            return "Neither ffmpeg nor Pillow was found, so no GIF was made"
        frames = sorted(Path(base).parent.glob(Path(base).name + '_*.png'))
        imgs = [Image.open(f).convert('P', palette=Image.Palette.ADAPTIVE) for f in frames]
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], loop=0,
                     duration=max(20, int(round(1000 / fps))))
        return None
    if gif:
        vf = 'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4'
        cmd = [exe, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f"{base}_%05d.png",
               '-vf', vf, '-loop', '0', out_path]
    else:
        cmd = [exe, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f"{base}_%05d.png",
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"{'GIF' if gif else 'MP4'} encoding failed: {r.stderr.strip()[:300]}"
    return None


class _Source:
    """
    Scenes by snapshot position. Whole positions are the snapshots themselves;
    fractional ones are in-between frames. At most a few scenes are kept, so a
    long movie needs memory for a couple of snapshots, whatever its length.
    """
    KEEP = 3

    def __init__(self, scene_for: Callable[[int], 'Scene'], locator=None):
        self.scene_for = scene_for
        self.cache: 'OrderedDict[int, Scene]' = OrderedDict()
        self.bad: Dict[int, str] = {}
        self.pair_key = None
        self.pair = None
        self.locator = locator
        self.dropped = 0
        self.limited: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def scene(self, i: int) -> 'Scene':
        if i in self.bad:
            raise RuntimeError(self.bad[i])
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        try:
            sc = self.scene_for(i)
        except Exception as e:
            self.bad[i] = f"{type(e).__name__}: {e}"
            raise
        self.cache[i] = sc
        while len(self.cache) > self.KEEP:
            self.cache.popitem(last=False)
        return sc

    def at(self, s: float, interpolate: bool) -> Tuple['Scene', int, float]:
        """(scene, snapshot index, fraction towards the next one)."""
        i = int(floor(s + 1e-9))
        f = s - i
        if f < 1e-6 or not interpolate:
            k = int(floor(s + 0.5))                # nearest snapshot (halves go forward)
            return self.scene(k), k, 0.0
        if f > 1 - 1e-6:
            return self.scene(i + 1), i + 1, 0.0
        if self.pair_key != (i, i + 1):
            from movie.interpolate import PairInterpolator
            a, b = self.scene(i), self.scene(i + 1)
            self.pair = None
            self.pair = PairInterpolator(a, b) if self.locator is None else PairInterpolator(a, b, self.locator)
            self.pair_key = (i, i + 1)
            self.dropped = max(self.dropped, self.pair.dropped)
            self.limited[self.pair_key] = (self.pair.limited, self.pair.n_curves)
        return self.pair.scene_at(f), i, f


def _plan(s: MovieSettings, camera: Camera, current_index: int, n_snap: int) -> List[Tuple[Camera, float]]:
    """(camera before recentring, snapshot position) for every frame."""
    if s.kind == 'rotating':
        d_az = s.angle / s.nframes
        return [(camera.rotated(k * d_az), float(current_index)) for k in range(s.nframes)]
    if s.kind == 'keyframes':
        from movie.keyframes import Keyframe, camera_path
        return camera_path([Keyframe.from_dict(k) for k in s.keyframes], s.nframes, s.ease)
    last = n_snap - 1 if s.last < 0 else s.last
    out = []
    for i in range(s.first, last + 1):
        out.append((camera, float(i)))
        if i < last:
            for j in range(1, s.interp_frames + 1):
                out.append((camera, i + j / (s.interp_frames + 1)))
    return out


class _Painter:
    """Turns a scene and a camera into an image, holding what must stay fixed across frames."""

    def __init__(self, s: MovieSettings):
        self.s = s
        self.range = None if s.smooth_range is None else tuple(map(float, s.smooth_range))

    def render(self, scene: 'Scene', cam: Camera) -> np.ndarray:
        s = self.s
        lines = list(scene.lines)
        if s.draw_box and scene.box_min is not None and scene.box_size is not None:
            lines.insert(0, (box_lines(scene.box_min, scene.box_size), (0.2, 0.2, 0.2, 1.0)))
        if s.style == 'smoothed':
            from movie.smoothed import choose_range, colourise, density_image
            img = density_image(scene, cam, s.width, s.height)
            if self.range is None:
                self.range = choose_range(img, s.smooth_log)
            bg = colourise(img, self.range[0], self.range[1], s.smooth_log, s.smooth_table)
            # The smoothed image is orthographic; draw the box, trails and selected particles the
            # same way, with a camera so far away (same frame width) that perspective is negligible.
            return render_frame(s.width, s.height, _orthographic(cam), None, None, scene.highlight_pos,
                                scene.highlight_colors, lines, point_size=s.point_size,
                                highlight_size=s.highlight_size, background=bg)
        return render_frame(s.width, s.height, cam, scene.pos, scene.colors,
                            scene.highlight_pos, scene.highlight_colors, lines,
                            point_size=s.point_size, highlight_size=s.highlight_size,
                            lit_pos=scene.lit_pos, lit_colors=scene.lit_colors, lit_solid=scene.lit_solid)

    def overlays(self, scene: 'Scene', cam: Camera):
        s = self.s
        bar = cbar = None
        if s.scale_bar and scene.length is not None:
            from movie.overlays import frame_width, scale_bar
            info = scene.length
            if scene.interpolated and info.comoving:
                info = replace(info, a=float(scene.time))
            bar = scale_bar(frame_width(cam), s.width, info, s.scale_units, perspective=s.style == 'points')
        if s.colour_bar:
            if s.style == 'smoothed' and self.range is not None:
                from movie.smoothed import TITLE
                cbar = {'table': s.smooth_table, 'lo': self.range[0], 'hi': self.range[1],
                        'log': s.smooth_log, 'title': TITLE}
            elif s.style == 'points':
                cbar = scene.colour_bar
        return bar, cbar


ORTHO_FACTOR = 1000.0


def _orthographic(cam: Camera) -> Camera:
    """A camera ORTHO_FACTOR times further away with the same frame width at the view centre."""
    from math import atan, degrees, radians, tan
    w = 2.0 * cam.distance * tan(radians(cam.fov) / 2.0)
    d = cam.distance * ORTHO_FACTOR
    return replace(cam, distance=d, fov=2.0 * degrees(atan(w / (2.0 * d))))


def _render(scene: 'Scene', cam: Camera, s: MovieSettings) -> np.ndarray:
    return _Painter(s).render(scene, cam)


def make_movie(settings: MovieSettings, camera: Camera, current_index: int,
               scene_for: Callable[[int], 'Scene'], snapshot_names: Sequence[str],
               progress: Callable[[int, int, str], None] = lambda *a: None,
               cancelled: Callable[[], bool] = lambda: False,
               extra_params: Optional[dict] = None, locator=None) -> MovieResult:
    s = settings
    n_snap = len(snapshot_names)
    err = s.validate(n_snap)
    if err:
        return MovieResult(False, err)
    tmp_dir = None
    if s.keep_frames:
        base = str(Path(s.basename).expanduser().resolve())
    else:
        out = Path(s.output_file).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix='cosmoviewer_movie_')
        base = str(Path(tmp_dir) / 'frame')
    out_dir = Path(base).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _make(s, camera, current_index, scene_for, snapshot_names, progress, cancelled,
                     extra_params, locator, base, out_dir)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _make(s, camera, current_index, scene_for, snapshot_names, progress, cancelled, extra_params,
          locator, base, out_dir) -> MovieResult:
    n_snap = len(snapshot_names)
    plan = _plan(s, camera, current_index, n_snap)
    source = _Source(scene_for, locator)
    painter = _Painter(s)
    interpolate = s.kind == 'evolving' or s.interpolate
    follow = s.recentre and s.kind != 'rotating'
    # Smooth the followed centre over the same stretch of time however many in-between frames.
    window = s.n_smooth * ((s.interp_frames + 1) if s.kind == 'evolving' else 1)
    frames, skipped, record = [], [], []
    raw_centres: List[np.ndarray] = []
    centre = np.asarray(camera.center, dtype=np.float64)
    t0 = time.time()
    n_interp = 0

    for n_done, (cam0, pos_s) in enumerate(plan):
        if cancelled():
            return MovieResult(False, "Cancelled by user", _kept(frames, s), skipped)
        try:
            scene, i, f = source.at(pos_s, interpolate)
        except Exception as e:
            k0 = int(floor(pos_s + 1e-9))
            j = next((j for j in (k0, k0 + 1) if j in source.bad), k0)
            bad = source.bad.get(j, f"{type(e).__name__}: {e}")
            name = snapshot_names[min(j, n_snap - 1)]
            if s.ignore_unreadable and s.kind != 'rotating':
                msg = f"{name}: {bad}"
                if msg not in skipped:
                    skipped.append(msg)
                progress(n_done + 1, len(plan), f"skipped {Path(name).name}")
                continue
            return MovieResult(False, f"{name}: {bad}", _kept(frames, s), skipped)
        cam = cam0
        if follow:
            if scene.centre is not None:
                centre = np.asarray(scene.centre, dtype=np.float64)
            elif s.kind == 'keyframes':
                centre = np.asarray(cam0.center, dtype=np.float64)
            raw_centres.append(centre.copy())
            use = smooth_centres(raw_centres, len(raw_centres) - 1, window, scene.box_period) \
                if window > 0 else centre
            cam = cam0.centred(use)
        k = len(frames)
        path = frame_path(base, k)
        rgb = painter.render(scene, cam)
        bar, cbar = painter.overlays(scene, cam)
        if not write_png(path, rgb, scene.label if s.draw_label else '', bar, cbar):
            return MovieResult(False, f"Unable to write PNG file {path}", _kept(frames, s), skipped)
        frames.append(path)
        n_interp += int(f > 0)
        record.append({'frame': k, 'snapshot': snapshot_names[i], 'snapshot_index': i,
                       'position': round(float(pos_s), 6), 'interpolated': f > 0,
                       'fraction': round(float(f), 6), 'time': float(scene.time), 'label': scene.label,
                       'camera': cam.to_dict(),
                       'selection_centre': None if scene.centre is None else list(map(float, scene.centre))})
        tag = f" (in-between {f:.2f})" if f > 0 else ''
        progress(n_done + 1, len(plan), f"frame {k + 1}/{len(plan)}: {Path(snapshot_names[i]).name}{tag}")
    if not frames:
        return MovieResult(False, "Unable to read any snapshots", [], skipped)

    params_file = None
    used_range = None if painter.range is None else list(map(float, painter.range))
    if s.keep_frames:
        params_file = f"{base}_movie.json"
        with open(params_file, 'w') as fh:
            json.dump({'settings': asdict(s), 'start_camera': camera.to_dict(), 'frames': record,
                       'skipped': skipped, 'interpolated_frames': n_interp,
                       'interpolation': ("in-between frames: particles of the earlier snapshot on cubic Hermite "
                                         "curves fitted to positions and velocities at both snapshots, matched "
                                         "by ParticleID; pictures only") if n_interp else None,
                       'particles_dropped_in_between': source.dropped,
                       'tangents_shortened': {f"{Path(snapshot_names[i]).name} -> {Path(snapshot_names[j]).name}":
                                              f"{n} of {tot} particles" for (i, j), (n, tot) in source.limited.items()},
                       'smooth_range_used': used_range,
                       'seconds': round(time.time() - t0, 2), **(extra_params or {})}, fh, indent=1)

    msg = f"Wrote {len(frames)} frames to {out_dir}" if s.keep_frames else f"Rendered {len(frames)} frames"
    if n_interp:
        msg += f" ({n_interp} in-between)"
    if skipped:
        msg += f" ({len(skipped)} snapshots skipped)"
    far = [f"{Path(snapshot_names[i]).name} -> {Path(snapshot_names[j]).name}"
           for (i, j), (n, tot) in source.limited.items() if tot and n > tot / 2]
    if far:
        msg += (f". Snapshots too far apart for smooth in-between frames ({', '.join(far)}): most particles "
                f"turned more than the curve can follow, so they move nearly straight there")
    outputs = []
    if s.keep_frames:
        if s.encode_mp4:
            outputs.append(f"{base}.mp4")
        if s.encode_gif:
            outputs.append(f"{base}.gif")
    else:
        outputs.append(str(Path(s.output_file).expanduser().resolve()))
    made = []
    for out in outputs:
        e = encode(base, s.fps, out)
        if e:
            msg += f". {e}"
            if not s.keep_frames:
                return MovieResult(False, msg, [], skipped)
        else:
            made.append(out)
            msg += f"; wrote {Path(out).name}" if s.keep_frames else f"; wrote {out}"
    mp4 = next((m for m in made if m.lower().endswith('.mp4')), None)
    return MovieResult(True, msg, _kept(frames, s), skipped, mp4, params_file, made, used_range)


def _kept(frames: List[str], s: MovieSettings) -> List[str]:
    return frames if s.keep_frames else []
