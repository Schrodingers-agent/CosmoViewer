"""
Off-screen software renderer for movie frames.

GadgetViewer renders movie frames with its own plotter, independent of the
window (movie.f90 calls plotter_make_image with an image size and a copy of
the view transform). This does the same with numpy, so frames have a fixed
size, do not depend on the OpenGL window or its context, and can be made in a
worker thread or headless.

The camera reproduces pyqtgraph's GLViewWidget (center, distance, elevation,
azimuth, horizontal field of view), so a movie starts from what is on screen.
Particles are added like the on-screen 'additive' blending; selected particles
are drawn opaque on top, as on screen.
"""

from dataclasses import dataclass, replace
from math import radians, tan
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class Camera:
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 100.0
    elevation: float = 30.0          # degrees, pyqtgraph default
    azimuth: float = 45.0            # degrees, pyqtgraph default
    fov: float = 60.0                # horizontal, degrees

    def rotated(self, d_azimuth: float) -> 'Camera':
        return replace(self, azimuth=self.azimuth + d_azimuth)

    def centred(self, center) -> 'Camera':
        return replace(self, center=tuple(float(x) for x in center))

    def to_dict(self) -> dict:
        return {'center': list(self.center), 'distance': self.distance, 'elevation': self.elevation,
                'azimuth': self.azimuth, 'fov': self.fov}


def _rot(angle_deg: float, axis) -> np.ndarray:
    """4x4 rotation matrix as QMatrix4x4.rotate(angle, axis) builds it."""
    x, y, z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    a = radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    m = np.eye(4)
    m[:3, :3] = [[x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, z * z * C + c]]
    return m


def _trans(dx, dy, dz) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (dx, dy, dz)
    return m


def view_matrix(cam: Camera) -> np.ndarray:
    """World -> eye transform, identical to GLViewWidget.viewMatrix() (euler mode)."""
    cx, cy, cz = cam.center
    return (_trans(0.0, 0.0, -cam.distance) @ _rot(cam.elevation - 90, (1, 0, 0))
            @ _rot(cam.azimuth + 90, (0, 0, -1)) @ _trans(-cx, -cy, -cz))


def project(points: np.ndarray, cam: Camera, width: int, height: int):
    """
    Pixel coordinates (x right, y down) and eye-space depth of ``points``.
    Returns (px, py, depth, visible) where visible excludes points behind the camera.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    m = view_matrix(cam)
    eye = p @ m[:3, :3].T + m[:3, 3]
    depth = -eye[:, 2]
    visible = depth > cam.distance * 1e-3              # pyqtgraph's near clip
    with np.errstate(divide='ignore', invalid='ignore'):
        tx = tan(0.5 * radians(cam.fov))
        ty = tx * height / width
        xn = eye[:, 0] / (depth * tx)
        yn = eye[:, 1] / (depth * ty)
    px = (xn + 1.0) * 0.5 * width
    py = (1.0 - yn) * 0.5 * height
    visible &= np.isfinite(px) & np.isfinite(py)
    return px, py, depth, visible


def _footprint(size: int):
    r = max(size, 1) / 2.0
    k = int(np.ceil(r))
    offs = [(dx, dy) for dy in range(-k, k + 1) for dx in range(-k, k + 1)
            if (dx * dx + dy * dy) <= r * r + 0.25]
    return offs or [(0, 0)]


def _splat(img, px, py, rgb, size: int, additive: bool):
    h, w = img.shape[:2]
    ix0 = np.floor(px).astype(np.int64)
    iy0 = np.floor(py).astype(np.int64)
    for dx, dy in _footprint(size):
        ix, iy = ix0 + dx, iy0 + dy
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        if not ok.any():
            continue
        flat = iy[ok] * w + ix[ok]
        if additive:
            out = img.reshape(-1, 3)                      # view: img is contiguous
            for ch in range(3):
                out[:, ch] += np.bincount(flat, weights=rgb[ok, ch], minlength=h * w).astype(np.float32)
        else:
            img.reshape(-1, 3)[flat] = rgb[ok]


def _splat_over(img, px, py, rgb, alpha, size: int):
    """Alpha blending over what is drawn (dst = src a + dst (1 - a)). Where points of one
    call overlap, the one written last (nearest, after sorting) wins: an approximation."""
    h, w = img.shape[:2]
    ix0 = np.floor(px).astype(np.int64)
    iy0 = np.floor(py).astype(np.int64)
    out = img.reshape(-1, 3)
    for dx, dy in _footprint(size):
        ix, iy = ix0 + dx, iy0 + dy
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        if ok.any():
            flat = iy[ok] * w + ix[ok]
            out[flat] = out[flat] * (1.0 - alpha[ok]) + rgb[ok] * alpha[ok]


def render_frame(width: int, height: int, cam: Camera,
                 pos: Optional[np.ndarray] = None, colors: Optional[np.ndarray] = None,
                 highlight_pos: Optional[np.ndarray] = None, highlight_colors: Optional[np.ndarray] = None,
                 lines: Sequence[Tuple[np.ndarray, Sequence[float]]] = (),
                 point_size: int = 2, highlight_size: int = 4,
                 background=(0.0, 0.0, 0.0),
                 lit_pos: Optional[np.ndarray] = None, lit_colors: Optional[np.ndarray] = None,
                 lit_solid: bool = False) -> np.ndarray:
    """
    Render one frame. ``colors``/``highlight_colors`` are RGBA in [0, 1]; ``lines``
    is a list of (points (k, 3), RGBA) polylines, NaN rows break a line.
    ``lit_pos``/``lit_colors`` (colouring by a property) are drawn over the other
    points: added up, or with ``lit_solid`` blended far to near so the nearest is in front.
    Returns uint8 RGB of shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:] = background

    if pos is not None and len(pos):
        px, py, _, vis = project(pos, cam, width, height)
        col = np.asarray(colors, dtype=np.float32).reshape(-1, 4)
        rgb = col[vis, :3] * col[vis, 3:4]              # additive: src * alpha + dst
        _splat(img, px[vis], py[vis], rgb, point_size, additive=True)

    if lit_pos is not None and len(lit_pos):
        px, py, depth, vis = project(lit_pos, cam, width, height)
        col = np.asarray(lit_colors, dtype=np.float32).reshape(-1, 4)[vis]
        if lit_solid:
            order = np.argsort(-depth[vis])             # far first, near written last
            _splat_over(img, px[vis][order], py[vis][order], col[order, :3], col[order, 3:4], point_size)
        else:
            _splat(img, px[vis], py[vis], col[:, :3] * col[:, 3:4], point_size, additive=True)

    for pts, rgba in lines:
        _draw_polyline(img, np.asarray(pts, dtype=np.float64), rgba, cam)

    if highlight_pos is not None and len(highlight_pos):
        px, py, depth, vis = project(highlight_pos, cam, width, height)
        col = np.asarray(highlight_colors, dtype=np.float32).reshape(-1, 4)
        order = np.argsort(-depth[vis])                 # far first, near drawn last
        _splat(img, px[vis][order], py[vis][order], col[vis][order, :3], highlight_size, additive=False)

    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _draw_polyline(img, pts, rgba, cam: Camera):
    h, w = img.shape[:2]
    if len(pts) < 2:
        return
    px, py, _, vis = project(pts, cam, w, h)
    good = vis & np.all(np.isfinite(pts), axis=1)
    rgb = np.asarray(rgba[:3], dtype=np.float32)
    alpha = float(rgba[3]) if len(rgba) > 3 else 1.0
    for i in range(len(pts) - 1):
        if not (good[i] and good[i + 1]):
            continue
        n = int(max(abs(px[i + 1] - px[i]), abs(py[i + 1] - py[i]))) + 1
        if n > 4 * (w + h):                              # segment far off screen
            continue
        t = np.linspace(0.0, 1.0, n + 1)
        x = np.floor(px[i] + t * (px[i + 1] - px[i])).astype(np.int64)
        y = np.floor(py[i] + t * (py[i + 1] - py[i])).astype(np.int64)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        img[y[ok], x[ok]] = img[y[ok], x[ok]] * (1 - alpha) + rgb * alpha


def box_lines(min_corner, size) -> np.ndarray:
    """The 12 edges of an axis-aligned box as one NaN-separated polyline."""
    x0, y0, z0 = (float(v) for v in min_corner)
    sx, sy, sz = (float(v) for v in size)
    c = np.array([[x0 + i * sx, y0 + j * sy, z0 + k * sz] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    out = []
    for a, b in edges:
        out += [c[a], c[b], [np.nan] * 3]
    return np.array(out)
