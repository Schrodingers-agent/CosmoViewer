"""
View frustum of the 3D window, used to decide which particles can be on screen.

The camera is pyqtgraph's GLViewWidget camera (centre, distance, elevation,
azimuth, horizontal field of view), whose view matrix movie/renderer.py
already reproduces. The projection is GLViewWidget.projectionMatrix(): near
and far clip at 0.001 and 1000 times the camera distance.
"""

from math import radians, tan

import numpy as np

from movie.renderer import Camera, view_matrix


def projection_matrix(fov_deg: float, width: int, height: int, near: float, far: float) -> np.ndarray:
    """glFrustum as GLViewWidget builds it: fov is horizontal, top/bottom follow the aspect ratio."""
    r = near * tan(0.5 * radians(fov_deg))
    t = r * height / max(width, 1)
    m = np.zeros((4, 4))
    m[0, 0] = near / r
    m[1, 1] = near / t
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -2.0 * far * near / (far - near)
    m[3, 2] = -1.0
    return m


def camera_mvp(cam: Camera, width: int, height: int) -> np.ndarray:
    near, far = cam.distance * 0.001, cam.distance * 1000.0
    return projection_matrix(cam.fov, width, height, near, far) @ view_matrix(cam)


def planes_from_mvp(mvp: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    The six clip planes (a, b, c, d) of a model-view-projection matrix, normals
    pointing inside, normalised so a*x + b*y + c*z + d is a distance.
    ``margin`` widens the side planes by that fraction of the view (so particles
    just off screen are already loaded when the view pans a little).
    """
    m = np.asarray(mvp, dtype=np.float64)
    r0, r1, r2, r3 = m[0], m[1], m[2], m[3]
    s = 1.0 + margin
    planes = np.array([r3 * s + r0, r3 * s - r0, r3 * s + r1, r3 * s - r1, r3 + r2, r3 - r2])
    planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
    return planes


def camera_planes(cam: Camera, width: int, height: int, margin: float = 0.0) -> np.ndarray:
    return planes_from_mvp(camera_mvp(cam, width, height), margin)


OUTSIDE, PARTIAL, INSIDE = 0, 1, 2


def classify_boxes(planes: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """OUTSIDE / PARTIAL / INSIDE for axis-aligned boxes (n, 3) against the frustum (conservative)."""
    lo = np.asarray(lo, dtype=np.float64).reshape(-1, 3)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1, 3)
    state = np.full(len(lo), INSIDE, dtype=np.int8)
    for a, b, c, d in planes:
        n = np.array([a, b, c])
        pos = n >= 0
        p_far = np.where(pos, hi, lo) @ n + d       # corner furthest along the normal
        p_near = np.where(pos, lo, hi) @ n + d      # corner furthest against it
        state[p_far < 0] = OUTSIDE
        state[(p_near < 0) & (state == INSIDE)] = PARTIAL
    return state


def points_inside(planes: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ok = np.ones(len(p), dtype=bool)
    for a, b, c, d in planes:
        ok &= (p[:, 0] * a + p[:, 1] * b + p[:, 2] * c + d) >= 0
    return ok
