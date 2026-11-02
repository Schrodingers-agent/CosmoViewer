"""
Keyframed camera paths for movies.

A keyframe is a view the user stored: camera centre, distance, angles, field
of view, and the snapshot shown. The movie's camera passes through every
keyframe in order on a Catmull-Rom spline (centre, log distance, angles,
field of view), the usual camera-path curve in ParaView and Blender. The
snapshot changes linearly between keyframes, so a path can combine a
fly-through with time evolution; it never runs backwards between two
keyframes unless the keyframes do.

Frames are spread evenly over the keyframe segments; with ``ease`` the camera
starts and stops gently (smoothstep on the whole path).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from movie.renderer import Camera


@dataclass
class Keyframe:
    camera: Dict                      # center, distance, elevation, azimuth, fov
    snapshot: int                     # index into the loaded snapshots
    name: str = ''

    def to_dict(self) -> Dict:
        return {'camera': {k: (list(map(float, v)) if k == 'center' else float(v))
                           for k, v in self.camera.items()},
                'snapshot': int(self.snapshot), 'name': self.name}

    @staticmethod
    def from_dict(d: Dict) -> 'Keyframe':
        cam = dict(d['camera'])
        cam['center'] = tuple(cam['center'])
        return Keyframe(cam, int(d['snapshot']), d.get('name', ''))


def _catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _params(kf: Keyframe) -> np.ndarray:
    c = kf.camera
    return np.array([*map(float, c['center']), np.log(max(float(c['distance']), 1e-300)),
                     float(c['elevation']), float(c['azimuth']), float(c['fov'])])


def camera_path(keyframes: Sequence[Keyframe], nframes: int, ease: bool = True) -> List[Tuple[Camera, float]]:
    """(camera, snapshot position) for every frame; the position is fractional between keyframes."""
    if len(keyframes) < 2:
        raise ValueError("A camera path needs at least two keyframes")
    if nframes < 2:
        raise ValueError("A camera path needs at least two frames")
    P = np.array([_params(k) for k in keyframes])
    # Azimuth: take the short way round between consecutive keyframes.
    az = P[:, 5].copy()
    for i in range(1, len(az)):
        az[i] = az[i - 1] + ((az[i] - az[i - 1] + 180.0) % 360.0 - 180.0)
    P[:, 5] = az
    snaps = np.array([k.snapshot for k in keyframes], dtype=np.float64)
    # Ends are padded by reflection so the curve leaves the first and last keyframe straight.
    pad = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    nseg = len(keyframes) - 1
    out = []
    for k in range(nframes):
        u = k / (nframes - 1)
        if ease:
            u = u * u * (3 - 2 * u)
        t = u * nseg
        i = min(int(np.floor(t)), nseg - 1)
        local = t - i
        v = _catmull_rom(pad[i], pad[i + 1], pad[i + 2], pad[i + 3], local)
        if local == 0.0:
            v = P[i]                       # exactly on a keyframe
        elif i == nseg - 1 and k == nframes - 1:
            v = P[-1]
        s = snaps[i] + local * (snaps[i + 1] - snaps[i])
        cam = Camera(center=tuple(v[:3]), distance=float(np.exp(v[3])),
                     elevation=float(np.clip(v[4], -90.0, 90.0)), azimuth=float(v[5]), fov=float(np.clip(v[6], 1.0, 179.0)))
        out.append((cam, float(s)))
    return out
