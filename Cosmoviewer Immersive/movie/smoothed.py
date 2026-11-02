"""
Smoothed movie frames: the SPH-kernel surface density of model/sph_projection.py
(the renderer behind the smoothed images in the window) instead of point splats.

* The image is an orthographic projection onto the plane through the view
  centre, as wide as the perspective view is at that depth, so the frame
  frames the same region as a point frame and a scale bar holds everywhere.
* Particles are drawn at their stored positions, as in point frames and
  under the box outline, not wrapped around the view centre.
* Every particle gets h = r_32 / 2 from its 32nd nearest neighbour in the
  drawn sample; masses are scaled by N_total / N_sample, as in the window.
* Colours map the surface density through one colour table with one range
  for the whole movie (fixed by the user, or taken from the first frame and
  then kept), on a log or linear scale, so brightness means the same thing in
  every frame. Empty pixels are background.

These are pictures; the sample and smoothing lengths change with sampling.
"""

from dataclasses import replace
from math import atan, degrees, radians, tan
from typing import Optional, Tuple

import numpy as np

from model.colour_tables import auto_range, lut, normalise
from model.sph_projection import knn_smoothing_length, project

N_NEIGHBOURS = 32
TITLE = "surface density (code mass / code length^2)"


def _layers(scene):
    pos, mass, hs = [], [], []
    for p, m, h in (('pos', 'mass', 'hsml'), ('lit_pos', 'lit_mass', 'lit_hsml')):
        x = getattr(scene, p)
        if x is None or len(x) == 0:
            continue
        mm = getattr(scene, m)
        pos.append(np.asarray(x, dtype=np.float64))
        mass.append(np.ones(len(x)) if mm is None else np.asarray(mm, dtype=np.float64))
        hs.append(getattr(scene, h))
    return pos, mass, hs


def ensure_smoothing_lengths(scene) -> None:
    """Compute (once) and store smoothing lengths on the scene's drawn layers."""
    pos, _, hs = _layers(scene)
    if not pos or all(h is not None for h in hs):
        return
    allpos = np.vstack(pos)
    h = knn_smoothing_length(allpos, N_NEIGHBOURS, float(scene.box_period or 0.0)) if len(allpos) > 1 \
        else np.ones(len(allpos))
    n0 = 0 if scene.pos is None else len(scene.pos)
    if scene.pos is not None and len(scene.pos):
        scene.hsml = h[:n0]
    if scene.lit_pos is not None and len(scene.lit_pos):
        scene.lit_hsml = h[n0:]


def density_image(scene, cam, width: int, height: int) -> np.ndarray:
    """(height, width) surface density, row 0 at the top of the frame."""
    ensure_smoothing_lengths(scene)
    pos, mass, hs = _layers(scene)
    npix = max(width, height)
    if not pos:
        return np.zeros((height, width))
    w_view = 2.0 * cam.distance * tan(radians(cam.fov) / 2.0)        # frame width at the centre
    side = w_view * npix / width
    fov_sq = 2.0 * degrees(atan(side / (2.0 * cam.distance)))
    cdict = {'center': cam.center, 'distance': cam.distance, 'elevation': cam.elevation,
             'azimuth': cam.azimuth, 'fov': fov_sq}
    m = np.concatenate(mass) * float(scene.mass_scale or 1.0)
    # No periodic wrap here (box=0): the box outline, trails and selected particles are drawn at
    # the stored positions, so the image must be too. Wrapping around the view centre moved the
    # picture off the cube outline by a box length whenever the centre was off the box centre.
    # The smoothing lengths above still see neighbours across the periodic boundary.
    img, _ = project(np.vstack(pos), m, np.concatenate(hs), cdict, npix=npix, box=0.0)
    img = img[::-1]                                  # row 0 = top (project's row 0 is the bottom)
    r0 = (npix - height) // 2
    c0 = (npix - width) // 2
    return img[r0:r0 + height, c0:c0 + width]


def choose_range(img: np.ndarray, log: bool) -> Tuple[float, float]:
    """Range from one frame: 1st to 99.9th percentile of the non-empty pixels."""
    return auto_range(img[img > 0], log, 1.0, 99.9)


def colourise(img: np.ndarray, lo: float, hi: float, log: bool, table: str,
              background=(0.0, 0.0, 0.0)) -> np.ndarray:
    """(h, w, 3) float RGB in [0, 1]."""
    x = normalise(img, lo, hi, log)
    t = lut(table)
    out = np.empty(img.shape + (3,), dtype=np.float32)
    out[:] = background
    ok = np.isfinite(x) & (img > 0)
    out[ok] = t[np.round(x[ok] * (len(t) - 1)).astype(np.int64), :3]
    return out
