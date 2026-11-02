"""
SPH-type smoothed images of the particles seen by the camera (GadgetViewer's
"smoothed density" and smoothed-property plots).

Each particle's mass is spread over the image with the 2D cubic-spline kernel
(support radius 2h), normalised per particle so mass is conserved however few
pixels a particle covers. Gas uses its SmoothingLength from the file when
available; other particles get h = r_k / 2 from their k-th nearest neighbour
in the display sample, so the support reaches the k-th neighbour.

Large kernels are deposited on coarser copies of the image (each level halves
the resolution, so no kernel is wider than MAX_R pixels) and the levels are
upsampled and summed. This keeps the cost to about (2 MAX_R + 1)^2 operations
per particle.

The image is built from the display sample with masses scaled by
N_total / N_sample, so it estimates the surface density, but it is a picture:
the sample and the smoothing lengths change with sampling. Nothing quantitative
should be read from it; that is stated in the panel.
"""

import os
from typing import Optional, Tuple

import numpy as np

MAX_R = 8
DEPOSIT_THREADS = max(1, min(8, (os.cpu_count() or 2) - 1))   # leave one core to the window


def cubic_spline_2d(q: np.ndarray) -> np.ndarray:
    """
    Unnormalised 2D cubic spline in q = r/h, support q < 2. The analytic factor
    10 / (7 pi h^2) is left out because _deposit_part normalises each particle's
    kernel over the pixels it covers.
    """
    w = np.zeros_like(q)
    a = q < 1
    b = (q >= 1) & (q < 2)
    w[a] = 1 - 1.5 * q[a] ** 2 + 0.75 * q[a] ** 3
    w[b] = 0.25 * (2 - q[b]) ** 3
    return w


def knn_smoothing_length(pos: np.ndarray, k: int = 32, box: float = 0.0) -> np.ndarray:
    from scipy.spatial import cKDTree
    p = np.asarray(pos, dtype=np.float64)
    tree = cKDTree(np.mod(p, box), boxsize=box) if box > 0 else cKDTree(p)
    k = min(k, max(len(p) - 1, 1))
    d, _ = tree.query(np.mod(p, box) if box > 0 else p, k=k + 1, workers=-1)   # +1: the particle itself
    return 0.5 * d[:, -1]


def camera_frame(cam: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, towards-camera) unit vectors for pyqtgraph's (elevation, azimuth)."""
    el, az = np.radians(cam['elevation']), np.radians(cam['azimuth'])
    back = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(back, right)
    return right, up, back


def _spline(q: np.ndarray) -> np.ndarray:
    """cubic_spline_2d without masked indexing (the hot loop of _deposit)."""
    t = np.clip(2.0 - q, 0.0, None)
    return np.where(q < 1.0, 1.0 - 1.5 * q * q + 0.75 * q * q * q, 0.25 * t * t * t)


def _deposit_part(flats, nx: int, ny: int, R: int, x, y, h, ws):
    """Kernels of radius R (pixels) for one share of the particles, added to ``flats``."""
    off = np.arange(-R, R + 1)
    ox, oy = np.meshgrid(off, off, indexing='ij')
    ox, oy = ox.ravel(), oy.ravel()
    centre = len(ox) // 2
    step = max(1, 400_000 // len(ox))                   # about 400k kernel pixels at a time
    for s in range(0, len(x), step):
        xs, ys = x[s:s + step], y[s:s + step]
        fx, fy = np.floor(xs), np.floor(ys)
        ix = fx.astype(np.int64)[:, None] + ox[None, :]
        iy = fy.astype(np.int64)[:, None] + oy[None, :]
        dx = (fx + 0.5 - xs)[:, None] + ox[None, :]
        dy = (fy + 0.5 - ys)[:, None] + oy[None, :]
        hh = np.maximum(h[s:s + step], 0.5)[:, None]    # at least half a pixel
        ker = _spline(np.sqrt(dx * dx + dy * dy) / hh)
        tot = ker.sum(axis=1)
        small = tot == 0
        ker *= (1.0 / np.where(small, 1.0, tot))[:, None]
        ker[:, centre] += small                         # kernel smaller than a pixel: centre pixel
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        idx = (iy * nx + ix)[ok]
        ker_ok = ker[ok]
        rows = np.broadcast_to(np.arange(len(xs))[:, None], ker.shape)[ok]
        for flat, w in zip(flats, ws):
            flat += np.bincount(idx, weights=ker_ok * w[s:s + step][rows], minlength=nx * ny)


def _deposit(imgs, x: np.ndarray, y: np.ndarray, h: np.ndarray, ws):
    """
    Add kernels (pixel units) to each image in ``imgs`` (ny, nx), weighted by the
    matching array in ``ws`` (the kernel is worked out once for all); every r <= MAX_R.
    A single image and weight array may be passed as they are. Large sets are
    shared between threads (numpy releases the interpreter lock in this work),
    each adding to its own copy of the images.
    """
    if isinstance(imgs, np.ndarray):
        imgs, ws = [imgs], [ws]
    ny, nx = imgs[0].shape
    r = np.clip(np.ceil(2 * h).astype(np.int64), 0, MAX_R)
    flats = [img.ravel() for img in imgs]
    for R in np.unique(r):
        sel = np.nonzero(r == R)[0]
        work = len(sel) * (2 * R + 1) ** 2
        nthreads = int(min(DEPOSIT_THREADS, max(1, work // 2_000_000)))
        if nthreads == 1:
            _deposit_part(flats, nx, ny, int(R), x[sel], y[sel], h[sel], [w[sel] for w in ws])
            continue
        from concurrent.futures import ThreadPoolExecutor
        parts = np.array_split(sel, nthreads)
        local = [[np.zeros(nx * ny) for _ in flats] for _ in parts]
        with ThreadPoolExecutor(nthreads) as pool:
            list(pool.map(lambda a: _deposit_part(a[0], nx, ny, int(R), x[a[1]], y[a[1]], h[a[1]],
                                                  [w[a[1]] for w in ws]), zip(local, parts)))
        for lf in local:
            for flat, add in zip(flats, lf):
                flat += add


def project(pos: np.ndarray, mass: np.ndarray, h: np.ndarray, cam: dict, npix: int = 400,
            weights: Optional[np.ndarray] = None, box: float = 0.0,
            region: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, float]:
    """
    Smoothed image of the region the camera looks at.

    Returns (image, width): surface density (mass / length^2 in code units) or,
    with ``weights``, the mass-weighted mean of ``weights`` (NaN where empty);
    ``width`` is the side of the square image in code length units. Rows are
    the screen's up axis, columns the right axis.

    ``region`` = (x0, x1, y0, y1), offsets from the camera centre along the
    screen's right and up axes, draws only that rectangle (a zoom into the
    picture) at ``npix`` pixels across and square pixels; ``width`` is then x1 - x0.
    """
    right, up, _ = camera_frame(cam)
    c = np.asarray(cam['center'], dtype=np.float64)
    d = np.asarray(pos, dtype=np.float64) - c
    if box > 0:
        d -= box * np.round(d / box)
    if region is None:
        width = 2.0 * cam['distance'] * np.tan(np.radians(cam['fov']) / 2.0)
        x0 = y0 = -width / 2.0
        nx = ny = npix
    else:
        x0, x1, y0, y1 = (float(r) for r in region)
        width = x1 - x0
        nx = npix
        ny = max(1, int(round(npix * (y1 - y0) / width)))
    pix = width / nx
    if region is None:
        x = (d @ right) / pix + npix / 2.0
        y = (d @ up) / pix + npix / 2.0
    else:
        x = (d @ right - x0) / pix
        y = (d @ up - y0) / pix
    hp = np.asarray(h, dtype=np.float64) / pix
    m = np.asarray(mass, dtype=np.float64)
    keep = (x + 2 * hp >= 0) & (x - 2 * hp < nx) & (y + 2 * hp >= 0) & (y - 2 * hp < ny) & np.isfinite(hp)
    if weights is not None:
        keep &= np.isfinite(weights)
    x, y, hp, m = x[keep], y[keep], hp[keep], m[keep]
    q = None if weights is None else np.asarray(weights, dtype=np.float64)[keep]

    def render(wlist):
        totals = [np.zeros((ny, nx)) for _ in wlist]
        level = 0
        remaining = np.ones(len(x), dtype=bool)
        while remaining.any():
            f = 2 ** level
            nly, nlx = int(np.ceil(ny / f)), int(np.ceil(nx / f))
            this = remaining & ((2 * hp / f <= MAX_R) | (max(nlx, nly) <= 4))
            if this.any():
                imgs = [np.zeros((nly, nlx)) for _ in wlist]
                _deposit(imgs, x[this] / f, y[this] / f, hp[this] / f, [w[this] for w in wlist])
                for total, img in zip(totals, imgs):
                    if f > 1:
                        img = np.repeat(np.repeat(img, f, axis=0), f, axis=1)[:ny, :nx] / (f * f)
                        from scipy.ndimage import uniform_filter
                        # Soften the block edges; a zoom's borders are inside the picture, so its
                        # edges continue outward rather than fading to nothing.
                        img = uniform_filter(img, size=f, mode='constant' if region is None else 'nearest')
                    total += img
            remaining &= ~this
            level += 1
        return totals

    if q is None:
        mass_img, = render([m])
    else:
        mass_img, num = render([m, m * q])          # one kernel pass for both
    # Upsampling and uniform_filter leave round-off (tiny, even negative) values far from
    # any particle; zero them so they do not show up on a log scale.
    tiny = mass_img <= 1e-10 * max(float(mass_img.max()), 1e-300)
    mass_img[tiny] = 0.0
    if q is None:
        return mass_img / (pix * pix), width
    with np.errstate(invalid='ignore', divide='ignore'):
        empty = tiny
        out = np.where(empty, np.nan, num / np.where(empty, 1.0, mass_img))
    return out, width
