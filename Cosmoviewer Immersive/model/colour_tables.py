"""
Colour tables and value-to-colour mapping for colouring particles by a property.

GadgetViewer's "colour by property" maps a particle property through a colour
table, linearly or logarithmically, between a minimum and maximum the user can
set, and has a "random colours for integer properties" option (with a
Randomize button) for halo membership. This module does the same in numpy, so
the window and the off-screen movie renderer produce identical colours.

The tables are perceptually uniform maps bundled with pyqtgraph (matplotlib's
viridis family and Kovesi's CET maps), sampled once into 256-entry arrays.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# name shown -> pyqtgraph colormap name
TABLES: Dict[str, str] = {
    'viridis': 'viridis',
    'magma': 'magma',
    'inferno': 'inferno',
    'plasma': 'plasma',
    'cividis (colour-blind safe)': 'cividis',
    'fire (CET-L3)': 'CET-L3',
    'grey (CET-L1)': 'CET-L1',
    'blue-white-red (CET-D1, diverging)': 'CET-D1',
    'turbo (rainbow)': 'turbo',
}
DEFAULT_TABLE = 'viridis'
DIVERGING_TABLE = 'blue-white-red (CET-D1, diverging)'

MISSING_RGBA = np.array([0.35, 0.35, 0.35, 0.08])     # particles without the property
NO_GROUP_RGBA = np.array([0.25, 0.25, 0.25, 0.15])    # integer value 0 = in no group

_LUTS: Dict[str, np.ndarray] = {}


def lut(table: str, n: int = 256) -> np.ndarray:
    """(n, 4) float RGBA in [0, 1]."""
    key = f"{table}:{n}"
    if key not in _LUTS:
        import pyqtgraph as pg
        cmap = pg.colormap.get(TABLES.get(table, table))
        _LUTS[key] = cmap.getLookupTable(nPts=n, alpha=True).astype(np.float64) / 255.0
    return _LUTS[key]


def auto_range(values: np.ndarray, log: bool, lo_pct: float = 0.5, hi_pct: float = 99.5) -> Tuple[float, float]:
    """Percentile range of the finite (and, for log, positive) values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if len(v) == 0:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo * 10 if (log and lo > 0) else lo + 1.0
    return float(lo), float(hi)


def normalise(values: np.ndarray, lo: float, hi: float, log: bool) -> np.ndarray:
    """Values scaled to [0, 1] (clipped); NaN where undefined (NaN, or <= 0 on a log scale)."""
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        if log:
            if lo <= 0 or hi <= 0:
                raise ValueError("log scale needs a positive minimum and maximum")
            x = (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            x[~(v > 0)] = np.nan
        else:
            x = (v - lo) / (hi - lo) if hi != lo else np.zeros_like(v)
    return np.clip(x, 0.0, 1.0)


def map_values(values: np.ndarray, lo: float, hi: float, log: bool, table: str,
               alpha: float = 0.7) -> np.ndarray:
    """(n, 4) RGBA; undefined values get MISSING_RGBA."""
    x = normalise(values, lo, hi, log)
    table_rgba = lut(table)
    out = np.empty((len(x), 4), dtype=np.float32)        # float32: what the GPU takes, no later copy
    out[:] = MISSING_RGBA
    ok = np.isfinite(x)
    out[ok] = table_rgba[np.round(x[ok] * (len(table_rgba) - 1)).astype(np.int64)]
    out[ok, 3] = alpha
    return out


def random_colours(values: np.ndarray, seed: int = 0, alpha: float = 0.85) -> np.ndarray:
    """
    Random but stable colour per integer value (group index). Value 0 (and NaN)
    means "in no group" and is drawn dim. The same value always gets the same
    colour for a given seed, in every snapshot and in movies.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) & (v != 0)
    out = np.empty((len(v), 4), dtype=np.float32)
    out[:] = NO_GROUP_RGBA
    if ok.any():
        k = v[ok].astype(np.int64)
        # Hash value and seed into a hue; saturated, bright colours stay visible on black.
        h = (k * np.int64(2654435761) + np.int64(seed) * np.int64(40503)) & np.int64(0xFFFFFFFF)
        hue = (h % 3600) / 3600.0
        sat = 0.55 + 0.45 * ((h >> 12) % 100) / 100.0
        val = 0.75 + 0.25 * ((h >> 20) % 100) / 100.0
        out[ok, :3] = _hsv_to_rgb(hue, sat, val)
        out[ok, 3] = alpha
    return out


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6).astype(np.int64) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = np.empty((len(h), 3))
    for k, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        rgb[m, 0], rgb[m, 1], rgb[m, 2] = r[m], g[m], b[m]
    return rgb


def colour_bar_image(table: str, width: int = 256, height: int = 12) -> np.ndarray:
    """(height, width, 4) uint8 RGBA image of the table, low values on the left."""
    row = (lut(table, width) * 255).astype(np.uint8)
    return np.repeat(row[None, :, :], height, axis=0)
