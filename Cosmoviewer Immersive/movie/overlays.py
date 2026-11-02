"""
Overlays burnt into movie frames: time or redshift, a scale bar, a colour bar.

The scale bar is measured in the plane through the view centre facing the
camera. Movie frames from particles are perspective views, so the bar is
exact only at the depth of the view centre and its label says so; smoothed
frames are orthographic and the bar holds across the whole frame.

Lengths are labelled with what they are: comoving (as stored in a
cosmological snapshot, usually per h) or physical (times a, divided by h).
A run without comoving integration has one kind of length and the label
only gives the unit.
"""

from dataclasses import dataclass
from math import floor, log10, radians, tan
from typing import Dict, Optional, Tuple

import numpy as np

# Length units the bar can switch between, in cm.
_UNITS = (('pc', 3.085678e18), ('kpc', 3.085678e21), ('Mpc', 3.085678e24), ('Gpc', 3.085678e27))


@dataclass
class LengthInfo:
    """How a stored length converts to a labelled length."""
    unit_cm: float = 3.085678e21      # UnitLength_in_cm
    comoving: bool = False            # stored lengths are comoving (cosmological run)
    a: float = 1.0
    h: float = 1.0
    a_scaling: float = 0.0            # physical = stored * a**a_scaling * h**h_scaling
    h_scaling: float = 0.0

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def length_info(header: Dict, params: Dict, coord_scaling: Dict) -> LengthInfo:
    from snapshot_io.id_lookup import scale_factor
    unit = float(params.get('UnitLength_in_cm', header.get('UnitLength_in_cm', 3.085678e21)))
    h = float(params.get('HubbleParam', header.get('HubbleParam', 1.0)) or 1.0)
    a = scale_factor(header, params)
    comoving = a != 1.0 or int(params.get('ComovingIntegrationOn', 0) or 0) == 1
    sa = coord_scaling.get('a_scaling', coord_scaling.get('aexp-scale-exponent'))
    sh = coord_scaling.get('h_scaling', coord_scaling.get('h-scale-exponent'))
    if sa is None:
        sa = 1.0 if comoving else 0.0                 # Gadget convention when unlabelled
    if sh is None:
        sh = -1.0 if comoving else 0.0
    return LengthInfo(unit, comoving, a, h, float(sa), float(sh))


def _nice(x: float) -> float:
    """Largest of 1, 2, 5 x 10^n not above x."""
    e = floor(log10(x))
    for m in (5, 2, 1):
        if m * 10 ** e <= x:
            return m * 10 ** e
    return 10 ** e


def scale_bar(frame_width_code: float, width_px: int, info: LengthInfo, kind: str,
              perspective: bool, fraction: float = 0.2) -> Tuple[int, str]:
    """
    (bar length in pixels, label) for a frame ``frame_width_code`` stored length units wide.
    ``kind`` is 'comoving' or 'physical'.
    """
    physical = kind == 'physical' or not info.comoving
    factor = info.a ** info.a_scaling * info.h ** info.h_scaling if physical else 1.0
    per_h = (not physical) and info.h_scaling != 0
    cm_per_code = info.unit_cm * factor
    target_cm = frame_width_code * fraction * cm_per_code
    name, ucm = _UNITS[0]
    for n, u in _UNITS:
        if target_cm >= u:
            name, ucm = n, u
    if target_cm < _UNITS[0][1]:
        name, ucm = _UNITS[0]
    value = _nice(target_cm / ucm)
    px = int(round(value * ucm / cm_per_code / frame_width_code * width_px))
    label = f"{value:g} {name}{'/h' if per_h else ''}"
    if info.comoving:
        label += " physical" if physical else " comoving"
    if perspective:
        label += ", at view centre"
    return px, label


def frame_width(cam) -> float:
    """Width of the view in stored length units, in the plane through the view centre."""
    return 2.0 * float(cam.distance) * tan(radians(float(cam.fov)) / 2.0)


def format_value(v: float) -> str:
    return f"{v:.3g}" if 1e-3 <= abs(v) < 1e4 or v == 0 else f"{v:.2e}"


def title_font(font, title: str, width: int):
    """``font``, shrunk until ``title`` fits in ``width`` pixels (it was cut off on the left)."""
    from PyQt6.QtGui import QFont, QFontMetrics
    f = QFont(font)
    while f.pixelSize() > 6 and QFontMetrics(f).horizontalAdvance(title) > width:
        f.setPixelSize(f.pixelSize() - 1)
    return f


def draw_overlays(img, label: str = '', bar: Optional[Tuple[int, str]] = None,
                  colour_bar: Optional[Dict] = None):
    """
    Paint the overlays on a QImage in place. ``colour_bar``: {'table', 'lo', 'hi', 'log', 'title'}.
    """
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
    w, h = img.width(), img.height()
    fs = max(10, h // 32)
    margin = max(6, w // 80)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont()
    font.setPixelSize(fs)
    p.setFont(font)
    text = QColor(230, 230, 230)
    p.setPen(text)
    if label:
        p.drawText(margin, max(6, h // 60) + fs, label)
    if bar is not None and bar[0] > 0:
        px, lab = bar
        y = h - margin - fs // 2
        p.setPen(QPen(text, max(2, h // 240)))
        p.drawLine(margin, y, margin + px, y)
        tick = max(4, fs // 3)
        p.drawLine(margin, y - tick, margin, y + tick)
        p.drawLine(margin + px, y - tick, margin + px, y + tick)
        p.setPen(text)
        p.drawText(margin, y - tick - 4, lab)
    if colour_bar:
        from model.colour_tables import colour_bar_image
        bw, bh = max(120, w // 4), max(8, h // 50)
        x0 = w - margin - bw
        y0 = h - margin - fs - bh - 4
        bar_rgba = np.ascontiguousarray(colour_bar_image(colour_bar['table'], bw, bh))
        qi = QImage(bar_rgba.data, bw, bh, 4 * bw, QImage.Format.Format_RGBA8888).copy()
        p.drawImage(x0, y0, qi)
        p.setPen(QPen(text, 1))
        p.drawRect(x0, y0, bw - 1, bh - 1)
        small = QFont()
        small.setPixelSize(max(9, int(fs * 0.8)))
        p.setFont(small)
        lo, hi = format_value(colour_bar['lo']), format_value(colour_bar['hi'])
        p.drawText(QRect(x0, y0 + bh + 2, bw, fs + 2), int(Qt.AlignmentFlag.AlignLeft), lo)
        p.drawText(QRect(x0, y0 + bh + 2, bw, fs + 2), int(Qt.AlignmentFlag.AlignRight), hi)
        title = colour_bar.get('title', '') + (' (log scale)' if colour_bar.get('log') else '')
        p.setFont(title_font(small, title, x0 + bw - margin))
        p.drawText(QRect(margin, y0 - fs - 4, x0 + bw - margin, fs + 2), int(Qt.AlignmentFlag.AlignRight), title)
    p.end()
