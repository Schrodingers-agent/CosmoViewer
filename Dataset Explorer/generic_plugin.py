"""
generic_plugin.py — fallback view for any numeric dataset without a plugin.

The hand-written plugins cover EAGLE gas, stars and black holes. Everything
else (dark matter, Gadget-4 names such as Masses, FoF / SUBFIND catalogue
fields) gets a generated plugin class from make_generic_leaves().

Units come only from the file:
  * Gadget-4 writes length_scaling / mass_scaling / velocity_scaling, which
    give the dimension, and to_cgs, which gives the factor.
  * EAGLE writes CGSConversionFactor without a dimension; the dimension is
    then taken from a short table of standard dataset names.
  * Otherwise the values are shown as stored and labelled "code units".
Nothing is guessed: an unknown dimension is said to be unknown.

Vector datasets (N×K) become one leaf per column, plus |·| for 3-vectors
that are not positions. Signed data is binned on a linear axis, and so are
positions (Coordinates, GroupPos, SubhaloPos, SubhaloCM, CentreOfMass, ...):
a log axis of a coordinate means nothing. Lengths are offered in kpc first,
the unit the Analyses report, with cm and Mpc in the unit dropdown.
"""
from __future__ import annotations

import re

import numpy as np

from plugin_base import BasePlugin

try:
    from astropy import units as u
except ImportError:          # the base pipeline has its own safefall
    u = None

# Groups that hold metadata or lookup tables, not per-object quantities.
SKIP_GROUPS = {"Header", "Config", "Parameters", "Units", "Constants",
               "RuntimePars", "HashTable"}

# Dimension of common dataset names when the file gives only a CGS factor.
_NAME_UNITS = {
    "Coordinates": "cm", "Velocity": "cm/s", "Velocities": "cm/s",
    "Mass": "g", "Masses": "g", "SmoothingLength": "cm", "Density": "g/cm3",
    "InternalEnergy": "erg/g", "Potential": "erg/g", "BH_Mass": "g",
    # CGSConversionFactor turns EAGLE's Msun/yr into g/s (6.30e25), so the CGS
    # value is in g/s; Msun/yr is offered first in the dropdown.
    "InitialMass": "g", "StarFormationRate": "g/s",
}

# Display choices per physical type (first entry is the default).
_DISPLAY_CHOICES = {
    "length":            ["cm", "kpc", "Mpc"],
    "mass":              ["g", "Msun"],
    "speed":             ["cm/s", "km/s"],
    "mass density":      ["g/cm3", "Msun/kpc3"],
    "time":              ["s", "Myr", "Gyr"],
    "specific energy":   ["erg/g", "km2/s2"],
    "energy":            ["erg"],
    "angular momentum":  ["g cm2/s", "Msun kpc km/s"],
    "mass rate":         ["g/s", "Msun/yr"],
}

# Display unit offered first, the default in the CGS system (Galactic and Cosmological
# still pick kpc and Mpc): lengths in kpc, as the Analyses report them.
_FIRST_CHOICE = {"length": "kpc", "mass rate": "Msun/yr"}

# EAGLE particle GroupNumber / SubGroupNumber: 2**30 marks a particle in no
# FoF group / no subgroup (EAGLE team 2017, arXiv:1706.09899). The value is
# present in RefL0025N0752 snapshot 28 for gas, stars and black holes.
_GROUP_FLAGS = {float(2 ** 30): "in no group"}

_POSITION_RE = re.compile(r"(Coordinates|Pos$|CM$|Centre|Center)")
_COMPONENT_NAMES = ("x", "y", "z")


def _base_unit(attrs: dict, dataset: str):
    """(astropy unit or None, note). The unit is what Stage 2 values are in."""
    if u is None:
        return None, ""
    has_factor = "to_cgs" in attrs or "CGSConversionFactor" in attrs
    if not has_factor:
        return None, "code units (no unit attributes in file)"
    exps = [attrs.get(k) for k in ("length_scaling", "mass_scaling", "velocity_scaling")]
    if all(e is not None for e in exps):
        l, m, v = (float(e) for e in exps)
        unit = (u.cm ** l) * (u.g ** m) * ((u.cm / u.s) ** v)
        return unit.decompose(bases=[u.cm, u.g, u.s]), ""
    leaf = dataset.split("/")[-1]
    if leaf in _NAME_UNITS:
        return u.Unit(_NAME_UNITS[leaf]), f"unit inferred from the name {leaf}"
    factor = attrs.get("CGSConversionFactor", attrs.get("to_cgs"))
    try:
        as_stored = float(factor) == 1.0
    except (TypeError, ValueError):
        as_stored = False
    if as_stored:
        # A factor of 1 with no named unit: counts, flags, fractions and expansion factors
        # (EAGLE's BH_CumlNumSeeds, OnEquationOfState, ...); the value is the stored one.
        return None, "as stored (CGS factor 1; the file does not name a unit)"
    return None, "CGS value; the file does not record which unit"


def _choices(base) -> list:
    if base is None:
        return []
    if base == u.dimensionless_unscaled:
        return [u.dimensionless_unscaled]
    try:
        ptype = str(base.physical_type)
    except Exception:
        ptype = ""
    if base.is_equivalent(u.g / u.s):     # astropy has no name for it ("unknown")
        ptype = "mass rate"
    out = [base]
    for name in _DISPLAY_CHOICES.get(ptype, []):
        cand = u.Unit(name)
        if cand != base and cand.is_equivalent(base):
            out.append(cand)
    first = _FIRST_CHOICE.get(ptype)
    if first is not None:
        out.sort(key=lambda unit: unit != u.Unit(first))     # stable: that unit first, others in order
    return out


class GenericPlugin(BasePlugin):
    """Base for generated per-leaf classes; do not register directly."""
    GROUP = ""
    DATASET = ""
    TABS = ["Histogram", "Line"]
    TREE_CONTRIBUTIONS: dict = {}
    DEFAULT_SCALE_X = "log"
    DEFAULT_SCALE_Y = "log"
    MAX_BINS = 100
    HISTOGRAM_AUTO_LINEAR = True
    COLUMN: int | None = None      # column of an N×K dataset; None = whole / magnitude
    MAGNITUDE: bool = False
    UNIT_NOTE: str = ""
    GENERIC: bool = True

    def reduce_vector(self, array):
        if self.MAGNITUDE:
            return np.linalg.norm(array, axis=1)
        if self.COLUMN is not None:
            return array[:, self.COLUMN]
        return array.reshape(len(array), -1)[:, 0]

    def lod_key_name(self, dataset_path):
        # [x], [y], |·| of one dataset need seperate index files.
        if self.MAGNITUDE:
            return f"{dataset_path}#norm"
        if self.COLUMN is not None:
            return f"{dataset_path}#{self.COLUMN}"
        return dataset_path

    def get_custom_info(self, cgs_data):
        return f"generic view; {self.UNIT_NOTE}" if self.UNIT_NOTE else "generic view"


def make_generic_leaves(group: str, dataset: str, shape: tuple, dtype) -> dict:
    """{leaf label: plugin class} for one dataset, or {} if it is not numeric."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "iuf" or len(shape) == 0 or shape[0] == 0 or len(shape) > 2:
        return {}
    is_int = dtype.kind in "iu"
    is_id = dataset.split("/")[-1].endswith(("IDs", "ID")) or dataset in ("ID",)
    # Positions: linear histogram bins and axes (the Line view's value axis is y).
    is_position = bool(_POSITION_RE.search(dataset))
    tabs = ["Line"] if is_id else ["Histogram", "Line"]

    def make(label, column=None, magnitude=False, display=None):
        name = re.sub(r"\W", "_", f"Generic_{group}_{label}")
        attrs = {
            "GROUP": group, "DATASET": dataset, "TABS": tabs,
            "TREE_CONTRIBUTIONS": {group: [label]},
            "DISPLAY_NAME": display or label,
            "COLUMN": column, "MAGNITUDE": magnitude,
            "DEFAULT_SCALE_X": "linear" if is_id or is_position else "log",
            "DEFAULT_SCALE_Y": "linear" if is_id or is_position else "log",
            "HISTOGRAM_LINEAR_X": is_position and not magnitude,
            "HISTOGRAM_FLAG_VALUES": (_GROUP_FLAGS if dataset in ("GroupNumber", "SubGroupNumber")
                                      else {}),
            "LINE_Y_LABEL": (f"{display or label} ({dtype} → float64)" if is_int else ""),
            "REQUIRES": [f"{group}/{dataset}"],
        }
        return type(name, (GenericPlugin,), attrs)

    if len(shape) == 1 or shape[1] == 1:
        return {dataset: make(dataset)}
    k = shape[1]
    leaves = {}
    if k == 3:
        if not _POSITION_RE.search(dataset):
            leaves[f"{dataset} |·|"] = make(f"{dataset} |·|", magnitude=True,
                                             display=f"|{dataset}|")
        for i, c in enumerate(_COMPONENT_NAMES):
            leaves[f"{dataset} [{c}]"] = make(f"{dataset} [{c}]", column=i)
    else:
        for i in range(k):
            leaves[f"{dataset} [{i}]"] = make(f"{dataset} [{i}]", column=i)
    return leaves


def configure_units(plugin_cls, attrs: dict) -> None:
    """Set BASE_CGS_UNIT / ALLOWED_UNITS on a generated class from dataset attrs."""
    base, note = _base_unit(attrs, plugin_cls.DATASET)
    plugin_cls.UNIT_NOTE = note
    if base is None:
        plugin_cls.BASE_CGS_UNIT = None
        plugin_cls.ALLOWED_UNITS = []          # [] = no dropdown; values shown as computed
        if "code units" in note:
            plugin_cls.DISPLAY_NAME = f"{plugin_cls.DISPLAY_NAME} (code units)"
        elif note.startswith("as stored"):
            plugin_cls.DISPLAY_NAME = f"{plugin_cls.DISPLAY_NAME} (as stored)"
        elif note:
            plugin_cls.DISPLAY_NAME = f"{plugin_cls.DISPLAY_NAME} (CGS, unit unknown)"
    else:
        plugin_cls.BASE_CGS_UNIT = base
        plugin_cls.ALLOWED_UNITS = _choices(base)
