"""
results.py — the Result contract every measurement returns.

A Result is named quantities (astropy Quantities) plus Provenance. Provenance
says which files, particles, centre and parameters produced the numbers, and
how many of the particles were used. Only a result computed from every
particle it claims to describe (fraction_used == 1) counts as a science
result; anything else is a preview and says so.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from astropy import units as u


@dataclass
class Provenance:
    measurement: str
    version: str
    files: list = field(default_factory=list)
    particle_types: list = field(default_factory=list)
    n_used: int = 0
    n_total: int = 0
    frame: str = "physical"            # "physical" | "comoving"
    time: dict = field(default_factory=dict)   # {"code": .., "a": .., "z": .., "Gyr": ..}
    centre: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    @property
    def fraction_used(self) -> float:
        return self.n_used / self.n_total if self.n_total else 0.0

    @property
    def is_science(self) -> bool:
        return self.n_total > 0 and self.n_used == self.n_total

    def status_line(self) -> str:
        """One line for the status bar: all particles, or PREVIEW with the fraction used."""
        if self.is_science:
            who = f"All {self.n_total:,} particles"
        else:
            who = f"PREVIEW: {self.n_used:,} of {self.n_total:,} particles"
        parts = [who]
        if self.centre:
            parts.append(f"centre: {self.centre.get('method', '?')}")
        parts.append(self.frame)
        return " · ".join(parts)


@dataclass
class Result:
    values: dict                 # name -> astropy Quantity (or plain ndarray for dimensionless)
    provenance: Provenance

    def __getitem__(self, key):
        return self.values[key]

    @property
    def is_science(self) -> bool:
        return self.provenance.is_science

    def to_dict(self) -> dict:
        """JSON-safe dict: each value as {"value": list, "unit": str}."""
        out = {}
        for k, v in self.values.items():
            if isinstance(v, u.Quantity):
                out[k] = {"value": np.asarray(v.value).tolist(), "unit": v.unit.to_string()}
            else:
                out[k] = {"value": np.asarray(v).tolist(), "unit": ""}
        prov = asdict(self.provenance)
        prov["fraction_used"] = self.provenance.fraction_used
        prov["is_science"] = self.provenance.is_science
        return {"values": out, "provenance": prov}

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        """Inverse of to_dict (used by the results cache)."""
        values = {}
        for k, v in d["values"].items():
            arr = np.asarray(v["value"])
            if arr.dtype == object:
                arr = np.asarray(v["value"], dtype=float)
            values[k] = arr * u.Unit(v["unit"]) if v["unit"] else (arr if arr.ndim else arr[()])
        prov = dict(d["provenance"])
        prov.pop("fraction_used", None)
        prov.pop("is_science", None)
        return cls(values=values, provenance=Provenance(**prov))

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, u.Quantity):
        return {"value": np.asarray(o.value).tolist(), "unit": o.unit.to_string()}
    return str(o)
