"""
cache.py — results cache across snapshots and runs.

A result is stored as JSON under a key built from the measurement's name and
arguments and, for every input file, its absolute path, size and modification
time (ns). Touching or replacing a snapshot changes its key, so only that
snapshot's results are recomputed; everything else is read back.

Default folder: ~/.cosmoviewer/results_cache, or $COSMOVIEWER_RESULTS_CACHE
(never inside the simulation folder). A cached result equals the Result a fresh
computation gives (test_profiles_energy_sanity checks this value for value).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .results import Result, _json_default

DEFAULT_DIR = Path(os.environ.get("COSMOVIEWER_RESULTS_CACHE",
                                  Path.home() / ".cosmoviewer" / "results_cache"))


def _file_stamp(path: str) -> list:
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, st.st_mtime_ns]


class ResultCache:
    def __init__(self, folder: str | os.PathLike | None = None):
        self.folder = Path(folder) if folder else DEFAULT_DIR
        self.hits = 0
        self.misses = 0

    def key(self, measurement: str, files, arguments: dict) -> str:
        blob = json.dumps({"m": measurement, "files": [_file_stamp(f) for f in files],
                           "args": arguments}, sort_keys=True, default=_json_default)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get_or_compute(self, measurement: str, files, arguments: dict, compute) -> Result:
        k = self.key(measurement, files, arguments)
        path = self.folder / f"{measurement}_{k}.json"
        if path.exists():
            try:
                with open(path) as fh:
                    res = Result.from_dict(json.load(fh))
                self.hits += 1
                return res
            except (OSError, ValueError, KeyError, TypeError):
                pass                               # unreadable entry: recompute it
        res = compute()
        self.misses += 1
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            res.to_json(str(tmp))
            os.replace(tmp, path)
        except OSError:
            pass                                   # a read-only home still computes
        return res
