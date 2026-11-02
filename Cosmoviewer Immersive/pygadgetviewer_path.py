"""
Locates cosmo_io_engine.py and puts its folder on sys.path.

cosmo_io_engine.py provides binary-format Gadget snapshot reading, octree
spatial indexing, and halo catalogue readers. For HDF5 snapshots (the primary
use case), the main application uses h5py directly and does not need this module.

Previously named pygadgetviewer.py; renamed for clarity.
"""

import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent

SEARCH_DIRS = [
    _ROOT,
    _ROOT.parent,
    _ROOT.parent / "cosmo_io_engine",
    # Legacy locations for backwards compatibility
    _ROOT.parent / "gadgetviewer-1.2.0 2",
    _ROOT.parent / "gadgetviewer-1.2.0",
]

# Module name to import (try new name first, fall back to old)
_MODULE_NAMES = ["cosmo_io_engine", "pygadgetviewer"]


def ensure_on_path() -> Optional[Path]:
    """Add the folder containing the I/O engine module to sys.path; return it, or None if not found."""
    for folder in SEARCH_DIRS:
        for name in _MODULE_NAMES:
            if (folder / f"{name}.py").is_file():
                if str(folder) not in sys.path:
                    sys.path.append(str(folder))
                return folder
    return None


def missing_message() -> str:
    searched = "\n".join(f"  {d}" for d in SEARCH_DIRS)
    return f"cosmo_io_engine.py was not found. Searched:\n{searched}"
