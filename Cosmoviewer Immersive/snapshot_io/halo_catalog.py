"""
Legacy wrapper around cosmo_io_engine's HaloCatalogReader (Gadget-4 and VELOCIraptor groups).

The application reads Gadget-4 catalogues with snapshot_io.group_catalogue instead; this module
is kept for code that imports HaloCatalogLoader. Its VELOCIraptor reader is untested (no test
files), and the window does not offer that format.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from pygadgetviewer_path import ensure_on_path

ensure_on_path()
try:
    from cosmo_io_engine import HaloCatalogReader
    HAS_HALO_READER = True
except ImportError:
    try:
        from pygadgetviewer import HaloCatalogReader
        HAS_HALO_READER = True
    except ImportError:
        HAS_HALO_READER = False


class HaloCatalogLoader:
    """Interface to parse and inject halo group membership into particle stores."""

    @staticmethod
    def load_gadget4_groups(catalog_file: str) -> Dict[str, Any]:
        if HAS_HALO_READER:
            return HaloCatalogReader.read_gadget4_groups(catalog_file)
        return {}

    @staticmethod
    def load_velociraptor_groups(catalog_file: str) -> Dict[str, Any]:
        if HAS_HALO_READER:
            return HaloCatalogReader.read_velociraptor_groups(catalog_file)
        return {}
