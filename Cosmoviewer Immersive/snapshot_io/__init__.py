"""
Snapshot I/O package for snapshot loading, halo catalogs, and FoF group finding.

The package-level names are imported on first use, so that importing one
module of the package (as the FoF worker process does with
snapshot_io.fof_process) does not load Qt and the snapshot loader.
"""

__all__ = ['AsyncSnapshotLoader', 'HaloCatalogLoader', 'FoFGroupsEngine']

_SOURCES = {'AsyncSnapshotLoader': 'snapshot_loader', 'HaloCatalogLoader': 'halo_catalog',
            'FoFGroupsEngine': 'fof_finder'}


def __getattr__(name):
    if name in _SOURCES:
        import importlib
        return getattr(importlib.import_module(f'.{_SOURCES[name]}', __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
