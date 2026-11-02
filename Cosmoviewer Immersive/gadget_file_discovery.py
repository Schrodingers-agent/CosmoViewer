"""
What is in a Gadget output folder, and what is in one snapshot, from metadata alone.

FileDiscoveryEngine sorts the files of an output folder by name into snapshots
(snapshot_* and snap_*, single or multi-file), backup snapshots (bak-*), Gadget-4
group catalogues (fof_tab_NNN, fof_subhalo_tab_NNN), the text diagnostics
Gadget writes (energy.txt, info.txt, ...) and parameter files, and checks that
each snapshot has a Header group. SnapshotInspector reports, for each of the six
particle types, how many particles a snapshot has and whether it stores
velocities and masses; for a multi-file snapshot the counts come from
NumPart_Total. Neither reads a particle array. The application controller uses
both when a folder is opened, and the layer panel shows the type list.
"""

import re
import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto


# ============================================================
#  FILE CLASSIFICATION
# ============================================================

class FileType(Enum):
    """Classification of Gadget output files."""
    SNAPSHOT        = auto()
    BACKUP_SNAPSHOT = auto()
    FOF_CATALOG     = auto()
    DIAGNOSTIC_TXT  = auto()
    PARAMETER_FILE  = auto()
    UNKNOWN         = auto()


@dataclass
class FileRecord:
    """Metadata for a single discovered file."""
    filepath:      Path
    filetype:      FileType
    index:         Optional[int]  = None
    filesize_mb:   float          = 0.0
    is_readable:   bool           = True
    error_message: Optional[str]  = None
    # Header attributes, filled in by the validation pass. Nothing reads them
    # at present; the snapshot loader opens the file itself.
    cached_header: Optional[dict] = field(default=None, repr=False)


@dataclass
class FileManifest:
    """Complete inventory of a Gadget output directory."""
    root_dir:         Path
    snapshots:        List[FileRecord]
    diagnostics:      Dict[str, FileRecord]
    parameters:       List[FileRecord]
    backup_snapshots: List[FileRecord]
    fof_catalogs:     List[FileRecord]
    unknown_files:    List[FileRecord]
    errors:           List[str]

    def get_snapshot_count(self) -> int:
        return len(self.snapshots)

    def get_snapshot_files(self) -> List[Path]:
        """Return snapshot file paths sorted by index."""
        return [s.filepath for s in self.snapshots]

    def get_timeline_range(self) -> Tuple[int, int]:
        """
        Return (min_index, max_index) from the numeric indices embedded in
        snapshot filenames. When no file name carries an index, the positions
        in the sorted list (0 .. N-1) are used, so the result is always a pair
        of integers.
        """
        if not self.snapshots:
            return (0, 0)

        valid = [s.index for s in self.snapshots if s.index is not None]
        if not valid:
            # Fallback: treat position in the sorted list as the index
            return (0, len(self.snapshots) - 1)

        return (min(valid), max(valid))

    def has_diagnostic(self, filename: str) -> bool:
        return filename in self.diagnostics

    def get_summary(self) -> str:
        """Human-readable inventory summary."""
        parts = [f"{len(self.snapshots)} snapshots"]
        if self.diagnostics:
            parts.append(f"{len(self.diagnostics)} diagnostics")
        if self.fof_catalogs:
            parts.append(f"{len(self.fof_catalogs)} FOF catalogs")
        if self.backup_snapshots:
            parts.append(f"{len(self.backup_snapshots)} backups (ignored)")
        if self.unknown_files:
            parts.append(f"{len(self.unknown_files)} unknown files")
        return ", ".join(parts)


# ============================================================
#  FILE DISCOVERY ENGINE
# ============================================================

class FileDiscoveryEngine:
    """Scans a Gadget output directory and builds a FileManifest."""

    KNOWN_DIAGNOSTICS = [
        'energy.txt',
        'info.txt',
        'density.txt',
        'hydro.txt',
        'domain.txt',
        'balance.txt',
        'growthfac.txt',
        'timebins.txt',
        'timings.txt',
        'cpu.txt',
        'cpu.csv',
        'sfr.txt',
        'memory.txt',
    ]

    KNOWN_PARAMETERS = [
        'parameters-usedvalues',
        'param.txt',
        'Config.sh',
        'config.txt',
    ]

    def __init__(self,
                 scan_subfolders:     bool = False,
                 include_backups:     bool = False,
                 validate_snapshots:  bool = True):
        self.scan_subfolders    = scan_subfolders
        self.include_backups    = include_backups
        self.validate_snapshots = validate_snapshots

    def discover(self, root_dir: Path) -> FileManifest:
        """
        Main entry point: scan directory and classify all files.

        Returns:
            FileManifest with every file categorised.
        """
        root_dir = Path(root_dir)

        if not root_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {root_dir}")
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_dir}")
        if not self._is_gadget_directory(root_dir):
            print(f"WARNING: {root_dir} may not be a Gadget output directory")

        manifest = FileManifest(
            root_dir=root_dir,
            snapshots=[],
            diagnostics={},
            parameters=[],
            backup_snapshots=[],
            fof_catalogs=[],
            unknown_files=[],
            errors=[],
        )

        files = (list(root_dir.rglob('*'))
                 if self.scan_subfolders
                 else list(root_dir.glob('*')))
        files = [f for f in files if f.is_file()]
        print(f"Found {len(files)} files in {root_dir}")

        for filepath in files:
            self._classify_and_add(filepath, manifest)

        manifest.snapshots.sort(
            key=lambda x: x.index if x.index is not None else 0)

        if self.validate_snapshots:
            self._validate_snapshots(manifest)

        return manifest

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _is_gadget_directory(self, root_dir: Path) -> bool:
        has_energy    = (root_dir / 'energy.txt').exists()
        has_info      = (root_dir / 'info.txt').exists()
        # snapshot_* (Gadget-4), snap_* (Gadget-2/3, EAGLE's snap_028_z000p000.0.hdf5), and
        # folders of multi-file snapshots (snapdir_*, EAGLE's snapshot_028_z000p000/).
        has_snapshots = any(
            next(root_dir.glob(pattern), None) is not None
            for pattern in ('snapshot_*.hdf5', 'snapshot_*.h5', 'snap_*.hdf5', 'snap_*.h5', 'snapdir_*')
        ) or any(p.is_dir() for p in root_dir.glob('snapshot_*'))
        return has_energy or has_info or has_snapshots

    def _classify_and_add(self, filepath: Path, manifest: FileManifest):
        """Classify one file and route it to the correct manifest list."""
        filename = filepath.name
        filetype, index = self._classify_file(filepath)

        try:
            filesize_mb = filepath.stat().st_size / (1024 * 1024)
        except OSError:
            filesize_mb = 0.0

        record = FileRecord(
            filepath=filepath,
            filetype=filetype,
            index=index,
            filesize_mb=filesize_mb,
        )

        if filetype == FileType.SNAPSHOT:
            manifest.snapshots.append(record)

        elif filetype == FileType.BACKUP_SNAPSHOT:
            if self.include_backups:
                manifest.backup_snapshots.append(record)
            else:
                manifest.errors.append(f"Skipped backup: {filename}")

        elif filetype == FileType.FOF_CATALOG:
            manifest.fof_catalogs.append(record)

        elif filetype == FileType.DIAGNOSTIC_TXT:
            manifest.diagnostics[filename] = record

        elif filetype == FileType.PARAMETER_FILE:
            manifest.parameters.append(record)

        elif filetype == FileType.UNKNOWN:
            if not filename.startswith('.'):
                manifest.unknown_files.append(record)

    def _classify_file(self, filepath: Path) -> Tuple[FileType, Optional[int]]:
        """Determine the FileType and extract a numeric index when applicable."""
        filename = filepath.name

        if filename.startswith('bak-') and 'snapshot' in filename.lower():
            return (FileType.BACKUP_SNAPSHOT, self._extract_snapshot_index(filename))

        if (filename.startswith('snapshot_') or filename.startswith('snap_')):
            if filename.endswith('.hdf5') or filename.endswith('.h5'):
                return (FileType.SNAPSHOT, self._extract_snapshot_index(filename))

        # Gadget-4 group catalogues: fof_tab_NNN (FoF only) and fof_subhalo_tab_NNN (with
        # SUBFIND), one file or pieces .0, .1, ...
        if re.match(r'fof_(?:subhalo_)?tab_\d+', filename.lower()):
            return (FileType.FOF_CATALOG, self._extract_snapshot_index(filename))

        if filename in self.KNOWN_DIAGNOSTICS:
            return (FileType.DIAGNOSTIC_TXT, None)

        if filename in self.KNOWN_PARAMETERS:
            return (FileType.PARAMETER_FILE, None)

        return (FileType.UNKNOWN, None)

    def _extract_snapshot_index(self, filename: str) -> Optional[int]:
        """Extract the numeric snapshot index from a filename."""
        patterns = [
            r'snapshot[_-](\d+)(?:\.\d+)?\.(?:hdf5|h5)',
            r'snap[_-](\d+)(?:\.\d+)?\.(?:hdf5|h5)',
            r'bak-snapshot[_-](\d+)(?:\.\d+)?\.(?:hdf5|h5)',
            r'fof_(?:subhalo_)?tab[_-](\d+)(?:\.\d+)?\.(?:hdf5|h5)',
        ]
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        return None

    def _validate_snapshots(self, manifest: FileManifest):
        """
        Quick validation: attempt to open each snapshot's HDF5 Header group,
        and keep its attributes in FileRecord.cached_header.
        """
        for record in manifest.snapshots:
            try:
                with h5py.File(record.filepath, 'r') as f:
                    if 'Header' not in f:
                        record.is_readable    = False
                        record.error_message  = "Missing Header group"
                        manifest.errors.append(
                            f"Snapshot {record.index}: Missing Header")
                    else:
                        record.cached_header = dict(f['Header'].attrs)
            except Exception as e:
                record.is_readable   = False
                record.error_message = str(e)
                manifest.errors.append(f"Snapshot {record.index}: {e}")


# ============================================================
#  SNAPSHOT INSPECTOR
# ============================================================

@dataclass
class ParticleTypeInfo:
    """Metadata about one particle type within a snapshot."""
    type_id:         int
    type_name:       str
    count:           int
    exists:          bool
    has_coordinates: bool = True
    has_velocities:  bool = True
    has_masses:      bool = True


class SnapshotInspector:
    """
    Lightweight inspection of snapshot particle-type structure.
    Reads HDF5 group/dataset metadata only — never loads particle arrays.
    """

    TYPE_NAMES: Dict[int, str] = {
        0: "Gas",
        1: "Dark Matter",
        2: "Disk Stars",
        3: "Bulge Stars",
        4: "Star Particles",
        5: "Black Holes",        # Gadget-4 and EAGLE (old Gadget-2 runs used it for boundary particles)
    }

    @staticmethod
    def inspect(snapshot_path: Path) -> List[ParticleTypeInfo]:
        """
        Return a list of exactly 6 ParticleTypeInfo entries (types 0–5).

        If the file cannot be read, all six entries are blank, wherever the
        failure happened.
        """
        result: List[ParticleTypeInfo] = []

        def _blank_types() -> List[ParticleTypeInfo]:
            return [
                ParticleTypeInfo(
                    type_id=tid,
                    type_name=SnapshotInspector.TYPE_NAMES.get(tid, f"Type {tid}"),
                    count=0, exists=False,
                    has_coordinates=False, has_velocities=False, has_masses=False,
                )
                for tid in range(6)
            ]

        try:
            with h5py.File(snapshot_path, 'r') as f:
                # A multi-file snapshot's counts are in NumPart_Total, not in this piece.
                hdr = f['Header'].attrs if 'Header' in f else {}
                totals = None
                if int(np.atleast_1d(hdr.get('NumFilesPerSnapshot', 1))[0]) > 1 and 'NumPart_Total' in hdr:
                    totals = np.asarray(hdr['NumPart_Total'], dtype=np.int64)
                    if 'NumPart_Total_HighWord' in hdr:
                        totals = totals + (np.asarray(hdr['NumPart_Total_HighWord'], dtype=np.int64) << 32)
                for type_id in range(6):
                    key = f'PartType{type_id}'

                    if totals is not None and type_id < len(totals) and totals[type_id] > 0:
                        count          = int(totals[type_id])
                        exists         = True
                        grp            = f.get(key)
                        has_velocities = grp is not None and ('Velocities' in grp or 'Velocity' in grp)
                        has_masses     = grp is not None and ('Masses' in grp or 'Mass' in grp)
                    elif key in f and 'Coordinates' in f[key]:
                        # len() on an HDF5 dataset reads shape only — no data load
                        count          = len(f[key]['Coordinates'])
                        exists         = count > 0
                        # EAGLE names these Velocity and Mass
                        has_velocities = 'Velocities' in f[key] or 'Velocity' in f[key]
                        has_masses     = 'Masses'     in f[key] or 'Mass'     in f[key]
                    else:
                        count          = 0
                        exists         = False
                        has_velocities = False
                        has_masses     = False

                    result.append(ParticleTypeInfo(
                        type_id=type_id,
                        type_name=SnapshotInspector.TYPE_NAMES.get(
                            type_id, f"Type {type_id}"),
                        count=count,
                        exists=exists,
                        has_coordinates=exists,
                        has_velocities=has_velocities,
                        has_masses=has_masses,
                    ))

        except Exception as e:
            print(f"Error inspecting snapshot {snapshot_path}: {e}")
            # A fresh list: ``result`` may allready hold some entries.
            return _blank_types()

        return result

    @staticmethod
    def get_existing_types(snapshot_path: Path) -> List[int]:
        """Return the list of type IDs that have at least one particle."""
        return [info.type_id for info in SnapshotInspector.inspect(snapshot_path)
                if info.exists]