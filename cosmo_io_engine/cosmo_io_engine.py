"""
CosmoViewer I/O Engine.

Readers for Gadget binary snapshots (SnapFormat 1 and 2) and, more simply, HDF5
snapshots, plus a few older helpers. The application reads HDF5 snapshots with
h5py itself (snapshot_io/snapshot_loader.py) and comes here only for binary
files, through SnapshotHarvester and GadgetBinaryReader.

The other classes are older code the application no longer relies on
(spatial/octree.py and snapshot_io/halo_catalog.py still wrap two of them). The
application's FoF is snapshot_io/fof.py, not FriendsOfFriends below, and
OctreeSpatialIndex builds no tree. Their docstrings say how each differs
from what its name suggests.

Previously named pygadgetviewer.py.
"""

import os
import glob
import re
import struct
import math
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# 1. GADGET CONSTANTS & HEADER SPECS
# ==============================================================================

PARTICLE_TYPE_NAMES = {
    0: "Gas (Type 0)",
    1: "Halo/DM (Type 1)",
    2: "Disk (Type 2)",
    3: "Bulge (Type 3)",
    4: "Stars (Type 4)",
    5: "Bndry/BH (Type 5)"
}

HEADER_STRUCT_FMT = "6I 6d d d i i 6I i i d d d d i i 6I i 60x"  # 256 bytes


class GadgetHeader:
    """Structure storing Gadget simulation header attributes."""
    def __init__(self):
        self.npart = np.zeros(6, dtype=np.uint32)
        self.massarr = np.zeros(6, dtype=np.float64)
        self.time = 0.0
        self.redshift = 0.0
        self.flag_sfr = 0
        self.flag_feedback = 0
        self.nall = np.zeros(6, dtype=np.uint64)
        self.flag_cooling = 0
        self.num_files = 1
        self.boxsize = 0.0
        self.omega0 = 0.0
        self.omega_lambda = 0.0
        self.hubble_param = 0.0
        self.flag_stellarage = 0
        self.flag_metals = 0
        self.nall_hw = np.zeros(6, dtype=np.uint32)
        self.flag_entropy_instead_with_eqp = 0
        self.endianness = '<'

    @property
    def total_particles(self) -> np.ndarray:
        return self.nall + (self.nall_hw.astype(np.uint64) << 32)


# ==============================================================================
# 2. FRIENDS-OF-FRIENDS (FoF) AUTOMATIC GROUP DETECTOR
# ==============================================================================

class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression and rank optimization."""
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, i: int) -> int:
        path = []
        curr = i
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]
        for node in path:
            self.parent[node] = curr
        return curr

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


class FriendsOfFriends:
    """
    Early FoF implementation, unused by the application (see snapshot_io/fof.py).

    It departs from Gadget-4 in ways that change the groups: N_min defaults to
    20 rather than 32, the default length uses (V/N)^(1/3) with V the bounding
    box of the particles when no box size is given, and the fallback without
    scipy ignores periodic wrapping. Group numbers start at 1, 0 = no group.
    """

    @staticmethod
    def find_groups(positions: np.ndarray,
                    boxsize: Optional[float] = None,
                    b_linking: float = 0.20,
                    r_link: Optional[float] = None,
                    min_group_size: int = 20) -> Tuple[np.ndarray, int, Dict[int, int]]:
        """
        Executes Friends-of-Friends group finding algorithm.
        
        Parameters:
            positions: (N, 3) particle coordinates array.
            boxsize: Periodic box size (optional).
            b_linking: Dimensionless linking length parameter (default 0.20).
            r_link: Explicit linking length in the units of positions (if None, b * (V/N)^(1/3)).
            min_group_size: Minimum particle count threshold to constitute a group (default 20).
            
        Returns:
            Tuple of (group_ids array of shape (N,), n_groups count, dict of group_id -> particle count).
        """
        n_part = len(positions)
        if n_part == 0:
            return np.zeros(0, dtype=np.int32), 0, {}

        # Calculate linking radius if not specified
        if r_link is None:
            if boxsize is not None and boxsize > 0:
                volume = boxsize ** 3
            else:
                min_c = np.min(positions, axis=0)
                max_c = np.max(positions, axis=0)
                bounds = np.maximum(max_c - min_c, 1.0)
                volume = float(np.prod(bounds))
            mean_interparticle_spacing = (volume / n_part) ** (1.0 / 3.0)
            r_link = b_linking * mean_interparticle_spacing

        uf = UnionFind(n_part)

        if HAS_SCIPY:
            tree = cKDTree(positions, boxsize=boxsize)
            pairs = tree.query_pairs(r=r_link, output_type='ndarray')
            for i, j in pairs:
                uf.union(i, j)
        else:
            # Fallback simple spatial grid bucket if scipy is unavailable
            grid_size = max(r_link, 1.0)
            min_c = np.min(positions, axis=0)
            grid_coords = np.floor((positions - min_c) / grid_size).astype(int)
            buckets = {}
            for idx, gc in enumerate(grid_coords):
                key = tuple(gc)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(idx)
            
            for key, p_list in buckets.items():
                # Check neighbor cells
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            n_key = (key[0]+dx, key[1]+dy, key[2]+dz)
                            if n_key in buckets:
                                for i in p_list:
                                    for j in buckets[n_key]:
                                        if i < j:
                                            d2 = np.sum((positions[i] - positions[j])**2)
                                            if d2 <= r_link**2:
                                                uf.union(i, j)

        # Collect root components
        root_map = {}
        for i in range(n_part):
            root = uf.find(i)
            if root not in root_map:
                root_map[root] = []
            root_map[root].append(i)

        # Filter clusters by min_group_size and sort descending by size
        valid_clusters = [pts for pts in root_map.values() if len(pts) >= min_group_size]
        valid_clusters.sort(key=len, reverse=True)

        group_ids = np.zeros(n_part, dtype=np.int32)
        group_sizes = {}
        for grp_idx, pts in enumerate(valid_clusters, start=1):
            group_ids[pts] = grp_idx
            group_sizes[grp_idx] = len(pts)

        return group_ids, len(valid_clusters), group_sizes


# ==============================================================================
# 3. GADGET BINARY READER (TYPE 1 & TYPE 2)
# ==============================================================================

class GadgetBinaryReader:
    """
    Reader for Gadget Type 1 (SnapFormat 1) and Type 2 (SnapFormat 2) unformatted binary snapshots.

    Every block is a Fortran record (4-byte length, payload, the same length again); Type 2 puts
    a record with a 4-character name and the next record's size before each block. Blocks are
    split by the particle counts in each file's own header: POS, VEL and ID over every type,
    MASS over the types whose MassTable entry is 0, U, RHO and HSML over gas. The element size
    is taken from the block length, so double-precision positions and 64-bit IDs are read as
    such. A multi-file snapshot (<base>.0, <base>.1, ...) is read whole from any of its files.

    Fixed 2026-10-27: Type 1 parsed each file's header into the reader's own header and then
    split the blocks with a blank one (every type read 0 particles), and re-read the header in
    the middle of the blocks; Type 2 split every file with the first file's counts and MASS over
    every type; IDs were always read as 32-bit.
    """

    _FILE_RE = re.compile(r"^(?P<base>.+)\.(?P<piece>\d+)$")

    def __init__(self, filename: str):
        self.filename = filename
        self.endianness = '<'
        self.is_type2 = False
        self.header = GadgetHeader()
        self._inspect()

    def _inspect(self):
        with open(self.filename, 'rb') as f:
            first = f.read(4)
        if len(first) < 4:
            raise ValueError(f"{self.filename}: too short for a Gadget binary file")
        rec_le, rec_be = struct.unpack('<I', first)[0], struct.unpack('>I', first)[0]
        if rec_le in (256, 8):
            self.endianness, self.is_type2 = '<', rec_le == 8
        elif rec_be in (256, 8):
            self.endianness, self.is_type2 = '>', rec_be == 8
        else:
            raise ValueError(f"Unrecognized Gadget binary format marker: {rec_le}")
        self.header = self._file_header(self.filename)
        self.header.endianness = self.endianness

    # ------------------------------------------------------------------ records

    def _record(self, f) -> Optional[bytes]:
        """One Fortran record's payload, or None at the end of the file."""
        b = f.read(4)
        if len(b) < 4:
            return None
        n = struct.unpack(self.endianness + 'I', b)[0]
        data = f.read(n)
        end = f.read(4)
        if len(data) != n or len(end) < 4 or struct.unpack(self.endianness + 'I', end)[0] != n:
            raise ValueError(f"{getattr(f, 'name', '')}: damaged record (length markers do not match)")
        return data

    def _blocks(self, f):
        """(name or None, payload) for every block of the file, the header first."""
        while True:
            if self.is_type2:
                label = self._record(f)
                if label is None:
                    return
                name = label[:4].decode('ascii', errors='ignore').strip()
            else:
                name = None
            data = self._record(f)
            if data is None:
                return
            yield name, data

    def _parse_header_bytes(self, data: bytes) -> GadgetHeader:
        h = GadgetHeader()
        unpacked = struct.unpack(self.endianness + HEADER_STRUCT_FMT, data[:256])
        h.npart = np.array(unpacked[0:6], dtype=np.uint32)
        h.massarr = np.array(unpacked[6:12], dtype=np.float64)
        h.time = unpacked[12]
        h.redshift = unpacked[13]
        h.flag_sfr = unpacked[14]
        h.flag_feedback = unpacked[15]
        h.nall = np.array(unpacked[16:22], dtype=np.uint64)
        h.flag_cooling = unpacked[22]
        h.num_files = unpacked[23]
        h.boxsize = unpacked[24]
        h.omega0 = unpacked[25]
        h.omega_lambda = unpacked[26]
        h.hubble_param = unpacked[27]
        h.flag_stellarage = unpacked[28]
        h.flag_metals = unpacked[29]
        h.nall_hw = np.array(unpacked[30:36], dtype=np.uint32)
        h.flag_entropy_instead_with_eqp = unpacked[36]
        h.endianness = self.endianness
        return h

    def _file_header(self, fname: str) -> GadgetHeader:
        with open(fname, 'rb') as f:
            for name, data in self._blocks(f):
                if self.is_type2 and name != 'HEAD':
                    raise ValueError(f"{fname}: the first block is {name!r}, not HEAD")
                return self._parse_header_bytes(data)
        raise ValueError(f"{fname}: no header")

    # ------------------------------------------------------------------ particles

    def read_particles(self) -> Dict[int, Dict[str, np.ndarray]]:
        all_data = {i: {} for i in range(6)}
        for fname in self._get_file_list():
            for t, fields in self._read_file(fname).items():
                for key, arr in fields.items():
                    all_data[t].setdefault(key, []).append(arr)
        for t in range(6):
            for key in list(all_data[t]):
                all_data[t][key] = np.concatenate(all_data[t][key], axis=0)
        return all_data

    _TYPE1_ORDER = ('POS', 'VEL', 'ID', 'MASS', 'U', 'RHO', 'HSML')
    _NAMES = {'POS': 'Coordinates', 'VEL': 'Velocities', 'ID': 'ParticleIDs', 'MASS': 'Masses',
              'U': 'InternalEnergy', 'RHO': 'Density', 'HSML': 'SmoothingLength'}

    def _read_file(self, fname: str) -> Dict[int, Dict[str, np.ndarray]]:
        out = {t: {} for t in range(6)}
        h = None
        order = list(self._TYPE1_ORDER)
        with open(fname, 'rb') as f:
            for name, data in self._blocks(f):
                if h is None:
                    h = self._parse_header_bytes(data)
                    npart = h.npart.astype(np.int64)
                    with_mass = [t for t in range(6) if npart[t] > 0 and h.massarr[t] == 0]
                    if not with_mass:
                        order.remove('MASS')      # Type 1 files have no MASS block then
                    continue
                if name is None:                  # Type 1: blocks come in a fixed order
                    if not order:
                        break                     # further blocks (e.g. code-specific) are not read
                    name = order.pop(0)
                    if name in ('U', 'RHO', 'HSML') and npart[0] == 0:
                        break
                key = self._NAMES.get(name)
                if key is None:
                    continue
                types = (range(6) if name in ('POS', 'VEL', 'ID') else
                         with_mass if name == 'MASS' else [0])
                counts = [int(npart[t]) for t in types]
                n = sum(counts)
                if n == 0:
                    continue
                ncomp = 3 if name in ('POS', 'VEL') else 1
                size = len(data) // (n * ncomp)
                if size * n * ncomp != len(data) or size not in (4, 8):
                    raise ValueError(f"{Path(fname).name}: {name} block holds {len(data)} bytes, "
                                     f"which does not fit {n} particles")
                kind = 'u' if name == 'ID' else 'f'
                arr = np.frombuffer(data, dtype=f"{self.endianness}{kind}{size}")
                if ncomp == 3:
                    arr = arr.reshape(-1, 3)
                start = 0
                for t, c in zip(types, counts):
                    if c:
                        out[t][key] = arr[start:start + c]
                    start += c
        if h is None:
            raise ValueError(f"{fname}: no header")
        for t in range(6):                        # equal masses are in the header only
            if npart[t] > 0 and 'Masses' not in out[t] and h.massarr[t] > 0:
                out[t]['Masses'] = np.full(int(npart[t]), h.massarr[t], dtype=np.float64)
        return out

    def _get_file_list(self) -> List[str]:
        if self.header.num_files <= 1:
            return [self.filename]
        m = self._FILE_RE.match(self.filename)
        if not m:
            raise ValueError(f"{self.filename}: the header says {self.header.num_files} files, "
                             f"but the name is not <base>.<i>")
        files = [f"{m.group('base')}.{i}" for i in range(self.header.num_files)]
        missing = [p for p in files if not os.path.exists(p)]
        if missing:                               # dropping a file would drop particles
            raise ValueError(f"{Path(missing[0]).name} and {len(missing) - 1} more of "
                             f"{self.header.num_files} files are missing")
        return files


# ==============================================================================
# 4. GADGET & SWIFT HDF5 READER
# ==============================================================================

class GadgetHDF5Reader:
    """Reader for Gadget, SWIFT, and EAGLE HDF5 snapshots: every dataset (or those named), read whole.

    If any piece of a multi-file snapshot is missing, only the named file is
    read, so particles go missing without an error. snapshot_io/snapshot_loader.py
    reads HDF5 itself and refuses incomplete sets instead.
    """

    def __init__(self, filename: str):
        if not HAS_H5PY:
            raise ImportError("h5py module is required to read HDF5 snapshots.")
        self.filename = filename
        self.header = {}
        self.particle_types = []
        self._inspect()

    def _inspect(self):
        with h5py.File(self.filename, 'r') as f:
            if 'Header' in f:
                hdr = f['Header']
                for k, v in hdr.attrs.items():
                    self.header[k] = v
            self.particle_types = [k for k in f.keys() if k.startswith('PartType')]

    def read_particles(self, datasets: Optional[List[str]] = None) -> Dict[int, Dict[str, np.ndarray]]:
        files = self._get_file_list()
        all_data = {i: {} for i in range(6)}

        for fname in files:
            with h5py.File(fname, 'r') as f:
                for ptype in range(6):
                    grp_name = f"PartType{ptype}"
                    if grp_name not in f:
                        continue
                    grp = f[grp_name]
                    for dname in grp.keys():
                        if datasets and dname not in datasets:
                            continue
                        val = grp[dname][:]
                        if dname not in all_data[ptype]:
                            all_data[ptype][dname] = [val]
                        else:
                            all_data[ptype][dname].append(val)

        for ptype in range(6):
            for key in list(all_data[ptype].keys()):
                all_data[ptype][key] = np.concatenate(all_data[ptype][key], axis=0)

        return all_data

    def _get_file_list(self) -> List[str]:
        nfiles = self.header.get('NumFilesPerSnapshot', 1)
        if isinstance(nfiles, np.ndarray):
            nfiles = nfiles[0]
        if nfiles <= 1:
            return [self.filename]
        
        base, ext = os.path.splitext(self.filename)
        prefix = base.rsplit('.', 1)[0]
        files = [f"{prefix}.{i}{ext}" for i in range(nfiles)]
        if all(os.path.exists(f) for f in files):
            return files
        return [self.filename]


# ==============================================================================
# 5. SPH KERNEL & SPATIAL ENGINE
# ==============================================================================

class SPHKernelEngine:
    """The 3-D cubic spline kernel W(r, h) in Gadget's convention: h is the full support radius.

    W = 8 / (pi h^3) * (1 - 6q^2 + 6q^3) for q = r/h <= 1/2, 2 (1 - q)^3 up to
    q = 1, and 0 beyond. Nothing in the application calls it.
    """

    @staticmethod
    def cubic_spline_kernel(r: np.ndarray, h: np.ndarray) -> np.ndarray:
        q = r / h
        norm = 8.0 / (np.pi * (h ** 3))
        w = np.zeros_like(q)
        
        mask1 = (q >= 0.0) & (q <= 0.5)
        mask2 = (q > 0.5) & (q <= 1.0)
        
        w[mask1] = norm[mask1] * (1.0 - 6.0 * (q[mask1]**2) + 6.0 * (q[mask1]**3))
        w[mask2] = norm[mask2] * (2.0 * ((1.0 - q[mask2])**3))
        return w


class OctreeSpatialIndex:
    """Box queries by a linear scan over ``pos``; despite the name, no tree is built."""

    def __init__(self, pos: np.ndarray, boxsize: float):
        self.pos = pos
        self.boxsize = boxsize

    def query_box(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> np.ndarray:
        mask = (self.pos[:, 0] >= xmin) & (self.pos[:, 0] <= xmax) & \
               (self.pos[:, 1] >= ymin) & (self.pos[:, 1] <= ymax) & \
               (self.pos[:, 2] >= zmin) & (self.pos[:, 2] <= zmax)
        return np.where(mask)[0]


# ==============================================================================
# 6. HALO CATALOG & GROUP READER
# ==============================================================================

class HaloCatalogReader:
    """Gadget-4 HDF5 and VELOCIraptor catalogues, every dataset read whole into a flat dict.

    Only one file of a split catalogue is read. The VELOCIraptor reader is
    untested; snapshot_io/group_catalogue.py is what the application uses.
    """

    @staticmethod
    def read_gadget4_groups(catalog_file: str) -> Dict[str, np.ndarray]:
        if not HAS_H5PY:
            raise ImportError("h5py required for Gadget-4 group catalog.")
        
        cat_data = {}
        with h5py.File(catalog_file, 'r') as f:
            if 'Group' in f:
                for k in f['Group'].keys():
                    cat_data[f"Group/{k}"] = f['Group'][k][:]
            if 'Subhalo' in f:
                for k in f['Subhalo'].keys():
                    cat_data[f"Subhalo/{k}"] = f['Subhalo'][k][:]
        return cat_data

    @staticmethod
    def read_velociraptor_groups(catalog_groups_file: str) -> Dict[str, np.ndarray]:
        if not HAS_H5PY:
            raise ImportError("h5py required for VELOCIraptor catalog.")
        
        data = {}
        with h5py.File(catalog_groups_file, 'r') as f:
            for k in f.keys():
                data[k] = f[k][:]
        return data


# ==============================================================================
# 7. UNIFIED FEATURE HARVESTER FOR COSMOVIEWER
# ==============================================================================

class SnapshotHarvester:
    """Picks a reader by file format and returns {ptype: {dataset name: array}}.

    HDF5 is recognised by the file suffix, binary by its first record marker
    (256 bytes for a Type 1 header, 8 for a Type 2 block label). sample_rate < 1
    draws an unseeded random subset, different on every call; the snapshot
    loader always reads everything and samples by ParticleID itself.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.format_type = self._detect_format()
        self.reader = None
        self._init_reader()

    def _detect_format(self) -> str:
        if self.filename.endswith('.hdf5') or self.filename.endswith('.h5'):
            return 'HDF5'
        with open(self.filename, 'rb') as f:
            first = f.read(4)
            rec, rec_be = struct.unpack('<I', first)[0], struct.unpack('>I', first)[0]
            if 256 in (rec, rec_be):
                return 'GADGET_BINARY_TYPE1'
            elif 8 in (rec, rec_be):              # the record holding a block name and size
                return 'GADGET_BINARY_TYPE2'
        raise ValueError(f"Unknown snapshot format for file {self.filename}")

    def _init_reader(self):
        if self.format_type == 'HDF5':
            self.reader = GadgetHDF5Reader(self.filename)
        else:
            self.reader = GadgetBinaryReader(self.filename)

    def harvest(self, species: Optional[List[int]] = None, sample_rate: float = 1.0) -> Dict[int, Dict[str, np.ndarray]]:
        raw_data = self.reader.read_particles()
        if species is None:
            species = list(raw_data.keys())

        harvested = {}
        for sp in species:
            if sp not in raw_data or not raw_data[sp]:
                continue
            
            harvested[sp] = {}
            n_part = len(next(iter(raw_data[sp].values())))
            
            if sample_rate < 1.0 and n_part > 0:
                indices = np.random.choice(n_part, size=int(n_part * sample_rate), replace=False)
                indices.sort()
                for key, arr in raw_data[sp].items():
                    harvested[sp][key] = arr[indices]
            else:
                for key, arr in raw_data[sp].items():
                    harvested[sp][key] = arr

        return harvested

    def harvest_to_pandas(self, particle_type: int = 0) -> Any:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas library is required for harvest_to_pandas()")

        harvested = self.harvest(species=[particle_type])
        if particle_type not in harvested:
            return pd.DataFrame()
        
        pdata = harvested[particle_type]
        df_dict = {}
        for k, v in pdata.items():
            if v.ndim == 1:
                df_dict[k] = v
            elif v.ndim == 2:
                for dim in range(v.shape[1]):
                    col_name = f"{k}_{['x', 'y', 'z'][dim] if v.shape[1]==3 else dim}"
                    df_dict[col_name] = v[:, dim]
        return pd.DataFrame(df_dict)


if __name__ == '__main__':
    print("PyGadgetViewer Core Engine with FoF Group Detector Ready.")
