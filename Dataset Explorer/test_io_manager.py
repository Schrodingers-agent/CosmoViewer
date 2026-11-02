"""
test_io_manager.py
──────────────────
pytest suite for VirtualSnapshotSet (io_manager.py).

Run with:
    pytest test_io_manager.py -v

Three unit tests verify the three core correctness properties:
    1. 64-bit HighWord particle count reconstruction
    2. Phase-continuous residual striding across file boundaries
    3. Missing-chunk FileNotFoundError guard

Two integration / smoke tests verify orchestrator-level wiring:
    4. Single-file legacy compatibility (no regression)
    5. Multi-file pipeline round-trip with concatenated array verification
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import h5py
import pytest

from io_manager import VirtualSnapshotSet


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build a minimal Gadget-4-style HDF5 chunk on disk
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(
    path:         str,
    data:         dict[str, np.ndarray],  # {"PartType0/Density": array, ...}
    n_part_total: list[int] | None = None,
    n_part_high:  list[int] | None = None,
    n_part_this:  list[int] | None = None,
    n_files:      int              = 1,
) -> None:
    """Write a minimal Gadget-4-compatible HDF5 file to disk."""
    n_part_total = n_part_total or [0] * 6
    n_part_high  = n_part_high  or [0] * 6
    n_part_this  = n_part_this  or list(n_part_total)

    with h5py.File(path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["NumPart_Total"]         = np.array(n_part_total, dtype=np.uint32)
        hdr.attrs["NumPart_Total_HighWord"] = np.array(n_part_high,  dtype=np.uint32)
        hdr.attrs["NumPart_ThisFile"]       = np.array(n_part_this,  dtype=np.uint32)
        hdr.attrs["NumFilesPerSnapshot"]    = n_files

        for ds_path, arr in data.items():
            grp_name, ds_name = ds_path.split("/", 1)
            grp = f.require_group(grp_name)
            grp.create_dataset(ds_name, data=arr)


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Test 1: 64-bit HighWord particle count reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def test_highword_calculation(tmp_path):
    """
    Mock an HDF5 header with:
        NumPart_Total         = [1000, 0, 0, 0, 0, 0]
        NumPart_Total_HighWord = [   1, 0, 0, 0, 0, 0]

    Expected 64-bit total for PartType0:
        (1 << 32) | 1000 = 4_294_968_296
    """
    chunk0 = str(tmp_path / "snap.0.hdf5")
    _make_chunk(
        chunk0,
        data         = {"PartType0/Density": np.ones(1000, dtype=np.float32)},
        n_part_total = [1000, 0, 0, 0, 0, 0],
        n_part_high  = [   1, 0, 0, 0, 0, 0],
        n_files      = 1,
    )

    vset = VirtualSnapshotSet(chunk0)
    expected = (1 << 32) | 1000   # = 4_294_968_296

    assert vset.particle_counts["PartType0"] == expected, (
        f"Expected {expected}, got {vset.particle_counts['PartType0']}"
    )
    assert vset.particle_counts["PartType1"] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Test 2: Phase-continuous residual striding across file boundaries
# ─────────────────────────────────────────────────────────────────────────────

def test_residual_striding(tmp_path):
    """
    Three files of 10 particles each hold the values 0..29. With max_particles=7
    the stride is max(1, 30 // 7) = 4, and the stride continues across file
    boundaries instead of restarting at row 0 of each file:
        file 0: rows 0, 4, 8   -> 0, 4, 8      (2 rows left over)
        file 1: rows 2, 6      -> 12, 16       (starts at 4 - 2 = 2)
        file 2: rows 0, 4, 8   -> 20, 24, 28
    Result: every 4th particle of the concatenated snapshot, [0, 4, ..., 28].
    """
    base    = str(tmp_path / "snap.")
    n_total = [30, 0, 0, 0, 0, 0]

    for i in range(3):
        arr       = np.arange(i * 10, (i + 1) * 10, dtype=np.float64)
        n_this    = [10, 0, 0, 0, 0, 0]
        _make_chunk(
            f"{base}{i}.hdf5",
            data         = {"PartType0/Density": arr},
            n_part_total = n_total,
            n_part_this  = n_this,
            n_files      = 3,
        )

    vset   = VirtualSnapshotSet(f"{base}0.hdf5")
    result = vset.extract_array("PartType0/Density", max_particles=7).data

    expected = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.float64)
    np.testing.assert_array_equal(result, expected,
        err_msg=f"Striding mismatch: got {result}, expected {expected}")


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Test 3: Missing chunk FileNotFoundError guard
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_file_guard(tmp_path):
    """
    NumFilesPerSnapshot = 3, but only files 0 and 2 exist on disk.
    VirtualSnapshotSet.__init__ must raise FileNotFoundError.
    """
    base    = str(tmp_path / "snap.")
    n_total = [30, 0, 0, 0, 0, 0]

    # Create file 0 (chunk 0 — must exist for VirtualSnapshotSet to read header)
    _make_chunk(
        f"{base}0.hdf5",
        data         = {"PartType0/Density": np.ones(10, dtype=np.float32)},
        n_part_total = n_total,
        n_files      = 3,                   # declares 3 chunks
    )
    # Create file 2 — deliberately skip file 1
    _make_chunk(
        f"{base}2.hdf5",
        data         = {"PartType0/Density": np.ones(10, dtype=np.float32)},
        n_part_total = n_total,
        n_files      = 3,
    )
    # File 1 does NOT exist on disk

    with pytest.raises(FileNotFoundError) as exc_info:
        VirtualSnapshotSet(f"{base}0.hdf5")

    assert "snap." in str(exc_info.value), (
        f"Error message should name the missing chunk: {exc_info.value}")


# ─────────────────────────────────────────────────────────────────────────────
#  Integration Test 4: Single-file legacy compatibility
# ─────────────────────────────────────────────────────────────────────────────

def test_single_file_legacy(tmp_path):
    """
    Load a plain Gadget-4 single-file snapshot (no numeric suffix pattern,
    NumFilesPerSnapshot = 1).  Verify that:
      - VirtualSnapshotSet initialises without error
      - self.files == [filepath]
      - extract_array returns the full dataset unchanged
    """
    snap = str(tmp_path / "snapshot_015.hdf5")   # no .N. numeric suffix
    data = np.arange(100, dtype=np.float32)
    _make_chunk(
        snap,
        data         = {"PartType0/Density": data},
        n_part_total = [100, 0, 0, 0, 0, 0],
        n_files      = 1,
    )

    vset = VirtualSnapshotSet(snap)

    assert vset.files == [snap], f"Expected single-file list, got {vset.files}"
    assert vset.particle_counts["PartType0"] == 100

    result = vset.extract_array("PartType0/Density", max_particles=None).data
    assert len(result) == 100
    np.testing.assert_allclose(result, data.astype(np.float64))


# ─────────────────────────────────────────────────────────────────────────────
#  Integration Test 5: Multi-file pipeline round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_multifile_pipeline_roundtrip(tmp_path):
    """
    2-file snapshot, 50 particles each (100 total).
    max_particles = None → full read, no striding.
    Verify concatenated output == np.arange(100).

    Also verifies:
      - global_header is populated
      - get_dtype returns correct string
      - get_attributes returns a dict
    """
    base    = str(tmp_path / "snap.")
    n_total = [100, 0, 0, 0, 0, 0]

    for i in range(2):
        arr    = np.arange(i * 50, (i + 1) * 50, dtype=np.float32)
        n_this = [50, 0, 0, 0, 0, 0]
        _make_chunk(
            f"{base}{i}.hdf5",
            data         = {"PartType0/Density": arr},
            n_part_total = n_total,
            n_part_this  = n_this,
            n_files      = 2,
        )

    vset = VirtualSnapshotSet(f"{base}0.hdf5")

    # Full read — no striding
    result = vset.extract_array("PartType0/Density", max_particles=None).data
    expected = np.arange(100, dtype=np.float64)
    np.testing.assert_allclose(result, expected,
        err_msg=f"Round-trip mismatch: {result[:10]}...")

    # Metadata helpers
    assert vset.particle_counts["PartType0"] == 100
    assert "NumFilesPerSnapshot" in vset.global_header
    assert vset.get_dtype("PartType0/Density") == "float32"   # on-disk dtype preserved
    attrs = vset.get_attributes("PartType0/Density")
    assert isinstance(attrs, dict)   # may be empty if no attrs set


# ─────────────────────────────────────────────────────────────────────────────
#  Integration Test 6: Histogram receives FULL data, Line receives downsampled
# ─────────────────────────────────────────────────────────────────────────────

def test_tab_budget_routing(tmp_path):
    """
    1000 particles in a single file.
    - max_particles=None  → all 1000 returned  (Histogram path)
    - max_particles=100   → every 10th particle (Line path)
    """
    snap = str(tmp_path / "snap.0.hdf5")
    data = np.random.rand(1000).astype(np.float32)
    _make_chunk(
        snap,
        data         = {"PartType0/Density": data},
        n_part_total = [1000, 0, 0, 0, 0, 0],
        n_files      = 1,
    )

    vset = VirtualSnapshotSet(snap)

    hist_arr = vset.extract_array("PartType0/Density", max_particles=None).data
    np.testing.assert_array_equal(hist_arr, data)                 # every particle, in file order

    line_arr = vset.extract_array("PartType0/Density", max_particles=100).data
    np.testing.assert_array_equal(line_arr, data[::10])           # stride 1000 // 100 = 10
