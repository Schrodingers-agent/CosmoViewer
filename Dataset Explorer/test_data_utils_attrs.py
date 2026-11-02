"""
Reading HDF5 attributes and bounded full reads (data_utils.py).

  * read_dataset_attrs returns the dataset's attributes, decoded, without
    reading the array itself.
  * _read_dataset_full refuses to run without a particle budget and returns
    the values as float64 when given one.
  * coerce_attrs turns numpy bytes and scalars into plain Python values.

    python3 -m pytest -q test_data_utils_attrs.py
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


def _make_fixture_hdf5() -> str:
    """A minimal HDF5 file with one dataset and its unit attributes; returns the path."""
    fd, path = tempfile.mkstemp(suffix=".hdf5")
    os.close(fd)
    with h5py.File(path, "w") as f:
        grp = f.create_group("PartType0")
        ds = grp.create_dataset("Density", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        ds.attrs["to_cgs"] = 6.77e-22
        ds.attrs["h-scale-exponent"] = 2.0
        ds.attrs["aexp-scale-exponent"] = -3.0
        ds.attrs["VarDescription"] = b"Comoving density"
    return path


class TestReadDatasetAttrs(unittest.TestCase):

    def setUp(self):
        self.path = _make_fixture_hdf5()

    def tearDown(self):
        os.unlink(self.path)

    def test_returns_decoded_attributes_and_no_data(self):
        from data_utils import read_dataset_attrs
        with h5py.File(self.path, "r") as f:
            attrs = read_dataset_attrs(f["PartType0"], "Density")
        self.assertIsInstance(attrs, dict)
        self.assertNotIn("data", attrs)
        self.assertNotIn("raw", attrs)
        self.assertAlmostEqual(attrs["to_cgs"], 6.77e-22, places=30)
        self.assertEqual(attrs["h-scale-exponent"], 2.0)
        self.assertEqual(attrs["aexp-scale-exponent"], -3.0)
        self.assertEqual(attrs["VarDescription"], "Comoving density")   # bytes decoded

    def test_never_reads_the_array(self):
        """Reading a dataset's values is forbidden while read_dataset_attrs runs."""
        from data_utils import read_dataset_attrs
        sentinel = RuntimeError("read_dataset_attrs read the array")

        def forbidden_getitem(self_ds, key):
            raise sentinel

        with h5py.File(self.path, "r") as f:
            with patch.object(h5py.Dataset, "__getitem__", forbidden_getitem):
                try:
                    attrs = read_dataset_attrs(f["PartType0"], "Density")
                except RuntimeError as exc:
                    if exc is sentinel:
                        self.fail("read_dataset_attrs read the array")
                    raise
        self.assertAlmostEqual(attrs["to_cgs"], 6.77e-22, places=30)


class TestReadDatasetFull(unittest.TestCase):

    def setUp(self):
        self.path = _make_fixture_hdf5()

    def tearDown(self):
        os.unlink(self.path)

    def test_refuses_without_a_budget(self):
        from data_utils import _read_dataset_full
        with h5py.File(self.path, "r") as f:
            with self.assertRaises(ValueError) as ctx:
                _read_dataset_full(f["PartType0"], "Density", max_particles=None)
        self.assertIn("_read_dataset_full called without a max_particles budget", str(ctx.exception))

    def test_returns_every_value_as_float64_within_the_budget(self):
        from data_utils import _read_dataset_full
        with h5py.File(self.path, "r") as f:
            result = _read_dataset_full(f["PartType0"], "Density", max_particles=100)
        self.assertEqual(result["raw"].dtype, np.float64)
        np.testing.assert_array_equal(result["raw"], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["attrs"]["to_cgs"], 6.77e-22, places=30)


class TestCoerceAttrs(unittest.TestCase):

    def test_numpy_bytes_scalars_and_arrays(self):
        from data_utils import coerce_attrs
        proxy = {"label": np.bytes_(b"hello"), "mass": np.float32(1.5),
                 "ids": np.array([1, 2, 3])}
        result = coerce_attrs(proxy)
        self.assertEqual(result["label"], "hello")
        self.assertAlmostEqual(result["mass"], 1.5, places=5)
        np.testing.assert_array_equal(result["ids"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
