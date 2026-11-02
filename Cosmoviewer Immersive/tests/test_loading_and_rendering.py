"""
Regression tests for snapshot loading and the 3D view (not FoF).

Run from the "Cosmoviewer Immersive" folder:
    QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_loading_and_rendering.py

Real-data tests run when COSMOVIEWER_DM_SNAPSHOT points at
DM-L50-N128/output/snapshot_000.hdf5; otherwise they are skipped.
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from snapshot_io import snapshot_loader as sl  # noqa: E402
from model.units import GadgetUnits  # noqa: E402

MPC_IN_CM = 3.085678e24
DM_SNAPSHOT = os.environ.get("COSMOVIEWER_DM_SNAPSHOT")


def _write_gadget4_like(path, n_dm=1000, n_gas=0, mass_dm=0.5, box=50.0):
    """Minimal Gadget-4 style file: DM masses only in MassTable, units in Parameters."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = box
        h.attrs["Time"] = 0.5
        h.attrs["Redshift"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["MassTable"] = np.array([0.0, mass_dm])
        h.attrs["NumPart_ThisFile"] = np.array([n_gas, n_dm])
        h.attrs["NumPart_Total"] = np.array([n_gas, n_dm])
        p = f.create_group("Parameters")
        p.attrs["UnitLength_in_cm"] = MPC_IN_CM
        p.attrs["UnitMass_in_g"] = 1.989e43
        p.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        p.attrs["HubbleParam"] = 0.678
        dm = f.create_group("PartType1")
        dm["Coordinates"] = (rng.random((n_dm, 3)) * box).astype(np.float32)
        dm["Velocities"] = rng.normal(0, 100, (n_dm, 3)).astype(np.float32)
        dm["ParticleIDs"] = np.arange(n_dm, dtype=np.uint32)
        if n_gas:
            gas = f.create_group("PartType0")
            gas["Coordinates"] = (rng.random((n_gas, 3)) * box).astype(np.float32)
            gas["Velocities"] = np.zeros((n_gas, 3), np.float32)
            gas["ParticleIDs"] = np.arange(n_dm, n_dm + n_gas, dtype=np.uint32)
            gas["Masses"] = np.full(n_gas, 0.1, np.float32)


def test_cosmo_io_engine_is_found():
    """The I/O engine module (cosmo_io_engine, formerly pygadgetviewer) is importable."""
    assert sl.HAS_PYGADGETVIEWER, sl.IMPORT_ERROR


def test_masses_come_from_mass_table(tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    _write_gadget4_like(snap, n_dm=1000, n_gas=200, mass_dm=0.5)
    d = sl.load_single_snapshot_multiformat((str(snap), 0, 10_000))
    assert "error" not in d
    np.testing.assert_allclose(d["m"][d["t"] == 1], 0.5, rtol=1e-6)
    np.testing.assert_allclose(d["m"][d["t"] == 0], 0.1, rtol=1e-6)
    assert len(d["c"]) == 1200


def test_units_come_from_parameters_group(tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    _write_gadget4_like(snap)
    d = sl.load_single_snapshot_multiformat((str(snap), 0, 10_000))
    units = GadgetUnits(d["unit_source"])
    assert units.UnitLength_in_cm == pytest.approx(MPC_IN_CM)
    assert units.HubbleParam == pytest.approx(0.678)


def test_unreadable_file_reports_an_error(tmp_path):
    bad = tmp_path / "snapshot_001.hdf5"
    bad.write_bytes(b"not an hdf5 file")
    d = sl.load_single_snapshot_multiformat((str(bad), 0, 10_000))
    assert "error" in d and d["filepath"] == str(bad)


def test_loader_thread_collects_errors(tmp_path):
    good = tmp_path / "snapshot_000.hdf5"
    bad = tmp_path / "snapshot_001.hdf5"
    _write_gadget4_like(good)
    bad.write_bytes(b"broken")
    loader = sl.AsyncSnapshotLoader([good, bad])
    got = {}
    loader.finished.connect(lambda snaps, units, out: got.update(snaps=snaps, units=units))
    loader.run()  # synchronous: the signal is delivered directly
    assert len(got["snaps"]) == 1
    assert len(loader.errors) == 1 and "snapshot_001.hdf5" in loader.errors[0]


@pytest.fixture(scope="module")
def controller():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
    c.view.close()


def _load_into(controller, snap_path):
    d = sl.load_single_snapshot_multiformat((str(snap_path), 0, 150_000))
    controller.model._loader_thread = None
    controller._on_load_complete([d], GadgetUnits(d["unit_source"]), None)
    return d


def test_particles_are_drawn_without_a_selection(controller, tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    _write_gadget4_like(snap, n_dm=500)
    _load_into(controller, snap)
    for s in controller.model.tracker.sets:
        s.clear()
    controller._update_frame()
    assert controller.view.scatter_3d.pos is not None
    assert len(controller.view.scatter_3d.pos) == 500
    assert len(controller.view.scatter_3d_highlight.pos) == 0


def test_velocity_colour_with_draw_selected_only(controller, tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    _write_gadget4_like(snap, n_dm=500)
    d = _load_into(controller, snap)
    controller.model.tracker.get_set(0).select_by_ids(d["id"][:50])
    controller.view.selection_panel.draw_selected_only = True
    controller.color_mode = "velocity"
    try:
        controller._update_frame()
        assert len(controller.view.scatter_3d.pos) == 50
        assert len(controller.view.scatter_3d.color) == 50
    finally:
        controller.view.selection_panel.draw_selected_only = False
        controller.color_mode = "component"
        controller.model.tracker.get_set(0).clear()


def test_units_shown_in_loaded_info(controller, tmp_path):
    snap = tmp_path / "snapshot_000.hdf5"
    _write_gadget4_like(snap)
    _load_into(controller, snap)
    # Time 0.5 and Redshift 1: cosmological, so the convention is stated
    assert "code units: L = 1 Mpc/h (comoving), M = 10¹⁰ M☉/h" in controller.view.lbl_loaded.text()


@pytest.mark.skipif(not DM_SNAPSHOT, reason="set COSMOVIEWER_DM_SNAPSHOT to DM-L50-N128 snapshot_000.hdf5")
def test_real_dm_l50_snapshot():
    d = sl.load_single_snapshot_multiformat((DM_SNAPSHOT, 0, 150_000))
    assert "error" not in d
    # The display sample is every particle whose ID hash u is below 150,000 / 2,097,152
    # (snapshot_loader.plan_hdf5_snapshot), not every 13th row as before: the same particles
    # in every snapshot, so movies do not flicker. 150,519 particles on DM-L50.
    from lod.store import id_uniforms
    with h5py.File(DM_SNAPSHOT, "r") as f:
        all_ids = f["PartType1/ParticleIDs"][:]
    want = np.sort(all_ids[id_uniforms(all_ids, sl.LOD_SEED) < 150_000 / 2_097_152])
    np.testing.assert_array_equal(np.sort(d["id"]), want)
    assert abs(len(d["c"]) - 150_000) < 3 * 150_000 ** 0.5        # a binomial draw around the budget
    np.testing.assert_allclose(d["m"], 0.5093952007948054, rtol=1e-6)
    units = GadgetUnits(d["unit_source"])
    assert units.UnitLength_in_cm == pytest.approx(MPC_IN_CM)
    assert units.HubbleParam == pytest.approx(0.678)
