"""
Regression tests for the submission audit of the Dataset Explorer (2026-10-27).

Each test failed on the code before the audit and passes after it:

1. SnapshotView.time_info divided an isolated run's time by HubbleParam.
   Gadget-4 writes h_scaling = 0 on every dataset when ComovingIntegrationOn = 0
   (src/io/io.cc), and its time unit is UnitLength / UnitVelocity
   (src/main/begrun.cc); the header bar and the formation-time plugin already
   used that. Only matters when an isolated run sets HubbleParam != 1.
2. io_manager.extract_array lost its stride phase after a chunk holding fewer
   particles than the pending offset (e.g. a file with 0 or 1 particles of a
   type), so a strided read of a multi-file snapshot took the wrong particles
   and no longer lined up with SnapshotView's strided IDs.
3. The histogram info bar printed scaling exponents with :.0f, so Gadget-4
   velocities (a^0.5) read "a^0", and in Comoving mode it said a^a_exp was
   applied when it was left out.
4. A dataset whose values are all equal reported the ±10 % padded bin range as
   its "Domain" and as its min/max markers.
5. energy_budget with a selection and a stored Potential multiplied the set's
   masses by the whole snapshot's Φ array (shape mismatch).

Synthetic files only; no environment variables needed.
"""
import os
import sys

import h5py
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from measurements.models import write_gadget4_snapshot  # noqa: E402


def _isolated(path, hubble=0.7, time=2.5, n=50, fields=None, mass=None):
    rng = np.random.default_rng(1)
    part = dict(pos=rng.normal(0, 1, (n, 3)), vel=rng.normal(0, 10, (n, 3)),
                mass=np.full(n, 1e-4) if mass is None else mass, ids=np.arange(1, n + 1))
    if fields:
        part["fields"] = fields
    return write_gadget4_snapshot(str(path), {1: part}, time=time, hubble=hubble)


# ── 1. isolated-run time unit ────────────────────────────────────────────────

def test_isolated_run_time_has_no_h(tmp_path):
    from measurements import SnapshotView
    from astro_units import cosmology_context
    from io_manager import VirtualSnapshotSet
    path = _isolated(tmp_path / "iso.hdf5", hubble=0.7, time=2.5)
    expected = 2.5 * 3.085678e21 / 1.0e5 / 3.15576e16          # Gyr, no h
    info = SnapshotView(path).time_info()
    assert info["Gyr"] == pytest.approx(expected, rel=1e-6)
    vs = VirtualSnapshotSet(path)
    bar = cosmology_context(vs.global_header, vs.global_params)
    assert info["Gyr"] == pytest.approx(bar["time_gyr"], rel=1e-6)   # the header bar agrees


# ── 2. strided multi-file reads ──────────────────────────────────────────────

def _multifile(folder, counts):
    """A Gadget-4-style distributed snapshot snap_000.{i}.hdf5 of PartType1 only;
    the value of each particle is its global index, its ID is index + 1."""
    total = np.zeros(6, dtype=np.uint64)
    total[1] = sum(counts)
    start = 0
    for i, n in enumerate(counts):
        this = np.zeros(6, dtype=np.uint64)
        this[1] = n
        with h5py.File(folder / f"snap_000.{i}.hdf5", "w") as f:
            hd = f.create_group("Header")
            hd.attrs.update({"NumFilesPerSnapshot": np.int32(len(counts)), "NumPart_ThisFile": this,
                             "NumPart_Total": total, "MassTable": np.array([0, 1e-4, 0, 0, 0, 0.]),
                             "Time": 1.0, "BoxSize": 0.0, "Redshift": 0.0})
            pa = f.create_group("Parameters")
            pa.attrs.update({"ComovingIntegrationOn": np.int32(0), "HubbleParam": 1.0})
            g = f.create_group("PartType1")
            idx = np.arange(start, start + n, dtype=np.float64)
            g.create_dataset("Coordinates", data=np.column_stack([idx, idx, idx]))
            g.create_dataset("Velocities", data=np.zeros((n, 3)))
            g.create_dataset("ParticleIDs", data=(idx + 1).astype(np.uint64))
        start += n
    return str(folder / "snap_000.0.hdf5")


@pytest.mark.parametrize("counts", [[10, 1, 10], [10, 0, 0, 11], [7, 2, 1, 3, 9]])
def test_strided_read_keeps_its_phase_across_small_chunks(tmp_path, counts):
    from io_manager import VirtualSnapshotSet
    vs = VirtualSnapshotSet(_multifile(tmp_path, counts))
    n = sum(counts)
    for max_p in (2, 3, 5, 6):
        stride = max(1, n // max_p)
        got = vs.extract_array("PartType1/Coordinates", max_particles=max_p).data[:, 0]
        np.testing.assert_array_equal(got, np.arange(n, dtype=float)[::stride])


def test_strided_positions_and_ids_line_up(tmp_path):
    from measurements import SnapshotView
    path = _multifile(tmp_path, [10, 1, 10])
    from measurements.snapshot_view import KPC_CM
    p = SnapshotView(path).particles([1], max_particles=5)
    # stored x is the global index (no unit attributes, so to_cgs = 1); the ID is index + 1
    index = np.round(p.pos[:, 0] * KPC_CM)
    np.testing.assert_array_equal(index, np.arange(21)[::4])
    np.testing.assert_array_equal(p.ids.astype(float), index + 1)


# ── 3. cosmology factor labels ───────────────────────────────────────────────

def _histogram(path, dataset, leaf, coordinates):
    from generic_plugin import configure_units, make_generic_leaves
    from data_utils import coerce_attrs
    with h5py.File(path, "r") as f:
        ds = f["PartType1"][dataset]
        cls = make_generic_leaves("PartType1", dataset, ds.shape, ds.dtype)[leaf]
        attrs = coerce_attrs(ds.attrs)
        configure_units(cls, attrs)
        data = {f"PartType1/{dataset}": ds[...].astype(float)}
        header, params = coerce_attrs(f["Header"].attrs), coerce_attrs(f["Parameters"].attrs)
    return cls().compute(data, {f"PartType1/{dataset}": attrs}, header, params, "Histogram",
                         "log", "log", (0, 0), {"coordinates": coordinates})


def test_velocity_label_shows_half_power_of_a(tmp_path):
    rng = np.random.default_rng(2)
    part = dict(pos=rng.uniform(0, 10, (200, 3)), vel=rng.normal(0, 100, (200, 3)), mass=None,
                ids=np.arange(1, 201))
    path = write_gadget4_snapshot(str(tmp_path / "cos.hdf5"), {1: part}, time=0.25, box=10.0,
                                  cosmological=True, hubble=0.7, omega0=0.3, omega_lambda=0.7,
                                  mass_table=[0, 1e-3, 0, 0, 0, 0])
    phys = _histogram(path, "Velocities", "Velocities [x]", "Physical")
    assert "a^0.5" in phys.info and "a^0 " not in phys.info and "a^0(" not in phys.info
    com = _histogram(path, "Coordinates", "Coordinates [x]", "Comoving")
    assert "a^1 not applied" in com.info
    assert "×a^1" not in com.info                      # it says what was multiplied
    phys = _histogram(path, "Coordinates", "Coordinates [x]", "Physical")
    assert "h^-1×a^1 applied" in phys.info


# ── 4. constant data ─────────────────────────────────────────────────────────

def test_constant_dataset_reports_its_value_not_the_padded_bins(tmp_path):
    from generic_plugin import configure_units, make_generic_leaves
    from data_utils import coerce_attrs
    path = _isolated(tmp_path / "const.hdf5", hubble=1.0)          # every mass 1e-4 code units
    with h5py.File(path, "r") as f:
        ds = f["PartType1"]["Masses"]
        cls = make_generic_leaves("PartType1", "Masses", ds.shape, ds.dtype)["Masses"]
        attrs = coerce_attrs(ds.attrs)
        configure_units(cls, attrs)
        req = cls().compute({"PartType1/Masses": ds[...]}, {"PartType1/Masses": attrs},
                            coerce_attrs(f["Header"].attrs), coerce_attrs(f["Parameters"].attrs),
                            "Histogram", "log", "log", (0, 0), {"coordinates": "Physical"})
    value = 1e-4 * 1.989e43 * req.unit_scalar
    lo, hi = (m.pos for m in req.markers if m.role in ("domain_min", "domain_max"))
    assert lo == pytest.approx(value) and hi == pytest.approx(value)
    assert f"Domain: [{value:.3e}, {value:.3e}]" in req.status
    assert np.nansum(req.curves[0].y) == 50                       # still binned, all counted


# ── 5. stored potential with a selection ─────────────────────────────────────

def test_energy_of_a_set_with_stored_potential(tmp_path):
    from measurements.energy import energy_budget
    from measurements.snapshot_view import MSUN_G
    n = 40
    phi = -np.linspace(1.0, 2.0, n)
    path = _isolated(tmp_path / "pot.hdf5", hubble=1.0, n=n, fields={"Potential": phi})
    ids = np.arange(5, 25)                                  # IDs 5..24 = rows 4..23
    res = energy_budget(path, [1], potential="stored", ids=ids)
    m_msun = 1e-4 * 1.989e43 / MSUN_G
    expected = 0.5 * m_msun * np.sum(phi[4:24] / 1.0e10)
    assert res.provenance.parameters["potential"] == "stored"
    assert res["W"].value == pytest.approx(expected, rel=1e-12)
