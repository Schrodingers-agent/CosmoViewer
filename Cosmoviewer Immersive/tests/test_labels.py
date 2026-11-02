"""
Words, units and times the researcher reads:
the units of a run in words, with the comoving and per-h convention of
cosmological runs; the time of a snapshot as a and z, or as code time; the
unit of the radius criterion; the level-of-detail status; and the renamed
tabs, 2D panel and playback button.
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_gadgetviewer_features import _FINISHED, _load, _wait, app, ctl, write_snapshot  # noqa: E402,F401
from controller.app_controller import CosmoViewerController as C, _power_text  # noqa: E402
from controller.console_commands import describe_snapshot  # noqa: E402
from controller.view_tools_controller import is_cosmological, time_text  # noqa: E402
from lod.view import ViewResult  # noqa: E402
from model.units import GadgetUnits  # noqa: E402

MPC, KPC = 3.085678e24, 3.085678e21


def _units(length_cm):
    return GadgetUnits({'UnitLength_in_cm': length_cm, 'UnitMass_in_g': 1.989e43,
                        'UnitVelocity_in_cm_per_s': 1e5})


# ----------------------------------------------------------------------------- pure functions

def test_units_in_words():
    assert (C._describe_units(_units(MPC), True)
            == "code units: L = 1 Mpc/h (comoving), M = 10¹⁰ M☉/h, V = 1 km/s")      # DM-L50-N128
    assert (C._describe_units(_units(KPC), False)
            == "code units: L = 1 kpc, M = 10¹⁰ M☉, V = 1 km/s")                     # CollidingGalaxiesSFR
    assert C._describe_units(_units(MPC)) == "code units: L = 1 Mpc, M = 10¹⁰ M☉, V = 1 km/s"   # not known
    assert C._describe_units(None) == "units unknown"
    # next to the radius (a 320 px panel): written as the 3D view's scale bar writes it
    assert C._length_unit_name(_units(MPC), True) == "cMpc/h"
    assert C._length_unit_name(_units(KPC), False) == "kpc"
    assert C._length_unit_name(_units(10 * KPC), True) == "× 10 ckpc/h"
    assert C._length_unit_name(None, True) == "code units"
    assert [_power_text(x) for x in (1e10, 4.3e10, 250.0, 9.99996e9)] == ["10¹⁰", "4.3×10¹⁰", "250", "10¹⁰"]


def test_time_of_a_snapshot():
    def snap(t, z=None, flag=None):
        hdr = {'Time': t} if z is None else {'Time': t, 'Redshift': z}
        return {'time': t, 'header': hdr, 'unit_source': {} if flag is None else {'ComovingIntegrationOn': flag}}

    assert time_text(snap(0.250338679973027, 2.9945884515638816, 1)) == "a = 0.250 (z = 2.99)"   # DM-L50 000
    assert time_text(snap(0.9999999999999997, 4.4e-16, 1)) == "a = 1.000 (z = 0.00)"           # DM-L50 002
    assert time_text(snap(1.0, -1e-16)) == "a = 1.000 (z = 0.00)"                              # never "-0.00"
    assert time_text(snap(1 / 128, 127.0)) == "a = 0.00781 (z = 127.0)"
    assert time_text(snap(0.5, flag=1)) == "a = 0.500 (z = 1.00)"                             # z from a
    assert time_text(snap(1.203125, 0.0, 0)) == "t = 1.203 (code units)"                     # CollidingGalaxies
    assert time_text(snap(0.0, 0.0, 0)) == "t = 0 (code units)"
    # Time = 1 and Redshift = 0 look like a = 1 from the header alone; the parameter decides.
    assert is_cosmological({'Time': 1.0, 'Redshift': 0.0})
    assert not is_cosmological({'Time': 1.0, 'Redshift': 0.0}, {'ComovingIntegrationOn': 0})
    assert time_text(snap(1.0, 0.0, 0)) == "t = 1 (code units)"
    s = {'filepath': '/run/snapshot_002.hdf5', 'c': np.zeros((4, 3)), 't': np.ones(4, np.uint8),
         'n_total': 8, 'step': 2, **snap(1.0, 0.0, 1)}
    assert describe_snapshot(s) == "snapshot_002.hdf5: a = 1.000 (z = 0.00); 4 of 8 particles (a uniform sample, about 1 in 2); type 1: 4"


# ----------------------------------------------------------------------------- in the window

def _non_cosmological_run(folder):
    """Like CollidingGalaxiesSFR: no comoving integration, kpc, h = 1; here at Time = 1, Redshift 0."""
    fn = Path(folder) / "snapshot_000.hdf5"
    write_snapshot(fn, n_gas=200, n_dm=300)
    with h5py.File(fn, 'r+') as f:
        f['Header'].attrs['BoxSize'] = 0.0
        p = f['Parameters'].attrs
        p['ComovingIntegrationOn'] = 0
        p['UnitLength_in_cm'] = KPC
        p['HubbleParam'] = 1.0
    return fn


def _window_words(app, ctl):
    """Tabs, 2D panel, playback, detail-level status and the load-error box, before any run."""
    from PyQt6 import sip
    from PyQt6.QtWidgets import QGroupBox
    v = ctl.view
    assert v.selection_panel.lbl_radius_unit.text() == "code units"            # before a run is loaded
    assert v.tab_widget.tabText(0) == "Explore 3D"
    assert v.tab_widget.tabText(v.tab_widget.indexOf(v.telemetry_view)).startswith("Log && console")
    v.set_telemetry_badge(2)
    assert v.tab_widget.tabText(v.tab_widget.indexOf(v.telemetry_view)) == "Log && console (2)"
    v.set_telemetry_badge(0)
    assert v.tab_widget.tabText(v.tab_widget.indexOf(v.telemetry_view)) == "Log && console"
    assert v.plot_2d.plotItem.titleLabel.text == "Z-slice (display sample)"
    assert "Z-SLICE (DISPLAY SAMPLE)" in [g.title() for g in v.findChildren(QGroupBox)]
    assert v.btn_play.text() == "Play snapshots"
    ctl._toggle_play()
    assert v.btn_play.text() == "Pause"
    ctl._toggle_play()
    assert v.btn_play.text() == "Play" and not ctl.is_playing

    def res(level, fraction, n, seconds):
        return ViewResult({'t': np.zeros(3)}, level, fraction, n, 'store+file', seconds)

    ctl.lod_enabled = True
    try:
        ctl._lod_view = (ctl.current_snap_idx, (), None, res(7, 0.088, 1_560_000, 1.674))
        assert ctl._lod_status() == " | Detail level 7: 8.8% of particles drawn near the focus (~1.56M in view, 1.7 s)"
        ctl._lod_view = (ctl.current_snap_idx, (), None, res(0, 1.0, 85_000, 0.12))
        assert ctl._lod_status() == " | Detail level 0: all particles drawn near the focus (~85k in view, 120 ms)"
        ctl._lod_view = (ctl.current_snap_idx + 1, (), None, res(0, 1.0, 85_000, 0.12))
        assert ctl._lod_status() == ""                                     # another snapshot's view
    finally:
        ctl._lod_view = None

    class Loader:
        errors = ["snapshot_009.hdf5: OSError: truncated file"]
        n_requested = 1

    ctl.model._loader_thread = Loader()
    ctl._on_load_complete([], None, None)
    box = ctl._load_error_box
    button = next(b for b in box.buttons() if b.text() == "Show in the log")
    button.click()
    app.processEvents()
    assert v.tab_widget.currentWidget() is v.telemetry_view
    if not sip.isdeleted(box):                  # the click usually closes (and deletes) it
        box.close()
    ctl.model._loader_thread = None
    v.tab_widget.setCurrentIndex(0)


def _cosmological_run(app, ctl, folder):
    folder.mkdir()
    write_snapshot(folder / "snapshot_000.hdf5", n_gas=200, n_dm=300)           # Time 1, z 0, 1 Mpc code length
    _load(app, ctl, folder)
    v = ctl.view
    units = "code units: L = 1 Mpc/h (comoving), M = 10¹⁰ M☉/h, V = 1 km/s"
    assert v.lbl_loaded.text() == f"1 snapshots loaded | {units}"
    assert v.lbl_info.text().startswith("a = 1.000 (z = 0.00) | N=500")
    assert v.selection_panel.lbl_radius_unit.text() == "cMpc/h"
    assert "comoving Mpc per h" in v.selection_panel.lbl_radius_unit.toolTip()
    assert v.selection_panel.spin_radius.value() == pytest.approx(10.0)         # default unchanged
    text = ctl.summary_text()
    assert f"Units: {units}" in text and "Cosmological: yes" in text
    assert "snapshot_000.hdf5: a = 1.000 (z = 0.00)" in text


def _isolated_run(app, ctl, folder):
    folder.mkdir()
    _non_cosmological_run(folder)
    _load(app, ctl, folder)
    v = ctl.view
    units = "code units: L = 1 kpc, M = 10¹⁰ M☉, V = 1 km/s"
    assert v.lbl_loaded.text() == f"1 snapshots loaded | {units}"
    assert v.lbl_info.text().startswith("t = 1 (code units) | N=500")
    assert v.selection_panel.lbl_radius_unit.text() == "kpc"
    text = ctl.summary_text()
    assert f"Units: {units}" in text and "Cosmological: no" in text
    assert ctl._length_unit(ctl.model.get_snapshot(0)) == (pytest.approx(1.0), False, False)   # scale bar: kpc


def _run_without_units(app, ctl, folder):
    folder.mkdir()
    fn = folder / "snapshot_000.hdf5"
    write_snapshot(fn, n_gas=200, n_dm=300)
    with h5py.File(fn, 'r+') as f:
        for k in ('UnitLength_in_cm', 'UnitMass_in_g', 'UnitVelocity_in_cm_per_s'):
            del f['Parameters'].attrs[k]
    _load(app, ctl, folder)
    assert ctl.view.lbl_loaded.text().endswith("(assumed: the files do not state their units)")
    assert ctl.view.selection_panel.lbl_radius_unit.text() == "code units"


def test_labels_in_the_window(app, ctl, tmp_path):
    """
    One window for all of it: every test window stays alive until the session ends (_FINISHED),
    and a few more of them were enough to make the full suite crash inside Qt.
    """
    _window_words(app, ctl)
    _cosmological_run(app, ctl, tmp_path / "cosmological")
    _isolated_run(app, ctl, tmp_path / "isolated")          # like CollidingGalaxiesSFR
    _run_without_units(app, ctl, tmp_path / "no_units")
