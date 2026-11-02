"""
Fixes from the UI audit of 2026-10-28:

* The view menu's section headers (VISIBILITY, INTERPOLATION) have a style sheet that parses;
  a doubled closing brace made Qt drop it with a warning.
* The Line view on a log Y axis says how many values ≤ 0 it leaves out (a velocity component
  lost half its points with no word about it).
* Line after Histogram does not keep the Histogram's x range (every point was off screen).
* Log axes carry no SI prefix ("g (x1e+27)" under ticks of 1e12-1e16).
* A field with a CGS factor of 1 and no named unit is titled "(as stored)".
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_view_menu_style_sheets_parse(app):
    from PyQt6.QtCore import qInstallMessageHandler
    from PyQt6.QtWidgets import QLabel
    from plot_panel import ViewToolbar
    seen = []
    previous = qInstallMessageHandler(lambda mode, ctx, msg: seen.append(msg))
    try:
        tb = ViewToolbar()
        for lbl in tb.findChildren(QLabel):
            lbl.ensurePolished()
        from PyQt6.QtWidgets import QMenu
        for menu in tb.findChildren(QMenu):
            for lbl in menu.findChildren(QLabel):
                lbl.ensurePolished()
        app.processEvents()
    finally:
        qInstallMessageHandler(previous)
    assert not [m for m in seen if "stylesheet" in m.lower()], seen


def _line_status(raw, scale_y):
    from generic_plugin import GenericPlugin
    p = GenericPlugin()
    return p._pipeline_line(raw, len(raw), len(raw), "", "linear", scale_y, "float32").status


def test_line_view_says_what_a_log_y_axis_leaves_out(app):
    raw = np.array([-3.0, -1.0, 0.0, 2.0, 5.0, np.nan, 7.0], np.float32)
    assert "leaves out 3 values ≤ 0" in _line_status(raw, "log")
    assert "leaves out" not in _line_status(raw, "linear")
    assert "leaves out" not in _line_status(np.abs(raw) + 1, "log")


def test_compare_arrays_explains_an_unreadable_file(app, tmp_path, monkeypatch):
    """Options > Compare arrays from files, given a text file of words, said
    "Calling nonzero on 0d arrays is not allowed ..."."""
    from PyQt6.QtWidgets import QFileDialog
    from plot_panel import PlotPanel
    bad = tmp_path / "bad.txt"
    bad.write_text("hello\nworld\n")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([str(bad)], "")))
    pp = PlotPanel()
    got = []
    pp.diagnostic_loaded.connect(got.append)
    pp._do_load_diagnostic()
    assert not got
    assert "bad.txt could not be read as numbers" in pp.lbl_status.text()


def _request(tab, x, y, log_x, log_y, presentation=False):
    from plugin_base import CurveSpec, PlotRequest
    r = PlotRequest(title="BH_Mass", x_label="x", y_label="y", tab=tab,
                    log_x=log_x, log_y=log_y, unit_dimension="mass",
                    curves=[CurveSpec(x=np.asarray(x, float), y=np.asarray(y, float))])
    r.is_presentation = presentation
    return r


def test_line_after_histogram_does_not_keep_the_histogram_range(app):
    """Histogram of BH masses (x = 1e38-1e43 g, log), then Line (x = particle index): the
    Line view kept x = [38, 43] and every point was off screen."""
    from plot_panel import PlotPanel
    pp = PlotPanel()
    pp.resize(800, 600)
    pp.show()
    pp.render(_request("Histogram", np.logspace(38, 43, 20), np.arange(1, 21), True, True))
    app.processEvents()
    pp.render(_request("Line", np.arange(500), np.logspace(38, 43, 500), True, True,
                       presentation=True))
    app.processEvents()
    (x0, x1), _ = pp.pw.viewRange()
    assert x0 <= 0.5 and x1 >= np.log10(499) - 0.1, (x0, x1)


def test_log_axes_carry_no_si_prefix(app):
    from plot_panel import PlotPanel
    pp = PlotPanel()
    pp.render(_request("Histogram", np.logspace(38, 43, 20), np.arange(1, 21), True, True))
    assert not pp.pw.getAxis("bottom").autoSIPrefix
    assert not pp.pw.getAxis("left").autoSIPrefix
    pp.render(_request("Line", np.arange(10), np.arange(10) * 1e27, False, False))
    assert pp.pw.getAxis("bottom").autoSIPrefix and pp.pw.getAxis("left").autoSIPrefix


def test_a_factor_one_field_is_titled_as_stored(app):
    """EAGLE's BH_CumlNumSeeds (a count, CGSConversionFactor 1) was titled
    "(CGS, unit unknown)"."""
    from generic_plugin import GenericPlugin, configure_units
    cls = type("Seeds", (GenericPlugin,), {"DISPLAY_NAME": "BH_CumlNumSeeds",
                                           "DATASET": "PartType5/BH_CumlNumSeeds"})
    configure_units(cls, {"CGSConversionFactor": 1.0, "h-scale-exponent": 0.0,
                          "aexp-scale-exponent": 0.0})
    assert cls.DISPLAY_NAME == "BH_CumlNumSeeds (as stored)"
    cls2 = type("Odd", (GenericPlugin,), {"DISPLAY_NAME": "Odd", "DATASET": "PartType0/Odd"})
    configure_units(cls2, {"CGSConversionFactor": 3.1e24})
    assert cls2.DISPLAY_NAME == "Odd (CGS, unit unknown)"
