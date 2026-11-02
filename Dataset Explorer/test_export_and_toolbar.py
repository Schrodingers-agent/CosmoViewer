"""
Export and toolbar regressions found by the GUI walkthrough of 2026-10-26.

    python3 -m pytest -q test_export_and_toolbar.py

B3  After an analysis failed, Export wrote the previous analysis's data under
    the new name. A failed view (and a picture with no curves) now exports
    nothing and says why.
B4  Analysis exports had only the y values of the first curve (Lagrangian
    radii without the radius). Every curve is now exported with its x values,
    checked against an independent recomputation from the snapshot.
B5  The previous analysis's view-tab buttons stayed drawn over the new ones
    until the event loop deleted them.
B2  The toolbar forced the window wider than a laptop screen; it now wraps
    onto further rows, keeping every control visible and none overlapping.
"""

import csv
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

import analysis_plugins  # noqa: E402
import cli_import  # noqa: E402
from measurements import SnapshotView, find_centre, lagrangian_radii  # noqa: E402
from measurements.centre import minimum_image  # noqa: E402
from measurements.models import plummer, write_gadget4_snapshot  # noqa: E402
from measurements.snapshot_view import MSUN_G  # noqa: E402
from plugin_base import BasePlugin, CurveSpec, ImageSpec, PlotRequest  # noqa: E402


def _wait(app, win, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        w = win._active_worker
        if w is None or not w.isRunning():
            app.processEvents()
            if win._active_worker is None or not win._active_worker.isRunning():
                return True
        time.sleep(0.02)
    return False


def _read_csv(path):
    """(comment lines, header, float rows) of a CSV written by Export."""
    lines = Path(path).read_text().splitlines()
    notes = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    header = next(csv.reader([body[0]]))
    rows = np.array([[float(v) for v in r] for r in csv.reader(body[1:])])
    return notes, header, rows


class TestAnalysisExport(unittest.TestCase):
    """Through the real window, its worker and the measurement layer."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        import density_explorer
        cls.dx = density_explorer
        cls.tmp = tempfile.TemporaryDirectory()
        p4, v4, m4 = plummer(5000, 1e5, 0.001, seed=1)
        p1, v1, _ = plummer(8000, 1e6, 0.003, seed=2)
        code_mass = 1.989e43 / MSUN_G
        cls.snap = Path(cls.tmp.name) / "snapshot_000.hdf5"
        write_gadget4_snapshot(cls.snap, {
            4: dict(pos=p4 + 2.0, vel=v4, mass=m4 / code_mass, ids=np.arange(5000)),
            1: dict(pos=p1 + 2.0, vel=v1, mass=None, ids=np.arange(5000, 13000))},
            mass_table=[0, 1e6 / 8000 / code_mass, 0, 0, 0, 0])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _window(self):
        win = self.dx.CosmoExplorer()
        win.show()
        self.addCleanup(win.close)
        return win

    def test_lagrangian_export_has_the_radius_and_matches_the_snapshot(self):
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap)]),
                                   "Analyses/Lagr. radii, PartType4", "Profile")
        self.assertTrue(_wait(self.app, win))
        out = Path(self.tmp.name) / "lagr.csv"
        win.plot_panel.export_to(str(out))
        notes, header, rows = _read_csv(out)
        self.assertEqual(header[0], "r [kpc  physical]")
        self.assertEqual(header[1], "enclosed mass fraction")
        self.assertTrue(any("Lagrangian radii" in n for n in notes))
        # Independent: sorted distances from the same centre, and the cumulative mass fraction
        p = SnapshotView(str(self.snap)).particles([4], velocities=False, ids=False)
        c = find_centre(p, analysis_plugins.DEFAULT_CENTRE)
        d = minimum_image(p.pos - c.pos, p.box)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        order = np.argsort(r, kind="stable")
        f = np.cumsum(p.mass[order]) / p.mass.sum()
        pick = np.unique(np.linspace(0, len(r) - 1, analysis_plugins.PROFILE_POINTS).astype(int))
        np.testing.assert_allclose(rows[:, 0], r[order][pick], rtol=1e-9)
        np.testing.assert_allclose(rows[:, 1], f[pick], rtol=1e-9)
        # and the half-mass radius read off the exported curve is the analysis's own
        r50 = float(lagrangian_radii(p, c, [0.5])["radius"].value[0])
        self.assertLess(abs(np.interp(0.5, rows[:, 1], rows[:, 0]) - r50) / r50, 0.01)

    def test_failed_analysis_exports_nothing(self):
        win = self._window()
        win.open_from_command_line(cli_import.resolve_sources([str(self.snap)]),
                                   "Analyses/Lagr. radii, PartType4", "Profile")
        self.assertTrue(_wait(self.app, win))
        good = Path(self.tmp.name) / "good.csv"
        win.plot_panel.export_to(str(good))
        self.assertTrue(good.exists())
        # The next analysis fails (refused as too large for memory), as a set of other types does
        old = analysis_plugins.MAX_ANALYSIS_PARTICLES
        analysis_plugins.MAX_ANALYSIS_PARTICLES = 10
        self.addCleanup(setattr, analysis_plugins, "MAX_ANALYSIS_PARTICLES", old)
        self.assertTrue(win.sidebar.select_leaf("Analyses", "Lagr. radii, PartType1"))
        self.assertTrue(_wait(self.app, win))
        self.assertIn("Error", win.plot_panel.lbl_status.text())
        for name in ("bad.csv", "bad.py"):
            bad = Path(self.tmp.name) / name
            win.plot_panel.export_to(str(bad))
            self.assertFalse(bad.exists(), f"{name} written after a failed analysis")
        self.assertIn("Nothing to export", win.plot_panel.lbl_status.text())


class TestPlotPanel(unittest.TestCase):
    """The panel alone, fed the requests the plugins produce."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from plot_panel import PlotPanel
        self.panel = PlotPanel()
        self.addCleanup(self.panel.deleteLater)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _curve_request(self):
        """A dispersion-like view: a curve with its error band, and one on other x values."""
        r_mid = np.array([0.1, 0.2, 0.4, 0.8])
        r_hi = np.array([0.15, 0.3, 0.6, 1.2, 2.4])
        s, e = np.array([10.0, 9.0, 8.0, 7.0]), np.array([1.0, 0.5, 0.5, 1.0])
        return PlotRequest(
            title="Radial profile, PartType1", x_label="r [kpc, physical]",
            y_label="velocity [km/s]", tab="Dispersion", status="All 8,000 particles",
            log_x=False, log_y=False,
            curves=[CurveSpec(x=r_mid, y=s, label="σ_r"),
                    CurveSpec(x=r_mid, y=s + e, dash=True),
                    CurveSpec(x=r_mid, y=s - e, dash=True),
                    CurveSpec(x=r_hi, y=2 * r_hi, label="v_circ", dash=True)])

    def test_every_curve_is_exported_with_its_x(self):
        req = self._curve_request()
        self.panel.render(req)
        out = Path(self.tmp.name) / "disp.csv"
        self.panel.export_to(str(out))
        notes, header, rows = _read_csv(out)
        self.assertEqual(header, ["r [kpc  physical]", "σ_r", "σ_r + err", "σ_r - err",
                                  "r [kpc  physical] for v_circ", "v_circ"])
        np.testing.assert_allclose(rows[:4, 0], req.curves[0].x)
        np.testing.assert_allclose(rows[:4, 1], req.curves[0].y)
        np.testing.assert_allclose(rows[:4, 2], req.curves[1].y)
        np.testing.assert_allclose(rows[:4, 3], req.curves[2].y)
        self.assertTrue(np.isnan(rows[4, :4]).all())          # padding, never invented values
        np.testing.assert_allclose(rows[:, 4], req.curves[3].x)
        np.testing.assert_allclose(rows[:, 5], req.curves[3].y)
        self.assertTrue(any("All 8,000 particles" in n for n in notes))

    def test_error_and_picture_views_clear_the_previous_export(self):
        self.panel.render(self._curve_request())
        for req, says in ((BasePlugin.error_request("None of the particles are of these types"),
                           "failed"),
                          (PlotRequest(title="Phase diagram, gas", x_label="", y_label="",
                                       tab="Phase diagram",
                                       images=[ImageSpec(data=np.ones((4, 4)))]),
                           "Matrix tab")):
            self.panel.render(self._curve_request())
            self.panel.render(req)
            out = Path(self.tmp.name) / f"after_{req.tab or 'error'}.csv"
            self.panel.export_to(str(out))
            self.assertFalse(out.exists())
            self.assertIn(says, self.panel.lbl_status.text())

    def test_old_view_tab_buttons_go_at_once(self):
        tb = self.panel._toolbar
        self.panel.show()
        tb.set_view_tabs(["Matrix", "Density", "Enclosed mass", "Dispersion", "Anisotropy"])
        old = list(tb._view_btns.values())
        tb.set_view_tabs(["Matrix", "Report"])
        # no return to the event loop: deferred deletion has not run yet
        self.assertTrue(all(not b.isVisible() for b in old))
        self.app.processEvents()
        shown = [b for b in tb.findChildren(QPushButton)
                 if b.objectName() == "tab_btn" and b.isVisible()]
        self.assertEqual(sorted(b.text() for b in shown), ["Matrix", "Report"])

    def test_toolbar_wraps_instead_of_widening_the_window(self):
        tb = self.panel._toolbar
        tb.set_view_tabs(["Matrix", "Density", "Enclosed mass", "Dispersion", "Anisotropy"],
                         default="Density")
        one_row = tb.sizeHint().width()
        self.assertLess(tb.minimumSizeHint().width(), one_row * 0.75)

        def visible_controls():
            return [w for w in tb.findChildren(QPushButton) + tb.findChildren(type(tb._combo_x))
                    if w.isVisible()]

        for width, rows in ((one_row + 40, 1), (int(one_row * 0.6), 2)):
            self.panel.resize(width, 600)
            self.panel.show()
            for _ in range(3):
                self.app.processEvents()
            self.assertEqual(tb.height(), 44 * rows, f"width {width}")
            rects = []
            for w in visible_controls():
                top_left = w.mapTo(tb, w.rect().topLeft())
                r = w.rect().translated(top_left)
                self.assertTrue(tb.rect().contains(r), f"{w.text() if hasattr(w, 'text') else w} "
                                                       f"outside the toolbar at width {width}")
                rects.append(r)
            for i in range(len(rects)):
                for j in range(i + 1, len(rects)):
                    self.assertFalse(rects[i].intersects(rects[j]), "controls overlap")
            # Export stays at the right-hand end of its row
            exp = tb._btn_export
            self.assertGreater(exp.mapTo(tb, exp.rect().topLeft()).x(), tb.width() // 2)


if __name__ == "__main__":
    unittest.main()
