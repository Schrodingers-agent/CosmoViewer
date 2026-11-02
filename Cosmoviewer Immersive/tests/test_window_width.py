"""
The window keeps its width (views/main_window.py).

The window used to grow wider than the screen, leaving controls out of reach: the header's
status line (rewritten on every frame) made the window at least as wide as its text, and Qt
never narrows a window again. Checks that it does not now, that the status line keeps its
whole text (in its tooltip when it is cut), and that the window starts within the screen.
(The Analyse view's toolbar still sets the window's width: a scrolling strip was tried and
withdrawn on 2026-10-26.) Synthetic Plummer snapshots, headless.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_host import app, host, _settle, _wait  # noqa: E402,F401  (fixtures shared with the Analyse view's tests)

LONG_STATUS = ("a = 1.000 (z = 0.00) | N=2,097,152 (lit 12,345) | selected 5,000 | "
               "fof_subhalo_tab_002.hdf5: 12,345 groups, 9,876 subhaloes | reading Density 45% | "
               "Detail level 7: 8.8% of particles drawn near the focus (~1.56M in view, 1.7 s)")
LONG_LOADED = ("3 snapshots loaded | code units: L = 1 Mpc/h (comoving), M = 10¹⁰ M☉/h, V = 1 km/s "
               "(assumed: the files do not state their units)")


def _shown(app, c, w=1000, h=800):
    v = c.view
    v.resize(w, h)
    v.show()
    for _ in range(4):
        app.processEvents()
    return v


def test_long_header_lines_do_not_widen_the_window(app, host):
    # The status line is the app's to rewrite: the level-of-detail build's progress ("Indexing
    # ... 1/1") and the next frame's time line replaced the long text before it was checked,
    # whenever they arrived during the test. Let the build finish and stop its poll first.
    assert _wait(app, lambda: host._lod_build is None or host._lod_build.isFinished(), 60)
    host._lod_timer.stop()
    v = _shown(app, host)
    min_before = v.minimumSizeHint().width()
    v.set_status_text(LONG_STATUS)
    v.set_loaded_info(LONG_LOADED)
    for _ in range(4):
        app.processEvents()
    assert v.lbl_info.fontMetrics().horizontalAdvance(LONG_STATUS) > 1000   # longer than the window is wide
    assert v.width() == 1000
    assert v.minimumSizeHint().width() <= min_before
    assert v.lbl_info.text() == LONG_STATUS and v.lbl_info.toolTip() == LONG_STATUS
    assert v.lbl_loaded.text() == LONG_LOADED and v.lbl_loaded.toolTip() == LONG_LOADED


def test_window_starts_within_the_screen(app, host):
    free = app.primaryScreen().availableGeometry()
    w, h = host.view.start_size()
    assert w <= 1600 and h <= 1000
    assert w <= free.width() and h <= free.height()


def test_3d_view_has_room_at_start(app, host):
    """Walkthrough bug B1: the 3D view started 0 px tall (splitter sizes [0, 660]), the 2D
    panel filling the Explore 3D tab. It now starts with about two thirds of the height."""
    v = _shown(app, host, 1400, 900)
    top, bottom = v.view_split.sizes()
    assert top > 0 and bottom > 0
    assert abs(top / (top + bottom) - 2 / 3) < 0.05
    assert v.view_split.widget(0).height() == top


def test_analyse_view_fits_a_laptop_screen(app, host):
    """Walkthrough bugs B2 and B6: visiting analyses raised the window's minimum width to
    1672 px on a 1470 px screen, the window was squeezed below it and the main tab labels
    were cut off. The Analyse toolbar now wraps onto a second row instead. Every analysis
    view here keeps the window's minimum width well under a 13-inch MacBook's 1470 px, and
    at 1000 px wide every control of the toolbar and every main tab label is shown whole."""
    v = _shown(app, host, 1000, 800)
    v.tab_widget.setCurrentWidget(v.analyse_tab)
    assert _settle(app, host)
    ex = host._explorer
    tb = ex.plot_panel._toolbar
    from PyQt6.QtWidgets import QPushButton
    for leaf in ("Lagr. radii, PartType1", "Energy, all types", "Profile, PartType1"):
        assert ex.sidebar.select_leaf("Analyses", leaf) and _settle(app, host)
        for name in list(tb._view_btns):
            tb._view_btns[name].click()
            assert _settle(app, host)
            for _ in range(3):
                app.processEvents()
            assert v.minimumSizeHint().width() <= 1100, (leaf, name, v.minimumSizeHint())
            assert v.width() == 1000
            for b in tb.findChildren(QPushButton):
                if b.isVisible():
                    r = b.rect().translated(b.mapTo(tb, b.rect().topLeft()))
                    assert tb.rect().contains(r), (leaf, name, b.text())
    bar = v.tab_widget.tabBar()
    for i in range(bar.count()):
        text = bar.tabText(i).replace("&&", "&")
        assert bar.tabRect(i).width() >= bar.fontMetrics().horizontalAdvance(text), text
        assert bar.tabRect(i).height() >= bar.fontMetrics().height()
