"""
The right-hand control panel of the 3D view (views/main_window.py).

Checks that it is shown only with the Explore 3D tab (not beside the Analyse view, which hosts
the Dataset Explorer, nor beside the log), that its width follows the splitter handle within its
limits and is kept when the user comes back to the 3D view, and that full screen hides it and
leaving full screen brings it back only on the 3D tab. Synthetic Plummer snapshots, headless.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_host import app, host, _settle  # noqa: E402,F401  (fixtures shared with the Analyse view's tests)


def _shown(app, c, w=1400, h=900):
    v = c.view
    v.resize(w, h)
    v.show()
    app.processEvents()
    return v


def test_panel_only_with_the_3d_tab(app, host):
    c = host
    v = _shown(app, c)
    tw = v.tab_widget
    assert tw.currentWidget() is v.explore_tab and v.side_scroll.isVisible()
    tw.setCurrentWidget(v.analyse_tab)
    assert _settle(app, c)
    assert c._explorer is not None                     # the Explorer is really shown in the tab
    assert not v.side_scroll.isVisible()
    assert v.tab_widget.width() > v.width() - 10        # the Analyse view takes the panel's room
    tw.setCurrentWidget(v.telemetry_view)
    app.processEvents()
    assert not v.side_scroll.isVisible()
    tw.setCurrentWidget(v.explore_tab)
    app.processEvents()
    assert v.side_scroll.isVisible()


def test_panel_width_follows_the_handle(app, host):
    c = host
    v = _shown(app, c)
    split = v.workspace_split
    assert split.widget(1) is v.side_scroll and not split.childrenCollapsible()
    assert v.side_panel_width >= v.SIDE_PANEL_WIDTH                    # 320 px, or what the controls need
    assert abs(v.side_scroll.width() - v.side_panel_width) <= 2
    total = sum(split.sizes())
    split.setSizes([total - 500, 500])                                 # what dragging the handle does
    app.processEvents()
    assert abs(v.side_scroll.width() - 500) <= 2
    split.setSizes([total - 100, 100])                                 # not narrower than the minimum
    app.processEvents()
    assert v.side_scroll.width() == v.side_panel_min_width >= v.SIDE_PANEL_MIN_WIDTH
    split.setSizes([200, total - 200])                                 # nor wider than the maximum
    app.processEvents()
    assert v.side_scroll.width() == v.SIDE_PANEL_MAX_WIDTH
    split.setSizes([total - 450, 450])
    app.processEvents()
    v.tab_widget.setCurrentWidget(v.telemetry_view)                   # away and back keeps the width
    app.processEvents()
    v.tab_widget.setCurrentWidget(v.explore_tab)
    app.processEvents()
    assert abs(v.side_scroll.width() - 450) <= 2


def test_full_screen_hides_panel_and_restores_it_on_the_3d_tab(app, host):
    c = host
    v = _shown(app, c)
    v.set_full_screen(True)
    app.processEvents()
    assert not v.side_scroll.isVisible() and v.tab_widget.currentWidget() is v.explore_tab
    v.set_full_screen(False)
    app.processEvents()
    assert v.side_scroll.isVisible()
    v.set_full_screen(True)
    v.tab_widget.setCurrentWidget(v.telemetry_view)                   # e.g. an error shows the log
    app.processEvents()
    assert not v.side_scroll.isVisible()
    v.set_full_screen(False)
    app.processEvents()
    assert not v.side_scroll.isVisible()                              # not beside the log


def test_controls_fit_the_panel(app, host):
    """Walkthrough bug B7: the panel was 320 px but its controls needed 350, so the Colour
    button, the Log check box and Auto were cut off on the right. The panel now starts, and
    cannot be dragged, narrower than its controls need; it never scrolls sideways."""
    v = _shown(app, host)
    for _ in range(3):
        app.processEvents()
    content = v.side_scroll.widget()
    assert v.side_scroll.viewport().width() >= content.minimumSizeHint().width()
    split = v.workspace_split
    total = sum(split.sizes())
    split.setSizes([total - 100, 100])                                 # dragged as narrow as it goes
    app.processEvents()
    assert v.side_scroll.viewport().width() >= content.minimumSizeHint().width()
    assert content.width() >= content.minimumSizeHint().width()
