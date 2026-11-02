"""
CosmoViewerWindow, the main window of CosmoViewer Immersive, and AnalyseBar, the
strip above the hosted Analyse view.

Menus and layout follow GadgetViewer's behaviour and defaults; no Fortran was
translated. The "Explore 3D" tab holds the pyqtgraph OpenGL view (with optional
side-by-side stereo) and the 2D panel; beside the tabs sits the side panel of
controls on a splitter. The log and console tab follows, and the Analyse tab is
inserted after the 3D view when controller/analysis_host.py asks for it. The
window computes nothing about the data: menu actions leave as signals, and the
controller hands it the points, colours and text to draw. Overlays, the
orthographic and anaglyph views and sessions are in views/view_extras.py.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
    QPushButton, QSplitter, QFrame, QTabWidget, QProgressBar, QGroupBox, QScrollArea, QCheckBox,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QAction, QVector3D, QKeySequence
import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
    HAS_OPENGL = True
except (ImportError, ModuleNotFoundError):
    gl = None
    HAS_OPENGL = False

from views.layer_panel import DynamicLayerPanel
from views.selection_panel import ParticleSelectionPanel
from views.colouring_panel import ColouringPanel, LAGRANGE, TYPE_MODE
from views.telemetry_view import TelemetryView
from views.collapsible import CollapsibleSection
from views.inspector_panel import ParticleInspectorPanel
from views import soft_points, view_extras
from gadget_file_discovery import ParticleTypeInfo
from model.simulation_model import ParticleTracker

STYLESHEET_DARK = """
QMainWindow { background-color: #050505; }
QWidget { color: #cccccc; font-family: 'SF Pro', 'Helvetica Neue', sans-serif; font-size: 11px; }
QFrame { border: none; }
QMenuBar { background-color: #1a1a1a; color: #fff; border-bottom: 1px solid #333; }
QMenuBar::item:selected { background-color: #007acc; }
QMenu { background-color: #1a1a1a; border: 1px solid #333; }
QTabWidget::pane { border: 1px solid #333; background: #0a0a0a; }
QTabBar::tab { background: #1a1a1a; color: #888; padding: 8px 16px; border: 1px solid #333; }
QTabBar::tab:selected { background: #007acc; color: #fff; }
QTabBar::tab:hover { background: #333; }
QFrame#ControlPanel { background-color: #121212; border-left: 1px solid #333; }
QFrame#Header { background-color: #0d0d0d; border-bottom: 1px solid #007acc; }
QGroupBox { border: 1px solid #333; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #007acc; font-weight: bold; }
QSlider::groove:horizontal { background: #222; height: 4px; }
QSlider::handle:horizontal { background: #007acc; width: 12px; margin: -4px 0; border-radius: 6px; }
QPushButton { background-color: #222; border: 1px solid #444; padding: 5px; border-radius: 3px; color: #fff; }
QPushButton:hover { background-color: #333; border-color: #007acc; }
QPushButton:pressed { background-color: #007acc; }
QPushButton:disabled { background-color: #111; color: #555; border-color: #222; }
QLabel#Title { font-size: 14px; font-weight: bold; color: #fff; letter-spacing: 1px; }
QLabel#StatVal { font-size: 12px; color: #00e5ff; font-weight: bold; }
QProgressBar { background-color: #222; border: 1px solid #444; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #007acc; }
"""


class AnalyseBar(QFrame):
    """The strip above the Analyse view. It says in words what is analysed (which snapshot,
    which population) and anything to know before trusting the result. The controller
    (controller/analysis_host.py) fills it in."""

    WARNING = "#ffb000"     # what limits the result: an incomplete set, another snapshot than the 3D view
    PROBLEM = "#ff6060"     # something could not be done

    def __init__(self):
        super().__init__()
        self.setObjectName("AnalyseBar")
        self.setStyleSheet("QFrame#AnalyseBar { background-color: #0d0d0d; border-bottom: 1px solid #333; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 4, 10, 4)
        lines = QVBoxLayout()
        lines.setSpacing(2)
        self.lbl_main = QLabel("")
        self.lbl_main.setStyleSheet("color: #ddd;")
        self.lbl_note = QLabel("")
        self.lbl_note.setStyleSheet("color: #888;")
        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet(f"color: {self.WARNING};")
        self.lbl_problem = QLabel("")
        self.lbl_problem.setStyleSheet(f"color: {self.PROBLEM};")
        for lbl in (self.lbl_main, self.lbl_note, self.lbl_warning, self.lbl_problem):
            lbl.setWordWrap(True)
            lines.addWidget(lbl)
        h.addLayout(lines, 1)
        self.btn_show_3d = QPushButton("Show in 3D")
        self.btn_show_3d.setToolTip("Back to the 3D view, centred on the active set (switched on if it was off)")
        h.addWidget(self.btn_show_3d)
        self.btn_every = QPushButton("Use every particle")
        self.btn_every.setToolTip("Analyse every particle of the snapshot instead of the set.\n"
                                  "Analyse ▸ in the selection panel hands the set over again.")
        h.addWidget(self.btn_every)
        self.show_state("")

    def show_state(self, main: str, note: str = "", warning: str = "", problem: str = ""):
        """main: what is analysed; note: context, in grey; warning and problem: in their colours."""
        self.lbl_main.setText(main)
        for lbl, text in ((self.lbl_note, note), (self.lbl_warning, warning), (self.lbl_problem, problem)):
            lbl.setText(text)
            lbl.setVisible(bool(text))


class CosmoViewerWindow(QMainWindow):
    """
    Main Application Window (View Component).
    Main window, laid out after GadgetViewer's.
    """

    directory_open_requested = pyqtSignal()
    rescan_requested = pyqtSignal()
    color_mode_set = pyqtSignal(str)
    play_toggled = pyqtSignal()
    frame_changed = pyqtSignal(int)
    z_slider_changed = pyqtSignal()

    # Gadgetviewer menu action signals
    fof_dialog_requested = pyqtSignal()
    group_dialog_requested = pyqtSignal()
    plot_window_requested = pyqtSignal()
    movie_dialog_requested = pyqtSignal()
    aux_data_requested = pyqtSignal()
    panel2d_changed = pyqtSignal()
    pick_requested = pyqtSignal(float, float, bool)   # click in the 3D view: x, y (widget px), add to set

    PANEL2D_MODES = ["Z-slice points", "Smoothed density (3D view)", "Smoothed colour property (3D view)"]
    Z_SLICE_TITLE = "Z-slice (display sample)"     # a picture of the display sample, not an analysis
    LOG_TAB_TITLE = "Log && console"     # "&&" shows as "&"; a single "&" would mark a keyboard shortcut
    SIDE_PANEL_WIDTH = 320              # right-hand control panel: starting width (px), or what its controls need
    SIDE_PANEL_MIN_WIDTH = 300          # narrowest the handle allows, or what its controls need if more
    SIDE_PANEL_MAX_WIDTH = 640          # widest the handle allows

    def __init__(self, tracker: ParticleTracker):
        super().__init__()
        self.tracker = tracker
        self.setWindowTitle("CosmoViewer Immersive")
        self.resize(*self.start_size())
        self._build_ui()
        self._build_menu()
        self.setStyleSheet(STYLESHEET_DARK)
        self._fit_side_panel(initial=True)          # after the style sheet: it sets the controls' widths
        # Scale bar, time and labels over the view, the 2D panel toggle, orthographic and
        # anaglyph views, sessions and Jump to coordinates (views/view_extras.py).
        self.extras = view_extras.ViewExtras(self)

    @staticmethod
    def start_size() -> Tuple[int, int]:
        """1600 x 1000 px, or less on a smaller screen: at most 92 % of the screen's free width
        and 88 % of its height, as the Dataset Explorer, so the window starts on screen."""
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1600, 1000
        free = screen.availableGeometry()
        return min(1600, int(free.width() * 0.92)), min(1000, int(free.height() * 0.88))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header Bar
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(45)
        h = QHBoxLayout(header)

        lbl_title = QLabel("COSMO VIEWER")
        lbl_title.setObjectName("Title")

        self.lbl_info = QLabel("LOAD DATA TO BEGIN")
        self.lbl_info.setObjectName("StatVal")

        self.lbl_loaded = QLabel("---")
        self.lbl_loaded.setStyleSheet("color: #888; font-size: 10px;")

        # Both lines are rewritten as the app runs (the status line on every frame) and do not
        # wrap. Ignored: their text never sets the window's minimum width, which otherwise grows
        # the window past the screen. A line too long for the space loses its end (the time and
        # particle counts come first and stay); its tooltip has it whole.
        for lbl in (self.lbl_loaded, self.lbl_info):
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        h.addWidget(lbl_title)
        h.addSpacing(15)
        h.addWidget(self.lbl_loaded, 1)
        h.addWidget(self.lbl_info, 2)
        root.addWidget(header)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # Workspace: the tabs, then the 3D view's control panel, split by a handle the user drags
        ws = QSplitter(Qt.Orientation.Horizontal)
        ws.setHandleWidth(4)
        ws.setChildrenCollapsible(False)
        ws.setStyleSheet("QSplitter::handle { background-color: #1c1c1c; }"
                         " QSplitter::handle:hover { background-color: #007acc; }")
        self.workspace_split = ws

        # Tab Widget
        self.tab_widget = QTabWidget()

        # Tab 1: Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setHandleWidth(1)
        self.view_split = split

        # 3D OpenGL Scatter
        if HAS_OPENGL:
            self.view_3d = view_extras.CosmoGLView()
            self.view_3d.setBackgroundColor(0, 0, 0)
            self.view_3d.setCameraPosition(distance=100)

            self.box_item = gl.GLBoxItem()
            self.box_item.setColor((50, 50, 50, 255))
            self.view_3d.addItem(self.box_item)

            # Two point layers (views/soft_points.py): scatter_3d holds every particle when
            # colouring by type and the unlit ones when colouring by a property; the lit
            # layer is drawn after it, so lit particles are always on top. Then selections.
            self.scatter_3d = soft_points.SoftPoints(size=4, pxMode=True)
            self.scatter_3d.setDepthValue(0)
            self.view_3d.addItem(self.scatter_3d)
            self.scatter_3d_lit = soft_points.SoftPoints(size=4, pxMode=True)
            self.scatter_3d_lit.setDepthValue(1)
            self.view_3d.addItem(self.scatter_3d_lit)

            self.scatter_3d_highlight = gl.GLScatterPlotItem(size=8, pxMode=True)
            self.scatter_3d_highlight.setGLOptions('opaque')
            self.scatter_3d_highlight.setDepthValue(2)
            self.view_3d.addItem(self.scatter_3d_highlight)

            # Trails of tracked particles (pairs of points, mode='lines')
            self.trails_3d = gl.GLLinePlotItem(mode='lines', width=1.0, antialias=True)
            self.trails_3d.setGLOptions('translucent')
            self.trails_3d.setDepthValue(3)
            self.view_3d.addItem(self.trails_3d)

            # Centre and radius of the radius criterion (three axis bars, one radius each way)
            self.radius_marker = gl.GLLinePlotItem(mode='lines', width=2.0, antialias=True,
                                                   color=(1.0, 0.85, 0.2, 0.9))
            self.radius_marker.setGLOptions('translucent')
            self.radius_marker.setDepthValue(4)
            self.radius_marker.setVisible(False)
            self.view_3d.addItem(self.radius_marker)

            # A click that does not move the camera picks a particle (controller decides when).
            self._press_pos = None
            self.view_3d.installEventFilter(self)

            # Colour bar caption drawn over the view (property, range, table)
            self.lbl_colour_overlay = QLabel(self.view_3d)
            self.lbl_colour_overlay.setStyleSheet(
                "color: #ddd; background-color: rgba(0,0,0,140); padding: 3px; font-size: 10px;")
            self.lbl_colour_overlay.move(8, 8)
            self.lbl_colour_overlay.hide()

            # Side-by-side stereo (GadgetViewer: View > Side by side stereo): a second
            # view of the same data, its camera shifted sideways by the eye separation.
            self.stereo_on = False
            self.eye_separation = 0.05          # fraction of the camera distance (GadgetViewer default)
            self.view_3d_right = None
            self._last_data = {}
            view_box = QWidget()
            vb = QVBoxLayout(view_box)
            vb.setContentsMargins(0, 0, 0, 0)
            vb.setSpacing(0)
            self.eyes_layout = QHBoxLayout()
            self.eyes_layout.setSpacing(2)
            self.eyes_layout.addWidget(self.view_3d)
            vb.addLayout(self.eyes_layout, 1)
            self.stereo_bar = QWidget()
            sbl = QHBoxLayout(self.stereo_bar)
            sbl.setContentsMargins(6, 2, 6, 2)
            sbl.addWidget(QLabel("Eye separation"))
            self.slide_eye = QSlider(Qt.Orientation.Horizontal)
            self.slide_eye.setRange(5, 100)             # 0.005 .. 0.1, as GadgetViewer
            self.slide_eye.setValue(int(self.eye_separation * 1000))
            self.slide_eye.valueChanged.connect(self._on_eye_separation)
            sbl.addWidget(self.slide_eye)
            self.stereo_bar.hide()
            vb.addWidget(self.stereo_bar)
            self._stereo_timer = QTimer(self)
            self._stereo_timer.setInterval(33)
            self._stereo_timer.timeout.connect(self._sync_stereo_camera)
            view_box.setMinimumHeight(120)
            split.addWidget(view_box)
        else:
            fallback_lbl = QLabel("3D View (PyOpenGL required for 3D Viewport)")
            fallback_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            split.addWidget(fallback_lbl)

        # 2D panel: z-slice of the sample, or a smoothed (SPH-type) image of the 3D view
        panel2d = QWidget()
        p2 = QVBoxLayout(panel2d)
        p2.setContentsMargins(0, 0, 0, 0)
        p2.setSpacing(0)
        bar2d = QHBoxLayout()
        bar2d.setContentsMargins(6, 2, 6, 2)
        bar2d.addWidget(QLabel("2D panel:"))
        from PyQt6.QtWidgets import QComboBox
        self.combo_2d = QComboBox()
        self.combo_2d.addItems(self.PANEL2D_MODES)
        self.combo_2d.setToolTip("Smoothed images use the 3D camera's centre, orientation and field of view,\n"
                                 "the drawn particle types, and every particle of snapshots up to 2 million\n"
                                 "(bigger ones: the display sample, masses scaled by the sampling factor).\n"
                                 "Zoom or pan with the mouse to redraw that part in detail.\n"
                                 "They are pictures, not measurements.")
        self.combo_2d.currentIndexChanged.connect(lambda _: self.panel2d_changed.emit())
        bar2d.addWidget(self.combo_2d)
        self.chk_2d_follow = QCheckBox("Follow camera")
        self.chk_2d_follow.setChecked(True)
        self.chk_2d_follow.setToolTip("Re-render when the 3D view stops moving")
        bar2d.addWidget(self.chk_2d_follow)
        self.btn_2d_render = QPushButton("Render")
        # The macOS style lays these out by rects that differ from what the stylesheet draws, so without
        # this the button sat 9 px over the end of "Follow camera".
        for w in (self.chk_2d_follow, self.btn_2d_render):
            w.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
        self.btn_2d_render.clicked.connect(lambda: self.panel2d_changed.emit())
        bar2d.addWidget(self.btn_2d_render)
        bar2d.addStretch()
        p2.addLayout(bar2d)

        # 2D panel: a picture of the display sample (or a smoothed image), not a measurement
        self.plot_2d = pg.PlotWidget(title=self.Z_SLICE_TITLE)
        self.plot_2d.setBackground('#080808')
        self.plot_2d.showGrid(x=True, y=True, alpha=0.2)
        self.plot_2d.setAspectLocked(True)
        self.scatter_2d = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(255, 255, 255, 100))
        self.plot_2d.addItem(self.scatter_2d)
        self.image_2d = pg.ImageItem()
        self.plot_2d.addItem(self.image_2d)
        self.image_2d.hide()
        p2.addWidget(self.plot_2d, 1)
        split.addWidget(panel2d)
        self.panel2d_widget = panel2d

        # The OpenGL view asks for no height, so left alone the splitter gave it none and the
        # 2D panel filled the tab. Start with two thirds for the 3D view, one third for the 2D panel.
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setSizes([2000, 1000])
        viz_layout.addWidget(split)
        self.tab_widget.addTab(viz_widget, "Explore 3D")
        self.explore_tab = viz_widget

        # Last tab: the log and the command console
        self.telemetry_view = TelemetryView()
        self.tab_widget.addTab(self.telemetry_view, self.LOG_TAB_TITLE)

        ws.addWidget(self.tab_widget)

        # Right Control Panel (scrolls: the tracking controls make it taller than a laptop screen)
        ctrl_panel = QFrame()
        ctrl_panel.setObjectName("ControlPanel")
        c = QVBoxLayout(ctrl_panel)
        c.setSpacing(10)
        c.setContentsMargins(15, 15, 15, 15)

        self.layer_panel = DynamicLayerPanel()
        c.addWidget(self.layer_panel)

        self.colouring_panel = ColouringPanel()
        # Every start opens on Visibility Layers and Colouring, with Particle Selection and the inspector folded
        # away (a pick unfolds the inspector), whatever was open when the app was last closed: no settings keys.
        self.colouring_section = CollapsibleSection("COLOURING", self.colouring_panel)
        c.addWidget(self.colouring_section)

        self.selection_panel = ParticleSelectionPanel(self.tracker)
        self.selection_section = CollapsibleSection("PARTICLE SELECTION (6 SETS)", self.selection_panel,
                                                    expanded=False)
        c.addWidget(self.selection_section)

        self.inspector_panel = ParticleInspectorPanel()
        self.inspector_section = CollapsibleSection("PARTICLE INSPECTOR", self.inspector_panel, expanded=False)
        c.addWidget(self.inspector_section)

        slicer_group = QGroupBox("Z-SLICE (DISPLAY SAMPLE)")
        sg = QVBoxLayout(slicer_group)
        self.slide_z = QSlider(Qt.Orientation.Horizontal)
        self.slide_z.setRange(0, 100)
        self.slide_z.setValue(50)
        self.slide_z.valueChanged.connect(self.z_slider_changed)
        sg.addWidget(self.slide_z)
        c.addWidget(slicer_group)

        playback_group = QGroupBox("PLAYBACK CONTROLS")
        pg_layout = QVBoxLayout(playback_group)
        self.btn_play = QPushButton("Play snapshots")       # steps through the saved snapshots
        self.btn_play.setToolTip("Step through the loaded snapshots in order, again and again")
        self.btn_play.setFixedHeight(30)
        self.btn_play.clicked.connect(self.play_toggled)
        self.btn_play.setEnabled(False)
        pg_layout.addWidget(self.btn_play)
        self.slide_time = QSlider(Qt.Orientation.Horizontal)
        self.slide_time.valueChanged.connect(self.frame_changed)
        pg_layout.addWidget(self.slide_time)
        c.addWidget(playback_group)

        c.addStretch()
        scroll = QScrollArea()
        self.side_scroll = scroll
        self.header_frame = header
        scroll.setWidget(ctrl_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Width set by the splitter handle; the minimum keeps the controls readable
        # (_fit_side_panel raises both to what the controls need).
        self.side_panel_width = self.SIDE_PANEL_WIDTH
        self.side_panel_min_width = self.SIDE_PANEL_MIN_WIDTH
        scroll.setMinimumWidth(self.SIDE_PANEL_MIN_WIDTH)
        scroll.setMaximumWidth(self.SIDE_PANEL_MAX_WIDTH)
        ws.addWidget(scroll)
        ws.setStretchFactor(0, 1)
        ws.setStretchFactor(1, 0)
        ws.setSizes([10_000, self.SIDE_PANEL_WIDTH])
        self._side_content = ctrl_panel
        ctrl_panel.installEventFilter(self)         # its controls change: _fit_side_panel again
        root.addWidget(ws, 1)
        # The panel controls the 3D view, so it is shown only with the Explore 3D tab
        self._full_screen_on = False
        self.tab_widget.currentChanged.connect(self._sync_side_panel)
        self._sync_side_panel()

    def _side_panel_need(self) -> int:
        """Width the panel needs to show its controls whole: their minimum, the vertical
        scroll bar beside them and the frame (the panel never scrolls sideways)."""
        s = self.side_scroll
        return (s.widget().minimumSizeHint().width() + s.verticalScrollBar().sizeHint().width()
                + 2 * s.frameWidth())

    def _fit_side_panel(self, initial: bool = False):
        """Keep the right-hand panel at least as wide as its controls need (they were cut off on
        the right at 320 px), within SIDE_PANEL_MAX_WIDTH. At start it also takes that width."""
        need = min(self._side_panel_need(), self.SIDE_PANEL_MAX_WIDTH)
        self.side_panel_min_width = max(self.SIDE_PANEL_MIN_WIDTH, need)
        self.side_scroll.setMinimumWidth(self.side_panel_min_width)
        if initial:
            self.side_panel_width = max(self.SIDE_PANEL_WIDTH, need)
            self.workspace_split.setSizes([10_000, self.side_panel_width])

    def _sync_side_panel(self, *_):
        """Show the right-hand control panel only while the Explore 3D tab is shown (not in full screen)."""
        on = self.tab_widget.currentWidget() is self.explore_tab and not self._full_screen_on
        self.side_scroll.setVisible(on)

    def _build_menu(self):
        menubar = self.menuBar()

        # File Menu (Matches Gadgetviewer's File Menu)
        file_menu = menubar.addMenu('File')

        open_action = QAction('Read simulation snapshot...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.directory_open_requested)
        file_menu.addAction(open_action)

        rescan_action = QAction('Rescan directory', self)
        rescan_action.setShortcut('Ctrl+N')
        rescan_action.triggered.connect(self.rescan_requested)
        file_menu.addAction(rescan_action)

        aux_menu = file_menu.addMenu('Auxiliary data')
        aux_action = QAction('Read additional data...', self)
        aux_action.triggered.connect(self.aux_data_requested)
        aux_menu.addAction(aux_action)

        shot_action = QAction('Save screenshot...', self)
        shot_action.setShortcut('Ctrl+Shift+S')
        shot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(shot_action)
        # (view_extras adds a separator and Save / Open session below.)

        # Structures: haloes and groups, found here or read from the simulation's catalogue.
        # GadgetViewer keeps these under File > Read groups, where they are hard to find and
        # where nothing says FoF can be run here; the halo table lists the groups found.
        structures_menu = menubar.addMenu('Structures')
        structures_menu.setToolTipsVisible(True)
        self.structures_menu = structures_menu

        fof_action = QAction('Find FoF groups...', self)
        fof_action.setToolTip("Run friends-of-friends on the snapshot on screen, at full resolution")
        fof_action.triggered.connect(self.fof_dialog_requested)
        structures_menu.addAction(fof_action)

        groups_action = QAction('Read halo catalogue...', self)
        groups_action.setToolTip("Attach a Gadget-4 fof_tab / fof_subhalo_tab catalogue (one found next to "
                                 "the snapshots is attached automatically)")
        groups_action.triggered.connect(self.group_dialog_requested)
        structures_menu.addAction(groups_action)

        structures_menu.addSeparator()
        self.act_halo_table = QAction('Halo table', self, checkable=True)
        self.act_halo_table.setToolTip("The groups of the snapshot on screen; selecting one selects all its members")
        structures_menu.addAction(self.act_halo_table)

        # Options Menu (Matches Gadgetviewer's Options Menu)
        options_menu = menubar.addMenu('Options')

        plot_action = QAction('Make plot...', self)
        plot_action.setShortcut('Ctrl+P')
        plot_action.triggered.connect(self.plot_window_requested)
        options_menu.addAction(plot_action)

        movie_action = QAction('Make movie...', self)
        movie_action.setShortcut('Ctrl+Shift+M')
        movie_action.triggered.connect(self.movie_dialog_requested)
        options_menu.addAction(movie_action)

        # View Menu (Matches Gadgetviewer's View Menu)
        view_menu = menubar.addMenu('View')
        view_menu.addAction('Color: Component', lambda: self.color_mode_set.emit('component'))
        view_menu.addAction('Color: Velocity Heat', lambda: self.color_mode_set.emit('velocity'))
        view_menu.addAction('Color: FoF Group Index', lambda: self.color_mode_set.emit('fof_group'))
        view_menu.addSeparator()
        self.act_stereo = QAction('Side by side stereo', self, checkable=True)
        self.act_stereo.toggled.connect(self.set_stereo)
        view_menu.addAction(self.act_stereo)
        self.act_fullscreen = QAction('Full screen', self, checkable=True)
        self.act_fullscreen.setShortcut(QKeySequence('F11'))
        self.act_fullscreen.toggled.connect(self.set_full_screen)
        view_menu.addAction(self.act_fullscreen)

    # ------------------------------------------------------------------ stereo, full screen, screenshot

    def set_stereo(self, on: bool):
        if not HAS_OPENGL:
            return
        if on and self.view_3d_right is None:
            r = view_extras.CosmoGLView()
            r.ortho = self.view_3d.ortho
            r.setBackgroundColor(0, 0, 0)
            self.box_item_r = gl.GLBoxItem()
            self.box_item_r.setColor((50, 50, 50, 255))
            r.addItem(self.box_item_r)
            self.scatter_3d_r = soft_points.SoftPoints(size=4, pxMode=True)
            self.scatter_3d_r.setDepthValue(0)
            r.addItem(self.scatter_3d_r)
            self.scatter_3d_lit_r = soft_points.SoftPoints(size=4, pxMode=True)
            self.scatter_3d_lit_r.setDepthValue(1)
            r.addItem(self.scatter_3d_lit_r)
            self.scatter_3d_highlight_r = gl.GLScatterPlotItem(size=8, pxMode=True)
            self.scatter_3d_highlight_r.setGLOptions('opaque')
            self.scatter_3d_highlight_r.setDepthValue(2)
            r.addItem(self.scatter_3d_highlight_r)
            self.trails_3d_r = gl.GLLinePlotItem(mode='lines', width=1.0, antialias=True)
            self.trails_3d_r.setGLOptions('translucent')
            self.trails_3d_r.setDepthValue(3)
            r.addItem(self.trails_3d_r)
            self.view_3d_right = r
            self.eyes_layout.addWidget(r)
        self.stereo_on = on
        if self.view_3d_right is not None:
            self.view_3d_right.setVisible(on)
        self.stereo_bar.setVisible(on or self.view_3d.anaglyph)
        if on:
            self._replay_to_right()
            self._sync_stereo_camera(force=True)
            self._stereo_timer.start()
        else:
            self._stereo_timer.stop()

    def _on_eye_separation(self, v: int):
        self.eye_separation = v / 1000.0
        self._sync_stereo_camera(force=True)

    def stereo_right_centre(self, cam: dict) -> tuple:
        """Right-eye view centre: the left centre moved along the screen's right vector."""
        az = np.radians(cam['azimuth'])
        right = np.array([-np.sin(az), np.cos(az), 0.0])
        return tuple(np.array(cam['center']) + right * self.eye_separation * cam['distance'])

    def _sync_stereo_camera(self, force: bool = False):
        if not self.stereo_on or self.view_3d_right is None:
            return
        cam = self.camera_state()
        key = (cam['center'], cam['distance'], cam['elevation'], cam['azimuth'], cam['fov'], self.eye_separation)
        if not force and key == getattr(self, '_stereo_key', None):
            return
        self._stereo_key = key
        c = self.stereo_right_centre(cam)
        r = self.view_3d_right
        r.opts['fov'] = cam['fov']
        r.setCameraPosition(pos=QVector3D(*[float(x) for x in c]), distance=cam['distance'],
                            elevation=cam['elevation'], azimuth=cam['azimuth'])

    def _replay_to_right(self):
        if not self.stereo_on or self.view_3d_right is None:
            return
        d = self._last_data
        if 'scatter' in d:
            self.scatter_3d_r.setData(pos=d['scatter'][0], color=d['scatter'][1])
        if 'lit' in d:
            self.scatter_3d_lit_r.setData(pos=d['lit'][0], color=d['lit'][1])
        if 'drawing' in d:
            self._style_layers(self.scatter_3d_r, self.scatter_3d_lit_r, *d['drawing'])
        if 'highlight' in d:
            self.scatter_3d_highlight_r.setData(pos=d['highlight'][0], color=d['highlight'][1])
        if 'trails' in d:
            pos, col = d['trails']
            if len(pos):
                self.trails_3d_r.setData(pos=pos, color=col, mode='lines')
            self.trails_3d_r.setVisible(bool(len(pos)))
        if 'box' in d:
            size, mn = d['box']
            self.box_item_r.setSize(size[0], size[1], size[2])
            self.box_item_r.resetTransform()
            self.box_item_r.translate(mn[0], mn[1], mn[2])

    def set_full_screen(self, on: bool):
        """Full screen with only the 3D view (and both eyes in stereo); F11 or Esc leaves."""
        self._full_screen_on = on
        for w in (self.header_frame, self.menuBar()):
            w.setVisible(not on)
        self.panel2d_widget.setVisible(not on and self.extras.panel_wanted)   # as the corner toggle left it
        if self.extras.toggle is not None:
            self.extras.toggle.setVisible(not on)
        self.tab_widget.tabBar().setVisible(not on)
        if on:
            self.tab_widget.setCurrentIndex(0)
            self.showFullScreen()
        else:
            self.showNormal()
        self._sync_side_panel()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.act_fullscreen.setChecked(False)
            return
        if ev.key() == Qt.Key.Key_F11 and self.isFullScreen():
            self.act_fullscreen.setChecked(False)
            return
        super().keyPressEvent(ev)

    def grab_view_image(self):
        """The 3D view as a QImage (both eyes side by side in stereo); None without OpenGL."""
        if not HAS_OPENGL:
            return None
        left = self.extras.compose_screenshot(self.view_3d.grabFramebuffer())
        if not (self.stereo_on and self.view_3d_right is not None):
            return left
        from PyQt6.QtGui import QImage, QPainter
        right = self.view_3d_right.grabFramebuffer()
        img = QImage(left.width() + right.width(), max(left.height(), right.height()), QImage.Format.Format_RGB32)
        img.fill(0)
        p = QPainter(img)
        p.drawImage(0, 0, left)
        p.drawImage(left.width(), 0, right)
        p.end()
        return img

    def save_screenshot(self, path: str = ""):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "cosmoviewer.png", "PNG image (*.png)")
            if not path:
                return False
        img = self.grab_view_image()
        if img is None or img.isNull() or not img.save(path):
            QMessageBox.warning(self, "Save screenshot", f"Could not save the view to {path}.")
            return False
        return True

    def set_colour_bar(self, colouring: dict):
        """Caption over the 3D view naming what the colours mean."""
        if not HAS_OPENGL:
            return
        prop = (colouring or {}).get('property', TYPE_MODE)
        if prop == TYPE_MODE:
            self.lbl_colour_overlay.hide()
            return
        if prop == LAGRANGE:
            text = f"{prop}: red = x, green = y, blue = z across the box"
        elif colouring.get('random'):
            text = f"{prop}: random colour per value (dim = 0 / none)"
        else:
            lo, hi = colouring.get('range', (0, 1))
            text = f"{prop}: {lo:.4g} → {hi:.4g}{' (log)' if colouring.get('log') else ''}  [{colouring.get('table')}]"
        if colouring.get('density_brightness'):
            text += "; brightness = local density"
        self.lbl_colour_overlay.setText(text)
        self.lbl_colour_overlay.adjustSize()
        self.lbl_colour_overlay.show()

    def set_progress_visible(self, visible: bool):
        self.progress_bar.setVisible(visible)

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_status_text(self, text: str):
        self.lbl_info.setText(text)
        self.lbl_info.setToolTip(text)

    def set_loaded_info(self, text: str):
        self.lbl_loaded.setText(text)
        self.lbl_loaded.setToolTip(text)

    def set_telemetry_badge(self, n_errors: int):
        idx = self.tab_widget.indexOf(self.telemetry_view)
        t = self.LOG_TAB_TITLE
        self.tab_widget.setTabText(idx, f"{t} ({n_errors})" if n_errors else t)

    def show_telemetry_tab(self):
        self.tab_widget.setCurrentWidget(self.telemetry_view)
        self.telemetry_view.input.setFocus()

    # ------------------------------------------------------------------ Analyse view (controller/analysis_host.py)

    def add_analyse_tab(self) -> QWidget:
        """Insert the Analyse tab after the 3D view: the bar saying what is analysed, above a
        frame that takes the Dataset Explorer's interface when the tab is first shown."""
        self.explore_tab = self.tab_widget.widget(0)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.analyse_bar = AnalyseBar()
        lay.addWidget(self.analyse_bar)
        self.analyse_frame = QWidget()
        frame_lay = QVBoxLayout(self.analyse_frame)
        frame_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.analyse_frame, 1)
        self.analyse_tab = page
        i = self.tab_widget.insertTab(1, page, "Analyse")
        self.tab_widget.setTabToolTip(i, "Exact analyses (profiles, Lagrangian radii, energy, sanity report) of the "
                                         "snapshot shown in 3D, for the active set or every particle")
        return page

    def host_analysis_view(self, widget: QWidget, stylesheet: str):
        """Show the Explorer's interface in the Analyse tab with the Explorer's own style sheet,
        which it loses once it leaves its window."""
        self.analyse_frame.setStyleSheet(stylesheet)
        self.analyse_frame.layout().addWidget(widget)

    def add_analysis_menu(self):
        """The Analysis menu: the Analyse view's actions, then its options once it has started."""
        return self.menuBar().addMenu('Analysis')

    def set_play_enabled(self, enabled: bool):
        self.btn_play.setEnabled(enabled)

    def set_play_button_text(self, text: str):
        self.btn_play.setText(text)

    def set_time_slider_max(self, maximum: int):
        self.slide_time.setMaximum(maximum)

    def set_time_slider_value(self, value: int):
        self.slide_time.blockSignals(True)
        self.slide_time.setValue(value)
        self.slide_time.blockSignals(False)

    def get_z_slider_value(self) -> int:
        return self.slide_z.value()

    def set_box_geometry(self, size: np.ndarray, min_corner: np.ndarray, center: np.ndarray, max_dim: float):
        """The simulation box outline, and the camera framing it. Only for the whole box: framing a set
        or a group is frame_camera(), which leaves the outline alone."""
        if HAS_OPENGL:
            self.box_item.setSize(size[0], size[1], size[2])
            self.box_item.resetTransform()
            self.box_item.translate(min_corner[0], min_corner[1], min_corner[2])
            self._last_data['box'] = (np.array(size), np.array(min_corner))
            self._replay_to_right()
            self.frame_camera(center, max_dim)

    def frame_camera(self, center: np.ndarray, max_dim: float):
        """Centre the camera on ``center`` at a distance that shows an object ``max_dim`` across."""
        if HAS_OPENGL:
            self.view_3d.setCameraPosition(
                pos=QVector3D(float(center[0]), float(center[1]), float(center[2])),
                distance=max_dim * 1.5,
            )

    def camera_state(self) -> dict:
        """Current 3D camera as plain numbers (center, distance, elevation, azimuth, fov)."""
        if not HAS_OPENGL:
            return {}
        o = self.view_3d.opts
        c = o['center']
        return {'center': (c.x(), c.y(), c.z()), 'distance': float(o['distance']),
                'elevation': float(o['elevation']), 'azimuth': float(o['azimuth']), 'fov': float(o['fov'])}

    def set_camera_state(self, state: dict):
        """Put the camera at a stored view (a movie keyframe): the inverse of camera_state()."""
        if HAS_OPENGL and state:
            c = state['center']
            self.view_3d.opts['fov'] = float(state.get('fov', self.view_3d.opts['fov']))
            self.view_3d.setCameraPosition(pos=QVector3D(float(c[0]), float(c[1]), float(c[2])),
                                           distance=float(state['distance']),
                                           elevation=float(state['elevation']), azimuth=float(state['azimuth']))

    def update_radius_marker(self, centre, radius: float, show: bool):
        """Show (or hide) the radius criterion's centre as three bars of length 2 * radius."""
        if not HAS_OPENGL:
            return
        if show:
            c = np.asarray(centre, dtype=np.float64)
            pts = np.repeat(c[None, :], 6, axis=0)
            for k in range(3):
                pts[2 * k, k] -= radius
                pts[2 * k + 1, k] += radius
            self.radius_marker.setData(pos=pts)
        self.radius_marker.setVisible(bool(show))

    def set_camera_centre(self, center):
        """Move the view centre, keeping distance and angles (GadgetViewer's recentre)."""
        if HAS_OPENGL:
            self.view_3d.setCameraPosition(pos=QVector3D(float(center[0]), float(center[1]), float(center[2])))

    def update_trails(self, pos: np.ndarray, colors: np.ndarray):
        if HAS_OPENGL:
            self._last_data['trails'] = (pos, colors)
            self._replay_to_right()
            if len(pos):
                self.trails_3d.setData(pos=pos, color=colors, mode='lines')
                self.trails_3d.setVisible(True)
            else:
                self.trails_3d.setVisible(False)

    def update_scatter_3d(self, pos: np.ndarray, colors: np.ndarray):
        if HAS_OPENGL:
            self.scatter_3d.setData(pos=pos, color=colors)
            self._last_data['scatter'] = (pos, colors)
            self._replay_to_right()

    def update_scatter_lit(self, pos: np.ndarray, colors: np.ndarray):
        if HAS_OPENGL:
            self.scatter_3d_lit.setData(pos=pos, color=colors)
            self._last_data['lit'] = (pos, colors)
            self._replay_to_right()

    def apply_drawing(self, drawing: dict, by_property: bool):
        """Size, blur and fade of the two layers; the unlit layer only exists when colouring by a property."""
        if not HAS_OPENGL or not drawing:
            return
        self._last_data['drawing'] = (drawing, by_property)
        self._style_layers(self.scatter_3d, self.scatter_3d_lit, drawing, by_property)
        if self.stereo_on and self.view_3d_right is not None:
            self._style_layers(self.scatter_3d_r, self.scatter_3d_lit_r, drawing, by_property)

    @staticmethod
    def _style_layers(base, lit, drawing: dict, by_property: bool):
        st = drawing['unlit' if by_property else 'lit']
        base.set_style(size=st['size'], blur=st['blur'], fade=st['fade'], mode=soft_points.GLOW)
        base.setVisible(drawing.get('show_unlit', True) or not by_property)
        st = drawing['lit']
        lit.set_style(size=st['size'], blur=st['blur'], fade=st['fade'], mode=drawing.get('blend', soft_points.GLOW))

    # ------------------------------------------------------------------ picking

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.LayoutRequest and obj is getattr(self, "_side_content", None):
            self._fit_side_panel()
        if HAS_OPENGL and obj is self.view_3d:
            t = ev.type()
            if t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._press_pos = ev.position()
            elif t == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                p0, self._press_pos = self._press_pos, None
                if p0 is not None and (ev.position() - p0).manhattanLength() <= 3:
                    shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                    self.pick_requested.emit(float(ev.position().x()), float(ev.position().y()), shift)
        return super().eventFilter(obj, ev)

    def screen_coords(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Widget pixel coordinates (x right, y down) of 3D points as the left/main view draws
        them, their clip-space depth, and whether they are in front of the camera.
        """
        v = self.view_3d
        w, h = max(v.width(), 1), max(v.height(), 1)
        try:
            proj = v.projectionMatrix((0, 0, w, h), (0, 0, w, h))
        except TypeError:                       # older pyqtgraph: projectionMatrix(region=None)
            proj = v.projectionMatrix()
        m = proj * v.viewMatrix()
        mvp = np.array(m.data(), dtype=np.float64).reshape(4, 4).T     # Qt stores column-major
        # float32 row by row: a few tens of ms for millions of points (positions are float32 already)
        p = np.asarray(pos, dtype=np.float32)
        row = lambda i: p @ mvp[i, :3].astype(np.float32) + np.float32(mvp[i, 3])
        wc = row(3)
        front = wc > 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (row(0) / wc + 1.0) * 0.5 * w
            y = (1.0 - row(1) / wc) * 0.5 * h
            z = row(2) / wc
        return x, y, z, front

    def update_scatter_highlight(self, pos: np.ndarray, colors: np.ndarray):
        if HAS_OPENGL:
            self.scatter_3d_highlight.setData(pos=pos, color=colors)
            self._last_data['highlight'] = (pos, colors)
            self._replay_to_right()

    def clear_scatter_highlight(self):
        if HAS_OPENGL:
            self.update_scatter_highlight(np.empty((0, 3)), np.empty((0, 4)))

    def clear_scatter_3d(self):
        if HAS_OPENGL:
            self.update_scatter_3d(np.empty((0, 3)), np.empty((0, 4)))
            self.update_scatter_lit(np.empty((0, 3)), np.empty((0, 4)))
            self.update_scatter_highlight(np.empty((0, 3)), np.empty((0, 4)))

    def panel2d_mode(self) -> int:
        return self.combo_2d.currentIndex()

    def update_scatter_2d(self, x: np.ndarray, y: np.ndarray, brushes: list):
        if self.image_2d.isVisible():
            self.plot_2d.enableAutoRange()      # the smoothed image had fixed the range
        self.image_2d.hide()
        self.plot_2d.setTitle(self.Z_SLICE_TITLE)
        self.scatter_2d.setData(x, y, brush=brushes)

    def show_image_2d(self, rgba: np.ndarray, width: float, title: str, rect=None, reset_range: bool = True):
        """
        Smoothed image (rows = screen up, columns = screen right), centred on 0, side
        ``width``; or a zoom into it covering ``rect`` = (x, y, width, height).
        ``reset_range`` shows the whole image.
        """
        from PyQt6.QtCore import QRectF
        self.scatter_2d.setData([])
        self.image_2d.setImage(np.transpose((rgba * 255).astype(np.uint8), (1, 0, 2)), levels=(0, 255))
        r = QRectF(-width / 2, -width / 2, width, width) if rect is None else QRectF(*rect)
        self.image_2d.setRect(r)
        self.image_2d.show()
        if reset_range:
            self.plot_2d.getViewBox().setRange(r, padding=0)
        self.plot_2d.setTitle(title)

    def clear_scatter_2d(self):
        self.scatter_2d.setData([])

    def rebuild_layer_panel(self, particle_types: List[ParticleTypeInfo]):
        self.layer_panel.rebuild_for_snapshot(particle_types)

    def get_visible_types(self, existing_types: List[int]) -> List[int]:
        return self.layer_panel.get_visible_types(existing_types)

    def clear_layer_panel(self):
        self.layer_panel._clear_layout()
