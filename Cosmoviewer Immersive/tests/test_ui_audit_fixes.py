"""
Fixes from the UI audit of 2026-10-28, each checked the way a user meets it:

* Every style sheet of the Analyse tab parses (walkthrough bug B6: one did not, and the
  main tab labels were clipped on the Analyse tab).
* The colouring note names the property now chosen, also for integer properties and with
  random colours (it went on naming the previous property).
* The inspector does not subtract group numbers: a group number is a label, so the change
  column says what it was, and the CSV has no d_ column for it.
Synthetic snapshots, headless.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_host import app, host, _settle, _wait  # noqa: E402,F401
from test_inspector import _write, ctl, _load, _all_read, _row  # noqa: E402,F401


def test_analyse_tab_style_sheets_all_parse(app, host):
    """Every style sheet the Analyse tab brings in parses. One did not (a doubled closing brace
    in the Explorer's view-menu headers, Dataset Explorer/plot_panel.py); Qt's warning about it,
    shown in the app's console, left the main tab bar unpolished and its labels clipped."""
    from PyQt6.QtCore import qInstallMessageHandler
    seen = []
    previous = qInstallMessageHandler(lambda mode, ctx, msg: seen.append(msg))
    try:
        v = host.view
        v.show()
        names = [v.tab_widget.tabText(i) for i in range(v.tab_widget.count())]
        v.tab_widget.setCurrentIndex(names.index("Analyse"))
        assert _settle(app, host, 120)
        _wait(app, lambda: False, 0.5)
    finally:
        qInstallMessageHandler(previous)
    assert host._explorer is not None
    assert not [m for m in seen if "stylesheet" in m.lower()], seen


def _group_pair(tmp_path):
    n = 20
    ids = np.arange(1, n + 1)
    pos = np.random.default_rng(1).random((n, 3)) * 10
    _write(tmp_path / "snapshot_000.hdf5", {1: dict(ids=ids, pos=pos, GroupNumber=np.full(n, 5, np.int32),
                                                    SubGroupNumber=np.zeros(n, np.int32))}, time_=0.5)
    _write(tmp_path / "snapshot_001.hdf5", {1: dict(ids=ids, pos=pos, GroupNumber=np.full(n, 3, np.int32),
                                                    SubGroupNumber=np.zeros(n, np.int32))}, time_=1.0)


def test_inspector_treats_group_numbers_as_labels(app, ctl, tmp_path):
    _group_pair(tmp_path)
    c = ctl
    _load(app, c, tmp_path)
    c.current_snap_idx = 1
    c._update_frame()
    c.inspect_particle(4)
    assert _wait(app, lambda: _all_read(c), 60)
    c._refresh_inspector()
    g = _row(c, "GroupNumber")
    assert g is not None and g[1] == "3" and g[2] == "was 5" and g[3] == "–"
    s = _row(c, "SubGroupNumber")
    assert s[2] == "same" and s[3] == "–"
    out = c._export_inspected(str(tmp_path / "p4.csv"))
    head = list(csv.reader(open(out)))[1]
    assert "GroupNumber" in head and "d_GroupNumber" not in head and "d_SubGroupNumber" not in head
    assert "d_Coordinates x" in head


def test_colour_note_names_the_property_now_chosen(app, ctl, tmp_path):
    _group_pair(tmp_path)
    c = ctl
    _load(app, c, tmp_path)
    cp = c.view.colouring_panel
    names = [cp.combo_prop.itemText(i) for i in range(cp.combo_prop.count())]
    assert "GroupNumber" in names and "Coordinates x" in names
    cp.combo_prop.setCurrentText("Coordinates x")
    assert _wait(app, lambda: cp.lbl_info.text().startswith("Coordinates x"), 30), cp.lbl_info.text()
    cp.combo_prop.setCurrentText("GroupNumber")          # integer: random colours, no range
    assert _wait(app, lambda: cp.lbl_info.text().startswith("GroupNumber"), 30), cp.lbl_info.text()


def test_bad_input_is_explained_in_plain_words(app):
    """Typed IDs, an ID file and Jump to coordinates said "invalid literal for int() with
    base 10" or "could not convert string to float"."""
    import pytest
    from model.tracking import parse_id_text
    from views.view_extras import JumpDialog
    for text in ("abc", "12 x7", "5-z"):
        with pytest.raises(ValueError, match="is not a particle ID"):
            parse_id_text(text)
    dlg = JumpDialog(None, (1.0, 2.0, 3.0), 4.0)
    dlg.edits[0].setText("abc")
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        dlg.values()


def test_eagle_folders_are_not_called_unfamiliar(tmp_path):
    """Opening an EAGLE snapshot folder warned that it 'may not be a Gadget output directory'."""
    from gadget_file_discovery import FileDiscoveryEngine
    d = FileDiscoveryEngine()
    snapdir = tmp_path / "snapshot_028_z000p000"
    snapdir.mkdir()
    (snapdir / "snap_028_z000p000.0.hdf5").touch()
    assert d._is_gadget_directory(snapdir)          # the folder of one multi-file snapshot
    assert d._is_gadget_directory(tmp_path)         # the run folder holding it
    other = tmp_path / "notes"
    other.mkdir()
    (other / "readme.txt").touch()
    assert not d._is_gadget_directory(other)


def test_cancelling_a_movie_is_not_reported_as_a_failure(app, ctl, monkeypatch):
    """Cancel in the movie progress window stopped the movie, then a warning said
    "Movie making failed: Cancelled by user"."""
    from PyQt6.QtWidgets import QMessageBox, QProgressDialog
    from movie.maker import MovieResult
    boxes = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: boxes.append(a[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: boxes.append(a[2]))
    dlg = QProgressDialog("Making movie...", "Cancel", 0, 10, ctl.view)
    ctl._on_movie_done(MovieResult(False, "Cancelled by user"), dlg)
    assert boxes == []
    ctl._on_movie_done(MovieResult(False, "Unable to read any snapshots"), dlg)
    assert len(boxes) == 1 and "failed" in boxes[0]


def test_empty_scatter_plot_says_why(app, ctl, tmp_path):
    """Make plot of a gas property against a star property drew an empty plot under a note
    about the display sample, with no word on why it was empty."""
    rng = np.random.default_rng(2)
    _write(tmp_path / "snapshot_000.hdf5",
           {0: dict(ids=np.arange(1, 11), pos=rng.random((10, 3)) * 10, Masses=np.full(10, 0.1),
                    Density=rng.random(10) + 1),
            4: dict(ids=np.arange(11, 16), pos=rng.random((5, 3)) * 10, Masses=np.full(5, 0.1),
                    StellarFormationTime=rng.random(5))})
    c = ctl
    _load(app, c, tmp_path)
    c._open_plot_window()
    pw = c.plot_window
    shown = []
    pw.show_message = lambda text: shown.append(text)
    assert _wait(app, lambda: (c._render_property_plot("scatter", "Density", "StellarFormationTime",
                                                       False, False) or True)
                 and shown and not shown[-1].startswith("Reading"), 30), shown
    assert "No drawn particle has values of both 'Density' and 'StellarFormationTime'" in shown[-1]


def test_particle_type_5_is_called_black_holes(tmp_path):
    """The snapshot inspector named EAGLE's and Gadget-4's PartType5 "Boundary"."""
    from gadget_file_discovery import SnapshotInspector
    assert SnapshotInspector.TYPE_NAMES[5] == "Black Holes"
