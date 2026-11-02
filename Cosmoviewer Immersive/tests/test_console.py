"""Telemetry Logs console: log capture, commands and background jobs."""

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from controller import console  # noqa: E402
from test_multifile_eagle import _write_eagle_like  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def ctl(app):
    console.install_log_capture()
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
    c.view.close()


def _text(c):
    return c.view.telemetry_view.log_edit.toPlainText()


def _run(c, line):
    c.view.telemetry_view.input.setText(line)
    c.view.telemetry_view.input.returnPressed.emit()


def _wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_print_and_exceptions_reach_console(app, ctl):
    # pytest swaps sys.stdout for its own capture, so drive the tee directly
    console._Tee(None, 'info').write("hello from print\npartial")
    try:
        raise ValueError("boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    app.processEvents()
    text = _text(ctl)
    assert "hello from print" in text
    assert "Unhandled ValueError: boom" in text
    assert ctl.view.telemetry_view.n_errors >= 1
    tabs = ctl.view.tab_widget                     # found by widget: an Analyse tab may sit before it
    assert tabs.tabText(tabs.indexOf(ctl.view.telemetry_view)).startswith("Log && console (")


def test_exception_in_slot_does_not_abort(app, ctl):
    def bad():
        raise RuntimeError("slot failure")
    QTimer.singleShot(0, bad)
    assert _wait(app, lambda: "slot failure" in _text(ctl), 5)


def test_help_unknown_and_history(app, ctl):
    _run(ctl, "help")
    _run(ctl, "nosuchcommand")
    _run(ctl, "history")
    app.processEvents()
    text = _text(ctl)
    assert "errors [n]" in text and "load <folder|file|glob>" in text
    assert "unknown command 'nosuchcommand'" in text
    assert "nosuchcommand" in text.split("> history")[-1]


def test_load_command_and_info(app, ctl, tmp_path):
    _write_eagle_like(tmp_path, nfiles=4)
    _run(ctl, f'load "{tmp_path}"')
    assert _wait(app, lambda: not ctl.loading and ctl.model.snapshots, 60)
    _run(ctl, "info")
    app.processEvents()
    text = _text(ctl)
    assert "1 snapshots loaded" in text
    assert "(+3 files)" in text
    assert "h-scale-exponent=-1.0" in text


def test_load_errors_go_to_console(app, ctl, tmp_path):
    (tmp_path / "snap_000.hdf5").write_bytes(b"broken")
    _run(ctl, f'load "{tmp_path}"')
    assert _wait(app, lambda: not ctl.loading, 60)
    app.processEvents()
    assert "Snapshot load failed: snap_000.hdf5" in _text(ctl)
    ctl._load_error_box.close()


def test_background_job_and_cancel(app, ctl):
    reg = ctl.console
    started = []

    def work(job):
        started.append(job.id)
        while not job.cancelled:
            time.sleep(0.01)
        return "stopped"
    job = reg.start_job("spin", work)
    assert _wait(app, lambda: started, 5)
    _run(ctl, "jobs")
    _run(ctl, f"cancel {job.id}")
    assert _wait(app, lambda: job.status == 'cancelled', 5)
    assert _wait(app, lambda: f"[job {job.id}] stopped" in _text(ctl), 5)


def test_info_describes_hashed_sample_as_a_sample():
    # The display sample is chosen by a hash of the ParticleID, not every step-th row:
    # 'info' used to print "(every 13.93...)", a stride that was never used.
    import numpy as np
    from controller.console_commands import describe_snapshot
    snap = {'filepath': '/x/snapshot_000.hdf5', 'c': np.zeros((150_519, 3)), 't': np.ones(150_519, np.uint8),
            'n_total': 2_097_152, 'step': 2_097_152 / 150_000, 'time': 1.0, 'header': {'Time': 1.0}}
    text = describe_snapshot(snap)
    assert "150,519 of 2,097,152 particles (a uniform sample, about 1 in 13.9)" in text
    assert "every" not in text
    snap.update(c=np.zeros((10, 3)), t=np.ones(10, np.uint8), n_total=10, step=1.0)
    assert "all 10 particles" in describe_snapshot(snap)
