"""Command-line import: sources named at launch, in the console, and --list."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import cli_import  # noqa: E402
from controller import console  # noqa: E402
from test_multifile_eagle import _write_eagle_like  # noqa: E402


# ── resolver ────────────────────────────────────────────────────────────────

def test_folders_files_globs_and_errors(tmp_path):
    _write_eagle_like(tmp_path, nfiles=3)
    one = tmp_path / "snap_028_z000p000.0.hdf5"
    plan = cli_import.resolve_sources([str(tmp_path), str(one), str(tmp_path / "*.hdf5"),
                                       str(tmp_path / "missing.hdf5"), str(tmp_path / "x*.hdf5")])
    assert plan.folders == [tmp_path]
    assert [p.name for p in plan.files] == [f"snap_028_z000p000.{i}.hdf5" for i in range(3)]
    assert len(plan.errors) == 2 and "no such file" in plan.errors[0] and "no files match" in plan.errors[1]


def test_schemes_are_reported_until_a_handler_exists(tmp_path):
    plan = cli_import.resolve_sources(["agora:ISO/snap_100", "ftp:x", "C:/data/snap.hdf5"])
    assert "not available yet" in plan.errors[0]
    assert "unknown source type 'ftp:'" in plan.errors[1]
    assert "no such file" in plan.errors[2]            # a drive letter, not a scheme
    target = tmp_path / "fetched.hdf5"
    target.write_bytes(b"")
    cli_import.register_source("agora", lambda spec: [target])
    try:
        assert cli_import.resolve_sources(["agora:ISO/snap_100"]).files == [target]
    finally:
        cli_import.SOURCE_HANDLERS["agora"] = None


def test_qt_arguments_pass_through():
    p = cli_import.build_parser("x", "")
    args, rest = cli_import.parse_args(p, ["prog", "a.hdf5", "-style", "--list"])
    assert args.list and "-style" in rest
    with pytest.raises(SystemExit):
        cli_import.parse_args(p, ["prog", "--nonsense"])


def test_list_prints_plan_without_a_window(tmp_path):
    _write_eagle_like(tmp_path, nfiles=2)
    out = subprocess.run([sys.executable, str(ROOT / "main.py"), str(tmp_path), "--list"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert f"folder: {tmp_path}" in out.stdout


# ── the app ─────────────────────────────────────────────────────────────────

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


def _wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _text(c):
    return c.view.telemetry_view.log_edit.toPlainText()


def test_launch_sources_load_then_commands_run(app, ctl, tmp_path):
    _write_eagle_like(tmp_path, nfiles=4)
    plan = cli_import.resolve_sources([str(tmp_path / "snap_028_z000p000.*.hdf5")])
    ctl.max_particles = 200
    ctl.start_from_command_line(plan, ["info"])
    assert _wait(app, lambda: ctl.model.snapshots and not ctl.loading)
    assert _wait(app, lambda: "1 snapshots loaded" in _text(ctl))
    assert len(ctl.model.snapshots) == 1                # 4 pieces are one snapshot
    assert ctl.model.snapshots[0]['n_total'] == 750
    assert ctl.model.manifest is None


def test_console_load_accepts_a_file(app, ctl, tmp_path):
    _write_eagle_like(tmp_path, nfiles=2)
    ctl.console.run(f'load "{tmp_path / "snap_028_z000p000.1.hdf5"}" sxs:BBH:0305')
    assert _wait(app, lambda: ctl.model.snapshots and not ctl.loading)
    text = _text(ctl)
    assert "importing from 'sxs:' is not available yet" in text
    assert len(ctl.model.snapshots) == 1
