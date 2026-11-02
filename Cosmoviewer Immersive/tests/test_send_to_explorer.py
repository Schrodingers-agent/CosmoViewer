"""
Send to Explorer: the active selection set is written as a shared set file
(selection_sets.py), an open Explorer for the same run is reused, and
otherwise one is started on the current snapshot with --set.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import selection_sets  # noqa: E402
from model.tracking import read_id_file  # noqa: E402
from test_inspector import _gas_star_pair, _load, app, ctl  # noqa: E402,F401


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(selection_sets, "FALLBACK_ROOT", tmp_path / "home" / "sets")
    monkeypatch.setattr(selection_sets, "INSTANCES_DIR", tmp_path / "home" / "explorer_instances")


def _with_set(app, c, tmp_path, ids):
    _gas_star_pair(tmp_path)
    _load(app, c, tmp_path)
    sp = c.view.selection_panel
    s = c.model.tracker.get_set(sp.active_set_idx)
    s.name = "my halo"
    c._apply_ids_to_set(s, np.asarray(ids, dtype=np.uint64), 'new')
    return s


def test_send_writes_the_set_next_to_the_snapshots(app, ctl, tmp_path, monkeypatch):
    c = ctl
    s = _with_set(app, c, tmp_path, [7, 1003, 1004])
    launched = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: launched.append((args, kw)) or object())
    path = c._send_set_to_explorer()
    assert Path(path).parent == tmp_path / "cosmoviewer_sets"
    st = selection_sets.read_set(path)
    np.testing.assert_array_equal(st.ids, np.array([7, 1003, 1004], dtype=np.uint64))
    assert st.name == "my halo" and st.colour == "#%02x%02x%02x" % tuple(s.color[:3])
    assert st.made_in == "CosmoViewer Immersive" and st.source_snapshot == "snapshot_000.hdf5"
    # loading the file back gives the same IDs
    np.testing.assert_array_equal(read_id_file(path), st.ids)
    # an Explorer was started on the current snapshot with the set (when the script is found)
    import controller.sets_controller as scm
    if scm.EXPLORER_SCRIPT.exists():
        args = launched[0][0]
        assert args[1] == str(scm.EXPLORER_SCRIPT) and args[2].endswith("snapshot_000.hdf5")
        assert args[3:] == ["--set", path]
    else:
        assert "was not found" in c.view.selection_panel.lbl_status.text()


def test_open_explorer_for_the_run_is_reused(app, ctl, tmp_path, monkeypatch):
    c = ctl
    _with_set(app, c, tmp_path, [7])
    selection_sets.register_instance(os.getpid(), [str(tmp_path / "snapshot_000.hdf5")])
    started = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: started.append(a))
    c._send_set_to_explorer()
    assert started == []
    assert "open Explorer" in c.view.selection_panel.lbl_status.text()


def test_launch_uses_the_explorer_next_to_immersive(app, ctl, tmp_path, monkeypatch):
    import controller.sets_controller as scm
    fake = tmp_path / "Dataset Explorer" / "density_explorer.py"
    fake.parent.mkdir()
    fake.write_text("")
    monkeypatch.setattr(scm, "EXPLORER_SCRIPT", fake)
    c = ctl
    _with_set(app, c, tmp_path, [7])
    got = []
    monkeypatch.setattr(scm.subprocess, "Popen", lambda args, **kw: got.append((args, kw)) or object())
    path = c._send_set_to_explorer()
    args, kw = got[0]
    assert args == [sys.executable, str(fake), str(tmp_path / "snapshot_000.hdf5"), "--set", path]
    assert kw["cwd"] == str(fake.parent)


def test_empty_set_is_not_sent(app, ctl, tmp_path, monkeypatch):
    c = ctl
    _gas_star_pair(tmp_path)
    _load(app, c, tmp_path)
    assert c._send_set_to_explorer() is None
    assert not (tmp_path / "cosmoviewer_sets").exists()
