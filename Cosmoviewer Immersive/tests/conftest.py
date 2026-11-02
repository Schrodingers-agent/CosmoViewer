"""
pytest session setup for the Immersive Mode tests.

One QApplication lives for the whole session, held here. The test modules
each have an ``app`` fixture that returns ``QApplication.instance()``; before
this file, the first module's fixture created the application and was the
only thing holding it, so when that module finished the application could be
destroyed while its closed windows still existed, and building the next
window crashed in Qt (seen as an intermittent segmentation fault while
constructing a plot widget). The Dataset Explorer's tests had the same
problem and the same fix (Dataset Explorer/conftest.py).

Closed main windows are deleted after each test, at a safe point. A test
drops its controller when it ends, and a controller sits in reference
cycles, so its window used to be destroyed whenever Python's cyclic garbage
collector next ran. That could be in the middle of a queued signal being
delivered to one of its slots, and PyQt then crashed (SIGSEGV in
PyQtSlot::call, main thread "<invalid frame>"). Measured with tests/test_console.py
on 2026-10-24: 11 crashes in 16 runs; with the window deleted at teardown,
0 in 16.
"""
import os
import sys
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = None
_CLOSED = []          # weak references to main windows closed during the current test


def _record_closed_main_windows():
    """Make QMainWindow.close() remember the window, so _settle_qt can delete it."""
    from PyQt6.QtWidgets import QMainWindow
    if getattr(QMainWindow.close, "_records_closed", False):
        return
    original = QMainWindow.close

    def close(self):
        done = original(self)
        _CLOSED.append(weakref.ref(self))
        return done
    close._records_closed = True
    QMainWindow.close = close


def _delete_closed_windows():
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    while _CLOSED:
        w = _CLOSED.pop()()
        if w is not None and not sip.isdeleted(w) and not w.isVisible():
            w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def pytest_configure(config):
    global _APP
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return
    _APP = QApplication.instance() or QApplication(sys.argv)
    _record_closed_main_windows()


@pytest.fixture(autouse=True)
def _settle_qt():
    """After each test, let finished worker threads deliver their signals, then delete
    the main windows the test closed (see the module docstring).

    Unlike the Explorer's conftest this does not force a garbage collection:
    Immersive tests drop controllers whose background lookups (QThreads) may
    still be running, and collecting a running QThread aborts the process
    ("Destroyed while thread is still running")."""
    yield
    if _APP is None:
        return
    for _ in range(3):
        _APP.processEvents()
    _delete_closed_windows()
    _APP.processEvents()


# ----------------------------------------------------------------------------- real Gadget-4 data
# Both suites read the same two example runs, but grew different variable names
# for them. Setting either name now serves both suites:
#   DM-L50-N128/output          COSMOVIEWER_G4_OUTPUT  = COSMOVIEWER_DM_OUTPUT
#   CollidingGalaxiesSFR/output COSMOVIEWER_CG_OUTPUT  = COSMOVIEWER_SFR_OUTPUT
# and COSMOVIEWER_DM_SNAPSHOT defaults to snapshot_000.hdf5 of the DM run.
# This runs before any test module reads os.environ.
_ALIASES = (("COSMOVIEWER_G4_OUTPUT", "COSMOVIEWER_DM_OUTPUT"),
            ("COSMOVIEWER_CG_OUTPUT", "COSMOVIEWER_SFR_OUTPUT"))
for _a, _b in _ALIASES:
    _v = os.environ.get(_a) or os.environ.get(_b)
    if _v:
        os.environ.setdefault(_a, _v)
        os.environ.setdefault(_b, _v)
if os.environ.get("COSMOVIEWER_G4_OUTPUT") and not os.environ.get("COSMOVIEWER_DM_SNAPSHOT"):
    _s = os.path.join(os.environ["COSMOVIEWER_G4_OUTPUT"], "snapshot_000.hdf5")
    if os.path.exists(_s):
        os.environ["COSMOVIEWER_DM_SNAPSHOT"] = _s

# With COSMOVIEWER_REQUIRE_REAL_DATA=1 (use it for release runs), a test skipped for
# want of real data fails instead, so a green run cannot hide that the real-data
# checks never ran. Without it, the run ends with a count of those skips.
_REAL_DATA_WORDS = ("COSMOVIEWER_", "dm_cut", "empty parameter set")
_real_data_skips = []


def _is_real_data_skip(report):
    if not report.skipped or getattr(report, "wasxfail", None) is not None:
        return False
    reason = str(report.longrepr[-1]) if isinstance(report.longrepr, tuple) else str(report.longrepr)
    return any(w in reason for w in _REAL_DATA_WORDS)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _is_real_data_skip(report):
        _real_data_skips.append(report.nodeid)
        if os.environ.get("COSMOVIEWER_REQUIRE_REAL_DATA") == "1":
            report.outcome = "failed"
            report.longrepr = ("COSMOVIEWER_REQUIRE_REAL_DATA=1 but this real-data test was skipped: "
                               + (str(report.longrepr[-1]) if isinstance(report.longrepr, tuple)
                                  else str(report.longrepr)))


def pytest_terminal_summary(terminalreporter):
    if not _real_data_skips:
        return
    if os.environ.get("COSMOVIEWER_REQUIRE_REAL_DATA") == "1":
        terminalreporter.write_line(f"{len(_real_data_skips)} real-data tests had no data and were "
                                    "failed (COSMOVIEWER_REQUIRE_REAL_DATA=1).", red=True)
    else:
        terminalreporter.write_line(
            f"{len(_real_data_skips)} real-data tests were skipped (no Gadget-4 output given). "
            "Set COSMOVIEWER_G4_OUTPUT / COSMOVIEWER_CG_OUTPUT to run them, and "
            "COSMOVIEWER_REQUIRE_REAL_DATA=1 to make their absence fail.", yellow=True)
