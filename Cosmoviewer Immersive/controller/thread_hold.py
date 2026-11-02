"""
Keeping background threads alive until nothing of theirs is queued.

A QThread that is let go (its last Python reference dropped) after its run()
has ended, while one of its signals still waits in the main thread's event
queue, is freed by Python's garbage collector; delivering that signal then
crashes the process (SIGSEGV in PyQtSlot::call). The Dataset Explorer met this
first and keeps its workers in CosmoExplorer._live_workers. Immersive
Mode kept a cancelled job only while ``isRunning()`` was true, and a lookup that
outlasted ``wait(5000)`` was dropped while still running.

``hold(thread)``, called just before ``thread.start()``, keeps the thread
referenced until QThread's own ``finished`` signal has been delivered. That
signal is emitted after run() returns, so once it arrives nothing of the thread
is queued and releasing it is safe. The base-class signal is used even where a
subclass declares its own ``finished`` (AsyncSnapshotLoader does).

When the application quits, threads still held are cancelled (if they can be)
and waited for, up to 10 s each, so none is destroyed while it runs.
"""
from PyQt6.QtCore import QCoreApplication, QThread

_HELD = set()
_QUIT_HOOKED = []


def _stop_all():
    for t in list(_HELD):
        if t.isRunning():
            cancel = getattr(t, 'cancel', None)
            if callable(cancel):
                cancel()
            t.wait(10000)


def hold(thread: QThread) -> QThread:
    """Keep ``thread`` referenced until its QThread.finished signal has been delivered."""
    if thread in _HELD:
        return thread
    _HELD.add(thread)
    app = QCoreApplication.instance()
    if app is not None and not _QUIT_HOOKED:
        app.aboutToQuit.connect(_stop_all)
        _QUIT_HOOKED.append(app)
    QThread.finished.__get__(thread, QThread).connect(lambda t=thread: _HELD.discard(t))
    return thread


def held() -> int:
    """How many threads are held (for tests)."""
    return len(_HELD)
