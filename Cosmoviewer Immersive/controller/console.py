"""
Console back end for the Telemetry Logs tab.

LogBus collects everything the app reports: print() output, warnings,
logging records and unhandled exceptions from any thread. CommandRegistry
holds the commands typed into the console. Long commands (future AGORA / SXS
import or streaming) run as background jobs that can report progress and be
cancelled, so the UI stays responsive.

Adding a command:

    def cmd_hello(ctx, args):
        ctx.write(f"hello {' '.join(args)}")

    registry.register("hello", cmd_hello, "say hello", usage="hello [name]")

Adding a background command:

    def cmd_fetch(ctx, args):
        def work(job):
            for i in range(10):
                if job.cancelled:
                    return "cancelled"
                job.progress(f"chunk {i + 1}/10")
            return "done"
        ctx.start_job("fetch " + args[0], work)
"""

import logging
import shlex
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from PyQt6 import sip
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

LEVELS = ('debug', 'info', 'warning', 'error', 'command')


@dataclass
class LogEntry:
    level: str
    text: str
    when: float = field(default_factory=time.time)


class LogBus(QObject):
    """Process-wide sink. Thread-safe: emitting from a worker thread is queued to the GUI."""
    entry = pyqtSignal(object)

    _instance: Optional['LogBus'] = None

    def __init__(self):
        super().__init__()
        self.history: List[LogEntry] = []     # kept so the console can show startup output
        self.max_history = 20_000
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'LogBus':
        # Qt can delete the C++ side when the QApplication goes away (tests make several).
        if cls._instance is None or sip.isdeleted(cls._instance):
            old = cls._instance
            cls._instance = LogBus()
            if old is not None:
                cls._instance.history = old.history
        return cls._instance

    def log(self, text: str, level: str = 'info'):
        e = LogEntry(level, str(text).rstrip('\n'))
        with self._lock:
            self.history.append(e)
            if len(self.history) > self.max_history:
                del self.history[:len(self.history) - self.max_history]
        self.entry.emit(e)

    def errors(self) -> List[LogEntry]:
        with self._lock:
            return [e for e in self.history if e.level == 'error']


def log(text: str, level: str = 'info'):
    try:
        LogBus.instance().log(text, level)
    except RuntimeError:        # a connected widget was deleted; never let logging raise
        pass


class _Tee:
    """Copies a stream to the terminal and to the LogBus, one line at a time."""

    def __init__(self, original, level: str):
        self.original = original
        self.level = level
        self._buf = ''
        self._local = threading.local()

    def write(self, text):
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        if getattr(self._local, 'busy', False):      # a slot printed while logging
            return len(text)
        self._local.busy = True
        try:
            self._buf += text
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                if line.strip():
                    log(line, self._classify(line))
        finally:
            self._local.busy = False
        return len(text)

    def _classify(self, line: str) -> str:
        low = line.lower()
        if self.level == 'error' or low.startswith(('error', 'traceback')) or 'exception' in low:
            return 'error'
        if low.startswith('warning') or 'warning:' in low:
            return 'warning'
        return self.level

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.original, name)


class _BusHandler(logging.Handler):
    def emit(self, record):
        level = ('error' if record.levelno >= logging.ERROR else
                 'warning' if record.levelno >= logging.WARNING else
                 'info' if record.levelno >= logging.INFO else 'debug')
        try:
            log(self.format(record), level)
        except Exception:
            pass


_installed = False


def install_log_capture():
    """Route stdout/stderr, logging and unhandled exceptions to the LogBus. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    LogBus.instance()
    sys.stdout = _Tee(sys.__stdout__, 'info')
    # stderr carries Python warnings and library chatter; only tracebacks are errors.
    sys.stderr = _Tee(sys.__stderr__, 'warning')

    handler = _BusHandler()
    handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.captureWarnings(True)

    def excepthook(exc_type, exc, tb):
        # PyQt6 aborts the app on an exception in a slot unless a hook is set;
        # with this hook the error is recorded in full and the app keeps running.
        text = ''.join(traceback.format_exception(exc_type, exc, tb))
        try:
            sys.__stderr__.write(text)
        except Exception:
            pass
        log(f"Unhandled {exc_type.__name__}: {exc}\n{text}", 'error')

    sys.excepthook = excepthook
    threading.excepthook = lambda a: excepthook(a.exc_type, a.exc_value, a.exc_traceback)

    from PyQt6.QtCore import QtMsgType, qInstallMessageHandler

    def qt_handler(mode, context, message):
        try:
            sys.__stderr__.write(message + "\n")
        except Exception:
            pass
        level = {QtMsgType.QtWarningMsg: 'warning', QtMsgType.QtCriticalMsg: 'error',
                 QtMsgType.QtFatalMsg: 'error'}.get(mode, 'debug')
        log(f"Qt: {message}", level)

    qInstallMessageHandler(qt_handler)


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

@dataclass
class Command:
    name: str
    func: Callable
    help: str
    usage: str = ''


class _JobSignals(QObject):
    finished = pyqtSignal(object)


class Job(QRunnable):
    """A background command. work(job) runs off the GUI thread and returns a message."""

    _next_id = 1

    def __init__(self, name: str, work: Callable[['Job'], Optional[str]]):
        super().__init__()
        self.id = Job._next_id
        Job._next_id += 1
        self.name = name
        self.work = work
        self.status = 'queued'
        self.cancelled = False
        self.signals = _JobSignals()
        self.setAutoDelete(False)

    def progress(self, text: str):
        log(f"[job {self.id}] {text}", 'info')

    def cancel(self):
        self.cancelled = True

    def run(self):
        self.status = 'running'
        try:
            msg = self.work(self)
            self.status = 'cancelled' if self.cancelled else 'done'
            if msg:
                log(f"[job {self.id}] {msg}", 'info')
        except Exception as e:
            self.status = 'failed'
            log(f"[job {self.id}] {self.name} failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}", 'error')
        self.signals.finished.emit(self)


class ConsoleContext:
    """What a command can use: the controller, output, and background jobs."""

    def __init__(self, registry: 'CommandRegistry', controller=None, clear: Callable = None):
        self.registry = registry
        self.controller = controller
        self._clear = clear

    def write(self, text: str, level: str = 'info'):
        log(text, level)

    def clear(self):
        if self._clear:
            self._clear()

    def start_job(self, name: str, work: Callable[[Job], Optional[str]]) -> Job:
        return self.registry.start_job(name, work)


class CommandRegistry:
    def __init__(self):
        self.commands: Dict[str, Command] = {}
        self.history: List[str] = []
        self.jobs: Dict[int, Job] = {}
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(2)
        self._register_builtins()

    def register(self, name: str, func: Callable, help: str, usage: str = ''):
        self.commands[name] = Command(name, func, help, usage or name)

    def start_job(self, name: str, work) -> Job:
        job = Job(name, work)
        self.jobs[job.id] = job
        log(f"[job {job.id}] started: {name}", 'info')
        self.pool.start(job)
        return job

    def execute(self, line: str, ctx: ConsoleContext):
        line = line.strip()
        if not line:
            return
        self.history.append(line)
        log(f"> {line}", 'command')
        try:
            parts = shlex.split(line)
        except ValueError as e:
            log(f"could not parse command: {e}", 'error')
            return
        name, args = parts[0], parts[1:]
        cmd = self.commands.get(name)
        if cmd is None:
            log(f"unknown command '{name}'. Type 'help' for the list.", 'warning')
            return
        try:
            cmd.func(ctx, args)
        except Exception as e:
            log(f"{name} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}", 'error')

    def complete(self, prefix: str) -> List[str]:
        return sorted(n for n in self.commands if n.startswith(prefix))

    # -- built-ins -------------------------------------------------------------

    def _register_builtins(self):
        def cmd_help(ctx, args):
            if args and args[0] in self.commands:
                c = self.commands[args[0]]
                ctx.write(f"{c.usage}\n    {c.help}")
                return
            width = max(len(c.usage) for c in self.commands.values())
            ctx.write("\n".join(f"  {c.usage.ljust(width)}  {c.help}"
                                for c in sorted(self.commands.values(), key=lambda c: c.name)))

        def cmd_clear(ctx, args):
            ctx.clear()

        def cmd_history(ctx, args):
            ctx.write("\n".join(f"{i + 1:4d}  {h}" for i, h in enumerate(self.history[-50:])))

        def cmd_errors(ctx, args):
            errs = LogBus.instance().errors()
            n = int(args[0]) if args else 10
            if not errs:
                ctx.write("no errors recorded")
                return
            ctx.write(f"{len(errs)} errors recorded; last {min(n, len(errs))}:")
            for e in errs[-n:]:
                ctx.write(f"[{time.strftime('%H:%M:%S', time.localtime(e.when))}] {e.text}", 'error')

        def cmd_jobs(ctx, args):
            if not self.jobs:
                ctx.write("no jobs")
            for j in self.jobs.values():
                ctx.write(f"  {j.id:3d}  {j.status:9s}  {j.name}")

        def cmd_cancel(ctx, args):
            if not args or not args[0].isdigit() or int(args[0]) not in self.jobs:
                ctx.write("usage: cancel <job id>  (see 'jobs')", 'warning')
                return
            self.jobs[int(args[0])].cancel()
            ctx.write(f"cancel requested for job {args[0]}")

        self.register('help', cmd_help, "list commands, or show one command's usage", 'help [command]')
        self.register('clear', cmd_clear, "clear the console")
        self.register('history', cmd_history, "show the last 50 commands")
        self.register('errors', cmd_errors, "show the last n recorded errors in full", 'errors [n]')
        self.register('jobs', cmd_jobs, "list background jobs")
        self.register('cancel', cmd_cancel, "cancel a background job", 'cancel <job id>')
