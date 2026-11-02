"""
"Log & console" tab: a terminal-style console.

Shows everything the app reports (output, warnings, errors with full
tracebacks) and takes typed commands. The commands themselves live in
controller/console.py; this widget only displays and collects input.
"""

import html
import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                             QVBoxLayout, QWidget)

COLOURS = {
    'debug': '#6a6a6a',
    'info': '#c8c8c8',
    'warning': '#e5c07b',
    'error': '#ff6b6b',
    'command': '#00e5ff',
}

CONSOLE_STYLE = """
QPlainTextEdit#ConsoleOutput { background: #000000; color: #c8c8c8; border: 1px solid #222;
                               selection-background-color: #007acc; }
QLineEdit#ConsoleInput { background: #000000; color: #ffffff; border: 1px solid #333; padding: 4px; }
QLabel#ConsolePrompt { color: #00e5ff; font-weight: bold; }
"""


class _InputLine(QLineEdit):
    """Command line with Up/Down history and Tab completion."""
    history_requested = pyqtSignal(int)
    complete_requested = pyqtSignal(str)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Up:
            self.history_requested.emit(-1)
        elif key == Qt.Key.Key_Down:
            self.history_requested.emit(1)
        elif key == Qt.Key.Key_Tab:
            self.complete_requested.emit(self.text())
        else:
            super().keyPressEvent(event)

    def focusNextPrevChild(self, forward):
        return False          # keep Tab for completion


class TelemetryView(QWidget):
    """View component displaying snapshot telemetry and diagnostic log messages."""

    command_entered = pyqtSignal(str)
    completion_requested = pyqtSignal(str)
    error_count_changed = pyqtSignal(int)

    MAX_LINES = 20_000

    def __init__(self):
        super().__init__()
        self.setStyleSheet(CONSOLE_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        top = QHBoxLayout()
        title = QLabel("LOG & CONSOLE")
        title.setObjectName("Title")
        top.addWidget(title)
        top.addStretch()
        self.lbl_counts = QLabel("0 errors · 0 warnings")
        top.addWidget(self.lbl_counts)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_log)
        top.addWidget(btn_clear)
        layout.addLayout(top)

        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(11)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setObjectName("ConsoleOutput")
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(mono)
        self.log_edit.setMaximumBlockCount(self.MAX_LINES)
        self.log_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.log_edit, stretch=1)

        row = QHBoxLayout()
        prompt = QLabel("cosmo>")
        prompt.setObjectName("ConsolePrompt")
        prompt.setFont(mono)
        row.addWidget(prompt)
        self.input = _InputLine()
        self.input.setObjectName("ConsoleInput")
        self.input.setFont(mono)
        self.input.setPlaceholderText("type 'help' for commands")
        self.input.returnPressed.connect(self._submit)
        self.input.history_requested.connect(self._step_history)
        self.input.complete_requested.connect(self.completion_requested)
        row.addWidget(self.input, stretch=1)
        layout.addLayout(row)

        self._history = []
        self._hist_pos = 0
        self.n_errors = 0
        self.n_warnings = 0

    # -- output ----------------------------------------------------------------

    def append_entry(self, level: str, text: str, when: float = None):
        stamp = time.strftime('%H:%M:%S', time.localtime(when or time.time()))
        colour = COLOURS.get(level, COLOURS['info'])
        body = html.escape(text).replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        tag = {'error': 'ERROR ', 'warning': 'WARN  '}.get(level, '')
        self.log_edit.appendHtml(
            f'<span style="color:#555">{stamp}</span> '
            f'<span style="color:{colour}">{tag}{body}</span>')
        if level == 'error':
            self.n_errors += 1
        elif level == 'warning':
            self.n_warnings += 1
        if level in ('error', 'warning'):
            self.lbl_counts.setText(f"{self.n_errors} errors · {self.n_warnings} warnings")
            self.error_count_changed.emit(self.n_errors)
        bar = self.log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def on_log_entry(self, e):
        self.append_entry(e.level, e.text, e.when)

    def append_log(self, text: str):
        self.append_entry('info', text)

    def clear_log(self):
        self.log_edit.clear()

    # -- input -----------------------------------------------------------------

    def _submit(self):
        line = self.input.text()
        self.input.clear()
        if line.strip():
            self._history.append(line)
        self._hist_pos = len(self._history)
        self.command_entered.emit(line)

    def _step_history(self, delta: int):
        if not self._history:
            return
        self._hist_pos = max(0, min(len(self._history), self._hist_pos + delta))
        self.input.setText(self._history[self._hist_pos] if self._hist_pos < len(self._history) else "")
        self.input.setCursorPosition(len(self.input.text()))

    def set_completion(self, text: str):
        self.input.setText(text)
        self.input.setCursorPosition(len(text))
