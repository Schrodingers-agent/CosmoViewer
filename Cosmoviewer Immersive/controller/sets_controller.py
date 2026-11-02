"""
Send a selection set to the Dataset Explorer.

"Send to Explorer" writes the active set as a selection-set file next to the
snapshots (selection_sets.py), with what the set means: its kind, definition,
the snapshot it was defined at and whether it is complete. An Explorer that
already has a snapshot of the same run open picks the file up by itself and
restricts its analyses to it; otherwise one is started on the current snapshot
with the set chosen. The file stays, so the set can also be used from a script
or loaded back here with Load IDs (which keeps its meaning).
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QMessageBox

from controller.console import log
from selection_sets import explorer_for, write_set

EXPLORER_SCRIPT = Path(__file__).resolve().parents[2] / "Dataset Explorer" / "density_explorer.py"


class SetsMixin:

    def _init_sets(self):
        self._explorer_proc: Optional[subprocess.Popen] = None
        self.view.selection_panel.send_to_explorer_requested.connect(self._send_set_to_explorer)

    @staticmethod
    def _set_colour_hex(s_set) -> str:
        r, g, b = (int(c) for c in s_set.color[:3])        # 0-255, as ParticleSelectionSet stores it
        return f"#{r:02x}{g:02x}{b:02x}"

    def _send_set_to_explorer(self, launch: bool = True) -> Optional[str]:
        sp = self.view.selection_panel
        s_set = self.model.tracker.get_set(sp.active_set_idx)
        snap = self.model.get_snapshot(self.current_snap_idx)
        if not snap or not s_set.selected_ids:
            QMessageBox.information(self.view, "Send to Explorer", "The active set is empty.")
            return None
        try:
            # the file carries what the set means (kind, definition, defining snapshot,
            # completeness), so the Explorer, its analyses and scripts read the same meaning
            path = write_set(snap['filepath'], s_set.id_array(), s_set.name or f"Set {s_set.set_id + 1}",
                             self._set_colour_hex(s_set), made_in="CosmoViewer Immersive",
                             description=s_set.describe(), **s_set.meaning())
        except (OSError, ValueError) as e:
            QMessageBox.warning(self.view, "Send to Explorer", f"Could not write the set: {e}")
            return None
        n = len(s_set.selected_ids)
        pid = explorer_for(snap['filepath'])
        warning = s_set.completeness_warning()
        if warning:
            log(f"{s_set.name}: {warning}", 'warning')
        if pid is not None:
            msg = f"Sent {s_set.name} ({n:,} IDs) to the open Explorer: {path}"
        elif launch:
            msg = self._launch_explorer(snap['filepath'], path, s_set.name, n)
        else:
            msg = f"Saved {s_set.name} ({n:,} IDs) for the Explorer: {path}"
        sp.lbl_status.setText(msg)
        sp.lbl_status.setWordWrap(True)
        log(msg)
        return path

    def _launch_explorer(self, snapshot: str, set_path: str, name: str, n: int) -> str:
        if not EXPLORER_SCRIPT.exists():
            return (f"Saved {name} ({n:,} IDs) to {set_path}. The Explorer was not found at "
                    f"{EXPLORER_SCRIPT}; open the snapshot there and choose the set under Sets.")
        try:
            self._explorer_proc = subprocess.Popen(
                [sys.executable, str(EXPLORER_SCRIPT), snapshot, "--set", set_path],
                cwd=str(EXPLORER_SCRIPT.parent))
        except OSError as e:
            return f"Saved {name} to {set_path}, but the Explorer could not be started: {e}"
        return f"Opening the Explorer with {name} ({n:,} IDs); set file {set_path}"
