"""
Run snapshot FoF in its own Python process, so a crash cannot take the viewer down.

FoF on a big snapshot reads every linking particle at full resolution, and the
memory that needs grows with the snapshot.  Run inside the viewer, running out
of memory (or any crash in the linker) ends the whole application.  Here the
viewer starts ``python -m snapshot_io.fof_process <work dir>`` instead and
watches it:

  * progress lines come back on the child's stdout and are passed to a callback;
  * the child's memory (resident plus swapped/compressed, see _footprint) is
    checked several times a second and the child is stopped if it passes the
    limit (MEMORY_FRACTION of physical memory by default), before the computer
    starts thrashing;
  * a child that fails, runs out of memory or is killed by the system is
    reported as a FoFProcessError with the reason; the viewer carries on;
  * cancelling stops the child at once.

The result comes back through .npy files in a temporary directory: particle
IDs sorted, the group of each, and the group lengths, which is all the viewer
uses (SnapshotFoF.lookup, result.group_len, result.linking_length).  The
per-primary result.group_index of an in-process run is not sent back, to keep
the transfer small; it is None in the returned object.

The child exits by itself if the viewer goes away (its progress writes fail).
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

MEMORY_FRACTION = float(os.environ.get('COSMOVIEWER_FOF_MEMORY_FRACTION', 0.6))
_TAG = '@@COSMOVIEWER-FOF '
_APP_ROOT = Path(__file__).resolve().parent.parent


class FoFProcessError(RuntimeError):
    """The FoF process failed, ran out of memory or was stopped; the message says which."""


class FoFCancelled(Exception):
    """The run was cancelled by the user."""


def physical_memory_bytes() -> Optional[int]:
    try:
        return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        return None


_libproc = None


def _footprint_macos(pid: int) -> Optional[int]:
    """phys_footprint from proc_pid_rusage: what Activity Monitor shows as Memory, swap included."""
    global _libproc
    import ctypes
    import ctypes.util
    if _libproc is None:
        _libproc = ctypes.CDLL(ctypes.util.find_library('proc') or '/usr/lib/libproc.dylib')
    buf = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(2), buf) != 0:   # RUSAGE_INFO_V2
        return None
    # rusage_info_v2: 16-byte uuid, then uint64 fields; ri_phys_footprint is the 8th
    return int.from_bytes(buf.raw[72:80], sys.byteorder)


def _footprint(pid: int) -> Optional[int]:
    """Memory the process holds, in bytes, counting what was swapped out; None if unknown."""
    try:
        if sys.platform.startswith('linux'):
            fields = {}
            with open(f'/proc/{pid}/status') as f:
                for line in f:
                    k, _, v = line.partition(':')
                    if k in ('VmRSS', 'VmSwap'):
                        fields[k] = int(v.split()[0]) * 1024
            return sum(fields.values()) if fields else None
        if sys.platform == 'darwin':
            fp = _footprint_macos(pid)
            if fp:
                return fp
        out = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) * 1024 if out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _describe_exit(code: int) -> str:
    if code < 0:
        sig = -code
        if sig == 9:
            return ("was stopped by the operating system (signal 9), which usually means the "
                    "computer ran out of memory")
        if sig == 11:
            return "crashed (segmentation fault)"
        return f"was stopped by signal {sig}"
    return f"exited with code {code}"


def run_fof_process(path: str, kwargs: dict, progress: Optional[Callable[[float, str], None]] = None,
                    cancelled: Optional[Callable[[], bool]] = None, memory_limit: Optional[int] = None,
                    poll: float = 0.2):
    """run_snapshot_fof(path, **kwargs) in a child process; returns its SnapshotFoF.

    memory_limit in bytes (default MEMORY_FRACTION of physical memory; 0 = none).
    Raises FoFProcessError (failure, crash, memory limit) or FoFCancelled.
    """
    from .fof import FoFResult
    from .fof_snapshot import SnapshotFoF

    if memory_limit is None:
        ram = physical_memory_bytes()
        memory_limit = int(MEMORY_FRACTION * ram) if ram else 0
    work = Path(tempfile.mkdtemp(prefix='cosmoviewer-fof-'))
    proc = None
    try:
        (work / 'args.json').write_text(json.dumps({'path': os.path.abspath(str(path)), 'kwargs': kwargs}))
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(_APP_ROOT)] + [p for p in sys.path if p and os.path.isdir(p)])
        env['PYTHONUNBUFFERED'] = '1'
        with open(work / 'stderr.txt', 'w') as err:
            proc = subprocess.Popen([sys.executable, '-m', 'snapshot_io.fof_process', str(work)],
                                    cwd=str(_APP_ROOT), env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        lines: 'queue.Queue[Optional[str]]' = queue.Queue()

        def pump():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()

        error, done, eof, peak, stopped = None, False, False, 0, None
        while True:
            try:
                line = lines.get(timeout=poll)
                while True:
                    if line is None:
                        eof = True
                    elif line.startswith(_TAG):
                        msg = json.loads(line[len(_TAG):])
                        if msg['kind'] == 'progress' and progress is not None:
                            progress(float(msg['fraction']), msg['text'])
                        elif msg['kind'] == 'error':
                            error = msg['text']
                        elif msg['kind'] == 'done':
                            done = True
                    line = lines.get_nowait()
            except queue.Empty:
                pass
            if cancelled is not None and cancelled():
                stopped = 'cancelled'
                break
            used = _footprint(proc.pid) if proc.poll() is None else None
            if used:
                peak = max(peak, used)
                if memory_limit and used > memory_limit:
                    stopped = 'memory'
                    break
            if eof and proc.poll() is not None:
                break
        if stopped:
            proc.kill()
            proc.wait()
            if stopped == 'cancelled':
                raise FoFCancelled()
            raise FoFProcessError(
                f"FoF needed more than {memory_limit / 2**30:.1f} GB of memory, the limit set for it "
                f"on this computer (which has {(physical_memory_bytes() or 0) / 2**30:.1f} GB), "
                f"so it was stopped before the computer ran out. The viewer is unaffected. "
                f"Link fewer particle types, or run FoF on a machine with more memory.")
        code = proc.wait()
        if error:
            raise FoFProcessError(error)
        if code != 0 or not done:
            tail = (work / 'stderr.txt').read_text(errors='replace').strip().splitlines()[-8:]
            detail = ("\n\nLast output:\n" + "\n".join(tail)) if tail else ""
            raise FoFProcessError(f"The FoF process {_describe_exit(code)} (peak memory about "
                                  f"{peak / 2**30:.1f} GB). The viewer is unaffected.{detail}")
        meta = json.loads((work / 'meta.json').read_text())
        ids = np.load(work / 'ids.npy')
        gidx = np.load(work / 'group_index.npy')
        glen = np.load(work / 'group_len.npy')
        result = FoFResult(None, glen, float(meta['linking_length']))
        return SnapshotFoF(result, ids, gidx, meta['boxsize'], meta['convention'], ids_sorted=True)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(work, ignore_errors=True)


def _child(work: Path) -> int:
    from .fof_snapshot import run_snapshot_fof

    def send(kind, **msg):
        sys.stdout.write(_TAG + json.dumps({'kind': kind, **msg}) + '\n')
        sys.stdout.flush()   # fails once the viewer is gone, which ends this process

    last = [0.0, None]

    def progress(fraction, text):
        now = time.monotonic()
        if text != last[1] or now - last[0] > 0.2:
            last[:] = [now, text]
            send('progress', fraction=fraction, text=text)

    try:
        os.nice(5)    # keep the viewer responsive
    except (AttributeError, OSError):
        pass
    try:
        args = json.loads((work / 'args.json').read_text())
        fof = run_snapshot_fof(args['path'], progress=progress, **args['kwargs'])
        gidx = fof.group_index
        if len(fof.result.group_len) < 2 ** 31:
            gidx = gidx.astype(np.int32)
        np.save(work / 'ids.npy', fof.ids)
        np.save(work / 'group_index.npy', gidx)
        np.save(work / 'group_len.npy', fof.result.group_len)
        (work / 'meta.json').write_text(json.dumps({'linking_length': fof.result.linking_length,
                                                    'boxsize': fof.boxsize, 'convention': fof.convention}))
    except (BrokenPipeError, KeyboardInterrupt):
        return 3
    except BaseException as e:  # reported to the viewer, never swallowed
        try:
            send('error', text=f"{type(e).__name__}: {e}")
        except BrokenPipeError:
            pass
        return 1
    send('done')
    return 0


if __name__ == '__main__':
    sys.exit(_child(Path(sys.argv[1])))
