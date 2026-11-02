"""
From a fresh clone to a first rendered frame, on macOS, Linux or Windows.

    python tools/first_run.py              make .venv, install, check, render
    python tools/first_run.py --no-venv    use the Python running this script
    python tools/first_run.py --offscreen  render without opening a window

On Windows, if `python` opens the Microsoft Store, use `py tools\\first_run.py`.

Steps: check the Python version, create .venv in the repository root (unless
--no-venv), install the packages listed in the README, check that they import,
ask both applications what they would open on the small test snapshot that
ships in _dev_tools/inspector_check/dm_cut, then open that snapshot in the 3D
viewer, check that particles were actually drawn in the 3D view, and save the
window as first_render.png in the repository root.

Everything the steps print, pip's output included, also goes to first_run.log
in the repository root; send that file along if a step fails.

Only the standard library is used here, so it runs before anything is installed.
Each step prints PASS, FAIL or SKIP; the script stops at the first failure.
"""

import argparse
import os
import platform
import struct
import subprocess
import sys
import textwrap
import threading
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMMERSIVE = ROOT / "Cosmoviewer Immersive"
EXPLORER = ROOT / "Dataset Explorer"
SAMPLE = ROOT / "_dev_tools" / "inspector_check" / "dm_cut"
LOG_PATH = ROOT / "first_run.log"
PACKAGES = ["PyQt6>=6.5", "pyqtgraph>=0.13", "PyOpenGL>=3.1", "numpy>=1.24",
            "h5py>=3.8", "scipy>=1.10", "astropy>=5.0"]
MIN_PARTICLE_PIXELS = 200       # red pixels the 3D view must show; a good render of dm_cut has about 6000 on a Mac

RENDER = textwrap.dedent(r"""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    import numpy as np
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from cli_import import resolve_sources
    from controller.app_controller import CosmoViewerController
    app = QApplication(sys.argv[:1])
    print("Qt platform:", app.platformName())
    c = CosmoViewerController()
    c.show()
    if not c.load_sources(resolve_sources([sys.argv[2]])):
        sys.exit("the snapshot folder could not be opened")
    t0 = time.time()
    while (c.loading or not c.model.snapshots) and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    if not c.model.snapshots:
        sys.exit("no snapshot loaded within 120 s")
    t1 = time.time()
    while time.time() - t1 < 2.0:          # let the view draw a few frames
        app.processEvents()
        time.sleep(0.02)
    snap = c.model.snapshots[0]
    print(f"loaded {len(c.model.snapshots)} snapshot(s), {len(snap['c']):,} particles drawn")
    # Dark matter is drawn red; the scale bar, labels and the no-OpenGL placeholder are
    # white or grey. Red pixels in the 3D view mean the particles really reached the screen.
    img = c.view.view_3d.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    bits = img.constBits()
    bits.setsize(img.sizeInBytes())
    px = np.frombuffer(bits, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w].astype(int)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    red = int(((r > 150) & (g < 90) & (b < 90)).sum())
    print(f"3D view: {w}x{h} pixels, {red} red particle pixels")
    if not c.view.grab().save(sys.argv[3]):
        sys.exit("could not write " + sys.argv[3])
    c.view.close()
""")

_log = None


def say(text, echo=True):
    """Print text and append it to first_run.log."""
    _log.write(text + "\n")
    _log.flush()
    if echo:
        print(text, flush=True)


def step(name, ok, detail="", output=""):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        tail = output.strip().splitlines()[-20:]
        if tail:
            say("  last lines of output:\n    " + "\n    ".join(tail))
        say(f"  full output: {LOG_PATH}")
        sys.exit(1)


def child_env(**extra):
    # UTF-8 throughout: on Windows a pipe defaults to the ANSI code page, which
    # cannot carry every path or unit symbol the applications print
    return dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", **extra)


def run(cmd, cwd=None, env=None, timeout=600, echo=False):
    """Run cmd and log its output; return (exit code, output). echo shows pip's progress lines live."""
    say("$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), echo=False)
    try:
        p = subprocess.Popen(cmd, cwd=cwd, env=env or child_env(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        say(str(e), echo=False)
        return 1, str(e)
    killed = []
    timer = threading.Timer(timeout, lambda: (killed.append(True), p.kill()))
    timer.start()
    lines = []
    try:
        for line in p.stdout:
            line = line.rstrip()
            lines.append(line)
            say(line, echo=False)
            if echo and line.startswith(("Collecting", "Successfully", "ERROR")):
                print("  " + line, flush=True)
        code = p.wait()
    finally:
        timer.cancel()
    if killed:
        lines.append(f"(stopped after {timeout} s)")
        say(lines[-1], echo=False)
    return code, "\n".join(lines)


def last(out):
    return out.strip().splitlines()[-1] if out.strip() else ""


def venv_python(where: Path) -> Path:
    return where / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    global _log
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--no-venv", action="store_true", help="use this Python instead of making .venv")
    ap.add_argument("--offscreen", action="store_true", help="render without a window (Qt offscreen)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    _log = open(LOG_PATH, "w", encoding="utf-8")
    say(f"first_run.py on {platform.platform()} ({platform.machine()}), repository {ROOT}", echo=False)

    v = sys.version_info
    step("Python version", (3, 10) <= v[:2] <= (3, 14), f"{v.major}.{v.minor}.{v.micro} (3.10 to 3.14 supported)")
    bits = struct.calcsize("P") * 8
    step("64-bit Python", bits == 64, f"{bits}-bit, {platform.machine()}")

    if args.no_venv:
        py = Path(sys.executable)
    else:
        where = ROOT / ".venv"
        if not venv_python(where).exists():
            print("Creating .venv ...")
            venv.EnvBuilder(with_pip=True).create(where)
        py = venv_python(where)
        step("virtual environment", py.exists(), str(where))
        print("Installing packages (a few minutes the first time) ...")
        code, out = run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        step("pip upgrade", code == 0, last(out), out)
        code, out = run([str(py), "-m", "pip", "install", "--progress-bar", "off", *PACKAGES],
                        timeout=1800, echo=True)
        if code != 0 and os.name == "nt":
            say("  On Windows a failed install is usually a path too long for Windows (clone into a short"
                " folder such as C:\\cv) or a Python with no ready-made packages (use 64-bit Python 3.12"
                " or 3.13 from python.org).")
        step("package install", code == 0, last(out), out)

    code, out = run([str(py), "-c", "import PyQt6.QtGui, pyqtgraph, OpenGL, numpy, h5py, scipy, astropy; print('ready')"])
    step("imports", code == 0 and out.endswith("ready"), last(out), out)

    step("test snapshot present", (SAMPLE / "snapshot_002.hdf5").exists(), str(SAMPLE))
    code, out = run([str(py), "main.py", str(SAMPLE), "--list"], cwd=IMMERSIVE)
    step("3D viewer accepts the folder", code == 0 and "dm_cut" in out, last(out), out)
    code, out = run([str(py), "density_explorer.py", str(SAMPLE / "snapshot_002.hdf5"), "--list"], cwd=EXPLORER)
    step("Dataset Explorer accepts the file", code == 0 and "snapshot_002" in out, last(out), out)

    env = child_env(PYTHONDONTWRITEBYTECODE="1")
    if args.offscreen:
        env["QT_QPA_PLATFORM"] = "offscreen"
    out_png = ROOT / "first_render.png"
    if out_png.exists():
        out_png.unlink()            # a stale picture must not pass for this run's
    print("Opening the snapshot in the 3D viewer ...")
    code, out = run([str(py), "-c", RENDER, str(IMMERSIVE), str(SAMPLE), str(out_png)],
                    cwd=IMMERSIVE, env=env, timeout=300)
    loaded = [l for l in out.splitlines() if l.startswith("loaded ")]
    step("window rendered", code == 0 and out_png.exists(),
         (loaded[-1] if loaded else last(out)) + f" -> {out_png.name}", out)

    view = [l for l in out.splitlines() if l.startswith("3D view: ")]
    red = int(view[-1].split(", ")[1].split()[0]) if view else 0
    if args.offscreen:
        say("[SKIP] particles visible in the 3D view: Qt's offscreen mode has no OpenGL here, so the 3D view"
            " stays blank; run without --offscreen to check it")
    else:
        step("particles visible in the 3D view", red >= MIN_PARTICLE_PIXELS,
             (view[-1][len("3D view: "):] if view else "not measured") +
             ("" if red >= MIN_PARTICLE_PIXELS else
              f", need {MIN_PARTICLE_PIXELS}: the window opened but OpenGL drew no particles (see first_render.png)"),
             out)

    act = r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(f"\nAll steps passed. To work with CosmoViewer: {act}, then\n"
          f"  cd \"Cosmoviewer Immersive\" && python main.py <snapshot folder>")


if __name__ == "__main__":
    main()
