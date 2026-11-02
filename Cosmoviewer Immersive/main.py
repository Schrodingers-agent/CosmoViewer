"""
Entry point for CosmoViewer Immersive.

    python main.py                         open the window, choose a folder there
    python main.py <folder|file|glob> ...  open these snapshots at launch
    python main.py <source> --list         print what would be opened, no window
    python main.py <source> --run info     run console commands once loaded

See cli_import.py for the kinds of source.

Arguments are parsed and sources resolved before PyQt6 is imported, so --list
works without a display. freeze_support() lets the worker processes that read
snapshots and build the level-of-detail store start in a frozen build.
"""
import sys
import multiprocessing as mp

from cli_import import build_parser, parse_args, resolve_sources



def _parser():
    p = build_parser("cosmoviewer", "CosmoViewer Immersive: 3-D viewer for Gadget / EAGLE snapshots.")
    p.add_argument("--max-particles", type=int, default=None, metavar="N",
                   help="particles drawn on screen, the level-of-detail budget (default 2000000); "
                        "snapshots also keep a base sample of up to this many for playback")
    p.add_argument("--run", action="append", default=[], metavar="COMMAND",
                   help="console command to run after loading; repeatable, e.g. --run info")
    return p


def main(argv=None):
    mp.freeze_support()
    args, qt_argv = parse_args(_parser(), list(sys.argv if argv is None else argv))
    plan = resolve_sources(args.sources)
    if args.list:
        print(plan.describe())
        sys.exit(1 if plan.errors else 0)

    from PyQt6.QtWidgets import QApplication
    from controller.app_controller import CosmoViewerController
    from controller.console import install_log_capture

    install_log_capture()      # everything printed or raised shows in the Telemetry Logs console
    app = QApplication(qt_argv)
    controller = CosmoViewerController()
    if args.max_particles:
        controller.max_particles = args.max_particles
    controller.show()
    controller.start_from_command_line(plan, args.run)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
