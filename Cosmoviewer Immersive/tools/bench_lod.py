"""
Measure the level-of-detail store on a snapshot (no window).

    python3 tools/bench_lod.py <snapshot file or first piece> [--cap N] [--budget N] [--workers N]

Prints the build (or cache open) time, peak memory, and the time and size of
view queries from far away down to a deep zoom on the densest region.
"""

import argparse
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lod.frustum import camera_planes  # noqa: E402
from lod.store import STORE_CAP, build_store  # noqa: E402
from lod.view import base_sample, eye_position, query_view  # noqa: E402
from movie.renderer import Camera, view_matrix  # noqa: E402


def pieces(first: str):
    p = Path(first)
    m = re.match(r'^(.+)\.(\d+)\.(hdf5|h5)$', p.name)
    if not m:
        return [str(p)]
    found = sorted(p.parent.glob(f"{m.group(1)}.*.{m.group(3)}"),
                   key=lambda q: int(q.name.split('.')[-2]))
    return [str(q) for q in found]


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    return r / 1e6 if sys.platform == 'darwin' else r / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot')
    ap.add_argument('--cap', type=int, default=STORE_CAP)
    ap.add_argument('--budget', type=int, default=2_000_000)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    files = pieces(a.snapshot)
    t = time.time()
    s = build_store(files, cap=a.cap, workers=a.workers,
                    progress=lambda i, n: print(f"\r  read {i}/{n} tasks", end='', flush=True))
    print(f"\nstore: {'opened from cache' if s.info.get('from_cache') else 'built'} in {time.time() - t:.1f} s; "
          f"{s.n_stored:,} of {s.layout['n_total']:,} particles (fraction {s.fraction:.4g}), depth {s.cells.depth}, "
          f"{len(s.cells.keys):,} cells, {len(s.blocks['start']):,} file blocks; peak RSS {rss_mb():.0f} MB")
    t = time.time()
    b = base_sample(s, 150_000)
    print(f"base sample 150k: {len(b['t']):,} in {1000 * (time.time() - t):.0f} ms")
    size = s.cells.size
    # densest leaf cell: zoom target
    k = int(np.argmax(s.cells.counts))
    from lod.cells import demorton
    target = s.cells.origin + (demorton(s.cells.keys[k:k + 1])[0] + 0.5) * size / (1 << s.cells.depth)
    print(f"{'camera distance':>16} {'level':>5} {'fraction':>9} {'in view':>13} {'drawn':>10} {'source':>6} "
          f"{'blocks':>7} {'rows read':>12} {'ms':>7}")
    for d in (1.5, 0.5, 0.2, 0.08, 0.03, 0.01, 0.004):
        centre = s.cells.origin + size / 2 if d >= 1 else target
        cam = Camera(center=tuple(centre), distance=d * size)
        r = query_view(s, camera_planes(cam, 1600, 1000, margin=0.15), a.budget, eye_position(view_matrix(cam)), cam.distance)
        print(f"{d:>11.3f} box {r.level:>5} {r.fraction:>9.4g} {r.n_in_view:>13,} {r.n_drawn:>10,} {r.source:>6} "
              f"{r.blocks_read:>7,} {r.rows_read:>12,} {1000 * r.seconds:>7.0f}")
    print(f"peak RSS {rss_mb():.0f} MB")


if __name__ == '__main__':
    main()
