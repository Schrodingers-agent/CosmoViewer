"""
Movie maker v2: in-between frames, camera paths, smoothed frames, overlays,
quick MP4/GIF output, lazy scene requests.

Synthetic tests check the mathematics exactly; the DM-L50 test checks the
comoving velocity factor against the snapshots themselves.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from movie.interpolate import PairInterpolator, dx_dtime_factor, hermite  # noqa: E402
from movie.keyframes import Keyframe, camera_path  # noqa: E402
from movie.maker import MovieSettings, Scene, can_write_mp4, make_movie  # noqa: E402
from movie.overlays import LengthInfo, scale_bar  # noqa: E402
from movie.renderer import Camera, project  # noqa: E402

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _scene(pos, ids, vel=None, t=0.0, box=0.0, hl=None, files=(), dxdt=1.0, label=None):
    n = len(pos)
    hl_pos = hl_ids = hl_vel = None
    if hl is not None:
        hl_pos, hl_ids, hl_vel = pos[hl], ids[hl], None if vel is None else vel[hl]
    return Scene(pos=np.asarray(pos, float), colors=np.tile([1, 1, 1, 0.5], (n, 1)), ids=np.asarray(ids, np.uint64),
                 vel=vel, mass=np.ones(n), time=t, box_period=box, dxdt=dxdt, files=list(files),
                 highlight_pos=hl_pos, highlight_colors=None if hl is None else np.tile([1, 0, 0, 1], (len(hl_pos), 1)),
                 highlight_ids=hl_ids, highlight_vel=hl_vel, label=label or f"t = {t:.4g} (code units)",
                 centre=None if hl_pos is None else hl_pos.mean(axis=0))


# ---------------------------------------------------------------- in-between frames

def test_hermite_end_points_exact():
    rng = np.random.default_rng(1)
    p0, p1, m0, m1 = (rng.normal(size=(50, 3)) for _ in range(4))
    assert np.array_equal(hermite(p0, p1, m0, m1, 0.0), p0)
    assert np.array_equal(hermite(p0, p1, m0, m1, 1.0), p1)


def test_pair_exact_for_accelerated_motion_across_periodic_boundary():
    """x(t) = x0 + v0 t + g t^2 / 2 is a cubic, so the Hermite curve must reproduce it exactly."""
    rng = np.random.default_rng(2)
    box, n = 10.0, 300
    x0 = rng.uniform(0, box, (n, 3))
    # Displacements stay under half the box, as between real snapshots (the unwrapping
    # takes the nearest image, so a particle crossing more than half the box is ambiguous).
    v0 = rng.uniform(-2, 2, (n, 3))
    g = rng.uniform(-1, 1, (n, 3))
    t0, t1 = 1.0, 2.5

    def at(t):
        dt = t - t0
        return np.mod(x0 + v0 * dt + 0.5 * g * dt * dt, box), v0 + g * dt

    ids = np.arange(1, n + 1, dtype=np.uint64)
    pa, va = at(t0)
    pb, vb = at(t1)
    perm = rng.permutation(n)                       # B's rows in another order
    a = _scene(pa, ids, va, t0, box)
    b = _scene(pb[perm], ids[perm], vb[perm], t1, box)
    pair = PairInterpolator(a, b)
    assert pair.dropped == 0
    for f in (0.25, 0.5, 0.8):
        sc = pair.scene_at(f)
        truth, _ = at(t0 + f * (t1 - t0))
        d = sc.pos - truth
        d -= box * np.round(d / box)
        assert np.max(np.abs(d)) < 1e-9
        assert sc.interpolated and "interpolated" in sc.label
        assert np.array_equal(sc.ids, ids)
    # Many particles cross the boundary between t0 and t1, so the unwrapping was exercised.
    raw = x0 + v0 * (t1 - t0) + 0.5 * g * (t1 - t0) ** 2
    assert np.sum(np.any((raw < 0) | (raw >= box), axis=1)) > 20


def test_pair_looks_up_missing_particles_and_drops_vanished():
    n = 100
    ids = np.arange(1, n + 1, dtype=np.uint64)
    pa = np.random.default_rng(3).uniform(0, 1, (n, 3))
    va = np.zeros((n, 3))
    a = _scene(pa, ids, va, 0.0, files=['A'])
    keep_b = ids <= 60                               # B's sample has 60 of A's particles
    b = _scene(pa[keep_b] + 1.0, ids[keep_b], va[keep_b], 1.0, files=['B'])
    asked = {}

    def locator(files, want):
        asked['files'], asked['ids'] = files, want.copy()
        found = want <= 90                           # 61-90 are in B's file, 91-100 no longer exist
        pos = np.full((len(want), 3), np.nan)
        pos[found] = pa[(want[found] - 1).astype(int)] + 1.0
        vel = np.where(found[:, None], 0.0, np.nan)
        return pos, vel, found

    pair = PairInterpolator(a, b, locator)
    assert asked['files'] == ['B'] and np.array_equal(asked['ids'], np.arange(61, 101))
    assert pair.dropped == 10 and pair.looked_up == 30
    sc = pair.scene_at(0.5)
    assert len(sc.pos) == 90 and len(sc.colors) == 90 and len(sc.mass) == 90
    assert np.allclose(sc.pos, pa[:90] + 0.5)        # zero velocities: Hermite gives 3f^2-2f^3 = 0.5


def test_velocity_factor_non_comoving_is_one():
    assert dx_dtime_factor({'BoxSize': 0.0, 'Time': 1.2}, {'ComovingIntegrationOn': 0}, 0.0) == 1.0


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_comoving_velocity_factor_matches_snapshots():
    """
    DM-L50: between a = 0.25 and a = 0.5 the growth is close to linear in a, so
    dx/da from the stored velocities (u a^-1.5 / H) at both ends must agree with
    (x(a1) - x(a0)) / (a1 - a0). A wrong factor (no sqrt(a), an h, H0 in the
    wrong units) is off by 30 % or more.
    """
    import h5py
    from snapshot_io.id_lookup import _attrs
    out = Path(G4_OUTPUT)
    data = []
    for name in ("snapshot_000.hdf5", "snapshot_001.hdf5"):
        with h5py.File(out / name, "r") as f:
            g = f["PartType1"]
            order = np.argsort(g["ParticleIDs"][:])
            x = g["Coordinates"][:][order].astype(np.float64)
            u = g["Velocities"][:][order].astype(np.float64)
            fac = dx_dtime_factor(_attrs(f["Header"]), _attrs(f["Parameters"]),
                                  float(g["Velocities"].attrs["a_scaling"]))
            data.append((float(f["Header"].attrs["Time"]), x, u * fac))
    (a0, x0, d0), (a1, x1, d1) = data
    dx = x1 - x0
    dx -= 50.0 * np.round(dx / 50.0)
    fd = dx / (a1 - a0)
    mean_deriv = 0.5 * (d0 + d1)
    # Compare the bulk flow in 16^3 cells (3.1 Mpc/h): inside haloes, orbital motion is
    # not traced by a straight finite difference, and it would only add scatter.
    cell = np.floor(np.mod(x0, 50.0) / 50.0 * 16).astype(int)
    key = (cell[:, 0] * 16 + cell[:, 1]) * 16 + cell[:, 2]
    cnt = np.bincount(key, minlength=4096)
    occ = cnt > 0

    def avg(v):
        return np.stack([np.bincount(key, v[:, k], minlength=4096) / np.maximum(cnt, 1) for k in range(3)], 1)[occ]
    A, B = avg(fd), avg(mean_deriv)
    slope = float(np.sum(A * B) / np.sum(B * B))
    corr = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    print(f"\nDM-L50 a {a0:.3f}->{a1:.3f}: bulk-flow slope {slope:.3f}, correlation {corr:.4f}")
    assert 0.95 < slope < 1.08 and corr > 0.99
    # The check can tell: dropping the sqrt(a) of the stored velocity misses by over 30 %.
    wrong = avg(0.5 * (d0 * np.sqrt(a0) + d1 * np.sqrt(a1)))
    assert float(np.sum(A * wrong) / np.sum(wrong * wrong)) > 1.3


# ---------------------------------------------------------------- camera paths

def test_camera_path_passes_through_keyframes():
    kfs = [Keyframe({'center': (0, 0, 0), 'distance': 10, 'elevation': 30, 'azimuth': 350, 'fov': 60}, 0),
           Keyframe({'center': (5, 0, 0), 'distance': 40, 'elevation': 10, 'azimuth': 20, 'fov': 50}, 2),
           Keyframe({'center': (5, 5, 0), 'distance': 20, 'elevation': -20, 'azimuth': 90, 'fov': 60}, 2)]
    path = camera_path(kfs, 21, ease=False)
    for k, kf in ((0, kfs[0]), (10, kfs[1]), (20, kfs[2])):
        cam, s = path[k]
        assert np.allclose(cam.center, kf.camera['center'])
        assert cam.distance == pytest.approx(kf.camera['distance'])
        assert (cam.azimuth - kf.camera['azimuth']) % 360 == pytest.approx(0) or \
            (cam.azimuth - kf.camera['azimuth']) % 360 == pytest.approx(360)
        assert s == kf.snapshot
    az = [c.azimuth for c, _ in path[:11]]
    assert max(np.abs(np.diff(az))) < 10            # 350 -> 20 goes the short way (30 degrees)
    snaps = [s for _, s in path]
    assert all(np.diff(snaps) >= 0) and snaps[5] == pytest.approx(1.0)
    eased = camera_path(kfs, 21, ease=True)
    assert np.allclose(eased[0][0].center, (0, 0, 0)) and np.allclose(eased[-1][0].center, (5, 5, 0))
    d_start = np.linalg.norm(np.subtract(eased[1][0].center, eased[0][0].center))
    assert d_start < np.linalg.norm(np.subtract(path[1][0].center, path[0][0].center))


def test_keyframe_round_trip():
    kf = Keyframe({'center': (1.0, 2.0, 3.0), 'distance': 5.0, 'elevation': 1.0, 'azimuth': 2.0, 'fov': 60.0}, 3, 'a')
    assert Keyframe.from_dict(json.loads(json.dumps(kf.to_dict()))) == kf


# ---------------------------------------------------------------- smoothed frames and overlays

def test_smoothed_frame_orientation_and_mass(app):
    from movie.smoothed import density_image
    cam = Camera(center=(0, 0, 0), distance=10, elevation=25, azimuth=60, fov=60)
    w, h = 320, 200
    # One clump right of and above the centre on screen, as the point renderer sees it.
    rng = np.random.default_rng(4)
    probe = np.array([[2.0, -1.0, 1.5]])
    px, py, _, _ = project(probe, cam, w, h)
    pts = probe + rng.normal(0, 0.05, (2000, 3))
    sc = Scene(pos=pts, colors=np.ones((2000, 4)), mass=np.full(2000, 0.5), mass_scale=2.0)
    img = density_image(sc, cam, w, h)
    r, c = np.unravel_index(np.argmax(img), img.shape)
    # Orthographic vs perspective differ a little off-centre, so compare sides, not pixels.
    assert np.sign(c - w / 2) == np.sign(px[0] - w / 2) and np.sign(r - h / 2) == np.sign(py[0] - h / 2)
    pix = 2 * 10 * np.tan(np.radians(30)) / w
    assert img.sum() * pix * pix == pytest.approx(2000 * 0.5 * 2.0, rel=1e-3)


def test_scale_bar_labels():
    info = LengthInfo(unit_cm=3.085678e24, comoving=True, a=0.5, h=0.678, a_scaling=1.0, h_scaling=-1.0)
    px, lab = scale_bar(10.0, 640, info, 'physical', perspective=False)
    # 10 Mpc/h comoving = 10 * 0.5 / 0.678 = 7.375 Mpc physical; 20 % is 1.47 -> 1 Mpc.
    assert lab == "1 Mpc physical" and px == round(1 / 7.3746 * 640)
    px, lab = scale_bar(10.0, 640, info, 'comoving', perspective=True)
    assert lab == "2 Mpc/h comoving, at view centre" and px == 128
    px, lab = scale_bar(60.0, 600, LengthInfo(), 'comoving', perspective=False)
    assert lab == "10 kpc" and px == 100            # non-comoving run: just the unit


# ---------------------------------------------------------------- whole movies

def _run_scenes(n_snap=3, n=400, box=10.0):
    rng = np.random.default_rng(6)
    ids = np.arange(1, n + 1, dtype=np.uint64)
    x0 = rng.uniform(0, box, (n, 3))
    v = np.tile([1.0, 0.0, 0.0], (n, 1))
    out = []
    for i in range(n_snap):
        perm = rng.permutation(n)
        pos = np.mod(x0 + v * i, box)[perm]
        sc = _scene(pos, ids[perm], v[perm], float(i), box, hl=np.arange(10), label=f"t = {i}")
        sc.box_min, sc.box_size = np.zeros(3), np.full(3, box)
        out.append(sc)
    return out


def test_evolving_in_between_frames_keep_snapshot_frames_exact(app, tmp_path):
    scenes = _run_scenes()
    calls = []

    def scene_for(i):
        calls.append(i)
        return scenes[i]
    cam = Camera(center=(5, 5, 5), distance=25)
    names = [f"s{i}.hdf5" for i in range(3)]
    plain = MovieSettings(kind='evolving', recentre=False, width=200, height=150, basename=str(tmp_path / 'a' / 'f'))
    r0 = make_movie(plain, cam, 0, scene_for, names)
    s = MovieSettings(kind='evolving', recentre=False, interp_frames=2, width=200, height=150,
                      basename=str(tmp_path / 'b' / 'f'), scale_bar=True)
    r = make_movie(s, cam, 0, scene_for, names)
    assert r0.success and r.success and len(r0.frames) == 3 and len(r.frames) == 7
    meta = json.load(open(r.params_file))
    flags = [f['interpolated'] for f in meta['frames']]
    assert flags == [False, True, True, False, True, True, False]
    assert [f['position'] for f in meta['frames']] == pytest.approx([0, 1 / 3, 2 / 3, 1, 4 / 3, 5 / 3, 2])
    assert meta['interpolated_frames'] == 4 and meta['particles_dropped_in_between'] == 0
    assert all('interpolated' in f['label'] for f in meta['frames'] if f['interpolated'])
    from PyQt6.QtGui import QImage

    def pixels(p):
        im = QImage(p).convertToFormat(QImage.Format.Format_RGB888)
        return np.frombuffer(im.constBits().asstring(im.sizeInBytes()), np.uint8).copy()
    # Frames at the snapshots are the snapshots themselves (the scale bar differs, the label does not).
    s2 = MovieSettings(kind='evolving', recentre=False, interp_frames=2, width=200, height=150,
                       basename=str(tmp_path / 'c' / 'f'))
    r2 = make_movie(s2, cam, 0, scene_for, names)
    for k0, k in ((0, 0), (1, 3), (2, 6)):
        assert np.array_equal(pixels(r0.frames[k0]), pixels(r2.frames[k]))
    assert max(calls) == 2 and len(set(calls)) == 3


def test_keyframe_movie_with_time_and_follow(app, tmp_path):
    scenes = _run_scenes()
    kfs = [Keyframe({'center': (5, 5, 5), 'distance': 30, 'elevation': 30, 'azimuth': 0, 'fov': 60}, 0).to_dict(),
           Keyframe({'center': (5, 5, 5), 'distance': 15, 'elevation': 10, 'azimuth': 90, 'fov': 60}, 2).to_dict()]
    s = MovieSettings(kind='keyframes', keyframes=kfs, nframes=9, ease=False, recentre=False,
                      width=160, height=120, basename=str(tmp_path / 'k' / 'f'))
    r = make_movie(s, Camera(), 0, lambda i: scenes[i], ['a', 'b', 'c'])
    assert r.success, r.message
    meta = json.load(open(r.params_file))
    assert [f['position'] for f in meta['frames']] == pytest.approx(np.linspace(0, 2, 9))
    assert meta['frames'][4]['interpolated'] is False and meta['frames'][1]['interpolated'] is True
    assert meta['settings']['keyframes'] == kfs
    # Without interpolation, fractional positions show the nearest snapshot.
    s.interpolate = False
    s.basename = str(tmp_path / 'k2' / 'f')
    r = make_movie(s, Camera(), 0, lambda i: scenes[i], ['a', 'b', 'c'])
    meta = json.load(open(r.params_file))
    assert meta['interpolated_frames'] == 0
    assert [f['snapshot_index'] for f in meta['frames']] == [0, 0, 1, 1, 1, 1, 2, 2, 2]   # nearest, halves forward


def test_smoothed_movie_keeps_one_colour_range(app, tmp_path):
    scenes = _run_scenes()
    s = MovieSettings(kind='evolving', recentre=False, style='smoothed', colour_bar=True, scale_bar=True,
                      width=160, height=120, basename=str(tmp_path / 's' / 'f'))
    r = make_movie(s, Camera(center=(5, 5, 5), distance=25), 0, lambda i: scenes[i], ['a', 'b', 'c'])
    assert r.success, r.message
    meta = json.load(open(r.params_file))
    assert meta['smooth_range_used'] == r.smooth_range and r.smooth_range[1] > r.smooth_range[0] > 0
    s.smooth_range = (1e-3, 1.0)
    s.basename = str(tmp_path / 's2' / 'f')
    r = make_movie(s, Camera(center=(5, 5, 5), distance=25), 0, lambda i: scenes[i], ['a', 'b', 'c'])
    assert r.smooth_range == [1e-3, 1.0]


@pytest.mark.parametrize("suffix", [".gif", ".mp4"])
def test_quick_movie_writes_only_the_movie(app, tmp_path, suffix):
    if suffix == ".mp4" and not can_write_mp4():
        pytest.skip("no MP4 encoder (ffmpeg or OpenCV) installed")
    scenes = _run_scenes()
    out = tmp_path / "out" / f"quick{suffix}"
    s = MovieSettings(kind='rotating', nframes=12, width=160, height=120, keep_frames=False,
                      output_file=str(out), fps=12)
    r = make_movie(s, Camera(center=(5, 5, 5), distance=25), 1, lambda i: scenes[i], ['a', 'b', 'c'])
    assert r.success, r.message
    assert out.exists() and out.stat().st_size > 1000
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]   # no frames, no JSON
    assert r.frames == [] and r.outputs == [str(out)]
    bad = MovieSettings(keep_frames=False, output_file=str(tmp_path / "x.avi"))
    assert "mp4 or .gif" in bad.validate(3)


def test_unreadable_snapshot_skips_its_in_between_frames(app, tmp_path):
    scenes = _run_scenes(n_snap=4)

    def scene_for(i):
        if i == 2:
            raise OSError("truncated file")
        return scenes[i]
    s = MovieSettings(kind='evolving', recentre=False, interp_frames=1, width=120, height=100,
                      basename=str(tmp_path / 'u' / 'f'))
    r = make_movie(s, Camera(center=(5, 5, 5), distance=25), 0, scene_for, ['a', 'b', 'c', 'd'])
    assert r.success and len(r.skipped) == 1 and "truncated" in r.skipped[0]
    meta = json.load(open(r.params_file))
    assert [f['position'] for f in meta['frames']] == pytest.approx([0, 0.5, 1, 3])


def test_smoothed_overlays_line_up_with_the_image(app):
    """Box, trails and selected particles on smoothed frames use the image's orthographic projection."""
    from movie.maker import _orthographic
    from movie.smoothed import density_image
    cam = Camera(center=(1, 2, 3), distance=10, elevation=35, azimuth=-70, fov=60)
    w, h = 400, 300
    probe = np.array([[4.0, 0.5, 5.0]])               # well off-centre, where perspective differs
    rng = np.random.default_rng(8)
    pts = probe + rng.normal(0, 0.01, (500, 3))
    img = density_image(Scene(pos=pts, colors=np.ones((500, 4)), mass=np.ones(500)), cam, w, h)
    r, c = np.unravel_index(np.argmax(img), img.shape)
    px, py, _, _ = project(probe, _orthographic(cam), w, h)
    assert abs(px[0] - (c + 0.5)) < 1.5 and abs(py[0] - (r + 0.5)) < 1.5
    ppx, ppy, _, _ = project(probe, cam, w, h)          # the perspective view puts it elsewhere
    assert np.hypot(ppx[0] - px[0], ppy[0] - py[0]) > 5


def test_long_movie_keeps_few_scenes_in_memory(app, tmp_path):
    """Lazy reading: however many snapshots, only a few scenes are alive at once."""
    import gc
    import weakref
    alive = []

    def scene_for(i):
        sc = _scene(np.random.default_rng(i).uniform(0, 1, (50, 3)), np.arange(1, 51, dtype=np.uint64),
                    np.zeros((50, 3)), float(i))
        alive.append(weakref.ref(sc))
        gc.collect()
        assert sum(r() is not None for r in alive) <= 4
        return sc
    s = MovieSettings(kind='evolving', recentre=False, interp_frames=1, width=100, height=100,
                      basename=str(tmp_path / 'long' / 'f'))
    r = make_movie(s, Camera(center=(0.5, 0.5, 0.5), distance=3), 0, scene_for, [f"s{i}" for i in range(40)])
    assert r.success and len(r.frames) == 79 and len(alive) == 40


def test_tangents_limited_when_snapshots_are_far_apart():
    """A particle on a circular orbit sampled once per orbit must not be flung out between snapshots."""
    from movie.interpolate import MAX_TANGENT
    ids = np.array([1, 2], dtype=np.uint64)
    p0 = np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    v = np.array([[0.0, 2 * np.pi, 0.0], [0.0, 0.0, 0.0]])    # period 1; the second does not move...
    p1 = p0 + np.array([[1e-3, 0, 0], [1.0, 0, 0]])            # ...except 1 unit in x
    a = _scene(p0, ids, v, 0.0)
    b = _scene(p1, ids, v, 1.0)
    pair = PairInterpolator(a, b)
    assert pair.limited == 1 and pair.n_curves == 2
    for f in np.linspace(0.05, 0.95, 19):
        x = pair.scene_at(f).pos
        assert np.linalg.norm(x[0] - p0[0]) < MAX_TANGENT * 1e-3      # stays put, not a 2 pi swing
    assert np.allclose(pair.scene_at(0.5).pos[1], [5.5, 0, 0])          # unaffected: linear-consistent


def test_smoothed_image_in_a_periodic_box_lines_up_with_the_box(app):
    """
    The smoothed image is drawn where the box outline, trails and selected particles are:
    at the stored positions, not wrapped into the periodic image around the view centre.
    Before the fix a clump near one face of the box showed up beyond the opposite face
    whenever the view centre was more than half a box away, and the cube outline sat off
    the picture by the shift (the halo movie in assets/movies).
    """
    from movie.maker import _orthographic
    from movie.renderer import box_lines
    from movie.smoothed import density_image
    box = 10.0
    cam = Camera(center=(8.0, 5.0, 5.0), distance=30, elevation=20, azimuth=-60, fov=60)
    w, h = 400, 300
    probe = np.array([[1.0, 5.0, 5.0]])                 # 7 from the centre along x: wraps to x = 11
    rng = np.random.default_rng(9)
    pts = probe + rng.normal(0, 0.01, (500, 3))
    sc = Scene(pos=pts, colors=np.ones((500, 4)), mass=np.ones(500), box_period=box,
               box_min=np.zeros(3), box_size=np.full(3, box))
    img = density_image(sc, cam, w, h)
    r, c = np.unravel_index(np.argmax(img), img.shape)
    ortho = _orthographic(cam)
    px, py, _, _ = project(probe, ortho, w, h)
    assert abs(px[0] - (c + 0.5)) < 1.5 and abs(py[0] - (r + 0.5)) < 1.5
    # And inside the drawn outline: between the projected x = 0 and x = box faces.
    corners = box_lines(sc.box_min, sc.box_size)
    corners = corners[np.all(np.isfinite(corners), axis=1)]
    bx, by, _, _ = project(corners, ortho, w, h)
    assert bx.min() <= c + 0.5 <= bx.max() and by.min() <= r + 0.5 <= by.max()


@pytest.mark.parametrize("size", [(480, 480), (320, 240), (1280, 720)])
def test_colour_bar_title_fits_the_frame(app, size):
    """The smoothed-frame colour bar title is drawn whole: it used to be cut off on the left."""
    from PyQt6.QtGui import QFont, QFontMetrics
    from movie.overlays import title_font
    from movie.smoothed import TITLE
    w, h = size
    fs, margin = max(10, h // 32), max(6, w // 80)
    small = QFont()
    small.setPixelSize(max(9, int(fs * 0.8)))
    bw = max(120, w // 4)
    room = (w - margin - bw) + bw - margin             # from the left margin to the bar's right end
    f = title_font(small, TITLE + ' (log scale)', room)
    assert QFontMetrics(f).horizontalAdvance(TITLE + ' (log scale)') <= room
    assert f.pixelSize() >= 6
