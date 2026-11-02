"""
Tests for particle tracking across snapshots and movie making.

Run from the "Cosmoviewer Immersive" folder:
    QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_tracking_and_movies.py

Real-data tests run when the environment variables point at the outputs:
    COSMOVIEWER_G4_OUTPUT   DM-L50-N128/output (snapshot_00[0-2].hdf5 + fof_tab_00[0-2].hdf5)
    COSMOVIEWER_CG_OUTPUT   CollidingGalaxiesSFR/output (snapshot_100..102.hdf5)
"""

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from snapshot_io.id_lookup import (locate_ids, select_in_sphere, physical_factor,  # noqa: E402
                                   scale_factor)
from model.tracking import (periodic_mean, sphere_wraps, unwrap_for_display, unwrap_series, smooth_centres,  # noqa: E402
                            build_trajectories, export_npz, export_csv, parse_id_text,
                            read_id_file, write_id_file)
from movie.renderer import Camera, view_matrix, project, render_frame  # noqa: E402
from movie.maker import MovieSettings, Scene, make_movie  # noqa: E402

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
CG_OUTPUT = os.environ.get("COSMOVIEWER_CG_OUTPUT")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _write_snapshot(path, parts, box=0.0, time=1.0, mass_table=None, nfiles=1):
    """parts: {ptype: (ids, coords[, masses])}."""
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = box
        h.attrs["Time"] = time
        h.attrs["Redshift"] = 1.0 / time - 1.0
        h.attrs["NumFilesPerSnapshot"] = nfiles
        h.attrs["MassTable"] = np.array(mass_table if mass_table is not None else [0.0] * 6)
        p = f.create_group("Parameters")
        p.attrs["HubbleParam"] = 0.7
        p.attrs["ComovingIntegrationOn"] = 1 if box > 0 else 0
        for t, spec in parts.items():
            g = f.create_group(f"PartType{t}")
            ids, c = spec[0], spec[1]
            g["ParticleIDs"] = np.asarray(ids, dtype=np.uint64)
            g["Coordinates"] = np.asarray(c, dtype=np.float32)
            g["Coordinates"].attrs["a_scaling"] = 1.0
            g["Coordinates"].attrs["h_scaling"] = -1.0
            g["Velocities"] = np.asarray(c, dtype=np.float32) * 10
            g["Velocities"].attrs["a_scaling"] = 0.5
            g["Velocities"].attrs["h_scaling"] = 0.0
            if len(spec) > 2:
                g["Masses"] = np.asarray(spec[2], dtype=np.float32)


# --------------------------------------------------------------------------- lookup by ID

def test_locate_ids_matches_brute_force_across_types_files_and_chunks(tmp_path):
    rng = np.random.default_rng(1)
    files = []
    all_ids, all_c, all_t, all_m = [], [], [], []
    next_id = 0
    for i in range(3):
        parts = {}
        for t in (0, 1, 4):
            n = int(rng.integers(50, 300))
            ids = rng.permutation(np.arange(next_id, next_id + n)) + 1000
            next_id += n
            c = rng.random((n, 3)) * 10
            m = rng.random(n) if t != 1 else None
            parts[t] = (ids, c) if m is None else (ids, c, m)
            all_ids.append(ids); all_c.append(c.astype(np.float32)); all_t.append(np.full(n, t))
            all_m.append(m.astype(np.float32) if m is not None else np.full(n, 0.25))
        p = tmp_path / f"snap_000.{i}.hdf5"
        _write_snapshot(p, parts, mass_table=[0, 0.25, 0, 0, 0, 0], nfiles=3)
        files.append(str(p))
    ids = np.concatenate(all_ids); c = np.concatenate(all_c)
    t = np.concatenate(all_t); m = np.concatenate(all_m)

    want = np.concatenate([rng.choice(ids, 200, replace=False), [5, 999_999]])   # 2 missing
    loc = locate_ids(files, want, chunk_rows=37)
    assert loc.n_found == 200
    assert not loc.found[np.isin(loc.ids, [5, 999_999])].any()
    order = {int(v): k for k, v in enumerate(ids)}
    for k in np.nonzero(loc.found)[0]:
        j = order[int(loc.ids[k])]
        np.testing.assert_array_equal(loc.pos[k], c[j])
        assert loc.types[k] == t[j]
        assert loc.mass[k] == pytest.approx(m[j])
        np.testing.assert_allclose(loc.vel[k], c[j] * 10, rtol=1e-6)
    assert np.all(np.isnan(loc.pos[~loc.found]))
    assert loc.duplicates == 0
    assert loc.scaling["Coordinates"] == {"a_scaling": 1.0, "h_scaling": -1.0}


def test_locate_ids_reports_duplicates_and_keeps_first(tmp_path):
    p = tmp_path / "s.hdf5"
    _write_snapshot(p, {0: ([7, 8, 7], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 1: ([8], [[4, 4, 4]])})
    loc = locate_ids([str(p)], [7, 8])
    assert loc.n_found == 2
    assert loc.duplicates == 2
    np.testing.assert_array_equal(loc.pos, [[1, 1, 1], [2, 2, 2]])


def test_type_change_is_followed(tmp_path):
    """A gas particle that becomes a star keeps its ID; the lookup reports the new type."""
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    _write_snapshot(a, {0: ([1, 2, 3], np.eye(3)), 4: ([10], [[0, 0, 0]])}, time=1.0)
    _write_snapshot(b, {0: ([1, 3], np.eye(3)[[0, 2]] + 1), 4: ([10, 2], [[0, 0, 0], [5, 5, 5]])}, time=2.0)
    la, lb = locate_ids([str(a)], [2, 3]), locate_ids([str(b)], [2, 3])
    assert list(la.types) == [0, 0] and list(lb.types) == [4, 0]
    tr = build_trajectories([la, lb], [str(a), str(b)])
    assert tr.types.tolist() == [[0, 0], [4, 0]]
    np.testing.assert_array_equal(tr.pos[1, 0], [5, 5, 5])


def test_select_in_sphere_periodic_matches_brute_force(tmp_path):
    rng = np.random.default_rng(2)
    box = 20.0
    c = rng.random((3000, 3)) * box
    ids = np.arange(3000) + 1
    p = tmp_path / "s.hdf5"
    _write_snapshot(p, {1: (ids, c)}, box=box)
    centre, r = np.array([0.5, 19.5, 10.0]), 3.0
    got = np.sort(select_in_sphere([str(p)], centre, r, box, chunk_rows=101))
    d = c.astype(np.float32).astype(np.float64) - centre
    d -= box * np.round(d / box)
    want = ids[np.einsum("ij,ij->i", d, d) <= r * r]
    np.testing.assert_array_equal(got, want)
    assert len(want) > 10                       # the sphere straddles a corner


def test_sphere_cut_by_a_face_is_drawn_in_one_piece_with_the_same_members():
    # The reported case: r = 10 about (38.3, 4.741, 29.13) in a 50 box reaches through y = 0,
    # so its members sit at y in [0, 14.7] and [44.7, 50) and used to be drawn as two pieces.
    rng = np.random.default_rng(7)
    box, centre, r = 50.0, np.array([38.3, 4.741, 29.13]), 10.0
    c = rng.random((200_000, 3)) * box
    d = c - centre
    d -= box * np.round(d / box)
    members = c[np.einsum("ij,ij->i", d, d) <= r * r]
    assert members[:, 1].max() > 44 and members[:, 1].min() < 1      # stored: split across y
    assert sphere_wraps(centre, r, box) == ["y"]
    drawn = unwrap_for_display(members, box)
    assert drawn.shape == members.shape
    np.testing.assert_allclose(np.mod(drawn, box), np.mod(members, box), atol=1e-9)  # same points
    np.testing.assert_allclose(np.linalg.norm(drawn - centre, axis=1).max(), r, atol=0.05)
    assert drawn[:, 1].min() > -5.3 and drawn[:, 1].max() < 14.8     # one ball poking out of y = 0
    np.testing.assert_array_equal(members, c[np.einsum("ij,ij->i", d, d) <= r * r])  # input untouched


def test_unwrap_for_display_leaves_box_filling_and_open_sets_alone():
    rng = np.random.default_rng(8)
    box = 10.0
    every = rng.random((5000, 3)) * box
    np.testing.assert_array_equal(unwrap_for_display(every, box), every)
    np.testing.assert_array_equal(unwrap_for_display(every, 0.0), every)  # no periodic box
    inside = rng.random((500, 3)) * 2 + 4                               # away from every face
    np.testing.assert_allclose(unwrap_for_display(inside, box), inside)
    assert sphere_wraps([5, 5, 5], 2, box) == [] and sphere_wraps([5, 5, 5], 2, 0.0) == []
    assert sphere_wraps([9.5, 0.5, 5], 1, box) == ["x", "y"]
    assert sphere_wraps([12, 5, 5], 1, box) == []                       # centre outside the box: wrapped


# --------------------------------------------------------------------------- centres and paths

def test_periodic_mean_across_the_boundary():
    box = 50.0
    pts = np.array([[49.9, 10, 10], [0.1, 10, 10], [49.8, 10, 10], [0.2, 10, 10]])
    c = periodic_mean(pts, box)
    assert min(abs(c[0]), abs(c[0] - box)) < 1e-9
    np.testing.assert_allclose(c[1:], [10, 10])
    assert periodic_mean(np.full((2, 3), np.nan), box) is None
    np.testing.assert_allclose(periodic_mean(pts), pts.mean(axis=0))   # open boundaries


def test_unwrap_series_removes_jumps_and_keeps_gaps():
    box = 10.0
    x = np.array([9.0, 9.8, 0.5, 1.5, np.nan, 2.5])     # crosses the boundary once
    pos = np.stack([x, np.zeros(6), np.zeros(6)], axis=1)[:, None, :]
    u = unwrap_series(pos, box)[:, 0, 0]
    np.testing.assert_allclose(u[[0, 1, 2, 3, 5]], [9.0, 9.8, 10.5, 11.5, 12.5])
    assert np.isnan(u[4])


def test_smooth_centres_like_gadgetviewer_and_periodic():
    cs = [np.array([float(i), 0, 0]) for i in range(5)]
    np.testing.assert_allclose(smooth_centres(cs, 4, 3), [3, 0, 0])          # mean of 2, 3, 4
    np.testing.assert_allclose(smooth_centres(cs, 1, 10), [0.5, 0, 0])       # fewer than n
    cs[3] = None
    np.testing.assert_allclose(smooth_centres(cs, 4, 3), [7 / 3, 0, 0])      # skips missing
    box = 10.0
    pc = [np.array([9.5, 0, 0]), np.array([0.5, 0, 0])]
    c = smooth_centres(pc, 1, 2, box)
    assert min(abs(c[0]), abs(c[0] - box)) < 1e-9                            # not 5.0


def test_physical_conversion_convention():
    assert physical_factor({"a_scaling": 1.0, "h_scaling": -1.0}, 0.25, 0.678) == pytest.approx(0.25 / 0.678)
    assert physical_factor({"a_scaling": 0.5, "h_scaling": 0.0}, 0.25, 0.678) == pytest.approx(0.5)
    assert physical_factor({}, 1, 1) is None
    assert scale_factor({"Time": 0.5, "BoxSize": 50}, {"ComovingIntegrationOn": 1}) == 0.5
    assert scale_factor({"Time": 1.2, "BoxSize": 0}, {"ComovingIntegrationOn": 0}) == 1.0


def test_trajectory_export_roundtrip(tmp_path):
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    _write_snapshot(a, {1: ([1, 2], [[49.0, 1, 1], [3, 3, 3]])}, box=50.0, time=0.5)
    _write_snapshot(b, {1: ([1, 2], [[1.0, 1, 1], [4, 4, 4]])}, box=50.0, time=1.0)
    tr = build_trajectories([locate_ids([str(a)], [1, 2]), locate_ids([str(b)], [1, 2])], [str(a), str(b)])
    npz = np.load(export_npz(tr, str(tmp_path / "t.npz")))
    np.testing.assert_allclose(npz["pos_unwrapped"][:, 0, 0], [49.0, 51.0])
    np.testing.assert_allclose(npz["pos_physical"][0, 1], np.array([3, 3, 3]) * 0.5 / 0.7, rtol=1e-6)
    np.testing.assert_allclose(npz["vel_physical"][1, 1], np.array([40, 40, 40]) * 1.0, rtol=1e-6)
    meta = json.loads(str(npz["metadata"]))
    assert meta["box_size"] == 50.0
    lines = open(export_csv(tr, str(tmp_path / "t.csv"))).read().splitlines()
    assert lines[0].startswith("# ") and lines[1].startswith("snapshot,file,time")
    assert len(lines) == 2 + 4


def test_id_text_and_files(tmp_path):
    np.testing.assert_array_equal(parse_id_text("3, 7-9 12"), [3, 7, 8, 9, 12])
    with pytest.raises(ValueError):
        parse_id_text("9-3")
    with pytest.raises(ValueError):
        parse_id_text("abc")
    p = write_id_file(str(tmp_path / "ids.txt"), {5, 1, 3}, header="test")
    np.testing.assert_array_equal(read_id_file(p), [1, 3, 5])


# --------------------------------------------------------------------------- renderer

def test_view_matrix_equals_pyqtgraph(qapp):
    import pyqtgraph.opengl as gl
    from PyQt6.QtGui import QVector3D
    w = gl.GLViewWidget()
    rng = np.random.default_rng(3)
    for _ in range(20):
        cam = Camera(tuple(rng.normal(0, 30, 3)), float(rng.uniform(1, 500)),
                     float(rng.uniform(-90, 90)), float(rng.uniform(-360, 360)))
        w.setCameraPosition(pos=QVector3D(*cam.center), distance=cam.distance,
                            elevation=cam.elevation, azimuth=cam.azimuth)
        ref = np.array(w.viewMatrix().data()).reshape(4, 4).T          # Qt is column-major
        np.testing.assert_allclose(view_matrix(cam), ref, atol=1e-4 * max(1.0, cam.distance))


def test_projection_centre_and_rotation():
    cam = Camera(center=(5, 5, 5), distance=20, elevation=0, azimuth=0)
    px, py, _, vis = project(np.array([[5, 5, 5]]), cam, 400, 300)
    assert vis[0] and px[0] == pytest.approx(200) and py[0] == pytest.approx(150)
    p = np.array([[5, 8, 5], [5, 5, 9]])
    px0, py0, _, _ = project(p, cam, 400, 300)
    px1, py1, _, _ = project(p, cam.rotated(180), 400, 300)
    np.testing.assert_allclose(px1, 400 - px0, atol=1e-6)            # mirror about the vertical
    np.testing.assert_allclose(py1, py0, atol=1e-6)
    assert py0[1] < 150                                              # +z is up on screen
    pxf, pyf, _, _ = project(p, cam.rotated(360), 400, 300)
    np.testing.assert_allclose(pxf, px0, atol=1e-6)


def test_render_additive_and_highlight():
    cam = Camera(center=(0, 0, 0), distance=10, elevation=0, azimuth=0)
    pts = np.zeros((4, 3))
    col = np.tile([0.1, 0.2, 0.3, 1.0], (4, 1))
    img = render_frame(100, 100, cam, pts, col, point_size=1)
    np.testing.assert_array_equal(img[50, 50], [102, 204, 255])      # 4 x (0.1, 0.2, 0.3), clipped
    img2 = render_frame(100, 100, cam, pts, col, highlight_pos=pts[:1],
                        highlight_colors=np.array([[1, 0, 0, 1.0]]), point_size=1, highlight_size=1)
    np.testing.assert_array_equal(img2[50, 50], [255, 0, 0])
    assert img.sum() == img[50, 50].astype(int).sum()                # nothing else drawn


# --------------------------------------------------------------------------- movies

def _scene_factory(centres, fail=()):
    def scene_for(i):
        if i in fail:
            raise OSError("unreadable")
        c = np.asarray(centres[i], dtype=float)
        return Scene(pos=c[None, :], colors=np.array([[1, 1, 1, 1.0]]), highlight_pos=c[None, :],
                     highlight_colors=np.array([[1, 0, 0, 1.0]]), centre=c, label=f"snap {i}")
    return scene_for


def test_rotating_movie(tmp_path, qapp):
    from PyQt6.QtGui import QImage
    s = MovieSettings(kind="rotating", nframes=8, angle=360, width=160, height=120,
                      basename=str(tmp_path / "rot" / "frame"), draw_box=False, draw_label=False)
    cam = Camera(center=(0, 0, 0), distance=10, elevation=0, azimuth=0)
    r = make_movie(s, cam, 0, _scene_factory([[0, 0, 0]]), ["s0"])
    assert r.success, r.message
    assert len(r.frames) == 8 and all(Path(f).exists() for f in r.frames)
    assert QImage(r.frames[0]).width() == 160
    meta = json.load(open(r.params_file))
    azs = [f["camera"]["azimuth"] for f in meta["frames"]]
    np.testing.assert_allclose(np.diff(azs), 45.0)


def test_evolving_movie_recentres_and_skips(tmp_path, qapp):
    from PyQt6.QtGui import QImage
    centres = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    s = MovieSettings(kind="evolving", first=0, last=3, n_smooth=0, width=120, height=100,
                      basename=str(tmp_path / "evo"), draw_box=False, draw_label=False)
    cam = Camera(center=(9, 9, 9), distance=10, elevation=0, azimuth=0)
    r = make_movie(s, cam, 0, _scene_factory(centres, fail={2}), ["a", "b", "c", "d"])
    assert r.success, r.message
    assert len(r.frames) == 3 and len(r.skipped) == 1
    for f in r.frames:                                   # selected particle drawn at frame centre
        img = QImage(f)
        assert img.pixelColor(60, 50).red() == 255 and img.pixelColor(60, 50).green() == 0
    meta = json.load(open(r.params_file))
    assert [f["snapshot"] for f in meta["frames"]] == ["a", "b", "d"]
    # Smoothing over 2 frames: camera centre is the mean of this and the previous centre.
    s2 = MovieSettings(kind="evolving", n_smooth=2, width=120, height=100,
                       basename=str(tmp_path / "evo2"), draw_box=False)
    r2 = make_movie(s2, cam, 0, _scene_factory(centres), ["a", "b", "c", "d"])
    xs = [f["camera"]["center"][0] for f in json.load(open(r2.params_file))["frames"]]
    np.testing.assert_allclose(xs, [0, 0.5, 1.5, 2.5])


def test_evolving_movie_fails_cleanly(tmp_path, qapp):
    s = MovieSettings(kind="evolving", ignore_unreadable=False, width=120, height=100,
                      basename=str(tmp_path / "x"))
    r = make_movie(s, Camera(), 0, _scene_factory([[0, 0, 0]] * 2, fail={1}), ["a", "b"])
    assert not r.success and "unreadable" in r.message
    assert MovieSettings(width=50).validate(3) is not None
    assert MovieSettings(kind="evolving", first=2, last=1).validate(3) is not None


def test_mp4_encoding_when_ffmpeg_is_available(tmp_path, qapp):
    from movie.maker import can_write_mp4
    if not can_write_mp4():
        pytest.skip("no MP4 encoder (ffmpeg or OpenCV) installed")
    s = MovieSettings(kind="rotating", nframes=5, width=161, height=121, encode_mp4=True,
                      basename=str(tmp_path / "m"))
    r = make_movie(s, Camera(), 0, _scene_factory([[0, 0, 0]]), ["s0"])
    assert r.success and r.mp4 and Path(r.mp4).stat().st_size > 0, r.message


def test_mp4_encoding_falls_back_to_opencv_without_ffmpeg(tmp_path, qapp, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import movie.maker as maker
    monkeypatch.setattr(maker, "find_ffmpeg", lambda: None)
    assert maker.can_write_mp4()
    s = MovieSettings(kind="rotating", nframes=7, width=161, height=121, encode_mp4=True,
                      basename=str(tmp_path / "m"), fps=10)
    r = make_movie(s, Camera(), 0, _scene_factory([[0, 0, 0]]), ["s0"])
    assert r.success and r.mp4, r.message
    cap = cv2.VideoCapture(r.mp4)
    n, shape = 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n, shape = n + 1, img.shape[:2]
    cap.release()
    assert n == 7 and shape == (122, 162)          # every frame, padded to even size like ffmpeg


# --------------------------------------------------------------------------- real Gadget-4 data

def _catalogue_group_ids(snap_path, tab_path, grp=0):
    with h5py.File(tab_path, "r") as f:
        off = int(f["Group/GroupOffsetType"][grp, 1])
        n = int(f["Group/GroupLenType"][grp, 1])
    with h5py.File(snap_path, "r") as f:
        return f["PartType1/ParticleIDs"][off:off + n].astype(np.uint64)


@pytest.mark.skipif(not G4_OUTPUT, reason="COSMOVIEWER_G4_OUTPUT not set")
def test_real_dm_track_largest_halo_back_in_time():
    """Largest z=0 FoF group (catalogue) located in all three snapshots, checked by direct read."""
    out = Path(G4_OUTPUT)
    ids = _catalogue_group_ids(out / "snapshot_002.hdf5", out / "fof_tab_002.hdf5")
    assert len(ids) == 34120
    located = []
    for s in ("000", "001", "002"):
        snap = out / f"snapshot_{s}.hdf5"
        loc = locate_ids([str(snap)], ids)
        assert loc.n_found == len(ids) and loc.duplicates == 0
        with h5py.File(snap, "r") as f:                   # independent check: full read + argsort
            all_ids = f["PartType1/ParticleIDs"][:].astype(np.uint64)
            rows = np.argsort(all_ids)[np.searchsorted(np.sort(all_ids), loc.ids)]
            np.testing.assert_array_equal(loc.pos, f["PartType1/Coordinates"][:][rows])
            assert np.all(loc.mass == pytest.approx(float(f["Header"].attrs["MassTable"][1])))
        located.append(loc)
    tr = build_trajectories(located, [str(out / f"snapshot_{s}.hdf5") for s in ("000", "001", "002")])
    # The halo's centre at z=0 agrees with the catalogue GroupPos.
    with h5py.File(out / "fof_tab_002.hdf5", "r") as f:
        gpos = f["Group/GroupPos"][0]
    c = periodic_mean(tr.pos[2], tr.box)
    d = c - gpos
    d -= tr.box * np.round(d / tr.box)
    assert np.max(np.abs(d)) < 1e-3
    # Physical conversion at a = 0.25: comoving Mpc/h * a / h.
    assert tr.pos_factor[0] == pytest.approx(tr.times[0] / 0.6777, rel=1e-3)
    # The progenitor region at z=3 is more extended (comoving) than the z=0 halo.
    def rms(p):
        d = p - periodic_mean(p, tr.box)
        d -= tr.box * np.round(d / tr.box)
        return np.sqrt((d ** 2).sum(axis=1).mean())
    assert rms(tr.pos[0]) > 2 * rms(tr.pos[2])


@pytest.mark.skipif(not CG_OUTPUT, reason="COSMOVIEWER_CG_OUTPUT not set")
def test_real_colliding_galaxies_gas_to_star():
    """Gas particles that form stars between snapshots 100 and 102 are followed into PartType4."""
    out = Path(CG_OUTPUT)
    files = [str(out / f"snapshot_{i}.hdf5") for i in (100, 101, 102)]
    with h5py.File(files[0], "r") as f:
        gas = f["PartType0/ParticleIDs"][:].astype(np.uint64)
    with h5py.File(files[2], "r") as f:
        stars = f["PartType4/ParticleIDs"][:].astype(np.uint64)
    converted = np.intersect1d(gas, stars)
    assert len(converted) > 0
    sample = np.concatenate([converted[:500], gas[:500]])
    locs = [locate_ids([p], sample) for p in files]
    tr = build_trajectories(locs, files)
    k = np.isin(tr.ids, converted[:500])
    assert np.all(tr.types[0, k] == 0) and np.all(tr.types[2, k] == 4)
    assert tr.pos_factor is not None and np.allclose(tr.pos_factor, 1.0)  # non-comoving: as stored
