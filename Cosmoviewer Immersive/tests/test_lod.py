"""
Level-of-detail display data (lod/).

The LOD code decides which particles are drawn; it must never change what a
particle is. These tests check that every drawn particle is a real particle
with its file values, that the sample is the documented one (u < f), that
levels are nested, that culling never drops a particle that is on screen,
and that the snap['rows'] mapping points at the right rows of the files.
Set COSMOVIEWER_DM_SNAPSHOT to a Gadget-4 snapshot to also run on real data.
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lod.store as store_mod  # noqa: E402
from lod.cells import level_for, level_fraction  # noqa: E402
from lod.frustum import camera_mvp, camera_planes, points_inside  # noqa: E402
from lod.store import build_store, id_uniforms  # noqa: E402
from lod.view import ViewRule, base_sample, eye_position, query_view, read_detail  # noqa: E402
from movie.renderer import view_matrix  # noqa: E402
from movie.renderer import Camera, project  # noqa: E402

DM_SNAPSHOT = os.environ.get("COSMOVIEWER_DM_SNAPSHOT")
BOX = 50.0


def _write_pieces(folder, nfiles=3, n_gas=30_000, n_dm=60_000, seed=1, sort=True):
    """Clustered multi-file snapshot. Returns the full data per (piece, type)."""
    rng = np.random.default_rng(seed)
    full = {}
    files = []
    for i in range(nfiles):
        p = Path(folder) / f"snap_000.{i}.hdf5"
        files.append(str(p))
        with h5py.File(p, "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = BOX
            h.attrs["NumFilesPerSnapshot"] = nfiles
            h.attrs["MassTable"] = np.array([0.0, 0.7, 0, 0, 0, 0])
            h.attrs["Time"] = 1.0
            for t, n in ((0, n_gas), (1, n_dm)):
                centres = rng.random((20, 3)) * BOX
                c = (centres[rng.integers(0, 20, n)] + rng.normal(0, 1.5, (n, 3))) % BOX
                if sort:                    # spatially coherent rows, like Peano-Hilbert ordered files
                    c = c[np.lexsort((c[:, 2], c[:, 1], np.floor(c[:, 0] / 5)))]
                g = f.create_group(f"PartType{t}")
                g.create_dataset("Coordinates", data=c, chunks=(4096, 3), compression="gzip")
                v = rng.normal(0, 100, (n, 3)).astype(np.float32)
                g.create_dataset("Velocities", data=v)
                ids = (np.arange(n) + 1_000_000 * (i * 6 + t)).astype(np.uint64)
                g.create_dataset("ParticleIDs", data=ids)
                if t == 0:
                    g.create_dataset("Masses", data=rng.random(n).astype(np.float32))
                full[(i, t)] = {"c": c, "v": v, "id": ids}
    return files, full


@pytest.fixture
def small_blocks(monkeypatch):
    monkeypatch.setattr(store_mod, "BLOCK_ROWS", 4096)
    monkeypatch.setattr(store_mod, "TASK_ROWS", 4096 * 4)
    import lod.view as view_mod
    monkeypatch.setattr(view_mod, "DETAIL_RUN_ROWS", 4096 * 4)


def _check_view_against_files(view, full):
    """Every drawn particle is the file's particle at its (piece, type, row)."""
    k = 0
    for (piece, t), rows in sorted(view["rows"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = len(rows)
        assert np.all(np.diff(rows) > 0)
        ref = full[(piece, t)]
        np.testing.assert_array_equal(view["id"][k:k + n], ref["id"][rows])
        np.testing.assert_array_equal(view["c"][k:k + n], ref["c"][rows].astype(np.float32))
        np.testing.assert_array_equal(view["v"][k:k + n], ref["v"][rows])
        assert np.all(view["t"][k:k + n] == t)
        k += n
    assert k == len(view["t"])


def _expected_rows(files, full, seed, f, planes=None, rule=None, above=None):
    """Brute force: rows with u < f (or u < rule's f, only where that exceeds ``above``), in the frustum.
    u is the hash of each particle's ParticleID."""
    out = {}
    for (piece, t), ref in full.items():
        u = id_uniforms(ref["id"], seed)
        if rule is not None:
            sel = (u < rule.f_points(ref["c"])) & rule.from_file(ref["c"], above)
        else:
            sel = u < f
        if planes is not None:
            sel &= points_inside(planes, ref["c"])
        if sel.any():
            out[(piece, t)] = np.flatnonzero(sel)
    return out


def test_full_store_holds_every_particle_once(tmp_path, small_blocks):
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=10**9, workers=1, use_cache=False)
    assert s.fraction == 1.0 and s.n_stored == sum(len(v["id"]) for v in full.values())
    view = s.subset(np.arange(s.n_stored))
    _check_view_against_files(view, full)
    assert len(np.unique(view["id"])) == s.n_stored


def test_sampled_store_is_exactly_u_below_fraction(tmp_path, small_blocks):
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=40_000, seed=7, workers=1, use_cache=False)
    view = s.subset(np.arange(s.n_stored))
    _check_view_against_files(view, full)
    expected = _expected_rows(files, full, 7, s.fraction)
    assert set(view["rows"]) == set(expected)
    for k in expected:
        np.testing.assert_array_equal(view["rows"][k], expected[k])


def test_levels_are_nested_and_uniform(tmp_path, small_blocks):
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=10**9, workers=1, use_cache=False)
    cells = np.arange(len(s.cells.keys))
    prev = None
    for level in range(8, -1, -1):
        rows = set(s.cells.take(cells, level_fraction(level)).tolist())
        if prev is not None:
            assert prev <= rows                 # a finer level keeps every coarser particle
        prev = rows
    # same fraction of every type and of every region (density preserved)
    view = base_sample(s, s.n_stored // 8)
    for t in (0, 1):
        n_t = sum(len(full[(i, t)]["id"]) for i in range(3))
        assert abs((view["t"] == t).sum() / n_t - 0.125) < 0.01
    h_all, _ = np.histogramdd(np.vstack([v["c"] for v in full.values()]), bins=4, range=[(0, BOX)] * 3)
    h_s, _ = np.histogramdd(view["c"], bins=4, range=[(0, BOX)] * 3)
    big = h_all > 2000
    assert np.all(np.abs(h_s[big] / h_all[big] - 0.125) < 0.03)


def test_rounding_keeps_expected_total(tmp_path, small_blocks):
    files, _ = _write_pieces(tmp_path)
    s = build_store(files, cap=10**9, workers=1, use_cache=False)
    cells = np.arange(len(s.cells.keys))
    for f in (0.5, 0.1, 0.01):
        n = len(s.cells.take(cells, f))
        assert abs(n - f * s.n_stored) < 4 * np.sqrt(len(cells)) + 2


@pytest.mark.parametrize("cam", [
    Camera(center=(25, 25, 25), distance=75, elevation=30, azimuth=45),
    Camera(center=(10, 12, 30), distance=8, elevation=-20, azimuth=170),
    Camera(center=(40, 5, 5), distance=3, elevation=80, azimuth=10, fov=40),
])
def test_culling_never_drops_an_on_screen_particle(tmp_path, small_blocks, cam):
    files, full = _write_pieces(tmp_path, sort=False)
    s = build_store(files, cap=10**9, workers=1, use_cache=False)
    w, h = 800, 500
    planes = camera_planes(cam, w, h)
    vis_rows = set(s.cells.take(s.cells.visible_cells(planes), 1.0).tolist())
    pos = np.asarray(s.arrays["pos"], dtype=np.float64)
    px, py, depth, ok = project(pos, cam, w, h)
    on_screen = ok & (px >= 0) & (px < w) & (py >= 0) & (py < h) & (depth < cam.distance * 1000)
    missing = np.flatnonzero(on_screen)[~np.isin(np.flatnonzero(on_screen), list(vis_rows))]
    assert len(missing) == 0
    # and the plane test agrees with the projection
    inside = points_inside(planes, pos)
    assert np.all(inside[on_screen])
    assert (inside & ~on_screen).sum() <= max(10, 1e-3 * on_screen.sum())   # edge pixels only


def _rule(s, cam, f_focus):
    ct = s.cells
    return ViewRule(f_focus, eye_position(view_matrix(cam)), cam.distance, ct.origin, ct.size, ct.depth)


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("falloff", [0.0, 2.0])
def test_file_detail_is_exactly_the_documented_sample(tmp_path, small_blocks, sort, falloff):
    files, full = _write_pieces(tmp_path, sort=sort)
    s = build_store(files, cap=20_000, seed=3, workers=1, use_cache=False)
    cam = Camera(center=(20, 20, 20), distance=6, elevation=10, azimuth=60)
    planes = camera_planes(cam, 800, 500)
    rule = _rule(s, cam, level_fraction(2))
    rule.falloff = falloff
    assert rule.f_focus > s.fraction
    res = read_detail(s, planes, rule)
    _check_view_against_files(res.data, full)
    par = read_detail(s, planes, rule, parallel=True)      # worker processes: same particles, same order
    for k in ("id", "c", "t"):
        np.testing.assert_array_equal(par.data[k], res.data[k])
    expected = _expected_rows(files, full, 3, None, planes, rule=rule, above=s.fraction)
    assert set(res.data["rows"]) == set(expected)
    for k in expected:
        np.testing.assert_array_equal(res.data["rows"][k], expected[k])
    # stored particles in view whose cell is read from the files are all in the detail (no popping)
    stored = s.subset(np.arange(s.n_stored))
    need = (points_inside(planes, stored["c"]) & rule.from_file(stored["c"], s.fraction)
            & (rule.f_points(stored["c"]) >= s.fraction))
    assert need.any() and np.isin(stored["id"][need], res.data["id"]).all()
    if sort:
        assert res.blocks_read < len(s.blocks["start"])     # coherent files: only some blocks read


def test_depth_falloff_concentrates_detail_at_the_focus(tmp_path, small_blocks):
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=10**9, workers=1, use_cache=False)
    densest = int(np.argmax(s.cells.counts))
    centre = s.cells.centres(np.array([densest]))[0]
    cam = Camera(center=tuple(centre), distance=3, elevation=10, azimuth=60)   # inside the densest cluster
    planes = camera_planes(cam, 800, 500)
    eye = eye_position(view_matrix(cam))
    flat = query_view(s, planes, 1_000, eye, cam.distance, falloff=0.0)
    deep = query_view(s, planes, 1_000, eye, cam.distance)
    assert deep.fraction > flat.fraction                  # sharper at the focus for the same budget
    assert deep.n_drawn <= 1_000 * 1.3 and flat.n_drawn <= 1_000 * 1.3
    near = lambda r: (np.linalg.norm(r.data["c"] - np.array(cam.center), axis=1) < cam.distance).sum()
    assert near(deep) > near(flat)


def test_query_respects_budget_and_switches_source(tmp_path, small_blocks):
    files, full = _write_pieces(tmp_path)
    n_all = sum(len(v["id"]) for v in full.values())
    s = build_store(files, cap=30_000, workers=1, use_cache=False)
    cam_far = Camera(center=(25, 25, 25), distance=80)
    far = camera_planes(cam_far, 800, 500)
    r = query_view(s, far, 5_000, eye_position(view_matrix(cam_far)), cam_far.distance)
    assert r.source == "store" and r.n_drawn <= 5_000 * 1.1
    assert abs(r.n_in_view - n_all) < 0.1 * n_all
    cam_near = Camera(center=(20, 20, 20), distance=4)
    near = camera_planes(cam_near, 800, 500)
    r2 = query_view(s, near, 5_000, eye_position(view_matrix(cam_near)), cam_near.distance)
    assert r2.fraction > r.fraction
    if r2.fraction > s.fraction:
        assert "file" in r2.source
    assert r2.n_drawn <= 5_000 * 1.3
    _check_view_against_files(r2.data, full)


def test_settled_camera_keeps_a_set_that_still_serves(tmp_path, small_blocks):
    """Small rotations reuse the drawn set; zooming or leaving the loaded region replaces it."""
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=30_000, workers=1, use_cache=False)

    def query(cam, previous=None):
        return query_view(s, camera_planes(cam, 800, 500, margin=0.15), 5_000, eye_position(view_matrix(cam)),
                          cam.distance, previous=previous, screen_planes=camera_planes(cam, 800, 500))

    cam = Camera(center=(20, 20, 20), distance=10, elevation=10, azimuth=60)
    first = query(cam)
    assert query(Camera(center=(20, 20, 20), distance=10, elevation=10, azimuth=61), first) is first
    # a big rotation moves the eye too far, a zoom changes the level, a pan leaves the region
    for other in (Camera(center=(20, 20, 20), distance=10, elevation=10, azimuth=120),
                  Camera(center=(20, 20, 20), distance=2, elevation=10, azimuth=60),
                  Camera(center=(20, 20, 20), distance=10, elevation=10, azimuth=60, fov=120)):
        r = query(other, first)
        assert r is not first
        fresh = query(other)
        np.testing.assert_array_equal(r.data["id"], fresh.data["id"])
        _check_view_against_files(r.data, full)


def test_idle_worker_pool_is_released_and_restarts(tmp_path, small_blocks):
    import lod.view as view_mod
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=30_000, workers=1, use_cache=False)
    cam = Camera(center=(20, 20, 20), distance=6, elevation=10, azimuth=60)
    planes = camera_planes(cam, 800, 500)
    rule = ViewRule(1.0, eye_position(view_matrix(cam)), cam.distance, s.cells.origin, s.cells.size, s.cells.depth)
    a = read_detail(s, planes, rule, parallel=True)
    assert view_mod._POOL is not None
    assert not view_mod.release_idle_pool(idle_seconds=3600)      # recently used: kept
    assert view_mod.release_idle_pool(idle_seconds=0) and view_mod._POOL is None
    b = read_detail(s, planes, rule, parallel=True)                 # starts again on demand
    np.testing.assert_array_equal(a.data["id"], b.data["id"])
    view_mod.release_idle_pool(idle_seconds=0)


def test_disk_cache_reopens_identically_and_detects_changes(tmp_path, small_blocks, monkeypatch):
    monkeypatch.setattr(store_mod, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "data").mkdir()
    files, full = _write_pieces(tmp_path / "data")
    a = build_store(files, cap=50_000, workers=1, use_cache=True)
    assert a.path is not None and not a.info["from_cache"]
    b = build_store(files, cap=50_000, workers=1, use_cache=True)
    assert b.info["from_cache"]
    for k in store_mod.FIELDS:
        np.testing.assert_array_equal(np.asarray(a.arrays[k]), np.asarray(b.arrays[k]))
    os.utime(files[1], ns=(1, 1))        # file changed: the cache must not be used
    c = build_store(files, cap=50_000, workers=1, use_cache=True)
    assert not c.info["from_cache"]


def test_frustum_matches_pyqtgraph_matrices():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    try:
        from PyQt6.QtWidgets import QApplication
        import pyqtgraph.opengl as gl
    except Exception as e:  # no OpenGL libraries on this machine
        pytest.skip(str(e))
    app = QApplication.instance() or QApplication([])
    w = gl.GLViewWidget()
    w.resize(800, 500)
    w.setCameraPosition(distance=12.0, elevation=25, azimuth=-70)
    w.opts["center"] = __import__("PyQt6.QtGui", fromlist=["QVector3D"]).QVector3D(1, 2, 3)
    o = w.opts
    cam = Camera(center=(1, 2, 3), distance=o["distance"], elevation=o["elevation"], azimuth=o["azimuth"],
                 fov=o["fov"])
    mine = camera_mvp(cam, w.width(), w.height())
    vp = (0, 0, w.width(), w.height())
    theirs = np.array((w.projectionMatrix(vp, vp) * w.viewMatrix()).data()).reshape(4, 4).T
    w.deleteLater()                # destroy now, not at a random garbage collection later
    app.processEvents()
    np.testing.assert_allclose(mine, theirs, rtol=1e-5, atol=1e-8)


def test_level_arithmetic():
    assert level_for(1_000, 2_000) == 0
    for n in (3_000_000, 50_000_000, 850_000_000):
        lv = level_for(n, 2_000_000)
        assert n * level_fraction(lv) <= 2_000_000 * (1 + 1e-9) < n * level_fraction(lv - 1)


@pytest.mark.skipif(not DM_SNAPSHOT, reason="set COSMOVIEWER_DM_SNAPSHOT to a Gadget-4 snapshot")
def test_real_snapshot_rows_point_at_file_rows():
    s = build_store([DM_SNAPSHOT], cap=300_000, workers=2, use_cache=False)
    view = base_sample(s, 50_000)
    with h5py.File(DM_SNAPSHOT, "r") as f:
        k = 0
        for (piece, t), rows in sorted(view["rows"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
            g = f[f"PartType{t}"]
            np.testing.assert_array_equal(view["id"][k:k + len(rows)], g["ParticleIDs"][:][rows])
            np.testing.assert_array_equal(view["c"][k:k + len(rows)], g["Coordinates"][:][rows].astype(np.float32))
            k += len(rows)


def _write_reshuffled(folder, full, seed=5, with_ids=True):
    """The same particles as ``full``, as the next snapshot of a run might write them: every type's
    particles shuffled across rows and files (Gadget writes in FoF group order)."""
    rng = np.random.default_rng(seed)
    pieces = sorted({p for p, _ in full})
    files = []
    split = {}
    for t in sorted({t for _, t in full}):
        cat = {k: np.concatenate([full[(p, t)][k] for p in pieces]) for k in ("c", "v", "id")}
        order = rng.permutation(len(cat["id"]))
        cuts = np.sort(rng.choice(np.arange(1, len(order)), len(pieces) - 1, replace=False))
        for p, part in zip(pieces, np.split(order, cuts)):
            split[(p, t)] = {k: v[part] for k, v in cat.items()}
    for p in pieces:
        path = Path(folder) / f"snap_001.{p}.hdf5"
        files.append(str(path))
        with h5py.File(path, "w") as f:
            h = f.create_group("Header")
            h.attrs["BoxSize"] = BOX
            h.attrs["NumFilesPerSnapshot"] = len(pieces)
            h.attrs["MassTable"] = np.array([0.0, 0.7, 0, 0, 0, 0])
            h.attrs["Time"] = 1.0
            for (q, t), d in split.items():
                if q == p:
                    g = f.create_group(f"PartType{t}")
                    g.create_dataset("Coordinates", data=d["c"])
                    g.create_dataset("Velocities", data=d["v"])
                    if with_ids:
                        g.create_dataset("ParticleIDs", data=d["id"])
                    g.create_dataset("Masses", data=np.ones(len(d["id"]), np.float32))
    return files, split


def test_same_particles_drawn_in_every_snapshot(tmp_path, small_blocks):
    """u follows the ParticleID: reordered files give the same store, base sample and view set."""
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    files_a, full = _write_pieces(tmp_path / "a")
    files_b, _ = _write_reshuffled(tmp_path / "b", full)
    sa = build_store(files_a, cap=40_000, seed=2, workers=1, use_cache=False)
    sb = build_store(files_b, cap=40_000, seed=2, workers=1, use_cache=False)
    ids = lambda d: set(np.asarray(d["id"]).tolist())
    assert ids(sa.arrays) == ids(sb.arrays)
    assert ids(base_sample(sa, 9_000)) == ids(base_sample(sb, 9_000))
    sources = set()
    for cam in (Camera(center=(25, 25, 25), distance=80), Camera(center=(20, 20, 20), distance=4)):
        planes = camera_planes(cam, 800, 500)
        eye = eye_position(view_matrix(cam))
        ra = query_view(sa, planes, 5_000, eye, cam.distance)
        rb = query_view(sb, planes, 5_000, eye, cam.distance)
        assert ra.level == rb.level and ra.source == rb.source
        sources.add(ra.source)
        assert ids(ra.data) == ids(rb.data) and ra.n_drawn > 300
    assert "store" in sources and any("file" in x for x in sources)   # store and file reads both covered


def test_store_below_is_u_below_f_in_every_cell(tmp_path, small_blocks):
    files, _ = _write_pieces(tmp_path)
    s = build_store(files, cap=60_000, workers=1, use_cache=False)
    rng = np.random.default_rng(0)
    cells = np.sort(rng.choice(len(s.cells.keys), len(s.cells.keys) // 2, replace=False))
    f = rng.random(len(cells)) * s.fraction * 1.2
    u = np.asarray(s.arrays["u"])
    cell_of = np.repeat(np.arange(len(s.cells.keys)), s.cells.counts)
    fmap = np.full(len(s.cells.keys), -1.0)
    fmap[cells] = f
    expected = np.flatnonzero(u < fmap[cell_of])
    np.testing.assert_array_equal(s.below(cells, f), expected)
    nested = s.below(cells, f * 0.5)
    assert np.isin(nested, expected).all()


def test_files_without_ids_still_sample_uniformly(tmp_path, small_blocks):
    """No ParticleIDs: u falls back to (seed, piece, type, block); ids are row numbers."""
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    _, full = _write_pieces(tmp_path / "a")
    files, split = _write_reshuffled(tmp_path / "b", full, with_ids=False)
    s = build_store(files, cap=40_000, seed=4, workers=1, use_cache=False)
    view = s.subset(np.arange(s.n_stored))
    for (p, t), r in view["rows"].items():
        n = len(split[(p, t)]["id"])
        u = np.concatenate([store_mod.block_uniforms(4, p, t, b, min(store_mod.BLOCK_ROWS, n - b))
                            for b in range(0, n, store_mod.BLOCK_ROWS)])
        np.testing.assert_array_equal(r, np.flatnonzero(u < s.fraction))
    assert abs(s.n_stored / sum(len(d["id"]) for d in split.values()) - s.fraction) < 0.01


def test_id_hash_is_uniform_and_seeded():
    sparse = (np.arange(200_000, dtype=np.uint64) * np.uint64(7_919_000_003)) + np.uint64(5_000_000_000)
    for ids in (np.arange(200_000, dtype=np.uint64), sparse):
        u = id_uniforms(ids, 0)
        assert u.dtype == np.float32 and u.min() >= 0 and u.max() < 1
        hist = np.histogram(u, bins=20, range=(0, 1))[0]
        assert np.all(np.abs(hist / len(u) - 0.05) < 0.003)
    assert not np.array_equal(id_uniforms(sparse, 0), id_uniforms(sparse, 1))
    np.testing.assert_array_equal(id_uniforms(sparse[::-1], 0), id_uniforms(sparse, 0)[::-1])


def test_deep_zoom_uses_the_budget_behind_the_focus(tmp_path, small_blocks):
    """At full resolution with budget to spare, the focus radius grows: the scene behind stays."""
    files, full = _write_pieces(tmp_path)
    s = build_store(files, cap=30_000, workers=1, use_cache=False)
    n_all = sum(len(v["id"]) for v in full.values())
    densest = s.cells.centres(np.array([int(np.argmax(s.cells.counts))]))[0]
    for budget in (5_000, 10**9):
        cam = Camera(center=tuple(densest), distance=0.3, elevation=10, azimuth=60)
        planes = camera_planes(cam, 800, 500)
        eye = eye_position(view_matrix(cam))
        r = query_view(s, planes, budget, eye, cam.distance)
        assert r.level == 0 and r.n_drawn <= budget * 1.1
        rule = r.info["rule"]
        assert rule.wide > cam.distance and rule.wide_cap == s.fraction   # background from the store
        in_view = sum(points_inside(planes, v["c"].astype(np.float32).astype(np.float64)).sum()
                      for v in full.values())                       # positions as stored (float32)
        # the whole view at the store's density at least, as far as the budget allows
        assert r.n_drawn >= 0.9 * min(0.5 * budget, s.fraction * in_view)
        _check_view_against_files(r.data, full)
