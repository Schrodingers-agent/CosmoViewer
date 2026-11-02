"""
GadgetViewer README features: colour by any property, halo catalogues on
particles, selection by property and by catalogue group, Make plot layers,
additional data files, smoothed images, stereo.

Synthetic tests always run. Real-data tests run when COSMOVIEWER_G4_OUTPUT
points at a DM-L50-N128/output folder (any of its snapshots with their
fof_tab files) and COSMOVIEWER_CG_OUTPUT at CollidingGalaxiesSFR/output.
"""

import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from model import colour_tables as ct  # noqa: E402
from model.sph_projection import project  # noqa: E402
from snapshot_io import properties as P  # noqa: E402
from snapshot_io import group_catalogue as G  # noqa: E402
from snapshot_io import aux_data as A  # noqa: E402
from snapshot_io.snapshot_loader import load_hdf5_snapshot  # noqa: E402

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT")
CG_OUTPUT = os.environ.get("COSMOVIEWER_CG_OUTPUT")


# ----------------------------------------------------------------------------- synthetic data

def write_snapshot(path, n_gas=3000, n_dm=5000, box=10.0, seed=0, groups=None, ntypes_cols=2,
                   file_index=None, n_files=1, id_offset=0):
    """
    Gadget-4-like snapshot: gas with Density, InternalEnergy, ElectronAbundance,
    SmoothingLength, Masses; DM with masses only in MassTable. With ``groups``
    (list of DM group lengths) the DM rows are in group order and a fof_tab is
    written next to it with ``ntypes_cols`` columns per type (Gadget-4 NTYPES).
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, 'w') as f:
        h = f.create_group('Header')
        h.attrs['BoxSize'] = box
        h.attrs['Time'] = 1.0
        h.attrs['Redshift'] = 0.0
        h.attrs['MassTable'] = np.array([0, 0.5, 0, 0, 0, 0], dtype=np.float64)
        h.attrs['NumPart_ThisFile'] = np.array([n_gas, n_dm, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs['NumFilesPerSnapshot'] = n_files
        p = f.create_group('Parameters')
        p.attrs['UnitLength_in_cm'] = 3.085678e24
        p.attrs['UnitMass_in_g'] = 1.989e43
        p.attrs['UnitVelocity_in_cm_per_s'] = 1e5
        p.attrs['HubbleParam'] = 0.7
        g0 = f.create_group('PartType0')
        g0['Coordinates'] = rng.uniform(0, box, (n_gas, 3)).astype(np.float32)
        g0['Velocities'] = rng.normal(0, 100, (n_gas, 3)).astype(np.float32)
        g0['ParticleIDs'] = np.arange(1, n_gas + 1, dtype=np.uint64) + id_offset
        g0['Masses'] = np.full(n_gas, 0.1, dtype=np.float32)
        g0['Density'] = (10 ** rng.uniform(-3, 2, n_gas)).astype(np.float32)
        g0['Density'].attrs['a_scaling'] = -3.0
        g0['Density'].attrs['h_scaling'] = 2.0
        g0['InternalEnergy'] = (10 ** rng.uniform(1, 5, n_gas)).astype(np.float32)
        g0['InternalEnergy'].attrs['to_cgs'] = 1e10
        g0['ElectronAbundance'] = rng.uniform(0, 1.158, n_gas).astype(np.float32)
        g0['SmoothingLength'] = rng.uniform(0.05, 0.5, n_gas).astype(np.float32)
        g1 = f.create_group('PartType1')
        g1['Coordinates'] = rng.uniform(0, box, (n_dm, 3)).astype(np.float32)
        g1['Velocities'] = rng.normal(0, 100, (n_dm, 3)).astype(np.float32)
        g1['ParticleIDs'] = np.arange(10_000_001, 10_000_001 + n_dm, dtype=np.uint64) + id_offset
    if groups is not None:
        stem = Path(path).name.replace('snapshot', 'fof_tab')
        with h5py.File(Path(path).parent / stem, 'w') as f:
            ln = np.zeros((len(groups), ntypes_cols), dtype=np.int32)
            ln[:, 1] = groups
            off = np.zeros((len(groups), ntypes_cols), dtype=np.int64)
            off[1:, 1] = np.cumsum(groups)[:-1]
            f.create_group('Header').attrs['Ngroups_Total'] = len(groups)
            f['Header'].attrs['NumFiles'] = 1
            g = f.create_group('Group')
            g['GroupLenType'] = ln
            g['GroupOffsetType'] = off
            g['GroupLen'] = np.asarray(groups, dtype=np.int32)


def snap_for(path, max_particles=1000):
    s = load_hdf5_snapshot([str(path)], 0, max_particles)
    assert 'error' not in s, s.get('error')
    return s


# ----------------------------------------------------------------------------- properties

def test_list_properties_and_masses_from_masstable(tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn)
    props = P.list_properties([fn])
    for k in ('Density', 'InternalEnergy', 'ElectronAbundance', 'SmoothingLength', 'Temperature',
              'Masses', 'Velocities x', 'Velocities |v|'):
        assert k in props, k
    assert 'ParticleIDs' not in props and 'Coordinates |v|' not in props
    assert props['Masses'].types == (0, 1)
    assert props['Density'].types == (0,)
    assert "a^-3 h^2 for physical" in props['Density'].unit_label
    s = snap_for(fn)
    m = P.read_sample_values(s, props['Masses'])
    np.testing.assert_allclose(m, s['m'])
    np.testing.assert_allclose(m[s['t'] == 1], 0.5)


def test_sample_values_aligned_by_id(tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn)
    s = snap_for(fn, 1000)
    with h5py.File(fn) as f:
        rho = dict(zip(f['PartType0/ParticleIDs'][:].tolist(), f['PartType0/Density'][:].tolist()))
    vals = P.read_sample_values(s, P.list_properties([fn])['Density'])
    gas = s['t'] == 0
    np.testing.assert_array_equal(vals[gas], [rho[int(i)] for i in s['id'][gas]])
    assert np.all(np.isnan(vals[~gas]))


def test_wrong_sampling_falls_back_to_id_matching(tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn)
    s = snap_for(fn, 1000)
    good = P.read_sample_values(dict(s), P.list_properties([fn])['Density'])
    s2 = dict(s)
    s2['step'] = s['step'] + 1             # rows reconstructed from this would be wrong
    s2.pop('_props', None)
    np.testing.assert_array_equal(P.read_sample_values(s2, P.list_properties([fn])['Density']), good)


def test_rows_from_loader_random_sample(tmp_path):
    """The LOD loader will provide snap['rows'] for a random sample; values must follow it."""
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn, n_gas=3000, n_dm=5000)
    rng = np.random.default_rng(3)
    r0 = np.sort(rng.choice(3000, 700, replace=False))
    r1 = np.sort(rng.choice(5000, 900, replace=False))
    with h5py.File(fn) as f:
        ids = np.concatenate([f['PartType0/ParticleIDs'][:][r0], f['PartType1/ParticleIDs'][:][r1]])
        rho = f['PartType0/Density'][:][r0]
    s = {'files': [str(fn)], 'filepath': str(fn), 'rows': {(0, 0): r0, (0, 1): r1},
         'c': np.zeros((1600, 3)), 'id': ids, 't': np.r_[np.zeros(700, int), np.ones(900, int)]}
    vals = P.read_sample_values(s, P.list_properties([fn])['Density'])
    np.testing.assert_array_equal(vals[:700], rho)


def test_temperature_formula():
    # u = 100 (km/s)^2, x_e = 1.158 (fully ionised, X = 0.76): mu = 4/(1+3X+4X*1.158)
    X = 0.76
    mu = 4 / (1 + 3 * X + 4 * X * 1.158)
    expected = (2 / 3) * 100 * 1e10 * mu * 1.67262192e-24 / 1.380649e-16
    assert P.temperature_from_u(np.array([100.0]), np.array([1.158]), 1e10)[0] == pytest.approx(expected)
    # neutral gas, x_e = 0: mu = 4/(1+3X) ~ 1.22
    t0 = P.temperature_from_u(np.array([100.0]), np.array([0.0]), 1e10)[0]
    assert t0 / expected == pytest.approx((4 / (1 + 3 * X)) / mu)


def test_select_by_range_full_resolution(tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn)
    info = P.list_properties([fn])['Density']
    ids = P.select_by_range([fn], info, 0.1, 1.0, [0, 1], chunk_rows=777)
    with h5py.File(fn) as f:
        rho, pid = f['PartType0/Density'][:], f['PartType0/ParticleIDs'][:]
    np.testing.assert_array_equal(np.sort(ids), np.sort(pid[(rho >= 0.1) & (rho <= 1.0)]))


def test_read_values_for_ids(tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn)
    info = P.list_properties([fn])['Masses']
    want = np.array([5, 10_000_003, 2, 99_999_999], dtype=np.uint64)
    v = P.read_values_for_ids([fn], info, want, chunk_rows=1000)
    # aligned with np.unique(want) = [2, 5, 10000003, 99999999]
    np.testing.assert_allclose(v[:3], [0.1, 0.1, 0.5], rtol=1e-6)
    assert np.isnan(v[3])


# ----------------------------------------------------------------------------- colour tables

def test_colour_mapping():
    v = np.array([1.0, 10.0, 100.0, np.nan, -1.0])
    x = ct.normalise(v, 1.0, 100.0, True)
    np.testing.assert_allclose(x[:3], [0, 0.5, 1])
    assert np.isnan(x[3]) and np.isnan(x[4])
    rgba = ct.map_values(v, 1.0, 100.0, True, 'viridis')
    np.testing.assert_allclose(rgba[3], ct.MISSING_RGBA)
    np.testing.assert_allclose(rgba[0, :3], ct.lut('viridis')[0, :3])
    with pytest.raises(ValueError):
        ct.normalise(v, 0.0, 1.0, True)
    lo, hi = ct.auto_range(np.r_[np.arange(1000.0), np.nan], False)
    assert lo == pytest.approx(4.995) and hi == pytest.approx(994.005)


def test_random_colours_stable():
    a = ct.random_colours(np.array([0, 1, 2, 1, np.nan]), seed=3)
    np.testing.assert_allclose(a[0], ct.NO_GROUP_RGBA)
    np.testing.assert_allclose(a[4], ct.NO_GROUP_RGBA)
    np.testing.assert_allclose(a[1], a[3])
    assert not np.allclose(a[1], a[2])
    b = ct.random_colours(np.array([1]), seed=4)
    assert not np.allclose(a[1], b[0])          # Randomize changes the scheme


# ----------------------------------------------------------------------------- catalogues

def test_catalogue_membership_with_fewer_type_columns(tmp_path):
    fn = tmp_path / "snapshot_002.hdf5"
    write_snapshot(fn, n_dm=5000, groups=[900, 400, 100, 32])
    cat = G.read_gadget4_catalogue(G.find_catalogue_for(str(fn)))
    assert cat.group_len.shape == (4, 6) and cat.n_groups == 4
    mem = G.membership(cat.group_offset, cat.group_len, 1, np.arange(5000))
    assert np.bincount(mem).tolist() == [5000 - 1432, 900, 400, 100, 32]
    assert mem[899] == 1 and mem[900] == 2 and mem[1431] == 4 and mem[1432] == 0
    assert not G.membership(cat.group_offset, cat.group_len, 0, np.arange(3000)).any()
    s = snap_for(fn, 1000)
    sm = G.sample_membership(s, cat)['FoFGroupIndex']
    with h5py.File(fn) as f:
        pid = f['PartType1/ParticleIDs'][:]
    lookup = dict(zip(pid.tolist(), mem.tolist()))
    dm = s['t'] == 1
    np.testing.assert_array_equal(sm[dm], [lookup[int(i)] for i in s['id'][dm]])
    np.testing.assert_array_equal(np.sort(G.group_ids_full([fn], cat, 2)), np.sort(pid[900:1300]))


def test_catalogue_overlapping_offsets_rejected(tmp_path):
    fn = tmp_path / "fof_tab_000.hdf5"
    with h5py.File(fn, 'w') as f:
        f.create_group('Header')
        g = f.create_group('Group')
        g['GroupLenType'] = np.array([[0, 10], [0, 10]])
        g['GroupOffsetType'] = np.array([[0, 0], [0, 5]])
    with pytest.raises(ValueError):
        G.read_gadget4_catalogue(str(fn))


def test_global_rows_multi_file(tmp_path):
    a, b = tmp_path / "snap_005.0.hdf5", tmp_path / "snap_005.1.hdf5"
    write_snapshot(a, n_gas=100, n_dm=300, n_files=2)
    write_snapshot(b, n_gas=50, n_dm=200, n_files=2, id_offset=1000, seed=1)
    plan = [(str(a), 1, np.array([0, 299])), (str(b), 1, np.array([0, 5]))]
    rows = G.global_rows([a, b], plan)
    assert rows[1][1].tolist() == [300, 305]


def test_find_catalogue_with_template(tmp_path):
    (tmp_path / "groups_007").mkdir()
    (tmp_path / "groups_007" / "my_cat_007.hdf5").touch()
    found = G.find_catalogue_for(str(tmp_path / "snapshot_007.hdf5"),
                                 template=str(tmp_path / "groups_003" / "my_cat_003.hdf5"))
    assert found == str(tmp_path / "groups_007" / "my_cat_007.hdf5")


def test_snapshot_number_replaced_only_in_its_own_places(tmp_path):
    # A number elsewhere in the path (a run folder) is not the snapshot number and must stay.
    run = tmp_path / "run_512"
    (run / "groups_007").mkdir(parents=True)
    (run / "groups_007" / "fof_subhalo_tab_007.hdf5").touch()
    found = G.find_catalogue_for(str(run / "snapshot_007.hdf5"),
                                 template=str(run / "groups_003" / "fof_subhalo_tab_003.hdf5"))
    assert found == str(run / "groups_007" / "fof_subhalo_tab_007.hdf5")
    assert A.path_for_snapshot(str(run / "pot_004.0.hdf5"), str(run / "snapshot_012.hdf5")) == \
        str(run / "pot_012.0.hdf5")
    assert G.renumber_path("/d/box_100/cat_0009.hdf5", "12") == "/d/box_100/cat_0012.hdf5"
    assert G.renumber_path("/d/run_512/cat.hdf5", "007") is None


# ----------------------------------------------------------------------------- additional data

def test_aux_hdf5_split_and_ascii(tmp_path):
    fn = tmp_path / "snapshot_004.hdf5"
    write_snapshot(fn, n_gas=300, n_dm=500)
    pot = np.arange(500, dtype=np.float64) * 2
    for i, part in enumerate(np.split(pot, [200])):
        with h5py.File(tmp_path / f"pot_004.{i}.hdf5", 'w') as f:
            f['Potential'] = part
    spec = A.AuxSpec('Potential', str(tmp_path / "pot_004.0.hdf5"), 1, 'hdf5', 'Potential')
    vals = A.read_aux_array(spec)
    np.testing.assert_array_equal(vals, pot)
    s = snap_for(fn, 200)
    att = A.attach_to_sample(s, spec, vals)
    with h5py.File(fn) as f:
        pid = f['PartType1/ParticleIDs'][:].tolist()
    dm = s['t'] == 1
    np.testing.assert_array_equal(att[dm], [pot[pid.index(int(i))] for i in s['id'][dm]])
    assert np.all(np.isnan(att[~dm]))
    np.savetxt(tmp_path / "grp_004.txt", np.c_[np.arange(300), np.arange(300) % 7], fmt='%d', header="id grp")
    spec2 = A.AuxSpec('grp', str(tmp_path / "grp_004.txt"), 0, 'ascii', column=1, header_lines=1, integer=True)
    v2 = A.read_aux_array(spec2)
    assert v2.dtype == np.int64 and v2[13] == 6
    with pytest.raises(ValueError):
        A.attach_to_sample(s, spec2, v2[:-1])
    assert A.path_for_snapshot(str(tmp_path / "pot_004.0.hdf5"), "/x/snapshot_012.hdf5").endswith("pot_012.0.hdf5")


# ----------------------------------------------------------------------------- smoothed images

def test_projection_conserves_mass():
    rng = np.random.default_rng(1)
    pos = rng.uniform(-0.4, 0.4, (5000, 3))
    m = rng.uniform(0.5, 1.5, 5000)
    h = 10 ** rng.uniform(-3, -0.8, 5000)
    cam = {'center': (0, 0, 0), 'distance': 1.0, 'elevation': 90, 'azimuth': 0, 'fov': 90}
    img, w = project(pos, m, h, cam, npix=200)
    assert img.sum() * (w / 200) ** 2 == pytest.approx(m.sum(), rel=1e-6)
    q, _ = project(pos, m, h, cam, npix=100, weights=np.full(5000, 7.0))
    np.testing.assert_allclose(q[np.isfinite(q)], 7.0, rtol=1e-4)


# ----------------------------------------------------------------------------- through the app

@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


_FINISHED = []


@pytest.fixture
def ctl(app, monkeypatch):
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    yield c
    c._reset_tracking()
    c._shutdown_lod()          # as on quit: no level-of-detail polling from windows of finished tests
    c.view.close()
    _FINISHED.append(c)        # never destroyed mid-run by the garbage collector (as the one app window)


def _wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _load(app, c, folder):
    c._load_directory(Path(folder))
    assert _wait(app, lambda: not c.loading and c.model.snapshots, 300)


def test_app_colour_select_plot_stereo(app, ctl, tmp_path):
    write_snapshot(tmp_path / "snapshot_002.hdf5", n_gas=3000, n_dm=5000, groups=[900, 400, 100, 32])
    _load(app, ctl, tmp_path)
    c, cp, sp = ctl, ctl.view.colouring_panel, ctl.view.selection_panel
    names = [cp.combo_prop.itemText(i) for i in range(cp.combo_prop.count())]
    assert 'Group (catalogue)' in names and 'Density' in names and 'Temperature' in names

    # Colour by Density (log): gas is the lit layer (drawn on top), DM the dim unlit layer
    assert cp.select_property('Density')
    cp.chk_log.setChecked(True)
    c._update_frame()
    snap = c.model.snapshots[0]
    unlit, lit = c.view.scatter_3d, c.view.scatter_3d_lit
    gas = snap['t'] == 0
    assert len(lit.pos) == gas.sum() and len(unlit.pos) == (~gas).sum()
    np.testing.assert_allclose(lit.pos, snap['c'][gas])
    np.testing.assert_allclose(unlit.color, np.tile(ct.MISSING_RGBA, ((~gas).sum(), 1)))
    assert np.all(lit.color[:, 3] == pytest.approx(0.7))
    assert lit.depthValue() > unlit.depthValue()             # lit drawn after unlit
    assert unlit.visible()
    assert "(lit 3,000)" in c.view.lbl_info.text()             # 3,000 gas of 8,000 drawn
    assert cp.lbl_info.text().startswith("Density: defined for 3,000 of 8,000 drawn particles")
    assert "Density" in c.view.lbl_colour_overlay.text()

    # Hide the unlit layer: only lit particles stay; style changes need no new colours
    cp.chk_unlit.setChecked(False)
    assert not unlit.visible() and lit.visible()
    cp.sliders[('lit', 'fade')].setValue(40)
    cp.sliders[('lit', 'blur')].setValue(50)
    cp.combo_blend.setCurrentText('Solid')
    assert lit.fade == pytest.approx(0.4) and lit.blur == pytest.approx(0.5) and lit.mode == 'Solid'
    cp.chk_unlit.setChecked(True)
    assert unlit.visible()

    # Movie frames bake the fades into the colours and drop a hidden unlit layer
    ui = c._ui_state()
    ui['drawing'] = dict(ui['drawing'], show_unlit=False)
    sc = c._build_scene(0, ui, compute=True)
    assert len(sc.pos) == 0 and len(sc.lit_pos) == gas.sum() and sc.lit_solid
    np.testing.assert_allclose(sc.lit_colors[:, 3], 0.7 * 0.4, rtol=1e-5)
    cp.combo_blend.setCurrentText('Glow')

    # Catalogue groups: random colours, members of a group share one colour; ungrouped are unlit
    assert cp.select_property('Group (catalogue)')
    assert cp.state()['random']
    c._update_frame()
    grp = c._groups_of(snap)['Group (catalogue)']
    lit_ids = c._pick_candidates[1]
    assert len(lit_ids) == (grp != 0).sum()
    g1 = np.isin(lit_ids, snap['id'][grp == 1])
    col = c.view.scatter_3d_lit.color
    assert g1.sum() > 1 and np.allclose(col[g1], col[g1][0])

    # Select catalogue group 2 at full resolution: all 400 members
    sp.chk_cat.setChecked(True)
    sp.spin_cat.setValue(2)
    c._apply_selection_criteria()
    assert len(c.model.tracker.get_set(0).selected_ids) == 400
    sp.chk_cat.setChecked(False)

    # Property range, full snapshot, against brute force
    sp.chk_prop.setChecked(True)
    sp.combo_prop.setCurrentText('Density')
    sp.edit_prop_min.setText('1')
    sp.edit_prop_max.setText('')
    sp.chk_full.setChecked(True)
    c._apply_selection_criteria()
    with h5py.File(tmp_path / "snapshot_002.hdf5") as f:
        brute = f['PartType0/ParticleIDs'][:][f['PartType0/Density'][:] >= 1]
    assert c.model.tracker.get_set(0).selected_ids == set(int(i) for i in brute)
    sp.chk_prop.setChecked(False)

    # Make plot: sample layer + set layer, set at full resolution
    c._open_plot_window()
    pw = c.plot_window
    pw.combo_type.setCurrentIndex(1)
    pw.combo_x.setCurrentText('Density')
    pw.chk_log_x.setChecked(True)
    pw._on_update_clicked()
    labels = [it.name() for it in pw.plot_widget.listDataItems()]
    assert len(labels) == 2 and labels[1].endswith(f"({len(brute):,})")
    pw.combo_type.setCurrentIndex(0)
    pw.combo_y.setCurrentText('Temperature')
    pw._on_update_clicked()
    assert len(pw.plot_widget.listDataItems()) == 2

    # Stereo: right eye gets the same points and a camera shifted sideways
    c.view.act_stereo.setChecked(True)
    c._update_frame()
    r = c.view.view_3d_right
    assert r is not None and len(c.view.scatter_3d_r.pos) == len(c.view.scatter_3d.pos)
    cam = c.view.camera_state()
    exp = np.array(c.view.stereo_right_centre(cam))
    rc = r.opts['center']
    np.testing.assert_allclose([rc.x(), rc.y(), rc.z()], exp, rtol=1e-5, atol=1e-5)
    assert np.linalg.norm(exp - np.array(cam['center'])) == pytest.approx(0.05 * cam['distance'])
    c.view.act_stereo.setChecked(False)

    # Smoothed density image in the 2D panel
    c.view.combo_2d.setCurrentIndex(1)
    assert c.view.image_2d.isVisible() or c.view.image_2d.image is not None
    ny, nx = c.view.image_2d.image.shape[:2]
    assert ny == nx and 400 <= nx <= 1200      # square, at the panel's resolution
    c.view.combo_2d.setCurrentIndex(2)          # smoothed colour property (Group -> random: falls to density)
    c.view.combo_2d.setCurrentIndex(0)


def test_app_aux_data_follows_snapshot(app, ctl, tmp_path, monkeypatch):
    for k in (0, 1):
        write_snapshot(tmp_path / f"snapshot_00{k}.hdf5", n_gas=300, n_dm=500, seed=k)
        with h5py.File(tmp_path / f"pot_00{k}.hdf5", 'w') as f:
            f['Potential'] = np.full(500, float(k + 1))
    _load(app, ctl, tmp_path)
    spec = A.AuxSpec('Potential', str(tmp_path / "pot_000.hdf5"), 1, 'hdf5', 'Potential')

    class FakeDialog:
        def __init__(self, *a):
            pass

        def exec(self):
            return True

        def spec(self):
            return spec
    import views.aux_data_dialog as d
    monkeypatch.setattr(d, "AuxDataDialog", FakeDialog)
    ctl._open_aux_dialog()
    assert len(ctl.model.snapshots) == 2
    for snap in ctl.model.snapshots:
        k = int(Path(snap['filepath']).stem[-1])
        vals = ctl._sample_values(snap, 'Potential')
        assert np.all(vals[snap['t'] == 1] == k + 1)


# ----------------------------------------------------------------------------- real outputs

def _g4_pairs():
    if not G4_OUTPUT:
        return []
    out = []
    for sp in sorted(Path(G4_OUTPUT).glob("snapshot_*.hdf5")):
        cat = G.find_catalogue_for(str(sp))
        if cat:
            out.append((str(sp), cat))
    return out


@pytest.mark.skipif(not _g4_pairs(), reason="set COSMOVIEWER_G4_OUTPUT to DM-L50-N128/output")
@pytest.mark.parametrize("pair", _g4_pairs() or [None])
def test_real_dm_catalogue_membership(pair):
    """Members by offsets: counts equal GroupLen and centres of mass equal GroupPos (independent)."""
    snap_path, cat_path = pair
    cat = G.read_gadget4_catalogue(cat_path)
    with h5py.File(snap_path) as f:
        pos = f['PartType1/Coordinates'][:].astype(np.float64)
        box = float(f['Header'].attrs['BoxSize'])
    with h5py.File(cat_path) as f:
        glen, gpos = f['Group/GroupLen'][:], f['Group/GroupPos'][:]
    mem = G.membership(cat.group_offset, cat.group_len, 1, np.arange(len(pos)))
    cnt = np.bincount(mem, minlength=cat.n_groups + 1)
    np.testing.assert_array_equal(cnt[1:], glen)
    order = np.argsort(mem, kind='stable')
    starts = np.r_[0, np.cumsum(cnt)]
    worst = 0.0
    for g in range(1, cat.n_groups + 1):
        p = pos[order[starts[g]:starts[g + 1]]]
        d = p - p[0]
        p = p[0] + d - box * np.round(d / box)
        e = p.mean(0) - gpos[g - 1]
        e -= box * np.round(e / box)
        worst = max(worst, float(np.abs(e).max()))
    assert worst < 1e-4 * box


@pytest.mark.skipif(not CG_OUTPUT, reason="set COSMOVIEWER_CG_OUTPUT to CollidingGalaxiesSFR/output")
def test_real_colliding_galaxies_properties():
    snaps = sorted(Path(CG_OUTPUT).glob("snapshot_*.hdf5"))
    fn = str(snaps[min(1, len(snaps) - 1)])
    props = P.list_properties([fn])
    assert {'Density', 'Temperature', 'StarFormationRate', 'Metallicity', 'Masses'} <= set(props)
    s = snap_for(fn, 150_000)
    T = P.read_sample_values(s, props['Temperature'])
    gas = s['t'] == 0
    assert np.isfinite(T[gas]).all() and np.isnan(T[~gas]).all()
    assert 10 < np.nanmin(T) and np.nanmax(T) < 1e9            # physically plausible gas
    m = P.read_sample_values(s, props['Masses'])
    np.testing.assert_allclose(m, s['m'], rtol=1e-6)            # MassTable types included
    ids = P.select_by_range([fn], props['Temperature'], 1e5, np.inf, [0])
    with h5py.File(fn) as f:
        g = f['PartType0']
        Tf = P.temperature_from_u(g['InternalEnergy'][:], g['ElectronAbundance'][:], 1e10)
        brute = g['ParticleIDs'][:][Tf >= 1e5]
    np.testing.assert_array_equal(np.sort(ids), np.sort(brute.astype(np.uint64)))


def test_read_values_for_many_unordered_ids(tmp_path):
    """More than 512 hits per chunk, IDs stored in reverse: rows read out of order must still align."""
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn, n_gas=3000)
    with h5py.File(fn, 'r+') as f:
        ids = f['PartType0/ParticleIDs'][:][::-1].copy()
        f['PartType0/ParticleIDs'][:] = ids
        rho = f['PartType0/Density'][:]
    want = ids[100:1700]
    v = P.read_values_for_ids([fn], P.list_properties([fn])['Density'], want)
    order = np.argsort(want)
    np.testing.assert_array_equal(v, rho[100:1700][order].astype(np.float64))


def _view_set(path, keep_fn):
    """A LOD-style view set (lod.store.make_view) of the rows of each type where keep_fn(pos) holds."""
    make_view = pytest.importorskip("lod.store").make_view
    parts = {k: [] for k in ('c', 'v', 'm', 'id', 't', 'piece', 'row')}
    with h5py.File(path, 'r') as f:
        mt = f['Header'].attrs['MassTable']
        for t in (0, 1):
            g = f[f'PartType{t}']
            pos = g['Coordinates'][...]
            rows = np.flatnonzero(keep_fn(pos))
            parts['c'].append(pos[rows].astype(np.float32))
            parts['v'].append(g['Velocities'][...][rows].astype(np.float32))
            m = g['Masses'][...][rows] if 'Masses' in g else np.full(len(rows), mt[t])
            parts['m'].append(m.astype(np.float32))
            parts['id'].append(g['ParticleIDs'][...][rows])
            parts['t'].append(np.full(len(rows), t, np.uint8))
            parts['piece'].append(np.zeros(len(rows), np.uint16))
            parts['row'].append(rows)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return make_view(cat['c'], cat['v'], cat['m'], cat['id'], cat['t'], cat['piece'], cat['row'])


def test_lod_view_set_keeps_features_aligned(app, ctl, tmp_path, monkeypatch):
    """A zoomed (non-uniform) view set gets its own group numbers, property values and density."""
    import controller.features_controller as F
    path = tmp_path / "snapshot_002.hdf5"
    write_snapshot(path, n_gas=3000, n_dm=5000, groups=[900, 400, 100, 32])
    _load(app, ctl, tmp_path)
    c = ctl
    base = c.model.snapshots[0]
    c._enable_catalogues('auto')
    view = dict(base)
    view.update(_view_set(path, lambda p: p[:, 0] < 3.0))
    view['_props'], view['lod'] = {}, object()
    assert len(view['c']) < len(base['c'])

    # Group numbers by ID equal the base sample's
    bg, vg = c._groups_of(base)[F.CAT_GROUP], c._groups_of(view)[F.CAT_GROUP]
    by_id = dict(zip(base['id'].tolist(), bg.tolist()))
    assert np.array_equal(vg, [by_id[i] for i in view['id'].tolist()])
    assert c._groups_of(base)[F.CAT_GROUP] is bg              # base cache untouched

    # Density by ID equals the file's
    with h5py.File(path, 'r') as f:
        dens = dict(zip(f['PartType0/ParticleIDs'][...].tolist(), f['PartType0/Density'][...].tolist()))
    vals = c._sample_values(view, 'Density')
    gas = view['t'] == 0
    np.testing.assert_allclose(vals[gas], [dens[i] for i in view['id'][gas].tolist()])
    assert np.all(np.isnan(vals[~gas]))

    # kNN density uses the uniform base sample: a drawn particle that is also in the base
    # sample gets the base sample's own estimate
    rho_b, rho_v = c._knn_density(base, 10.0), c._knn_density(view, 10.0)
    pos_b = {i: k for k, i in enumerate(base['id'].tolist())}
    k = [pos_b[i] for i in view['id'][:50].tolist()]
    np.testing.assert_allclose(rho_v[:50], rho_b[k], rtol=1e-5)

    # Large view sets read file properties in a worker; the frame is redrawn when it finishes
    monkeypatch.setattr(F, 'BACKGROUND_ROWS', 10)
    view2 = dict(view)
    view2['_props'] = {}
    assert c._sample_values(view2, 'InternalEnergy') is None
    assert c._prop_jobs and "reading InternalEnergy" in c._features_status(view2)
    assert _wait(app, lambda: not c._prop_jobs, 60)
    assert 'InternalEnergy' in view2['_props']
    assert c._sample_values(view2, 'InternalEnergy') is not None

    # One job per quantity: new view sets while it runs start no second reader
    views = [dict(view, _props={}) for _ in range(4)]
    for v in views:
        assert c._sample_values(v, 'SmoothingLength') is None
    assert list(c._prop_jobs) == ['SmoothingLength']
    assert _wait(app, lambda: not c._prop_jobs, 60)
    assert c._sample_values(views[-1], 'SmoothingLength', wait=True) is not None

    # Local density of a large set is computed in a worker too, from the base sample
    v5 = dict(view, _props={})
    assert c._knn_density(v5, 10.0) is None and F.DENSITY in c._prop_jobs
    assert _wait(app, lambda: not c._prop_jobs, 60)
    np.testing.assert_allclose(c._knn_density(v5, 10.0), rho_v, rtol=1e-6)


# ----------------------------------------------------------------------------- read-once property cache

def test_property_cache_on_disk_multi_file(tmp_path, monkeypatch):
    """Disk path (one process per file): values by (piece, row), range selection and top values."""
    from snapshot_io import property_cache as PC
    monkeypatch.setattr(PC, 'DISK_MIN_ROWS', 10)
    monkeypatch.setattr(PC, 'CACHE_DIR', tmp_path / "cache")
    a, b = tmp_path / "snap_005.0.hdf5", tmp_path / "snap_005.1.hdf5"
    write_snapshot(a, n_gas=1000, n_dm=1500, n_files=2)
    write_snapshot(b, n_gas=700, n_dm=900, n_files=2, id_offset=100_000, seed=1)
    files = [str(a), str(b)]
    info = P.list_properties(files)['Density']
    rho, pid = {}, {}
    for fn in files:
        with h5py.File(fn) as f:
            rho.update(zip(f['PartType0/ParticleIDs'][:].tolist(), f['PartType0/Density'][:].tolist()))
    rng = np.random.default_rng(5)
    rows = {(0, 0): np.sort(rng.choice(1000, 300, replace=False)), (1, 0): np.sort(rng.choice(700, 200, replace=False)),
            (0, 1): np.arange(0, 1500, 7)}
    ids, ts = [], []
    for (piece, t), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        with h5py.File(files[piece]) as f:
            ids.append(f[f'PartType{t}/ParticleIDs'][:][r])
        ts.append(np.full(len(r), t))
    ids, ts = np.concatenate(ids), np.concatenate(ts)
    s = {'files': files, 'filepath': files[0], 'rows': rows, 'c': np.zeros((len(ids), 3)), 'id': ids, 't': ts}
    vals = P.read_sample_values(s, info)
    gas = ts == 0
    np.testing.assert_array_equal(vals[gas], [rho[int(i)] for i in ids[gas]])
    assert np.all(np.isnan(vals[~gas]))
    assert len(list((tmp_path / "cache").glob("*/Density_t0.npy"))) == 1
    PC._memory.clear()
    s2 = dict(s, _props={})
    np.testing.assert_array_equal(P.read_sample_values(s2, info), vals)       # reopened from disk

    got = P.select_by_range(files, info, 0.1, 1.0, [0, 1])
    want = [i for i, v in rho.items() if 0.1 <= v <= 1.0]
    assert sorted(got.tolist()) == sorted(want)

    top_ids, top_t, top_v = PC.top_values(files, info, [0, 1], k=5)
    best = sorted(rho.items(), key=lambda kv: -kv[1])[:5]
    assert top_ids.tolist() == [i for i, _ in best] and np.all(top_t == 0)
    np.testing.assert_allclose(top_v, [v for _, v in best], rtol=1e-6)


def test_property_ids_mismatch_uses_id_matching(tmp_path):
    """Rows that disagree with the set's IDs are never trusted."""
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn, n_gas=3000, n_dm=5000)
    with h5py.File(fn) as f:
        pid, dens = f['PartType0/ParticleIDs'][:], f['PartType0/Density'][:]
    r = np.arange(10, 3000, 10)
    s = {'files': [str(fn)], 'filepath': str(fn), 'rows': {(0, 0): r + 1},    # off by one row
         'c': np.zeros((len(r), 3)), 'id': pid[r], 't': np.zeros(len(r), int)}
    np.testing.assert_array_equal(P.read_sample_values(s, P.list_properties([fn])['Density']), dens[r])


# ----------------------------------------------------------------------------- picking lit particles

def test_nearest_on_screen_prefers_nearest_depth():
    from controller.picking_controller import PickingMixin
    x = np.array([10.0, 10.5, 50.0, 10.2])
    y = np.array([10.0, 10.0, 50.0, 10.1])
    depth = np.array([0.9, 0.2, 0.1, 0.5])
    front = np.array([True, True, True, False])
    assert PickingMixin.nearest_on_screen(x, y, depth, front, 10.1, 10.0, 6) == 1   # within a pixel: nearer
    assert PickingMixin.nearest_on_screen(x, y, depth, front, 30, 30, 6) is None


def test_app_pick_by_click_and_from_list_then_tracked(app, ctl, tmp_path):
    for k in (0, 1):
        write_snapshot(tmp_path / f"snapshot_00{k}.hdf5", n_gas=3000, n_dm=5000, seed=k)
    _load(app, ctl, tmp_path)
    c, cp, sp = ctl, ctl.view.colouring_panel, ctl.view.selection_panel
    c.view.resize(1200, 800)
    assert cp.select_property('Density')
    c._update_frame()
    pos, ids, types, vals = c._pick_candidates
    assert len(ids) == 3000 and np.all(types == 0) and vals is not None

    # Click exactly on one lit particle (screen position from the view's own matrices)
    x, y, depth, front = c.view.screen_coords(pos)
    ok = np.flatnonzero(front & (x > 5) & (x < c.view.view_3d.width() - 5) & (y > 5) & (y < c.view.view_3d.height() - 5))
    k = int(ok[np.argmin(depth[ok])])
    sp.chk_pick.setChecked(True)
    c.view.pick_requested.emit(float(x[k]), float(y[k]), False)
    s_set = c.model.tracker.get_set(sp.active_set_idx)
    got = next(iter(s_set.selected_ids))
    assert len(s_set.selected_ids) == 1
    near = np.flatnonzero((x - x[k]) ** 2 + (y - y[k]) ** 2 <= 1.0)
    assert got in set(int(i) for i in ids[near])
    assert "Density" in sp.lbl_pick.text()
    # ...and it is looked up by ID in every snapshot (tracking across snapshots)
    assert _wait(app, lambda: all(c._tracked(i, s_set) is not None for i in range(2)), 60)
    assert all(c._tracked(i, s_set).found.all() for i in range(2))

    # Shift+click adds
    k2 = int(ok[np.argmax(depth[ok])])
    c.view.pick_requested.emit(float(x[k2]), float(y[k2]), True)
    assert len(s_set.selected_ids) == 2

    # List of the highest Density in the full snapshot; choosing one picks it
    c._list_top()
    assert _wait(app, lambda: sp.combo_top.count() > 0, 60)
    with h5py.File(c.model.get_snapshot(c.current_snap_idx)['filepath']) as f:   # the shown snapshot
        rho, pid = f['PartType0/Density'][:], f['PartType0/ParticleIDs'][:]
    best = int(pid[np.argmax(rho)])
    assert c._top_list[0][0] == best
    sp.radio_new.setChecked(True)
    c._pick_from_list(0)
    assert s_set.selected_ids == {best}


# ----------------------------------------------------------------------------- halo membership stored in the snapshot

def add_snapshot_halos(path, seed=0):
    """EAGLE-style GroupNumber / SubGroupNumber per particle (2**30 = none); returns {id: (group, sub)}."""
    rng = np.random.default_rng(seed)
    out = {}
    with h5py.File(path, 'r+') as f:
        for t in (0, 1):
            g = f[f'PartType{t}']
            n = g['ParticleIDs'].shape[0]
            gn = rng.integers(1, 6, n).astype(np.int32)
            sg = rng.integers(0, 3, n).astype(np.int32)
            gn[rng.random(n) < 0.3] = P.NO_HALO
            sg[gn == P.NO_HALO] = P.NO_HALO
            sg[rng.random(n) < 0.1] = P.NO_HALO           # in a group but in no subhalo
            g['GroupNumber'], g['SubGroupNumber'] = gn, sg
            out.update(zip(g['ParticleIDs'][:].tolist(), zip(gn.tolist(), sg.tolist())))
    return out


def test_snapshot_halo_membership(tmp_path, monkeypatch):
    from snapshot_io import property_cache as PC
    monkeypatch.setattr(PC, 'DISK_MIN_ROWS', 1000)       # also through the on-disk cache
    monkeypatch.setattr(PC, 'CACHE_DIR', tmp_path / 'cache')
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn)
    truth = add_snapshot_halos(fn)
    props = P.list_properties([fn])
    grp, sub = props[P.SNAP_GROUP], props[P.SNAP_SUBHALO]
    assert grp.integer and sub.integer and grp.types == (0, 1) and sub.types == (0, 1)
    s = snap_for(fn, 1000)
    g_exp = np.array([truth[int(i)][0] if truth[int(i)][0] != P.NO_HALO else 0 for i in s['id']])
    s_exp = np.array([g * P.SUBHALO_STRIDE + truth[int(i)][1] if g and truth[int(i)][1] != P.NO_HALO else 0
                      for g, i in zip(g_exp, s['id'])])
    np.testing.assert_array_equal(P.read_sample_values(s, grp), g_exp)
    np.testing.assert_array_equal(P.read_sample_values(s, sub), s_exp)
    # Full-resolution selection of one group and one subhalo
    ids = P.select_by_range([fn], grp, 3, 3, [0, 1])
    assert set(ids.tolist()) == {i for i, (g, _) in truth.items() if g == 3}
    key = 3 * P.SUBHALO_STRIDE + 0
    ids = P.select_by_range([fn], sub, key, key, [0, 1])
    assert set(ids.tolist()) == {i for i, (g, sg) in truth.items() if g == 3 and sg == 0}


def test_app_snapshot_halos_colour_and_select(app, ctl, tmp_path):
    fn = tmp_path / "snapshot_000.hdf5"
    write_snapshot(fn, n_gas=3000, n_dm=5000)
    truth = add_snapshot_halos(fn)
    _load(app, ctl, tmp_path)
    c, cp, sp = ctl, ctl.view.colouring_panel, ctl.view.selection_panel
    assert cp.select_property(P.SNAP_GROUP)
    c._update_frame()
    pos, ids, types, vals = c._pick_candidates
    in_group = {i for i, (g, _) in truth.items() if g != P.NO_HALO}
    shown = c.model.get_snapshot(c.current_snap_idx)['id']
    assert set(int(i) for i in ids) == {int(i) for i in shown if int(i) in in_group}   # lit = in a group
    assert all(truth[int(i)][0] == int(v) for i, v in zip(ids, vals))
    # The Catalogue criterion uses the snapshot's own membership when no catalogue is loaded
    sp.radio_new.setChecked(True)
    sp.chk_cat.setChecked(True)
    sp.combo_cat.setCurrentText('group')
    sp.spin_cat.setValue(2)
    c._apply_selection_criteria()
    s_set = c.model.tracker.get_set(sp.active_set_idx)
    assert s_set.selected_ids == {i for i, (g, _) in truth.items() if g == 2}
    sp.combo_cat.setCurrentText('subhalo')
    sp.spin_cat.setValue(2 * P.SUBHALO_STRIDE + 1)
    c._apply_selection_criteria()
    assert s_set.selected_ids == {i for i, (g, sg) in truth.items() if g == 2 and sg == 1}
    sp.chk_cat.setChecked(False)


def test_histogram_bins_cover_selection_sets():
    # A selection set is read at full resolution and can hold values the display sample never
    # reaches (the densest particles of a halo). The bins used to come from the sample alone, and
    # np.histogram drops values outside them, so the set's tail vanished while its legend still
    # counted every member.
    from types import SimpleNamespace
    from controller.features_controller import FeaturesMixin
    got = {}
    pw = SimpleNamespace(show_message=lambda t: got.setdefault('msg', t),
                         draw_histogram_layers=lambda layers, *a, **k: got.setdefault('layers', layers))
    sample = np.linspace(1.0, 10.0, 1000)
    members = np.r_[np.linspace(5.0, 10.0, 50), np.linspace(10.0, 100.0, 50)]   # half beyond the sample
    s_set = SimpleNamespace(name="Set 1", selected_ids={1})
    snap = {'filepath': '/x/snapshot_000.hdf5', 't': np.ones(1000, np.uint8), 'c': np.zeros((1000, 3)),
            'n_total': 14000}
    stub = SimpleNamespace(
        plot_window=pw, current_snap_idx=0, _prop_jobs={},
        model=SimpleNamespace(get_snapshot=lambda i: snap, tracker=SimpleNamespace(get_set=lambda k: s_set)),
        _ui_state=lambda: {'visible': [1], 'sets': [(0, True, (255, 0, 0, 255), 0)]},
        _scene_values=lambda snap, nm, mask, ui: sample,
        _plot_values_for_set=lambda i, s, nm: members)
    FeaturesMixin._render_property_plot(stub, 'histogram', 'Density', '', False, False)
    (_, e0, n0, _), (label, e1, n1, _) = got['layers']
    assert n0.sum() == 1000 and label.endswith("(100)")
    assert n1.sum() == 100                     # every member is in some bin
