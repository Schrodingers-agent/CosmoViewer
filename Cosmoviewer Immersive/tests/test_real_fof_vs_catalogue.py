"""
Real-data check of the FoF selection repair and of the "group" population,
against Gadget-4's own FoF catalogue.

    COSMOVIEWER_G4_OUTPUT=<...>/DM-L50-N128/output \\
        QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_real_fof_vs_catalogue.py

Data: DM-L50-N128 snapshot_002 (z = 0, 2,097,152 DM particles) and fof_tab_002
(2,774 groups, b = 0.2, N_min = 32, DM links). Gadget-4 writes snapshots in
group order, so catalogue group g (0-based) is PartType1 rows
GroupOffsetType[g, 1] .. + GroupLenType[g, 1].

What is checked:
  * CosmoViewer's FoF finds exactly Gadget-4's partition of snapshot_002: the
    same 2,774 member sets. Only the numbering of equal-length groups differs
    (CosmoViewer breaks ties by the smallest ParticleID; the catalogue's tie
    order follows no ID rule), so "FoF group k" and "catalogue group k" are the
    same halo only while group k's length is unique (k <= 120 here).
  * In the application, at the default budget (2M, below the 2,097,152
    particles) and at 500k: the FoF window run with its own defaults, then
    Highlight Halo on group 1, gives a complete "group" population whose IDs
    are exactly catalogue group 1, with n_expected, b, N_min, linked types and
    the snapshot it was defined at; the "FoF Group ID" criterion gives the same IDs.
  * The population survives selection_sets.write_set version 2 and reaches the
    Dataset Explorer (hosted in this process, as the Analyse view will be):
    its Lagrangian radii and mass equal a headless measurements computation on
    the catalogue membership.
  * Send to Explorer writes the same meaning into the set file.

P1 below names the bug this file guards against: Highlight Halo used to select
only the members that happened to be in the display sample.

The run is copied to a temporary folder first, so nothing the app writes
(cosmoviewer_sets/, caches) lands next to the original output. Skipped when
COSMOVIEWER_G4_OUTPUT is unset or lacks snapshot_000..002 and fof_tab_002.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXPLORER = ROOT.parent / "Dataset Explorer"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

G4_OUTPUT = os.environ.get("COSMOVIEWER_G4_OUTPUT", "")
NAMES = [f"snapshot_00{i}.hdf5" for i in range(3)] + ["fof_tab_002.hdf5"]
HAVE_DATA = bool(G4_OUTPUT) and all((Path(G4_OUTPUT) / n).exists() for n in NAMES)
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="set COSMOVIEWER_G4_OUTPUT to DM-L50-N128/output "
                                                      "(snapshot_000..002.hdf5 and fof_tab_002.hdf5)")

N_PART = 2_097_152
N_GROUPS = 2_774
LARGEST_20 = [34120, 31375, 22599, 18404, 17054, 16682, 16571, 16191, 15001, 13394,
              12986, 12722, 12594, 12566, 11858, 11465, 11450, 11123, 11097, 11026]
L_LINK = 0.078125                      # 0.2 x (50 Mpc/h) / 128, Gadget-4's mass-based mean separation


# ----------------------------------------------------------------------------- data and helpers

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Plain copies of the three snapshots and the z = 0 catalogue (the app writes next to snapshots)."""
    d = tmp_path_factory.mktemp("dm_l50") / "output"
    d.mkdir()
    for n in NAMES:
        shutil.copyfile(Path(G4_OUTPUT) / n, d / n)
    return d


@pytest.fixture(scope="module")
def catalogue(run_dir):
    with h5py.File(run_dir / "fof_tab_002.hdf5", "r") as f:
        off = f["Group/GroupOffsetType"][:, 1].astype(np.int64)
        ln = f["Group/GroupLenType"][:, 1].astype(np.int64)
        pos = f["Group/GroupPos"][:].astype(np.float64)
    with h5py.File(run_dir / "snapshot_002.hdf5", "r") as f:
        ids = f["PartType1/ParticleIDs"][:].astype(np.uint64)
    return {"off": off, "len": ln, "pos": pos, "snap_ids": ids,
            "members": lambda g: np.unique(ids[off[g - 1]:off[g - 1] + ln[g - 1]])}   # g 1-based


def _wait(app, cond, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _explorer_idle(app, win, timeout=600):
    """As the Explorer's own tests wait: until no analysis worker is running."""
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        w = win._active_worker
        if w is None or not w.isRunning():
            app.processEvents()
            if win._active_worker is None or not win._active_worker.isRunning():
                return True
        time.sleep(0.02)
    return False


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module", params=[2_000_000, 500_000], ids=["budget2M", "budget500k"])
def loaded(request, qapp, run_dir, tmp_path_factory):
    """The run loaded at a particle budget, snapshot_002 on screen, FoF run from its window's defaults."""
    budget = request.param
    home = tmp_path_factory.mktemp(f"home_{budget}")
    mp = pytest.MonkeyPatch()
    from PyQt6.QtWidgets import QMessageBox
    messages = []
    for name in ("information", "warning", "critical", "question"):      # a modal box would block offscreen
        mp.setattr(QMessageBox, name, staticmethod(
            lambda *a, _n=name, **k: (messages.append((_n, str(a[2]) if len(a) > 2 else "")),
                                      QMessageBox.StandardButton.Yes if _n == "question"
                                      else QMessageBox.StandardButton.Ok)[1]))
    import lod.store
    import selection_sets
    import snapshot_io.property_cache as pc
    mp.setattr(lod.store, "CACHE_DIR", home / "lod_cache")
    mp.setattr(pc, "CACHE_DIR", home / "property_cache")
    mp.setattr(selection_sets, "FALLBACK_ROOT", home / "sets")
    mp.setattr(selection_sets, "INSTANCES_DIR", home / "explorer_instances")
    from controller.app_controller import CosmoViewerController
    c = CosmoViewerController()
    c.max_particles = budget
    t0 = time.perf_counter()
    c._load_directory(run_dir)
    assert _wait(qapp, lambda: not c.loading and len(c.model.snapshots) == 3)
    load_s = time.perf_counter() - t0
    c.view.slide_time.setValue(2)
    qapp.processEvents()
    assert c.current_snap_idx == 2
    snap = c.model.get_snapshot(2)
    assert Path(snap["filepath"]).name == "snapshot_002.hdf5"

    c._open_fof_dialog()
    dlg = c.fof_dialog
    emitted = []
    dlg.fof_run_requested.connect(emitted.append)
    t0 = time.perf_counter()
    dlg._on_run_clicked()                       # the window's own defaults, as a user runs it
    assert _wait(qapp, lambda: 'fof' in snap and c._fof_worker is not None and not c._fof_worker.isRunning())
    fof_s = time.perf_counter() - t0
    yield {"ctl": c, "snap": snap, "budget": budget, "settings": emitted[0], "messages": messages,
           "load_s": load_s, "fof_s": fof_s, "home": home}
    c._shutdown_fof()
    c._shutdown_lod()
    c.view.close()
    qapp.processEvents()
    mp.undo()


def _active(ctl):
    return ctl.model.tracker.get_set(ctl.view.selection_panel.active_set_idx)


# ----------------------------------------------------------------------------- the finder itself

def test_fof_partition_equals_gadget4_catalogue(run_dir, catalogue):
    """Exact agreement with fof_tab_002: number, sizes and member sets of every group."""
    from snapshot_io.fof_snapshot import run_snapshot_fof
    fof = run_snapshot_fof(str(run_dir / "snapshot_002.hdf5"), b=0.2, min_group_size=32,
                           primary_types=(1,), convention="gadget4")
    assert fof.result.linking_length == pytest.approx(L_LINK, rel=1e-7)
    gl = np.asarray(fof.result.group_len)
    assert len(gl) == len(catalogue["len"]) == N_GROUPS
    assert gl[:20].tolist() == catalogue["len"][:20].tolist() == LARGEST_20
    np.testing.assert_array_equal(gl, catalogue["len"])            # every size, rank by rank

    ours = fof.lookup(catalogue["snap_ids"].astype(fof.ids.dtype))  # our group of each snapshot row
    ref = np.full(N_PART, -1, dtype=np.int64)
    for g, (o, n) in enumerate(zip(catalogue["off"], catalogue["len"])):
        ref[o:o + n] = g
    assert np.array_equal(ref < 0, ours < 0)                      # the same particles are in groups
    pairs, cnt = np.unique(np.stack([ref[ref >= 0], ours[ref >= 0]], 1), axis=0, return_counts=True)
    assert len(pairs) == N_GROUPS                                  # one-to-one: identical member sets
    assert len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1])) == N_GROUPS
    np.testing.assert_array_equal(cnt, catalogue["len"][pairs[:, 0]])
    renumbered = pairs[pairs[:, 0] != pairs[:, 1]]
    # numbers differ only between groups of equal length (tie order), never for a unique length
    np.testing.assert_array_equal(catalogue["len"][renumbered[:, 0]], catalogue["len"][renumbered[:, 1]])
    vals, counts = np.unique(catalogue["len"], return_counts=True)
    first_tie = int(np.nonzero(np.isin(catalogue["len"], vals[counts > 1]))[0][0]) + 1
    assert first_tie == 121 and renumbered[:, 0].min() + 1 >= first_tie
    g1 = np.unique(fof.ids[fof.group_index == 0].astype(np.uint64))
    np.testing.assert_array_equal(g1, catalogue["members"](1))


# ----------------------------------------------------------------------------- in the application

def test_fof_window_reports_the_catalogue_groups(loaded):
    snap, st = loaded["snap"], loaded["settings"]
    assert st == {'b_linking': pytest.approx(0.2), 'min_size': 32, 'r_link': None, 'link_types': [1]}
    assert len(snap['fof_sizes']) == N_GROUPS
    assert [snap['fof_sizes'][g] for g in range(1, 21)] == LARGEST_20
    assert snap['fof_linking_length'] == pytest.approx(L_LINK, rel=1e-7)
    status = loaded["ctl"].fof_dialog.lbl_status.text()
    assert "Detected 2,774 FoF groups in snapshot_002.hdf5" in status, status
    # the display sample is smaller than the snapshot, which is what made P1 bite
    assert snap.get('n_total') == N_PART and len(snap['id']) < N_PART


def test_highlight_halo_1_is_the_complete_catalogue_group(loaded, catalogue):
    ctl, snap = loaded["ctl"], loaded["snap"]
    ctl.view.selection_panel.combo_sets.setCurrentIndex(0)
    ctl._highlight_fof_halo(1)
    s = _active(ctl)
    ids = np.sort(s.id_array())
    np.testing.assert_array_equal(ids, catalogue["members"](1))    # every member, as ParticleIDs
    assert len(ids) == 34_120
    in_sample = int(np.isin(ids, np.asarray(snap['id'], dtype=np.uint64)).sum())
    assert in_sample < len(ids)                                    # what P1 used to select instead
    print(f"\nbudget {loaded['budget']:,}: display sample {len(snap['id']):,} of {N_PART:,}; "
          f"group 1 members in it {in_sample:,} of {len(ids):,} (what P1 used to select); "
          f"load {loaded['load_s']:.1f} s, FoF {loaded['fof_s']:.1f} s")

    assert (s.kind, s.complete, s.n_expected) == ('group', True, 34_120)
    d = s.definition
    assert (d['finder'], d['group'], d['subhalo']) == ('fof', 1, False)
    assert d['b'] == pytest.approx(0.2) and d['min_size'] == 32 and d['link_types'] == [1]
    assert d['convention'] == 'gadget4'
    assert d['linking_length_derived'] == pytest.approx(L_LINK, rel=1e-7)
    assert 'linking_length' not in d                                # b was the choice, l derived from it
    at = s.defined_at
    assert (at['file'], at['index']) == ('snapshot_002.hdf5', 2)
    assert at['time'] == pytest.approx(1.0) and at['redshift'] == pytest.approx(0.0, abs=1e-12)
    assert s.describe() == ("FoF group 1 (b = 0.2, N_min = 32, DM) at snapshot_002.hdf5 · complete · "
                            "34,120 particles")
    assert s.completeness_warning() is None


def test_fof_criterion_selects_the_same_members(loaded, catalogue):
    ctl = loaded["ctl"]
    sp = ctl.view.selection_panel
    sp.combo_sets.setCurrentIndex(1)
    for chk in (sp.chk_ids, sp.chk_radius, sp.chk_prop, sp.chk_cat):
        chk.setChecked(False)
    sp.chk_fof.setChecked(True)
    sp.spin_fof_id.setValue(1)
    try:
        ctl._apply_selection_criteria()
    finally:
        sp.chk_fof.setChecked(False)
    s = ctl.model.tracker.get_set(1)
    np.testing.assert_array_equal(np.sort(s.id_array()), catalogue["members"](1))


def test_group_population_reaches_the_explorer_with_the_headless_numbers(loaded, catalogue, run_dir, qapp):
    """Population -> set file version 2 -> Explorer in this process -> Lagrangian radii and mass."""
    if not (EXPLORER / "density_explorer.py").exists():
        pytest.skip("Dataset Explorer folder not found next to Immersive")
    import selection_sets
    ctl = loaded["ctl"]
    ctl.view.selection_panel.combo_sets.setCurrentIndex(0)
    ctl._highlight_fof_halo(1)
    pop = _active(ctl).as_selection_set("#ffaa00")
    path = selection_sets.write_set(str(run_dir / "snapshot_002.hdf5"), pop.ids, "FoF group 1", pop.colour,
                                    made_in=pop.made_in, path=str(loaded["home"] / "g1.hdf5"),
                                    kind=pop.kind, definition=pop.definition, defined_at=pop.defined_at,
                                    complete=pop.complete, n_expected=pop.n_expected)
    st = selection_sets.read_set(path)
    assert (st.kind, st.complete, st.n_expected, st.defined_at, st.definition) == \
        ('group', True, 34_120, pop.defined_at, pop.definition)
    np.testing.assert_array_equal(st.ids, catalogue["members"](1))

    if str(EXPLORER) not in sys.path:
        sys.path.append(str(EXPLORER))          # after Immersive's folder: only the shared files coincide
    import density_explorer
    from measurements import SnapshotView, find_centre, lagrangian_radii
    from plot_panel import _format_cell
    win = density_explorer.CosmoExplorer()
    try:
        win.show()
        assert win.open_path(str(run_dir / "snapshot_002.hdf5"))
        assert win._activate_set(path)
        win.sidebar.select_leaf("Analyses", "Lagr. radii, all types")
        win.plot_panel._toolbar._switch_view("Matrix")
        assert _explorer_idle(qapp, win)
        t = win.plot_panel._matrix_view._table
        radii_shown = [t.item(i, 1).text() for i in range(t.rowCount())]
        notes = win.plot_panel._export_columns[0]
        status = win.plot_panel.lbl_status.text()
        info = win.plot_panel.lbl_info.text()
    finally:
        win.close()
        qapp.processEvents()

    p = SnapshotView(str(run_dir / "snapshot_002.hdf5")).particles([1], velocities=False, ids=True)
    p = p.subset(np.isin(p.ids.astype(np.uint64), catalogue["members"](1)))
    ref = lagrangian_radii(p, find_centre(p, "shrinking_sphere"))
    assert ref.provenance.n_used == ref.provenance.n_total == 34_120
    assert radii_shown == [_format_cell(float(v), "Auto") for v in ref["radius"].value]
    m_ref = float(ref["mass_total"].value)
    from data_utils import latex_to_html
    assert latex_to_html(f"M = {m_ref:.4e} Msun") in info, info
    print(f"\nbudget {loaded['budget']:,}: Explorer M = {m_ref:.6e} Msun; Lagrangian radii "
          + ", ".join(f"{f:g}: {r:.4g} kpc" for f, r in zip(ref['fraction'], ref['radius'].value)))
    # 34,120 x 0.5093952 x 1.989e43 g / h (h = 0.678), in astropy's solar mass (1.98841e33 g)
    assert m_ref == pytest.approx(34_120 * 0.5093952 * 1.989e43 / 1.98840987e33 / 0.678, rel=1e-6)
    assert "All 34,120 particles" in status, status
    assert any("is: FoF group 1 (b = 0.2, N_min = 32, DM) at snapshot_002.hdf5 · complete · 34,120 particles"
               in n for n in notes), notes
    assert not any("WARNING" in n for n in notes), notes


def test_send_to_explorer_keeps_what_the_set_means(loaded):
    """Send to Explorer used to write plain IDs of unknown completeness; this fails on a
    tree without the fix."""
    import selection_sets
    ctl = loaded["ctl"]
    ctl.view.selection_panel.combo_sets.setCurrentIndex(0)
    ctl._highlight_fof_halo(1)
    path = ctl._send_set_to_explorer(launch=False)
    assert path and Path(path).parent.parent == Path(loaded["snap"]["filepath"]).parent   # the temporary copy
    st = selection_sets.read_set(path)
    assert (st.kind, st.complete, st.n_expected) == ('group', True, 34_120)


def test_sample_only_radius_selection_is_marked_incomplete(loaded, catalogue):
    """Radius criterion on the display sample ("Search full snapshot" unticked) at a budget below N.

    Known defect: while the selection criteria do not build populations, the set is plain
    IDs of unknown completeness. The test then reports an expected failure (xfail), not a
    skip, so the run shows the defect; once fixed, the assertions below run."""
    if loaded["budget"] != 500_000:
        pytest.skip("checked at the 500k budget")
    ctl = loaded["ctl"]
    sp = ctl.view.selection_panel
    centre = catalogue["pos"][0]                                   # catalogue group 1's centre
    ctl._view_centre = lambda: np.array(centre, dtype=np.float64)  # where a user would have navigated
    sp.combo_sets.setCurrentIndex(2)
    for chk in (sp.chk_ids, sp.chk_fof, sp.chk_prop, sp.chk_cat):
        chk.setChecked(False)
    sp.chk_radius.setChecked(True)
    sp.spin_radius.setValue(1.0)
    try:
        sp.chk_full.setChecked(True)
        ctl._apply_selection_criteria()
        full = ctl.model.tracker.get_set(2).id_array()
        sp.chk_full.setChecked(False)
        ctl._apply_selection_criteria()
        s = ctl.model.tracker.get_set(2)
    finally:
        sp.chk_radius.setChecked(False)
        sp.chk_full.setChecked(False)
        del ctl._view_centre
    sample = s.id_array()
    assert 0 < len(sample) < len(full)                             # the sample really is incomplete here
    assert np.all(np.isin(sample, full))
    print(f"\nradius r = 1 about catalogue group 1: display sample {len(sample):,}, full snapshot {len(full):,}")
    if s.kind == 'ids' and s.complete is None:
        pytest.xfail(f"known defect: a radius selection on the display sample "
                    f"({len(sample):,} of the {len(full):,} particles inside r = 1) is still plain IDs "
                    f"of unknown completeness")
    assert s.complete is False
    assert s.completeness_warning() and "display sample" in s.completeness_warning()
