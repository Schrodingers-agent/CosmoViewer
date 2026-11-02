# Algorithms

Every method CosmoViewer uses, with the file it lives in, the equations it implements and the reason for the choices
that are not obvious. Figure numbers refer to [figures/](figures/). Constants are quoted from the code; if a number
here and the code ever disagree, the code is right and this page is stale.

Contents: [1 Snapshot reading and units](#1-snapshot-reading-and-units) · [2 Level of detail](#2-level-of-detail) ·
[3 Friends-of-Friends](#3-friends-of-friends) · [4 Selections and tracking](#4-selections-and-tracking) ·
[5 Measurements](#5-measurements) · [6 The Explorer's histogram index](#6-the-explorers-histogram-index) ·
[7 Smoothed images and movies](#7-smoothed-images-and-movies)

---

## 1 Snapshot reading and units

**Files.** `snapshot_io/snapshot_loader.py`, `gadget_file_discovery.py`, `Dataset Explorer/io_manager.py`,
`Dataset Explorer/data_utils.py`, `model/units.py`.

A multi-file snapshot `<base>.<i>.hdf5` is one snapshot. The set is accepted only if every piece named by
`NumFilesPerSnapshot` exists; a partial set is refused, because a partial set gives wrong totals with no error.
Particle counts come from `NumPart_Total` with `NumPart_Total_HighWord`, so counts above 2³² are exact.
The Explorer addresses particles by a global row through per-group prefix sums over the pieces
(`VirtualSnapshotSet.read_particle`).

Masses come from the `Masses` (or EAGLE `Mass`) dataset, or from `Header/MassTable` when Gadget omitted the array
because a type has one mass. `MassTable` has no unit attributes, so its conversion is borrowed from another type's
mass dataset in the same file, and the result records that it did.

Physical values are built from the file's own attributes:

$$x_{\rm phys} = x_{\rm stored}\; a^{\,a_{\rm scaling}}\; h^{\,h_{\rm scaling}}\; \texttt{to\_cgs}$$

with the EAGLE names (`aexp-scale-exponent`, `h-scale-exponent`, `CGSConversionFactor`) accepted as aliases.
Whether a run is comoving is read from the header (`ComovingIntegrationOn`), never from a setting. Gadget-4 writes
`h_scaling = 0` when comoving integration is off, and isolated runs are treated accordingly. When a file states no
units the values are labelled *assumed*.

Two constants differ from IAU/CODATA values on purpose. Gadget-4 uses 1.989 × 10⁴³ g for 10¹⁰ M☉ (3.0 × 10⁻⁴
heavier than the IAU solar mass) and G = 6.6738 × 10⁻⁸ cgs, H₀ = 3.2407789 × 10⁻¹⁸ h s⁻¹. Reported masses use the
IAU M☉; the FoF linking length and the sanity report's mass budget use Gadget's G and H₀, which makes the
DM-L50-N128 budget exact to five digits (with astropy's G it is off by 7.5 × 10⁻⁵).

Stored Gadget velocities are w = v_pec / √a in comoving runs. Measurements convert them to peculiar km/s in both
display frames; the Hubble flow is never added, and results say so.

## 2 Level of detail

**Files.** `lod/store.py`, `lod/cells.py`, `lod/view.py`, `lod/budget.py`, `lod/frustum.py`. **Figures** 4 and 7.

**The hash.** Each particle gets a fixed number u ∈ [0, 1): the splitmix64 hash of its ParticleID (xored with a seed),
top 24 bits. The same particle gets the same u in every snapshot wherever Gadget writes it, so a movie or the time
slider draws the same particles in every frame (whole-box views: 99 % of drawn particles shared between consecutive
DM-L50 snapshots, against 11 % with row-based sampling). Files without ParticleIDs fall back to a generator seeded by
(seed, piece, type, block start). 24 bits give about 48 distinct levels; `MAX_LEVEL = 80` is headroom.

**The store.** Particles with u < φ, φ = min(1, `STORE_CAP` / N) with `STORE_CAP` = 16 M, are kept, sorted by octree
leaf cell (128³ leaves, `MAX_DEPTH = 7`) and by u inside each cell. Taking the first part of every cell is then the same
as a threshold on u, so every level is a set of contiguous slices and a finer level contains every particle of a
coarser one: zooming in never pops. Stores of 8 M particles or more (`DISK_CACHE_MIN`) are cached as memory-mapped
`.npy` files in `~/.cosmoviewer/lod_cache`, keyed by path, size and mtime, under an 8 GB least-recently-used limit.
Positions are float32: this is display data only.

**The view rule.** For a camera at distance D from the point it orbits, a particle at distance d from the eye is drawn
when u < f(d):

$$f(d) = \min\!\left(1,\; f_{\rm focus}\left(\frac{D}{\max(d, D)}\right)^{2}\right)$$

Everything up to the focus is drawn at the same fraction, so the density there is faithful; beyond it the fraction
falls as d⁻², like the screen area a region covers. f_focus is the finest level 2^(−k/2) whose expected count fits the
budget X (default 2,000,000). When the focus is already at full resolution and budget is left, a second, wider falloff
spends it on the background from the store alone. A cell whose nearest point needs f above φ is read from the
snapshot files instead (only blocks that touch the view), so every particle has exactly one source.

**Why the fraction is the same in every visible cell.** The drawn particles are then a uniform sample of what is in
view, so on-screen density stays proportional to true density. Cells exist only to skip what is off screen and to
read contiguous runs.

**The rule that matters most.** Nothing in this section feeds a number. FoF, tracking and every measurement read
full-resolution data (section 3 onwards). Figure `docs/images/display_sample_bias.png` shows why: a halo measured from a
fraction s of its particles keeps its half-mass radius and dispersion within ~3 % but its mass falls as s and its
virial ratio rises as 1/s.

## 3 Friends-of-Friends

**Files.** `snapshot_io/fof.py`, `fof_snapshot.py`, `fof_process.py`, `fof_finder.py`. **Figure** 5.

**Definition** (Davis et al. 1985, Gadget-4 conventions from `src/fof/fof.cc`). Primary particles closer than the
linking length l are friends (minimum image in a periodic box); groups are connected components; groups below
`FOF_GROUP_MIN_LEN` = 32 are dropped; groups are numbered from 0 by decreasing length.

$$l = b\,\langle d\rangle,\qquad \langle d\rangle = \left(\frac{\bar m}{\rho_{\rm link}}\right)^{1/3},\qquad
\rho_{\rm link} = \begin{cases}(\Omega_0-\Omega_b)\,\rho_{\rm crit} & \text{run has gas}\\ \Omega_0\,\rho_{\rm crit} & \text{otherwise}\end{cases}$$

with b = 0.2 and m̄ the mean primary mass. For an equal-mass dark-matter box this is (L³/N)^{1/3}. Only
`FOF_PRIMARY_LINK_TYPES` (bit mask, default 2 = PartType1) link. On runs that do not write Ω and units in a
`Parameters` group (EAGLE, AGORA, Gadget-2/3, zoom-ins) the default cannot be formed; the user types l
(`docs/fof_linking_length.md`).

**Full resolution.** `fof_snapshot` reads Coordinates, ParticleIDs and Masses of the primary types straight from the
files and returns membership keyed by ParticleID. Running on the display sample would change both ⟨d⟩ and which
particles can link.

**Grid linker.** Space is cut into cubic cells of side s ≤ l/√3, so all particles in one cell are friends and are
joined outright. Cell pairs are then joined across neighbour offsets, nearest first, and a pair is examined only while
its two cells are still in different groups. Inside haloes almost every pair is already joined after a few offsets,
which bounds time and memory. Remaining pairs are tested particle by particle, first on a few particles per cell and
then exhaustively, so the result is exact. A pair of crowded cells (more than 2¹⁸ particle pairs) is tested with a
k-d tree. Memory is about 16 bytes per particle plus ~100 per occupied cell. A k-d tree linker is kept only as a test
cross-check.

**Isolation.** The run happens in a child process (`python -m snapshot_io.fof_process`). Its resident plus
swapped memory is polled several times a second and the child is stopped above 60 % of physical memory
(`COSMOVIEWER_FOF_MEMORY_FRACTION`); a crash, a kill by the system or a cancel is reported and the viewer carries on.
Results return as `.npy` files: sorted IDs, their group, group lengths.

**Evidence.** DM-L50-N128 z = 0: 2,774 of 2,774 groups identical to Gadget-4's catalogue; all three snapshots
8,030 of 8,032, the two differences being float32 ties at the linking length. Numbering matches up to rank 120;
beyond it equal-length groups swap. Figure 5 repeats the comparison on the 400,000-particle test cut that ships with
the repository (29 of 29 complete groups identical). FoF on 2.1 M particles: about 10–11 s, about 0.5 GB.

## 4 Selections and tracking

**Files.** `selection_sets.py` (identical in both apps), `model/simulation_model.py`, `snapshot_io/id_lookup.py`,
`snapshot_io/id_index.py`, `model/tracking.py`. **Figures** 2, 3, 8.

**A selection is a population with a record** (Fig. 3): the IDs, plus `kind` (ids, group, region, query), its
`definition`, the snapshot it was `defined_at`, `complete`, `n_expected` and `incomplete_reason`. A selection made on
the display sample is stored with complete = False and reason `display_sample`; editing it by hand cannot make it
complete. Version-1 files (IDs only) still read, with completeness unknown.

**At another snapshot** a selection means the same particles, found by ID; its definition is not re-evaluated
("where did these particles go" and "where is this halo now" are different questions, and the second needs FoF or
the catalogue at that snapshot).

**Finding IDs.** `id_lookup` reads the ParticleIDs column of every type in 4 M-row chunks and matches with a sorted
search; rows with few hits are read individually. For snapshots of 1 M to 200 M particles, `id_index` builds once a
sorted (ID, global row) table, 12 bytes per particle, cached with the LOD store, and later lookups are a binary
search plus reads of only the chunks that hold hits (DM-L50: 0.11–0.15 s by scan, 0.03–0.04 s by index once built, about 0.15 s to build). Both give
identical results, including repeated IDs (first occurrence in scan order wins) and IDs absent from the snapshot.
A gas particle that becomes a star keeps its ID and is found as PartType4.

**Follow and trails.** The view recentres on the mean position of the set, with the first member as the reference
for the periodic wrap (GadgetViewer's convention); trails are unwrapped across the box.

## 5 Measurements

**Files.** `Dataset Explorer/measurements/`. Pure Python, no Qt: the window and an exported script call the same
functions and get the same numbers (checked digit for digit).

| Measurement | Method | Notes |
|---|---|---|
| Centre (`centre.py`) | centre of mass (circular mean of Bai & Breen 2008 for the periodic reference); shrinking sphere (Power et al. 2003); density centre (Casertano & Hut 1985, kNN density); potential minimum; one particle | Shrinking sphere stops at 1,000 particles (capped at N/10): with 100 it biased r₁₀–r₅₀ high by ~1σ on a Plummer sphere. A warning fires when the density centre lands in a low-density gap between two nuclei |
| Lagrangian radii (`lagrangian.py`) | smallest particle radius where cumulative mass reaches f M | Errors: binomial σ_f = √(f(1−f)/N_eff), N_eff = (Σm)²/Σm², through dr/df; and a bootstrap. Neither includes the centre's error |
| Radial profile (`profiles.py`) | log or equal-count shells: ρ, M(<r), v_c, σ_r, σ_θ, σ_φ, β = 1 − (σ_θ²+σ_φ²)/2σ_r² | Dispersions about each shell's own mean velocity; inner region below max(2.8 ε, Power radius) shaded as unresolved |
| Energy (`energy.py`) | K = ½Σmv², U = Σmu, W = −½ΣΣ G mᵢmⱼ g(rᵢⱼ) with Gadget's spline softening, max(εᵢ, εⱼ) | W direct up to 30,000 particles; above that an unbiased Monte Carlo estimate from 4 M random pairs with its standard error. Pairs are drawn on ID order, so the same pairs recur in every snapshot and the drift ΔW is far more precise than W. No W in periodic boxes (that needs Ewald summation) |
| Virial ratio | Q = K/\|W\| | 0.4996 ± 0.0002 on Plummer spheres, N = 10⁵ |
| Phase diagram (`phase.py`) | mass- or count-weighted 2D histogram over every gas particle | bins + out-of-range + dropped = total, exactly |
| Sanity report (`sanity.py`) | nine families: counts, finite, positive, ids, positions, box, velocities, mass budget, softening | One pass over every particle, bounded memory except the ID and position checks. Catches all 15 planted defects |
| Run sets (`runset.py`, `cache.py`) | one measurement over many snapshots and runs | Results cached as JSON keyed by arguments and each input file's path, size and mtime |

Results carry their provenance (`results.py`): a result counts as science only when every particle was used;
otherwise its status line says PREVIEW and the fraction used.

## 6 The Explorer's histogram index

**Files.** `Dataset Explorer/plugin_base.py`, `compute_worker.py`, `lod_index.py`.

The Explorer draws a field's histogram from a strided first pass (LOD-1, at most 25 M particles), then builds in the
background a CSR index from histogram bins to *global* HDF5 rows (LOD-2): `indices[offsets[b]:offsets[b+1]]` are the
rows in bin b. The build is two parallel passes over the files (count, then fill non-overlapping ranges without
locks), cached beside the LOD store. Clicking a bin then lists its actual particles, read back from the file.

Bin counts follow Freedman–Diaconis, computed on log₁₀ of the values when the axis is logarithmic (on linear values
FD hits its cap and leaves the sparse tail empty), capped at the Rice rule for small sets. Integer fields get bins
that contain integers, and zero-heavy positive fields keep the log axis with the zeros counted in the status line.

## 7 Smoothed images and movies

**Files.** `model/sph_projection.py`, `movie/interpolate.py`, `movie/maker.py`, `movie/keyframes.py`.

**Smoothed map.** Each particle's mass is spread with the 2D cubic-spline (M4) kernel, support 2h, normalised per
particle over the pixels it covers so mass is conserved at any resolution. Gas uses its SmoothingLength; other types
get h = r₃₂/2 from the 32nd nearest neighbour. Large kernels are deposited on coarser copies of the image and
upsampled, bounding cost to about (2R+1)² per particle. The map is built from the display sample with masses scaled
by N/N_sample: a picture, labelled as one.

**In-between frames.** Between snapshots A and B each particle follows a cubic Hermite curve through its two
positions with its two velocities as tangents, matched by ID; B's position is first moved to the image nearest A's.
The parameter is the snapshot time variable (a in comoving runs), with dx/da = w a^{s−2}/H(a), s = 0.5 for Gadget-4's
stored velocities. Tangents are shortened when particles turn faster than the snapshot spacing resolves. Every
in-between frame is labelled and none feeds a measurement.

**Camera paths.** Keyframes (view plus snapshot) are joined by a spline; the colour range is fixed from the first frame
or typed, so frames stay comparable.
