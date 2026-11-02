# measurements: the Explorer's measurement layer

Status as of 2026-10-18. The work came in two stages: the first (snapshot access, centring, Lagrangian radii, run sets) and the second (sanity report, profiles, energies, phase diagram, cache, script export).

Everything in this folder is pure Python. It uses no Qt and draws no plots. It reads files through the Explorer's own `io_manager.VirtualSnapshotSet` and converts units with the Explorer's own `data_utils` functions.

| File | Contents |
|---|---|
| `snapshot_view.py` | `SnapshotView`: one snapshot with explicit units. `ParticleSet`: particles in kpc, km/s and Msun |
| `centre.py` | `find_centre` with five methods: `com`, `shrinking_sphere`, `density`, `potential_min` and `particle`. All are periodic-aware |
| `lagrangian.py` | `lagrangian_radii`, with a binomial sampling error and an optional bootstrap |
| `runset.py` | `discover_snapshots`, `Run`, `RunSet.apply` and `ResultTable` (long format, CSV export) |
| `results.py` | `Result` and `Provenance`. A result counts as science only when every particle was used |
| `models.py` | Plummer and Hernquist samplers, planted anisotropy, and a writer for Gadget-4-style HDF5 files (tests only) |
| `sanity.py` | `sanity_report`, one pass over every particle with PASS/WARN/FAIL/INFO per check |
| `profiles.py` | `radial_profile` (ρ, M(<r), v_circ, σ_r, σ_t, β with errors) and `surface_density` |
| `energy.py` | `energy_budget` (K, U, W, E, Q, P, L), `conservation` over a run, `read_energy_txt` |
| `phase.py` | `property_histogram2d`, mass- or count-weighted, exact weight accounting |
| `cache.py` | `ResultCache`, keyed by measurement, arguments and each file's path, size and mtime |
| `scripts.py` | `script_for`, the Python that Export ▸ .py writes |

Example (headless):

```python
from measurements import SnapshotView, find_centre, lagrangian_radii, RunSet
snap = SnapshotView("CollidingGalaxiesSFR/output/snapshot_100.hdf5")
stars = snap.particles([4])
r = lagrangian_radii(stars, find_centre(stars, "density"))
print(r["radius"], r.provenance.status_line())

table = RunSet.from_folder("CollidingGalaxiesSFR/output").apply(
    lambda s: lagrangian_radii(p := s.particles([4]), find_centre(p, "density")))
table.to_csv("r_lagr_stars.csv")
```

## Acceptance criteria, first stage

| Criterion | Result | Evidence |
|---|---|---|
| SnapshotView equals the plugin path on every field of both real runs | **PASS** | `TestRealOutputs.test_every_field_*`. The arrays are bit-identical to plugin Stage 2 (`assert_array_equal`) for every dataset and vector column of DM-L50-N128 `snapshot_000` and CollidingGalaxiesSFR `snapshot_100`, in both Physical and Comoving modes |
| Plummer Lagrangian radii within 2σ at N = 10⁴ and 10⁵ | **PASS** | With the true centre: 100 seeds at N = 10⁴ (800 radii), 93–98% within 2σ (95.4% expected), and no bias per fraction. 20 seeds at N = 10⁵ also pass. The bootstrap and binomial errors agree within 25% |
| Centring recovers a planted offset within error | **PASS** | 30 seeds at N = 10⁴, unbiased for all three methods. Measured rms centre error: density 0.02 a, centre of mass 0.04 a, shrinking sphere 0.05 a |
| Periodic boxes | **PASS** | A system wrapped across a box corner gives the same centre (< 10⁻⁹ kpc) and Lagrangian radii (10⁻⁹ relative) as the unwrapped one. On DM-L50 `fof_tab_000`, our centre of mass matches Gadget-4's `GroupPos` for **all 2,225 groups** to within 4 float32 ulp, including groups that straddle the boundary |
| A headless script reproduces a GUI result | **PASS** | `test_analyses_window.py` (6 tests). The Matrix cells equal the headless result formatted the same way, string for string. The exported columns are bit-identical. Checked for a Masses type, a MassTable type and the real DM-L50 snapshot (2,097,152 particles) |

Further checks on real data (Verified):
- DM-L50 `snapshot_000`: total mass 1.5761×10¹⁶ M☉ against Ω₀ρ_crit(L/h)³ = 1.5757×10¹⁶ (0.03%). Physical box 18,461.56 kpc = 50 Mpc/h × a.
- CollidingGalaxiesSFR `snapshot_100`: 455,187 stars. Time is 1.203 code units = 1.176 Gyr. The masses of PT1–PT3 come from MassTable, converted with the attributes of another type's `Masses` dataset in the same file, and the result records this.

Run all Explorer tests: `COSMOVIEWER_DM_OUTPUT=… COSMOVIEWER_SFR_OUTPUT=… python3 -m pytest -q`. When the first stage was finished, the 60 existing tests, 18 measurement tests and 6 window tests passed (84 total, Linux container, Qt offscreen, 6 of 6 full runs).

## Findings worth knowing

1. **The centre choice biases Lagrangian radii.** A centre that is off by δ inflates r_f by roughly δ²/r_f. For a Plummer sphere with N = 10⁴, the shrinking-sphere centre biases r₁₀–r₅₀ high by about 1σ when the sphere stops at 100 particles. The fix is to stop at 1,000 particles (Power et al. 2003), capped at N/10. That cuts the centre error from 0.15 a to 0.05 a, and the bias falls below 0.4σ at N = 10⁴ and below 0.75σ at N = 10⁵. The density centre (Casertano & Hut) is the most accurate: 0.02 a at N = 10⁴ and 0.006 a at N = 10⁵. But the density centre is a *weighted mean*. On CollidingGalaxiesSFR `snapshot_100` it lands between the two galaxies, at (1.0, 0.3, 0.2) kpc, while the nuclei are at (7.9, 4.7, 0.5) and (−7.1, −4.9, −0.2) kpc. That inflates r₁ to 3.6 kpc. The shrinking sphere finds the (−7.1, −4.9) nucleus, where r₁ = 0.05 kpc. `find_centre` now warns when the density near the density centre is below 1% of the peak, and a test with two Plummer spheres checks that warning. **Proposed default: shrinking sphere everywhere, with density offered for single relaxed clusters where it is the most precise.**
2. **Gadget-4's GroupVel is not the stored velocity unit.** GroupVel equals the mean of the members' stored `Velocities` × a^1.5. Because the stored velocity is the peculiar velocity / √a, GroupVel = **a × the mean peculiar velocity**, and GroupVel / a is the group's peculiar velocity in km/s. The catalogue datasets carry no unit attributes. It matters for the halo table and is recorded so nobody reads GroupVel as km/s.
   - Checked on DM-L50 groups 1, 2 and 100 at a = 0.2503 and a = 0.4993 (`fof_tab_000` and `_001`). The ratio GroupVel / mean stored velocity equals a^1.5 to about 1×10⁻⁶.
   - **Correction (2026-10-24):** an earlier version of this finding said "peculiar × a^1.5 = stored × a²". That is off by √a, and the corrected relation was checked independently.
   - The halo table does not show GroupVel.
3. **Gadget's unit constants are not IAU values.** Gadget-4 uses 3.085678e21 cm for a kpc and 1.989e43 g for 10¹⁰ M☉. Converted to astropy kpc and M☉, lengths are 1.4×10⁻⁷ longer and masses 2.9×10⁻⁴ heavier. This is correct behaviour, and it is the reason a test that hard-codes "×1000 per Mpc" fails at the 10⁻⁷ level.
4. Velocities in cosmological runs are peculiar: the Hubble flow is not added. Every result records this.

## Acceptance criteria, second stage

| Criterion | Result | Evidence |
|---|---|---|
| The sanity report catches every planted defect | **PASS** | `TestSanityPlantedDefects`, 15 cases. Each defect (header count, dataset length, NaN, inf, negative mass and density, duplicate IDs across types, duplicate positions, particle on the box edge, outside the box, speed above c, speed outlier, MassTable and Masses both set, no mass at all, cosmological mass budget off by 5 %) is caught by its own check and by no other. A clean file has no FAIL or WARN |
| The sanity report flags the two edge particles in DM-L50 snapshot_002 | **PASS** | WARN on PartType1: index 1969905 (ID 180231) and index 2002078 (ID 545792), both with z = BoxSize = 50. An independent h5py read finds the same two. Everything else passes, and the total mass is 1.00000 × Ω₀ ρ_crit V when Gadget-4's own G and H₀ are used |
| Plummer Q = 0.50 ± 0.01 at N = 10⁵ | **PASS** | Q = 0.4993 ± 0.0002. W is within 0.13 % of −3πGM²/(32a) (Literature, from memory). Monte Carlo pair potential agrees with the exact direct sum within 3σ on 6 mixed-softening systems |
| The energy budget agrees with Gadget-4 energy.txt | **PASS** for K, U and mass; **NOT TESTED** for W | energy.txt columns verified: time; total thermal, potential, kinetic; then per type thermal, potential, kinetic; then mass per type. CollidingGalaxiesSFR snapshot_000 equals row 0 to all 8 digits for K, U, K per type and mass per type. DM-L50: K = ½ Σ m v_pec² in code units, within 5×10⁻⁴ of the file interpolated to the snapshot time (Partially verified: the file has no row at snapshot times). The potential column is 0 in both runs, so W cannot be compared |
| Plummer and Hernquist profiles within error | **PASS** | 8 seeds × 20 bins at N = 5×10⁴: density pulls have mean ≈ 0 and spread ≈ 1 for both models; σ_r of Plummer likewise; β = 0 for the isotropic Plummer and β = 0.75 for a planted anisotropy; projected Plummer Σ(R) within error. β errors are conservative (spread ≈ 0.8) because they ignore the σ_r–σ_t correlation. Osipkov–Merritt β is NOT TESTED |
| Cache: cached = uncached; touching one file recomputes only its rows | **PASS** | `TestResultsCache`: 3 runs × 2 snapshots, every row identical; after touching one file, 1 miss and 5 hits |
| Script export: the exported script reproduces the GUI numbers exactly | **PASS** | `test_*_script_reproduces_the_window`: the script runs in a fresh Python and its CSV equals the window's export digit for digit (profile and energy; sanity counts too) |
| Phase diagram: weighted totals exact; ρ–T equals an independent h5py computation | **PASS** | Bins + out of range + dropped = total. On CollidingGalaxiesSFR snapshot_100 every bin equals an h5py-only computation to 10⁻¹² |

Tests: `test_profiles_energy_sanity.py` (32) and `test_analyses_window_profiles_energy.py` (12). With the existing suites, 140 passed and 1 skipped (a permissions test that cannot run as root), in 4 of 4 full runs (Linux container, Qt offscreen).

### What the second stage found

1. **Gadget-4's G and H₀ are not CODATA values.** Gadget-4 uses G = 6.6738×10⁻⁸ and H₀ = 3.2407789×10⁻¹⁸ h/s (`src/data/constants.h`, read on the Mac copy). With astropy's G the DM-L50 mass budget is off by 7.5×10⁻⁵; with Gadget's, it is exact to 5 digits. The sanity report uses Gadget's constants.
2. **No snapshot stores `Potential`, and energy.txt's potential column is 0.** So W is computed: exactly by direct sum up to 30,000 particles (about 10 s), and above that by a Monte Carlo estimate from 4 million random pairs with its standard error (0.2 % on CollidingGalaxiesSFR, 3 s). The pairs are the same in every snapshot of a run while the particles stay the same, so energy drift is much more precise than W itself. An exact tree potential for large N would need a new dependency (pytreegrav with numba, or similar); that is not added.
3. **energy.txt in a cosmological run reports the peculiar kinetic energy** ½ Σ m v_pec², i.e. stored velocity² × a (Partially verified by interpolation).
4. CollidingGalaxiesSFR loses 3.8 % of its total energy between t = 0 and t = 1.2. That is expected physically: the gas cools and forms stars. The window says so when gas is present.

## In the window

The tree has a new **Analyses** group with the leaves "Lagr. radii, all types" and "Lagr. radii, PartType0" to "PartType5". A leaf is greyed out when the open file lacks that type. Each leaf has a **Matrix** tab (fraction, radius, sampling error; Export writes a provenance header) and a **Profile** tab (cumulative mass curve with a marker at each radius). **Options ▸ Centre** picks shrinking sphere (the default), density-weighted or centre of mass, and recomputes. The status bar says how many particles were used and which centre. An analysis never samples: a selection over 50 million particles is refused with a message.

Code: `analysis_plugins.py` (new). Small edits to `density_explorer.py`, `plot_panel.py`, `compute_worker.py` and `plugin_registry.py`. `conftest.py` (new) makes one QApplication for the whole test session and settles Qt events after each test. Without it, about 1 full run in 4 crashed in the first window test. The cause is not proven (Partially verified).

### Second-stage leaves in the window

**Sanity report** (Report tab: status, check, scope, particles, detail; PASS/WARN/FAIL coloured), **Energy, all types** (Budget tab), **Energy over run** (Energy drift, Virial ratio and Angular momentum tabs, over the snapshots in the open file's folder; more than 60 are thinned to 60 evenly spaced, every particle in each; results are cached in ~/.cosmoviewer/results_cache), **Phase diagram, gas** (mass-weighted ρ–T), and **Profile, all types / PartType0–5** (Density, Enclosed mass, Dispersion and Anisotropy tabs; the unresolved region inside 2.8 ε is shaded). Each Matrix tab holds the full table. Export can save it as CSV, or as a Python script that reproduces it. The active selection set restricts profiles, energies and the phase diagram; the sanity report always checks the whole file. The Anisotropy tab is drawn on a linear y axis whatever the Y setting, since β can be negative.

Code: `analysis_plugins_m2.py` (new). Small additive edits: `cells` and `script` on `_MatrixRequest` (`compute_worker.py`); text tables and the .py export in `plot_panel.py`; `rect` on `ImageSpec` (`plugin_base.py`) so a map sits in axis units; two hooks in `density_explorer.py`; `**ANALYSES_M2` in `plugin_registry.py`.

## Next

- A general property-vs-property window (any two fields, sets overlaid), reusing Immersive's Make plot window. The phase diagram is its first fixed case.
