# CosmoViewer Immersive: performance and level of detail

## The problem

A Gadget/EAGLE snapshot can hold close to a billion particles (EAGLE RefL0025N0752 holds 850,512,099 across 16 files, about 58 GB).
Before this work the viewer read a fixed strided sample once, using `step = N / --max-particles`.

- Zoomed out, the sample is fine.
- Zoomed in, it stays the same few thousand points, so small structures dissolve into noise.
- Loading the first sample of EAGLE took 32.5 s.

## What the viewer does now

**There is a fixed on-screen budget, X = 2,000,000 particles.** The viewer always tries to draw about X particles chosen for the current view. Zooming in replaces distant particles with more of the particles near the camera. At a deep enough zoom it draws every particle in the region (full resolution).

### Why X = 2M

- It is GadgetViewer's own default sample size, so the two apps look comparable.
- At 2M points, a full scene rebuild (colours, sizes, selection) costs 140–260 ms of CPU. The GPU upload (`setData`) costs about 10 ms. Rotating the camera re-uploads nothing, because pyqtgraph keeps points in VBOs. Those numbers come from the container; Mac frame rates are listed under "Not verified".
- Memory is about 41 B per drawn particle, so the drawn set is roughly 80 MB.

To change X, use View → Particle budget (0.5M / 1M / 2M / 5M) or `--max-particles N` on the command line. Both set the same budget; there is no second flag.

### Levels

There are 81 nested levels. Level k keeps a fraction f_k = 2^(-k/2) of the particles, two levels per octave.

- Every particle gets a fixed random number u, a hash (splitmix64) of its ParticleID. The same particle therefore has the same u in every snapshot of a run, however Gadget orders the files, so movies draw the same particles in every frame. Files without ParticleIDs fall back to a number from (seed, file piece, type, block).
- Level k is exactly the set of particles with u < f_k. Because of this, every level is an unbiased uniform sample, and each finer level contains the coarser one, so zooming only ever adds points.
- The viewer picks the finest level whose expected visible count fits the budget, using bisection.

### Depth-aware detail

At the focus the level fraction is f. Elsewhere it is f · (D / max(d, D))², where d is the particle's distance from the eye and D is the camera distance.

Without this falloff, a deep zoom still pays for every particle behind the focus and never gets sharp. With it, the 0.004-box zoom on EAGLE reaches f = 1 at the focus.

### Where particles come from

1. **LOD store.** A uniform sample of up to 16M particles (u < s, with s = 16M/N), sorted by octree cell and then by u. Any coarse view is a prefix of each visible cell, so no file reads are needed.
   - For snapshots of 8M+ particles, the store is cached on disk at `~/.cosmoviewer/lod_cache` as memory-mapped `.npy` files, about 0.8 GB at the 16M cap (each row also keeps its u and its rank in file order). The oldest stores are evicted above 8 GB.
   - The cache location and limit can be changed with `COSMOVIEWER_LOD_CACHE` and `COSMOVIEWER_LOD_CACHE_GB`.
   - The cache key includes each file's size and modification time, so a rewritten snapshot is re-indexed.
2. **File detail reads.** Where a view needs more than the store's fraction, the viewer reads those rows from the HDF5 files.
   - It only reads blocks of 16,384 rows whose bounding box and 4×4×4 occupancy map touch the view.
   - Large reads (over 2M rows) are spread over worker processes: half the cores, at most 4, at low priority (`nice 10`), in reads of at most 1M rows. They are started with `spawn` because the reads begin in a Qt thread, and they are stopped after 30 s without reads to give their memory back.
   - Building the store uses the same worker count and priority.

Every query runs in a background thread. It starts after the camera has been still for 120 ms and is skipped during playback. A newer camera move cancels it, and the window never waits for a cancelled job.

A settled camera keeps the particles already drawn when they still serve it. The previous set is kept when all of these hold:

- The level is within one step (a factor of √2 in density).
- The eye moved less than a quarter of the camera distance.
- The screen lies inside the region that was loaded.

Small rotations and pans then cost no reload, no redraw and no colour re-read. Resample view now (Ctrl+R) always makes a new set.

### Effect on the rest of the viewer

- **Science never sees LOD.** FoF, tracking, property plots and every number the app computes read full-resolution data or the base sample, never the view set. The view set is a shallow display copy (`_display_snapshot`), and the base snapshot arrays are never replaced. FoF colours on a view set are looked up by particle ID from the full-resolution group finder.
- **Movies** render from the base sample, so frames are reproducible.
- **Base sample.** Every snapshot keeps a base sample of up to X particles for playback and for the analysis features. Its size is capped so the whole series fits in about 1.5 GB: clamp(1.5 GB / (41 B × snapshots), 50k, X).
  - Snapshots over 8M particles take their base sample from the LOD store. That sample is the same uniform u-sample, not a stride.
  - `snap['rows']` records exactly which file rows were loaded, for the GadgetViewer features.
- **Controls** live in the existing View menu: Particle budget, Level of detail (on/off), and Resample view now (Ctrl+R). The status bar shows e.g. `Detail level 7: 8.8% of particles drawn near the focus (~1.56M in view, 1.7 s)` (until 2026-10: `LOD 7: 0.088 of the particles at the focus (~1.56M in view, store+file, 1674 ms)`).

## Measurements

Environment labels:

- **Mac VM**: the Linux VM on the M3 Mac (4 cores, 3.9 GB RAM, Python 3.10), reading the real files from Downloads.
- **Container**: the cloud container (4 cores, Python 3.13).

Neither is the native macOS app.

### Loading

| Case | Before | After | Env |
|---|---|---|---|
| EAGLE 850M, first sample (150k) | 32.5 s | store build 59.8 s **once**, then 0.1 s reopen; base sample 0.19 s | Mac VM |
| EAGLE store build peak memory | — | 1.3 GB (cap 8M because of the VM's RAM); 2.1 GB at the app's 16M cap on the native Mac (1 Nov 2026) | Mac VM |
| DM-L50-N128 / CollidingGalaxiesSFR load | 0.05–0.07 s | unchanged (below 8M, no store) | Mac VM |
| Synthetic 50M, 8 gzip pieces | — | build 21 s, peak 2.3 GB, reopen 0.0 s | Container |

### Zooming on EAGLE (850M particles, budget 2M, camera on the densest cell)

Measured in the Mac VM with the final code from the Mac copy. The store cap was 8M (the real app uses 16M, which halves file reads at the middle zooms).

| Camera distance | Level | Fraction at focus | Drawn | Rows read from files | Query time | Drawn, re-measured 1 Nov 2026 |
|---|---|---|---|---|---|---|
| 1.5 box | 18 | 0.0020 | 1.46M | 0 | 0.34 s | 1.46M |
| 0.5 box | 14 | 0.0078 | 1.79M | 0 | 0.37 s | 1.78M |
| 0.2 box | 11 | 0.022 | 1.55M | 48.8M | 2.9 s | 1.56M |
| 0.08 box | 9 | 0.044 | 1.60M | 38.9M | 2.0 s | 1.62M |
| 0.03 box | 7 | 0.088 | 1.56M | 33.1M | 1.7 s | 1.56M |
| 0.01 box | 3 | 0.35 | 1.60M | 20.6M | 1.1 s | 1.97M |
| 0.004 box | 0 | **1.0 (full resolution)** | 0.59M | 19.2M | 1.0 s | 1.99M |

Peak memory during the zoom sequence was 1.0 GB.

Re-measured on 1 Nov 2026 with `tools/bench_lod.py` at the app's 16M cap on the native M3 (16 GB, heavily loaded): 1.46M–1.99M drawn at every distance, queries 0.3–2.2 s. The 0.59M at 0.004 box above predates the deep-zoom fix, which now fills the budget at full resolution.

### Tests

- Full suite in the container (18 Oct, after the latency, same-particle and ParticleID-index changes): 206 passed, 18 skipped, 0 failed.
- Tests that pinned the old every-step-th sample (`test_multifile_eagle.py`, `test_tracking_app.py`, `test_lod_app.py`) now check the ID rule exactly instead (same strictness, new rule).
- `tests/test_lod.py` and `tests/test_lod_app.py` check the following:
  - The stored sample is exactly u < s.
  - Levels are nested and uniform.
  - Culling never drops an on-screen particle, for three cameras.
  - File detail matches the rule exactly, and parallel reads equal serial reads.
  - The frustum matches pyqtgraph's own matrices.
  - Budget and source switching work.
  - The disk cache detects changed files.
  - FoF colours use full-resolution groups, and the base sample is untouched.
  - Disabling LOD restores the base sample.
  - Real-snapshot checks run when `COSMOVIEWER_DM_SNAPSHOT` or `COSMOVIEWER_G4_OUTPUT` is set.
- `test_lod_view_set_keeps_features_aligned` (in `tests/test_gadgetviewer_features.py`) passes.

### Freeze on zoom (reported 18 Oct, measured natively on the M3 Mac)

**Report.** Zooming with LOD on froze the window and the whole laptop.

**Cause.** The laptop-wide freeze came from memory pressure:

- The Mac was already 10.6 GB into swap before CosmoViewer started.
- Every settled camera made a new particle set.
- Each new set started work that piled up: reads on every core but one, and, when colouring by a property, a fresh read of that property per set. The second part has since been fixed in the colouring code.

**Results.** Old against new code on EAGLE, measured in the native macOS app with Python 3.14 and pyqtgraph 0.14:

| | Old | New |
|---|---|---|
| Peak memory, all Python processes, 20 s of rapid zooming (125 moves) | 4.4 GB | 3.1 GB |
| Swap added during that test | +2 GB | none |
| Small rotations (3°) that kept the drawn set | 0 of 5 | 4 of 5 |
| Longest window pause | 0.3 s | 0.3 s |
| Query time per new view | 1.2–2.9 s | 1.2–2.3 s |
| Memory after 45 s idle | 1.17–1.42 GB in the main process, plus ~0.6 GB in workers | 0.87–0.93 GB, workers gone (3 of 4 runs) |

Re-measured on 1 Nov 2026 (same machine, fresh caches, about 19 GB of swap in use by other work): peak 3.5–4.0 GB across all processes (main process 2.7–2.8 GB), 1.7–2.2 GB after 45 s idle with the workers gone, no set drawn while the camera moved, longest window pause 74–273 ms. The 0.87–0.93 GB idle figure above was not reproduced.

**Caveats.**

- Neither version froze the window itself in the scripted tests. "Window freeze fixed" is therefore Partially verified, while the memory and swap improvement is Verified.
- In one of the four idle runs, the workers were still alive after 45 s. That run was swapping heavily, so a query was probably still running.
- The remaining 100–250 ms pauses after each new set come from the scene rebuild (`_build_scene`) and the z-slicer's 2D scatter, both in the colouring code.

### Latency when exploring (reported 18 Oct, second report)

**Report.** With LOD on, exploring felt slower than with LOD off: each load of new particles, even 200 ms, is noticeable.

**Where the time goes** (container, 50M-particle synthetic snapshot, budget 2M, offscreen Qt):

| Step | Time | Thread |
|---|---|---|
| Camera stop to query start | about 150 ms | waits for the camera to settle |
| Store query | 0.6–1.0 s | background |
| Drawing the new set | 185–870 ms, about 600 ms on average | window (this is the pause you feel) |

In the native macOS app (1 Nov 2026) a new set is applied in 12–45 ms and the longest main-thread stall per step is 27–132 ms, so the 600 ms above is a container figure. Only the default colouring was timed natively.

The pause in the window is the scene rebuild in the colouring code: `_build_scene` 270 ms, an unconditional white colour array 146 ms, the z-slicer's 2D scatter 154 ms, float64 to float32 conversion 65 ms. All of that is outside the LOD code.

**Changes in the LOD code.**

- A new set that arrives while the camera moves, or while a mouse button is held, is not drawn. It is drawn once the camera stops, or replaced by a fresh query for where it stopped (which may keep the held set). The pause can therefore no longer land in the middle of a drag or a scroll (test `test_new_set_waits_until_the_camera_stops`).
- The store query orders its rows by a rank stored with each row instead of sorting, and store and file particles are merged without a sort. In the container this cut the store lookup by about 20–30%; the container's timings vary by ±40% from run to run, so treat this as modest.

### Same particles in every snapshot

- u is now the hash of the ParticleID (see Levels). The store keeps u with each row, and a cell's particles are those with u below the cell's fraction exactly, not a rounded count.
- The base sample of a big snapshot is the n particles with the smallest u. For a small snapshot (under 8M particles) the loader keeps the particles with u < budget / N instead of every step-th row, so its size is the budget ± about 1%, not exactly the budget.
- Check on DM-L50 (2,097,152 particles, snapshots 0–2, Mac VM), old code against new:

| | Old (row-based u) | New (ID-hash u) |
|---|---|---|
| Loader sample of 500k, identical between snapshots | no | yes |
| LOD store and its 200k base sample, identical between snapshots | no | yes |
| Whole-box view at the same level: share of drawn particles also drawn in the next snapshot | 11% | 99% |

  In the new code the remaining 1% are particles that moved out of the view (442) or into a cell whose depth weight differs (1,581). At a 0.15-box zoom, 99% of the particles drawn only in one snapshot had moved out of the view.
- The cache format is now version 3: existing stores are rebuilt once on first open, and old ones are evicted under the size limit like any other.

### ParticleID index for tracking

- `snapshot_io/id_index.py` builds, on the first tracking lookup in a snapshot, the IDs sorted with their rows (12 bytes per particle) and keeps them in the LOD cache under the same 8 GB limit. Later lookups use a binary search and read only the file chunks that hold a requested particle.
- Results equal the chunked scan exactly, including repeated IDs and IDs absent from the snapshot (`tests/test_id_index.py`).
- Snapshots below 1M particles do not need it. Snapshots above 200M particles (`COSMOVIEWER_ID_INDEX_MAX`), or whose index would take more than half the cache limit, keep the scan. **EAGLE (850M particles, sparse 64-bit IDs) is one of those:** its index would take about 10 GB. Tracking on EAGLE is therefore unchanged by this.

### Deep zoom and navigation (reported 19 Oct)

**Report.** Zoomed in far, LOD stopped loading and the scene stopped brightening. The camera only zoomed toward the centre, with no way to pan, so the edges of the box could not be approached.

**Deep zoom: cause.** Once the focus is at full resolution (level 0), the depth falloff still thinned everything behind the focus as (D/d)², and with a tiny camera distance D almost nothing behind the focus was drawn. Most of the budget went unused: 16,478 particles drawn for a budget of 2M at a 0.001-box zoom on the 50M synthetic snapshot.

**Deep zoom: fix.** At level 0, the spare budget now goes to the background. A second, wider falloff draws particles out to the radius that fills the budget, at most at the store's density, so the background costs no extra file reads. The rule stays exact and nested; the focus is still full resolution.

| 50M synthetic, budget 2M | Drawn before | Drawn after | Query after |
|---|---|---|---|
| Densest cell, 0.03 box | 552,662 | 1,987,260 | 2.9 s (first read of the region) |
| Densest cell, 0.001 box | 16,478 | 1,399,511 | 1.1 s |
| Densest cell, 0.0001 box | 254 | 39,474 | 0.5 s |
| Near the edge, 0.003 box | 12 | 23,652 | 0.3 s |

**Navigation.** In the 3D view (`views/view_extras.py`, `CosmoGLView`):

- The scroll wheel zooms toward the point under the cursor, which stays put on screen, so any corner can be approached directly.
- Shift-drag or right-drag pans. Plain drag still orbits, and Cmd-drag and middle-drag still pan as before.
- Cmd-scroll still changes the field of view.

Movie keyframes take the camera as it is, so they follow these moves.

**Limit.** The camera centre is single precision (pyqtgraph's QVector3D). Beyond a zoom of about 10⁻⁵ of the box, the view starts to step by a pixel at a time.

**Tests.** `tests/test_navigation.py` checks:

- the point under the cursor stays put through a 200× zoom, in both projections;
- Cmd-scroll changes the field of view;
- right-drag and Shift-drag pan without orbiting, and plain drag still orbits.

`test_deep_zoom_uses_the_budget_behind_the_focus` checks the deep-zoom fix.

**Suite.** In the container the full suite gives 229 passed, 18 skipped. The intermittent offscreen OpenGL crash in the stereo tests also happens on the unchanged app: 1 of 3 runs on each.

### FoF on big snapshots (reported 19 Oct)

**Report.** FoF kept crashing on the Agora snapshot, using too much memory. Wanted: a progress bar, and crash protection even at some cost in speed.

**Cause (measured).** Two crowded neighbouring cells that were not yet joined were tested pair by pair in one go. That needs memory proportional to the product of their counts. Two cells of 8,000 particles each took 6.3 GB and 40 s. With 20,000 each it would need about 40 GB. Such cells are what a galaxy disc or a merger produces. FoF also ran inside the viewer, so running out of memory ended the whole app. The Agora snapshot itself was not found on the Mac; see Not verified.

**Fix.**

- **Own process.** FoF now runs in its own process (`snapshot_io/fof_process.py`).
  - If FoF fails, crashes or is killed by the system, the viewer shows the reason and carries on.
  - The process's memory, including swapped memory, is checked several times a second. It is stopped at 60% of physical memory (`COSMOVIEWER_FOF_MEMORY_FRACTION`) before the computer starts thrashing.
- **Crowded cells.** Crowded cell pairs are tested with a k-d tree, with the same answer as testing every pair (`_cells_touch`). Rounding ties at exactly the linking length are re-checked pair by pair.
  - 8,000 per cell: 2 MB and 0.01 s, previously 6.3 GB and 40 s.
- **Reading and linking.**
  - Particles are read into arrays allocated once.
  - The grid linker sorts positions in place and builds keys one axis at a time.
  - Masses are freed once the linking length is known.
- **Progress.** The FoF window has a progress bar showing reading, linking and numbering, plus a Cancel button. Quitting the viewer stops a running FoF.
- **Lookups.** The result comes back sorted by ID, so colouring LOD particles by group no longer sorts all IDs on every lookup.

**Results unchanged.**

- On 8M-particle synthetic snapshots (one clustered, one uniform), the groups, lengths, linking length and ID lookup are identical to the old code.
- In 72 linker cases (periodic and open, with crowded cells, and with every remaining pair forced through the tree), partitions are identical to the old code. The k-d tree reference linker agrees too.

| 8M particles, in the container | Old peak | New peak | Old time | New time |
|---|---|---|---|---|
| Clustered | 1.59 GB | 0.92 GB | 55 s | 52 s |
| Uniform (worst case for the grid) | 1.81 GB | 0.93 GB | 90 s | 79 s |

Per particle, this is 217 B before and 107 B now, not counting imports. The up-front check now assumes 120 B (`FOF_BYTES_PER_PARTICLE`). Linking 8M uniform particles on the native M3 (1 Nov 2026) took 50 s and 128 B per particle, so the check can underestimate by about 15 %. With 16 GB, it allows about 67M linking particles. EAGLE's 425M dark-matter particles would need about 51 GB, so FoF on EAGLE is still refused, with a message, on a 16 GB Mac.

**Tests.** `tests/test_fof_process.py` checks:

- the process gives the same groups and lookups as an in-process run;
- the memory stop, a system kill, a failure and cancel are each reported, with no process left behind;
- two cells of 20,000 particles each need under 64 MB;
- the crowded-cell test is exact;
- the window shows progress, gets the groups, and reports failure and cancel.

**Suite.** In the container the full suite gives 243 passed, 18 skipped.

## Acceptance criteria

| Criterion | Result | Evidence |
|---|---|---|
| Many LOD levels | PASS | 81 nested levels, two per octave (tests) |
| About X particles on screen at every zoom | PASS | 1.46M–1.79M drawn for X = 2M from 1.5 to 0.01 box on EAGLE. At 0.004 box full resolution is reached at the focus (0.59M drawn in the VM run; 1.99M since the deep-zoom fix, re-measured 1 Nov 2026) |
| Sharp when zoomed in | PASS | Full resolution (f = 1) at the focus at 0.004 box on EAGLE |
| X chosen and justified | PASS | 2M, see above. Settable from the menu and `--max-particles` |
| Scientific results never use LOD | PASS | Tests: FoF/full-resolution groups, base sample untouched, movies use the base |
| Bounded memory | PARTIAL | Native Mac, EAGLE: 3.5–4.0 GB peak during rapid zooming, 1.7–2.2 GB idle (re-measured 1 Nov 2026; first measured as 3.1 and 0.9 GB). The store build peaks at 1.3–2.3 GB (2.1 GB at the 16M cap) |
| GUI stays responsive | PARTIAL | Native Mac, EAGLE: longest pause 0.3 s during rapid zooming. Machine-wide load cut (see Freeze on zoom). Frame rate itself not measured |
| Existing features (movies, tracking, FoF) keep working | PASS | Full suite, as above |
| Frame rate at 2M points on the M3 GPU | NOT TESTED | Needs the app run natively on the Mac |
| No pause while the camera moves | PASS (container) | New sets wait for the camera to stop (test). Not yet felt on the Mac |
| Pause when a new set lands under 200 ms | PARTIAL | About 600 ms in the container; under 140 ms in the native macOS app (12–45 ms to apply a set, 1 Nov 2026). Colouring by a property not timed natively |
| Same particles in consecutive snapshots | PASS | DM-L50 snapshots 0–2 on the Mac (VM): see the table under "Same particles in every snapshot". Also a reshuffled synthetic snapshot gives identical sets (test) |
| ParticleID index equals scan | PASS on DM-L50, not applicable on EAGLE | DM-L50 snapshots 0–2, 50,000 random IDs plus 2 absent: identical results; lookup 0.11–0.13 s by scan, 0.03–0.04 s with the index (built in about 0.15 s). EAGLE gets no index (too big for the cache) |
| FoF cannot crash the viewer | PASS (container) | Separate process; memory stop, kill, failure and cancel each reported (tests) |
| FoF progress bar | PASS (container) | Progress through reading, linking and numbering in the window (test). Not yet seen on the Mac |
| FoF groups unchanged | PASS | Identical to the old code on 8M-particle snapshots and 72 linker cases |
| FoF on the Agora snapshot | NOT TESTED | Snapshot not found on the Mac |

## Not verified

- Frame rate and feel in the native macOS app. Everything above ran in the Mac's Linux VM or the cloud container, without a real GPU.
- The pyqtgraph version on the Mac. Older pyqtgraph versions re-upload points on every frame.
- A 16M-cap store on EAGLE. The VM's 3.9 GB of RAM limited the test to an 8M cap.
- FoF on the Agora snapshot. It is not on the Mac or its connected drives.
- The macOS memory reading for the FoF stop (`phys_footprint` via `proc_pid_rusage`, falling back to `ps`). It is code-reviewed only; Linux was tested.

## Backlog

- Stream the store build to disk, to cut its 1.3–2.3 GB peak.
- Dim points in proportion to the depth falloff, so that distant regions don't look sparser than they are.
- Add a second, denser store tier, to remove most file reads at the 0.2–0.03 box zooms.
- Measure GUI frame rate on the Mac.
- Cut the main process's ~0.75 GB of large arrays during zooming: scene and colour copies, and transient copies when merging detail reads.
- FoF on snapshots bigger than memory (for example EAGLE's 425M DM on 16 GB) would need an out-of-core or domain-split linker.

## Files

- `lod/frustum.py`: camera frustum, plane extraction, box tests.
- `lod/cells.py`: octree cells, Morton order, level arithmetic.
- `lod/store.py`: sample store, file block index, disk cache.
- `lod/view.py`: view rule, level choice, detail reads.
- `lod/budget.py`: budget and base-sample sizing.
- `snapshot_io/id_index.py`: cached ParticleID index for tracking lookups.
- `snapshot_io/fof_process.py`: FoF in its own process, with a memory stop, progress and cancel.
- `controller/lod_controller.py`: background jobs, menu, status.
- `tools/bench_lod.py`: the zoom benchmark above (`python3 tools/bench_lod.py <first .hdf5 piece>`).
- Other files changed by this work:
  - `app_controller.py`: mixin, budget, `snap['fof']`.
  - `tracking_controller.py`: draws `_display_snapshot`.
  - `snapshot_loader.py`: `rows`, sampling from the store for big snapshots.
  - `main.py`: `--max-particles` help text.
  - FoF: `snapshot_io/fof.py` (bounded-memory linker, progress), `fof_snapshot.py` (read, progress, sorted result), `fof_finder.py` (`separate_process`), `snapshot_io/__init__.py` (names imported on first use, so the FoF process does not load Qt), `views/fof_dialog.py` (progress bar, Cancel), `app_controller.py` (`FoFWorker`).
