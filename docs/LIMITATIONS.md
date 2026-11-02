# Limitations and future work

What CosmoViewer does not do, does only partly, or has not been tested on. Each item names where it lives in the
code. The README's *Known limitations* lists the three a user meets first; this is the full list.

## Formats and readers

- **Group catalogues: Gadget-4 HDF5 only.** `snapshot_io/group_catalogue.py` reads `fof_tab` and `fof_subhalo_tab`
  (FoF groups and SUBFIND subhaloes as Gadget-4 writes them). Old binary SubFind `group_tab`/`sub_tab` and
  VELOCIraptor are not read: there were no test files to write them against, and the catalogue dialog offers only
  Gadget-4. EAGLE's per-particle `GroupNumber` and `SubGroupNumber` are used directly.
- **Gadget binary (SnapFormat 1 and 2) is a fallback**, through `cosmo_io_engine`, and has not been tested on real
  binary output recently. HDF5 goes through h5py.
- **cosmo_io_engine's own HDF5 reader** (`GadgetHDF5Reader`) reads one file when a piece of a multi-file snapshot is
  missing, without an error. The application does not use it for HDF5: `snapshot_io/snapshot_loader.py` refuses an
  incomplete set instead.
- **Remote and non-Gadget sources.** The command line reserves `agora:`, `sxs:` and `http(s):` source names; none is
  implemented.

## Structure finding

- **FoF needs a typed linking length on runs that are not Gadget-4.** The default b·⟨d⟩ is formed from Ω and units
  in the `Parameters` group, which EAGLE, AGORA, Gadget-2/3 and most zoom-ins do not write. On a zoom-in, falling back
  to (BoxSize³/N)^{1/3} would give a length about 8–10 times too long. Workaround:
  [fof_linking_length.md](fof_linking_length.md). Fix (not made): read Ω and units from `Header`/`Units` and use
  Gadget-4's formula for every set of link types.
- **FoF memory.** The linker runs at full resolution in its own process and stops above 60 % of physical memory.
  It has been measured to 8 M linking particles; EAGLE's 425 M dark-matter particles on a 16 GB machine would need an
  out-of-core or domain-split linker.
- **Group numbering** matches Gadget-4 up to rank 120; beyond that, groups of equal length come in arbitrary order.
- **Secondary particle types** (`fof.attach_secondaries`, Gadget-4's FOF_SECONDARY_LINK_TYPES) are implemented but not
  offered in the dialog, because they have not been validated.
- **No subhalo finder** of its own; subhaloes come from a SUBFIND catalogue when there is one.

## Tracking and selections

- **No ParticleID index above 200 M particles** (`snapshot_io/id_index.py`, `COSMOVIEWER_ID_INDEX_MAX`). Tracking on
  EAGLE-sized snapshots falls back to a chunked scan of every ID column, once per snapshot.
- **Field histograms ignore the active selection.** In the Analyse tab, histograms, lines and tables of one dataset
  always show the whole particle type; the bar above the plot says so. Profiles, energies and the phase diagram do
  use the selection.
- **The radius search on huge multi-file snapshots** runs on the interface thread, with a busy cursor.

## Measurements

- **No potential energy in periodic boxes.** W would need Ewald summation; the energy budget reports W only for
  isolated systems. The Layzer–Irvine equation for cosmological energy conservation is not implemented.
- **Large-N potential energy** is a Monte Carlo estimate from 4 M random pairs above 30,000 particles, with its
  standard error; an exact tree potential would need a new dependency.
- **Lagrangian radii errors** do not include the uncertainty of the centre. Coverage is slightly low at small
  fractions.
- **Anisotropy errors** ignore the σ_r–σ_θ correlation, so they are conservative; Osipkov–Merritt profiles have not
  been tested.
- **Velocities are peculiar** in cosmological runs; the Hubble flow is never added.

## Rendering and performance

- **Frame rate on the GPU has not been measured.** Level-of-detail timings are for queries and reads.
- **A new set of points can pause the window briefly** when it lands. In the native macOS app the longest stall
  was under 140 ms (about 600 ms in a Linux container), and only the default colouring was timed. New sets wait
  until the camera stops, so they do not interrupt a drag.
- **The LOD store build** still peaks at 1.3–2.3 GB of memory on very large snapshots, once per snapshot.
- **Memory while zooming** on EAGLE peaks at 3.5–4.0 GB across all processes and settles at 1.7–2.2 GB when idle.
- **The FoF memory guard** assumes 120 B per linking particle; 8M uniform particles used 128 B, so it can
  underestimate by about 15 %.
- **No periodic replicas in the 3D view**: near the box edge a structure shows on one side only. Trails and centres
  are unwrapped correctly.
- **Smoothed maps** are built from the display sample with rescaled masses. They are pictures; nothing quantitative
  should be read from them, and the panel says so.

## Code that is present but unused

Kept because it is small and documented where it stands; none of it is on a path the application runs.

- `spatial/octree.py` and `cosmo_io_engine.OctreeSpatialIndex`: box and sphere queries by linear scan. No tree is
  built and nothing calls them; FoF and selection use SciPy's cKDTree and the chunked file scans.
- `cosmo_io_engine.FriendsOfFriends` and `UnionFind`: an early FoF (minimum 20 members, bounding-box volume), not
  Gadget-4's conventions. The application's FoF is `snapshot_io/fof.py`.
- `cosmo_io_engine.SPHKernelEngine`: the 3D cubic spline in Gadget's convention; the smoothed maps use
  `model/sph_projection.py`.
- `gadget_file_discovery.FileRecord.cached_header`: filled during discovery, read by nothing.
- `plugin_base.STREAMING_HISTOGRAM`: a streamed histogram path that no plugin switches on, because none has a test
  showing its counts equal the default path's.

## Platforms

Developed and tested on macOS (Apple M3) and a Linux VM. Windows has not been tested; `tools/first_run.py` is
written to run there.
