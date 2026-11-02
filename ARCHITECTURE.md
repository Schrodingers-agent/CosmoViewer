# Architecture

This document explains how CosmoViewer is constructed, how its major components interact, and where the important engineering decisions are.

## Two components, one window

CosmoViewer has two main codebases that were developed separately and are now integrated.

**Immersive Mode** (`Cosmoviewer Immersive/`) is the 3D viewer. It loads Gadget-4/EAGLE snapshots, renders particles with OpenGL, and provides the interactive exploration experience: orbiting, panning, zooming, selecting particles, finding FoF groups, tracking particles across snapshots, and making movies.

**Dataset Explorer** (`Dataset Explorer/`) is the analysis workbench. It reads the same HDF5 files and produces quantitative results: histograms, scatter plots, radial profiles, energy budgets, Lagrangian radii, sanity reports, and phase diagrams. It has a plugin architecture where each physical quantity is a separate module.

**The integration** (`controller/analysis_host.py`) runs the Explorer inside Immersive's window as the "Analyse" tab. The active selection — with its recorded meaning and completeness status — is handed to the Explorer as a population. The Explorer follows the snapshot shown in the 3D view.

The Explorer still runs standalone for analysis-only work.

## Immersive Mode architecture

```
main.py
  └── CosmoViewerController (controller/app_controller.py)
        ├── CosmoViewerWindow (views/main_window.py)
        │     ├── CosmoGLView (views/view_extras.py) — OpenGL 3D view
        │     ├── ParticleSelectionPanel — selection criteria, tracking
        │     ├── ColouringPanel — colour by type, property, FoF group
        │     ├── StructuresPanel — halo table, catalogue groups
        │     ├── ParticleInspectorPanel — inspect individual particles
        │     ├── TelemetryView — log & console
        │     └── AnalyseBar — Analyse view status strip
        ├── SimulationModel (model/simulation_model.py)
        │     └── Snapshots (coords, velocities, masses, IDs, types)
        ├── LOD system (lod/)
        │     ├── LODStore — octree-based sample cache
        │     ├── LODView — level selection, detail reads
        │     ├── Cells — Morton-ordered octree cells
        │     ├── Frustum — camera frustum, visibility testing
        │     └── Budget — particle budget, base-sample sizing
        ├── Controller mixins
        │     ├── TrackingMixin — particle tracking, trails, trajectory export
        │     ├── FeaturesMixin — colouring, property reads, derived quantities
        │     ├── LODMixin — background LOD jobs, status updates
        │     ├── PickingMixin — mouse picking of particles
        │     ├── InspectorMixin — particle inspector integration
        │     ├── SetsMixin — selection set management
        │     ├── ViewToolsMixin — units, time display, cosmology detection
        │     ├── StructuresMixin — halo table, catalogue loading
        │     └── AnalysisHostMixin — Explorer integration
        └── Snapshot I/O (snapshot_io/)
              ├── SnapshotLoader — HDF5 and binary loading
              ├── FoF group finder (fof.py) — Friends-of-Friends algorithm
              ├── FoF process (fof_process.py) — separate-process execution
              ├── GroupCatalogue — Gadget-4 catalogue reader
              ├── IDIndex — cached ParticleID index for fast lookups
              └── Properties — snapshot property reading and caching
```

### The controller uses mixins

The `CosmoViewerController` class composes its functionality from ~10 mixin classes. Each mixin handles one responsibility (tracking, FoF, LOD, analysis hosting, etc.). This avoids a monolithic controller while keeping all mixins able to access the same model and window state.

### Data flow for a typical session

1. **Loading**: `FileDiscoveryEngine` scans a directory, classifies files, and builds a `FileManifest`. `AsyncSnapshotLoader` reads snapshots in a background thread. Each snapshot becomes a dict with coordinates, velocities, masses, IDs, types, and header attributes.

2. **LOD store construction**: For large snapshots (>8M particles), `LODStore` builds an octree-based sample of up to 16M particles, cached on disk as memory-mapped `.npy` files.

3. **Rendering**: `LODView` selects the detail level for the current camera and merges store particles with file-read particles for regions needing more detail. The merged set is sent to `CosmoGLView` as a point cloud.

4. **Selection**: A user can select particles by criteria (radius, property range, ID list), by FoF group, or by catalogue group. Each `SelectionSet` records its kind, definition, completeness, and the snapshot it was defined at.

5. **Analysis**: "Analyse ▸" hands the active selection to the hosted Explorer. The Explorer opens the same HDF5 file, reads the data arrays specified by the active plugin, runs the 6-stage compute pipeline, and renders the result.

## Dataset Explorer architecture

```
density_explorer.py (orchestrator)
  ├── Plugin system
  │     ├── plugin_base.py — 6-stage compute pipeline (template method)
  │     ├── plugin_registry.py — plugin discovery and tree schema
  │     └── *_plugin.py — 30+ physics plugins
  ├── I/O layer
  │     ├── io_manager.py — VirtualSnapshotSet (multi-file abstraction)
  │     └── lod_index.py — CSR bin-index sidecar for fast drill-down
  ├── Measurements layer (measurements/)
  │     ├── snapshot_view.py — SnapshotView with explicit units
  │     ├── centre.py — 5 centre-finding algorithms
  │     ├── lagrangian.py — Lagrangian radii with sampling errors
  │     ├── profiles.py — radial profiles (ρ, M(<r), σ_r, σ_t, β)
  │     ├── energy.py — energy budget, conservation, virial ratio
  │     ├── sanity.py — 15+ automatic data-quality checks
  │     ├── phase.py — 2D weighted histograms
  │     ├── runset.py — multi-snapshot analysis
  │     ├── cache.py — result cache (keyed by file path, size, mtime)
  │     └── scripts.py — export analysis as reproducible Python
  ├── Rendering
  │     ├── plot_panel.py — pyqtgraph-based plot rendering
  │     └── compute_worker.py — background computation thread
  └── Units
        ├── astro_units.py — cosmological unit conversion
        └── data_utils.py — CGS calibration, domain conditioning
```

### The plugin pipeline

Each plugin (density, metallicity, temperature, etc.) is a class that declares:
- `REQUIRES`: which HDF5 dataset paths it needs
- Configuration: axis labels, units, scaling, log/linear defaults

The orchestrator reads the required datasets once, passes them to `plugin.compute()`, and receives a `PlotRequest` (pure data, no Qt). The `PlotPanel` renders it.

The `BasePlugin.compute()` method runs a 6-stage pipeline:
1. **Ingestion** — shape resolution, vector component extraction
2. **CGS calibration** — apply cosmological factors from the file's unit attributes
3. **Unit transformation** — convert to display units (kpc, km/s, Msun, etc.)
4. **Domain conditioning** — log-space safety, clipping, NaN handling
5. **Statistical reduction** — histograms, profiles, statistics
6. **Assembly** — annotation, labels, provenance

Child plugins configure this pipeline through class-level attributes. Only `DiagnosticPlugin` overrides `compute()` directly.

## Level-of-detail system

The LOD system is documented in detail in [PERFORMANCE.md](Cosmoviewer%20Immersive/PERFORMANCE.md). Key design decisions:

**Why 81 levels?** Two levels per octave of particle density, covering 2⁴⁰ ≈ 10¹² range. Each level k keeps fraction f_k = 2^(-k/2) of the particles. Levels are nested: every coarser level is a subset of every finer one.

**Why hash the ParticleID?** A particle's membership in a level must be deterministic and independent of file ordering, so the same visual sample appears in every snapshot of a run. The hash (splitmix64) of the ParticleID provides this.

**Why an octree store?** The store sorts particles by octree cell, so a view query can enumerate only the visible cells and read a prefix of each cell's particle list. This avoids scanning the full store for every camera position.

**Why depth-aware sampling?** Without it, zooming in deeply pays for every particle behind the focus and never reaches full resolution at the camera's target. The depth falloff (D/d)² concentrates the budget near the camera, reaching f = 1.0 (full resolution) at a 0.004-box zoom on 850M particles.

**Why worker processes for file reads?** Large detail reads (>2M rows) are spread across worker processes to avoid blocking the main thread. Workers run at low priority (`nice 10`) and are stopped after 30 seconds of inactivity to return their memory.

## FoF group finder

The FoF implementation is in `snapshot_io/fof.py`. It follows the Davis et al. (1985) definition using Gadget-4's conventions for linking length calculation.

**Key engineering decisions:**

- **Separate process** (`fof_process.py`): FoF can require substantial memory. Running it in a separate process means a crash or out-of-memory condition stops only that process, not the viewer. Memory is monitored and FoF is stopped at 60% of physical RAM.

- **k-d tree for crowded cells**: Two neighboring grid cells with thousands of particles each would require O(n₁ × n₂) pair tests. The k-d tree reduces this to O(n log n), cutting memory from 6.3 GB to 2 MB and time from 40 seconds to 0.01 seconds per crowded pair. Ties at exactly the linking length are re-checked pair by pair for exactness.

- **Results sorted by ID**: The FoF result stores particles sorted by ParticleID. This means colouring LOD particles by group can use a binary search instead of sorting all IDs on every lookup.

## Selection semantics

Every selection records:
- **Kind**: particle IDs, FoF group, catalogue group, radius query, or property criterion
- **Definition**: the parameters that created it (e.g., "FoF group 1 from snapshot_002")
- **Completeness**: whether the selection contains every member. A selection made from the display sample is marked incomplete. A selection from a file that lacks some members is marked with the reason.
- **Expected size**: how many particles the complete selection should contain

This information travels through set files, sessions, exports, and the Analyse view. An incomplete selection carries a visible warning. The Explorer's analysis views say "incomplete" when the population is a subset.

The rationale: the most serious scientific error found during development was that selecting a FoF halo kept only the particles in the display sample — about a quarter of the halo at the default budget. The mass was wrong by a factor of four, with no warning. Recording completeness prevents this class of error.

## File format support

| Format | Reader | Status |
|---|---|---|
| Gadget-4 HDF5 (single file) | `snapshot_loader.py` (h5py) | Primary path, fully tested |
| Gadget-4 HDF5 (multi-file, `*.N.hdf5`) | `snapshot_loader.py` | Tested on EAGLE (16 files) |
| EAGLE / OWLS HDF5 | `snapshot_loader.py` | Velocity and Mass field name variants handled |
| Gadget binary Type 1 & 2 | `cosmo_io_engine.py` | Optional fallback; not tested recently |
| Gadget-4 FoF catalogues | `group_catalogue.py` | Groups and subhaloes |

## Dependencies

All dependencies are standard scientific Python packages available via pip:

| Package | Purpose |
|---|---|
| PyQt6 | GUI framework |
| pyqtgraph | OpenGL 3D rendering and 2D plotting |
| PyOpenGL | OpenGL bindings (used by pyqtgraph) |
| numpy | Array computation |
| h5py | HDF5 file reading |
| scipy | k-d trees (FoF), sparse graphs (connected components), interpolation |

Optional: `ffmpeg` (system command) or `imageio-ffmpeg` for MP4 movie encoding, with OpenCV (`cv2`, H.264 through AVFoundation on macOS) as the fallback when neither is found. Movies are always saved as PNG frames regardless.
