# assets/

Pictures, exported results and movies made with CosmoViewer itself. Nothing here is used by the code or the tests.

## images/

| File | What it shows |
|---|---|
| `00_first_render_dm_l50_cut.png` | The 3D viewer on the 400,000-particle DM-L50-N128 cut that ships with the repository, as `tools/first_run.py` saves it |
| `01_merger_opened.jpg` | CollidingGalaxiesSFR opened: 18 snapshots on the time slider |
| `02_fof_groups.jpg` | Friends-of-Friends groups found in the merger |
| `03_galaxy_selected.jpg` | One galaxy selected at full resolution |
| `04_tracked_through_merger.jpg` | The same particles at t = 0, 0.6, 1.2 and 4 (code units), coloured by particle type (gas cyan, dark matter red) |
| `05_analyse_tab.jpg` | The selection opened in the Analyse tab |
| `06_energy_budget.jpg` | Energy budget of the selection |
| `07_angular_momentum_over_run.jpg` | Angular momentum over the run |
| `08_camera_follows_set.jpg` | The camera following the selected set |

Images 01 to 08 come from the worked example, [docs/Following_a_galaxy_through_a_merger.pdf](../docs/Following_a_galaxy_through_a_merger.pdf).

## outputs/

Results exported from the Analyse tab during that worked example, exactly as the app wrote them (CSV) with the plot
it showed. The Lagrangian radii of one galaxy before (`snap000`) and after (`snap334`) the merger give the manual's
answer: the 10 % radius shrinks from 0.79 to 0.47 kpc while the 90 % radius grows from 9.2 to 23 kpc.

## movies/

`dm_l50_halo0_smoothed.*`: the largest z = 0 halo of DM-L50-N128 traced back to z ≈ 3, made with the custom movie
maker (evolving, smoothed surface density, 11 in-between frames per snapshot pair, colour range fixed from the first
frame). The in-between frames are interpolated for viewing and labelled as such.
