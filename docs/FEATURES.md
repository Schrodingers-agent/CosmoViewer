# CosmoViewer features in detail

This is the full tour of CosmoViewer's features. Each entry gives the problem it addresses, what it does, and how it changes the work. The [README](../README.md) has the short version.

Contents: [Interface](#interface) · [3D viewer](#3d-viewer-explore-3d) · [Colouring and shaders](#colouring-and-shaders) · [Structure finding](#structure-finding) · [Particle selection](#particle-selection) · [Particle inspector](#particle-inspector) · [Analysis](#analysis-analyse) · [Movies](#movies)

## Interface

Good figures come from iteration: change one thing, look, change another. Every part of the interface is built to make that loop seconds long, so a research-grade illustration comes straight from tuning a view.

### Do many things at once

The 3D view, the 2D map, the halo table, the inspector and the selection and colouring panels all live in one window. Panels collapse when you don't need them. The halo table docks or floats. Analysis is one tab away, and plots and movies open in their own windows. Change the snapshot and all of it follows. *Why it matters:* you never lose your place, and you never re-open a file to look at it a different way. The question you had a second ago is still on screen when the answer arrives.

### Dual-domain synchronised viewport

A 3D view is good for finding structure and poor for reading it; a 2D map is the reverse. Here they sit together, 3D on top and 2D below, always showing the same region. A button in the corner of the view shows or hides the 2D panel, and dragging it sets the panel's height. The 2D panel has three modes:

- **Z-slice points** (the default). A thin slice at a depth you choose with a slider, like an MRI scan. *Why it matters:* you see structure in cross-section, with nothing in front to hide it. A shell, a cavity or a disc's thickness reads at once.
- **Smoothed density (3D view).** The SPH-smoothed surface density (mass per unit area, projected along the line of sight) of exactly what the 3D camera sees, on a log magma scale. *Why it matters:* this is the picture most papers print. Here it comes from the camera angle you just chose, with no camera parameters copied into a script.
- **Smoothed colour property (3D view).** A colour-aware projection: the mass-weighted average of whatever you are colouring by, drawn with the same colour table, range and log scale as the 3D view, from the particle types you have on screen. See [Colour-aware projections](#colour-aware-projections).

With *Follow camera* on, the panel redraws whenever the camera stops. Zoom or pan in it to redraw that part in more detail. Up to 2M particles it uses every one; beyond that, the display sample. *Why it matters:* you explore in 3D and read the structure in 2D at the same moment, and the two can never drift apart.

### Console and command line

Some steps are faster typed than clicked, and some need to be repeated exactly. The *Log & console* tab shows everything the app reports, warnings and errors with full tracebacks included, and takes commands (`help`, `info`, `load`, `track`, `track export`) with tab completion. An error badge on the tab tells you when something went wrong. `--run` passes the same commands at launch. Launch with a folder, a file or a glob pattern; naming one piece of a multi-file snapshot opens all of them. *Why it matters:* an error is never hidden in a terminal you closed, and a set of steps you worked out by hand can be replayed exactly.

### Sessions

Rebuilding a view by hand loses time, and sometimes loses the figure. *Save session* writes a small JSON file with everything needed to come back to the same picture: the snapshot files and the one on screen, the camera, colouring and point style, visible particle types, the selection sets with their IDs, names, colours and meanings, the 2D panel, view options, labelled points, additional data files, the catalogue setting and the settings of every FoF run. *Open session* loads the snapshots if they are not already open and applies the rest. It never stores particle data, so a session stays small. FoF is not re-run on opening: a reopened session still says which FoF run defined a group, and recomputes it only when you ask. *Why it matters:* a figure can be remade exactly, by you next month or by a colleague on another machine, and a selection comes back knowing what it was.

### Live data and additional data

A running simulation keeps writing snapshots. *Rescan directory* (Ctrl+N) reloads the folder, so new snapshots appear without restarting. *Read additional data* attaches extra per-particle datasets from other files: you give the property name, integer or real, the file, its format (ASCII with header lines and a column, or an HDF5 dataset) and the particle type it applies to. *Why it matters:* you can watch a run while it is still going, and a quantity computed in post-processing can be coloured, selected and plotted like any stored field, with no rewrite of the snapshot.

## 3D viewer (Explore 3D)

### Multi-species layers with counts

In a galaxy simulation, the dark matter outnumbers and hides everything else. Each particle type gets a checkbox, a colour and its count: "Gas (584,281)", "Dark Matter (600,000)". *Why it matters:* turn one off and the stars and gas stand alone. You study one component without the others in the way, and you always know exactly how many particles you are looking at.

### Level of detail

Large snapshots usually force a choice: a thinned-out picture, or a viewer that crawls. Here a fixed budget of points is spent where you look. Set it anywhere from 500k to 5M (default 2M) to suit your machine. Whenever the camera settles, the particles in view are refilled at the finest level that fits the budget, from a memory-mapped octree or, once you zoom in past its resolution, straight from the snapshot files. Snapshots larger than memory still open, because detail streams from disk. The sample is keyed on a hash of ParticleID, so the same particles are drawn in every snapshot. *Why it matters:* you explore an 850-million-particle run on a laptop as smoothly as a small one, and a movie through time does not flicker because the sample changed. The view set is only ever drawn. FoF, tracking, plots and movies read the files or the base sample.

### Particle budget and resampling

Every machine has a different limit. Set the budget from the *Particle budget* menu, turn level of detail on or off, or *Resample view now* to refill the current view. *Why it matters:* the viewer runs smoothly on a laptop and uses the full power of a workstation, and you decide where the line is.

### HUD overlays

A picture without a scale or a time is hard to trust and harder to publish. A scale bar that resizes as you zoom, the time or redshift, and labelled points are drawn over the view and burnt into screenshots (*Save screenshot*). Each can be switched off from the menu. *Why it matters:* a screenshot is ready for a talk or a draft as it is, and nobody has to guess how big the thing is or when it happened.

### Labelled points

Marking known objects usually means annotating a figure afterwards. *Read labelled points* reads a text file of positions and names (`x y z [name]`), and the labels sit in the 3D view and move with it; *Clear labelled points* removes them. *Why it matters:* you can find a known galaxy, or check a halo against a published position, at a glance.

### Property plots

Comparing two quantities usually means exporting them first. *Make plot* opens a histogram of one property or a scatter plot of one against another, in its own window, with linear or log axes. The drawn particles are shown in grey and each enabled selection set in its own colour. With *Auto update* on, the plot redraws when the snapshot, the sample or a selection changes. Besides every stored property, it offers distance from the view centre and local density. *Why it matters:* a relation you suspect from the picture can be checked while the picture is still on screen, and a selection shows up as its own cloud against everything else.

### Projection, stereo and navigation

Perspective hides true sizes, and a flat screen hides depth. Switch to *Orthographic projection* for sizes that do not shrink with distance, or to red/cyan *Anaglyph* or *Side by side stereo* to see depth. *Jump to coordinates* takes you straight to a known position. *Why it matters:* orthographic views give honest sizes for comparison, stereo makes a filament's depth obvious, and a known position is one dialog away. These apply to the window and to screenshots; movies have their own camera.

### Playback and full screen

*Play snapshots* steps through the run again and again, so you watch evolution before you decide to make a movie of it. *Full screen* gives the view the whole display for a talk or a demo. *Why it matters:* you see whether a story is worth a movie before you spend time rendering one.

### Simulation summary

Checking what a run actually contains usually means opening its files by hand. *Simulation summary* lists the snapshots, the units, the header, cosmology and parameters in one copyable page. *Why it matters:* "what box, what h, which code settings?" is one menu away, and the answer can be pasted into a notebook or a methods section.

### Particle inspector and List top

Click a particle to read every property it has in every snapshot, or list the 200 most extreme particles of whatever you are colouring by. Both have full sections: [Particle inspector](#particle-inspector) and [Picking by hand](#picking-by-hand).

## Colouring and shaders

Only the person making a figure knows what it is for. In a script, every change to a picture means editing a number and running it again. Here every option in the colouring panel redraws the view as you move it, so you find the picture that shows the structure in seconds, and you can see why it works before you commit to it.

The panel has two halves. The first decides *what colour* each particle gets. The second, *Point style*, decides *how it is drawn*: two layers of points, each with its own size, blur and fade, drawn by shaders written for CosmoViewer. Some of these tools are standard in the field, and CosmoViewer does them carefully. Others are new here, and they are labelled **experimental**: they work well in daily use, they have trade-offs, and the trade-offs are listed.

### Standard tools, done carefully

- **Colour by anything.** Particle type, any stored or derived property, or group membership, in any of the bundled colour tables, linear or log, with a colour bar. This is GadgetViewer's colour-by-property, rebuilt so the window and the movie renderer produce identical colours. *Why it matters:* the question you are asking decides the colouring. Temperature shows shocks, metallicity shows where enriched gas has travelled, group membership shows what the halo finder decided. Switching between them takes one menu.
- **Perceptually uniform tables.** viridis, magma, inferno, plasma, cividis (colour-blind safe), fire and grey from Kovesi's CET maps, a blue-white-red diverging map for signed quantities, and turbo for when you want a rainbow. *Why it matters:* in a uniform table, equal steps in value look like equal steps in colour, so the picture does not invent edges the data do not have. Cividis keeps a figure readable for colour-blind readers.
- **Auto range, or your own.** One click sets a display range from the 0.5th to the 99.5th percentile of the drawn particles, then holds it fixed across snapshots and movies, so frames stay comparable. Or type in your own range for reproducibility. *Why it matters:* a range that rescales itself in every frame makes a cooling halo look unchanged; a fixed range makes the change visible. A typed range means the same figure can be made again, months later or by a referee. One quirk: a log scale with a range that reaches zero or below is drawn linear.
- **Random colours for integers.** Every halo gets its own colour; group 0, "in no group", stays grey. *Randomize* reshuffles the colours when two neighbours look alike. A group keeps its colour across snapshots. *Why it matters:* thousands of groups on a single colour gradient blur into each other. Distinct colours make each halo's boundary readable, show at once where two groups touch or a finder has split one object in two, and let you follow one halo by eye through time.
- **Line-of-sight velocity.** Velocity along the viewing direction, relative to the mean velocity of the drawn particles, on the diverging map: red moving away from you, blue towards you, on a symmetric range. It updates as you rotate. *Why it matters:* this is how an observer sees a galaxy. Rotation, outflows and infall stand out the way they would in a spectrum. Values are in stored code units (for a comoving run, multiply by a^0.5 for peculiar km/s), and the zero point depends on which particles are drawn.
- **Distance from view centre.** Measured across the periodic boundary in a box. *Why it matters:* radial structure, gradients and shells read directly, centred wherever you are looking.
- **Brightness from local density.** The colour keeps its hue from the property and takes its brightness from the local density, for Splotch-style pictures. *Why it matters:* you see two quantities at once, what the gas is doing and where most of it is, so dense, hot regions stand out from diffuse ones of the same colour. *Trade-off:* the brightness is scaled between the 1st and 99.5th percentile of the coloured particles in each frame, so it is a picture, and brightness is not comparable from one frame to the next. The density comes from the display sample (below).

### New in CosmoViewer (experimental)

- **Lit and unlit layers, and *Show uncoloured particles*.** When you colour by a property, the particles that carry a value form the lit layer, and the rest form the unlit layer, drawn dim underneath, or not at all. Colour an AGORA or dark-matter run by FoF group and every halo lights up, but you lose the macro structure. Click *Show uncoloured particles*, and BAM: adjust their transparency and you see everything, the web and every halo in it. *Why it matters:* haloes only make sense in context: which filament feeds them, which void they border. This keeps the context on screen without hiding the subject. *Trade-off:* the lit layer is always drawn on top, so an unlit filament in front of a lit halo does not hide it. The layers give up true depth order between them in exchange for keeping the subject visible.
- **Point style: Size, Blur and Fade for each layer.** Size runs from 1 to 12 pixels. Blur blends each point from a hard disc into a Gaussian profile. Fade scales its opacity. These are shader settings, so moving a slider redraws at once with nothing re-uploaded to the graphics card. *Why it matters:* the structure you care about can be sharp and bright while its surroundings stay soft and faint, so the eye goes where you want it to, on screen and in a figure. *Trade-offs:* a blurred point is softer and also dimmer, since its edges are more transparent. If your installed pyqtgraph has a different point shader, the layers fall back to plain points: Fade still works through the colours, and Blur is unavailable. Movie frames are drawn without a shader, so Fade, *Show uncoloured particles* and the overlap mode carry into movies, and Blur does not.
- **Lit overlap: Glow or Solid.** *Glow* adds lit particles together, in any order, so dense regions brighten; Fade sets how much overlap it takes to saturate. *Solid* depth-tests lit particles among themselves only (the unlit layer writes no depth), so the nearest lit particle is in front. *Why it matters:* Glow reveals where matter piles up along the line of sight; Solid shows the surface nearest you. Flip between them to tell a dense core from a chance alignment. *Trade-offs:* Glow saturates, and a dense core can burn out to a flat bright patch unless you lower Fade. Solid hides what is behind, and it discards the faintest edges of blurred points so they do not punch holes in the depth buffer, which makes very soft points look a little cut out. A technique that gives both at once (weighted blended order-independent transparency, McGuire & Bavoil 2013) needs extra render targets that pyqtgraph does not expose. In movies, Solid is approximated by drawing far to near.
- **Lagrangian origin.** Each particle is coloured by where it was in the earliest loaded snapshot: red for x, green for y, blue for z across the box, found by ParticleID at full resolution. Matter that started together keeps one colour. *Why it matters:* mixing and infall become visible. A halo made of one colour formed from one patch; a halo made of many colours gathered matter from all over the box, and you can see which filament brought what. *Trade-offs:* the colour table and range do not apply. Three coordinates squeezed into one colour means two distant origins can look alike, and colours are kept at least 20% bright so nothing vanishes on the black background. "Earliest" means the earliest snapshot you have loaded, and particles missing from it go to the unlit layer. For large snapshots the lookup runs in the background, and the colours appear when it finishes.
- **Local density (display sample).** A k-nearest-neighbour density (k = 32) of the drawn particles, scaled by the sampling factor, on magma. It is computed once on the base sample, and a zoomed-in view takes each particle's value from its nearest base-sample neighbour, so zooming neither changes the estimate nor throws it away. *Why it matters:* structure shows up in any run, even one with no stored density, such as pure dark matter. *Trade-off:* it is estimated from the display sample, so its resolution is the sample's. It is a visual guide, and it is labelled "display sample" so it is never mistaken for a measurement.

### Colour-aware projections

A 3D point cloud is a good way to explore and a poor way to publish. Points overlap, depth is ambiguous, and a printed page has no rotate button. The usual fix is a projection: a smoothed map along the line of sight. But a projection is normally made afterwards, in a script, from a camera position copied out of the viewer by hand, and every adjustment means another run.

In CosmoViewer the projection is the view. Colour the 3D particles by temperature, metallicity or any other property, and the panel below draws the mass-weighted projection of that same property, with the same colour table, the same range and the same log scale. Change the colour table and both update. Narrow the range to bring out a shock front, and the map sharpens with it. Hide the dark matter, and the projection drops it too. Rotate the camera until the disc is edge-on, let go, and the map is redrawn from exactly that angle.

Say you are a researcher in a code-comparison project such as AGORA, preparing the figure that sets the runs side by side. The figure has to show the same quantity, on the same scale, from the same angle, in every run. Here you tune it live: find the angle where the outflow is clearest, pick the range that separates enriched gas from pristine, fix it with *Auto range*, and step through the snapshots with nothing else changing. What you see in the panel is the figure, and a screenshot of the 3D view carries the scale bar and time with it.

## Structure finding

Finding haloes is usually a separate program, a separate run and a separate file format, so asking "is that a halo?" waits on a batch job. Here it is a menu item, and the answer is drawn on the particles you are looking at.

### Friends-of-Friends

*Structures ▸ Find FoF groups* runs a halo finder on the snapshot on screen.

- **Settings.** Choose which particle types link (dark matter by default), a linking length as *b* × the mean particle separation (*b* = 0.2, for periodic boxes) or a fixed length in the run's units, and the smallest group to keep (*N*min = 32). These are Gadget-4's conventions. On an isolated run, with no box to define a mean separation, it asks you for a fixed length. *Why it matters:* the defaults match what the simulation code itself would find, so your groups are comparable with its catalogue, and every setting is in front of you.
- **Runs from other codes.** On EAGLE, AGORA, Gadget-2/3, GIZMO or any zoom-in, *b* × mean separation does not work yet: it either stops with an error or, with more than dark matter ticked, gives a length several times too long. Link dark matter only and type a fixed length computed as in [FoF on runs from other codes](fof_linking_length.md). *Trade-off:* this is a documented workaround; the code still reads the cosmology from Gadget-4's `Parameters` group only.
- **Every particle, in its own process.** FoF reads every particle from the files, never the display sample, and runs in a separate process, so a big run cannot freeze the viewer. Before it starts, it checks that the run will fit in memory and refuses one that will not, with the reason. *Why it matters:* you see the groups seconds after you wonder about them, at full resolution, and a run that would have exhausted your machine is turned away up front instead of crashing it halfway.
- **Colour, select, recentre.** *Color Particles by FoF Group* colours the box by group in one click. *Select halo (all members)* puts every member of a chosen group into the active selection set, at full resolution, and *Recentre View* flies to it. *Why it matters:* from "there are haloes" to "these are this halo's particles" takes two clicks.
- **Remembered per snapshot.** Each snapshot keeps its own FoF result and the settings that made it, and sessions save those settings. *Why it matters:* a group number always refers to the run that produced it, and you can tell which linking length defined a halo weeks later.

### Catalogues: use the code's own answers

Re-finding haloes the simulation already found wastes time and invites mismatches. *Structures ▸ Read halo catalogue* attaches the catalogue your simulation wrote (Gadget-4 HDF5, `fof_tab` and `fof_subhalo_tab`) and you work with its groups and subhaloes directly. Membership is read by the catalogue's own offsets, at full resolution. If the snapshot file lacks some of a group's members, as in a cut snapshot, the selection says it is incomplete and how many of how many it holds. *Why it matters:* your selection matches the catalogue your paper cites, particle for particle, and a partial halo can never pass for a whole one.

### Halo table

A list of numbers is hard to connect to the picture. *Structures ▸ Halo table* lists every group of the snapshot on screen, one row per group, docked or floating.

- **Groups from.** The simulation's catalogue for this snapshot, or FoF run in this window on this snapshot. The table follows the snapshot on screen and never shows one snapshot's groups as another's; on a snapshot without FoF it says which snapshots have it. *Why it matters:* a halo number on screen always belongs to the snapshot you are looking at.
- **Columns in honest units.** Size, mass and centre, as stored in the files, in the run's code units, with each column header naming its unit (for example 10¹⁰ M☉/h, Mpc/h comoving). A value that has not been measured yet shows as "—", never as zero. *Why it matters:* nobody reads a missing mass as a massless halo, or a comoving length as a physical one.
- **Sort, filter, jump.** Sort any column, hide groups below *Min N* members, or *Go to group* by number. The table holds its columns as arrays, so a catalogue of a million groups costs the view only the rows on screen. *Why it matters:* browsing haloes becomes as fast as scrolling a list, even in a big run.
- **Click a row.** The whole group goes into the active selection set, at full resolution, and the camera flies to it. *Analyse* sends it to the Analyse tab. *Why it matters:* the halo you read about in the table is in front of you, and measured, in a click each.
- **Check against catalogue.** Measure the selected group from its member particles at full resolution, and compare with the catalogue: members found against its length, summed member mass against its mass, and periodic centre of mass against its position. Centres are computed the way Gadget-4 computes them, each particle at its nearest periodic image, in double precision. *Why it matters:* you can check that the catalogue and the snapshot agree before you build on them, which catches a mismatched file or a cut snapshot at once.
- **Measure all groups.** The same measurement for every group in one pass over the snapshot, with *Cancel*. For FoF groups it fills in the mass and centre columns. *Why it matters:* a whole catalogue is checked, or a FoF list given masses and centres, in one click.

## Particle selection

A selection that loses its meaning is worse than no selection. In a script, "the halo" is whatever an array of IDs happened to hold when the cell last ran; a week later nobody remembers whether it was the whole halo, a sphere around it, or the part that was on screen. Here a selection keeps its definition, its completeness and its particles, wherever it goes. Everything below lives in the *Selection* panel beside the 3D view.

### Six sets, each with a name and a colour

- **Set combo, name, colour, enable.** Pick one of six sets ("Set 1: Halo A"), rename it, give it a colour, and switch it on or off. The colour follows the set into the 3D view, the 2D panel, property plots, the Analyse tab and movies. *Why it matters:* real questions are comparisons: a halo and its neighbour, the gas that falls in and the gas that doesn't, a group at two masses. Six sets on screen at once make the comparison visible, and the colours tell you which is which in every figure without a legend lookup.
- **The meaning line.** Under the set name, one line says what the active set is: how it was made, on which snapshot, and whether it holds every particle it should. A set known to be incomplete turns amber, and the tooltip carries the full definition and the reason. An empty set says so and tells you how to fill it. *Why it matters:* you never have to remember how a set was built. The panel tells you before you measure it.

### How a criterion adds to a set

- **New, Add, Subset.** *New* replaces the set with what the criteria find. *Add* puts the new particles in alongside the old ones. *Subset* keeps only the particles that are in both. The same three modes apply to criteria, to clicks, to *List top* and to *Load IDs*. *Why it matters:* most interesting populations are built in steps: everything in this halo, then only the hot part, then add that satellite. Each step is one click, and you can see the result after every step.
- **Edits are recorded.** A set made by a definition (a sphere, a group, a property range) that you then add to or cut down becomes a plain list of IDs, and it remembers which definition it was edited from. *Why it matters:* an edited set honestly stops claiming to be "FoF group 12". Its record says what it was before and that you changed it, so the provenance survives the edit.

### Criteria

Tick any criteria, then *Apply Criteria*. Every ticked criterion must hold, so ticking a radius and a temperature range gives you the hot gas inside that sphere.

- **Radius r ≤.** Every particle within a distance of the centre of the 3D view, measured across the periodic boundary in a box, in the run's own length unit. *Why it matters:* "everything within 50 kpc of this galaxy" is the most common first question, and here you answer it by flying to the galaxy and typing a number. A halo that straddles the box edge is selected whole.
- **Search full snapshot, not the sample.** On by default. The search reads every particle from the snapshot files; turn it off to search only the drawn particles, for a quick look. *Why it matters:* the drawn sample is a picture, and a count made from it is a fraction of the truth. With this on, a sphere holds every particle in it, however few are drawn. With it off, the set is marked as made from the display sample, and that warning follows it everywhere.
- **FoF Group ID =.** Every member of a Friends-of-Friends group found in this window. If FoF has not been run on this snapshot, the panel says so. *Why it matters:* you click from "there is a halo" to "these are its particles" with no export, no matching script and no catalogue lookup.
- **Property range.** Pick any property and type a minimum, a maximum or both; leave one empty for an open bound, so "temperature above 10⁶ K" is one field. Properties stored in the snapshot are searched in the full snapshot; properties attached from other files are searched in the drawn sample, and you are warned. *Why it matters:* this is the bounds-based selection every analysis script writes by hand, with the bounds visible and the result on screen at once, and a record of exactly which bounds were used.
- **Catalogue group / subhalo N.** Every member of a group or subhalo of the loaded halo catalogue, read from the snapshot at full resolution (1 is the first in the catalogue). Without a catalogue, the membership stored in the snapshot is used: EAGLE's `GroupNumber`, with a subhalo numbered group × 100000 + `SubGroupNumber`. *Why it matters:* the simulation already decided which particles belong to which halo. This uses its answer, so your selection matches the catalogue your paper cites, particle for particle.
- **IDs.** Type particle IDs and ranges directly: `1021, 5000-5010`. *Why it matters:* a colleague sends you the IDs of a strange particle, a paper lists a tracer population, or you noted one down last week. You paste them and they are on screen.

### Files

- **Load IDs.** Read a list of IDs from text, `.dat`, `.csv` or `.npy`, or a selection-set file (`.hdf5`). A selection-set file brings back exactly what the set meant when it was saved; a plain list becomes "IDs from that file". *Why it matters:* a population chosen in a script, by a collaborator or by another tool comes straight into the viewer, ready to look at and track.
- **Save IDs.** Write the active set as text, one ID per line, with its description in the header line; or as a selection-set file that stores its definition, its snapshot and its completeness. *Why it matters:* a set made by eye becomes an input to a paper pipeline, and the file says how it was made. Months later, you, a referee or a collaborator can read where it came from.
- **Analyse ▸.** Send the active set to the Analyse tab. *Why it matters:* the set you have just checked by eye is the set you measure, with no file in between.

### Acting on a set

- **Apply Criteria** fills the active set; **Clear Set** empties it. The status line gives the count.
- **Recentre View.** Fly the camera to the centre of the set, from its full-resolution positions when they have been looked up, unwrapped across the periodic boundary. *Why it matters:* a set split across the box edge still gets the right centre, and the camera lands on the object.
- **Draw Selected Only.** Hide everything else. *Why it matters:* you check what you selected before you measure it. A stray clump or a missing satellite shows up here, before it becomes a strange number.

### Picking by hand

- **Click a particle to pick it.** Tick the box, and a click that does not move the camera picks the nearest lit particle within a few pixels of the cursor into the active set; Shift+click adds to it. Picked particles are followed through every snapshot by ID, like any other selection, and the click also opens them in the [Particle inspector](#particle-inspector). *Why it matters:* the particle you noticed is the particle you get. "What is that bright speck?" turns from a coordinate hunt into one click.
- **List top.** The 200 lit particles with the highest values of whatever you are colouring by, read from the full snapshot for stored properties. Choose one to pick it (New or Add, as above) and centre the view on it. *Why it matters:* the most extreme particles are often the most interesting: the hottest gas, the densest clump, the most massive black hole. Finding them usually means a sort in a notebook; here they are a drop-down.

### Tracking through time

- **Follow selection when the snapshot changes.** On by default. Step to any snapshot and every set is found again by ID, at full resolution, even after gas has turned into stars. *Why it matters:* "where was this halo's matter at *z* = 3?" is one drag of the time slider, with no ID matching by hand in every file.
- **Trails, up to N particles.** Draw where the set's particles went, for up to N particles per set (100 by default), unwrapped across the periodic box. The path of the set's centre is always drawn. *Why it matters:* a trail shows at a glance whether a halo's gas fell in along a filament or arrived in a merger, which a series of snapshots shows only to someone who already knows where to look.
- **Export trajectories.** Every particle of the active set in every loaded snapshot, as a NumPy archive or a CSV with one row per particle per snapshot. The file carries the set's name, definition and completeness, and the log says how many particles were found in each snapshot. *Why it matters:* tracks found by eye become data for a paper, and their provenance goes with them.

## Particle inspector

Following one particle through a run normally takes a script per question: open every snapshot, find the ID, read each property, line them up, difference them, convert units. The inspector does all of it the moment you click. Its design follows OVITO's data inspector, which shows one picked element's properties, and ParaView's *Plot Selection Over Time*, which shows its values at every step; the change from snapshot to snapshot is CosmoViewer's addition.

Every value is read from the snapshot files by ID, at full resolution, in the background, so the view stays responsive. The drawn sample is never the source of a number in the inspector.

- **The header.** ID, type, snapshot number and time: scale factor, redshift and cosmic time in a cosmological run, simulation time otherwise. Under it, the time since the snapshot you are comparing with, how far the particle moved, and a change of type when there is one ("Type changed: Gas → Stars"). If the particle is missing from the other snapshot, the header says so. *Why it matters:* the most important facts about a particle's history, when it is, how far it went and whether it formed a star, are visible before you read a single row.
- **Recent picks.** The last ten particles you inspected, by name. *Why it matters:* comparisons often mean going back and forth between two particles. You flip between them in one menu without finding them again in the view.
- **Every property, with its change.** A table of every property stored for the particle, with its value, the change Δ and the change in per cent, against the snapshot you choose. Hover over a row for its unit, the value in the other snapshot and the rate of change per unit time. The table also shows the particle's catalogue group and subhalo, and its FoF group in this window. *Why it matters:* "what happened to this gas?" is answered by reading one table: its temperature tripled, its density rose a hundredfold, its metallicity went up. The whole story is in one place.
- **Change since: previous, next or first snapshot.** Choose what Δ is measured against. Positions use the nearest periodic image, so a particle that crosses the box edge shows its real displacement. *Why it matters:* previous shows what just happened; first shows everything that has happened since the start; next looks ahead. A particle that wraps around the box would otherwise appear to jump a whole box length.
- **Physical units.** Multiply each value by its a and h scaling, read from the dataset's own attributes. Physical changes include the expansion of the universe between snapshots; the stored, comoving changes do not. A property with no scaling attributes says so and shows no physical value. *Why it matters:* a comoving displacement and a physical one answer different questions, and confusing them is one of the commonest errors in the field. The toggle makes the difference explicit and uses the units the simulation itself recorded.
- **Only changed.** Hide every property that stayed the same. *Why it matters:* a gas particle carries dozens of fields, and most of them do not change between two snapshots. Hiding them leaves the few that tell the story.
- **History of.** Choose a property, or click a row in the table, and a plot shows its value in every loaded snapshot, with the current snapshot marked. *Why it matters:* the history of one quantity over the whole run, a temperature spike, the moment of star formation, a slow enrichment, is one click away. It is the plot you would otherwise write a loop to make.
- **Select its catalogue group (N) / Select its FoF group (N).** Put every member of the inspected particle's group into the active set, at full resolution, and centre the view on it; the button shows the group number. When there is no group to select, the button is off and its tooltip says why: no particle picked, no catalogue attached, FoF not yet run on this snapshot, or the particle is in no group. *Why it matters:* you go from "this particle is odd" to "is its whole halo odd?" in one click, and from there straight to tracking or analysis.
- **Export CSV.** Every property of the particle in every loaded snapshot, with its changes, in one file, with the snapshot, file, type, time, scale factor and cosmic time of each row. *Why it matters:* the particle you found by eye becomes data for a paper or a script, with its whole history in one table.

## Analysis (Analyse)

Measuring something you have just seen usually means leaving the viewer. Here the measurement is one click from the picture and knows exactly what it measured. The Analyse tab is the Dataset Explorer, built into the same window; it also runs on its own.

### Seeing and measuring together

- **One tab away.** Analysis lives in a tab of the same window and follows the 3D view's snapshot. *Analyse ▸* sends it the active selection; *Show in 3D* sends the result's particles back, highlighted. *Why it matters:* you move between seeing and measuring without copying anything.
- **Every result says what it measured.** A bar above every result names the population measured and whether anything is missing. *Use every particle* measures the whole snapshot instead of the selection. *Why it matters:* a number never turns up without its population. You know whether it came from a whole halo or a partial one.
- **Every particle, every time.** An analysis never samples. A selection too large to hold in memory (over 50 million particles) is refused with a message. *Why it matters:* a science number comes from all the data, and a result counts as science only when every particle was used.
- **The window and a script agree.** Each analysis calls the same measurement layer a script would, so the window and a script give the same numbers. *Why it matters:* a result you found by clicking can be reproduced, extended and checked in code without reimplementing it.

### Quantities

- **Plugins for every field.** Every derived quantity normally needs its own conversion code. 24 plugins cover density, temperature, entropy, element abundances, stellar and black-hole properties, and a generic plugin reads any other numeric field: dark matter, Gadget-4 names such as `Masses`, and FoF or SUBFIND catalogue fields. Units come only from the file's own attributes; an unknown dimension is said to be unknown. Fields a file cannot supply can be shown greyed out. *Why it matters:* you see at once what a run contains, and no unit is ever guessed.
- **Metallicity.** Enrichment is easy to misread without a reference. *Z* is shown in absolute and solar units (*Z*☉ = 0.0134, Asplund et al. 2009), with the solar value and median marked, and one line sums it up: "Metal-free (Z=0): 11,827 (7.9%) | Median Z = 1.17e-02 (0.87 Z☉) | Above solar: 39.1%". *Why it matters:* how enriched a galaxy is, and how much pristine gas is left, takes a single glance.
- **Views.** Each quantity can be read as a matrix (the full table), a histogram or a line, with linear or log axes and export. *Why it matters:* the same data answers different questions in different forms, and switching costs one click.

### Measurements

- **Centre.** *Options ▸ Centre* picks shrinking sphere (the default), density-weighted or centre of mass, and every measurement that needs a centre recomputes. All are periodic-aware. *Why it matters:* the centre choice biases radii, and in a merger a density-weighted centre can land between the two galaxies. The default finds a nucleus, and the choice is always visible.
- **Lagrangian radii.** The radii that hold given fractions of the mass, such as 10%, 50% and 90%, with sampling errors, for all types or one. *Why it matters:* collapse and expansion show up as numbers you can compare across a run.
- **Profiles.** Density, enclosed mass, velocity dispersion and anisotropy, for all types or one, with the region inside 2.8 softening lengths shaded as unresolved. *Why it matters:* you see exactly where the profile can be trusted.
- **Energies.** Kinetic, thermal and potential energy, the virial ratio, momentum and angular momentum, for one snapshot, or across a run with energy drift, virial ratio and angular momentum over time. The potential energy is computed exactly up to 30,000 particles and by a Monte Carlo estimate with its standard error above that. *Why it matters:* you get a first look at whether a halo is bound, or a merger is settling, on the spot; the virial ratio is a first diagnostic.
- **Phase diagram.** Gas density against temperature, weighted by mass, with every unit of weight accounted for. *Why it matters:* where the gas sits, cold disc, hot halo or shock, is visible in one plot.
- **Sanity report.** Broken data produces convincing nonsense. One pass over every particle checks header counts, dataset lengths, bad values, negative masses, duplicate IDs and positions, particles on or outside the box edge, impossible speeds, missing or doubled mass, and the cosmological mass budget, each marked PASS, WARN or FAIL. It always checks the whole file. *Why it matters:* a problem in the files is caught before it reaches a result.

### Getting results out

- **Export.** Every result exports as CSV with a provenance header, or as a Python script that reproduces it from the snapshot. *Why it matters:* every result should be reproducible, and the GUI is never a dead end for a paper.
- **Cache.** Results are cached, keyed on the measurement, its settings and each file's path, size and modification time. *Why it matters:* a repeat comes straight from the cache, and a changed file is recomputed, only for its own rows.
- **Compare arrays (advanced, hidden by default).** *Compare arrays from files* overlays one or two `.npy`, `.csv` or `.txt` arrays, as lines or as heatmaps, with no resampling. It is a developer diagnostic, so it is listed under Options only when CosmoViewer is started with `COSMOVIEWER_ADVANCED=1` (DECISIONS X-5). Its Matrix tab shows only the first array.

## Movies

A good simulation movie usually takes a pipeline: export the data, script a camera, render frames, stitch them together, and do it all again when something looks wrong. CosmoViewer has two movie makers, and both start from the view you already have on screen, with its colours, selections and camera.

### Quick

*For:* a movie in under a minute of setup, for a talk, a group meeting or a message to a collaborator.

*What it does:* pick one of two movies, *turn around the current view* or *play through all snapshots*, then a length in seconds, a size, a look (points as on screen, or smoothed) and MP4 or GIF. It follows the selected particles, adds in-between frames (experimental, see *Smooth time* below) so the movie lasts as long as you asked, and burns in the time, a scale bar and a colour bar. It writes that one file and nothing else.

*What it cannot do:* no keyframed camera paths, no PNG frames, and no control over the overlays. Smoothed Quick movies use at most 500,000 particles per snapshot, to keep rendering time down.

### Custom

*For:* the movie for a paper, a press release or a public talk, where every frame matters.

*What it does:*

- **Motion.** Rotate around a target, evolve through the snapshots, or fly a **camera path through keyframes**: *Add current view* stores the camera and the snapshot on screen, and the path between keyframes is a smooth spline, with optional ease in and out. Because each keyframe also stores its snapshot, one path can fly through space and time together. Keyframes can be reloaded from an earlier movie.
- **Smooth time (experimental).** In-between frames follow a cubic Hermite path through each particle's positions and velocities at the two snapshots, matched by ID and across the periodic box, so gas turning into stars keeps moving. They are entirely for viewing, as acceleration is usually non-linear: every one is labelled, and none feeds a measurement.
- **Look.** Points, or smoothed surface density with a colour table, log scale and one colour range for the whole movie, taken from the first frame or typed in. Optional bounding box, time or redshift label, scale bar in comoving or physical units, and colour bar. Frames can be centred on the selected particles.
- **Output and long runs.** PNG frames, plus MP4 and GIF. Each snapshot can be read from disk while rendering, with a fixed number of particles, so a movie can cover more snapshots than fit in memory. Unreadable snapshots can be skipped. Every movie's settings, camera and snapshot per frame are saved beside it, so it can be remade or extended later.

*What it cannot do:*

- In-between frames are only as smooth as the snapshot spacing allows. When snapshots are far apart, particles orbit in between, so their paths are limited to stay close to a straight line, and the motion looks more like a cross-fade. The movie reports when this happened.
- In-between frames keep the earlier snapshot's colours, trails and smoothing.
- Snapshots read from disk while rendering have no FoF-group colours; turn that option off to colour by FoF group.
- Smoothed frames are slow: one frame of 2 million particles took about 14 seconds in testing (Linux container, 640 × 480).
- Movies have their own camera and renderer: orthographic projection, stereo and the window's overlays stay in the window (movies draw their own time label, scale bar and colour bar), and points are drawn without Blur.
- Inset graphs and two-channel colour are not available yet.
