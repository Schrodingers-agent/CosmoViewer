# Particle Selection panel: manual and hand-test workflows

Immersive Mode, right-hand side panel, **PARTICLE SELECTION (6 SETS)**.

This file has two jobs. Part 1 says what every control in the panel does. Part 2 is a
step-by-step test for each control, with the numbers the app should show on the DM-L50
run. Every expected number in Part 2 was produced by driving the real window on
2026-11-01 and checking it against an independent `h5py` computation that uses no
CosmoViewer code.

Status words used below:

- **Verified**: the driver run gave this result and an independent check agreed.
- **Verified (behaviour)**: the driver run showed this behaviour; there is no number to cross-check.
- **Known bug**: an open bug already on the list; the step records what you will see today.
- **New bug**: found while writing this manual, not fixed.

The panel ends with two boxes, "Pick lit particles" and "Tracking across snapshots".
They are covered at the end of each part (P1 to P3 and T1 to T3).

---

## Part 1. What each control does

The panel holds six selection sets. Every control acts on the **active set**, the one
chosen in the Set dropdown.

| Control | What it does |
|---|---|
| **Set** dropdown | Chooses the active set (Set 1 to Set 6). The line under it says what the set is: its kind, definition, snapshot, whether it is complete, and its size. It turns amber with ⚠ when the set is known to be incomplete. Hover over it for the full text. |
| **Enable** | Draws the set's particles in its colour (on) or hides them (off). The set keeps its particles either way. |
| Name field | The set's name, shown in the dropdown, in saved files and in the Analyse tab. |
| **Colour** | Opens the colour picker. The set is drawn in this colour in the 3D view, plots and movies. The button and the status line take the colour. |
| **Selection Mode**: New / Add / Subset | What Apply Criteria does with the result: **New** replaces the set, **Add** adds the result to it (union), **Subset** keeps only the particles that are in both (intersection). Add or Subset turns a defined set into "particle IDs, edited from …". |
| **Radius r ≤** [value] [unit] | A sphere of this radius about the 3D view's centre. Distances are periodic in a cosmological box. The unit label is the run's length unit (cMpc/h on DM-L50). |
| Centre line and yellow marker | Ticking Radius shows "about the view centre (x, y, z)" under the radius, and three yellow bars of length 2r through the centre in the 3D view. When the sphere crosses a face of the box the line adds "crosses the box edge in y, so it wraps (periodic)". |
| **Box centre** | Moves the view centre (and so the sphere's centre) to the middle of the box, (25, 25, 25) on DM-L50. |
| **Search full snapshot, not the sample** | On (default): the radius and property searches read every particle from the file. Off: they search only the particles held for display, and the set is marked "display sample only" in amber. |
| **FoF Group ID =** | Every member of group N of the FoF run done in the app (Structures ▸ Find FoF groups...) on the snapshot shown. Groups are numbered from 1, largest first. |
| Property range (tick box, dropdown, min, max) | Particles whose property lies in [min, max]. An empty box leaves that end open. |
| **Catalogue** [group/subhalo] [N] | Every member of group (or subhalo) N of the halo catalogue next to the snapshots (fof_tab / fof_subhalo_tab), read at full resolution. Numbered from 1. |
| **IDs:** | Typed particle IDs: whole numbers and ranges, for example `1021, 5000-5010`. |
| **Load IDs...** | Reads IDs into the active set, using the Selection Mode. A `.hdf5` selection-set file restores what the set meant; a text list becomes "particle IDs from <file>". |
| **Save IDs...** | Writes the active set: `.txt` (one ID per line, with a header line describing the set) or `.hdf5` (keeps the full definition, so Load IDs restores it). |
| **Analyse ▸** | Opens the Analyse tab with the active set as the population. |
| **Apply Criteria** | Runs every ticked criterion and combines them with AND, then applies the result to the active set with the Selection Mode. |
| **Clear Set** | Empties the active set. |
| **Recentre View** | Centres the 3D view on the active set (the periodic mean of its members) and zooms to fit it. |
| **Draw Selected Only** | Hides every particle that is not in an enabled set. |
| *Pick lit particles box* | |
| **Click a particle in the 3D view to pick it** | While ticked, a left click that does not move the camera picks the nearest lit particle within a few pixels of the cursor into the active set, as a new set. Shift+click adds it instead. "Lit" means the particles coloured by the Colouring property; with "Particle type" colouring every drawn particle can be picked. The line under the box says what was picked. |
| **List top** and its dropdown | With Colouring set to a property, lists the 200 particles with its highest values (from the whole file for file properties). Choosing a row picks that particle (New, or Add when the mode is Add) and centres the view on it. |
| *Tracking across snapshots box* | |
| **Follow selection when the snapshot changes** | On a snapshot change, centres the view on the active set (the periodic mean of its members in that snapshot). |
| **Trails, up to [N] particles** | Draws the path of the first N members of each enabled set through the snapshots up to the one shown, plus the path of the set's centre in white. |
| **Export trajectories...** | Writes the active set's positions in every loaded snapshot, by ID, to `.npz` or `.csv`. |
| Tracking line | Under the box: for each non-empty set, how many members were found in this snapshot and in how many snapshots they have been looked up. |

Things to know before testing:

- The radius is measured from the **view centre**, which pan, Recentre View, Box centre,
  View ▸ Jump to coordinates (Ctrl+J) and wheel zoom all move. Always read the centre
  line before you click Apply Criteria.
- Criteria that are ticked but not wanted still count: untick everything you are not testing.
- The side panel is long. Collapse **COLOURING** (click its heading) or scroll down to
  reach the selection panel.

---

## Part 2. Hand-test workflows

### Setup (do once)

The app may write files next to the snapshots, so test on a copy. `cp -c` makes APFS
clones that take no extra disk space.

```bash
mkdir -p ~/cv_panel_test && cp -c ~/"Active Projects/gadget4/examples/DM-L50-N128/output"/{snapshot_00?.hdf5,fof_tab_00?.hdf5} ~/cv_panel_test/
```

```bash
cd ~/"Active Projects/CosmoViewer/Cosmoviewer Immersive" && python3 main.py ~/cv_panel_test
```

**S1. Load and go to snapshot_002.** Wait until the header says "3 snapshots loaded".
Move the snapshot slider to the last snapshot.
*Expected:* the header shows `a = 1.000 (z = 0.00)` and `fof_tab_002.hdf5: 2,774 groups`.
Dark Matter (2,097,152) under Visibility Layers. The radius unit reads `cMpc/h`. The view
centre is (25, 25, 25). The display sample holds 2,000,560 of the 2,097,152 particles.
**Verified.**

**S2. Default set colours.** Look at the Colour button of Set 1.
*Expected today:* it is red (255, 0, 0), and dark matter is also drawn red, so Set 1 is
hard to see against the background (screenshot `04_radius_applied_one_piece.jpg`).
**Known bug: red default set colour on DM.**

### W1. Set dropdown

1. Open the Set dropdown.
   *Expected:* six items, `Set 1: Selection Set 1` to `Set 6: Selection Set 6`.
2. Choose Set 2.
   *Expected:* the status line reads `Set 2: 0 particles selected` in green, the Colour
   button turns green, and the line under the dropdown reads `Empty set`.
3. Choose Set 1 again.
   *Expected:* `Set 1: 0 particles selected`, red. **Verified (behaviour).**

### W2. Name field

1. With Set 1 active, select the text in the name field and type `Sphere A`.
   *Expected:* the dropdown changes to `Set 1: Sphere A` as you type. **Verified.**
2. Delete the whole name.
   *Expected today:* the field is blank and the dropdown reads `Set 1: Selection 1`,
   which is not the original `Selection Set 1`. **New bug N1 (minor): an emptied name
   falls back to "Selection N", not the default "Selection Set N", and the field stays blank.**
3. Type `Sphere A` again for the following tests.

### W3. Radius, centre line, marker, wrap note

1. Selection Mode = **New**. Untick every criterion. Tick **Radius r ≤** and set it to `10`.
   Keep **Search full snapshot** ticked.
   *Expected:* the centre line reads `about the view centre (25, 25, 25)` and three yellow
   bars appear through the centre of the 3D view. **Verified.**
2. View ▸ **Jump to coordinates...** (Ctrl+J). Enter x = `38.3`, y = `4.741`, z = `29.13`, OK.
   *Expected:* `about the view centre (38.3, 4.741, 29.13); crosses the box edge in y, so it
   wraps (periodic)`. The yellow bars move there. **Verified.**
3. Click **Apply Criteria**.
   *Expected:* after about 1.5 s, `Set 1: 39,135 particles selected`. The line under the
   dropdown reads `sphere r = 10 Mpc/h (comoving) about (38.3, 4.741, 29.13), wraps y
   (periodic) at snapshot_002.hdf5 · complete · 39,135 particles`. The set is drawn as one
   ball through the y = 0 face (drawn y from −5.20 to 14.66), not as a big piece plus a
   thin cap at y = 50. **Verified:** the 39,135 IDs are exactly the particles within a
   periodic distance of 10 of that point in `snapshot_002.hdf5`, and every drawn point
   is within 10.0 of the centre.

### W4. Recentre View

1. Ctrl+J, enter `10`, `10`, `10`, OK. Then click **Recentre View**.
   *Expected:* the view centres on Sphere A and zooms to it. The centre line reads
   `about the view centre (36.25, 4.327, 30.27); crosses the box edge in y …`. This is the
   mean of the members, which lies off the sphere's centre because the matter inside is
   not uniform. **Verified:** an independent periodic mean of the 39,135 members is
   (36.2482, 54.3271 ≡ 4.3271, 30.2702).

### W5. Colour

1. Click **Colour**, choose a light blue (for example R 0, G 200, B 255), OK.
   *Expected:* the button, the status line and the drawn set all turn that colour.
   **Verified (behaviour).** Screenshot `06_colour_cyan.jpg`.

### W6. Enable

1. Untick **Enable**.
   *Expected:* the set disappears from the 3D view. The status line still says 39,135.
2. Tick it again.
   *Expected:* the set is drawn again. **Verified** (39,135 drawn → 0 → 39,135).

### W7. Draw Selected Only

1. Tick **Draw Selected Only**.
   *Expected:* at once, every other particle disappears (header `N=0`) and only the 39,135
   set members are drawn (screenshot `08_draw_selected_only.jpg`).
2. Untick it.
   *Expected:* the background comes back at once. **Verified (behaviour).** The redraw is
   immediate since the Draw Selected Only fix.

### W8. Box centre and a full-snapshot sphere

1. Click **Box centre**.
   *Expected:* the centre line reads `about the view centre (25, 25, 25)`. **Verified.**
2. Choose Set 2 in the dropdown and name it `Centre r10 full`. Mode New. Only **Radius**
   ticked, `10`. **Search full snapshot** ticked. Click **Apply Criteria**.
   *Expected:* `Set 2: 172,030 particles selected`, `… · complete · 172,030 particles`.
   **Verified** against h5py.

### W9. Search full snapshot (off)

1. Choose Set 3 and name it `Centre r10 sample`. Same sphere (r = 10 about (25, 25, 25)).
   **Untick** Search full snapshot. Click **Apply Criteria**.
   *Expected:* `Set 3: 164,161 particles selected`. The line under the dropdown is amber:
   `⚠ sphere r = 10 … · display sample only · 164,161 particles`. Hover for the warning
   "This set was taken from the display sample, not the whole snapshot …".
   **Verified:** all 164,161 are in the 172,030 of W8 (95.4 %), as expected from a sample
   of 2,000,560 out of 2,097,152.
2. Tick **Search full snapshot** again.

### W10. Selection Mode: New, Add, Subset

1. Choose Set 4 and name it `Two spheres`. Mode **New**. Only Radius ticked, set to `5`.
   Click **Box centre**, then **Apply Criteria**.
   *Expected:* `71,424 particles`. **Verified.**
2. Ctrl+J to `38.3`, `4.741`, `29.13`. Mode **Add**. **Apply Criteria**.
   *Expected:* `75,316 particles` (the union of the two r = 5 spheres). The description
   becomes `particle IDs, edited (add) from: sphere r = 5 … · 71,424 particles · 75,316 particles`.
   **Verified** against h5py.
3. Untick Radius. Tick **IDs:** and type `1-1000000`. Mode **Subset**. **Apply Criteria**.
   *Expected:* `25,097 particles` (the union, restricted to IDs ≤ 1,000,000). **Verified.**
4. Mode **Add**, IDs `1-10`, Apply: `25,107`. Mode **Subset**, IDs `1-2000000`, Apply:
   still `25,107`.
   *Expected today:* the description nests one more "particle IDs, edited (…) from:" on
   every step, and lists every earlier count (`· 71,424 particles · 75,316 particles ·
   25,097 particles · 25,107 particles …`). The counts are right; the text is not useful.
   **Known bug: nested descriptions on repeated Add/Subset.**

### W11. IDs field

1. Choose Set 5 and name it `Typed IDs`. Mode New. Only **IDs:** ticked, type
   `1021, 5000-5010`. **Apply Criteria**.
   *Expected:* `Set 5: 12 particles selected`, `particle IDs · complete · 12 particles`.
   **Verified:** exactly IDs 1021 and 5000 to 5010, all of which exist in the snapshot.
2. Type `abc`, **Apply Criteria**.
   *Expected:* a warning: "Could not apply the criteria: 'abc' is not a particle ID: use
   whole numbers and ranges such as 5000-5010". The set keeps its 12. **Verified.**
3. Type `99999999999` (no such particle), **Apply Criteria**.
   *Expected today:* no warning; `Set 5: 1 particles selected`, `particle IDs · complete ·
   1 particle`. **New bug N2: typed IDs are not checked against the snapshot, so a set of
   IDs that do not exist is called "complete".** **New bug N3 (cosmetic): "1 particles".**

### W12. Property range

1. Choose Set 6 and name it `Fast |v|>1000`. Mode New. Only the property box ticked.
   Property `Velocities |v|`, min `1000`, max empty. Search full snapshot ticked.
   **Apply Criteria**.
   *Expected:* `85,934 particles`, `Velocities |v| >= 1000 [code units, × a^0.5 for
   physical] at snapshot_002.hdf5 · complete · 85,934 particles`. **Verified** against
   |Velocities| in the file.
2. Property `Coordinates x`, min `10`, max `10.5`. **Apply Criteria**.
   *Expected:* `16,266 particles`, `10 <= Coordinates x <= 10.5 […] · complete`. **Verified.**
3. Min `5`, max `1`, **Apply Criteria**.
   *Expected:* the warning "property minimum is above the maximum"; the set is unchanged.
   **Verified.**

The dropdown lists: Group (catalogue), Coordinates x/y/z, Masses, Velocities x/y/z, Velocities |v|.

### W13. Catalogue

1. Set 6, name `Catalogue group 1`. Only **Catalogue** ticked, `group`, `1`. **Apply Criteria**.
   *Expected:* `34,120 particles`, `catalogue group 1 of fof_tab_002.hdf5 at
   snapshot_002.hdf5 · complete · 34,120 particles`. **Verified:** the same IDs as
   `GroupLen[0]` from `GroupOffsetType[0]` in the snapshot.
2. Number `2774` (the last group). *Expected:* `32 particles`. **Verified.**
3. Number `2775`. *Expected:* warning "group 2775 is not in the catalogue". **Verified.**
4. `subhalo`, `1`. *Expected:* warning "subhalo 1 is not in the catalogue". DM-L50 has
   no subhalo catalogue, so the refusal is right. **New bug N4 (minor): the message
   should say that the catalogue has no subhaloes at all.**
5. Set it back to `group`.

### W14. FoF Group ID

1. Set 5, name `FoF group 1`. Only **FoF Group ID** ticked, `1`. **Apply Criteria**
   before running FoF.
   *Expected:* warning "FoF has not been run on the snapshot shown (snapshot_002.hdf5);
   run it there (Structures > Find FoF groups...) or untick the FoF criterion." **Verified.**
2. Structures ▸ **Find FoF groups...**. Leave b = 0.20 and N_min = 32. Click **Run FoF
   Auto-Detection**. Wait (about 10 s here), then close the window.
   *Expected:* 2,774 groups, the same number as Gadget-4's fof_tab_002. **Verified.**
3. **Apply Criteria**.
   *Expected:* `34,120 particles`, `FoF group 1 (b = 0.2, N_min = 32, DM) at
   snapshot_002.hdf5 · complete · 34,120 particles`. **Verified:** exactly the same IDs as
   catalogue group 1 in W13.
4. FoF Group ID `2775`. *Expected:* warning "FoF group 2775 does not exist in
   snapshot_002.hdf5: it has groups 1 to 2774." **Verified.**

### W15. Save IDs and Load IDs

1. Choose Set 1 (Sphere A, 39,135). Click **Save IDs...**, save as `sphereA.txt` in
   `~/cv_panel_test`.
   *Expected:* the file starts with `# Sphere A: sphere r = 10 Mpc/h (comoving) about
   (38.3, 4.741, 29.13), wraps y (periodic) at snapshot_002.hdf5 · complete · 39,135
   particles; saved at snapshot_002.hdf5`, followed by 39,135 IDs. **Verified.**
   The save dialog suggests the set's name with no folder, so it opens wherever the
   dialog last was. Choose the folder yourself.
2. **Save IDs...** again, as `sphereA.hdf5` (choose the "Selection set" file type).
3. Choose Set 3, Mode New, **Load IDs...**, pick `sphereA.txt`.
   *Expected:* `39,135 particles`, `particle IDs from sphereA.txt · 39,135 particles`.
   The IDs are identical to Set 1. A text list does not carry the definition, by design.
   **Verified.**
4. Choose Set 4, Mode New, **Load IDs...**, pick `sphereA.hdf5`.
   *Expected:* `39,135 particles`, with the full sphere description restored (`… wraps y
   (periodic) … · complete · 39,135 particles`). **Verified.**
5. Choose an empty set and click **Save IDs...**.
   *Expected:* "The active set is empty." and no dialog. **Verified.**

### W16. Apply Criteria with nothing ticked, and Clear Set

1. Choose Set 3. Untick every criterion. Click **Apply Criteria**.
   *Expected today:* nothing happens, with no message; the set keeps its 39,135.
   **Known bug: Apply with no criterion is silent.**
2. Click **Clear Set**.
   *Expected:* `Set 3: 0 particles selected`, `Empty set`, and Set 3's particles leave the
   view (other enabled sets stay drawn). **Verified (behaviour).**

### W17. Analyse ▸

1. With an empty set active, click **Analyse ▸**.
   *Expected:* the status line reads `Set N is empty: select particles first, then
   Analyse ▸.` and the tab does not change. **Verified.**
2. Choose Set 1 (Sphere A) and click **Analyse ▸**.
   *Expected:* the Analyse tab opens with `Analysing snapshot_002.hdf5 (snapshot 3 of 3) ·
   Population: Sphere A: sphere r = 10 … · 39,135 particles`. **Verified (behaviour).**
   Go back to Explore 3D.

### W18. Wheel zoom moves the radius centre

1. Tick Radius, click **Box centre**: centre line `(25, 25, 25)`.
2. Put the mouse over the upper right of the 3D view and scroll one notch.
   *Expected today:* the centre line changes (one run gave (23.18, 26.3, 25.64), another
   (26.01, 26.98, 25.72): the exact value depends on the camera angle), and the yellow
   marker moves with it. A sphere applied now is not about the box centre.
   **Known bug: wheel zoom moves the radius centre.** Until it is fixed, set the centre with
   Box centre or Ctrl+J after zooming, and read the centre line before Apply Criteria.

### W19. Z-slice panel (below the 3D view)

1. Look at the x axis of the "Z-slice (display sample)" plot.
   *Expected today:* it runs from about −170 to 220, although the box is 0 to 50, so the
   particles fill only a small square in the middle. **Known bug: Z-slice axis too wide.**

### P1. List top with type colouring

Start from a fresh launch (Setup, S1). Set 1 active.

1. Leave Colouring ▸ Colour by on `Particle type`. Click **List top**.
   *Expected:* the line under the box reads "Colour by a property first; the list shows its
   highest values." and the dropdown stays empty. **Verified (behaviour).**

### P2. List top by a property

1. Colouring ▸ Colour by = `Velocities |v|`. Click **List top**.
   *Expected:* after a moment the dropdown holds 200 rows, starting
   `1. Velocities |v| 2605  ID 882776 (DM)`, `2. Velocities |v| 2553  ID 884183 (DM)`,
   `3. Velocities |v| 2494  ID 931412 (DM)`. **Verified:** the three fastest particles in
   snapshot_002 by h5py are 882776 (2605.41), 884183 (2552.87) and 931412 (2493.61).
2. Ctrl+J to `10`, `10`, `10`. Selection Mode **New**. Open the dropdown and choose row 1.
   *Expected:* `Set 1: 1 particles selected` (see N3),
   `highest Velocities |v| (from a list of 200) at snapshot_002.hdf5 · complete · 1 particle`,
   and the view centre jumps to the particle, (24.085, 35.536, 31.199). **Verified**
   against the particle's position in the file.
3. Mode **Add**. Choose row 2.
   *Expected:* `2 particles selected` (IDs 882776 and 884183). The line under the box reads
   "Picked ID 884183 (DM), Velocities |v| = 2553 into Selection Set 1; it is followed
   through the snapshots by ID." **Verified.**

### P3. Click to pick

Keep Colour by = `Velocities |v|`.

1. With **Click a particle…** unticked, click on a particle in the 3D view without moving the mouse.
   *Expected:* nothing is picked; the set keeps its 2. **Verified.**
2. Mode **New**. Tick **Click a particle in the 3D view to pick it**. Zoom in until single
   particles are a few pixels apart, then click exactly on one.
   *Expected:* `Set 1: 1 particles selected`, `picked particles · complete · 1 particle`,
   and the line under the box names the ID and its |v|. **Verified:** in the driver run the
   particle clicked (ID 163938, 9 px from its nearest neighbour on screen) is the one picked.
   Where particles overlap on screen the nearest one to the camera is taken, so in a dense
   region you may get a neighbour of the one you meant.
3. Shift+click another isolated particle.
   *Expected:* `2 particles selected`; the description becomes `particle IDs, edited (add)
   from: picked particles …`. **Verified.**
4. Click on empty black space.
   *Expected:* "No lit particle under the cursor."; the set keeps its 2. **Verified.**
5. Untick **Click a particle…** and set Colour by back to `Particle type`.

### T1. Tracking line and Follow selection

1. Set 1, Mode New, only Radius `10`, Ctrl+J to `38.3`, `4.741`, `29.13`, **Apply Criteria**
   (Sphere A again, 39,135).
   *Expected:* the tracking line reads `Set 1: 39,135 of 39,135 found here; 3/3 snapshots`
   once the lookups finish. **Verified.**
2. Keep **Follow selection** ticked. Ctrl+J to `10`, `10`, `10`. Move the snapshot slider
   to the first snapshot (snapshot_000).
   *Expected:* the view jumps to (36.51, 3.270, 29.07), the centre of the same 39,135
   particles in that snapshot. **Verified** against the periodic mean of those IDs in snapshot_000.hdf5.
3. Move back to snapshot_002.
   *Expected:* the view centre is (36.25, 4.327, 30.27), as in W4. **Verified.**
4. Untick **Follow selection**. Ctrl+J to `10`, `10`, `10`. Move to snapshot_000.
   *Expected:* the view stays at (10, 10, 10). **Verified.** Tick it again and go back to snapshot_002.

### T2. Trails

Sphere A in Set 1, snapshot_002 shown.

1. Tick **Trails, up to** with `100`.
   *Expected:* 100 coloured trails, each joining the particle's three positions, and one
   white line for the set's centre (screenshot `shots_pick_track/04_trails_100.jpg`).
   **Verified:** 101 lines of 2 segments are drawn.
2. Set the number to `0`.
   *Expected:* only the white centre path. **Verified.**
3. Set it to `5`, then move to snapshot_001.
   *Expected:* 5 trails plus the centre, each one segment shorter (they stop at the snapshot
   shown). **Verified** (6 lines of 2 segments, then of 1).
4. Untick **Trails**.
   *Expected:* the trails disappear. **Verified.** Go back to snapshot_002.

### T3. Export trajectories

1. With Sphere A active, click **Export trajectories...**. Save as `sphereA_traj.npz`
   in `~/cv_panel_test`.
   *Expected:* the file holds `ids`, `pos`, `pos_unwrapped`, `pos_physical`, `vel`,
   `vel_physical`, `mass`, `types`, `times`, `redshifts`, `centre` and `metadata`; `pos`
   has shape (3, 39135, 3). **Verified:** the 39,135 IDs are the set, and `pos[0]` and
   `pos[2]` equal the coordinates of those IDs in snapshot_000 and snapshot_002 exactly.
   The dialog suggests `<set name with _ for spaces>_trajectories.npz`.
2. Export again as `sphereA_traj.csv`.
   *Expected:* a `#` metadata line, a header row, then 117,405 rows (39,135 × 3 snapshots).
   **Verified.**
3. Choose an empty set, click **Export trajectories...**.
   *Expected:* "Select particles in the active set first." **Verified.**

---

## Summary of results (driver run, 2026-11-01)

| Workflow | Result |
|---|---|
| S1 load, W1 Set, W5 Colour, W6 Enable, W7 Draw Selected Only, W17 Analyse ▸ | PASS (behaviour) |
| W3 sphere 39,135 with wrap note and one-piece drawing | PASS, matches h5py |
| W4 Recentre View | PASS, matches an independent periodic mean |
| W8 sphere at box centre 172,030 | PASS, matches h5py |
| W9 sample search 164,161, amber warning | PASS (subset of the full result) |
| W10 Add 75,316 and Subset 25,097 | PASS on counts; description FAIL (known bug) |
| W11 typed IDs 12; bad text refused | PASS; non-existent IDs accepted as "complete" FAIL (new bug N2) |
| W12 property ranges 85,934 and 16,266 | PASS, match h5py |
| W13 catalogue group 1 = 34,120, group 2774 = 32 | PASS, match fof_tab_002 |
| W14 app FoF: 2,774 groups, group 1 = catalogue group 1 | PASS, identical IDs |
| W15 Save/Load .txt and .hdf5 | PASS |
| S2 default colour, W16 silent Apply, W18 wheel zoom, W19 Z-slice | FAIL (known bugs) |
| W2 emptied name | FAIL (new minor bug N1) |
| P1, P2 List top: top three IDs match h5py; centring on the pick | PASS |
| P3 click and Shift+click pick the particle under the cursor | PASS |
| T1 tracking line; Follow selection centres match h5py in snapshots 000 and 002 | PASS |
| T2 trails for 100, 0 and 5 particles | PASS |
| T3 export: .npz positions equal the files exactly; .csv has 117,405 rows | PASS |

Not covered: multi-file snapshots, gas or star runs, and the subhalo catalogue path
(DM-L50 has none). Click-to-pick was tested on an isolated particle only.
