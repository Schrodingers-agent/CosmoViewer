# Following a small set of particles through time

A short manual for following a hand-picked set of a few particles with the Particle Selection and
Particle Inspector panels, worked through on the DM-L50-N128 example run (snapshots at z = 3, 1 and 0,
with `fof_tab` catalogues).

**The question:** were the particles that make up today's biggest halo already together at z = 3?

Every step was carried out in the real window. Every number was checked against h5py reads that use no
CosmoViewer code. Labels: **Verified** means the number shown matched that independent read.

## Before you start

- Open the folder holding `snapshot_000`–`002.hdf5` and `fof_tab_000`–`002.hdf5`. The catalogue is
  attached by itself, and the header shows `fof_tab_002.hdf5: 2,774 groups`.
- At start-up only **Visibility Layers** and **Colouring** are open. Click **PARTICLE SELECTION** to
  open it. The inspector opens by itself when you pick a particle.

## 1. Pick the set (z = 0)

1. Move the slider to the last snapshot (z = 0).
2. **View ▸ Jump to coordinates…** and enter halo 1's position from the catalogue: `24.184`, `35.265`,
   `31.075` (cMpc/h).
3. Zoom with the wheel until single particles in the halo are a few pixels apart (camera about
   3 cMpc/h away).
4. In **PARTICLE SELECTION**, choose Set 1 and **New**, then tick **Click a particle in the 3D view to
   pick it**.
5. Click one particle, then **Shift+click** seven more. Each click picks the nearest particle under the
   cursor and adds it to the set. Each pick also becomes the inspected particle and joins the
   inspector's **Particle** list, which keeps the last 10.
6. Type a name, e.g. `Halo-1 core`, and press Return.
7. Click **Colour** and choose yellow. Set 1's default red is the same red as dark matter, so the 8
   points cannot be seen until you change it.

*Expected:* `Set 1: 8 particles selected`, and the tracking line reads `8 of 8 found here; 3/3 snapshots`.
**Verified:** the 8 IDs clicked are exactly the 8 in the set, and each one is in catalogue group 1 at z = 0.

## 2. Watch them move

1. Tick **Follow selection** and **Trails**.
2. Move the slider to z = 3, then to z = 1 and z = 0.

*Expected:* the view centres on the set's mean position in each snapshot. Trails join each particle's
positions up to the snapshot shown, and a white line traces the set's centre.
**Verified** (centre path recorded in `results.json`). At z = 3 the members are up to 8 cMpc/h from their
centre, so zoom out to see all eight.

## 3. Read each member in the inspector

1. Open the inspector's **Particle** list and choose one member.
2. Move the slider through z = 3, 1 and 0. The **Group (catalogue)** row gives the halo the particle
   was in at that snapshot (0 = in no group). Δ shows `was 18` and similar.
3. Click the **Group (catalogue)** row to plot its history. Group numbers are drawn as separate points,
   not joined: they are labels, and 18 → 1 does not pass through 17.
4. Repeat for each member.

**Verified:** the group number and x coordinate of all 8 members at all 3 snapshots (24 of each) equal
the h5py values.

| ID | z = 3 | z = 1 | z = 0 |
|---|---|---|---|
| 735199 | none | group 18 (4,579 particles) | group 1 |
| 833252 | none | group 2602 (37) | group 1 |
| 816598, 1026029, 1242193, 910551, 1291459, 995678 | none | none | group 1 |

## 4. Export

**Export trajectories…** writes the set's positions and velocities in every snapshot (stored, unwrapped
and physical). **Export CSV…** in the inspector writes one particle's full record.
**Verified:** the exported positions equal the snapshot files exactly (maximum difference 0).

## The answer

For these eight particles: no. At z = 3 none was in any FoF group, and they were spread over a region about
8 cMpc/h across. By z = 1, two had fallen into separate haloes (groups 18 and 2602) while six were still
outside any group. All eight end in halo 1 at z = 0, within 0.67 cMpc/h of their centre.

Eight particles are an anecdote, not a measurement. Over all 34,120 particles of halo 1, an h5py check
gives the same picture: 72 % were in no group at z = 3, and the rest were split among 58 groups. At z = 1,
39 % were in no group and the rest were split among 48 groups, the largest giving 4,558 and 4,488
particles. Halo 1 assembled from many small progenitors plus diffuse matter. (Those whole-halo figures
come from h5py alone, not from the app.)

## Things to know

- The pick takes the particle nearest the cursor on screen; where several overlap, it takes the one
  nearest the camera. In a dense core you may get a neighbour, so zoom in first.
- The inspector holds one particle at a time and remembers the last 10 picks. For more than about 10
  particles, use the set's **Export trajectories…** instead.
- The set's description line repeats ("particle IDs, edited (add) from: particle IDs, …") after several
  Shift+clicks. This is known and harmless.
