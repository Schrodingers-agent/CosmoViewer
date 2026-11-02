# Movies: the Quick and the Custom movie maker

CosmoViewer makes movies from the view already on screen, with its colours, selection sets and camera.
*Options ▸ Make movie…* (Ctrl+Shift+M) opens one window with two tabs: **Quick**, for a movie in under a minute of
setup, and **Custom**, for a movie where every frame matters. Both are rendered off screen, so the movie's size does
not depend on the window's and movies can be made with no display at all.

Contents: [Quick](#quick) · [Custom](#custom) · [How frames are made](#how-frames-are-made) ·
[In-between frames](#in-between-frames-smooth-time) · [Smoothed frames](#smoothed-frames) ·
[Following a selection](#following-a-selection) · [Output](#output-files-and-encoders) ·
[Checks](#what-has-been-checked) · [Performance](#performance) · [Differences from GadgetViewer](#differences-from-gadgetviewer) ·
[Limitations](#limitations) · [A worked example](#a-worked-example)

## Quick

Pick a movie, *turn around the current view* or *play through all snapshots*, then:

| Setting | Choices |
|---|---|
| Length | seconds; in-between frames are added so the movie lasts that long (at most 100 per snapshot pair) |
| Size | 640 × 480, 1280 × 720, 1920 × 1080 or 1080 × 1080 |
| Look | points as on screen, or smoothed surface density |
| Format | MP4 (25 fps) or GIF (15 fps) |

It follows the selected particles, burns in the time or redshift, a scale bar and a colour bar, and writes that one
file and nothing else (frames go to a temporary folder). Smoothed Quick movies read at most 500,000 particles per
snapshot, which keeps a frame to a few seconds. Quick has no keyframes, no PNG frames and no overlay settings; use
Custom for those.

## Custom

Three tabs.

**Motion.**
- *Rotating*: the camera turns about the scene's z axis through a chosen angle in a chosen number of frames.
- *Evolving*: one frame per snapshot, from a first to a last, optionally centred on the selected particles with the
  centre smoothed over the last N snapshots, unreadable snapshots skipped, and *In-between frames per snapshot pair*.
- *Camera path through keyframes*: *Add current view* stores the camera and the snapshot on screen. Keyframes can be
  shown, reordered, removed, or loaded from an earlier movie's `_movie.json`. The path between them is a Catmull–Rom
  spline in centre, log distance, angles and field of view (the azimuth goes the short way round), the snapshot
  changes linearly between keyframes, and ease in and out is optional. One path can therefore fly through space and
  time together. *Move particles smoothly between snapshots* turns on in-between positions along the path.

**Look.** Point size and highlight size, or smoothed surface density with a colour table, log scale and one colour
range for the whole movie (taken from the first frame and then fixed, or typed in). Optional bounding box, time
label, scale bar in comoving or physical units, and colour bar.

**Output.** PNG frames, plus MP4 and/or GIF, and *Read each snapshot from disk while rendering, with N particles*
(on by default), which keeps memory bounded however many snapshots the movie covers.

## How frames are made

- **One scene for window and movie.** The window and the movie build their points, colours and selections from the
  same function (`controller/tracking_controller.py`, `_build_scene`), so a movie cannot drift from what you saw.
- **The same camera.** `movie/renderer.py` reproduces pyqtgraph's camera (centre, distance, elevation, azimuth,
  field of view); its view matrix equals `GLViewWidget.viewMatrix()` for random cameras, so a movie starts from the
  view on screen.
- **Rendered in NumPy**, like GadgetViewer's own plotter: points are 2-pixel splats, so Fade, *Show uncoloured
  particles* and the overlap mode carry into movies and Blur does not. *Solid* overlap is approximated by drawing far
  to near. Orthographic, stereo and the window's overlays stay in the window; movies draw their own.
- **One frame loop** (`movie/maker.py`) serves every kind of movie and asks for scenes one snapshot at a time,
  keeping at most three in memory.

## In-between frames (smooth time)

A run with few snapshots makes a jumpy movie. Between snapshots A and B, `movie/interpolate.py` moves every particle
along a cubic Hermite curve through its position in A and in B, with its velocity in each as the tangent:

- particles are matched by ParticleID; one missing from B's drawn sample is looked up in B's files at full
  resolution, and one that is not in B at all is dropped and counted;
- in a periodic box B's position is first moved to the image nearest A's;
- the curve's parameter is the snapshot's time variable (the scale factor a in a comoving run). Stored velocities are
  converted to d(position)/d(time variable): for Gadget-4 comoving runs the stored velocity is u = v_pec/√a, so
  dx/da = u a^(s−2)/H(a) with s = 0.5 (the `Velocities` a_scaling), H₀ taken from the file's units;
- when snapshots are far apart compared with how fast particles turn, the velocity tangents are much longer than the
  straight line between the two positions and an unlimited curve overshoots; tangents are then capped at three times
  the straight line, and the movie's message and JSON (`tangents_shortened`) say how many particles were affected.

Every in-between frame is labelled "(interpolated)" in the frame and in the JSON, frames at a snapshot are the
snapshot itself, and no in-between frame feeds a measurement. Colours, trails and smoothing lengths in an in-between
frame are the earlier snapshot's.

## Smoothed frames

`movie/smoothed.py` draws the SPH-kernel surface density of `model/sph_projection.py`: each particle's mass is spread
with the 2D cubic spline, normalised per particle, with h = r₃₂/2 from its 32nd nearest neighbour in the drawn
sample (gas may use its SmoothingLength), and masses scaled by N_total/N_sample. The image is an orthographic
projection onto the plane through the view centre, as wide as the perspective view at that depth, so a scale bar
holds everywhere. Particles are drawn at their stored positions, as under the box outline and in point frames. One
colour range holds for the whole movie. A smoothed frame is a picture of the sample; nothing quantitative should be
read from it.

**Overlays** (`movie/overlays.py`): a time label (a and z, or t in code units), a scale bar (comoving keeps /h,
physical multiplies by a and divides by h, non-comoving runs show the unit only, perspective frames say "at view
centre"), and a colour bar whose title shrinks to fit.

## Following a selection

Selected particles are found in every snapshot by ParticleID at full resolution (`snapshot_io/id_lookup.py`, or the
on-disk ID index), never only where they happen to fall in the drawn sample, and a gas particle that becomes a star is
followed into PartType4. With *recentre* on, the frame centre is the selected particles' periodic mean (every member
moved to the image nearest the first), smoothed over the last N frames as in GadgetViewer, but averaged across the
periodic boundary correctly. Trails draw the paths of up to N particles per set and the set's centre, unwrapped
across the box. Positions are in the code units stored in the file; `track export` writes physical values too when
the file carries a_scaling and h_scaling.

## Output files and encoders

| File | Content |
|---|---|
| `<base>_00000.png …` | frames, numbered consecutively for every kind of movie |
| `<base>.mp4`, `<base>.gif` | encoded movies (Custom) or the single file (Quick) |
| `<base>_movie.json` | the settings, the start camera, and per frame the camera, snapshot, time, whether it is interpolated, the selection centre and the smoothing range used, so a movie can be remade or extended |

MP4 uses ffmpeg when it is installed and OpenCV (H.264 through AVFoundation on macOS) otherwise; GIF uses ffmpeg's
palette filter or Pillow. With none of these, PNG frames are still written and the log gives the ffmpeg command to
encode them later.

## What has been checked

| Claim | Status | Evidence |
|---|---|---|
| Frames at a snapshot are the snapshot itself | Verified | frames 0, 3, 6 of a movie with two in-betweens are pixel-identical to the plain movie (`test_evolving_in_between_frames_keep_snapshot_frames_exact`) |
| The Hermite curve is exact at both ends and for any cubic motion, across the periodic boundary | Verified | error below 10⁻⁹ (`test_hermite_end_points_exact`, `test_pair_exact_for_accelerated_motion_across_periodic_boundary`) |
| Comoving velocity factor dx/da = u a^(s−2)/H(a) | Verified on DM-L50-N128 | bulk flow in 3.1 Mpc/h cells against finite differences between snapshots: slope 1.027, correlation 0.997 (z ≈ 3 → 1) and 1.015 (z ≈ 1 → 0); dropping the √a gives a slope above 1.3 and fails |
| Non-comoving runs use the stored velocity as dx/dt | Code-reviewed only | a_scaling is 0 there; no snapshot pair is close enough for the same finite-difference check |
| Particles that change type keep moving smoothly | Verified | CollidingGalaxiesSFR movies: none dropped between gas and star snapshots |
| Smoothed frames conserve mass and line up with point frames and the box outline | Verified | synthetic: Σ × pixel area = total mass to 0.1 %; overlays line up to 1.5 px; image stays inside the box outline wherever the view centre is |
| Scale bar labels | Verified | `test_scale_bar_labels` (10 Mpc/h at a = 0.5, h = 0.678 is 7.37 Mpc) |
| Lazy reading keeps memory bounded | Verified (synthetic), partially verified (real) | a 40-snapshot synthetic movie never holds more than 4 scenes; a DM-L50 movie read 419k particles per snapshot with 105k loaded. A 335-snapshot movie has not been made |
| The largest z = 0 halo of DM-L50 (34,120 particles) is found in every snapshot | Verified | 34,120 of 34,120 in each of snapshots 000–002, positions identical to an independent full read sorted by ID |
| Movie camera equals the window's | Verified | view matrices equal for 20 random cameras |
| MP4 on a Mac without ffmpeg | Verified | Quick and Custom MP4 of DM-L50 through OpenCV, decoded frame by frame |

## Performance

| Movie or step | Time | Machine |
|---|---|---|
| Smoothed frame, 2M particles, 640 × 480 | 3.1 s for the first frame (with smoothing lengths), 0.9 s per later frame | native M3, 1 Nov 2026 |
| DM-L50, 9 frames, points, 2.1M particles read per snapshot, halo followed | 13 s | Linux container |
| DM-L50, 9 frames, smoothed, 1M particles per snapshot | 116 s | Linux container |
| CollidingGalaxiesSFR, 11 frames, smoothed gas and stars | 21 s | Linux container |
| CollidingGalaxiesSFR, 60-frame camera path, points | 18 s | Linux container |
| Frame of 161k points at 640 × 480 / 1920 × 1080 | 0.07 / 0.20 s | Linux container |
| Find 100 / 34,120 / 1,000,000 IDs in one DM-L50 snapshot | 0.28 / 0.27 / 1.05 s | Linux container |

The lookup cache holds about 28 bytes per particle per snapshot; above 20 million entries (60,000 particles over
335 snapshots, say) the app asks before tracking in every snapshot.

## Differences from GadgetViewer

The behaviour and defaults follow GadgetViewer 1.2.0 (200 frames, 360°, 640 × 480, centre smoothed over 10
snapshots, recentre on, unreadable snapshots skipped, six selection sets), written anew in Python. Then:

- frames are numbered consecutively for every kind of movie, so they encode directly; the snapshot of each frame is
  in the JSON;
- rotation turns about the scene's z axis, like dragging in the 3D view;
- the smoothed centre is averaged across the periodic boundary;
- an evolving movie can use the loaded snapshots or read each from disk while rendering;
- camera paths, in-between frames, smoothed frames and burnt-in scale and colour bars are additions;
- there is no graph inset in frames.

## Limitations

- In-between frames are only as smooth as the snapshot spacing allows; across large gaps they look like a
  cross-fade, and the movie says so.
- Snapshots read lazily carry no FoF-group colours (the group index lives on the loaded sample); turn lazy reading
  off for that colouring.
- Smoothed deposition is the slow step; Quick caps it at 500,000 particles per snapshot.
- Two-channel colour and inset graphs are not implemented.

## A worked example

1. `cd "Cosmoviewer Immersive" && python main.py <DM-L50-N128>/output`.
2. Move to snapshot 2. In the selection panel click *Load IDs…* and choose
   [`tracking_examples/dm_l50_halo0_z0_ids.txt`](../tracking_examples/dm_l50_halo0_z0_ids.txt), the largest z = 0 halo.
3. Step back through the snapshots: the same 34,120 particles stay highlighted and the view follows them. Tick
   *Trails* to see their paths.
4. *Options ▸ Make movie…*, Custom, Evolving, 11 in-between frames, smoothed, and *Make movie*. Frames go to
   `~/Movies/cosmoviewer/` unless you change the path. [`assets/movies/`](../assets/movies/) holds the result.
