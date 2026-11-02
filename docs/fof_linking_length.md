# FoF on runs from other codes: set the linking length by hand

*Written 28 Oct 2026. This page describes a workaround; the code itself has not been changed.*

Use this page when you run *Structures ▸ Find FoF groups* on anything that is not a Gadget-4 output: EAGLE, AGORA, Gadget-2/3, GIZMO, AREPO, or any zoom-in.

## Why the default fails on these runs

The default, *b × mean separation*, works on Gadget-4 files only. On other files it fails in one of two ways:

- **With dark matter ticked (the default)** it stops with *"snapshot has no cosmology/units in its Parameters group; use convention='box'"*. CosmoViewer reads Ω and the units only from the `Parameters` group, which Gadget-4 writes and other codes leave empty or omit. The dialog has no 'box' choice.
- **With any other type ticked as well** it silently uses (BoxSize³ / N)^(1/3). In a zoom-in, the high-resolution particles fill a small part of the box, so this length is about 8 to 10 times too long and the whole high-resolution region becomes one group. On a synthetic Gadget-3-style zoom-in this gave l = 149-188 kpc/h instead of 18.75 and one group of every particle (Verified, synthetic data). That giant group is the likely cause of the 19 Oct out-of-memory crash on the AGORA snapshot (Not verified: no AGORA file was available).

The linker itself is fine: given the correct length by hand, the same synthetic zoom-in gave back all 20 planted haloes (Verified, synthetic data).

## The workaround

1. Open *Structures ▸ Find FoF groups*.
2. Under *Link types*, tick **DM only**. In a zoom-in, this is the high-resolution dark matter (PartType1); untick Disk and Bulge, which hold the low-resolution particles in most zoom-ins.
3. Choose **Fixed linking length** and type the *l* computed below, in the file's own length units (the unit shown next to the box; "code units" means the units of `Coordinates` in the file).
4. Keep *N*min = 32 and run.
5. Check the result (see *Sanity checks* below).

## Computing *l*

This is Gadget-4's own definition (`src/fof/fof.cc`, `fof_get_comoving_linking_length`, lines 552-573):

```latex
l = b \left( \frac{m_\mathrm{DM}}{\Omega_\mathrm{dm} \, \rho_\mathrm{crit}} \right)^{1/3}, \qquad b = 0.2
```

- *m*DM: the mass of one (high-resolution) dark-matter particle, `Header/MassTable[1]`, or the mean of `PartType1/Masses` if that entry is 0.
- Ωdm = Ω0 − Ωb if the run has gas (`NumPart_Total[0] > 0`), otherwise Ω0. This matters: leaving out Ωb in a hydro run makes *l* about 6% too short.
- ρcrit = 3H0²/(8πG) = 2.775 × 10¹¹ h² M☉ Mpc⁻³. In Gadget's usual units that is 27.75 (10¹⁰ M☉/h) / (Mpc/h)³, or 2.775 × 10⁻⁸ (10¹⁰ M☉/h) / (kpc/h)³.

Where each input is stored:

| File from | Ω0, Ωb | Units |
|---|---|---|
| Gadget-4 | `Parameters` (the default already works) | `Parameters` |
| EAGLE | `Header` (`Omega0`, `OmegaBaryon`) | `Units` (Mpc/h and 10¹⁰ M☉/h) |
| Gadget-2/3, GIZMO | `Header/Omega0` only; **no Ωb** | usually not stored; Gadget's default is kpc/h and 10¹⁰ M☉/h |
| AGORA CosmoRun | Ωm = 0.272, Ωb = 0.0455, H0 = 70.2, so Ωdm = 0.2265 (Kim et al. 2014, MUSIC configuration in the appendix) | depends on the code that wrote it |

If Ωb or the units are not in the file, take them from the run's parameter file or paper. Do not guess.

### A script that does it for you

Run it on the first file of a snapshot: `python fof_length.py snapshot_000.hdf5`. It reads only the header and, if needed, 100,000 masses. It stops if the run has gas and no `OmegaBaryon`; and it assumes kpc/h and 10¹⁰ M☉/h when no units are stored, which you should check for your run.

```python
import sys, h5py, numpy as np
RHO_CRIT = 2.775366e11   # Msun/h per (Mpc/h)^3, i.e. 3 H0^2 / (8 pi G) with H0 = 100 h km/s/Mpc
def attr(f, key):
    for g in ('Parameters', 'Units', 'Header'):
        if g in f and key in f[g].attrs:
            return float(np.atleast_1d(f[g].attrs[key])[0])
    return None
path, b = sys.argv[1], 0.2
with h5py.File(path, 'r') as f:
    hdr = f['Header'].attrs
    m = float(np.atleast_1d(hdr['MassTable'])[1])          # high-res DM particle mass, code units
    if m == 0:
        m = float(np.mean(f['PartType1/Masses'][:100000]))
    om, ob = attr(f, 'Omega0'), attr(f, 'OmegaBaryon')
    gas = int(np.atleast_1d(hdr['NumPart_Total'])[0]) > 0
    ul = attr(f, 'UnitLength_in_cm') or 3.085678e21           # default: Gadget kpc/h
    um = attr(f, 'UnitMass_in_g') or 1.989e43                 # default: 1e10 Msun/h
    if gas and ob is None:
        sys.exit('gas present but no OmegaBaryon in the file: look it up for this run')
    om_dm = om - ob if gas else om
    mpc, msun = 3.085678e24, 1.989e33
    rho = om_dm * RHO_CRIT * msun / um * (ul / mpc) ** 3        # DM density in code units
    print(f'linking length l = {b * (m / rho) ** (1/3):.6g} code length units')
```

## Worked examples

| Run | *l* from the script | Independent check | Status |
|---|---|---|---|
| DM-L50-N128, z = 0 (Gadget-4, 50 Mpc/h, 128³) | 0.078119 Mpc/h | 0.2 × 50 / 128 = 0.078125 | Verified (agree to 0.008%) |
| EAGLE RefL0025N0752, `snap_028` (16.94 Mpc/h, 752³, with gas) | 0.0045061 Mpc/h | 0.2 × 16.9425 / 752 = 0.0045060 | Verified (header only; FoF not run, see limits) |
| Synthetic Gadget-3 zoom-in (60 Mpc/h box, 6 Mpc/h zoom, 64³ high-res) | 18.75 kpc/h | 20 planted haloes recovered | Verified (synthetic) |
| AGORA | not computed | needs a real AGORA file | Not verified |

## Sanity checks after the run

- **No giant group.** If the largest group holds most of the linked particles, *l* is too long, or low-resolution particles were linked.
- **Full-box runs.** For dark matter only in a full periodic box, *l* must equal 0.2 × BoxSize / N^(1/3), with N the number of DM particles.
- **Compare with a catalogue** when one exists (*Load group catalogue*). On DM-L50 the Gadget-4 default reproduces 2,774 of 2,774 groups at z = 0.

## Known limits of the workaround

- **Four decimals.** The length box takes 4 decimal places. EAGLE's 0.0045061 Mpc/h must be entered as 0.0045, 0.13% short. On runs in Mpc units with small *l*, expect a rounding error of up to 0.00005 / *l* (Verified by reading `views/fof_dialog.py`).
- **Memory.** FoF refuses runs that need more than half the RAM at about 120 bytes per linked particle. The full EAGLE `snap_028` (425M dark-matter particles, about 51 GB) is refused on a 16 GB Mac.
- **Isolated runs with a box size.** CosmoViewer treats a run as periodic whenever `Header/BoxSize > 0`, even with a fixed length. Older codes often store a box size for isolated runs too (for example an AGORA isolated disc), and their particles would then be wrapped into the box. Check that your positions lie in [0, BoxSize) before trusting the groups (Code-reviewed only; no such file tested).
- **Groups are labelled by the length you typed**, not as "Gadget-4 convention", in the saved settings.

The code fix (reading Ω and units from `Header`/`Units`, and using Gadget-4's formula for every set of link types) has not been made yet.
