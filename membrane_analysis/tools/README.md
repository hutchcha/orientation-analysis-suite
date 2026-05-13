# `traj-tools` — Trajectory & Topology Utilities

A small command-line companion to the analysis pipeline for the routine "I just need to clean up this trajectory" jobs: stripping water, centering, wrapping, aligning, striding, and writing topology subsets. Everything is built on MDAnalysis, so any format MDAnalysis can read or write is supported (DCD, XTC, TRR, PSF, PDB, GRO, TPR, …).

The tool is installed as a console script when you `pip install -e .` the parent package — `traj-tools` will be on your `$PATH` inside the `memb_analysis` env.

---

## Subcommands

```
traj-tools <command> [options]
```

| Command | Purpose |
|---------|---------|
| [`write`](#write--write-a-subtrajectory-with-optional-transformations)    | Write a (sub)trajectory with optional centering, wrapping, alignment, and striding. Always emits a matching PDB reference topology alongside. |
| [`topology`](#topology--write-a-topology-file-for-a-subset-of-atoms) | Write a single-frame topology file (PDB/GRO/PSF) for a subset of atoms. |

Run `traj-tools` with no arguments to see the top-level help, or `traj-tools <command> --help` for full per-command flags.

---

## `write` — write a (sub)trajectory with optional transformations

Writes a new trajectory containing only the atoms matched by `--select`. Optional transformations (`--center`, `--wrap`, `--align`) are stacked as MDAnalysis on-the-fly transformations, so they all happen in one pass and you don't pay for an intermediate file.

The output format is inferred from the extension on `--out` (so `.xtc`, `.dcd`, `.trr` all work). A matching `.pdb` reference topology with the same atom subset is **always** written next to the trajectory — that's the file you'll point downstream tools at as the topology.

### Flags

| Flag | Required | Default | Description |
|------|---------|---------|-------------|
| `--top`         | yes      | —       | Input topology (PSF, PDB, GRO, TPR, …) |
| `--traj`        | yes      | —       | Input trajectory (DCD, XTC, TRR, …) |
| `--out`         | yes      | —       | Output trajectory path (format inferred from extension) |
| `--select`      | no       | `all`   | MDAnalysis selection for atoms to write |
| `--center`      | no       | none    | Selection to center at the box origin (uses `transformations.center_in_box`) |
| `--wrap`        | no       | off     | Wrap atoms into the primary unit cell. Combined with `--center` for the standard "center on protein, wrap everything" workflow. |
| `--align`       | no       | none    | Selection for RMSD alignment to `--ref-frame` (uses `transformations.fit_rot_trans`) |
| `--ref-frame`   | no       | `0`     | Reference frame index for `--align` |
| `--stride`      | no       | `1`     | Write every Nth frame |
| `--start`       | no       | `0`     | First frame index (inclusive) |
| `--stop`        | no       | end     | Last frame index (exclusive) |

### Recipes

**Strip water + ions, write XTC**
```bash
traj-tools write --top sys.psf --traj prod.dcd \
    --select "not resname TIP3 SOD CLA" \
    --out nowat.xtc
```

**Center on protein, wrap everything**
```bash
traj-tools write --top sys.psf --traj prod.dcd \
    --select "all" \
    --center "protein" --wrap \
    --out centered.xtc
```

**Align on protein backbone, write only protein**
```bash
traj-tools write --top sys.psf --traj prod.dcd \
    --select "protein" \
    --align "protein and backbone" \
    --out aligned.xtc
```

**Stride a trajectory by 10**
```bash
traj-tools write --top sys.psf --traj prod.dcd \
    --select "all" --stride 10 \
    --out strided.xtc
```

**The whole-pipeline recipe — strip water, center, wrap, align, stride**
```bash
traj-tools write --top sys.psf --traj prod.dcd \
    --select "not resname TIP3 SOD CLA" \
    --center "protein" --wrap \
    --align "protein and name CA" \
    --stride 10 \
    --out final.xtc
```

**Trim a frame range (e.g. drop pre-equilibration)**
```bash
traj-tools write --top sys.psf --traj prod.dcd \
    --select "all" --start 5000 --stop 50000 \
    --out post_eq.xtc
```

---

## `topology` — write a topology file for a subset of atoms

Writes a single-frame topology (PDB, GRO, or PSF) containing only the selected atoms. Useful for generating a clean reference topology for visualization or as input to a separate pipeline run.

### Flags

| Flag | Required | Default | Description |
|------|---------|---------|-------------|
| `--top`     | yes  | —      | Input topology (PSF, PDB, GRO, …) |
| `--traj`    | no   | none   | Optional trajectory; needed when the input topology lacks coordinates and for PSF subset writing |
| `--out`     | yes  | —      | Output topology path. Format inferred from extension (`.pdb`, `.gro`, `.psf`) |
| `--select`  | no   | `all`  | MDAnalysis selection |
| `--frame`   | no   | `0`    | Frame to use for coordinates (when a trajectory is provided) |

### Recipes

**Write a clean PDB of just the protein + lipids (no water)**
```bash
traj-tools topology --top sys.psf --traj prod.dcd \
    --select "protein or resname POPC POPE POPS CHL1" \
    --out nowat.pdb
```

**Pull a single frame as PDB at frame 5000**
```bash
traj-tools topology --top sys.psf --traj prod.dcd \
    --select "all" --frame 5000 \
    --out frame_5000.pdb
```

**PSF subset (requires `parmed`)**
```bash
pip install parmed
traj-tools topology --top sys.psf --traj prod.dcd \
    --select "protein or resname POPC POPE POPS" \
    --out subset.psf
```

> **PSF caveat:** MDAnalysis cannot write PSF files directly. The PSF writer falls back to [`parmed`](https://parmed.github.io/ParmEd/), which preserves bond/angle topology for the kept atoms. If `parmed` isn't installed, a PDB is written alongside in its place and the tool prints a hint to install it.

---

## Format notes

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PSF    | yes  | via parmed | MDAnalysis reads PSF natively; PSF subset writing requires `parmed`. |
| PDB    | yes  | yes   | Single-frame topology + multi-frame trajectory both work. |
| GRO    | yes  | yes   | GROMACS topology format. |
| TPR    | yes  | no    | Read-only; TPR cannot be re-written. |
| DCD    | yes  | yes   | NAMD/CHARMM trajectory. |
| XTC    | yes  | yes   | GROMACS compressed trajectory (recommended for long sims). |
| TRR    | yes  | yes   | GROMACS uncompressed trajectory. |

---

## Tips

- **Always check the printed atom count.** The tool reports `Selection 'X' -> N atoms` before writing. If `N` is `0`, the selection didn't match — fix it before re-running.
- **The auto-emitted PDB is the matching topology for the new trajectory.** Don't reuse the original PSF/topology with the subset trajectory — atom indices won't match.
- **Stack transformations in one pass.** `--center` + `--wrap` + `--align` + `--stride` all run in a single trajectory iteration, so combining them is free — much faster than chaining intermediate files.
- **Selection syntax** is standard [MDAnalysis selection language](https://userguide.mdanalysis.org/stable/selections.html). Common patterns: `"protein"`, `"resname POPC"`, `"protein and name CA"`, `"not resname TIP3 SOD CLA"`, `"resid 100:200"`, `"protein and resid 50 and name CA"`.
- **For `--align`, pick a stable reference selection.** `"protein and backbone"` or `"protein and name CA"` work well; flexible loops will smear your alignment.
