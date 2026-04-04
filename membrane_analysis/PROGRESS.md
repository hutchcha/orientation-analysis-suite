# Membrane Analysis Pipeline — Development Progress

## Project Goal
Modular, YAML-configured pipeline for MD simulation equilibration and membrane
orientation analysis of Ras superfamily GTPases (Rheb, RhoA).  Refactored from
a monolithic Jupyter notebook into a reusable module structure.

---

## Environment Setup

**Conda environment:** `memb_analysis` (Python 3.11)

```bash
# Create and activate
conda env create -f environment.yml
conda activate memb_analysis

# Or manually
conda create -n memb_analysis python=3.11
pip install MDAnalysis numpy scipy pandas matplotlib tqdm PyYAML scikit-learn seaborn lipyphilic hdbscan
```

**Run the pipeline:**
```bash
conda activate memb_analysis
cd membrane_analysis/
python run_analysis.py testing/test_config.yaml
python run_analysis.py testing/test_config.yaml --mode equilibration
python run_analysis.py testing/test_config.yaml --mode analysis
python run_analysis.py testing/test_config.yaml --plot-only
python run_analysis.py testing/test_config.yaml --recompute tilt_rotation
```

---

## Module Status

### Core infrastructure
| Module | Status | Notes |
|--------|--------|-------|
| `core/config.py` | Done | YAML loading, universe construction, accessor helpers |
| `core/io.py` | Done | pickle cache-or-load, save/load text |
| `core/plotting.py` | Done | line_plot, style_axes, save_figure |
| `run_analysis.py` | Done | CLI entry point, module registry |

### Equilibration analyses
| Module | Status | Notes |
|--------|--------|-------|
| `analyses/rmsd.py` | Done | Cα RMSD, `superposition=True` explicit |
| `analyses/pp_distance.py` | Done | P-P bilayer thickness |
| `analyses/apl.py` | Done | Area per lipid via lipyphilic |
| `analyses/anchor_insertion.py` | Done | Anchor insertion depth |

### Orientation analyses
| Module | Status | Notes |
|--------|--------|-------|
| `analyses/tilt_rotation.py` | Done | See algorithm details below |
| `analyses/lobe_com.py` | Done | Lobe COM Z-distance + 2D KDE |
| `analyses/hbonds.py` | Done | Anchor RMSD + H-bond counting |
| `analyses/clustering.py` | Done | HDBSCAN + K-means on orientation data |

---

## Tilt/Rotation Algorithm

### Axis definition
The protein axis is the **a5 helix vector**: from the C-terminal CA of a5
(`orientation_axis_start`) to the N-terminal CA of a5 (`orientation_axis_end`).

For RhoA: resid 177 (C-term) → resid 169 (N-term)
For Rheb: resid 168 (C-term) → resid 161 (N-term)

### Tilt
Angle between the a5 axis unit vector and the membrane normal.  Range: [0°, 180°].

### Membrane normal
Estimated at frame 0 from upper-leaflet P COM minus lower-leaflet P COM,
normalised.  Falls back to +Z if no phosphorus selection is given.
**Can also be specified manually in the config** (planned — see TODO).

### Auto-flip convention
If the frame-0 axis vector is antiparallel to the estimated normal, start/end
selections are swapped.  This ensures tilt angles are consistently < 90° for a
normally oriented protein, regardless of which direction the helix points in
the coordinate frame.  Sign convention of rotation is arbitrary and consistent
between systems — only relative comparisons matter.

### Rotation
Measured in the plane perpendicular to the a5 axis, anchored at the N-terminal
end of a5:

1. **Reference vector**: a5 axis vector from the minimum-tilt frame (the frame
   where the protein is most "upright").  This defines rotation = 0.
2. **Pointer vector**: vector from the a5 N-terminal anchor COM to the
   `rotation_reference` group COM (e.g. a2/switch-II helix, resid 89:93 for RhoA).
3. Both vectors are projected onto the plane perpendicular to the current frame's
   a5 axis and normalised.
4. **Rotation = directed angle** from the projected reference to the projected
   pointer, signed by the a5 axis direction (atan2 formula).

Range: (−180°, +180°].

### Config keys required per system
```yaml
orientation_axis_start:  "protein and resid 177 and name CA"  # a5 C-term
orientation_axis_end:    "protein and resid 169 and name CA"  # a5 N-term
rotation_reference:      "protein and resid 89:93"            # pointer group
phosphorus:              "resname POPC POPE POPS and name P"  # for normal est.
```

### Validated output (RhoA-GTP, 9921 frames)
- Membrane normal: [−0.033, −0.029, +0.999]  (essentially +Z, as expected)
- Axis auto-flipped at frame 0 (a5 initially antiparallel to +Z)
- Tilt:     mean=46.4°  std=12.1°  range=[0.5°, 95.1°]
- Rotation: mean=−102.9°  std=98.2°  range=[−180°, 180°]

---

## Clustering Module (`analyses/clustering.py`)

Loads tilt/rotation data from the `tilt_rotation` pickle — no trajectory access.

**HDBSCAN**: density-based, no fixed cluster count.  Operates on 3D Cartesian
unit-vector embeddings of (tilt, rotation) so spherical geometry is respected.
NaN frames are excluded before clustering then re-expanded as label −1 (noise).

**K-means**: centroid-based, requires `n_clusters` in config.  Same Cartesian
sphere embedding.

Config block:
```yaml
clustering:
  enabled: true
  hdbscan:
    min_cluster_size: 200
    min_samples: null      # defaults to min_cluster_size // 2
  kmeans:
    n_clusters: 3
```

---

## Testing

### Test files
- `testing/test_config.yaml` — full pipeline config for RhoA-GTP test trajectory
- `testing/test_tilt_rotation.py` — standalone tilt/rotation test, no package imports
- `testing/generate_test_data.py` — generates a synthetic PDB+DCD if needed

### Run standalone tilt/rotation test
```bash
conda activate memb_analysis
cd membrane_analysis/
python testing/test_tilt_rotation.py
# Outputs to testing/results/tilt_rotation/
```

### Test trajectory location
```
membrane_analysis/RhoA-GTP.psf
membrane_analysis/RhoA-GTP_stride20.xtc   (9921 frames, stride-20 from full traj)
```

---

## TODO

### Completed since initial refactor
- [x] `membrane_normal` manual override in per-system config (`membrane_normal: [x,y,z]`)
- [x] Full pipeline end-to-end validated on RhoA-GTP (9921 frames, all modules passing)
- [x] All selections filled in for test system
- [x] APL: replaced lipyphilic leaflet assignment with static midplane-based assignment
- [x] APL: fixed POPS missing from apl_headgroup selection
- [x] Consistent multi-system plotting: shared axes grid, `fig.supxlabel`/`fig.supylabel`
- [x] Publication-quality rcParams (Arial, axes.linewidth=1.5, pdf/ps/svg fonttype)
- [x] pip-installable package (`pip install -e .`) with `membrane_analysis.*` namespace
- [x] Rotation sign convention corrected (negated directed angle)
- [x] `.gitignore` covering trajectories (.psf/.pdb/.xtc/.dcd) and testing directory

### Analyses
- [ ] Comparison plot generation (side-by-side pairs, 2x2 grids, triplets)
- [ ] Smart auto-grouping of systems for comparison plots
- [ ] Residue contact frequency module (Chase has standalone code)
- [ ] Frame extraction module (representative structures from orientation windows)
- [ ] Orientation-state contact comparison module

### Pipeline
- [ ] Per-system config validation (warn on missing selections before loading)
- [ ] Optional parallel computation across systems
- [ ] SVG/PDF output format toggle in config
