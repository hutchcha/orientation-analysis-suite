# Orientation Dynamics Analysis Suite

A modular, YAML-configured pipeline for MD simulation equilibration checks and membrane protein orientation analysis. Originally developed for Ras superfamily small GTPases (Rheb, RhoA), but designed to work with any membrane-bound protein system.

## Installation

Requires Python >= 3.10. Install via pip in editable mode:

```bash
# Create conda environment (recommended)
conda create -n memb_analysis python=3.11
conda activate memb_analysis

# Install the package
cd Orientation_dynamics_analysis_suite
pip install -e .
```

### Dependencies

Core:
- MDAnalysis (>= 2.7)
- numpy, scipy, pandas, matplotlib
- PyYAML, tqdm

Analysis-specific:
- lipyphilic (APL module)
- hdbscan (HDBSCAN clustering / explorer)
- scikit-learn (K-means clustering, GMM microstates)
- seaborn (optional, KDE plots)
- pyemma (kinetics / HMM module)
- networkx (flux network graphs)

All core dependencies install automatically with `pip install -e .`. For the optional analysis packages:

```bash
pip install lipyphilic hdbscan pyemma networkx
```

## Quick Start

```bash
# Copy and edit the example config
cp membrane_analysis/example_config.yaml my_config.yaml
# Edit my_config.yaml with your paths and selections...

# Run full pipeline
run-analysis my_config.yaml

# Equilibration checks only
run-analysis my_config.yaml --mode equilibration

# Orientation analysis only
run-analysis my_config.yaml --mode analysis

# Re-plot from cached data (no trajectory loading)
run-analysis my_config.yaml --plot-only

# Force recompute specific analyses
run-analysis my_config.yaml --recompute rmsd tilt_rotation
```

## Architecture

```
membrane_analysis/
├── run_analysis.py              # CLI entry point
├── example_config.yaml          # Documented example configuration
├── core/
│   ├── config.py                # YAML parsing, universe construction
│   ├── io.py                    # Pickle cache-or-load helpers
│   └── plotting.py              # Unified plotting utilities
└── analyses/
    ├── rmsd.py                  # Protein RMSD
    ├── pp_distance.py           # P-P bilayer thickness
    ├── apl.py                   # Area per lipid
    ├── anchor_insertion.py      # Lipid anchor insertion depth
    ├── tilt_rotation.py         # Orientation tilt & rotation angles
    ├── lobe_com.py              # Protein lobe COM Z-distances
    ├── hbonds.py                # H-bond counting + anchor RMSD
    ├── clustering.py            # HDBSCAN + K-means clustering
    ├── hdbscan_explorer.py      # HDBSCAN parameter exploration
    ├── contacts.py              # Residue-lipid contact frequency
    ├── inter_residue_distance.py# Inter-selection COM distance
    └── kinetics.py              # Bayesian HMM kinetics (pyemma)
```

## Configuration

All system-specific information (topology/trajectory paths, atom selections, analysis parameters) lives in a single YAML config file. See `membrane_analysis/example_config.yaml` for a fully documented template.

Key config sections:

- **`global`** — output directory, simulation length, force recompute flag
- **`style`** — matplotlib font family and DPI
- **`systems`** — one entry per simulation with topology, trajectory, and all atom selections
- **`analyses`** — grouped into `equilibration` and `analysis`, each module individually enabled/disabled with its own parameters

### Cache-or-load pattern

Every analysis module follows the same contract:
- `compute(cfg, universes)` checks for a cached pickle; if found, loads it. Otherwise computes and saves.
- `plot(cfg, results)` generates figures from the cached results.
- `--plot-only` skips trajectory loading entirely and regenerates plots from cache.
- `--recompute <key>` forces specific analyses to re-run.

## Modules

### Equilibration

| Module | Description | Required selections |
|--------|-------------|-------------------|
| **rmsd** | C-alpha RMSD vs frame 0 with backbone superposition | `rmsd` |
| **pp_distance** | P-P bilayer thickness (upper/lower leaflet phosphorus COM Z-distance) | `phosphorus` |
| **apl** | Area per lipid via lipyphilic, with per-lipid-type breakdown | `apl_headgroup` + `lipid_headgroups` dict |
| **anchor_insertion** | Lipid anchor insertion depth (membrane COM Z - anchor COM Z) | `anchor`, `membrane_heavy` |

All equilibration modules produce per-system time series with raw trace + moving average overlay. When multiple systems are configured, a comparison overlay plot is also generated.

### Orientation Analysis

| Module | Description | Required selections |
|--------|-------------|-------------------|
| **tilt_rotation** | Helix axis tilt (vs membrane normal) and rotation angle | `orientation_axis_start`, `orientation_axis_end`, `rotation_reference`, `phosphorus` (optional) |
| **lobe_com** | Protein lobe COM Z-distances from membrane + 2D KDE contour | `lobe1`, `lobe2`, `membrane_com` |
| **hbonds** | H-bond counting between protein anchor region and lipids | `anchor_rmsd`, `align` + `hbonds` config block |
| **contacts** | Residue-lipid contact frequency with structural annotation bands | `protein_heavy`, `lipid_heavy` |
| **inter_residue_distance** | COM distance between two atom selections per frame | `distance_sel1`, `distance_sel2` |

### Clustering & Kinetics

| Module | Description |
|--------|-------------|
| **clustering** | HDBSCAN + K-means on spherical embeddings of tilt/rotation. Optional auto MCS via DBCV sweep. |
| **hdbscan_explorer** | Standalone HDBSCAN exploration: MCS sweep, DBCV validation, condensed tree, scatter plots. |
| **kinetics** | Bayesian HMM via pyemma: implied timescale scan, transition matrix, MFPT, state populations, dwell times, CK test, flux network. |

See [CLUSTERING_KINETICS.md](CLUSTERING_KINETICS.md) for detailed documentation of the clustering and kinetics modules.

## Tilt & Rotation Algorithm

The orientation analysis follows the method described in:

> Travers, T., Lopez, C. A., Van, Q. N. et al. Molecular recognition of RAS/RAF complex at the membrane: Role of RAF cysteine-rich domain. *Sci Rep* 8, 8461 (2018). https://doi.org/10.1038/s41598-018-26832-4

The protein orientation is defined by a helix axis vector (typically alpha-5) and a structural reference group (e.g., alpha-2/switch-II):

- **Tilt (theta_t)**: angle between the helix axis unit vector (z') and the membrane normal (z). Range: [0, 180] degrees. 0 = axis parallel to normal (upright), 90 = axis in membrane plane.

- **Rotation (theta_r)**: at each frame, define the plane S perpendicular to the helix axis (z') and passing through the reference group COM. Project the membrane normal (z) onto S to get vector p. Construct the pointer vector a from the helix endpoint to the reference group COM (already in S by construction). The rotation angle is the directed angle from a to p around z'.

The membrane normal serves as the rotation reference, so rotation = 0 means the reference group is pointing in the same in-plane direction as the membrane normal's projection. This is well-defined regardless of whether the protein ever samples low tilt angles.

**Membrane normal**: estimated from upper/lower leaflet phosphorus COMs at frame 0, or manually specified via `membrane_normal: [x, y, z]` in the per-system config.

**Auto-flip**: at frame 0, if the axis vector is antiparallel to the membrane normal, the start/end selections are swapped so tilt angles remain physically interpretable.

## Plotting

All plots use a consistent publication-quality style:
- Arial font, `axes.linewidth = 1.5`
- PDF/PS/SVG text preserved as text (not paths)
- Multi-system figures use shared axes grids (max 2 columns)
- Comparison overlays generated automatically when > 1 system is configured
- Raw traces at alpha=0.25 with centered moving average overlay

## Example Selections (RhoA)

```yaml
selections:
  rmsd: "protein and name CA and not (resid 25:42 58:78 120:140 178:191)"
  anchor: "resname CYSG"
  membrane_heavy: "resname POPC POPE POPS PLA18 PSM"
  phosphorus: "name P"
  orientation_axis_start: "protein and resid 177 and name CA"  # a5 C-term
  orientation_axis_end:   "protein and resid 169 and name CA"  # a5 N-term
  rotation_reference: "protein and resid 89:93"                # a2/SwII
  protein_heavy: "(protein and not name H*) or (resname CYSG and not name H*)"
  lipid_heavy:   "resname POPC POPE POPS PLA18 PSM and not name H*"
```

## References

This pipeline relies on several open-source tools and is based on published methods. If you use this software, please cite the relevant papers:

**Orientation analysis method:**

1. Travers, T., Lopez, C. A., Van, Q. N. et al. Molecular recognition of RAS/RAF complex at the membrane: Role of RAF cysteine-rich domain. *Sci Rep* 8, 8461 (2018). https://doi.org/10.1038/s41598-018-26832-4

**Core dependencies:**

2. Gowers, R. J. et al. MDAnalysis: A Python Package for the Rapid Analysis of Molecular Dynamics Simulations. *Proceedings of the 15th Python in Science Conference* 98-105 (2016). doi:10.25080/Majora-629e541a-00e.
3. Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat Methods* 17, 261-272 (2020).

**Clustering:**

4. McInnes, L., Healy, J. & Astels, S. hdbscan: Hierarchical density based clustering. *Journal of Open Source Software* 2, 205 (2017).
5. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12, 2825-2830 (2011).

**Kinetics:**

6. Scherer, M. K. et al. PyEMMA 2: A Software Package for Estimation, Validation, and Analysis of Markov Models. *J. Chem. Theory Comput.* 11, 5525-5542 (2015).

**Lipid analysis:**

7. Smith, P. & Lorenz, C. D. LiPyphilic: A Python Toolkit for the Analysis of Lipid Membrane Simulations. *J. Chem. Theory Comput.* 17, 5907-5919 (2021).

## License

This project is currently private. Contact the repository owner for access.
