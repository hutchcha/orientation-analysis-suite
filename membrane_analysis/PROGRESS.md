# Membrane Analysis Pipeline — Development Progress

## Project Goal
Modular, YAML-configured pipeline for MD simulation equilibration and membrane
orientation analysis of Ras superfamily GTPases (Rheb, RhoA).  Designed to be
reusable across any membrane-bound protein system, with a mix-and-match
statistical analysis layer that operates on arbitrary combinations of computed
observables.

---

## Environment Setup

**Conda environment:** `memb_analysis` (Python 3.11)

```bash
conda env create -f environment.yml
conda activate memb_analysis
pip install -e .
```

**Windows note:** set `PYTHONIOENCODING=utf-8` (run `setx PYTHONIOENCODING utf-8`
in Command Prompt once) to allow Unicode symbols in console output.

**Run the pipeline:**
```bash
run-analysis config.yaml
run-analysis config.yaml --mode equilibration
run-analysis config.yaml --mode analysis
run-analysis config.yaml --plot-only
run-analysis config.yaml --recompute tilt_rotation
run-analysis config.yaml --stats stats.yaml      # also run statistical layer
```

---

## Module Status

### Core infrastructure
| Module | Status | Notes |
|--------|--------|-------|
| `core/config.py` | Done | YAML loading, universe construction, accessor helpers, stats-config loader |
| `core/io.py` | Done | pickle cache-or-load, `save_per_system` for per-system subfolders |
| `core/plotting.py` | Done | line_plot, style_axes, save_figure, overlay_line_plot |
| `core/features.py` | Done | Feature assembly: resolve cached arrays by key, align lengths, apply transforms (none/unit_sphere/spherical/auto) |
| `run_analysis.py` | Done | CLI entry point, module registry, `--stats` flag for statistical pipeline |

### Equilibration analyses
| Module | Status | Notes |
|--------|--------|-------|
| `analyses/rmsd.py` | Done | Ca RMSD with superposition |
| `analyses/pp_distance.py` | Done | P-P bilayer thickness |
| `analyses/apl.py` | Done | Area per lipid (static leaflet assignment) |
| `analyses/anchor_insertion.py` | Done | Lipid anchor insertion depth |

### Orientation / structural analyses
| Module | Status | Notes |
|--------|--------|-------|
| `analyses/tilt_rotation.py` | Done | a5 helix tilt + rotation, polar density, membrane-normal rotation reference |
| `analyses/lobe_com.py` | Done | Lobe COM Z-distance + 2D KDE |
| `analyses/hbonds.py` | Done | Anchor RMSD + H-bond counting |
| `analyses/contacts.py` | Done | Residue-lipid contact frequency (parallelised) |
| `analyses/inter_residue_distance.py` | Done | COM-COM distance between two selections |

### Statistical analysis (downstream consumers)
| Module | Status | Notes |
|--------|--------|-------|
| `analyses/clustering.py` | Done | HDBSCAN + K-means; dual-signature `compute()` supports main pipeline (tilt_rotation cache) and stats pipeline (feature assembly) |
| `analyses/hdbscan_explorer.py` | Done | 2D sweep (MCS x min_samples), condensed tree, eom/leaf |
| `analyses/gmm.py` | Done | Gaussian mixture model with BIC/AIC sweep, state assignment, polar/scatter/timeseries plots |
| `analyses/kinetics.py` | Done | Bayesian HMM via pyemma; dual-signature `compute()` for main + stats pipelines |

### CLI utilities
| Module | Status | Notes |
|--------|--------|-------|
| `traj-tools` | Done | Trajectory and topology manipulation CLI (separate entry point) |

---

## Architecture — Three-layer design (now in place)

```
Layer 1: Data-producing modules
    Each computes per-frame arrays and caches them as pickles.
    Each declares OUTPUT_TYPE metadata in core/features.py
    ("scalar" or "angular").

Layer 2: Feature assembly (core/features.py)
    Reads a list of feature keys from stats config, resolves them against
    cached pickles, aligns lengths, drops NaN frames, applies a transform.
    Transforms: "none" (column-stack), "unit_sphere" (rot+tilt -> 3D unit vec),
                "spherical" (rot+tilt+scalar -> 3D Cartesian),
                "auto" (inferred from registered feature types).

Layer 3: Statistical consumers (clustering, gmm, kinetics)
    Receive a (N, D) feature matrix from features.py.
    Don't care where the data came from — any module that follows the
    Layer 1 contract can be used as a feature.
```

### Per-system subfolder output

Every data-producing module saves both a combined pickle (all systems, as
before) and per-system subfolders with individual data + plots:

```
results/rmsd/
    rmsd.pkl                    # combined {sys1: arr, sys2: arr}
    rmsd_all.png                # multi-system grid
    RhoA-GTP/
        rmsd.pkl                # single system
        rmsd.png
    Rheb-GDP/
        rmsd.pkl
        rmsd.png
```

### Stats config split

Statistical analysis configuration lives in a separate `stats.yaml` to keep
the main config focused on systems/selections/trajectories:

```bash
run-analysis config.yaml --stats stats.yaml
```

See [`example_stats.yaml`](example_stats.yaml) for full documentation.

```yaml
feature_sets:
  orientation_2d:
    features: [tilt_rotation.rotation, tilt_rotation.tilt]
    transform: unit_sphere

  orientation_3d:
    features:
      - tilt_rotation.rotation
      - tilt_rotation.tilt
      - inter_residue_distance
    transform: spherical
    radial_scale: 0.6

clustering:
  feature_set: orientation_2d
  hdbscan: {min_cluster_size: 200, cluster_selection_method: eom}
  kmeans: {n_clusters: 3}

gmm:
  feature_set: orientation_3d
  n_components: [3, 4, 5, 6]
  covariance_type: full

kinetics:
  feature_set: orientation_3d
  K: 4
  lag: 500
  dt_time: 0.1
```

### Adding a new data-producing module

A new module just needs to:
1. Follow the `compute(cfg, universes)` / `plot(cfg, results)` contract
2. Return `{system_name: 1D_array}` (or dict of 1D arrays) from `compute()`
3. Declare `ANALYSIS_KEY` and register its output type with
   `core.features.register_feature(...)` (or rely on the default "scalar")
4. Register in the appropriate registry in `run_analysis.py`

After that, users reference it in `feature_sets` by its `ANALYSIS_KEY`.

---

## Completed work

- [x] Full pipeline end-to-end validated on RhoA-GTP (9921 frames, all modules)
- [x] `membrane_normal` manual override per system
- [x] APL: static midplane leaflet assignment, POPS fix
- [x] Multi-system shared-axes grid plots + comparison overlays
- [x] Publication-quality rcParams (Arial, Angstrom/mu/degree Unicode symbols)
- [x] pip-installable package (`pip install -e .`)
- [x] HDBSCAN: 2D parameter sweep, condensed tree, eom/leaf, scatter plots
- [x] Contacts module (parallelised, structural annotation bands)
- [x] Inter-residue distance module
- [x] HMM kinetics module (spherical + direct modes)
- [x] **Phase 1**: feature assembly layer + stats config + OUTPUT_TYPE metadata
- [x] **Phase 2a**: GMM module with BIC/AIC sweep via feature assembly
- [x] **Phase 2b**: clustering refactored onto feature assembly (dual-signature)
- [x] **Phase 2c**: kinetics refactored onto feature assembly (dual-signature)
- [x] **Phase 3**: `--stats` CLI flag, example `stats.yaml` documented
- [x] Per-system subfolder output across all data-producing modules
- [x] `traj-tools` CLI for trajectory/topology manipulation
- [x] Membrane normal used as rotation reference (Travers et al. 2018)
- [x] Tilt/rotation citations updated (Neale & Garcia 2020, Li & Buck 2017)
- [x] GitHub repo (hutchcha/orientation-analysis-suite)
- [x] README.md + CLUSTERING_KINETICS.md documentation
- [x] Track `initial_systems/` Rheb-GDP/GTP starting structures (.pdb/.psf)

---

## Roadmap — Future work

- [ ] Frame extraction module (representative structures from orientation windows)
- [ ] Orientation-state contact comparison module (compare contacts between
      HMM/GMM-defined states to identify state-specific lipid interactions)
- [ ] Per-system config validation (warn on missing selections before loading)
- [ ] Optional parallel computation across systems
- [ ] SVG/PDF output format toggle in config
- [ ] `gmm` as microstate source for `kinetics` (`microstate_source: gmm`)

---

## Tilt/Rotation Algorithm

### Axis definition
The protein axis is the **a5 helix vector**: from the C-terminal CA of a5
(`orientation_axis_start`) to the N-terminal CA of a5 (`orientation_axis_end`).

For RhoA: resid 177 (C-term) to resid 169 (N-term)
For Rheb: resid 168 (C-term) to resid 161 (N-term)

### Tilt
Angle between the a5 axis unit vector and the membrane normal.  Range: [0, 180].

### Membrane normal
Estimated at frame 0 from upper-leaflet P COM minus lower-leaflet P COM.
Falls back to +Z if no phosphorus selection is given.
Can be manually specified via `membrane_normal: [x, y, z]` in per-system config.

### Auto-flip convention
If the frame-0 axis vector is antiparallel to the estimated normal, start/end
selections are swapped to keep tilt angles physically interpretable.

### Rotation
Measured in the plane perpendicular to the membrane normal (Travers et al.
2018 convention; see `repo_citations.md`):

1. Reference vector: a5 axis from the minimum-tilt frame (rotation = 0).
2. Pointer vector: axis endpoint to rotation_reference group COM.
3. Both projected onto the plane perpendicular to the membrane normal.
4. Directed angle via atan2. Range: (-180, +180].

---

## Testing

### Test files
- `testing/test_config.yaml` — full pipeline config for RhoA-GTP test trajectory
- `testing/test_stats.yaml` — minimal stats config (orientation_2d + GMM)
- `testing/hdbscan_explorer/` — standalone HDBSCAN exploration with Rheb-GDP data

### Test trajectory
```
membrane_analysis/RhoA-GTP.psf
membrane_analysis/RhoA-GTP_stride20.xtc   (9921 frames)
```

---

## Legacy / prototype scripts

The following untracked scripts in this directory are the **original
exploratory / monolithic versions** that the modular pipeline was extracted
from.  They are kept locally for reference but are NOT the source of truth
and should not be edited or imported from:

| File | Purpose | Superseded by |
|------|---------|---------------|
| `3D_V2.py` | Original analysis script — early monolithic version that several pipeline modules were extracted from | The modular `analyses/*.py` modules |
| `pyemma_gmm_v2.py` | Standalone PyEMMA + GMM prototype | `analyses/gmm.py` and `analyses/kinetics.py` |
| `contacts_parralel.py` | Standalone parallel contacts prototype | `analyses/contacts.py` |
| `Rheb_GDP{rotation,tilt}angles.txt` | Old text-format angle exports (Jul 2025) | tilt_rotation cache pickles |

These are listed in `.gitignore`-equivalent local-only state — if you see
them and wonder "should I be using this?", the answer is **no**, use the
corresponding pipeline module instead.
