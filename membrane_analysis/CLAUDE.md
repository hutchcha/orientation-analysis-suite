# CLAUDE.md — Membrane Analysis Pipeline

## Project Overview

This is a modular, YAML-configured pipeline for MD simulation equilibration and membrane orientation analysis of Ras superfamily small GTPases (Rheb, RhoA, and similar systems). It is designed to be reusable across any membrane-bound protein system.

The code was refactored from a monolithic Jupyter notebook into a clean module structure with a cache-or-load pattern so analyses are only computed once and plots can be regenerated from saved data.

## Architecture

```
membrane_analysis/
├── run_analysis.py          # CLI entry point (argparse)
├── example_config.yaml      # Fully documented example config
├── core/
│   ├── io.py                # pickle cache-or-load, save/load text
│   ├── config.py            # YAML parsing, universe construction, accessor helpers
│   └── plotting.py          # Unified line_plot, style_axes, save_figure
├── analyses/
│   ├── rmsd.py              # [equilibration] Cα RMSD
│   ├── pp_distance.py       # [equilibration] P-P bilayer thickness
│   ├── apl.py               # [equilibration] Area per lipid (lipyphilic)
│   ├── anchor_insertion.py  # [equilibration] Lipid anchor insertion depth
│   ├── tilt_rotation.py     # [analysis] Orientation angles + HDBSCAN clustering
│   ├── lobe_com.py          # [analysis] Lobe COM Z-distance + 2D KDE contours
│   └── hbonds.py            # [analysis] Anchor RMSD + H-bond counting
```

## Key Design Decisions

### Cache-or-load pattern
Every analysis module follows the same contract:
- `compute(cfg, universes)` → checks for cached pickle; if found and not force_recompute, loads it; otherwise computes and saves.
- `plot(cfg, results)` → generates figures from the results dict.
- The `--plot-only` flag skips universe loading entirely and just re-plots from cache.
- `--recompute rmsd apl` forces only specific analyses to re-run.

### Configuration
- All system-specific info (paths, selections, strides) lives in a YAML config file.
- Per-system `stride_overrides` take priority over the analysis-level default stride.
- H-bonds default to `guess_hydrogens()` if `donors: null`; explicit `between:` lists override.
- APL uses a `lipid_headgroups` dict per system mapping resname → head-atom selection (handles CHL1 O3 vs phospholipid P).

### Plotting
- Single plot per system per analysis (no comparison plots yet — that's a future feature).
- One unified `line_plot()` function: raw trace at alpha=0.25, centered moving average on top carrying the legend label.
- One unified `style_axes()` for publication formatting.
- All matplotlib rcParams set once via `setup_style()`.

### Module groups
- **Equilibration**: RMSD, P-P distance, APL, anchor insertion depth
- **Analysis**: tilt & rotation, lobe COM, anchor-RMSD + H-bonds
- Selectable via `--mode equilibration|analysis|all`

## Logic Quirks Preserved from Original Notebook

1. **Lobe COM sign convention**: `−1 × (lobe_z − membrane_z)`, so positive values = lobe closer to membrane. This is intentional.

2. **P-P distance midplane**: Defined once at frame 0 from phosphorus COM, then used as the split threshold for all subsequent frames. The midplane is NOT recalculated per frame.

3. **APL per-lipid skip behavior**: If a lipid type isn't in the system, it's silently skipped with a log message. Upper/Lower POPC breakdown is computed when POPC exists.

4. **HDBSCAN on the sphere**: Tilt/rotation are embedded as 3D Cartesian unit vectors before clustering. The `min_cluster_size` default is 200. Cluster centers are computed as vector means on the sphere, converted back to (rot, tilt).

5. **Polar density plots**: Use circular padding + Gaussian smoothing to handle the 360° wrap. Edge bins are blended for seamless display.

6. **Anchor insertion depth**: Convention is `membrane_COM_z − anchor_COM_z`. The membrane selection uses heavy atoms (no hydrogens).

## Dependencies

- MDAnalysis, numpy, pandas, matplotlib, scipy, PyYAML, tqdm
- lipyphilic (for APL)
- hdbscan (optional, for tilt/rotation clustering)
- seaborn (optional, for some KDE plots)

## How to Run

```bash
# Full pipeline
python run_analysis.py my_config.yaml

# Equilibration only
python run_analysis.py my_config.yaml --mode equilibration

# Re-plot from cached data (no trajectory loading)
python run_analysis.py my_config.yaml --plot-only

# Force recompute specific analyses
python run_analysis.py my_config.yaml --recompute rmsd tilt_rotation
```

## Next Steps / TODO

- [ ] Comparison plot generation (side-by-side pairs, 2x2 grids, triplets)
- [ ] Smart auto-grouping of systems for comparison plots
- [ ] Residue contact frequency module (Chase has better standalone code for this)
- [ ] Frame extraction module (representative structures from lobe/orientation windows)
- [ ] Orientation-state contact comparison module
- [ ] Per-system config validation (warn on missing selections before loading trajectories)
- [ ] Optional parallel computation across systems
- [ ] SVG/PDF output format toggle in config
