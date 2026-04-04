# Clustering & Kinetics Modules

Detailed documentation for the orientation-state clustering and HMM kinetics modules. These modules operate downstream of the `tilt_rotation` analysis and optionally the `inter_residue_distance` analysis.

## Clustering (`analyses/clustering.py`)

Clusters orientation states from tilt/rotation angle data using two methods: HDBSCAN (density-based) and K-means (centroid-based). Both operate on 3D Cartesian unit-vector embeddings so that the circular nature of rotation angles is handled correctly.

### Spherical embedding

Tilt and rotation angles are converted to unit vectors on a sphere:

```
x = sin(tilt) * cos(rotation)
y = sin(tilt) * sin(rotation)
z = cos(tilt)
```

This avoids wrap-around artifacts at +/-180 degrees that would break Euclidean distance-based clustering.

### HDBSCAN

Density-based clustering that finds arbitrarily shaped clusters without requiring a fixed cluster count. Points in low-density transition regions are labelled as noise (-1).

HDBSCAN has three key parameters:
- **`min_cluster_size`** — the minimum number of points a dense region must contain to be called a cluster. Larger values produce fewer, broader clusters.
- **`min_samples`** — controls how conservative the density estimate is. Lower values allow more points into clusters (less noise); higher values require denser regions. Must be <= min_cluster_size.
- **`cluster_selection_method`** — `"eom"` (Excess of Mass, default) selects the most persistent clusters from the hierarchy, tending toward fewer large clusters. `"leaf"` selects leaf nodes, producing more granular sub-states.

**Fixed parameter mode** (default and recommended for production):
```yaml
clustering:
  enabled: true
  hdbscan:
    min_cluster_size: 200
    min_samples: null       # null = min_cluster_size // 2
    cluster_selection_method: eom   # "eom" or "leaf"
    auto_mcs: false
```

**Exploration mode** — runs a 2-D sweep over (min_cluster_size x min_samples) and scores each combination with DBCV. This is a diagnostic tool to help you understand the parameter landscape and choose values manually. The "optimal" parameters selected by peak DBCV are a starting point, not a final answer — DBCV tends to favor small, tight clusters and may fragment what is physically a single state into micro-clusters, especially when the data has one dominant orientation. Always inspect the sweep heatmap and condensed tree, then set your parameters in fixed mode for production runs.

```yaml
clustering:
  hdbscan:
    auto_mcs: true
    cluster_selection_method: eom   # "eom" or "leaf"
    sweep_n_mcs: 15        # MCS values in the 2-D sweep grid
    sweep_n_ms: 8          # min_samples values in the grid
    mcs_min: 50            # smallest MCS to sweep
    mcs_max: null          # null = 5% of subsample size
    ms_min: 5              # smallest min_samples to sweep
    ms_max: null           # null = mcs_max
    sweep_subsample: 20000 # frames used during sweep (subsampled for speed)
```

**Outputs (per system):**
- `*_hdbscan_sweep_2d.png` — DBCV heatmap over the (MCS, min_samples) grid, annotated with cluster count per cell, optimal cell starred (exploration mode only)
- `*_hdbscan_sweep_1d.png` — Noise fraction + DBCV vs MCS at the optimal min_samples (exploration mode only)
- `*_hdbscan_condensed_tree.png` — Cluster persistence bars (exploration mode only)
- `*_hdbscan_scatter.png` — Rotation vs tilt scatter with points colored by cluster
- `*_hdbscan_polar.png` — Polar density contour with cluster-centre stars
- `*_hdbscan_timeseries.png` — Frame-by-frame tilt/rotation colored by cluster

**Interpreting the 2-D sweep heatmap:**

Each cell shows the DBCV score (color) and number of clusters (k=N annotation) for that (MCS, min_samples) combination. Green = higher DBCV, red = lower. The starred cell is the DBCV maximum.

Key things to look for:
- A clear green region suggests genuine cluster structure. Choose parameters from within that region.
- If the entire heatmap is near-zero DBCV (all red/dark), the data likely has only one orientation state — clustering is not appropriate.
- Sharp transitions (green to red) as MCS increases indicate the boundary where a real cluster gets absorbed into noise.

**Interpreting the condensed tree:**

Each bar represents a cluster node in HDBSCAN's hierarchy. Bar length = the lambda range over which that cluster persists (longer = more stable). Bar thickness = relative cluster size. Colored bars with `*` are selected by HDBSCAN; grey bars were rejected. Use this to understand which clusters are robust vs. transient, and to decide whether `eom` or `leaf` is more appropriate for your data.

**Recommended workflow:**
1. Start with `auto_mcs: true` to generate the sweep heatmap and condensed tree.
2. Inspect the plots. If DBCV is near zero everywhere, your data probably doesn't have distinct sub-states.
3. If there is a clear green region, note the MCS and min_samples range that produces physically reasonable clusters (check the scatter plot).
4. Try both `eom` and `leaf` — leaf may resolve sub-states that eom merges.
5. Set `auto_mcs: false` with your chosen parameters for production runs.

### K-means

Centroid-based clustering with a fixed number of clusters. All points are assigned (no noise category).

```yaml
clustering:
  kmeans:
    n_clusters: 3
```

**Outputs:** same scatter, polar, and timeseries plots as HDBSCAN (prefixed `*_kmeans_*`).

### When to use which

- **HDBSCAN** when you don't know how many states exist, or when states have irregular shapes/sizes. The noise category is useful for identifying transition pathways. Best suited for systems with well-separated density peaks (e.g., Rheb with 3-4 distinct orientation states).
- **K-means** when you know the number of states (e.g., from prior KDE analysis or polar density plots) and want clean assignments with no noise. More appropriate when the density landscape has overlapping states that HDBSCAN might not resolve.

---

## HDBSCAN Explorer (`analyses/hdbscan_explorer.py`)

Standalone exploration module for HDBSCAN parameter selection. Can be called directly from Python without running the full pipeline — useful for working with angle data from external sources or for quickly iterating on parameter choices before committing to a pipeline run.

**Important:** The explorer's automatic parameter selection (peak DBCV) is a diagnostic aid, not a black-box optimizer. DBCV tends to favor tightly separated clusters and will often pick parameters that fragment a single broad density peak into micro-clusters. The real value of the explorer is the 2-D sweep heatmap and condensed tree — use those to understand your data's density structure, then choose parameters yourself.

### Standalone usage

```python
from membrane_analysis.analyses.hdbscan_explorer import run_exploration
import numpy as np

tilt = np.loadtxt("my_tilt_angles.txt")
rot  = np.loadtxt("my_rotation_angles.txt")

results = run_exploration(
    tilt, rot,
    outdir="hdbscan_output/",
    label="MySystem",
    method="eom",          # or "leaf"
    sweep_n_mcs=15,
    sweep_n_ms=8,
    mcs_min=50,
    ms_min=5,
    sweep_subsample=20000,
)

# Inspect the sweep heatmap and condensed tree, then decide parameters
# The auto-selected "best" is a starting point:
print(f"Auto-selected: MCS={results['best']['mcs']}, ms={results['best']['ms']}")
print(f"Clusters: {len(results['clusters'])}")
```

### Workflow

1. 2-D sweep of (min_cluster_size x min_samples) on a subsample, scoring each with DBCV.
2. Identify the auto-selected peak DBCV parameters (starting point only).
3. Run final HDBSCAN on all frames at those parameters.
4. Generate: 2-D heatmap, 1-D slice, condensed tree, polar density, scatter, time series.
5. **You decide:** inspect the outputs, adjust parameters if needed, re-run in fixed mode.

### Available functions

| Function | Purpose |
|----------|---------|
| `sweep_params()` | 2-D grid sweep over (MCS, min_samples), returns all results + best |
| `run_hdbscan()` | Run HDBSCAN at specific parameters on the full dataset |
| `plot_sweep_2d()` | DBCV heatmap over the parameter grid |
| `plot_sweep_1d()` | Noise fraction + DBCV vs MCS at the optimal min_samples |
| `plot_condensed_tree()` | Cluster persistence bar chart |
| `plot_polar_clustered()` | Polar density with cluster-centre overlays |
| `plot_scatter_clustered()` | 2D scatter colored by cluster |
| `plot_timeseries_clustered()` | Two-panel scatter time series colored by cluster |
| `run_exploration()` | High-level entry point that runs all of the above |

---

## Kinetics (`analyses/kinetics.py`)

Bayesian Hidden Markov Model analysis via pyemma. Constructs a coarse-grained kinetic model from orientation/distance features to extract state transition rates, mean first passage times, and flux networks.

### Data modes

**Spherical mode** — assembles features from the pipeline's `tilt_rotation` and `inter_residue_distance` caches:

```
(rotation, tilt, distance) → spherical_to_cartesian → (x, y, z)
```

The `radial_scale` parameter controls how much weight the distance coordinate gets relative to the angular coordinates.

```yaml
kinetics:
  enabled: true
  mode: spherical
  radial_scale: 1.0
```

Requires `tilt_rotation` and `inter_residue_distance` to have run first.

**Direct mode** — loads arbitrary feature files (one value per line) and stacks them as columns. No spherical transform is applied.

```yaml
kinetics:
  enabled: true
  mode: direct
  custom_features:
    - /path/to/feature1.txt
    - /path/to/feature2.txt
    - /path/to/feature3.txt
```

### Workflow

**Step 1: ITS scan** — set `lag: 0` to run the implied timescale scan without fitting the HMM:

```yaml
kinetics:
  enabled: true
  K: 4
  lag: 0
  max_lag: 5000
  dt_time: 0.1
```

Run the pipeline, inspect the ITS plot. The implied timescales should plateau at some lag time. Pick a lag value in the plateau region.

**Step 2: Full HMM** — set `lag` to your chosen value and rerun:

```yaml
kinetics:
  lag: 500    # chosen from ITS plot
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `spherical` | Feature assembly mode (`spherical` or `direct`) |
| `radial_scale` | `1.0` | Scaling factor for the radial (distance) coordinate in spherical mode |
| `micro_method` | `kmeans` | Microstate discretisation method (`kmeans` or `gmm`) |
| `K_micro` | `300` | Number of microstates for discretisation |
| `K` | `4` | Number of hidden (macro) states in the HMM |
| `lag` | `0` | HMM lag time in frames; 0 = ITS scan only |
| `max_lag` | `5000` | Maximum lag for the ITS scan (frames) |
| `dt_time` | `0.1` | Time per frame in nanoseconds |
| `nsamples` | `100` | Number of Bayesian HMM samples |
| `state_labels` | auto | Custom state labels, e.g. `{0: "OS1", 1: "OS2", ...}` |
| `mfpt_order` | identity | Custom ordering for MFPT and population plots, e.g. `[2, 1, 0, 3]` |

### Outputs (per system, when lag > 0)

| File | Description |
|------|-------------|
| `*_its.png` | Implied timescale scan (log scale) with lag time limit |
| `*_T_matrix.png` | Transition probability matrix heatmap |
| `*_mfpt.png` | Mean first passage time matrix (ns) |
| `*_populations.png` | Stationary distribution bar chart |
| `*_dwell_times.png` | Mean dwell time per state (ns), from diagonal of T |
| `*_ck_test.png` | Chapman-Kolmogorov test: model prediction vs data at lag multiples |
| `*_flux_network.png` | Directed flux network graph with edge weights |

### Interpreting results

**ITS plot**: look for timescale convergence (plateau). If timescales keep rising or cross the lag-time line, the model is not Markovian at that lag. Pick a lag in the plateau region.

**CK test**: model prediction (black solid) should match data estimation (blue dashed) for all state pairs. Systematic deviation indicates the model is not capturing the true dynamics at the chosen lag.

**Transition matrix**: diagonal elements should dominate (high self-transition probability). Off-diagonal elements show which state transitions are kinetically accessible.

**MFPT**: asymmetric MFPT indicates kinetic traps — if MFPT(A->B) >> MFPT(B->A), state A is kinetically trapped.

**Flux network**: edge thickness proportional to flux (stationary probability * transition probability). Red = forward, blue = reverse. Identifies the dominant kinetic pathways.

### State renumbering

States are automatically sorted by stationary probability (most populated = S0). If you know the physical meaning of each state, use `state_labels` and `mfpt_order` to map them to meaningful names:

```yaml
kinetics:
  state_labels:
    0: "OS3"    # S0 (most stable) maps to OS3
    1: "OS2"
    2: "OS1"
    3: "OS4"
  mfpt_order: [2, 1, 0, 3]   # plot order: OS1, OS2, OS3, OS4
```
