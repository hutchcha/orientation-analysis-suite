"""Orientation-state clustering (HDBSCAN and K-means).

Loads the cached tilt/rotation data produced by the tilt_rotation module
and clusters the orientation states using two independent methods:

  HDBSCAN — density-based, no fixed cluster count required.
             Operates on 3-D Cartesian unit-vector embeddings of the
             (tilt, rotation) angles so that spherical geometry is respected.
             Frames not assigned to any cluster are labelled -1 (noise).

             When ``auto_mcs: true`` is set, runs a min_cluster_size sweep
             using DBCV scores to automatically pick the optimal MCS value.

  K-means — centroid-based, requires a fixed ``n_clusters``.
             Also uses the Cartesian sphere embedding so that the circular
             nature of rotation is handled naturally.

Both methods write per-frame labels and cluster-centre (tilt, rotation)
angles to a pickle cache.  Plots include a polar density contour with
cluster centres overlaid, a 2D scatter colored by cluster, and a
time-series coloured by cluster state.

Config fields (under analyses.analysis.clustering)
---------------------------------------------------
  enabled      : true/false
  hdbscan:
    min_cluster_size : int   (default 200; used when auto_mcs is false)
    min_samples      : null  (default: min_cluster_size)
    auto_mcs         : bool  (default false; run DBCV sweep to find optimal MCS)
    sweep_n          : int   (default 25; number of MCS values in the sweep)
    mcs_min          : int   (default 50; smallest MCS to sweep)
    mcs_max          : null  (default: 5% of subsample size)
    sweep_subsample  : int   (default 20000; frames used during sweep)
  kmeans:
    n_clusters       : int   (default 3)
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import atan2

from membrane_analysis.core.io import cached_compute
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_sim_length,
    is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import style_axes, save_figure
from membrane_analysis.analyses.tilt_rotation import polar_density_plot
from membrane_analysis.analyses.hdbscan_explorer import (
    _sph_to_cart, _cart_to_angles, _cluster_color, _COLORS,
    sweep_mcs, find_optimal_mcs, run_hdbscan as explorer_run_hdbscan,
    plot_sweep, plot_condensed_tree, plot_polar_clustered,
    plot_scatter_clustered, plot_timeseries_clustered,
)


ANALYSIS_KEY = "clustering"


# ── Cluster summary helper ───────────────────────────────────────────────────

def _cluster_summary(labels, X_cart, n_frames):
    """Return a list of cluster-info dicts sorted by population (largest first).

    Each dict contains: label, rotation_deg_mean, tilt_deg_mean,
    pop_fraction, size.
    """
    clusters = []
    for lab in sorted(set(labels) - {-1}):
        idx  = labels == lab
        mean = X_cart[idx].sum(axis=0)
        norm = np.linalg.norm(mean)
        mean_unit = mean / norm if norm > 1e-12 else mean
        rot_m, tilt_m = _cart_to_angles(mean_unit)
        clusters.append(dict(
            label=int(lab),
            rotation_deg_mean=float(rot_m),
            tilt_deg_mean=float(tilt_m),
            size=int(idx.sum()),
            pop_fraction=float(idx.sum() / n_frames),
        ))
    clusters.sort(key=lambda d: d["size"], reverse=True)
    return clusters


# ── HDBSCAN ──────────────────────────────────────────────────────────────────

def _run_hdbscan(tilt, rot, params):
    """Cluster (tilt, rot) with HDBSCAN on the unit sphere.

    If ``auto_mcs`` is true in params, runs a DBCV sweep to pick the optimal
    min_cluster_size.  Otherwise uses the configured value directly.

    Returns dict with keys: clusters, labels, auto_mcs, optimal_mcs,
    sweep (if auto), condensed_tree_df.
    """
    auto = params.get("auto_mcs", False)

    if auto:
        sweep_n    = int(params.get("sweep_n", 25))
        mcs_min    = int(params.get("mcs_min", 50))
        mcs_max    = params.get("mcs_max")
        subsample  = int(params.get("sweep_subsample", 20_000))
        ms_final   = params.get("min_samples")

        n_sub    = min(subsample, len(tilt))
        _mcs_max = int(mcs_max) if mcs_max is not None else max(mcs_min + 1, n_sub // 20)
        mcs_range = np.unique(np.linspace(mcs_min, _mcs_max, sweep_n, dtype=int))

        print(f"    Auto MCS sweep: {mcs_range[0]}-{mcs_range[-1]} "
              f"({len(mcs_range)} values, subsample={n_sub})...")
        mcs_arr, noise_fracs, dbcv_scores = sweep_mcs(
            tilt, rot, mcs_range, subsample=n_sub)

        if len(mcs_arr) == 0:
            print("    Sweep produced no results — falling back to default MCS.")
            mcs = int(params.get("min_cluster_size", 200))
            sweep_data = None
        else:
            mcs, best_dbcv = find_optimal_mcs(mcs_arr, dbcv_scores)
            print(f"    Optimal MCS = {mcs}  (DBCV = {best_dbcv:.4f})")
            sweep_data = {
                "mcs_values":  mcs_arr,
                "noise_fracs": noise_fracs,
                "dbcv_scores": dbcv_scores,
            }

        clusterer, labels, clusters = explorer_run_hdbscan(
            tilt, rot, mcs, min_samples=ms_final)

        # store condensed tree as DataFrame for plotting later
        try:
            ct_df = clusterer.condensed_tree_.to_pandas()
        except Exception:
            ct_df = None

        return {
            "clusters":           clusters,
            "labels":             labels,
            "auto_mcs":           True,
            "optimal_mcs":        mcs,
            "sweep":              sweep_data,
            "condensed_tree_df":  ct_df,
            "n_points":           len(labels),
        }

    else:
        # Fixed MCS — original behaviour
        try:
            import hdbscan
        except ImportError:
            print("    hdbscan not installed — skipping HDBSCAN.")
            return {
                "clusters": [], "labels": np.full(len(tilt), -1, dtype=int),
                "auto_mcs": False, "optimal_mcs": None,
                "sweep": None, "condensed_tree_df": None,
                "n_points": len(tilt),
            }

        mcs = int(params.get("min_cluster_size", 200))
        ms  = params.get("min_samples") or max(5, mcs // 2)

        X = _sph_to_cart(rot, tilt)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            gen_min_span_tree=True,
        ).fit_predict(X)

        clusters = _cluster_summary(labels, X, len(tilt))
        noise_frac = (labels == -1).sum() / len(labels)
        print(f"    HDBSCAN: {len(clusters)} clusters, {noise_frac:.1%} noise")
        for c in clusters:
            print(f"      cluster {c['label']}: rot={c['rotation_deg_mean']:.1f}  "
                  f"tilt={c['tilt_deg_mean']:.1f}  pop={c['pop_fraction']:.3f}")

        return {
            "clusters": clusters, "labels": labels,
            "auto_mcs": False, "optimal_mcs": mcs,
            "sweep": None, "condensed_tree_df": None,
            "n_points": len(tilt),
        }


# ── K-means ──────────────────────────────────────────────────────────────────

def _run_kmeans(tilt, rot, params):
    """Cluster (tilt, rot) with K-means on the unit sphere.

    Returns dict with keys: clusters, labels.
    """
    from sklearn.cluster import KMeans

    k = params.get("n_clusters", 3)
    X = _sph_to_cart(rot, tilt)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    clusters = _cluster_summary(labels, X, len(tilt))
    print(f"    K-means (k={k}):")
    for c in clusters:
        print(f"      cluster {c['label']}: rot={c['rotation_deg_mean']:.1f}  "
              f"tilt={c['tilt_deg_mean']:.1f}  pop={c['pop_fraction']:.3f}")
    return {"clusters": clusters, "labels": labels}


# ── Top-level compute ────────────────────────────────────────────────────────

def compute(cfg, universes):
    """Load tilt/rotation results and cluster orientation states.

    ``universes`` is not used; the function reads from the tilt_rotation cache.

    Returns
    -------
    dict : {name: {"hdbscan": {...}, "kmeans": {...}}}
    """
    outdir     = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache      = os.path.join(outdir, "clustering.pkl")
    force      = is_force_recompute(cfg)
    params     = get_analysis_params(cfg, ANALYSIS_KEY)

    # path to the tilt_rotation results
    tr_cache = os.path.join(get_output_dir(cfg), "tilt_rotation", "tilt_rotation.pkl")

    def _run():
        if not os.path.exists(tr_cache):
            raise FileNotFoundError(
                f"tilt_rotation cache not found at {tr_cache}. "
                "Run tilt_rotation first."
            )
        with open(tr_cache, "rb") as fh:
            tr_data = pickle.load(fh)

        hdb_params = params.get("hdbscan", {})
        km_params  = params.get("kmeans",  {})

        results = {}
        for name in get_system_names(cfg):
            if name not in tr_data:
                print(f"  [{name}] No tilt_rotation data found, skipping.")
                continue

            tilt = tr_data[name]["tilt"]
            rot  = tr_data[name]["rotation"]

            # drop NaN frames before clustering
            valid = np.isfinite(tilt) & np.isfinite(rot)
            if valid.sum() < 10:
                print(f"  [{name}] Too few valid frames for clustering.")
                continue

            tilt_v = tilt[valid]
            rot_v  = rot[valid]

            print(f"  [{name}] HDBSCAN...")
            hdb_result = _run_hdbscan(tilt_v, rot_v, hdb_params)

            print(f"  [{name}] K-means...")
            km_result = _run_kmeans(tilt_v, rot_v, km_params)

            # expand labels back to full-length arrays (NaN frames -> -1)
            full_hdb = np.full(len(tilt), -1, dtype=int)
            full_km  = np.full(len(tilt), -1, dtype=int)
            full_hdb[valid] = hdb_result["labels"]
            full_km[valid]  = km_result["labels"]
            hdb_result["labels"] = full_hdb
            km_result["labels"]  = full_km

            results[name] = {
                "hdbscan": hdb_result,
                "kmeans":  km_result,
            }
        return results

    return cached_compute(cache, _run, force_recompute=force)


# ── Top-level plot ───────────────────────────────────────────────────────────

def plot(cfg, results):
    """Generate clustering diagnostic plots for HDBSCAN and K-means per system."""
    outdir  = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us  = get_sim_length(cfg)

    # load the raw angles for plotting
    tr_cache = os.path.join(get_output_dir(cfg), "tilt_rotation", "tilt_rotation.pkl")
    if not os.path.exists(tr_cache):
        print("  tilt_rotation cache missing — cannot generate clustering plots.")
        return
    with open(tr_cache, "rb") as fh:
        tr_data = pickle.load(fh)

    for name, data in results.items():
        if name not in tr_data:
            continue
        tilt = tr_data[name]["tilt"]
        rot  = tr_data[name]["rotation"]
        time = np.linspace(0, sim_us, len(tilt))

        # ── HDBSCAN plots ────────────────────────────────────────────────
        hdb      = data["hdbscan"]
        clusters = hdb["clusters"]
        labels   = hdb["labels"]

        # Auto MCS: sweep and condensed tree plots
        if hdb.get("auto_mcs") and hdb.get("sweep") is not None:
            sw = hdb["sweep"]
            plot_sweep(
                sw["mcs_values"], sw["noise_fracs"], sw["dbcv_scores"],
                hdb["optimal_mcs"],
                os.path.join(outdir, f"{name}_hdbscan_sweep.png"),
            )

        if hdb.get("condensed_tree_df") is not None:
            # Reconstruct a minimal object for the tree plot
            _plot_condensed_tree_from_df(
                hdb["condensed_tree_df"], labels, hdb["n_points"],
                os.path.join(outdir, f"{name}_hdbscan_condensed_tree.png"),
                label=f"{name} HDBSCAN",
            )

        # Polar, scatter, time series (always)
        plot_polar_clustered(rot, tilt, clusters,
                             os.path.join(outdir, f"{name}_hdbscan_polar.png"),
                             label=f"{name} HDBSCAN")
        plot_scatter_clustered(tilt, rot, labels, clusters,
                               os.path.join(outdir, f"{name}_hdbscan_scatter.png"),
                               label=f"{name} HDBSCAN")
        plot_timeseries_clustered(tilt, rot, labels,
                                  os.path.join(outdir, f"{name}_hdbscan_timeseries.png"),
                                  label=f"{name} HDBSCAN")

        # ── K-means plots ────────────────────────────────────────────────
        km       = data["kmeans"]
        km_clust = km["clusters"]
        km_labs  = km["labels"]

        plot_polar_clustered(rot, tilt, km_clust,
                             os.path.join(outdir, f"{name}_kmeans_polar.png"),
                             label=f"{name} K-means")
        plot_scatter_clustered(tilt, rot, km_labs, km_clust,
                               os.path.join(outdir, f"{name}_kmeans_scatter.png"),
                               label=f"{name} K-means")
        plot_timeseries_clustered(tilt, rot, km_labs,
                                  os.path.join(outdir, f"{name}_kmeans_timeseries.png"),
                                  label=f"{name} K-means")


def _plot_condensed_tree_from_df(tree_df, labels, n_points, outpath, label=""):
    """Build a condensed tree persistence plot from a stored DataFrame.

    This avoids needing the full clusterer object at plot time.
    """
    try:
        selected = set(np.unique(labels[labels >= 0]))

        cluster_nodes = sorted(tree_df[tree_df["child"] >= n_points]["child"].unique())

        bars = []
        for i, node in enumerate(cluster_nodes):
            births = tree_df[tree_df["child"] == node]["lambda_val"]
            birth  = float(births.min()) if len(births) > 0 else 0.0
            deaths = tree_df[tree_df["parent"] == node]["lambda_val"]
            death  = float(deaths.max()) if len(deaths) > 0 else birth
            size   = int(tree_df[tree_df["parent"] == node]["child_size"].sum())
            bars.append((i, birth, death, size, i in selected))

        if not bars:
            print("  Condensed tree: no cluster nodes found.")
            return

        fig, ax = plt.subplots(figsize=(10, max(3, len(bars) * 0.5 + 1)),
                               constrained_layout=True)

        max_size = max(b[3] for b in bars)
        for i, birth, death, size, is_sel in bars:
            color = _cluster_color(i) if is_sel else "lightgrey"
            lw    = 2 + 10 * (size / max_size)
            ax.plot([birth, death], [i, i], color=color, linewidth=lw,
                    solid_capstyle="butt")
            pct = 100 * size / n_points
            sel_tag = " *" if is_sel else ""
            ax.text(death, i, f"  cluster {i} ({pct:.1f}%){sel_tag}",
                    va="center", fontsize=10, color=color if is_sel else "grey")

        style_axes(ax, xlabel="lambda (1 / distance)", ylabel="Cluster",
                   title=label if label else "Condensed tree")
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([f"{i}" for i in range(len(bars))])
        ax.invert_yaxis()

        save_figure(fig, outpath)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  Condensed tree plot failed: {exc}")
