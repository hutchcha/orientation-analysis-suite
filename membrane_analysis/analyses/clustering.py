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

from membrane_analysis.core.io import (
    cached_compute, save_per_system, load_cache_metadata, get_time_bounds,
)
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_sim_length,
    is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import style_axes, save_figure
from membrane_analysis.analyses.tilt_rotation import polar_density_plot
from membrane_analysis.analyses.hdbscan_explorer import (
    _sph_to_cart, _cart_to_angles, _cluster_color, _COLORS,
    sweep_params, run_hdbscan as explorer_run_hdbscan,
    plot_sweep_2d, plot_sweep_1d,
    plot_condensed_tree, plot_polar_clustered,
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

    If ``auto_mcs`` is true, runs a 2-D (MCS x min_samples) DBCV sweep
    to pick the optimal parameters.  Otherwise uses configured values.

    Supports ``cluster_selection_method``: "eom" (default) or "leaf".
    """
    auto   = params.get("auto_mcs", False)
    method = params.get("cluster_selection_method", "eom")

    if auto:
        sweep_n_mcs = int(params.get("sweep_n_mcs", params.get("sweep_n", 15)))
        sweep_n_ms  = int(params.get("sweep_n_ms", 8))
        mcs_min     = int(params.get("mcs_min", 50))
        mcs_max     = params.get("mcs_max")
        ms_min      = int(params.get("ms_min", 5))
        ms_max      = params.get("ms_max")
        subsample   = int(params.get("sweep_subsample", 20_000))

        n_sub    = min(subsample, len(tilt))
        _mcs_max = int(mcs_max) if mcs_max is not None else max(mcs_min + 1, n_sub // 20)
        _ms_max  = int(ms_max) if ms_max is not None else _mcs_max

        mcs_range = np.unique(np.linspace(mcs_min, _mcs_max, sweep_n_mcs, dtype=int))
        ms_range  = np.unique(np.linspace(ms_min, _ms_max, sweep_n_ms, dtype=int))

        total = sum(1 for m in mcs_range for s in ms_range
                    if s <= m and m < n_sub // 2)
        print(f"    Auto sweep: {len(mcs_range)} MCS x {len(ms_range)} ms "
              f"= {total} combos (method={method}, subsample={n_sub})...")

        sweep_results, best = sweep_params(
            tilt, rot, mcs_range, ms_range,
            subsample=n_sub, method=method,
        )

        if best is None:
            print("    Sweep produced no results — falling back to defaults.")
            mcs = int(params.get("min_cluster_size", 200))
            ms  = int(params.get("min_samples") or max(5, mcs // 2))
            sweep_data = None
        else:
            mcs = best["mcs"]
            ms  = best["ms"]
            print(f"    Optimal: MCS={mcs}  ms={ms}  k={best['n_clusters']}  "
                  f"noise={best['noise_frac']:.1%}  DBCV={best['dbcv']:.4f}")
            sweep_data = {"results": sweep_results, "best": best}

        clusterer, labels, clusters = explorer_run_hdbscan(
            tilt, rot, mcs, min_samples=ms, method=method)

        try:
            ct_df = clusterer.condensed_tree_.to_pandas()
        except Exception:
            ct_df = None

        return {
            "clusters":           clusters,
            "labels":             labels,
            "auto_mcs":           True,
            "optimal_mcs":        mcs,
            "optimal_ms":         ms,
            "method":             method,
            "sweep":              sweep_data,
            "condensed_tree_df":  ct_df,
            "n_points":           len(labels),
        }

    else:
        # Fixed parameters
        try:
            import hdbscan
        except ImportError:
            print("    hdbscan not installed — skipping HDBSCAN.")
            return {
                "clusters": [], "labels": np.full(len(tilt), -1, dtype=int),
                "auto_mcs": False, "optimal_mcs": None, "optimal_ms": None,
                "method": method, "sweep": None, "condensed_tree_df": None,
                "n_points": len(tilt),
            }

        mcs = int(params.get("min_cluster_size", 200))
        ms  = params.get("min_samples") or max(5, mcs // 2)
        ms  = int(ms)

        X = _sph_to_cart(rot, tilt)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method=method,
            gen_min_span_tree=True,
        ).fit_predict(X)

        clusters = _cluster_summary(labels, X, len(tilt))
        noise_frac = (labels == -1).sum() / len(labels)
        print(f"    HDBSCAN: {len(clusters)} clusters, {noise_frac:.1%} noise "
              f"(MCS={mcs}, ms={ms}, method={method})")
        for c in clusters:
            print(f"      cluster {c['label']}: rot={c['rotation_deg_mean']:.1f}  "
                  f"tilt={c['tilt_deg_mean']:.1f}  pop={c['pop_fraction']:.3f}")

        return {
            "clusters": clusters, "labels": labels,
            "auto_mcs": False, "optimal_mcs": mcs, "optimal_ms": ms,
            "method": method, "sweep": None, "condensed_tree_df": None,
            "n_points": len(tilt),
        }


# ── K-means ──────────────────────────────────────────────────────────────────

def _run_kmeans(X, params):
    """Cluster feature matrix X with K-means.

    Returns dict with keys: clusters, labels.
    """
    from sklearn.cluster import KMeans

    k = params.get("n_clusters", 3)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    clusters = _cluster_summary(labels, X, len(X))
    print(f"    K-means (k={k}):")
    for c in clusters:
        print(f"      cluster {c['label']}: rot={c['rotation_deg_mean']:.1f}  "
              f"tilt={c['tilt_deg_mean']:.1f}  pop={c['pop_fraction']:.3f}")
    return {"clusters": clusters, "labels": labels}


# ── Top-level compute ────────────────────────────────────────────────────────

def compute(cfg, universes_or_stats_cfg=None):
    """Cluster orientation states.

    Supports two calling conventions:
      - ``compute(cfg, universes)`` — main pipeline, reads tilt_rotation cache
        with params from the main config's ``analyses.analysis.clustering`` block.
      - ``compute(cfg, stats_cfg)``  — stats pipeline, assembles features via
        core/features.py using the stats config.

    The stats pipeline path is used when universes_or_stats_cfg is a dict
    containing a ``feature_sets`` key.
    """
    # Detect which path to use
    is_stats = (isinstance(universes_or_stats_cfg, dict)
                and "feature_sets" in universes_or_stats_cfg)

    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "clustering.pkl")

    if is_stats:
        stats_cfg  = universes_or_stats_cfg
        from membrane_analysis.core.config import get_stats_params, get_feature_set
        from membrane_analysis.core.features import assemble_features

        params     = get_stats_params(stats_cfg, ANALYSIS_KEY)
        fs_name    = params.get("feature_set")
        fs_cfg     = get_feature_set(stats_cfg, fs_name)
        hdb_params = params.get("hdbscan", {})
        km_params  = params.get("kmeans", {})
        metadata   = {
            "analysis_key": ANALYSIS_KEY,
            "mode":         "stats",
            "feature_set":  {"name": fs_name, "cfg": dict(fs_cfg)},
            "hdbscan":      dict(hdb_params),
            "kmeans":       dict(km_params),
            "system_names": list(get_system_names(cfg)),
        }
    else:
        params     = get_analysis_params(cfg, ANALYSIS_KEY)
        hdb_params = params.get("hdbscan", {})
        km_params  = params.get("kmeans", {})
        metadata   = {
            "analysis_key": ANALYSIS_KEY,
            "mode":         "main",
            "params":       dict(params),
            "system_names": list(get_system_names(cfg)),
        }

    force = is_force_recompute(cfg) if not is_stats else True

    def _run():
        results = {}
        for name in get_system_names(cfg):
            if is_stats:
                # ── Stats pipeline: use feature assembly ─────────────────
                try:
                    X, columns, meta = assemble_features(
                        get_output_dir(cfg), fs_cfg, name)
                except (FileNotFoundError, KeyError) as e:
                    print(f"  [{name}] {e}")
                    continue

                print(f"  [{name}] Feature matrix: {X.shape} "
                      f"(transform={meta['transform']})")

                # For HDBSCAN we need tilt/rot angles. If the transform was
                # unit_sphere or spherical, we can extract from the raw cache.
                # The HDBSCAN functions operate on tilt/rot directly for the
                # sphere embedding. For non-angular features, we'd need a
                # different approach — for now, fall through to the old
                # tilt/rot path if angular features are present.
                tilt, rot, valid_mask, full_len = _load_tilt_rot(cfg, name)
            else:
                # ── Main pipeline: load tilt_rotation directly ───────────
                tilt, rot, valid_mask, full_len = _load_tilt_rot(cfg, name)

            if tilt is None:
                continue

            tilt_v = tilt[valid_mask]
            rot_v  = rot[valid_mask]

            if len(tilt_v) < 10:
                print(f"  [{name}] Too few valid frames for clustering.")
                continue

            print(f"  [{name}] HDBSCAN...")
            hdb_result = _run_hdbscan(tilt_v, rot_v, hdb_params)

            print(f"  [{name}] K-means...")
            X_sphere = _sph_to_cart(rot_v, tilt_v)
            km_result = _run_kmeans(X_sphere, km_params)

            # expand labels back to full-length arrays
            full_hdb = np.full(full_len, -1, dtype=int)
            full_km  = np.full(full_len, -1, dtype=int)
            full_hdb[valid_mask] = hdb_result["labels"]
            full_km[valid_mask]  = km_result["labels"]
            hdb_result["labels"] = full_hdb
            km_result["labels"]  = full_km

            results[name] = {
                "hdbscan": hdb_result,
                "kmeans":  km_result,
            }

        save_per_system(results, outdir, ANALYSIS_KEY, metadata=metadata)
        return results

    return cached_compute(cache, _run, force_recompute=force, metadata=metadata)


def _load_tilt_rot(cfg, name):
    """Load tilt/rotation arrays from the tilt_rotation cache.

    Returns (tilt, rot, valid_mask, full_length) or (None, None, None, None).
    """
    from membrane_analysis.core.io import load_cache_data
    tr_cache = os.path.join(get_output_dir(cfg), "tilt_rotation", "tilt_rotation.pkl")
    if not os.path.exists(tr_cache):
        print(f"  [{name}] tilt_rotation cache not found.")
        return None, None, None, None

    tr_data = load_cache_data(tr_cache)

    if name not in tr_data:
        print(f"  [{name}] Not found in tilt_rotation cache.")
        return None, None, None, None

    tilt = tr_data[name]["tilt"]
    rot  = tr_data[name]["rotation"]
    valid = np.isfinite(tilt) & np.isfinite(rot)

    return tilt, rot, valid, len(tilt)


# ── Top-level plot ───────────────────────────────────────────────────────────

def plot(cfg, results_or_stats_cfg=None, results=None):
    """Generate clustering plots.

    Supports two call signatures:
      - ``plot(cfg, results)``             — main pipeline
      - ``plot(cfg, stats_cfg, results)``  — stats pipeline
    """
    if results is None:
        results = results_or_stats_cfg
    _plot_impl(cfg, results)


def _plot_impl(cfg, results):
    """Generate clustering diagnostic plots for HDBSCAN and K-means per system."""
    from membrane_analysis.core.io import load_cache_data
    outdir  = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us  = get_sim_length(cfg)

    # load the raw angles for plotting (using tilt_rotation's cached metadata
    # for per-system time-axis bounds when available)
    tr_cache = os.path.join(get_output_dir(cfg), "tilt_rotation", "tilt_rotation.pkl")
    if not os.path.exists(tr_cache):
        print("  tilt_rotation cache missing — cannot generate clustering plots.")
        return
    tr_data = load_cache_data(tr_cache)
    tr_meta = load_cache_metadata(tr_cache)

    for name, data in results.items():
        if name not in tr_data:
            continue
        tilt = tr_data[name]["tilt"]
        rot  = tr_data[name]["rotation"]
        s_us, e_us = get_time_bounds(tr_meta, name, sim_us)
        time = np.linspace(s_us, e_us, len(tilt))

        # ── HDBSCAN plots ────────────────────────────────────────────────
        hdb      = data["hdbscan"]
        clusters = hdb["clusters"]
        labels   = hdb["labels"]

        # Auto MCS: sweep and condensed tree plots
        if hdb.get("auto_mcs") and hdb.get("sweep") is not None:
            sw = hdb["sweep"]
            plot_sweep_2d(
                sw["results"], sw["best"],
                os.path.join(outdir, f"{name}_hdbscan_sweep_2d.png"),
            )
            plot_sweep_1d(
                sw["results"], sw["best"],
                os.path.join(outdir, f"{name}_hdbscan_sweep_1d.png"),
            )

        if hdb.get("condensed_tree_df") is not None:
            # Reconstruct a minimal object for the tree plot
            _plot_condensed_tree_from_df(
                hdb["condensed_tree_df"], labels, hdb["n_points"],
                os.path.join(outdir, f"{name}_hdbscan_condensed_tree.png"),
                label=f"{name} HDBSCAN",
            )

        # Polar, scatter, time series (always)
        plot_polar_clustered(tilt, rot, clusters,
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

        plot_polar_clustered(tilt, rot, km_clust,
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
