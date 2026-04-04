"""Gaussian Mixture Model state discretisation.

Fits GMMs over a range of n_components and selects the best model via
BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion).
Produces per-frame state labels, cluster centres, and diagnostic plots.

The GMM operates on feature matrices assembled by core/features.py, so it
can work with any combination of computed observables.

Config (in stats.yaml under "gmm")
------------------------------------
  feature_set      : str   — name of a feature_set defined in stats.yaml
  n_components     : list of int — values to sweep (e.g. [2, 3, 4, 5, 6])
  covariance_type  : str   — "full", "tied", "diag", or "spherical" (default "full")
  random_state     : int   — random seed (default 42)
  best_criterion   : str   — "bic" or "aic" (default "bic")

Outputs per system
------------------
  *_bic_aic.png         — BIC/AIC curve with optimal n_components marked
  *_scatter.png         — feature scatter colored by GMM state
  *_polar.png           — polar density with state centres (if angular features)
  *_timeseries.png      — per-frame state assignment colored scatter
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from membrane_analysis.core.io import cached_compute, save_per_system
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_feature_set, get_stats_params,
)
from membrane_analysis.core.features import assemble_features, get_feature_type, _parse_feature_key
from membrane_analysis.core.plotting import style_axes, save_figure
from membrane_analysis.analyses.hdbscan_explorer import (
    _cluster_color, _cart_to_angles,
    plot_polar_clustered, plot_scatter_clustered, plot_timeseries_clustered,
)


ANALYSIS_KEY = "gmm"


# ── BIC/AIC sweep ────────────────────────────────────────────────────────────

def _sweep_components(X, n_range, cov_type="full", random_state=42):
    """Fit GMMs over a range of n_components and return BIC/AIC scores.

    Returns
    -------
    results : list of dicts with keys: n, bic, aic, model
    """
    results = []
    for n in n_range:
        gmm = GaussianMixture(
            n_components=n, covariance_type=cov_type,
            random_state=random_state, reg_covar=1e-6,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        results.append({"n": n, "bic": bic, "aic": aic, "model": gmm})
        print(f"    n={n:3d}  BIC={bic:.0f}  AIC={aic:.0f}")
    return results


def _select_best(sweep_results, criterion="bic"):
    """Pick the model with the lowest BIC or AIC."""
    return min(sweep_results, key=lambda r: r[criterion])


# ── Cluster summary ──────────────────────────────────────────────────────────

def _gmm_cluster_summary(labels, X, n_frames):
    """Build cluster info dicts similar to HDBSCAN format."""
    clusters = []
    for lab in sorted(set(labels)):
        idx  = labels == lab
        size = int(idx.sum())
        mean = X[idx].mean(axis=0)
        clusters.append(dict(
            label=int(lab),
            center=mean.tolist(),
            size=size,
            pop_fraction=float(size / n_frames),
        ))
    clusters.sort(key=lambda d: d["size"], reverse=True)
    return clusters


def _add_angular_centres(clusters, feature_keys):
    """If features are angular (unit_sphere/spherical), compute rot/tilt centres."""
    # Check if the first 3 columns are from a sphere transform
    has_angular = any(
        get_feature_type(*_parse_feature_key(k)) == "angular"
        for k in feature_keys
    )
    if not has_angular:
        return

    for c in clusters:
        centre = np.array(c["center"])
        if len(centre) >= 3:
            v = centre[:3]
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            rot_m, tilt_m = _cart_to_angles(v)
            c["rotation_deg_mean"] = float(rot_m)
            c["tilt_deg_mean"] = float(tilt_m)


# ── Top-level compute ────────────────────────────────────────────────────────

def compute(cfg, stats_cfg):
    """Run GMM analysis for all systems using the stats config.

    Parameters
    ----------
    cfg       : dict — main pipeline config (for system names, output dir)
    stats_cfg : dict — stats YAML config

    Returns
    -------
    dict : {name: {"labels": ndarray, "clusters": list, "sweep": list,
                    "best_n": int, "meta": dict}}
    """
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "gmm.pkl")

    gmm_cfg   = get_stats_params(stats_cfg, ANALYSIS_KEY)
    fs_name   = gmm_cfg.get("feature_set")
    if not fs_name:
        print("  GMM: no feature_set specified, skipping.")
        return {}

    fs_cfg    = get_feature_set(stats_cfg, fs_name)
    n_range   = gmm_cfg.get("n_components", [2, 3, 4, 5, 6])
    cov_type  = gmm_cfg.get("covariance_type", "full")
    seed      = int(gmm_cfg.get("random_state", 42))
    criterion = gmm_cfg.get("best_criterion", "bic")

    def _run():
        results = {}
        for name in get_system_names(cfg):
            print(f"  [{name}] Assembling features...")
            try:
                X, columns, meta = assemble_features(
                    get_output_dir(cfg), fs_cfg, name)
            except (FileNotFoundError, KeyError) as e:
                print(f"  [{name}] {e}")
                continue

            print(f"  [{name}] Feature matrix: {X.shape}")
            print(f"  [{name}] GMM sweep (n={n_range}, cov={cov_type})...")

            sweep = _sweep_components(X, n_range, cov_type, seed)
            best  = _select_best(sweep, criterion)

            print(f"  [{name}] Best: n={best['n']} ({criterion.upper()}={best[criterion]:.0f})")

            labels   = best["model"].predict(X)
            clusters = _gmm_cluster_summary(labels, X, len(X))
            _add_angular_centres(clusters, fs_cfg["features"])

            # Expand labels back to full length using the valid mask
            full_labels = np.full(meta["min_len"], -1, dtype=int)
            full_labels[meta["valid_mask"]] = labels

            # Store sweep without model objects (not picklable reliably)
            sweep_data = [{"n": r["n"], "bic": r["bic"], "aic": r["aic"]}
                          for r in sweep]

            entry = {
                "labels":       full_labels,
                "clusters":     clusters,
                "sweep":        sweep_data,
                "best_n":       best["n"],
                "best_bic":     best["bic"],
                "best_aic":     best["aic"],
                "meta":         meta,
                "feature_keys": fs_cfg["features"],
            }
            results[name] = entry

        # Save per-system subfolders
        save_per_system(results, outdir, ANALYSIS_KEY)
        return results

    return cached_compute(cache, _run, force_recompute=True)


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_bic_aic(sweep, best_n, criterion, outpath):
    """BIC/AIC curve with optimal marked."""
    ns   = [r["n"] for r in sweep]
    bics = [r["bic"] for r in sweep]
    aics = [r["aic"] for r in sweep]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(ns, bics, "o-", color="#457B9D", lw=2, ms=6, label="BIC")
    ax.plot(ns, aics, "s--", color="#E63946", lw=2, ms=6, label="AIC")
    ax.axvline(best_n, color="black", ls="--", lw=1.5,
               label=f"Best ({criterion.upper()}) = {best_n}")

    style_axes(ax, xlabel="Number of components",
               ylabel="Information criterion score")
    ax.legend(fontsize=11, frameon=False)
    ax.set_xticks(ns)
    save_figure(fig, outpath)


def plot(cfg, stats_cfg, results):
    """Generate GMM diagnostic plots for all systems."""
    outdir    = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    gmm_cfg   = get_stats_params(stats_cfg, ANALYSIS_KEY)
    criterion = gmm_cfg.get("best_criterion", "bic")

    for name, data in results.items():
        sys_dir = os.path.join(outdir, name)
        os.makedirs(sys_dir, exist_ok=True)

        # Load the raw angles for angular plots
        meta = data["meta"]
        feature_keys = data["feature_keys"]
        labels   = data["labels"]
        clusters = data["clusters"]
        valid    = meta["valid_mask"]

        # BIC/AIC curve
        _plot_bic_aic(data["sweep"], data["best_n"], criterion,
                      os.path.join(sys_dir, "bic_aic.png"))

        # If angular features, produce polar/scatter/timeseries
        has_angular = any(
            get_feature_type(*_parse_feature_key(k)) == "angular"
            for k in feature_keys
        )

        if has_angular and all("rotation_deg_mean" in c for c in clusters):
            # Load raw tilt/rotation for plots
            import pickle
            tr_cache = os.path.join(get_output_dir(cfg), "tilt_rotation", "tilt_rotation.pkl")
            if os.path.exists(tr_cache):
                with open(tr_cache, "rb") as f:
                    tr_data = pickle.load(f)
                if name in tr_data:
                    tilt = tr_data[name]["tilt"]
                    rot  = tr_data[name]["rotation"]

                    plot_polar_clustered(
                        tilt, rot, clusters,
                        os.path.join(sys_dir, "polar.png"),
                        label=f"{name} GMM (k={data['best_n']})")
                    plot_scatter_clustered(
                        tilt, rot, labels, clusters,
                        os.path.join(sys_dir, "scatter.png"),
                        label=f"{name} GMM (k={data['best_n']})")
                    plot_timeseries_clustered(
                        tilt, rot, labels,
                        os.path.join(sys_dir, "timeseries.png"),
                        label=f"{name} GMM (k={data['best_n']})")
