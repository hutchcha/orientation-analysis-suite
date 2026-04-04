"""Orientation-state clustering (HDBSCAN and K-means).

Loads the cached tilt/rotation data produced by the tilt_rotation module
and clusters the orientation states using two independent methods:

  HDBSCAN — density-based, no fixed cluster count required.
             Operates on 3-D Cartesian unit-vector embeddings of the
             (tilt, rotation) angles so that spherical geometry is respected.
             Frames not assigned to any cluster are labelled −1 (noise).

  K-means — centroid-based, requires a fixed ``n_clusters``.
             Also uses the Cartesian sphere embedding so that the circular
             nature of rotation is handled naturally.

Both methods write per-frame labels and cluster-centre (tilt, rotation)
angles to a pickle cache.  Plots include a polar density contour with
cluster centres overlaid and a time-series coloured by cluster state.

Config fields (under analyses.analysis.clustering)
---------------------------------------------------
  enabled      : true/false
  hdbscan:
    min_cluster_size : int   (default 200)
    min_samples      : null  (default: min_cluster_size // 2)
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


ANALYSIS_KEY = "clustering"

# Cluster colours for up to 10 labelled states; noise (−1) is always grey.
_CLUSTER_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261",
    "#6A0572", "#1D3557", "#52B788", "#D62828", "#023E8A",
]


# ── Spherical geometry helpers ────────────────────────────────────────────────

def _sph_to_cart(rot_deg, tilt_deg):
    """(rotation φ, tilt θ) in degrees → (N, 3) Cartesian unit vectors."""
    phi   = np.deg2rad(rot_deg)
    theta = np.deg2rad(tilt_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _cart_to_angles(v):
    """Unit vector → (rotation_deg, tilt_deg)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.rad2deg(atan2(y, x)), np.rad2deg(np.arccos(np.clip(z, -1.0, 1.0)))


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


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def _run_hdbscan(tilt, rot, params):
    """Cluster (tilt, rot) with HDBSCAN on the unit sphere.

    Returns (clusters_list, labels_array).
    """
    try:
        import hdbscan
    except ImportError:
        print("    hdbscan not installed — skipping HDBSCAN.")
        return [], np.full(len(tilt), -1, dtype=int)

    mcs = params.get("min_cluster_size", 200)
    ms  = params.get("min_samples") or max(5, mcs // 2)

    X = _sph_to_cart(rot, tilt)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms
    ).fit_predict(X)

    clusters = _cluster_summary(labels, X, len(tilt))
    noise_frac = (labels == -1).sum() / len(labels)
    print(f"    HDBSCAN: {len(clusters)} clusters, {noise_frac:.1%} noise")
    for c in clusters:
        print(f"      OS{c['label']}: rot={c['rotation_deg_mean']:.1f}°  "
              f"tilt={c['tilt_deg_mean']:.1f}°  pop={c['pop_fraction']:.3f}")
    return clusters, labels


# ── K-means ───────────────────────────────────────────────────────────────────

def _run_kmeans(tilt, rot, params):
    """Cluster (tilt, rot) with K-means on the unit sphere.

    Returns (clusters_list, labels_array).
    """
    from sklearn.cluster import KMeans

    k = params.get("n_clusters", 3)
    X = _sph_to_cart(rot, tilt)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    # Normalise centroids onto the sphere
    centers_cart = km.cluster_centers_
    centers_cart /= np.linalg.norm(centers_cart, axis=1, keepdims=True)

    clusters = _cluster_summary(labels, X, len(tilt))
    print(f"    K-means (k={k}):")
    for c in clusters:
        print(f"      OS{c['label']}: rot={c['rotation_deg_mean']:.1f}°  "
              f"tilt={c['tilt_deg_mean']:.1f}°  pop={c['pop_fraction']:.3f}")
    return clusters, labels


# ── Top-level compute ─────────────────────────────────────────────────────────

def compute(cfg, universes):
    """Load tilt/rotation results and cluster orientation states.

    ``universes`` is not used; the function reads from the tilt_rotation cache.

    Returns
    -------
    dict : {name: {"hdbscan": {...}, "kmeans": {...}}}
          Each sub-dict contains "clusters" (list of dicts) and "labels" (ndarray).
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
            hdb_clusters, hdb_labels_v = _run_hdbscan(tilt_v, rot_v, hdb_params)

            print(f"  [{name}] K-means...")
            km_clusters, km_labels_v = _run_kmeans(tilt_v, rot_v, km_params)

            # expand back to full-length arrays (NaN frames → −1)
            full_hdb = np.full(len(tilt), -1, dtype=int)
            full_km  = np.full(len(tilt), -1, dtype=int)
            full_hdb[valid] = hdb_labels_v
            full_km[valid]  = km_labels_v

            results[name] = {
                "hdbscan": {"clusters": hdb_clusters, "labels": full_hdb},
                "kmeans":  {"clusters": km_clusters,  "labels": full_km},
            }
        return results

    return cached_compute(cache, _run, force_recompute=force)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _cluster_color(label):
    if label < 0:
        return "lightgrey"
    return _CLUSTER_COLORS[label % len(_CLUSTER_COLORS)]


def _plot_polar_with_clusters(rot, tilt, clusters, ax, title=""):
    """Polar density plot with cluster-centre markers."""
    polar_density_plot(rot, tilt, ax)
    ax.set_title(title, fontsize=14, pad=20)

    for c in clusters:
        phi = np.deg2rad(c["rotation_deg_mean"])
        r   = np.deg2rad(c["tilt_deg_mean"])
        color = _cluster_color(c["label"])
        ax.plot(phi, r, marker="*", ms=14, color=color,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=5, label=f"OS{c['label']} ({c['pop_fraction']:.1%})")
    if clusters:
        ax.legend(loc="lower right", fontsize=9, frameon=True,
                  framealpha=0.7, bbox_to_anchor=(1.25, -0.05))


def _plot_timeseries_by_cluster(time, tilt, labels, ax_tilt, ax_rot, rot=None):
    """Scatter time-series of tilt (and optionally rotation) coloured by cluster."""
    unique = sorted(set(labels))
    for lab in unique:
        idx = labels == lab
        c   = _cluster_color(lab)
        lbl = f"OS{lab}" if lab >= 0 else "noise"
        ax_tilt.scatter(time[idx], tilt[idx], s=1, c=c, label=lbl, alpha=0.6, rasterized=True)
        if rot is not None and ax_rot is not None:
            ax_rot.scatter(time[idx], rot[idx],  s=1, c=c,           alpha=0.6, rasterized=True)


# ── Top-level plot ────────────────────────────────────────────────────────────

def plot(cfg, results):
    """Generate polar + time-series plots for HDBSCAN and K-means per system."""
    outdir  = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us  = get_sim_length(cfg)

    # also load the raw angles for plotting
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

        for method in ("hdbscan", "kmeans"):
            mdata    = data[method]
            clusters = mdata["clusters"]
            labels   = mdata["labels"]
            tag      = f"{name}_{method}"

            # ── polar density + centres ────────────────────────────────────
            fig, ax = plt.subplots(figsize=(5, 5),
                                   subplot_kw={"projection": "polar"},
                                   constrained_layout=True)
            _plot_polar_with_clusters(rot, tilt, clusters, ax,
                                      title=f"{name} — {method.upper()}")
            save_figure(fig, os.path.join(outdir, f"{tag}_polar.png"))

            # ── cluster-coloured time series ───────────────────────────────
            fig, (ax_rot, ax_tilt) = plt.subplots(2, 1, figsize=(10, 6),
                                                   constrained_layout=True)
            _plot_timeseries_by_cluster(time, tilt, labels, ax_tilt, ax_rot, rot=rot)

            ax_rot.set_ylabel("Rotation (°)", fontsize=14)
            style_axes(ax_rot, title=f"{name} — {method.upper()}")
            ax_rot.legend(markerscale=8, fontsize=9, frameon=False,
                          loc="upper right")

            ax_tilt.set_ylabel("Tilt (°)", fontsize=14)
            ax_tilt.set_xlabel("Time (μs)", fontsize=14)
            style_axes(ax_tilt)

            save_figure(fig, os.path.join(outdir, f"{tag}_timeseries.png"))
