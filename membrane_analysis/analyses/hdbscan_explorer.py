"""HDBSCAN exploration and validation for orientation-state clustering.

Use this module exploratorily — before committing to hyperparameters in the
main clustering module.  It sweeps min_cluster_size to find the optimal value
via DBCV score and noise fraction, generates the HDBSCAN condensed tree for
visual hyperparameter inspection, and produces polar/time-series plots of the
final clustering at the chosen parameters.

Typical workflow
----------------
1. Load tilt/rotation arrays (from .txt files or the tilt_rotation pickle).
2. Call ``run_exploration()`` to generate all diagnostic outputs in one shot.
3. Inspect the sweep plot and condensed tree to choose min_cluster_size.
4. Put that value in the main config under ``analyses.analysis.clustering``.

Standalone usage::

    from membrane_analysis.analyses.hdbscan_explorer import run_exploration
    import numpy as np

    tilt = np.loadtxt("tilt.txt")
    rot  = np.loadtxt("rotation.txt")
    run_exploration(tilt, rot, outdir="testing/hdbscan_explorer/RhebGDP",
                    label="Rheb-GDP")

Parameters accepted by ``run_exploration``
-------------------------------------------
tilt, rot       : 1-D ndarrays of angles in degrees
outdir          : directory for all output figures
label           : string prefix / title used in plots and filenames
sweep_n         : number of MCS values to sweep (default 25)
mcs_min/mcs_max : range of min_cluster_size values for the sweep
                  (defaults: 50, 5 % of subsample size)
sweep_subsample : number of frames used for the sweep (default 20 000)
                  Full data is always used for the final clustering.
min_samples     : passed to HDBSCAN for the final run; None = MCS value
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from membrane_analysis.core.plotting import style_axes, save_figure
from membrane_analysis.analyses.tilt_rotation import polar_density_plot


# ── Sphere geometry (duplicated from clustering.py to keep module standalone) ─

def _sph_to_cart(rot_deg, tilt_deg):
    """(rotation, tilt) in degrees → (N, 3) Cartesian unit vectors."""
    phi   = np.deg2rad(rot_deg)
    theta = np.deg2rad(tilt_deg)
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _cart_to_angles(v):
    """Unit vector → (rotation_deg, tilt_deg)."""
    from math import atan2
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.rad2deg(atan2(y, x)), np.rad2deg(np.arccos(np.clip(z, -1.0, 1.0)))


# Colours for up to 10 clusters; noise is always grey.
_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261",
    "#6A0572", "#1D3557", "#52B788", "#D62828", "#023E8A",
]

def _cluster_color(label):
    return "lightgrey" if label < 0 else _COLORS[label % len(_COLORS)]


# ── MCS sweep ────────────────────────────────────────────────────────────────

def sweep_mcs(tilt, rot, mcs_range, subsample=20_000):
    """Sweep min_cluster_size on a subsample and record noise fraction + DBCV.

    The sweep uses ``min_samples = min_cluster_size`` (HDBSCAN default).

    Parameters
    ----------
    tilt, rot   : full-length angle arrays (degrees)
    mcs_range   : sequence of int min_cluster_size values to test
    subsample   : number of frames to use for the sweep (for speed)

    Returns
    -------
    mcs_arr     : (M,) int array of MCS values actually tested
    noise_fracs : (M,) float array — fraction of frames labelled noise
    dbcv_scores : (M,) float array — DBCV (relative_validity_) score
    """
    try:
        import hdbscan as hdb_lib
    except ImportError:
        raise ImportError("hdbscan is not installed. Run: pip install hdbscan")

    X_full = _sph_to_cart(rot, tilt)

    if subsample and len(X_full) > subsample:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(X_full), size=subsample, replace=False)
        X    = X_full[idx]
    else:
        X = X_full

    n = len(X)
    mcs_values, noise_fracs, dbcv_scores = [], [], []

    for mcs in mcs_range:
        mcs = int(mcs)
        if mcs >= n // 2:
            break
        try:
            clust = hdb_lib.HDBSCAN(min_cluster_size=mcs,
                                    gen_min_span_tree=True).fit(X)
            noise_fracs.append((clust.labels_ == -1).mean())
            dbcv_scores.append(float(clust.relative_validity_))
            mcs_values.append(mcs)
            n_clust = len(set(clust.labels_) - {-1})
            print(f"    MCS={mcs:5d}  clusters={n_clust:2d}  "
                  f"noise={noise_fracs[-1]:.1%}  DBCV={dbcv_scores[-1]:.4f}")
        except Exception as exc:
            print(f"    MCS={mcs:5d}  ERROR: {exc}")

    return np.array(mcs_values), np.array(noise_fracs), np.array(dbcv_scores)


def find_optimal_mcs(mcs_values, dbcv_scores):
    """Return the MCS value that maximises the DBCV score."""
    best_idx = int(np.argmax(dbcv_scores))
    return int(mcs_values[best_idx]), float(dbcv_scores[best_idx])


# ── Final HDBSCAN run ─────────────────────────────────────────────────────────

def run_hdbscan(tilt, rot, mcs, min_samples=None):
    """Run HDBSCAN on the full dataset at the chosen MCS.

    Returns
    -------
    clusterer : fitted HDBSCAN object (has .condensed_tree_, .labels_, etc.)
    labels    : (N,) int array including -1 for noise
    clusters  : list of dicts sorted by population (label, tilt/rot means, pop)
    """
    try:
        import hdbscan as hdb_lib
    except ImportError:
        raise ImportError("hdbscan is not installed. Run: pip install hdbscan")

    X = _sph_to_cart(rot, tilt)
    ms = min_samples if min_samples is not None else mcs

    clusterer = hdb_lib.HDBSCAN(min_cluster_size=int(mcs),
                                 min_samples=int(ms),
                                 gen_min_span_tree=True).fit(X)
    labels = clusterer.labels_

    # Summarise clusters
    clusters = []
    for lab in sorted(set(labels) - {-1}):
        idx  = labels == lab
        mean = X[idx].sum(axis=0)
        norm = np.linalg.norm(mean)
        unit = mean / norm if norm > 1e-12 else mean
        rot_m, tilt_m = _cart_to_angles(unit)
        clusters.append(dict(
            label=int(lab),
            rotation_deg_mean=float(rot_m),
            tilt_deg_mean=float(tilt_m),
            size=int(idx.sum()),
            pop_fraction=float(idx.sum() / len(labels)),
        ))
    clusters.sort(key=lambda d: d["size"], reverse=True)

    n_clust     = len(clusters)
    noise_frac  = (labels == -1).mean()
    dbcv        = float(clusterer.relative_validity_)
    print(f"  Final HDBSCAN: MCS={mcs}  clusters={n_clust}  "
          f"noise={noise_frac:.1%}  DBCV={dbcv:.4f}")
    for c in clusters:
        print(f"    cluster {c['label']}: "
              f"rot={c['rotation_deg_mean']:.1f}  "
              f"tilt={c['tilt_deg_mean']:.1f}  "
              f"pop={c['pop_fraction']:.3f}")

    return clusterer, labels, clusters


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_sweep(mcs_values, noise_fracs, dbcv_scores, optimal_mcs, outpath):
    """Dual y-axis plot: noise fraction (left, blue) and DBCV (right, red)."""
    fig, ax1 = plt.subplots(figsize=(8, 4), constrained_layout=True)

    color_noise = "#457B9D"
    color_dbcv  = "#E63946"

    ax1.plot(mcs_values, noise_fracs, color=color_noise,
             linewidth=2, marker="o", ms=4, label="Noise fraction")
    ax1.set_xlabel("min_cluster_size", fontsize=14)
    ax1.set_ylabel("Noise fraction", fontsize=14, color=color_noise)
    ax1.tick_params(axis="y", labelcolor=color_noise, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(mcs_values, dbcv_scores, color=color_dbcv,
             linewidth=2, marker="s", ms=4, label="DBCV")
    ax2.set_ylabel("DBCV score (relative_validity_)", fontsize=14, color=color_dbcv)
    ax2.tick_params(axis="y", labelcolor=color_dbcv, labelsize=12)

    # mark optimal
    ax2.axvline(optimal_mcs, color="black", linestyle="--", linewidth=1.5,
                label=f"Optimal MCS = {optimal_mcs}")

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, frameon=False)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    save_figure(fig, outpath)


def plot_condensed_tree(clusterer, outpath, label=""):
    """Plot the HDBSCAN condensed tree as cluster persistence bars.

    Each bar spans from the cluster's birth lambda to its death lambda,
    with width proportional to cluster size.  Selected clusters are coloured;
    unselected ones are grey.  This is a manual implementation that avoids
    the hdbscan / matplotlib version incompatibility in the built-in plot.
    """
    try:
        ct     = clusterer.condensed_tree_
        tree   = ct.to_pandas()  # parent, child, lambda_val, child_size
        labels = clusterer.labels_
        n_pts  = len(labels)

        # A "cluster node" has child > max(parent seen in points), i.e. child >= n_pts
        cluster_nodes = sorted(tree[tree["child"] >= n_pts]["child"].unique())
        selected      = set(np.unique(labels[labels >= 0]))  # selected cluster ids

        # For each cluster node, compute birth/death lambda and total size
        bars = []
        for i, node in enumerate(cluster_nodes):
            # birth = lambda when this node first appears as a child
            births = tree[tree["child"] == node]["lambda_val"]
            birth  = float(births.min()) if len(births) > 0 else 0.0
            # death = max lambda of points/sub-clusters falling out of this node
            deaths = tree[tree["parent"] == node]["lambda_val"]
            death  = float(deaths.max()) if len(deaths) > 0 else birth
            size   = int(tree[tree["parent"] == node]["child_size"].sum())
            bars.append((i, birth, death, size, i in selected))

        if not bars:
            print("  Condensed tree: no cluster nodes found.")
            return

        fig, ax = plt.subplots(figsize=(10, max(3, len(bars) * 0.5 + 1)),
                               constrained_layout=True)

        max_size = max(b[3] for b in bars)
        for i, birth, death, size, is_sel in bars:
            color = _cluster_color(i) if is_sel else "lightgrey"
            lw    = 2 + 10 * (size / max_size)  # bar thickness ~ cluster size
            ax.plot([birth, death], [i, i], color=color, linewidth=lw, solid_capstyle="butt")
            pct = 100 * size / n_pts
            sel_tag = " *" if is_sel else ""
            ax.text(death, i, f"  cluster {i} ({pct:.1f}%){sel_tag}",
                    va="center", fontsize=10, color=color if is_sel else "grey")

        style_axes(ax, xlabel="lambda (1 / distance)", ylabel="Cluster",
                   title=f"Condensed tree — {label}" if label else "Condensed tree")
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([f"{i}" for i in range(len(bars))])
        ax.invert_yaxis()

        save_figure(fig, outpath)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  Condensed tree plot failed: {exc}")


def plot_polar_clustered(tilt, rot, clusters, outpath, label=""):
    """Polar density with cluster-centre star markers."""
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw={"projection": "polar"},
                           constrained_layout=True)
    polar_density_plot(rot, tilt, ax)
    if label:
        ax.set_title(label, fontsize=14, pad=20)

    for c in clusters:
        phi   = np.deg2rad(c["rotation_deg_mean"])
        r     = np.deg2rad(c["tilt_deg_mean"])
        color = _cluster_color(c["label"])
        ax.plot(phi, r, marker="*", ms=16, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                label=f"cluster {c['label']} ({c['pop_fraction']:.1%})")
    if clusters:
        ax.legend(loc="lower right", fontsize=9, frameon=True,
                  framealpha=0.7, bbox_to_anchor=(1.3, -0.05))

    save_figure(fig, outpath)


def plot_scatter_clustered(tilt, rot, labels, clusters, outpath, label=""):
    """2D scatter of rotation vs tilt with points coloured by cluster assignment."""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # noise first (background)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(rot[noise_mask], tilt[noise_mask], s=1, c="lightgrey",
                   alpha=0.3, rasterized=True, label="noise", zorder=1)

    # clusters on top
    for c in clusters:
        lab  = c["label"]
        mask = labels == lab
        color = _cluster_color(lab)
        ax.scatter(rot[mask], tilt[mask], s=2, c=color, alpha=0.5,
                   rasterized=True, label=f"cluster {lab} ({c['pop_fraction']:.1%})",
                   zorder=2)
        # mark centre
        ax.plot(c["rotation_deg_mean"], c["tilt_deg_mean"],
                marker="*", ms=16, color=color,
                markeredgecolor="black", markeredgewidth=0.8, zorder=3)

    style_axes(ax, title=label if label else "HDBSCAN clusters",
               xlabel="Rotation (°)", ylabel="Tilt (°)")
    ax.set_xlim(-185, 185)
    ax.set_ylim(0, max(tilt.max() + 5, 135))
    ax.legend(fontsize=10, frameon=True, framealpha=0.8, markerscale=5,
              loc="upper right")

    save_figure(fig, outpath)


def plot_timeseries_clustered(tilt, rot, labels, outpath, label=""):
    """Two-panel scatter time series (rotation top, tilt bottom) coloured by cluster."""
    fig, (ax_rot, ax_tilt) = plt.subplots(2, 1, figsize=(12, 6),
                                           constrained_layout=True, sharex=True)
    time = np.arange(len(tilt))

    for lab in sorted(set(labels)):
        idx  = labels == lab
        c    = _cluster_color(lab)
        name = f"cluster {lab}" if lab >= 0 else "noise"
        ax_rot.scatter(time[idx],  rot[idx],  s=0.5, c=c, label=name,
                       alpha=0.5, rasterized=True)
        ax_tilt.scatter(time[idx], tilt[idx], s=0.5, c=c,
                        alpha=0.5, rasterized=True)

    style_axes(ax_rot,  title=label, ylabel="Rotation (°)")
    style_axes(ax_tilt, xlabel="Frame", ylabel="Tilt (°)")
    ax_rot.legend(markerscale=10, fontsize=9, frameon=False, loc="upper right")

    save_figure(fig, outpath)


# ── High-level entry point ────────────────────────────────────────────────────

def run_exploration(tilt, rot, outdir,
                    label="",
                    sweep_n=25,
                    mcs_min=50,
                    mcs_max=None,
                    sweep_subsample=20_000,
                    min_samples=None):
    """Run the full HDBSCAN exploration workflow and write all outputs to outdir.

    Steps
    -----
    1. Sweep min_cluster_size on a subsample → noise fraction + DBCV plot.
    2. Pick optimal MCS (peak DBCV).
    3. Run final HDBSCAN on ALL frames at the optimal MCS.
    4. Plot condensed tree, polar density with cluster centres, time series.

    Parameters
    ----------
    tilt, rot        : 1-D angle arrays (degrees)
    outdir           : output directory (created if needed)
    label            : short name used in titles and filenames
    sweep_n          : number of MCS steps in the sweep
    mcs_min          : smallest MCS to sweep
    mcs_max          : largest MCS to sweep (default: 5 % of subsample size)
    sweep_subsample  : frames to use during sweep (None = use all)
    min_samples      : override min_samples for the final run (None = use MCS)
    """
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, label.replace(" ", "_") + "_") if label else outdir + "/"

    valid = np.isfinite(tilt) & np.isfinite(rot)
    tilt  = tilt[valid]
    rot   = rot[valid]
    print(f"  {len(tilt)} valid frames ({(~valid).sum()} NaN dropped)")

    n_sub   = min(sweep_subsample, len(tilt)) if sweep_subsample else len(tilt)
    _mcs_max = mcs_max if mcs_max is not None else max(mcs_min + 1, n_sub // 20)
    mcs_range = np.unique(
        np.linspace(mcs_min, _mcs_max, sweep_n, dtype=int)
    )

    # ── 1. Sweep ──────────────────────────────────────────────────────────────
    print(f"\n[1/4] Sweeping MCS {mcs_range[0]}–{mcs_range[-1]} "
          f"({len(mcs_range)} values, subsample={n_sub})...")
    mcs_arr, noise_fracs, dbcv_scores = sweep_mcs(tilt, rot, mcs_range, subsample=n_sub)

    optimal_mcs, optimal_dbcv = find_optimal_mcs(mcs_arr, dbcv_scores)
    print(f"\n  Optimal MCS = {optimal_mcs}  (DBCV = {optimal_dbcv:.4f})")

    plot_sweep(mcs_arr, noise_fracs, dbcv_scores, optimal_mcs,
               prefix + "sweep.png")

    # ── 2. Final clustering on all frames ────────────────────────────────────
    print(f"\n[2/4] Final HDBSCAN on all {len(tilt)} frames (MCS={optimal_mcs})...")
    clusterer, labels, clusters = run_hdbscan(tilt, rot, optimal_mcs,
                                              min_samples=min_samples)

    # ── 3. Condensed tree ─────────────────────────────────────────────────────
    print("\n[3/4] Condensed tree...")
    plot_condensed_tree(clusterer, prefix + "condensed_tree.png", label=label)

    # ── 4. Clustered polar + time series ─────────────────────────────────────
    print("\n[4/4] Polar density, scatter, and time series...")
    plot_polar_clustered(tilt, rot, clusters,
                         prefix + "polar_clustered.png", label=label)
    plot_scatter_clustered(tilt, rot, labels, clusters,
                           prefix + "scatter_clustered.png", label=label)
    plot_timeseries_clustered(tilt, rot, labels,
                              prefix + "timeseries_clustered.png", label=label)

    print(f"\nDone. Outputs in: {outdir}")
    return {
        "mcs_sweep":    (mcs_arr, noise_fracs, dbcv_scores),
        "optimal_mcs":  optimal_mcs,
        "clusterer":    clusterer,
        "labels":       labels,
        "clusters":     clusters,
    }
