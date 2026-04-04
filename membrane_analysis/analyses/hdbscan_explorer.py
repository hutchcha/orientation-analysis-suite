"""HDBSCAN exploration and validation for orientation-state clustering.

Sweeps min_cluster_size and min_samples in a 2-D grid, scoring each
combination with DBCV and noise fraction.  Produces a heatmap of the
grid, a condensed tree for the optimal parameters, and polar / scatter /
time-series diagnostic plots.

Supports ``cluster_selection_method`` = "eom" (Excess of Mass, default)
or "leaf" (selects leaf clusters — typically more granular).

Standalone usage::

    from membrane_analysis.analyses.hdbscan_explorer import run_exploration
    import numpy as np

    tilt = np.loadtxt("tilt.txt")
    rot  = np.loadtxt("rotation.txt")
    run_exploration(tilt, rot, outdir="explorer_out/", label="MySystem",
                    method="leaf")
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from membrane_analysis.core.plotting import style_axes, save_figure
from membrane_analysis.analyses.tilt_rotation import polar_density_plot


# ── Sphere geometry ──────────────────────────────────────────────────────────

def _sph_to_cart(rot_deg, tilt_deg):
    phi   = np.deg2rad(rot_deg)
    theta = np.deg2rad(tilt_deg)
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _cart_to_angles(v):
    from math import atan2
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.rad2deg(atan2(y, x)), np.rad2deg(np.arccos(np.clip(z, -1.0, 1.0)))


_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261",
    "#6A0572", "#1D3557", "#52B788", "#D62828", "#023E8A",
]

def _cluster_color(label):
    return "lightgrey" if label < 0 else _COLORS[label % len(_COLORS)]


# ── 2-D parameter sweep ─────────────────────────────────────────────────────

def sweep_params(tilt, rot, mcs_range, ms_range, subsample=20_000,
                 method="eom"):
    """Sweep (min_cluster_size, min_samples) on a subsample.

    Parameters
    ----------
    tilt, rot    : angle arrays (degrees)
    mcs_range    : 1-D array of MCS values
    ms_range     : 1-D array of min_samples values
    subsample    : frames to subsample (None = use all)
    method       : "eom" or "leaf"

    Returns
    -------
    results : list of dicts, each with keys:
              mcs, ms, n_clusters, noise_frac, dbcv
    best    : dict for the row with highest DBCV
    """
    try:
        import hdbscan as hdb_lib
    except ImportError:
        raise ImportError("hdbscan is not installed.")

    X_full = _sph_to_cart(rot, tilt)
    if subsample and len(X_full) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_full), size=subsample, replace=False)
        X = X_full[idx]
    else:
        X = X_full

    n = len(X)
    results = []

    for mcs in mcs_range:
        mcs = int(mcs)
        if mcs >= n // 2:
            continue
        for ms in ms_range:
            ms = int(ms)
            if ms > mcs:
                continue  # min_samples > min_cluster_size is invalid
            try:
                clust = hdb_lib.HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms,
                    cluster_selection_method=method,
                    gen_min_span_tree=True,
                ).fit(X)
                nf = float((clust.labels_ == -1).mean())
                dbcv = float(clust.relative_validity_)
                nc = len(set(clust.labels_) - {-1})
                results.append(dict(mcs=mcs, ms=ms, n_clusters=nc,
                                    noise_frac=nf, dbcv=dbcv))
                print(f"    MCS={mcs:5d}  ms={ms:4d}  k={nc:2d}  "
                      f"noise={nf:.1%}  DBCV={dbcv:.4f}")
            except Exception as exc:
                print(f"    MCS={mcs:5d}  ms={ms:4d}  ERROR: {exc}")

    if not results:
        return results, None

    best = max(results, key=lambda r: r["dbcv"])
    return results, best


# ── Final HDBSCAN run ────────────────────────────────────────────────────────

def run_hdbscan(tilt, rot, mcs, min_samples=None, method="eom"):
    """Run HDBSCAN on the full dataset at the chosen parameters.

    Returns (clusterer, labels, clusters).
    """
    try:
        import hdbscan as hdb_lib
    except ImportError:
        raise ImportError("hdbscan is not installed.")

    X = _sph_to_cart(rot, tilt)
    ms = min_samples if min_samples is not None else mcs

    clusterer = hdb_lib.HDBSCAN(
        min_cluster_size=int(mcs), min_samples=int(ms),
        cluster_selection_method=method,
        gen_min_span_tree=True,
    ).fit(X)
    labels = clusterer.labels_

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

    noise_frac = (labels == -1).mean()
    dbcv = float(clusterer.relative_validity_)
    print(f"  Final HDBSCAN: MCS={mcs}  ms={ms}  method={method}  "
          f"clusters={len(clusters)}  noise={noise_frac:.1%}  DBCV={dbcv:.4f}")
    for c in clusters:
        print(f"    cluster {c['label']}: "
              f"rot={c['rotation_deg_mean']:.1f}  "
              f"tilt={c['tilt_deg_mean']:.1f}  "
              f"pop={c['pop_fraction']:.3f}")

    return clusterer, labels, clusters


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_sweep_2d(results, best, outpath, metric="dbcv"):
    """2-D heatmap of DBCV (or noise_frac) over the (MCS, min_samples) grid.

    Also annotates n_clusters in each cell and marks the optimal cell.
    """
    if not results:
        return

    mcs_vals = sorted(set(r["mcs"] for r in results))
    ms_vals  = sorted(set(r["ms"]  for r in results))

    grid = np.full((len(ms_vals), len(mcs_vals)), np.nan)
    k_grid = np.full((len(ms_vals), len(mcs_vals)), np.nan)
    mcs_idx = {v: i for i, v in enumerate(mcs_vals)}
    ms_idx  = {v: i for i, v in enumerate(ms_vals)}

    for r in results:
        i, j = ms_idx[r["ms"]], mcs_idx[r["mcs"]]
        grid[i, j] = r[metric]
        k_grid[i, j] = r["n_clusters"]

    fig, ax = plt.subplots(figsize=(max(6, len(mcs_vals) * 0.7),
                                    max(4, len(ms_vals) * 0.5)),
                           constrained_layout=True)

    cmap = "RdYlGn" if metric == "dbcv" else "RdYlGn_r"
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar(im, ax=ax, label="DBCV" if metric == "dbcv" else "Noise fraction",
                 shrink=0.8)

    # annotate cells with n_clusters
    for i in range(len(ms_vals)):
        for j in range(len(mcs_vals)):
            if np.isfinite(k_grid[i, j]):
                k = int(k_grid[i, j])
                val = grid[i, j]
                c = "white" if (metric == "dbcv" and val < 0.15) else "black"
                ax.text(j, i, f"k={k}", ha="center", va="center",
                        fontsize=8, color=c)

    # mark optimal
    if best:
        bi, bj = ms_idx[best["ms"]], mcs_idx[best["mcs"]]
        ax.plot(bj, bi, marker="*", ms=18, color="black",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    ax.set_xticks(range(len(mcs_vals)))
    ax.set_xticklabels(mcs_vals, fontsize=9, rotation=45)
    ax.set_yticks(range(len(ms_vals)))
    ax.set_yticklabels(ms_vals, fontsize=9)

    method_str = f" ({best.get('method', '')})" if best and 'method' in best else ""
    style_axes(ax, xlabel="min_cluster_size", ylabel="min_samples",
               title=f"HDBSCAN parameter sweep{method_str}")

    save_figure(fig, outpath)


def plot_sweep_1d(results, best, outpath):
    """Noise fraction + DBCV vs MCS (collapsed across min_samples at optimal ms)."""
    if not results or not best:
        return

    opt_ms = best["ms"]
    rows = [r for r in results if r["ms"] == opt_ms]
    if len(rows) < 2:
        return

    rows.sort(key=lambda r: r["mcs"])
    mcs_arr = np.array([r["mcs"] for r in rows])
    noise   = np.array([r["noise_frac"] for r in rows])
    dbcv    = np.array([r["dbcv"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(8, 4), constrained_layout=True)

    c_noise, c_dbcv = "#457B9D", "#E63946"

    ax1.plot(mcs_arr, noise, color=c_noise, lw=2, marker="o", ms=4, label="Noise fraction")
    ax1.set_xlabel("min_cluster_size", fontsize=14)
    ax1.set_ylabel("Noise fraction", fontsize=14, color=c_noise)
    ax1.tick_params(axis="y", labelcolor=c_noise, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(mcs_arr, dbcv, color=c_dbcv, lw=2, marker="s", ms=4, label="DBCV")
    ax2.set_ylabel("DBCV", fontsize=14, color=c_dbcv)
    ax2.tick_params(axis="y", labelcolor=c_dbcv, labelsize=12)

    ax2.axvline(best["mcs"], color="black", ls="--", lw=1.5,
                label=f"Optimal MCS={best['mcs']}")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=11, frameon=False)
    ax1.set_title(f"min_samples = {opt_ms}", fontsize=13)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    save_figure(fig, outpath)


def plot_condensed_tree(clusterer, outpath, label=""):
    """Cluster persistence bar chart from the HDBSCAN condensed tree."""
    try:
        ct    = clusterer.condensed_tree_
        tree  = ct.to_pandas()
        labels = clusterer.labels_
        n_pts  = len(labels)

        cluster_nodes = sorted(tree[tree["child"] >= n_pts]["child"].unique())
        selected = set(np.unique(labels[labels >= 0]))

        bars = []
        for i, node in enumerate(cluster_nodes):
            births = tree[tree["child"] == node]["lambda_val"]
            birth  = float(births.min()) if len(births) > 0 else 0.0
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
            lw = 2 + 10 * (size / max_size)
            ax.plot([birth, death], [i, i], color=color, linewidth=lw,
                    solid_capstyle="butt")
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
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(rot[noise_mask], tilt[noise_mask], s=3, c="silver",
                   alpha=0.5, rasterized=True, label="noise", zorder=1)

    for c in clusters:
        lab  = c["label"]
        mask = labels == lab
        color = _cluster_color(lab)
        ax.scatter(rot[mask], tilt[mask], s=4, c=color, alpha=0.6,
                   rasterized=True, label=f"cluster {lab} ({c['pop_fraction']:.1%})",
                   zorder=2)
        ax.plot(c["rotation_deg_mean"], c["tilt_deg_mean"],
                marker="*", ms=16, color=color,
                markeredgecolor="black", markeredgewidth=0.8, zorder=3)

    style_axes(ax, title=label if label else "HDBSCAN clusters",
               xlabel="Rotation (\u00b0)", ylabel="Tilt (\u00b0)")
    ax.set_xlim(-185, 185)
    ax.set_ylim(0, max(tilt.max() + 5, 135))
    ax.legend(fontsize=10, frameon=True, framealpha=0.8, markerscale=5,
              loc="upper right")
    save_figure(fig, outpath)


def plot_timeseries_clustered(tilt, rot, labels, outpath, label=""):
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

    style_axes(ax_rot,  title=label, ylabel="Rotation (\u00b0)")
    style_axes(ax_tilt, xlabel="Frame", ylabel="Tilt (\u00b0)")
    ax_rot.legend(markerscale=10, fontsize=9, frameon=False, loc="upper right")
    save_figure(fig, outpath)


# ── High-level entry point ────────────────────────────────────────────────────

def run_exploration(tilt, rot, outdir,
                    label="",
                    method="eom",
                    sweep_n_mcs=15,
                    sweep_n_ms=8,
                    mcs_min=50,
                    mcs_max=None,
                    ms_min=5,
                    ms_max=None,
                    sweep_subsample=20_000):
    """Run the full HDBSCAN exploration workflow.

    Steps
    -----
    1. 2-D sweep of (min_cluster_size, min_samples) on a subsample.
    2. Pick optimal pair (peak DBCV).
    3. Run final HDBSCAN on ALL frames at optimal parameters.
    4. Plot: 2-D heatmap, 1-D slice, condensed tree, polar, scatter, timeseries.

    Parameters
    ----------
    method       : "eom" or "leaf"
    sweep_n_mcs  : number of MCS values in the grid
    sweep_n_ms   : number of min_samples values in the grid
    mcs_min/max  : MCS sweep range (max default: 5% of subsample)
    ms_min/max   : min_samples range (max default: mcs_max)
    sweep_subsample : frames for the sweep
    """
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, label.replace(" ", "_") + "_") if label else outdir + "/"

    valid = np.isfinite(tilt) & np.isfinite(rot)
    tilt  = tilt[valid]
    rot   = rot[valid]
    print(f"  {len(tilt)} valid frames ({(~valid).sum()} NaN dropped)")

    n_sub = min(sweep_subsample, len(tilt)) if sweep_subsample else len(tilt)
    _mcs_max = int(mcs_max) if mcs_max is not None else max(mcs_min + 1, n_sub // 20)
    _ms_max  = int(ms_max) if ms_max is not None else _mcs_max

    mcs_range = np.unique(np.linspace(mcs_min, _mcs_max, sweep_n_mcs, dtype=int))
    ms_range  = np.unique(np.linspace(ms_min, _ms_max, sweep_n_ms, dtype=int))

    # ── 1. 2-D sweep ────────────────────────────────────────────────────────
    total = sum(1 for mcs in mcs_range for ms in ms_range if ms <= mcs and mcs < n_sub // 2)
    print(f"\n[1/4] 2-D sweep: {len(mcs_range)} MCS x {len(ms_range)} ms "
          f"= {total} combos (method={method}, subsample={n_sub})...")
    results, best = sweep_params(tilt, rot, mcs_range, ms_range,
                                  subsample=n_sub, method=method)

    if best is None:
        print("  Sweep produced no valid results.")
        return {"sweep_results": results, "best": None}

    best["method"] = method
    print(f"\n  Optimal: MCS={best['mcs']}  ms={best['ms']}  "
          f"k={best['n_clusters']}  noise={best['noise_frac']:.1%}  "
          f"DBCV={best['dbcv']:.4f}")

    plot_sweep_2d(results, best, prefix + "sweep_2d.png")
    plot_sweep_1d(results, best, prefix + "sweep_1d.png")

    # ── 2. Final clustering ─────────────────────────────────────────────────
    print(f"\n[2/4] Final HDBSCAN on all {len(tilt)} frames "
          f"(MCS={best['mcs']}, ms={best['ms']}, method={method})...")
    clusterer, labels, clusters = run_hdbscan(
        tilt, rot, best["mcs"], min_samples=best["ms"], method=method)

    # ── 3. Condensed tree ────────────────────────────────────────────────────
    print("\n[3/4] Condensed tree...")
    plot_condensed_tree(clusterer, prefix + "condensed_tree.png", label=label)

    # ── 4. Diagnostic plots ──────────────────────────────────────────────────
    print("\n[4/4] Polar density, scatter, and time series...")
    plot_polar_clustered(tilt, rot, clusters,
                         prefix + "polar_clustered.png", label=label)
    plot_scatter_clustered(tilt, rot, labels, clusters,
                           prefix + "scatter_clustered.png", label=label)
    plot_timeseries_clustered(tilt, rot, labels,
                              prefix + "timeseries_clustered.png", label=label)

    print(f"\nDone. Outputs in: {outdir}")
    return {
        "sweep_results": results,
        "best":          best,
        "clusterer":     clusterer,
        "labels":        labels,
        "clusters":      clusters,
    }
