"""Lobe COM Z-distance analysis.

Tracks the Z-distance of two protein lobes from the membrane midplane.
Convention: -1 * (lobe_z - membrane_z), so positive = lobe closer to membrane.

Config fields per system:
  selections.lobe1         — first lobe (e.g. N-terminal domain)
  selections.lobe2         — second lobe (e.g. C-terminal domain)
  selections.membrane_com  — membrane selection for COM reference
                             (defaults to "resname POPC POPE SAPI CHL1 and name C*")
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde

from membrane_analysis.core.io import (
    cached_compute, save_per_system, load_cache_metadata, get_time_bounds,
)
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection, get_system,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
    get_frame_window_for_analysis, build_cache_metadata,
)
from membrane_analysis.core.plotting import line_plot, style_axes, save_figure, multi_system_figure
import matplotlib.pyplot as plt


ANALYSIS_KEY = "lobe_com"
OUTPUT_TYPE = "scalar"
OUTPUT_FIELDS = {"Lobe1": "scalar", "Lobe2": "scalar"}

DEFAULT_MEM_SEL = "resname POPC POPE SAPI CHL1 and name C*"


def _compute_one(u, lobe1_sel, lobe2_sel, mem_sel, start, stop, stride):
    """Return DataFrame with Lobe1 and Lobe2 Z-distances per frame."""
    d1, d2 = [], []
    for ts in tqdm(u.trajectory[start:stop:stride]):
        lobe1 = u.select_atoms(f"protein and {lobe1_sel}")
        lobe2 = u.select_atoms(f"protein and {lobe2_sel}")
        memcen = u.select_atoms(mem_sel)
        mem_z = memcen.center_of_mass()[2]
        # convention: positive = closer to membrane
        d1.append(-1 * (lobe1.center_of_mass()[2] - mem_z))
        d2.append(-1 * (lobe2.center_of_mass()[2] - mem_z))
    return pd.DataFrame({"Lobe1": np.array(d1), "Lobe2": np.array(d2)})


def compute(cfg, universes):
    """Compute lobe COM distances for all systems. Returns dict {name: DataFrame}."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "lobe_com.pkl")
    force  = is_force_recompute(cfg)
    metadata = build_cache_metadata(cfg, universes, ANALYSIS_KEY)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            l1 = get_selection(cfg, name, "lobe1")
            l2 = get_selection(cfg, name, "lobe2")
            if l1 is None or l2 is None:
                print(f"  [{name}] Missing lobe1/lobe2 selections, skipping.")
                continue
            mem_sel = get_selection(cfg, name, "membrane_com") or DEFAULT_MEM_SEL
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            start, stop = get_frame_window_for_analysis(
                cfg, name, universes[name], ANALYSIS_KEY)
            print(f"  [{name}] Computing lobe COM distances "
                  f"(frames {start}:{stop}:{stride})...")
            results[name] = _compute_one(
                universes[name], l1, l2, mem_sel, start, stop, stride)
        save_per_system(results, outdir, ANALYSIS_KEY, metadata=metadata)
        return results

    return cached_compute(cache, _run, force_recompute=force, metadata=metadata)


def plot(cfg, results):
    """Plot lobe COM time series and 2D KDE contours. Multi-system: shared axes."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "lobe_com.pkl")
    sim_us = get_sim_length(cfg)
    ma     = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 200)
    names  = list(results.keys())
    meta   = load_cache_metadata(cache)

    # ── time series: one panel per system, shared axes ────────────────────────
    fig, axes = multi_system_figure(len(names), sharex=True, sharey=True,
                                    ax_w=7, ax_h=4)
    for ax, name in zip(axes, names):
        df   = results[name]
        s_us, e_us = get_time_bounds(meta, name, sim_us)
        time = np.linspace(s_us, e_us, len(df))
        line_plot(time, df["Lobe1"].values, ax, title=name,
                  color="orange", z=1, label="Lobe1",
                  ma_window=ma, ma_color="orange", ma_z=2)
        line_plot(time, df["Lobe2"].values, ax,
                  color="blue",   z=1, label="Lobe2",
                  ma_window=ma, ma_color="blue",   ma_z=2)
        ax.legend(fontsize=12, frameon=False)

    fig.supxlabel("Time (μs)", fontsize=20)
    fig.supylabel("Z_lobe (Å)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "lobe_timeseries_all.png"))

    # ── 2D KDE contours: shared colorbar, one panel per system ───────────────
    import math
    ncols = min(len(names), 2)
    nrows = math.ceil(len(names) / ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                sharex=True, sharey=True, constrained_layout=True)
    axes2_flat = [axes2] if len(names) == 1 else list(np.asarray(axes2).flatten())
    for ax in axes2_flat[len(names):]:
        ax.set_visible(False)

    vmin, vmax = 0.02, 1.0
    cf_last = None
    for ax, name in zip(axes2_flat, names):
        df  = results[name]
        lb1 = df["Lobe1"].values
        lb2 = df["Lobe2"].values
        x   = lb1[np.isfinite(lb1) & np.isfinite(lb2)]
        y   = lb2[np.isfinite(lb1) & np.isfinite(lb2)]
        if len(x) < 10:
            ax.set_title(name, fontsize=20)
            continue

        kde  = gaussian_kde(np.vstack([x, y]))
        xg   = np.linspace(x.min() - 5, x.max() + 5, 80)
        yg   = np.linspace(y.min() - 5, y.max() + 5, 80)
        X, Y = np.meshgrid(xg, yg)
        Z    = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        Z    = Z / Z.max()
        Z[Z < 0.02] = np.nan

        cf_last = ax.contourf(X, Y, Z, levels=np.linspace(vmin, vmax, 10),
                              vmin=vmin, vmax=vmax, cmap="viridis")
        style_axes(ax, title=name)

    if cf_last is not None:
        fig2.colorbar(cf_last, ax=axes2_flat, orientation="vertical",
                      fraction=0.02, pad=0.02, label="Probability")
    fig2.supxlabel("Z_Lobe1 (Å)", fontsize=20)
    fig2.supylabel("Z_Lobe2 (Å)", fontsize=20)
    save_figure(fig2, os.path.join(outdir, "lobe_contour_all.png"))

    # Per-system individual plots
    for name in names:
        df   = results[name]
        s_us, e_us = get_time_bounds(meta, name, sim_us)
        time = np.linspace(s_us, e_us, len(df))

        # Time series: both lobes on same axes
        fig_s, ax_s = plt.subplots(figsize=(5, 4), constrained_layout=True)
        line_plot(time, df["Lobe1"].values, ax_s, title=name,
                  color="orange", z=1, label="Lobe1",
                  ma_window=ma, ma_color="orange", ma_z=2)
        line_plot(time, df["Lobe2"].values, ax_s,
                  color="blue", z=1, label="Lobe2",
                  ma_window=ma, ma_color="blue", ma_z=2)
        ax_s.legend(fontsize=12, frameon=False)
        ax_s.set_xlabel("Time (μs)", fontsize=14)
        ax_s.set_ylabel("Z_lobe (Å)", fontsize=14)
        save_figure(fig_s, os.path.join(outdir, name, "lobe_timeseries.png"))

        # KDE contour plot
        lb1 = df["Lobe1"].values
        lb2 = df["Lobe2"].values
        x   = lb1[np.isfinite(lb1) & np.isfinite(lb2)]
        y   = lb2[np.isfinite(lb1) & np.isfinite(lb2)]
        if len(x) >= 10:
            fig_c, ax_c = plt.subplots(figsize=(5, 4), constrained_layout=True)
            kde  = gaussian_kde(np.vstack([x, y]))
            xg   = np.linspace(x.min() - 5, x.max() + 5, 80)
            yg   = np.linspace(y.min() - 5, y.max() + 5, 80)
            X, Y = np.meshgrid(xg, yg)
            Z    = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            Z    = Z / Z.max()
            Z[Z < 0.02] = np.nan
            cf = ax_c.contourf(X, Y, Z, levels=np.linspace(0.02, 1.0, 10),
                               vmin=0.02, vmax=1.0, cmap="viridis")
            style_axes(ax_c, title=name)
            fig_c.colorbar(cf, ax=ax_c, orientation="vertical",
                           fraction=0.02, pad=0.02, label="Probability")
            ax_c.set_xlabel("Z_Lobe1 (Å)", fontsize=14)
            ax_c.set_ylabel("Z_Lobe2 (Å)", fontsize=14)
            save_figure(fig_c, os.path.join(outdir, name, "lobe_contour.png"))
