"""Protein RMSD (equilibration check).

Computes Cα RMSD vs. frame 0 for each system using the per-system
rmsd_selection (which typically excludes flexible loops).
"""

import os
import numpy as np
import MDAnalysis.analysis.rms as rms

from membrane_analysis.core.io import (
    cached_compute, save_per_system, load_cache_metadata, get_time_bounds,
)
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection,
    get_stride, get_sim_length, is_force_recompute,
    get_frame_window_for_analysis, build_cache_metadata,
    get_analysis_params,
)
from membrane_analysis.core.plotting import (
    line_plot, save_figure, multi_system_figure, overlay_line_plot,
)
import matplotlib.pyplot as plt


ANALYSIS_KEY = "rmsd"
OUTPUT_TYPE = "scalar"


def _compute_one(u, sel, start, stop, stride):
    """Align on *sel* then compute RMSD on *sel* vs frame ``start``."""
    R = rms.RMSD(u, u, ref_frame=start, select=sel, superposition=True)
    R.run(start=start, stop=stop, step=stride, verbose=True)
    return R.results.rmsd[:, 2]


def compute(cfg, universes):
    """Compute RMSD for all systems. Returns dict {name: 1D array}."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "rmsd.pkl")
    force  = is_force_recompute(cfg)
    metadata = build_cache_metadata(cfg, universes, ANALYSIS_KEY)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            sel = get_selection(cfg, name, "rmsd")
            if sel is None:
                print(f"  [{name}] No rmsd selection, skipping.")
                continue
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            start, stop = get_frame_window_for_analysis(
                cfg, name, universes[name], ANALYSIS_KEY)
            print(f"  [{name}] Computing RMSD "
                  f"(frames {start}:{stop}:{stride})...")
            results[name] = _compute_one(universes[name], sel, start, stop, stride)
        save_per_system(results, outdir, ANALYSIS_KEY, metadata=metadata)
        return results

    return cached_compute(cache, _run, force_recompute=force, metadata=metadata)


def plot(cfg, results):
    """Plot RMSD time series. Multi-system: shared-axes grid. Single: solo panel."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "rmsd.pkl")
    sim_us = get_sim_length(cfg)
    ma     = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 50)
    names  = list(results.keys())
    meta   = load_cache_metadata(cache)

    fig, axes = multi_system_figure(len(names), sharex=True, sharey=True)
    time_bounds = {}
    for ax, name in zip(axes, names):
        s_us, e_us = get_time_bounds(meta, name, sim_us)
        time_bounds[name] = (s_us, e_us)
        time = np.linspace(s_us, e_us, len(results[name]))
        line_plot(time, results[name], ax, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)

    fig.supxlabel("Time (μs)", fontsize=20)
    fig.supylabel("Cα RMSD (Å)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "rmsd_all.png"))

    overlay_line_plot(results, sim_us, "Cα RMSD (Å)",
                      os.path.join(outdir, "rmsd_comparison.png"),
                      ma_window=ma, time_bounds=time_bounds)

    # Per-system individual plots
    for name in names:
        s_us, e_us = get_time_bounds(meta, name, sim_us)
        fig_s, ax_s = plt.subplots(figsize=(5, 4), constrained_layout=True)
        time = np.linspace(s_us, e_us, len(results[name]))
        line_plot(time, results[name], ax_s, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
        ax_s.set_xlabel("Time (μs)", fontsize=14)
        ax_s.set_ylabel("Cα RMSD (Å)", fontsize=14)
        save_figure(fig_s, os.path.join(outdir, name, "rmsd.png"))
