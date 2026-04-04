"""Protein RMSD (equilibration check).

Computes Cα RMSD vs. frame 0 for each system using the per-system
rmsd_selection (which typically excludes flexible loops).
"""

import os
import numpy as np
import MDAnalysis.analysis.rms as rms

from membrane_analysis.core.io import cached_compute, save_pickle
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection,
    get_stride, get_sim_length, is_force_recompute,
)
from membrane_analysis.core.plotting import line_plot, save_figure, multi_system_figure, style_axes
from membrane_analysis.core.config import get_analysis_params
import matplotlib.pyplot as plt


ANALYSIS_KEY = "rmsd"


def _compute_one(u, sel, stride):
    """Align on *sel* then compute RMSD on *sel* vs frame 0.

    ``superposition=True`` (the default) superimposes mobile onto the reference
    before measuring RMSD, so translational/rotational drift is removed.
    The same selection is used for both alignment and RMSD measurement.
    """
    R = rms.RMSD(u, u, ref_frame=0, select=sel, superposition=True)
    R.run(step=stride, verbose=True)
    return R.results.rmsd[:, 2]  # column 2 = RMSD in Å


def compute(cfg, universes):
    """Compute RMSD for all systems. Returns dict {name: 1D array}."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache = os.path.join(outdir, "rmsd.pkl")
    force = is_force_recompute(cfg)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            sel = get_selection(cfg, name, "rmsd")
            if sel is None:
                print(f"  [{name}] No rmsd selection, skipping.")
                continue
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            print(f"  [{name}] Computing RMSD (stride={stride})...")
            results[name] = _compute_one(universes[name], sel, stride)
        return results

    return cached_compute(cache, _run, force_recompute=force)


def plot(cfg, results):
    """Plot RMSD time series. Multi-system: shared-axes grid. Single: solo panel."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us = get_sim_length(cfg)
    ma     = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 50)
    names  = list(results.keys())

    fig, axes = multi_system_figure(len(names), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        time = np.linspace(0, sim_us, len(results[name]))
        line_plot(time, results[name], ax, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)

    fig.supxlabel("Time (us)", fontsize=20)
    fig.supylabel("Ca RMSD (A)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "rmsd_all.png"))
