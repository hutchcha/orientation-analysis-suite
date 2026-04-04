"""Lipid anchor insertion depth (equilibration check).

Computes the Z-distance between the membrane heavy-atom COM and the
lipid anchor COM per frame.  Convention: membrane_COM_z - anchor_COM_z,
so positive values indicate the anchor is below the membrane midplane.
"""

import os
import numpy as np
from tqdm import tqdm

from membrane_analysis.core.io import cached_compute, save_per_system
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import line_plot, save_figure, multi_system_figure, overlay_line_plot
import matplotlib.pyplot as plt


ANALYSIS_KEY = "anchor_insertion"
OUTPUT_TYPE = "scalar"


def _compute_one(u, anchor_sel, membrane_sel, stride):
    """Return 1D array of insertion depths (Å) for one universe."""
    depths = []
    for ts in tqdm(u.trajectory[::stride]):
        anchor_z = u.select_atoms(anchor_sel).center_of_mass()[2]
        mem_z = u.select_atoms(membrane_sel).center_of_mass()[2]
        depths.append(mem_z - anchor_z)
    return np.array(depths)


def compute(cfg, universes):
    """Compute anchor insertion depth for all systems."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache = os.path.join(outdir, "anchor_insert_depth.pkl")
    force = is_force_recompute(cfg)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            anchor_sel = get_selection(cfg, name, "anchor")
            membrane_sel = get_selection(cfg, name, "membrane_heavy")
            if anchor_sel is None or membrane_sel is None:
                print(f"  [{name}] Missing anchor or membrane_heavy selection, skipping.")
                continue
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            print(f"  [{name}] Computing anchor insertion depth (stride={stride})...")
            results[name] = _compute_one(universes[name], anchor_sel, membrane_sel, stride)
        save_per_system(results, outdir, ANALYSIS_KEY)
        return results

    return cached_compute(cache, _run, force_recompute=force)


def plot(cfg, results):
    """Plot anchor insertion depth. Multi-system: shared-axes grid."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us = get_sim_length(cfg)
    ma     = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 50)
    names  = list(results.keys())

    fig, axes = multi_system_figure(len(names), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        time = np.linspace(0, sim_us, len(results[name]))
        line_plot(time, results[name], ax, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)

    fig.supxlabel("Time (μs)", fontsize=20)
    fig.supylabel("Insertion Depth (Å)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "anchor_insertion_all.png"))

    overlay_line_plot(results, sim_us, "Insertion Depth (Å)",
                      os.path.join(outdir, "anchor_insertion_comparison.png"), ma_window=ma)

    # Per-system individual plots
    for name in names:
        fig_s, ax_s = plt.subplots(figsize=(5, 4), constrained_layout=True)
        time = np.linspace(0, sim_us, len(results[name]))
        line_plot(time, results[name], ax_s, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
        ax_s.set_xlabel("Time (μs)", fontsize=14)
        ax_s.set_ylabel("Insertion Depth (Å)", fontsize=14)
        save_figure(fig_s, os.path.join(outdir, name, "anchor_insertion.png"))
