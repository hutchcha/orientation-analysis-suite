"""P-P distance / bilayer thickness (equilibration check).

Computes the Z-distance between upper and lower leaflet phosphorus (or
head-atom) COM per frame.  The membrane midplane is defined once at t=0
from the phosphorus_selection, then upper/lower leaflets are split by
that z-value each frame.
"""

import os
import numpy as np
from tqdm import tqdm

from membrane_analysis.core.io import (
    cached_compute, save_per_system, load_cache_metadata, get_time_bounds,
)
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
    get_frame_window_for_analysis, build_cache_metadata,
)
from membrane_analysis.core.plotting import line_plot, save_figure, multi_system_figure, overlay_line_plot
import matplotlib.pyplot as plt


ANALYSIS_KEY = "pp_distance"
OUTPUT_TYPE = "scalar"


def _compute_one(u, phos_sel, start, stop, stride):
    """Return 1D array of P-P z-distances (Å)."""
    ag = u.select_atoms(phos_sel)
    # Midplane defined from the first analysed frame (start of window).
    # This avoids using a pre-equilibration frame's midplane when a
    # window skips early simulation time.
    u.trajectory[start]
    mid_z = ag.center_of_mass()[2]

    zdist = []
    for ts in tqdm(u.trajectory[start:stop:stride]):
        top_ag = u.select_atoms(f"({phos_sel}) and prop z > {mid_z}")
        bot_ag = u.select_atoms(f"({phos_sel}) and prop z < {mid_z}")
        if top_ag.n_atoms == 0 or bot_ag.n_atoms == 0:
            zdist.append(np.nan)
            continue
        zdist.append(top_ag.center_of_mass()[2] - bot_ag.center_of_mass()[2])
    return np.array(zdist)


def compute(cfg, universes):
    """Compute P-P distances for all systems."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "pp_distance.pkl")
    force  = is_force_recompute(cfg)
    metadata = build_cache_metadata(cfg, universes, ANALYSIS_KEY)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            phos_sel = get_selection(cfg, name, "phosphorus")
            if phos_sel is None:
                print(f"  [{name}] No phosphorus selection, skipping.")
                continue
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            start, stop = get_frame_window_for_analysis(
                cfg, name, universes[name], ANALYSIS_KEY)
            print(f"  [{name}] Computing P-P distance "
                  f"(frames {start}:{stop}:{stride})...")
            results[name] = _compute_one(
                universes[name], phos_sel, start, stop, stride)
        save_per_system(results, outdir, ANALYSIS_KEY, metadata=metadata)
        return results

    return cached_compute(cache, _run, force_recompute=force, metadata=metadata)


def plot(cfg, results):
    """Plot P-P distance. Multi-system: shared-axes grid. Single: solo panel."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "pp_distance.pkl")
    sim_us = get_sim_length(cfg)
    ma     = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 500)
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
    fig.supylabel("Bilayer Thickness (Å)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "pp_distance_all.png"))

    overlay_line_plot(results, sim_us, "Bilayer Thickness (Å)",
                      os.path.join(outdir, "pp_distance_comparison.png"),
                      ma_window=ma, time_bounds=time_bounds)

    # Per-system individual plots
    for name in names:
        s_us, e_us = get_time_bounds(meta, name, sim_us)
        fig_s, ax_s = plt.subplots(figsize=(5, 4), constrained_layout=True)
        time = np.linspace(s_us, e_us, len(results[name]))
        line_plot(time, results[name], ax_s, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
        ax_s.set_xlabel("Time (μs)", fontsize=14)
        ax_s.set_ylabel("Bilayer Thickness (Å)", fontsize=14)
        save_figure(fig_s, os.path.join(outdir, name, "pp_distance.png"))
