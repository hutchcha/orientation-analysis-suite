"""Inter-residue (or inter-selection) distance measurement.

Computes the Euclidean distance between the COM of two atom selections
per frame.  Useful as a radial coordinate for the kinetics module's
spherical embedding (rotation, tilt, distance -> Cartesian).

Config selections per system
-----------------------------
  distance_sel1 : first atom selection  (e.g. "protein and resid 50 and name CA")
  distance_sel2 : second atom selection (e.g. "protein and resid 170 and name CA")
"""

import os
import numpy as np
from tqdm import tqdm

from membrane_analysis.core.io import cached_compute
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import (
    line_plot, save_figure, multi_system_figure, overlay_line_plot,
)
import matplotlib.pyplot as plt


ANALYSIS_KEY = "inter_residue_distance"
OUTPUT_TYPE = "scalar"


def _compute_one(u, sel1, sel2, stride):
    """Return 1D array of COM-COM distances (Angstrom)."""
    dists = []
    for ts in tqdm(u.trajectory[::stride]):
        com1 = u.select_atoms(sel1).center_of_mass()
        com2 = u.select_atoms(sel2).center_of_mass()
        dists.append(np.linalg.norm(com1 - com2))
    return np.array(dists)


def compute(cfg, universes):
    """Compute inter-residue distances for all systems."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "inter_residue_distance.pkl")
    force  = is_force_recompute(cfg)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            sel1 = get_selection(cfg, name, "distance_sel1")
            sel2 = get_selection(cfg, name, "distance_sel2")
            if sel1 is None or sel2 is None:
                print(f"  [{name}] Missing distance_sel1 or distance_sel2, skipping.")
                continue
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            print(f"  [{name}] Computing inter-residue distance (stride={stride})...")
            results[name] = _compute_one(universes[name], sel1, sel2, stride)
        return results

    return cached_compute(cache, _run, force_recompute=force)


def plot(cfg, results):
    """Plot distance time series."""
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
    fig.supylabel("Distance (Å)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "inter_residue_distance_all.png"))

    overlay_line_plot(results, sim_us, "Distance (Å)",
                      os.path.join(outdir, "inter_residue_distance_comparison.png"),
                      ma_window=ma)
