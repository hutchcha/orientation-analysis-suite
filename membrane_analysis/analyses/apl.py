"""Area per lipid (equilibration check).

Uses lipyphilic AreaPerLipid for the actual Voronoi-based APL calculation, but
replaces lipyphilic's AssignLeaflets with a custom static assignment based on
the membrane midplane at frame 0.  This is robust, fast, and correct for
simulations where lipids never undergo flip-flop.

Custom leaflet assignment
--------------------------
At frame 0:
  1. The midplane z is computed from the COM of all head atoms in apl_headgroup.
  2. Each lipid is assigned +1 (upper) or -1 (lower) based on whether its head
     atom z is above or below the midplane.
  3. The assignment is static — tiled across all frames — since flip-flop does
     not occur in these simulations.

Config fields used per system
------------------------------
  selections.apl_headgroup  — head-atom selection covering ALL lipid types
                               (must include every resname in lipid_headgroups)
  lipid_headgroups          — dict {resname: head-atom-sel} for per-type APL
                               e.g. {"POPC": "name P", "POPS": "name P",
                                     "CHL1": "name O3"}
"""

import os

import numpy as np
import pandas as pd

from membrane_analysis.core.io import cached_compute
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_system, get_selection,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import line_plot, save_figure, multi_system_figure
import matplotlib.pyplot as plt


ANALYSIS_KEY = "apl"

LIPID_COLORS = {
    "Total": "black", "POPC": "blue", "POPE": "green", "POPS": "gold",
    "PLA18": "purple", "PSM": "pink", "SAPI": "orange", "CHL1": "red",
    "Upper POPC": "cyan", "Lower POPC": "magenta",
}


# ── Leaflet assignment ────────────────────────────────────────────────────────

def _assign_leaflets_static(u, apl_sel, n_frames):
    """Assign leaflets based on frame-0 midplane; return (n_lipids, n_frames) array.

    Upper leaflet (+1): head atom z > midplane at frame 0.
    Lower leaflet (-1): head atom z < midplane at frame 0.

    The assignment is static because these simulations have no flip-flop.
    """
    u.trajectory[0]
    membrane = u.select_atoms(apl_sel)

    # one representative atom per residue (the head atom, first atom in the
    # selection for that residue)
    head_z = np.array([res.atoms[0].position[2]
                       for res in membrane.residues])

    mid_z = membrane.center_of_mass()[2]
    leaflet_0 = np.where(head_z > mid_z, 1, -1).astype(np.int8)

    # tile across all frames (shape: n_lipids × n_frames)
    return np.tile(leaflet_0[:, np.newaxis], (1, n_frames))


# ── Per-lipid row-index map ───────────────────────────────────────────────────

def _resid_row_map(u, apl_sel):
    """Map resid → row index in AreaPerLipid.areas (= position in membrane.residues)."""
    membrane = u.select_atoms(apl_sel)
    return {res.resid: i for i, res in enumerate(membrane.residues)}


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_one(u, apl_sel, lipid_headgroups, stride):
    """
    Compute APL for one universe.

    Returns a DataFrame: rows = sampled frames, columns = Total + per-lipid
    + Upper/Lower POPC (when POPC is present).  Values are mean APL (Å²).

    Shape note: AreaPerLipid.areas is (n_lipids, n_frames).
    """
    from lipyphilic.analysis.area_per_lipid import AreaPerLipid

    # count strided frames
    n_frames_total = len(u.trajectory)
    n_frames = len(range(0, n_frames_total, stride))

    # custom static leaflet assignment
    print(f"    Assigning leaflets (static, frame-0 midplane)...")
    leaflets = _assign_leaflets_static(u, apl_sel, n_frames)
    n_upper = int((leaflets[:, 0] == 1).sum())
    n_lower = int((leaflets[:, 0] == -1).sum())
    print(f"    Upper: {n_upper} lipids,  Lower: {n_lower} lipids")

    # APL calculation
    print(f"    Running AreaPerLipid...")
    areas = AreaPerLipid(universe=u, lipid_sel=apl_sel, leaflets=leaflets)
    areas.run(step=stride, verbose=True)

    # shape: (n_lipids, n_frames)
    arr = areas.areas
    resid_to_row = _resid_row_map(u, apl_sel)

    apl_df = pd.DataFrame()
    apl_df["Total"] = arr.mean(axis=0)   # mean over lipids per frame

    # per-lipid type
    for resname, head_sel in lipid_headgroups.items():
        ag = u.select_atoms(f"resname {resname} and {head_sel}")
        if ag.n_atoms == 0:
            print(f"    {resname}: not present, skipping.")
            continue

        rows = [resid_to_row[r] for r in ag.residues.resids if r in resid_to_row]
        if not rows:
            print(f"    {resname}: resids not in APL array — check apl_headgroup "
                  f"selection includes resname {resname}.")
            continue

        apl_df[resname] = arr[rows, :].mean(axis=0)

    # upper/lower leaflet POPC (if present)
    if "POPC" in lipid_headgroups:
        popc_ag = u.select_atoms("resname POPC and name P")
        if popc_ag.n_atoms > 0:
            all_rows = np.array([resid_to_row[r]
                                 for r in popc_ag.residues.resids
                                 if r in resid_to_row])
            # leaflets[:, 0] is the static assignment (same for all frames)
            upper_rows = all_rows[leaflets[all_rows, 0] == 1]
            lower_rows = all_rows[leaflets[all_rows, 0] == -1]
            if upper_rows.size:
                apl_df["Upper POPC"] = arr[upper_rows, :].mean(axis=0)
            if lower_rows.size:
                apl_df["Lower POPC"] = arr[lower_rows, :].mean(axis=0)

    return apl_df


# ── Pipeline interface ────────────────────────────────────────────────────────

def compute(cfg, universes):
    """Compute APL for all systems. Returns dict {name: DataFrame}."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "apl.pkl")
    force  = is_force_recompute(cfg)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            apl_sel  = get_selection(cfg, name, "apl_headgroup")
            sys_cfg  = get_system(cfg, name)
            lipid_hg = sys_cfg.get("lipid_headgroups", {})

            if apl_sel is None:
                print(f"  [{name}] No apl_headgroup selection, skipping.")
                continue
            if not lipid_hg:
                print(f"  [{name}] No lipid_headgroups defined, skipping.")
                continue

            stride = get_stride(cfg, name, ANALYSIS_KEY)
            print(f"  [{name}] Computing APL (stride={stride})...")
            results[name] = _compute_one(universes[name], apl_sel, lipid_hg, stride)
        return results

    return cached_compute(cache, _run, force_recompute=force)


def plot(cfg, results):
    """Plot APL per lipid type. Multi-system: shared-axes grid, one panel per system."""
    outdir    = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us    = get_sim_length(cfg)
    ma        = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 500)
    names     = list(results.keys())
    plot_cols = ["Total", "POPC", "POPE", "POPS", "PLA18", "PSM", "SAPI",
                 "CHL1", "Upper POPC", "Lower POPC"]

    fig, axes = multi_system_figure(len(names), sharex=True, sharey=True,
                                    ax_w=7, ax_h=4)
    for ax, name in zip(axes, names):
        df   = results[name]
        time = np.linspace(0, sim_us, len(df))
        for col in plot_cols:
            if col not in df.columns:
                continue
            c = LIPID_COLORS.get(col, "gray")
            line_plot(time, df[col].values, ax, title=name, color=c, z=1,
                      label=col, ma_window=ma, ma_color=c, ma_z=2)
        ax.legend(loc="upper right", fontsize=10, frameon=False)

    fig.supxlabel("Time (us)", fontsize=20)
    fig.supylabel("Area per Lipid (A^2)", fontsize=20)
    save_figure(fig, os.path.join(outdir, "apl_all.png"))
