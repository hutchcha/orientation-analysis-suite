"""Anchor-region RMSD and hydrogen bond analysis.

Computes:
  1. RMSD of the lipid anchor region (after whole-protein backbone alignment)
  2. H-bond counts between user-defined donors (or auto-guessed) and membrane lipids,
     broken down by lipid type (skipping lipids not present).

Config fields per system:
  selections.anchor_rmsd     — anchor region Cα selection for RMSD
  selections.align            — alignment selection (default: "protein and backbone")
  hbonds.donors               — donor selection string, or null for guess_hydrogens()
  hbonds.acceptor_lipids      — list of lipid resnames to check for H-bonds
  hbonds.between              — optional explicit 'between' list for HBA
"""

import os
import numpy as np
from tqdm import tqdm

from membrane_analysis.core.io import cached_compute
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection, get_system,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import line_plot, style_axes, save_figure
import matplotlib.pyplot as plt

from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


ANALYSIS_KEY = "hbonds"
OUTPUT_TYPE = "scalar"

LIPID_ACCEPTOR_ATOMS = "(name O* or name N* or name P*)"


def _lipid_present(u, resname):
    return u.select_atoms(f"resname {resname}").n_atoms > 0


def _frames_and_time(u, stride=1):
    """Return (frame_indices, time_in_us) arrays."""
    frames = np.arange(0, len(u.trajectory), stride, dtype=int)
    dt_ps = getattr(u.trajectory, "dt", None)

    u.trajectory[frames[0]]
    has_time = hasattr(u.trajectory.ts, "time") and u.trajectory.ts.time is not None

    if has_time:
        t_ps = np.empty_like(frames, dtype=float)
        for i, fr in enumerate(frames):
            u.trajectory[fr]
            t_ps[i] = u.trajectory.ts.time
    elif dt_ps is not None:
        t_ps = frames * float(dt_ps)
    else:
        t_ps = frames.astype(float)

    return frames, t_ps / 1e6


def _aligned_roi_rmsd(u, roi_sel, align_sel="protein and backbone",
                      ref_frame=0, stride=1):
    """Compute RMSD of a ROI after backbone alignment. Returns (time_us, rmsd_Å)."""
    r = RMSD(u, u, select=align_sel, groupselections=[roi_sel],
             ref_frame=ref_frame)
    r.run(step=stride, verbose=True)
    arr = r.results.rmsd
    time_us = arr[:, 1] / 1e6  # ps → µs
    roi_rmsd = arr[:, 3]       # first groupselection
    return time_us, roi_rmsd


def _hbond_counts(u, between_groups, acceptor_lipids, stride=1):
    """
    Run HBA and return per-frame total counts + per-lipid counts.
    If between_groups is None, uses guess_hydrogens mode.
    """
    frames, t_us = _frames_and_time(u, stride)
    n = len(frames)

    # figure out which lipids are actually in the system
    present = [r for r in acceptor_lipids if _lipid_present(u, r)]
    if not present:
        return {"time_us": t_us, "total": np.zeros(n, dtype=int), "by_lipid": {}}

    lipid_union = " or ".join(f"resname {r}" for r in present)

    # total H-bonds
    if between_groups is not None:
        h = HydrogenBondAnalysis(universe=u, between=between_groups)
    else:
        # default: all protein–lipid H-bonds with auto hydrogen guessing
        h = HydrogenBondAnalysis(
            universe=u,
            donors_sel=f"protein",
            acceptors_sel=f"({lipid_union}) and {LIPID_ACCEPTOR_ATOMS}",
        )
        h.guess_hydrogens("protein")
        h.guess_acceptors(f"({lipid_union}) and {LIPID_ACCEPTOR_ATOMS}")

    h.run(step=stride, verbose=True)

    hb = h.results.hbonds
    total_counts = np.zeros(n, dtype=int)
    if hb is not None and len(hb) > 0:
        frame_to_i = {fr: i for i, fr in enumerate(frames)}
        for fr in hb[:, 0].astype(int):
            i = frame_to_i.get(fr)
            if i is not None:
                total_counts[i] += 1

    # per-lipid breakdown (re-run per lipid type)
    by_lipid = {}
    for r in present:
        if between_groups is not None:
            # rebuild between for this specific lipid
            between_lip = [between_groups[0], f"resname {r}"]
            h_lip = HydrogenBondAnalysis(universe=u, between=between_lip)
        else:
            h_lip = HydrogenBondAnalysis(
                universe=u,
                donors_sel="protein",
                acceptors_sel=f"resname {r} and {LIPID_ACCEPTOR_ATOMS}",
            )
            h_lip.guess_hydrogens("protein")
            h_lip.guess_acceptors(f"resname {r} and {LIPID_ACCEPTOR_ATOMS}")

        h_lip.run(step=stride, verbose=True)
        hb_lip = h_lip.results.hbonds
        counts = np.zeros(n, dtype=int)
        if hb_lip is not None and len(hb_lip) > 0:
            for fr in hb_lip[:, 0].astype(int):
                i = frame_to_i.get(fr)
                if i is not None:
                    counts[i] += 1
        by_lipid[r] = counts

    return {"time_us": t_us, "total": total_counts, "by_lipid": by_lipid}


def compute(cfg, universes):
    """Compute anchor RMSD + H-bonds for all systems."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache = os.path.join(outdir, "hbonds_rmsd.pkl")
    force = is_force_recompute(cfg)

    def _run():
        results = {"rmsd": {}, "hbonds": {}}
        for name in get_system_names(cfg):
            sys_cfg = get_system(cfg, name)
            stride = get_stride(cfg, name, ANALYSIS_KEY)

            # anchor RMSD
            anchor_sel = get_selection(cfg, name, "anchor_rmsd")
            align_sel = get_selection(cfg, name, "align") or "protein and backbone"
            if anchor_sel:
                print(f"  [{name}] Anchor RMSD (stride={stride})...")
                t_us, rmsd_A = _aligned_roi_rmsd(
                    universes[name], anchor_sel, align_sel, stride=stride
                )
                results["rmsd"][name] = {"time_us": t_us, "rmsd_A": rmsd_A}

            # H-bonds
            hb_cfg = sys_cfg.get("hbonds", {})
            donors = hb_cfg.get("donors")  # None → guess
            acc_lipids = hb_cfg.get("acceptor_lipids", [])
            between = hb_cfg.get("between")  # explicit between list

            if acc_lipids or between:
                print(f"  [{name}] H-bonds (stride={stride})...")
                hb = _hbond_counts(
                    universes[name], between, acc_lipids, stride=stride
                )
                results["hbonds"][name] = hb
            else:
                print(f"  [{name}] No acceptor_lipids or between specified, skipping H-bonds.")

        return results

    return cached_compute(cache, _run, force_recompute=force)


# ─── Lipid colors for H-bond plots ───────────────────────────────────────────
LIPID_COLORS = {
    "TOTAL": "black", "POPC": "tab:blue", "POPS": "tab:orange",
    "POPE": "tab:green", "PLA18": "tab:purple", "PSM": "tab:pink",
    "SAPI": "tab:brown", "CHL1": "tab:red",
}


def plot(cfg, results):
    """Generate anchor-RMSD and H-bond time-series plots per system."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    sim_us = get_sim_length(cfg)
    ma = get_analysis_params(cfg, ANALYSIS_KEY).get("ma_window", 21)

    # anchor RMSD plots
    for name, data in results.get("rmsd", {}).items():
        t = data["time_us"]
        y = data["rmsd_A"]
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        line_plot(t, y, ax, title=name, color="black", z=1,
                  label="ROI RMSD", ma_window=ma, ma_color="red", ma_z=3)
        ax.set_xlabel("Time (μs)", fontsize=16)
        ax.set_ylabel("Anchor RMSD (Å)", fontsize=16)
        save_figure(fig, os.path.join(outdir, f"{name}_anchor_rmsd.png"))

    # H-bond plots
    for name, hb in results.get("hbonds", {}).items():
        t = hb["time_us"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

        # total
        line_plot(t, hb["total"], ax, color="black", z=1,
                  label="Total", ma_window=ma, ma_color="black", ma_z=3)

        # per lipid
        for r, counts in hb["by_lipid"].items():
            c = LIPID_COLORS.get(r, "gray")
            line_plot(t, counts, ax, color=c, z=1,
                      label=r, ma_window=ma, ma_color=c, ma_z=3)

        style_axes(ax, title=name, xlabel="Time (μs)", ylabel="# H-bonds")
        ax.legend(fontsize=10, frameon=False, loc="upper right")
        save_figure(fig, os.path.join(outdir, f"{name}_hbonds.png"))
