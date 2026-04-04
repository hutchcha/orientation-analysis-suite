"""Residue-lipid contact frequency (RCF) analysis.

Computes the fraction of sampled frames in which each protein residue is
within *cutoff* Angstroms of any lipid heavy atom.  Computation is
parallelised over trajectory chunks using multiprocessing.

Config fields
-------------
Per-system selections:
  protein_heavy  — protein heavy-atom selection
                   e.g. "protein and not name H*"
  lipid_heavy    — lipid heavy-atom selection
                   e.g. "resname POPC POPE POPS CHL1 and not name H*"

Analysis params (under analyses.analysis.contacts):
  cutoff        : float  — contact distance in Angstroms (default 4.0)
  n_jobs        : int    — worker processes; -1 = cpu_count - 1 (default -1)
  chunk_size    : int    — frames per worker task (default 200)
  regions       : list of dicts, each with keys label/start/end/color
                  (optional; structural annotation bands on the bar chart)
  residue_markers : dict {system_name: [[label, resid], ...]}
                  (optional; per-system tick+label markers above specific bars)
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import capped_distance

from membrane_analysis.core.io import cached_compute
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_system, get_selection,
    get_stride, is_force_recompute, get_analysis_params,
)
from membrane_analysis.core.plotting import style_axes, save_figure


ANALYSIS_KEY = "contacts"
OUTPUT_TYPE = "per_residue"

# ── Multiprocessing worker ────────────────────────────────────────────────────

def _chunk_contact_counts(args):
    """Count per-residue contact frames for a chunk of trajectory frames.

    Returns
    -------
    contacted_counts : ndarray (n_res,)
    n_frames         : int
    unique_resids    : ndarray (n_res,)
    """
    top, traj, cutoff, frame_indices, prot_sel, lipid_sel = args

    u = mda.Universe(top, traj)
    prot = u.select_atoms(prot_sel)
    lip  = u.select_atoms(lipid_sel)

    prot_resids = prot.resids
    unique_resids = np.unique(prot_resids)
    resid_to_idx = {int(rid): i for i, rid in enumerate(unique_resids)}
    prot_atom_res_idx = np.array(
        [resid_to_idx[int(rid)] for rid in prot_resids], dtype=np.int32
    )

    n_res = unique_resids.size
    contacted_counts = np.zeros(n_res, dtype=np.int64)
    n_frames = 0

    for fi in frame_indices:
        u.trajectory[fi]
        pairs = capped_distance(
            prot.positions, lip.positions,
            max_cutoff=cutoff, box=u.dimensions, return_distances=False,
        )
        if pairs is not None and len(pairs) > 0:
            if isinstance(pairs, tuple):
                prot_atom_idx = pairs[0]
            else:
                pairs = np.asarray(pairs)
                prot_atom_idx = pairs[:, 0]
            res_idx = prot_atom_res_idx[prot_atom_idx]
            contacted_counts[np.unique(res_idx)] += 1
        n_frames += 1

    return contacted_counts, n_frames, unique_resids


# ── Per-system RCF computation ────────────────────────────────────────────────

def _compute_one(u0, top, traj, prot_sel, lipid_sel,
                 stride, cutoff, n_jobs, chunk_size):
    """Return RCF DataFrame for one trajectory."""
    prot0    = u0.select_atoms(prot_sel)
    meta     = {
        int(r.resid): (r.resname, f"{r.resname}-{int(r.resid)}")
        for r in prot0.residues
    }

    all_frames = np.arange(0, u0.trajectory.n_frames, stride, dtype=np.int64)
    if all_frames.size == 0:
        raise ValueError(
            f"No frames selected (n_frames={u0.trajectory.n_frames}, stride={stride})."
        )

    chunks    = [all_frames[i:i + chunk_size]
                 for i in range(0, all_frames.size, chunk_size)]
    args_list = [(top, traj, cutoff, chunk, prot_sel, lipid_sel)
                 for chunk in chunks]

    if n_jobs == 1:
        chunk_results = [_chunk_contact_counts(a) for a in args_list]
    else:
        with Pool(processes=n_jobs) as pool:
            chunk_results = pool.map(_chunk_contact_counts, args_list)

    total_counts = None
    total_frames = 0
    worker_resids = chunk_results[0][2]

    for counts, n_frames, _ in chunk_results:
        total_frames += n_frames
        total_counts = counts.copy() if total_counts is None else total_counts + counts

    rcf = total_counts / float(total_frames)

    out_resids   = worker_resids.astype(int)
    out_resnames = [meta.get(int(r), ("UNK", f"UNK-{int(r)}"))[0] for r in out_resids]
    out_labels   = [meta.get(int(r), ("UNK", f"UNK-{int(r)}"))[1] for r in out_resids]

    return pd.DataFrame({
        "resid":            out_resids,
        "resname":          out_resnames,
        "label":            out_labels,
        "rcf":              rcf,
        "n_frames_sampled": int(total_frames),
    })


# ── Pipeline interface ────────────────────────────────────────────────────────

def compute(cfg, universes):
    """Compute RCF for all systems. Returns dict {name: DataFrame}."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "contacts.pkl")
    force  = is_force_recompute(cfg)

    params     = get_analysis_params(cfg, ANALYSIS_KEY)
    cutoff     = float(params.get("cutoff", 4.0))
    chunk_size = int(params.get("chunk_size", 200))
    n_jobs_cfg = params.get("n_jobs", -1)
    n_jobs     = max(1, cpu_count() - 1) if n_jobs_cfg == -1 else max(1, int(n_jobs_cfg))

    def _run():
        results = {}
        for name in get_system_names(cfg):
            prot_sel  = get_selection(cfg, name, "protein_heavy")
            lipid_sel = get_selection(cfg, name, "lipid_heavy")
            if prot_sel is None or lipid_sel is None:
                print(f"  [{name}] Missing protein_heavy or lipid_heavy selection, skipping.")
                continue
            sys_cfg = get_system(cfg, name)
            top  = sys_cfg["topology"]
            traj = sys_cfg["trajectory"]
            stride = get_stride(cfg, name, ANALYSIS_KEY)
            print(f"  [{name}] Computing RCF "
                  f"(stride={stride}, cutoff={cutoff}A, n_jobs={n_jobs})...")
            results[name] = _compute_one(
                universes[name], top, traj,
                prot_sel, lipid_sel, stride, cutoff, n_jobs, chunk_size,
            )
        return results

    return cached_compute(cache, _run, force_recompute=force)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _add_bands(ax, regions, alpha=0.22):
    """Shade structural annotation regions; return legend handles."""
    handles = []
    for r in regions:
        ax.axvspan(r["start"], r["end"],
                   ymin=0, ymax=1, facecolor=r["color"], alpha=alpha, edgecolor=None)
        handles.append(mpatches.Patch(facecolor=r["color"], alpha=alpha, label=r["label"]))
    return handles


def _add_band_labels(ax, regions, fontsize=9):
    """Place region name text centered above each shaded band (axes-y coordinates)."""
    for r in regions:
        x_mid = 0.5 * (r["start"] + r["end"])
        ax.text(x_mid, 1.05, r["label"],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=fontsize, color=r["color"],
                clip_on=False)


def _add_residue_markers(ax, resid_to_rcf, markers, line_frac=0.06, text_frac=0.02):
    """Vertical tick + label above specific residue bars."""
    y0, y1 = ax.get_ylim()
    yr     = y1 - y0
    line_h = line_frac * yr
    pad    = text_frac  * yr

    for label, rid in markers:
        if rid not in resid_to_rcf:
            continue
        y       = float(resid_to_rcf[rid])
        y_start = y + 0.01 * yr
        y_end   = y_start + line_h
        ax.plot([rid, rid], [y_start, y_end], linewidth=1.0, color="black", clip_on=False)
        ax.text(rid, y_end + pad, label,
                ha="center", va="bottom", fontsize=9, color="black", clip_on=False)


# ── Top-level plot ────────────────────────────────────────────────────────────

def plot(cfg, results):
    """Plot RCF bar charts, one row per system, with optional structural bands."""
    outdir  = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    params  = get_analysis_params(cfg, ANALYSIS_KEY)
    regions = params.get("regions", [])
    markers_cfg = params.get("residue_markers", {})
    names   = list(results.keys())
    nrows   = len(names)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(10.0, 3.5 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]

    all_rcf     = pd.concat(results.values())
    max_resid   = int(all_rcf["resid"].max())
    global_ymax = max(1.0, float(all_rcf["rcf"].max()) + 0.35)

    for i, (ax, name) in enumerate(zip(axes, names)):
        df = results[name].sort_values("resid")
        x  = df["resid"].to_numpy()
        y  = df["rcf"].to_numpy()

        ax.set_ylim(0, global_ymax)

        if regions:
            _add_bands(ax, regions)
            if i == 0:
                _add_band_labels(ax, regions)

        ax.bar(x, y, color="black", width=1)

        # system label on the right outside the axes
        ax.text(1.012, 0.50, name,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=11, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black"))

        markers = markers_cfg.get(name, [])
        if markers:
            resid_to_rcf = {int(rid): float(val)
                            for rid, val in zip(df["resid"], df["rcf"])}
            _add_residue_markers(ax, resid_to_rcf, markers)

        ax.grid(axis="y", linestyle=":")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.minorticks_off()
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=3.5, direction="out")
        ax.set_yticks(np.arange(0, global_ymax + 0.01, 0.5))
        ax.set_xlim(0, max_resid + 5)

    fig.supxlabel("Residue", fontsize=20)
    fig.supylabel("RCF", fontsize=20)
    save_figure(fig, os.path.join(outdir, "contacts_all.png"))
