"""Unified plotting utilities for publication-quality figures."""

import math
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

COMPARISON_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ── Global style ──────────────────────────────────────────────────────────────

def setup_style(font_family="Arial", dpi=600):
    """Set global matplotlib rcParams for publication-quality output."""
    plt.rcParams["font.family"]     = font_family
    plt.rcParams["axes.linewidth"]  = 1.5
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["savefig.dpi"]  = dpi


# ── Axes styling ──────────────────────────────────────────────────────────────

def style_axes(ax, title=None, xlabel=None, ylabel=None,
               title_fontsize=20, label_fontsize=20, tick_fontsize=16):
    """Apply clean publication-style formatting to an axes object."""
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.5, direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.minorticks_off()
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


# ── Line plot ─────────────────────────────────────────────────────────────────

def line_plot(xvalues, yvalues, ax, title=None, color="black", z=1,
              label=None, ma_window=5, ma_color="red", ma_z=2):
    """Raw data (alpha=0.25) with a centred moving-average overlay.

    The MA line carries the legend label; the raw trace is unlabeled.
    """
    xvalues  = np.asarray(xvalues, dtype=float)
    yvalues  = np.asarray(yvalues, dtype=float)
    mov_avg  = pd.Series(yvalues).rolling(window=ma_window, center=True).mean()

    ax.plot(xvalues, yvalues,       alpha=0.25, color=color,    zorder=z,    label=None)
    ax.plot(xvalues, mov_avg.values, alpha=1.0,  color=ma_color, zorder=ma_z, label=label)

    style_axes(ax, title=title)


# ── Multi-system figure factory ───────────────────────────────────────────────

def multi_system_figure(n_systems, sharex=True, sharey=True, ax_w=5, ax_h=4):
    """Create a grid figure sized for *n_systems* panels.

    Layout: up to 2 columns, as many rows as needed.
    Returns (fig, axes_list) where axes_list has exactly *n_systems* Axes.
    Any trailing unused panel is hidden automatically.
    """
    ncols = min(n_systems, 2)
    nrows = math.ceil(n_systems / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ax_w * ncols, ax_h * nrows),
        sharey=sharey, sharex=sharex,
        constrained_layout=True,
    )
    if n_systems == 1:
        return fig, [axes]
    axes_flat = np.asarray(axes).flatten()
    for ax in axes_flat[n_systems:]:
        ax.set_visible(False)
    return fig, list(axes_flat[:n_systems])


# ── Overlay comparison plot ───────────────────────────────────────────────────

def overlay_line_plot(results_dict, sim_us, ylabel, outpath, ma_window=50,
                      ax_w=8, ax_h=4, time_bounds=None):
    """All systems overlaid on a single axes with unique colors.

    Each system's raw trace is drawn at low alpha; the moving average carries
    the legend label.  Only generates a file when results_dict has > 1 entry.

    Parameters
    ----------
    results_dict : dict {name: 1D array}
    sim_us : float  — fallback simulation length in microseconds for any
             system without per-system bounds.
    ylabel : str
    outpath : str
    ma_window : int
    time_bounds : dict {name: (start_us, end_us)} or None
                  Per-system time-axis bounds.  When provided, each system's
                  trace is drawn against its own analysed window so windows
                  that start at different times line up correctly in
                  absolute simulation time.  Falls back to (0, sim_us) for
                  any missing system.
    """
    if len(results_dict) < 2:
        return

    fig, ax = plt.subplots(figsize=(ax_w, ax_h), constrained_layout=True)
    for i, (name, arr) in enumerate(results_dict.items()):
        color   = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        arr     = np.asarray(arr, dtype=float)
        if time_bounds is not None and name in time_bounds:
            s_us, e_us = time_bounds[name]
        else:
            s_us, e_us = 0.0, float(sim_us)
        xvalues = np.linspace(s_us, e_us, len(arr))
        mov_avg = pd.Series(arr).rolling(window=ma_window, center=True).mean()
        ax.plot(xvalues, arr,            alpha=0.15, color=color, zorder=1)
        ax.plot(xvalues, mov_avg.values, alpha=1.0,  color=color, zorder=2,
                label=name, linewidth=1.5)
    ax.legend(fontsize=12, frameon=False)
    style_axes(ax, xlabel="Time (us)", ylabel=ylabel)
    save_figure(fig, outpath)


# ── Figure saving ─────────────────────────────────────────────────────────────

def save_figure(fig, path, dpi=None, close=True):
    """Save a figure to *path*, creating directories as needed."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(path, **kwargs)
    print(f"  Figure: {path}")
    if close:
        plt.close(fig)
