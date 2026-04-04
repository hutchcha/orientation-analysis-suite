"""Unified plotting utilities for publication-quality figures."""

import math
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


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
