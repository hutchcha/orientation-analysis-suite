"""Protein tilt and rotation orientation analysis.

Tilt and rotation are defined relative to the membrane normal using a specific
protein helix axis vector (typically α5) and a secondary reference group
(typically the α2 helix / switch II region) that acts as a rotational pointer.

Tilt
----
Angle between the helix axis vector and the membrane normal. Range: [0°, 180°].
0° = axis parallel to and pointing along the normal (straight up from membrane).
90° = axis parallel to membrane plane.

Rotation
--------
Following Travers et al. (Sci Rep 8, 8461, 2018):
  - z' = helix axis direction (current frame)
  - S  = plane perpendicular to z', passing through the rotation reference
         group COM (e.g. H2/switch-II)
  - p  = projection of the membrane normal (z) onto plane S
  - a  = vector in S from the helix endpoint to the reference group COM

theta_r is the directed angle from a to p around z'.  This definition uses
the membrane normal as the rotation reference, so it is well-defined
regardless of whether the protein ever samples low tilt angles.

Range: (−180°, +180°].

Config selections required per system
--------------------------------------
  orientation_axis_start   : C-terminal end of α5 (e.g. "protein and resid 177 and name CA")
  orientation_axis_end     : N-terminal end of α5 (e.g. "protein and resid 169 and name CA")
  rotation_reference       : pointer group COM (e.g. "protein and resid 89:93")
  phosphorus               : (optional) for membrane-normal estimation

Membrane normal
---------------
Estimated from upper/lower leaflet phosphorus COMs at frame 0 (upper − lower),
giving the normal pointing from lower to upper leaflet.  Falls back to +Z if
no phosphorus selection is provided or the leaflets cannot be resolved.

Auto-flip
---------
At frame 0 the axis vector is checked against the estimated normal.  If they are
antiparallel (dot product < 0), start and end selections are swapped so that
the axis consistently points in the same half-space as the normal, and tilt
angles remain physically interpretable.
"""

import os
from math import atan2

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt

from membrane_analysis.core.io import (
    cached_compute, save_txt, save_per_system,
    load_cache_metadata, get_time_bounds,
)
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_selection,
    get_stride, get_sim_length, is_force_recompute, get_analysis_params,
    get_frame_window_for_analysis, build_cache_metadata,
)
from membrane_analysis.core.plotting import (
    line_plot, style_axes, save_figure, multi_system_figure, overlay_line_plot,
)


ANALYSIS_KEY = "tilt_rotation"
OUTPUT_TYPE = "angular"
OUTPUT_FIELDS = {"tilt": "angular", "rotation": "angular"}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _project_onto_plane(axis, vector):
    """Project *vector* onto the plane perpendicular to *axis*.

    Equivalent to removing the component of *vector* along *axis*:
        v_proj = v − (v·n̂ / |n|²) n
    """
    return vector - (np.dot(axis, vector) / np.dot(axis, axis)) * axis


def _directed_angle_deg(v1, v2, normal):
    """Signed angle from *v1* to *v2* around *normal* (degrees).

    Uses atan2 so the result covers (−180°, +180°] and is well-behaved at 0°
    and ±180°.  *v1* and *v2* must be unit vectors in the plane perpendicular
    to *normal*.
    """
    sin_a = np.dot(np.cross(v1, v2), normal)
    cos_a = np.dot(v1, v2)
    return -np.rad2deg(atan2(sin_a, cos_a))


# ── Membrane normal ───────────────────────────────────────────────────────────

def estimate_membrane_normal(u, phosphorus_sel):
    """Estimate the membrane normal from leaflet phosphorus COMs at frame 0.

    Returns a unit vector pointing from the lower to the upper leaflet (i.e.
    along +Z for a standard CHARMM-GUI bilayer).  Falls back to +Z if the
    selection is absent or the leaflets cannot be resolved.
    """
    fallback = np.array([0.0, 0.0, 1.0])
    if not phosphorus_sel:
        return fallback

    u.trajectory[0]
    ag = u.select_atoms(phosphorus_sel)
    if ag.n_atoms < 2:
        return fallback

    mid_z = ag.center_of_mass()[2]
    upper = u.select_atoms(f"({phosphorus_sel}) and prop z > {mid_z}")
    lower = u.select_atoms(f"({phosphorus_sel}) and prop z < {mid_z}")
    if upper.n_atoms == 0 or lower.n_atoms == 0:
        return fallback

    v = upper.center_of_mass() - lower.center_of_mass()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else fallback


# ── Core angle computation ────────────────────────────────────────────────────

def compute_angles(u, start_sel, end_sel, ref_sel, normal,
                   start=0, stop=None, stride=1):
    """Compute per-frame tilt and rotation angles over a frame window.

    Tilt (theta_t)
        Angle between the helix axis unit vector (z') and the membrane
        normal (z).  Range: [0, 180] degrees.

    Rotation (theta_r)
        Following Travers et al. (2018):
        - z' = helix axis direction (current frame)
        - S  = plane perpendicular to z', passing through H2COM
        - p  = projection of the membrane normal z onto plane S
        - a  = vector in S from the z' axis to H2COM (pointer vector)
        - theta_r = directed angle from a to p around z'

        The membrane normal serves as the rotation reference, so rotation=0
        means the reference group points in the same in-plane direction as
        the membrane normal's projection.  This is well-defined regardless
        of whether the protein ever samples low tilt.

    Parameters
    ----------
    u : MDAnalysis.Universe
    start_sel : str   — C-terminal anchor of the helix axis
    end_sel : str     — N-terminal anchor of the helix axis
    ref_sel : str     — rotation pointer group (e.g. H2/switch-II)
    normal : (3,) ndarray — membrane normal unit vector
    start, stop : int    — frame slice (stop exclusive)
    stride : int

    Returns
    -------
    tilt_deg, rotation_deg : (N,) ndarray
    """
    if stop is None:
        stop = int(u.trajectory.n_frames)

    start_ag = u.select_atoms(start_sel)
    end_ag   = u.select_atoms(end_sel)
    ref_ag   = u.select_atoms(ref_sel)

    # ── Pass 1: tilt + axis vectors ──────────────────────────────────────────
    axis_vectors = []
    tilts = []

    for _ts in tqdm(u.trajectory[start:stop:stride], desc="  tilt"):
        com_start = start_ag.center_of_mass()
        com_end   = end_ag.center_of_mass()
        v = com_end - com_start
        mag = np.linalg.norm(v)
        if mag < 1e-12:
            axis_vectors.append(np.full(3, np.nan))
            tilts.append(np.nan)
            continue
        v_unit = v / mag
        tilt = np.rad2deg(np.arccos(np.clip(np.dot(v_unit, normal), -1.0, 1.0)))
        axis_vectors.append(v)
        tilts.append(tilt)

    tilts        = np.array(tilts)
    axis_vectors = np.array(axis_vectors)  # shape (N, 3), raw (un-normalised)

    # Auto-flip: if first analysed frame's axis is antiparallel to the normal,
    # reverse convention.
    v0 = axis_vectors[0]
    if np.isfinite(v0).all() and np.dot(v0, normal) < 0:
        axis_vectors = -axis_vectors
        tilts = 180.0 - tilts
        start_sel, end_sel = end_sel, start_sel
        start_ag, end_ag = end_ag, start_ag
        print("    Axis auto-flipped: first analysed frame's vector was "
              "antiparallel to normal.")

    # ── Pass 2: rotation ─────────────────────────────────────────────────────
    # Reference vector = membrane normal (projected onto each frame's
    # helix-perpendicular plane).  This replaces the old min-tilt-frame
    # reference and is well-defined for any tilt angle.
    rots = []

    for f_idx, _ts in enumerate(tqdm(u.trajectory[start:stop:stride],
                                     desc="  rotation")):
        a5vec = axis_vectors[f_idx]
        if not np.isfinite(a5vec).all():
            rots.append(np.nan)
            continue

        com_end = end_ag.center_of_mass()
        com_ref = ref_ag.center_of_mass()
        h2vec   = com_ref - com_end  # pointer (a): axis tip -> reference group

        # p = projection of membrane normal onto plane perp to z'
        proj_normal = _project_onto_plane(a5vec, normal)
        # a = projection of pointer onto same plane
        proj_h2     = _project_onto_plane(a5vec, h2vec)

        n_norm = np.linalg.norm(proj_normal)
        n_h2   = np.linalg.norm(proj_h2)
        if n_norm < 1e-12 or n_h2 < 1e-12:
            rots.append(np.nan)
            continue

        axis_unit = a5vec / np.linalg.norm(a5vec)
        rot = _directed_angle_deg(
            proj_h2     / n_h2,     # a (pointer)
            proj_normal / n_norm,   # p (normal projection)
            axis_unit,
        )
        rots.append(rot)

    return tilts, np.array(rots)


# ── Top-level compute ─────────────────────────────────────────────────────────

def compute(cfg, universes):
    """Compute tilt/rotation angles for all systems.

    Returns
    -------
    dict : {name: {"tilt": ndarray, "rotation": ndarray,
                   "membrane_normal": ndarray}}
    """
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "tilt_rotation.pkl")
    force  = is_force_recompute(cfg)
    metadata = build_cache_metadata(cfg, universes, ANALYSIS_KEY)

    def _run():
        results = {}
        for name in get_system_names(cfg):
            from membrane_analysis.core.config import get_system
            sys_cfg   = get_system(cfg, name)
            start_sel = get_selection(cfg, name, "orientation_axis_start")
            end_sel   = get_selection(cfg, name, "orientation_axis_end")
            ref_sel   = get_selection(cfg, name, "rotation_reference")
            phos_sel  = get_selection(cfg, name, "phosphorus")

            if not start_sel or not end_sel:
                print(f"  [{name}] Missing orientation_axis selections, skipping.")
                continue
            if not ref_sel:
                print(f"  [{name}] Missing rotation_reference selection, skipping.")
                continue

            u = universes[name]

            # Manual override: membrane_normal: [x, y, z] in per-system config
            manual_normal = sys_cfg.get("membrane_normal")
            if manual_normal is not None:
                v = np.array(manual_normal, dtype=float)
                normal = v / np.linalg.norm(v)
                print(f"  [{name}] Using manual membrane normal: "
                      f"({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
            else:
                normal = estimate_membrane_normal(u, phos_sel)
                print(f"  [{name}] Estimated membrane normal: "
                      f"({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")

            stride = get_stride(cfg, name, ANALYSIS_KEY)
            start, stop = get_frame_window_for_analysis(cfg, name, u, ANALYSIS_KEY)
            print(f"  [{name}] Computing tilt/rotation "
                  f"(frames {start}:{stop}:{stride})...")
            tilt, rot = compute_angles(
                u, start_sel, end_sel, ref_sel, normal,
                start=start, stop=stop, stride=stride,
            )

            save_txt(tilt, os.path.join(outdir, f"{name}_tilt.txt"))
            save_txt(rot,  os.path.join(outdir, f"{name}_rotation.txt"))

            results[name] = {
                "tilt":             tilt,
                "rotation":         rot,
                "membrane_normal":  normal,
            }
        save_per_system(results, outdir, ANALYSIS_KEY, metadata=metadata)
        return results

    return cached_compute(cache, _run, force_recompute=force, metadata=metadata)


# ── Polar density plot ────────────────────────────────────────────────────────

def polar_density_plot(rot_deg, tilt_deg, ax,
                       num_theta_bins=360, num_radius_bins=100,
                       sigma=(5, 2), cmap="cividis",
                       tilt_max_deg=180):
    """Smoothed polar density histogram on a pre-created polar axes.

    Uses a 2-D histogram with circular padding in θ (rotation), Gaussian
    smoothing, and edge blending to produce a seamless wrap at ±180°.
    The radial axis spans [0°, tilt_max_deg].

    Parameters
    ----------
    rot_deg, tilt_deg : array-like
    ax : matplotlib PolarAxes
    num_theta_bins, num_radius_bins : int
    sigma : (float, float)  —  Gaussian sigma for (theta, radius)
    cmap : str or Colormap
    tilt_max_deg : float  — radial extent (180 to allow full range)
    """
    theta_rad = np.deg2rad(rot_deg)
    tilt_rad  = np.deg2rad(tilt_deg)

    theta_bins  = np.linspace(-np.pi, np.pi,          num_theta_bins  + 1)
    radius_bins = np.linspace(0,       np.deg2rad(tilt_max_deg), num_radius_bins + 1)

    H, _, _ = np.histogram2d(theta_rad, tilt_rad, bins=[theta_bins, radius_bins])

    pad      = 20
    H_pad    = np.pad(H, ((pad, pad), (0, 0)), mode="wrap")
    H_smooth = gaussian_filter(H_pad, sigma=sigma, mode="wrap")
    H_smooth = H_smooth[pad:-pad, :].T           # (radius, theta)

    theta_centers  = (theta_bins[:-1]  + theta_bins[1:])  / 2
    radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
    Theta, Radius  = np.meshgrid(theta_centers, radius_centers)

    H_smooth /= H_smooth.max() if H_smooth.max() > 0 else 1.0
    H_smooth[:, 0]  = (H_smooth[:, 0] + H_smooth[:, -1]) / 2
    H_smooth[:, -1] = H_smooth[:, 0]

    levels = np.linspace(0.02, 1.0, 11)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0\u00b0", "45\u00b0", "90\u00b0", "135\u00b0", "180\u00b0",
                        "-135\u00b0", "-90\u00b0", "-45\u00b0"])
    ax.set_yticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
    ax.set_yticklabels(["", "30", "", "90", "", "150", ""])
    ax.tick_params(axis="x", labelsize=14, pad=6)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(linestyle="-", color="black", alpha=0.5, zorder=1)
    ax.contourf(Theta, Radius, H_smooth, levels=levels, cmap=cmap, zorder=0)


# ── Top-level plot ────────────────────────────────────────────────────────────

def plot(cfg, results):
    """Plot polar density and tilt/rotation time series. Multi-system: shared axes."""
    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "tilt_rotation.pkl")
    sim_us = get_sim_length(cfg)
    params = get_analysis_params(cfg, ANALYSIS_KEY)
    ma     = params.get("ma_window", 500)
    names  = list(results.keys())
    meta   = load_cache_metadata(cache)
    time_bounds = {n: get_time_bounds(meta, n, sim_us) for n in names}

    # ── polar density: one panel per system in a grid ─────────────────────────
    import math
    ncols = min(len(names), 2)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             subplot_kw={"projection": "polar"},
                             constrained_layout=True)
    axes_flat = [axes] if len(names) == 1 else list(np.asarray(axes).flatten())
    for ax in axes_flat[len(names):]:
        ax.set_visible(False)

    for ax, name in zip(axes_flat, names):
        polar_density_plot(results[name]["rotation"], results[name]["tilt"], ax)
        ax.set_title(name, fontsize=14, pad=20)
    save_figure(fig, os.path.join(outdir, "polar_density_all.png"))

    # ── tilt time series ───────────────────────────────────────────────────────
    fig_t, axes_t = multi_system_figure(len(names), sharex=True, sharey=True)
    for ax, name in zip(axes_t, names):
        s_us, e_us = time_bounds[name]
        time = np.linspace(s_us, e_us, len(results[name]["tilt"]))
        line_plot(time, results[name]["tilt"], ax, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
    fig_t.supxlabel("Time (μs)", fontsize=20)
    fig_t.supylabel("Tilt (°)", fontsize=20)
    save_figure(fig_t, os.path.join(outdir, "tilt_timeseries_all.png"))

    # ── rotation time series ───────────────────────────────────────────────────
    fig_r, axes_r = multi_system_figure(len(names), sharex=True, sharey=True)
    for ax, name in zip(axes_r, names):
        s_us, e_us = time_bounds[name]
        time = np.linspace(s_us, e_us, len(results[name]["rotation"]))
        line_plot(time, results[name]["rotation"], ax, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
    fig_r.supxlabel("Time (μs)", fontsize=20)
    fig_r.supylabel("Rotation (°)", fontsize=20)
    save_figure(fig_r, os.path.join(outdir, "rotation_timeseries_all.png"))

    # ── comparison overlays ────────────────────────────────────────────────────
    overlay_line_plot({n: results[n]["tilt"]     for n in names}, sim_us,
                      "Tilt (°)",
                      os.path.join(outdir, "tilt_comparison.png"),
                      ma_window=ma, time_bounds=time_bounds)
    overlay_line_plot({n: results[n]["rotation"] for n in names}, sim_us,
                      "Rotation (°)",
                      os.path.join(outdir, "rotation_comparison.png"),
                      ma_window=ma, time_bounds=time_bounds)

    # Per-system individual plots
    for name in names:
        s_us, e_us = time_bounds[name]
        # Polar density
        fig_p, ax_p = plt.subplots(figsize=(5, 5),
                                   subplot_kw={"projection": "polar"},
                                   constrained_layout=True)
        polar_density_plot(results[name]["rotation"], results[name]["tilt"], ax_p)
        ax_p.set_title(name, fontsize=14, pad=20)
        save_figure(fig_p, os.path.join(outdir, name, "polar_density.png"))

        # Tilt time series
        fig_t_s, ax_t_s = plt.subplots(figsize=(5, 4), constrained_layout=True)
        time = np.linspace(s_us, e_us, len(results[name]["tilt"]))
        line_plot(time, results[name]["tilt"], ax_t_s, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
        ax_t_s.set_xlabel("Time (μs)", fontsize=14)
        ax_t_s.set_ylabel("Tilt (°)", fontsize=14)
        save_figure(fig_t_s, os.path.join(outdir, name, "tilt_timeseries.png"))

        # Rotation time series
        fig_r_s, ax_r_s = plt.subplots(figsize=(5, 4), constrained_layout=True)
        time = np.linspace(s_us, e_us, len(results[name]["rotation"]))
        line_plot(time, results[name]["rotation"], ax_r_s, title=name,
                  color="black", z=1, ma_window=ma, ma_color="red", ma_z=2)
        ax_r_s.set_xlabel("Time (μs)", fontsize=14)
        ax_r_s.set_ylabel("Rotation (°)", fontsize=14)
        save_figure(fig_r_s, os.path.join(outdir, name, "rotation_timeseries.png"))
