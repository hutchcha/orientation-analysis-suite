"""Configuration loader and helpers."""

import os
import warnings

import yaml
import MDAnalysis as mda


def load_config(path):
    """Load a YAML configuration file and return the dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_output_dir(cfg):
    """Return the global output directory, creating it if needed."""
    out = cfg.get("global", {}).get("output_dir", "./results")
    os.makedirs(out, exist_ok=True)
    return out


def get_sim_length(cfg):
    """Return simulation length in microseconds."""
    return cfg.get("global", {}).get("sim_length_us", 20)


def is_force_recompute(cfg):
    return cfg.get("global", {}).get("force_recompute", False)


def get_style(cfg):
    """Return style settings dict with defaults."""
    defaults = {"font_family": "Arial", "dpi": 600}
    defaults.update(cfg.get("style", {}))
    return defaults


def get_system_names(cfg):
    """Return ordered list of system names."""
    return list(cfg.get("systems", {}).keys())


def get_system(cfg, name):
    """Return the config dict for a single system."""
    return cfg["systems"][name]


def get_selection(cfg, name, sel_key, default=None):
    """Get a specific selection string for a system."""
    sys_cfg = get_system(cfg, name)
    return sys_cfg.get("selections", {}).get(sel_key, default)


def get_stride(cfg, name, analysis_key):
    """
    Get stride for a given system + analysis.
    Priority: system stride_overrides > analysis default stride > 1
    """
    sys_cfg = get_system(cfg, name)
    override = sys_cfg.get("stride_overrides", {}).get(analysis_key)
    if override is not None:
        return int(override)
    # search in analyses.equilibration and analyses.analysis
    analyses = cfg.get("analyses", {})
    for group in ("equilibration", "analysis"):
        group_cfg = analyses.get(group, {})
        if analysis_key in group_cfg:
            return int(group_cfg[analysis_key].get("stride", 1))
    return 1


def is_analysis_enabled(cfg, analysis_key):
    """Check if an analysis is enabled in the config."""
    analyses = cfg.get("analyses", {})
    for group in ("equilibration", "analysis"):
        group_cfg = analyses.get(group, {})
        if analysis_key in group_cfg:
            return group_cfg[analysis_key].get("enabled", False)
    return False


def get_analysis_params(cfg, analysis_key):
    """Return the full parameter dict for an analysis."""
    analyses = cfg.get("analyses", {})
    for group in ("equilibration", "analysis"):
        group_cfg = analyses.get(group, {})
        if analysis_key in group_cfg:
            return group_cfg[analysis_key]
    return {}


def load_stats_config(path):
    """Load a stats YAML configuration file and return the dict."""
    with open(path, "r") as f:
        stats_cfg = yaml.safe_load(f)
    return stats_cfg


def get_feature_set(stats_cfg, name):
    """Return a named feature_set dict from the stats config.

    Parameters
    ----------
    stats_cfg : dict — parsed stats YAML
    name      : str  — feature set name (e.g. "orientation_3d")

    Returns
    -------
    dict with keys: features (list), transform (str), radial_scale (float), ...
    """
    sets = stats_cfg.get("feature_sets", {})
    if name not in sets:
        raise KeyError(
            f"Feature set '{name}' not found in stats config. "
            f"Available: {list(sets.keys())}"
        )
    return sets[name]


def get_stats_params(stats_cfg, module_key):
    """Return the config block for a statistical module (clustering, gmm, kinetics)."""
    return stats_cfg.get(module_key, {})


def build_universe(cfg, name):
    """Create an MDAnalysis Universe for a named system."""
    sys_cfg = get_system(cfg, name)
    top = sys_cfg["topology"]
    traj = sys_cfg["trajectory"]
    print(f"  Loading universe: {name}")
    return mda.Universe(top, traj)


def build_all_universes(cfg):
    """Build all universes. Returns dict {name: Universe}."""
    universes = {}
    for name in get_system_names(cfg):
        universes[name] = build_universe(cfg, name)
    return universes


# ── Analysis window helpers ──────────────────────────────────────────────────
#
# A per-system YAML block selects a sub-range of the trajectory to analyse:
#
#   systems:
#     RhoA-GTP:
#       analysis_window:
#         start_us: 1.0          # skip the first 1 us
#         stop_us:  null         # null = run to end of trajectory
#
# By default the window applies to every analysis.  A group-level or
# per-analysis ``ignore_window: true`` flag opts an analysis out
# (typically used for equilibration plots that should show the full
# trajectory including the equilibration phase):
#
#   analyses:
#     equilibration:
#       ignore_window: true       # all equilibration modules show full traj
#       rmsd:
#         enabled: true
#         ignore_window: false    # ...except this one, which respects window
#

def get_analysis_window(cfg, name):
    """Return the per-system analysis_window dict, or None if not configured."""
    sys_cfg = get_system(cfg, name)
    win = sys_cfg.get("analysis_window")
    if not win:
        return None
    return win


def _frame_to_us(frame_idx, n_frames, full_sim_length_us):
    """Convert a frame index to simulation time in microseconds."""
    if n_frames <= 1:
        return 0.0
    return (frame_idx / (n_frames - 1)) * full_sim_length_us


def _us_to_frame(t_us, n_frames, full_sim_length_us):
    """Convert a simulation time in microseconds to a frame index."""
    if full_sim_length_us <= 0 or n_frames <= 1:
        return 0
    return int(round((t_us / full_sim_length_us) * (n_frames - 1)))


def get_frame_window(cfg, name, universe):
    """Resolve analysis_window into (start_frame, stop_frame) frame indices.

    ``stop_frame`` is exclusive (Python slice convention).  If no window is
    configured for this system, returns (0, n_frames).  Out-of-range values
    are clipped with a warning.
    """
    n_frames = int(universe.trajectory.n_frames)
    full_us  = float(get_sim_length(cfg))
    win      = get_analysis_window(cfg, name)

    if not win:
        return 0, n_frames

    start_us = win.get("start_us")
    stop_us  = win.get("stop_us")

    if start_us is None and stop_us is None:
        return 0, n_frames

    if start_us is not None and stop_us is not None and start_us >= stop_us:
        raise ValueError(
            f"[{name}] analysis_window: start_us ({start_us}) must be "
            f"strictly less than stop_us ({stop_us})."
        )

    start_f = 0 if start_us is None else _us_to_frame(start_us, n_frames, full_us)
    stop_f  = n_frames if stop_us is None else (_us_to_frame(stop_us, n_frames, full_us) + 1)

    # Clip with warnings
    if start_f < 0:
        warnings.warn(f"[{name}] start_us={start_us} < 0; clipping to frame 0.")
        start_f = 0
    if start_f >= n_frames:
        warnings.warn(
            f"[{name}] start_us={start_us} is past end of trajectory "
            f"(sim_length={full_us} us, n_frames={n_frames}); clipping to last frame."
        )
        start_f = n_frames - 1
    if stop_f > n_frames:
        warnings.warn(
            f"[{name}] stop_us={stop_us} is past end of trajectory "
            f"(sim_length={full_us} us, n_frames={n_frames}); clipping to end."
        )
        stop_f = n_frames
    if stop_f <= start_f:
        warnings.warn(
            f"[{name}] analysis_window has zero or negative length after clipping; "
            f"using single-frame window at start_f={start_f}."
        )
        stop_f = start_f + 1

    return int(start_f), int(stop_f)


def _find_analysis_group_cfg(cfg, analysis_key):
    """Return (group_name, group_cfg) where the analysis lives, or (None, None)."""
    analyses = cfg.get("analyses", {}) or {}
    for group_name, group_cfg in analyses.items():
        if isinstance(group_cfg, dict) and analysis_key in group_cfg:
            return group_name, group_cfg
    return None, None


def should_apply_window(cfg, analysis_key):
    """Whether the analysis_window applies to this analysis.

    Precedence (first match wins):
      1. Per-analysis ``ignore_window`` (under ``analyses.<group>.<key>``)
      2. Group-level ``ignore_window`` (under ``analyses.<group>``)
      3. Default: True (apply window)
    """
    _, group_cfg = _find_analysis_group_cfg(cfg, analysis_key)
    if group_cfg is None:
        return True

    ana_cfg = group_cfg.get(analysis_key, {})
    if isinstance(ana_cfg, dict) and "ignore_window" in ana_cfg:
        return not bool(ana_cfg["ignore_window"])

    if "ignore_window" in group_cfg:
        return not bool(group_cfg["ignore_window"])

    return True


def get_frame_window_for_analysis(cfg, name, universe, analysis_key):
    """(start, stop) frame indices honouring per-analysis ignore_window flags.

    If the window doesn't apply to this analysis, returns the full
    trajectory range.
    """
    if not should_apply_window(cfg, analysis_key):
        return 0, int(universe.trajectory.n_frames)
    return get_frame_window(cfg, name, universe)


def get_effective_window_us(cfg, name, universe, analysis_key):
    """(start_us, stop_us) of the analysed window in absolute simulation time.

    Honours per-analysis ignore_window.  Returns the bounds suitable for the
    plot x-axis: time values along the analysed range, in microseconds of
    simulation time (NOT relative to the window start).
    """
    full_us  = float(get_sim_length(cfg))
    n_frames = int(universe.trajectory.n_frames)
    start_f, stop_f = get_frame_window_for_analysis(cfg, name, universe, analysis_key)
    start_us = _frame_to_us(start_f, n_frames, full_us)
    # stop_f is exclusive — last analysed frame is stop_f - 1
    end_us   = _frame_to_us(max(start_f, stop_f - 1), n_frames, full_us)
    return start_us, end_us


def get_effective_sim_length_us(cfg, name, universe, analysis_key):
    """Duration of the analysed window in microseconds (for time-axis spans)."""
    start_us, end_us = get_effective_window_us(cfg, name, universe, analysis_key)
    return max(0.0, end_us - start_us)


# ── Cache metadata ───────────────────────────────────────────────────────────

def build_cache_metadata(cfg, universes, analysis_key):
    """Build the metadata dict to embed in this analysis's cache file.

    Per-system metadata captures: window resolution, stride, analysis params,
    system selections, and the cached time-axis bounds (so plot-only mode
    can rebuild the time axis without loading universes).
    """
    _, group_cfg = _find_analysis_group_cfg(cfg, analysis_key)
    analysis_params = {}
    group_options   = {}
    if group_cfg is not None:
        ana_block = group_cfg.get(analysis_key, {})
        if isinstance(ana_block, dict):
            analysis_params = dict(ana_block)
        # group-level options excluding nested analysis dicts
        group_options = {k: v for k, v in group_cfg.items()
                         if not (isinstance(v, dict) and "enabled" in v)}

    per_system = {}
    for name in get_system_names(cfg):
        sys_cfg = get_system(cfg, name)
        u = universes.get(name) if universes is not None else None

        if u is not None:
            start_f, stop_f = get_frame_window_for_analysis(cfg, name, u, analysis_key)
            start_us, end_us = get_effective_window_us(cfg, name, u, analysis_key)
            n_frames = int(u.trajectory.n_frames)
        else:
            start_f, stop_f, start_us, end_us, n_frames = None, None, None, None, None

        per_system[name] = {
            "window_cfg":     get_analysis_window(cfg, name),
            "start_frame":    start_f,
            "stop_frame":     stop_f,
            "start_us":       start_us,
            "end_us":         end_us,
            "n_frames_total": n_frames,
            "stride":         get_stride(cfg, name, analysis_key),
            "selections":     dict(sys_cfg.get("selections", {})),
            "ignored_window": not should_apply_window(cfg, analysis_key),
        }

    return {
        "analysis_key":    analysis_key,
        "full_sim_length_us": float(get_sim_length(cfg)),
        "analysis_params": analysis_params,
        "group_options":   group_options,
        "per_system":      per_system,
    }
