"""Configuration loader and helpers."""

import os
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
