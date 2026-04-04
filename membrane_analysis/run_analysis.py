#!/usr/bin/env python3
"""
Membrane analysis pipeline — main entry point.

Usage:
    python run_analysis.py config.yaml                     # run all enabled analyses
    python run_analysis.py config.yaml --mode equilibration  # equilibration only
    python run_analysis.py config.yaml --mode analysis       # analysis only
    python run_analysis.py config.yaml --plot-only           # re-plot from cached data
    python run_analysis.py config.yaml --recompute rmsd apl  # force recompute specific
"""

import argparse
import sys
import os

from membrane_analysis.core.config import (
    load_config, load_stats_config, is_analysis_enabled, get_style,
    get_output_dir, get_system_names, get_feature_set, get_stats_params,
    build_all_universes,
)
from membrane_analysis.core.plotting import setup_style
from membrane_analysis.core.features import assemble_features

# ─── Registry of analysis modules ─────────────────────────────────────────────
# Maps analysis_key → (module, group)
EQUILIBRATION_MODULES = {
    "rmsd":             "membrane_analysis.analyses.rmsd",
    "pp_distance":      "membrane_analysis.analyses.pp_distance",
    "apl":              "membrane_analysis.analyses.apl",
    "anchor_insertion":  "membrane_analysis.analyses.anchor_insertion",
}

ANALYSIS_MODULES = {
    "tilt_rotation":           "membrane_analysis.analyses.tilt_rotation",
    "lobe_com":                "membrane_analysis.analyses.lobe_com",
    "hbonds":                  "membrane_analysis.analyses.hbonds",
    "clustering":              "membrane_analysis.analyses.clustering",
    "contacts":                "membrane_analysis.analyses.contacts",
    "inter_residue_distance":  "membrane_analysis.analyses.inter_residue_distance",
    "kinetics":                "membrane_analysis.analyses.kinetics",
}

ALL_MODULES = {**EQUILIBRATION_MODULES, **ANALYSIS_MODULES}


def import_module(module_path):
    """Dynamically import an analysis module."""
    import importlib
    return importlib.import_module(module_path)


def run_pipeline(cfg, mode="all", plot_only=False, recompute_keys=None):
    """
    Run the analysis pipeline.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    mode : str
        "all", "equilibration", or "analysis"
    plot_only : bool
        If True, skip computation and only regenerate plots from cache.
    recompute_keys : list or None
        If provided, force recompute only these specific analyses.
    """
    # select which modules to run
    if mode == "equilibration":
        modules = EQUILIBRATION_MODULES
    elif mode == "analysis":
        modules = ANALYSIS_MODULES
    else:
        modules = ALL_MODULES

    # filter to enabled analyses
    enabled = {k: v for k, v in modules.items() if is_analysis_enabled(cfg, k)}
    if not enabled:
        print("No analyses are enabled in the config. Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"  Membrane Analysis Pipeline - mode: {mode}")
    print(f"  Enabled analyses: {', '.join(enabled.keys())}")
    print(f"{'='*60}\n")

    # apply per-key recompute overrides
    if recompute_keys:
        original_force = cfg.get("global", {}).get("force_recompute", False)
        for key in recompute_keys:
            if key in enabled:
                print(f"  Will force recompute: {key}")

    # build universes (skip if plot-only — compute functions will load from cache)
    universes = None
    if not plot_only:
        print("Loading universes...")
        universes = build_all_universes(cfg)
        print(f"  Loaded {len(universes)} universe(s).\n")

    # run each analysis
    for key, module_path in enabled.items():
        print(f"\n{'-'*40}")
        print(f"  Running: {key}")
        print(f"{'-'*40}")

        mod = import_module(module_path)

        # handle per-key force recompute
        if recompute_keys and key in recompute_keys:
            cfg.setdefault("global", {})["force_recompute"] = True
        elif recompute_keys:
            cfg.setdefault("global", {})["force_recompute"] = original_force

        if plot_only:
            # force load from cache by setting force_recompute = False
            cfg.setdefault("global", {})["force_recompute"] = False

        try:
            results = mod.compute(cfg, universes)
            print(f"  Plotting {key}...")
            mod.plot(cfg, results)
        except FileNotFoundError as e:
            if plot_only:
                print(f"  No cached data for {key}: {e}")
                print(f"  Run without --plot-only first to generate data.")
            else:
                raise
        except Exception as e:
            print(f"  ERROR in {key}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")


# ── Statistical analysis pipeline ────────────────────────────────────────────

STATS_MODULES = {
    "clustering": "membrane_analysis.analyses.clustering",
    "gmm":        "membrane_analysis.analyses.gmm",
    "kinetics":   "membrane_analysis.analyses.kinetics",
}


def run_stats_pipeline(cfg, stats_cfg):
    """Run statistical analysis modules using the stats config.

    Each stats module reads from the feature assembly layer, which
    resolves cached data from the main pipeline.
    """
    print(f"\n{'='*60}")
    print("  Statistical Analysis Pipeline")
    print(f"{'='*60}\n")

    for module_key, module_path in STATS_MODULES.items():
        module_cfg = get_stats_params(stats_cfg, module_key)
        if not module_cfg:
            continue

        feature_set_name = module_cfg.get("feature_set")
        if not feature_set_name:
            print(f"  [{module_key}] No feature_set specified, skipping.")
            continue

        print(f"\n{'-'*40}")
        print(f"  Running: {module_key}")
        print(f"  Feature set: {feature_set_name}")
        print(f"{'-'*40}")

        mod = import_module(module_path)

        try:
            # Stats modules take (cfg, stats_cfg) instead of (cfg, universes)
            results = mod.compute(cfg, stats_cfg)
            if results:
                print(f"  Plotting {module_key}...")
                mod.plot(cfg, stats_cfg, results)
        except Exception as e:
            print(f"  ERROR in {module_key}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Stats pipeline complete.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Membrane equilibration & orientation analysis pipeline"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--mode", choices=["all", "equilibration", "analysis"],
                        default="all", help="Which module group to run")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, regenerate plots from cached data")
    parser.add_argument("--recompute", nargs="+", metavar="KEY",
                        help="Force recompute specific analyses (e.g. rmsd apl)")
    parser.add_argument("--stats", metavar="STATS_YAML",
                        help="Path to stats YAML for clustering/GMM/kinetics")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # setup matplotlib style
    style_cfg = get_style(cfg)
    setup_style(font_family=style_cfg["font_family"], dpi=style_cfg["dpi"])

    # Main analysis pipeline (unless --stats-only in the future)
    if not args.stats:
        run_pipeline(cfg, mode=args.mode, plot_only=args.plot_only,
                     recompute_keys=args.recompute)
    else:
        # If --stats is given, run main pipeline first (loads from cache),
        # then run stats pipeline
        run_pipeline(cfg, mode=args.mode, plot_only=args.plot_only,
                     recompute_keys=args.recompute)
        stats_cfg = load_stats_config(args.stats)
        run_stats_pipeline(cfg, stats_cfg)


if __name__ == "__main__":
    main()
