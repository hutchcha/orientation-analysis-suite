"""Cache-or-load utilities for pickle and numpy data."""

import os
import pickle
import numpy as np


def ensure_dir(path):
    """Create parent directories for a file path if they don't exist."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def save_pickle(data, path):
    """Save data to a pickle file, creating directories as needed."""
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved: {path}")


def load_pickle(path):
    """Load and return data from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded: {path}")
    return data


def cached_compute(cache_path, compute_fn, force_recompute=False):
    """
    If cache_path exists and force_recompute is False, load from cache.
    Otherwise, call compute_fn() and save the result.
    Returns the data either way.
    """
    if os.path.exists(cache_path) and not force_recompute:
        return load_pickle(cache_path)
    data = compute_fn()
    save_pickle(data, cache_path)
    return data


def save_per_system(results, outdir, analysis_key):
    """Save individual pickle files into per-system subfolders.

    Given a results dict {system_name: data}, saves each system's data
    to ``outdir/{system_name}/{analysis_key}.pkl``.

    Parameters
    ----------
    results      : dict {str: any}
    outdir       : str — analysis output directory (e.g. results/rmsd/)
    analysis_key : str — e.g. "rmsd"
    """
    for name, data in results.items():
        sys_dir = os.path.join(outdir, name)
        os.makedirs(sys_dir, exist_ok=True)
        sys_cache = os.path.join(sys_dir, f"{analysis_key}.pkl")
        with open(sys_cache, "wb") as f:
            pickle.dump(data, f)


def save_txt(arr, path):
    """Save a numpy array to a text file."""
    ensure_dir(path)
    np.savetxt(path, arr)


def load_txt(path):
    """Load a numpy array from a text file."""
    return np.loadtxt(path)
