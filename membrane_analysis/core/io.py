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


def save_txt(arr, path):
    """Save a numpy array to a text file."""
    ensure_dir(path)
    np.savetxt(path, arr)


def load_txt(path):
    """Load a numpy array from a text file."""
    return np.loadtxt(path)
