"""Cache-or-load utilities for pickle and numpy data.

Cache file format
-----------------
Caches written by ``cached_compute(...)`` use one of two formats:

1. **Bare pickle** — `data` itself.  This is the legacy format and the
   format used when ``metadata`` is not provided.

2. **Wrapped with metadata** — ``{"_metadata": <dict>, "data": <data>}``.
   Used when ``metadata`` is provided.  The metadata captures the
   analysis parameters (window, stride, selections, etc.), which lets the
   loader detect stale caches and trigger an automatic recompute.

The format is detected on load via the presence of both ``_metadata`` and
``data`` keys at the top level.
"""

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


def _is_wrapped_cache(obj):
    """Detect the metadata-wrapped cache format."""
    return (isinstance(obj, dict)
            and "_metadata" in obj
            and "data" in obj
            and len(obj) == 2)


def _diff_metadata(old, new, prefix=""):
    """Print human-readable summary of metadata differences."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        if old != new:
            print(f"    {prefix or '(root)'}: {old!r} -> {new!r}")
        return
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "<missing>"), new.get(k, "<missing>")
        if a == b:
            continue
        sub = f"{prefix}.{k}" if prefix else k
        if isinstance(a, dict) and isinstance(b, dict):
            _diff_metadata(a, b, sub)
        else:
            print(f"    {sub}: {a!r} -> {b!r}")


def cached_compute(cache_path, compute_fn, force_recompute=False, metadata=None):
    """Cache-or-compute with optional metadata validation.

    If ``cache_path`` exists and ``force_recompute`` is False:
      - If ``metadata`` is None, return the cached payload as-is.
      - If ``metadata`` is provided and the cached metadata matches,
        return the cached data.
      - If ``metadata`` is provided and the cached metadata differs (or
        the cache is in legacy bare format), warn and recompute.

    The new cache is written in wrapped format when ``metadata`` is given,
    and in bare format otherwise.
    """
    if os.path.exists(cache_path) and not force_recompute:
        cached = load_pickle(cache_path)

        if _is_wrapped_cache(cached):
            cached_meta = cached["_metadata"]
            if metadata is None or cached_meta == metadata:
                return cached["data"]
            print(f"  Cache metadata mismatch — recomputing.")
            print(f"  Differences:")
            _diff_metadata(cached_meta, metadata)
        else:
            # Legacy bare format
            if metadata is None:
                return cached
            print(f"  Cache in legacy format (no metadata) — recomputing "
                  f"to capture current analysis parameters.")

    data = compute_fn()

    if metadata is not None:
        save_pickle({"_metadata": metadata, "data": data}, cache_path)
    else:
        save_pickle(data, cache_path)
    return data


def load_cache_metadata(cache_path):
    """Return the metadata dict embedded in a cache, or None.

    Returns None if the cache doesn't exist, is in legacy format, or
    does not contain metadata.  Plot-only mode uses this to recover
    the analysed time-axis bounds without loading a Universe.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
    except Exception:
        return None
    if _is_wrapped_cache(cached):
        return cached["_metadata"]
    return None


def load_cache_data(cache_path):
    """Load the data portion of a cache, transparent to the wrapping format.

    Use this in place of raw ``pickle.load`` when reading another module's
    cache — it correctly unwraps the new metadata-wrapped format and
    leaves legacy bare caches untouched.
    """
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    if _is_wrapped_cache(cached):
        return cached["data"]
    return cached


def save_per_system(results, outdir, analysis_key, metadata=None):
    """Save individual pickle files into per-system subfolders.

    Given a results dict {system_name: data}, saves each system's data
    to ``outdir/{system_name}/{analysis_key}.pkl``.

    If ``metadata`` is provided (the full analysis metadata dict from
    ``build_cache_metadata``), each per-system file is written in wrapped
    format with the metadata for *that system* extracted from
    ``metadata["per_system"][name]``.  Top-level metadata fields
    (analysis_params, group_options, etc.) are also included so each
    per-system file is self-describing.
    """
    for name, data in results.items():
        sys_dir = os.path.join(outdir, name)
        os.makedirs(sys_dir, exist_ok=True)
        sys_cache = os.path.join(sys_dir, f"{analysis_key}.pkl")

        if metadata is not None:
            sys_meta = {
                "analysis_key":       metadata.get("analysis_key"),
                "full_sim_length_us": metadata.get("full_sim_length_us"),
                "analysis_params":    metadata.get("analysis_params"),
                "group_options":      metadata.get("group_options"),
                "system":              metadata.get("per_system", {}).get(name),
            }
            payload = {"_metadata": sys_meta, "data": data}
        else:
            payload = data

        with open(sys_cache, "wb") as f:
            pickle.dump(payload, f)


def get_time_bounds(cache_metadata, system_name, fallback_us):
    """Return (start_us, end_us) for a system from cache metadata.

    Used by plot functions to build per-system time axes.  Falls back to
    ``(0.0, fallback_us)`` (the global sim_length) if metadata is absent
    or doesn't contain bounds for this system — preserves backward-compat
    behaviour with legacy caches.
    """
    if cache_metadata is None:
        return 0.0, float(fallback_us)
    sys_meta = cache_metadata.get("per_system", {}).get(system_name)
    if sys_meta is None:
        return 0.0, float(fallback_us)
    start_us = sys_meta.get("start_us")
    end_us   = sys_meta.get("end_us")
    if start_us is None or end_us is None:
        return 0.0, float(fallback_us)
    return float(start_us), float(end_us)


def save_txt(arr, path):
    """Save a numpy array to a text file."""
    ensure_dir(path)
    np.savetxt(path, arr)


def load_txt(path):
    """Load a numpy array from a text file."""
    return np.loadtxt(path)
