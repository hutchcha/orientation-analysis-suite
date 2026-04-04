"""Feature assembly layer for statistical analysis modules.

Resolves feature keys from cached pickle files, aligns array lengths,
and applies coordinate transforms to produce a (N, D) feature matrix
suitable for clustering, GMM, or HMM analysis.

Feature keys
-------------
Each key refers to a cached analysis output:
  - ``"rmsd"``                  → 1D array from rmsd.pkl
  - ``"tilt_rotation.tilt"``    → sub-key from tilt_rotation.pkl dict
  - ``"lobe_com.Lobe1"``       → column from lobe_com.pkl DataFrame

Transforms
----------
  - ``"none"``         — column-stack all features (Euclidean).
  - ``"unit_sphere"``  — exactly 2 angular features (rotation, tilt) embedded
                         on the unit sphere as 3D Cartesian.
  - ``"spherical"``    — exactly 2 angular + 1 scalar → spherical-to-Cartesian
                         with configurable ``radial_scale``.
  - ``"auto"``         — inspect OUTPUT_TYPE metadata to pick the right transform.

Usage
-----
::

    from membrane_analysis.core.features import assemble_features

    X = assemble_features(cfg, stats_cfg, "orientation_3d", "RhoA-GTP")
    # X is (N, D) ndarray ready for clustering / GMM / HMM
"""

import os
import pickle

import numpy as np
import pandas as pd


# ── Feature type registry ────────────────────────────────────────────────────
# Maps (analysis_key, sub_key_or_None) → output type.
# Populated by _register_defaults() and can be extended by custom modules.

_FEATURE_TYPES = {}


def register_feature(analysis_key, output_type, sub_key=None):
    """Register the output type for a feature.

    Parameters
    ----------
    analysis_key : str   — e.g. "rmsd", "tilt_rotation"
    output_type  : str   — "scalar" or "angular"
    sub_key      : str or None — e.g. "tilt", "Lobe1"
    """
    _FEATURE_TYPES[(analysis_key, sub_key)] = output_type


def get_feature_type(analysis_key, sub_key=None):
    """Look up the output type for a feature key."""
    return _FEATURE_TYPES.get((analysis_key, sub_key), "scalar")


def _register_defaults():
    """Register output types for all built-in modules."""
    # Equilibration — all scalar
    for key in ("rmsd", "pp_distance", "anchor_insertion", "inter_residue_distance"):
        register_feature(key, "scalar")

    # Tilt/rotation — angular sub-keys
    register_feature("tilt_rotation", "angular", sub_key="tilt")
    register_feature("tilt_rotation", "angular", sub_key="rotation")

    # Lobe COM — scalar sub-keys
    register_feature("lobe_com", "scalar", sub_key="Lobe1")
    register_feature("lobe_com", "scalar", sub_key="Lobe2")

    # APL — scalar (DataFrame columns vary per system, treat as scalar)
    register_feature("apl", "scalar", sub_key="Total")

    # Contacts — not per-frame time series, excluded from features


_register_defaults()


# ── Feature resolution ───────────────────────────────────────────────────────

def _parse_feature_key(key_str):
    """Parse "analysis_key" or "analysis_key.sub_key" into (analysis_key, sub_key)."""
    parts = key_str.split(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], None


def _load_cached_array(output_dir, analysis_key, sub_key, system_name):
    """Load a single 1D array from the cached pickle for one system.

    Returns
    -------
    arr : 1D ndarray
    """
    # Standard cache path: {output_dir}/{analysis_key}/{analysis_key}.pkl
    cache_dir = os.path.join(output_dir, analysis_key)
    cache_file = os.path.join(cache_dir, f"{analysis_key}.pkl")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cache not found: {cache_file}. Run the '{analysis_key}' analysis first."
        )

    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    if system_name not in data:
        raise KeyError(
            f"System '{system_name}' not found in {cache_file}. "
            f"Available: {list(data.keys())}"
        )

    val = data[system_name]

    # Extract sub-key if needed
    if sub_key is not None:
        if isinstance(val, dict):
            if sub_key not in val:
                raise KeyError(
                    f"Sub-key '{sub_key}' not found in {analysis_key} for "
                    f"'{system_name}'. Available: {list(val.keys())}"
                )
            val = val[sub_key]
        elif isinstance(val, pd.DataFrame):
            if sub_key not in val.columns:
                raise KeyError(
                    f"Column '{sub_key}' not found in {analysis_key} DataFrame "
                    f"for '{system_name}'. Available: {list(val.columns)}"
                )
            val = val[sub_key].values
        else:
            raise TypeError(
                f"Cannot extract sub-key '{sub_key}' from {type(val).__name__} "
                f"in {analysis_key} for '{system_name}'."
            )

    # Ensure 1D numpy array
    if isinstance(val, pd.Series):
        val = val.values
    elif isinstance(val, pd.DataFrame):
        raise TypeError(
            f"Feature '{analysis_key}' for '{system_name}' is a DataFrame. "
            f"Specify a sub-key (e.g. '{analysis_key}.ColumnName')."
        )

    return np.asarray(val, dtype=float).ravel()


# ── Transforms ───────────────────────────────────────────────────────────────

def _unit_sphere(rotation_deg, tilt_deg):
    """(rotation, tilt) in degrees → (N, 3) Cartesian unit vectors."""
    phi   = np.deg2rad(rotation_deg)
    theta = np.deg2rad(tilt_deg)
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _spherical_to_cartesian(rotation_deg, tilt_deg, distance, radial_scale=1.0):
    """(rotation, tilt, distance) → (N, 3) Cartesian."""
    phi   = np.deg2rad(rotation_deg)
    theta = np.deg2rad(tilt_deg)
    r     = np.asarray(distance, dtype=float) * float(radial_scale)
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


def _detect_transform(feature_keys):
    """Auto-detect the appropriate transform from feature types.

    Returns
    -------
    transform : str — "unit_sphere", "spherical", or "none"
    """
    types = []
    for key_str in feature_keys:
        ak, sk = _parse_feature_key(key_str)
        types.append(get_feature_type(ak, sk))

    n_angular = types.count("angular")
    n_scalar  = types.count("scalar")

    if n_angular == 2 and n_scalar == 0:
        return "unit_sphere"
    elif n_angular == 2 and n_scalar >= 1:
        return "spherical"
    else:
        return "none"


# ── Main assembly function ───────────────────────────────────────────────────

def assemble_features(output_dir, feature_set_cfg, system_name):
    """Assemble a feature matrix for one system from a feature_set config block.

    Parameters
    ----------
    output_dir      : str — path to the results directory
    feature_set_cfg : dict — feature set config with keys:
                      "features" (list of str), "transform" (str),
                      optionally "radial_scale" (float)
    system_name     : str

    Returns
    -------
    X       : (N, D) ndarray — feature matrix
    columns : list of str — feature names for each column
    meta    : dict — transform info for downstream use
    """
    feature_keys = feature_set_cfg["features"]
    transform    = feature_set_cfg.get("transform", "auto")
    radial_scale = float(feature_set_cfg.get("radial_scale", 1.0))

    # Load all 1D arrays
    arrays = []
    for key_str in feature_keys:
        ak, sk = _parse_feature_key(key_str)
        arr = _load_cached_array(output_dir, ak, sk, system_name)
        arrays.append((key_str, arr))

    # Align to shortest length
    min_len = min(len(arr) for _, arr in arrays)
    arrays = [(name, arr[:min_len]) for name, arr in arrays]

    # Drop NaN frames (intersection of all finite)
    valid = np.ones(min_len, dtype=bool)
    for _, arr in arrays:
        valid &= np.isfinite(arr)

    arrays_clean = [(name, arr[valid]) for name, arr in arrays]
    n_valid = valid.sum()
    n_dropped = min_len - n_valid
    if n_dropped > 0:
        print(f"    Features: {n_valid} valid frames ({n_dropped} NaN dropped)")

    # Auto-detect transform if needed
    if transform == "auto":
        transform = _detect_transform(feature_keys)
        print(f"    Auto-detected transform: {transform}")

    # Apply transform
    if transform == "unit_sphere":
        if len(arrays_clean) != 2:
            raise ValueError(
                f"unit_sphere transform requires exactly 2 features "
                f"(rotation, tilt), got {len(arrays_clean)}."
            )
        # Convention: first angular = rotation, second = tilt
        rot_arr  = arrays_clean[0][1]
        tilt_arr = arrays_clean[1][1]
        types = [get_feature_type(*_parse_feature_key(k)) for k in feature_keys]
        # If tilt is listed first, swap
        if "tilt" in feature_keys[0].lower() and "rot" in feature_keys[1].lower():
            tilt_arr, rot_arr = arrays_clean[0][1], arrays_clean[1][1]
        X = _unit_sphere(rot_arr, tilt_arr)
        columns = ["x", "y", "z"]

    elif transform == "spherical":
        # Separate angular and scalar features
        angular_arrs = []
        scalar_arrs  = []
        for key_str, arr in arrays_clean:
            ak, sk = _parse_feature_key(key_str)
            ft = get_feature_type(ak, sk)
            if ft == "angular":
                angular_arrs.append((key_str, arr))
            else:
                scalar_arrs.append((key_str, arr))

        if len(angular_arrs) != 2:
            raise ValueError(
                f"spherical transform requires exactly 2 angular features, "
                f"got {len(angular_arrs)}."
            )
        if len(scalar_arrs) < 1:
            raise ValueError(
                f"spherical transform requires at least 1 scalar feature, "
                f"got {len(scalar_arrs)}."
            )

        # Identify rotation and tilt
        rot_arr, tilt_arr = angular_arrs[0][1], angular_arrs[1][1]
        if "tilt" in angular_arrs[0][0].lower():
            tilt_arr, rot_arr = angular_arrs[0][1], angular_arrs[1][1]

        # Use first scalar as the radial distance
        dist_arr = scalar_arrs[0][1]

        X = _spherical_to_cartesian(rot_arr, tilt_arr, dist_arr, radial_scale)
        columns = ["x", "y", "z"]

        # If there are additional scalar features, append them as extra columns
        if len(scalar_arrs) > 1:
            extra = np.column_stack([arr for _, arr in scalar_arrs[1:]])
            X = np.hstack([X, extra])
            columns += [name for name, _ in scalar_arrs[1:]]

    elif transform == "none":
        X = np.column_stack([arr for _, arr in arrays_clean])
        columns = [name for name, _ in arrays_clean]

    else:
        raise ValueError(f"Unknown transform: '{transform}'")

    meta = {
        "transform":    transform,
        "radial_scale": radial_scale,
        "feature_keys": feature_keys,
        "n_valid":      n_valid,
        "n_dropped":    n_dropped,
        "valid_mask":   valid,
        "min_len":      min_len,
    }

    return X, columns, meta
