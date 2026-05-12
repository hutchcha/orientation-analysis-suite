"""Hidden Markov Model kinetics analysis (pyemma).

Constructs a Bayesian HMM from orientation/distance features or arbitrary
input data.  Produces implied timescale scans, transition matrices, MFPT
heatmaps, state populations, dwell times, Chapman-Kolmogorov validation,
and flux network graphs.

Calling conventions
-------------------
``compute(cfg, universes)``
    Main-pipeline mode.  Parameters live under
    ``analyses.analysis.kinetics`` in the main config.  Two data modes are
    supported (``mode: spherical|direct``):

    - **spherical** — assembles (rotation, tilt, distance) from the
      tilt_rotation and inter_residue_distance pipeline caches, transforms
      to 3-D Cartesian via spherical_to_cartesian with configurable
      radial_scale.
    - **direct** — loads arbitrary column files (one value per line per
      file) and stacks them as feature columns.  No spherical transform.

``compute(cfg, stats_cfg)``
    Stats-pipeline mode.  Parameters live under ``kinetics`` in
    ``stats.yaml`` and the feature matrix is assembled by
    ``core.features.assemble_features`` from the named ``feature_set``.

Config fields — main pipeline (``analyses.analysis.kinetics``)
---------------------------------------------------------------
  enabled       : true/false
  mode          : "spherical" or "direct"

  # spherical mode
  radial_scale  : float  (default 1.0)

  # direct mode
  custom_features:
    - /path/to/feature1.txt
    - /path/to/feature2.txt

  # microstate discretisation, HMM parameters (shared with stats mode)
  micro_method  : "kmeans" or "gmm"  (default kmeans)
  K_micro       : int  (default 300)
  K             : int    (number of hidden states, default 4)
  lag           : int    (HMM lag in frames; 0 = ITS scan only)
  max_lag       : int    (max lag for ITS scan, default 5000)
  dt_time       : float  (ns per frame, default 0.1)
  nsamples      : int    (Bayesian HMM samples, default 100)

  # optional state labelling
  state_labels  : {0: "OS1", 1: "OS2", ...}
  mfpt_order    : [2, 1, 0, 3]   # custom ordering for MFPT/population plots

Config fields — stats pipeline (``stats.yaml`` ``kinetics`` block)
-------------------------------------------------------------------
Same as above, except ``mode``, ``radial_scale`` and ``custom_features``
are dropped — feature selection is driven by ``feature_set`` instead.
"""

import os
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

from membrane_analysis.core.io import cached_compute, save_per_system
from membrane_analysis.core.config import (
    get_output_dir, get_system_names, get_sim_length,
    is_force_recompute, get_analysis_params,
    get_stats_params, get_feature_set,
)
from membrane_analysis.core.features import assemble_features
from membrane_analysis.core.plotting import style_axes, save_figure

ANALYSIS_KEY = "kinetics"

# Suppress noisy eigenvalue warnings from pyemma
warnings.filterwarnings("ignore", message="Using eigenvalues with non-zero imaginary part")


# ── Coordinate transforms ────────────────────────────────────────────────────

def spherical_to_cartesian(rot_deg, tilt_deg, distance, radial_scale=1.0):
    """(rotation, tilt, distance) -> (N, 3) Cartesian."""
    phi   = np.deg2rad(rot_deg)
    theta = np.deg2rad(tilt_deg)
    r     = np.asarray(distance, dtype=float) * float(radial_scale)
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


def circular_mean(angles_deg):
    rads = np.deg2rad(angles_deg)
    return np.degrees(np.arctan2(np.sum(np.sin(rads)), np.sum(np.cos(rads))))


# ── Feature assembly ─────────────────────────────────────────────────────────

def _assemble_features_spherical(cfg, name):
    """Load tilt/rotation + distance from pipeline caches, return Cartesian."""
    params = get_analysis_params(cfg, ANALYSIS_KEY)
    outdir = get_output_dir(cfg)

    tr_cache = os.path.join(outdir, "tilt_rotation", "tilt_rotation.pkl")
    if not os.path.exists(tr_cache):
        raise FileNotFoundError(f"tilt_rotation cache not found: {tr_cache}")
    with open(tr_cache, "rb") as f:
        tr_data = pickle.load(f)

    if name not in tr_data:
        raise KeyError(f"System '{name}' not in tilt_rotation cache.")

    rot  = tr_data[name]["rotation"]
    tilt = tr_data[name]["tilt"]

    dist_cache = os.path.join(outdir, "inter_residue_distance", "inter_residue_distance.pkl")
    if not os.path.exists(dist_cache):
        raise FileNotFoundError(
            f"inter_residue_distance cache not found: {dist_cache}. "
            "Run inter_residue_distance first."
        )
    with open(dist_cache, "rb") as f:
        dist_data = pickle.load(f)

    if name not in dist_data:
        raise KeyError(f"System '{name}' not in inter_residue_distance cache.")

    dist = dist_data[name]

    # Align lengths (different strides may produce different counts)
    n = min(len(rot), len(tilt), len(dist))
    rot, tilt, dist = rot[:n], tilt[:n], dist[:n]

    # Drop NaN frames
    valid = np.isfinite(rot) & np.isfinite(tilt) & np.isfinite(dist)
    rot, tilt, dist = rot[valid], tilt[valid], dist[valid]

    radial_scale = float(params.get("radial_scale", 1.0))
    pts = spherical_to_cartesian(rot, tilt, dist, radial_scale)

    return pts, {"rotation": rot, "tilt": tilt, "distance": dist}


def _assemble_features_direct(cfg, name):
    """Load arbitrary feature files, stack as columns."""
    params = get_analysis_params(cfg, ANALYSIS_KEY)
    files  = params.get("custom_features", [])
    if not files:
        raise ValueError("direct mode requires 'custom_features' list in config.")

    cols = []
    for fpath in files:
        arr = np.loadtxt(fpath)
        cols.append(arr)

    n = min(len(c) for c in cols)
    pts = np.column_stack([c[:n] for c in cols])

    valid = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid]

    return pts, None


def _assemble_features_via_stats(cfg, stats_cfg, name):
    """Stats-mode: build feature matrix via core.features.assemble_features.

    Returns
    -------
    pts     : (N, D) ndarray
    columns : list of str — feature column names
    meta    : dict — includes "transform", "radial_scale", "valid_mask",
              "min_len", "feature_keys"
    """
    kin_cfg = get_stats_params(stats_cfg, ANALYSIS_KEY)
    fs_name = kin_cfg.get("feature_set")
    if not fs_name:
        raise ValueError("stats-mode kinetics requires 'feature_set' in stats config.")
    fs_cfg = get_feature_set(stats_cfg, fs_name)
    return assemble_features(get_output_dir(cfg), fs_cfg, name)


# ── Microstate clustering ────────────────────────────────────────────────────

def _cluster_microstates(pts, method, K_micro):
    """Discretise feature space into microstates. Returns dtraj (int array)."""
    import pyemma

    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=K_micro, covariance_type="full",
                              reg_covar=1e-9, random_state=42)
        gmm.fit(pts[::2])
        return gmm.predict(pts).astype(np.int32)
    else:
        cl = pyemma.coordinates.cluster_kmeans([pts], k=K_micro, max_iter=500,
                                                fixed_seed=1)
        return cl.dtrajs[0]


# ── HMM ITS scan ────────────────────────────────────────────────────────────

def _its_scan(dtrajs, K, dt_time, max_lag, nsamples=20, n_points=50):
    """Bayesian HMM implied timescale scan. Returns (valid_lags, ts_array)."""
    import pyemma

    lags = np.unique(np.geomspace(2, max_lag, num=n_points).astype(int))
    lags = lags[lags >= 2]
    n_modes = K - 1

    ts_results, valid_lags = [], []

    for lag in lags:
        try:
            hmm = pyemma.msm.bayesian_hidden_markov_model(
                dtrajs, nstates=K, lag=int(lag),
                nsamples=nsamples, connectivity="largest",
            )
            T = np.array(hmm.transition_matrix, dtype=float)
            eig = np.real(np.real_if_close(np.linalg.eigvals(T), tol=1000))
            ev = np.sort(eig)[::-1][1:1 + n_modes]
            ev = np.clip(ev, 1e-12, 1.0 - 1e-12)
            its = -float(lag) * dt_time / np.log(ev)
            if np.all(np.isfinite(its)):
                ts_results.append(its)
                valid_lags.append(lag)
                n_clust = len(set(hmm.metastable_assignments) if hasattr(hmm, 'metastable_assignments') else range(K))
                print(f"    lag={lag:5d}  ITS={[f'{t:.1f}' for t in its]} ns")
        except Exception:
            pass

    if not valid_lags:
        print("    WARNING: No valid lags in ITS scan.")
        return np.array([]), np.array([])

    return np.array(valid_lags), np.vstack(ts_results)


# ── HMM fitting ─────────────────────────────────────────────────────────────

def _fit_hmm(dtrajs, K, lag, nsamples, dt_time, pts_xyz, on_sphere=True):
    """Fit Bayesian HMM and extract all analysis quantities.

    Parameters
    ----------
    on_sphere : bool
        If True, state centres are computed as a "spherical mean" (project
        the per-state COM onto the sphere of mean radius).  Use this when
        the feature space is unit_sphere or spherical.  If False, plain
        Euclidean COM is used (correct for arbitrary scalar features).

    Returns dict with T_sorted, pi_sorted, mfpt_ns, flux, sort_order,
    dtraj_macro, timescales_ns, ck_test data.
    """
    import pyemma

    print(f"    Fitting Bayesian HMM (K={K}, lag={lag}, nsamples={nsamples})...")
    hmm = pyemma.msm.bayesian_hidden_markov_model(
        dtrajs, nstates=K, lag=lag, nsamples=nsamples, connectivity="all",
    )

    pi = hmm.stationary_distribution
    sort_order = np.argsort(pi)[::-1]

    T_sorted  = hmm.transition_matrix[sort_order][:, sort_order]
    pi_sorted = pi[sort_order]

    # Viterbi path → macro trajectory
    hidden = np.concatenate(hmm.hidden_state_trajectories)
    remap  = {old: new for new, old in enumerate(sort_order)}
    hidden_sorted = np.array([remap[s] for s in hidden], dtype=int)
    ratio = max(1, round(len(dtrajs[0]) / len(hidden_sorted)))
    dtraj_macro = np.repeat(hidden_sorted, ratio)[:len(dtrajs[0])]

    # MFPT matrix
    mfpt = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                mfpt[i, j] = hmm.mfpt(i, j) * dt_time
    mfpt_sorted = mfpt[sort_order][:, sort_order]

    # Flux
    flux = pi_sorted[:, None] * T_sorted

    # HMM timescales
    timescales_ns = [float(t * dt_time) for t in hmm.timescales()]

    # CK test data
    ck = _ck_test_data(dtrajs, lag, K, T_sorted, dt_time, dtraj_macro)

    # State centres in feature space
    centers = np.zeros((K, pts_xyz.shape[1]))
    for k in range(K):
        mask = dtraj_macro == k
        if mask.sum() > 0:
            com = pts_xyz[mask].mean(axis=0)
            if on_sphere:
                radii = np.linalg.norm(pts_xyz[mask], axis=1)
                if np.linalg.norm(com) > 1e-9:
                    centers[k] = (com / np.linalg.norm(com)) * radii.mean()
                else:
                    centers[k] = com
            else:
                centers[k] = com

    return {
        "T_sorted":       T_sorted,
        "pi_sorted":      pi_sorted,
        "mfpt_ns":        mfpt_sorted,
        "flux":           flux,
        "sort_order":     sort_order,
        "dtraj_macro":    dtraj_macro,
        "timescales_ns":  timescales_ns,
        "ck":             ck,
        "centers":        centers,
        "K":              K,
        "lag":            lag,
    }


def _ck_test_data(dtrajs, msm_lag, K, T_model, dt_time, dtraj_macro, steps=9):
    """Compute CK test: model prediction vs. data estimate at multiples of lag."""
    import pyemma

    test_lags = np.arange(1, steps + 1) * msm_lag
    pred = np.zeros((len(test_lags), K, K))
    est  = np.zeros((len(test_lags), K, K))

    for k_idx, k in enumerate(range(1, steps + 1)):
        Tk = np.real_if_close(np.linalg.matrix_power(T_model, k), tol=1000)
        pred[k_idx] = Tk

    for k_idx, lag in enumerate(test_lags):
        try:
            msm_tmp = pyemma.msm.estimate_markov_model(
                [dtraj_macro], lag=int(lag), reversible=False, connectivity="largest")
            est[k_idx] = msm_tmp.transition_matrix
        except Exception:
            est[k_idx] = np.full((K, K), np.nan)

    return {
        "test_lags":  test_lags,
        "prediction": pred,
        "estimation": est,
        "dt_time":    dt_time,
    }


# ── Top-level compute ────────────────────────────────────────────────────────

def compute(cfg, universes_or_stats_cfg=None):
    """Run HMM kinetics analysis for all systems.

    Supports two calling conventions:
      - ``compute(cfg, universes)`` — main pipeline; uses
        ``analyses.analysis.kinetics`` config and the spherical/direct
        assembly modes.
      - ``compute(cfg, stats_cfg)``  — stats pipeline; assembles features via
        ``core/features.py`` from the named ``feature_set`` in stats config.

    The stats path is detected when the second arg is a dict with a
    ``feature_sets`` key.
    """
    is_stats = (isinstance(universes_or_stats_cfg, dict)
                and "feature_sets" in universes_or_stats_cfg)

    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)
    cache  = os.path.join(outdir, "kinetics.pkl")

    if is_stats:
        stats_cfg = universes_or_stats_cfg
        params    = get_stats_params(stats_cfg, ANALYSIS_KEY)
        force     = True   # stats runs always recompute (feature_set may differ)
    else:
        params = get_analysis_params(cfg, ANALYSIS_KEY)
        force  = is_force_recompute(cfg)

    def _run():
        micro_meth  = params.get("micro_method", "kmeans")
        K_micro     = int(params.get("K_micro", 300))
        K           = int(params.get("K", 4))
        lag         = int(params.get("lag", 0))
        max_lag     = int(params.get("max_lag", 5000))
        dt_time     = float(params.get("dt_time", 0.1))
        nsamples    = int(params.get("nsamples", 100))

        # Main-pipeline assembly mode (ignored in stats mode)
        mode = params.get("mode", "spherical")

        results = {}
        for name in get_system_names(cfg):
            transform = None
            columns   = None
            meta      = None
            raw       = None

            try:
                if is_stats:
                    print(f"  [{name}] Assembling features (stats mode)...")
                    pts, columns, meta = _assemble_features_via_stats(
                        cfg, stats_cfg, name)
                    transform = meta.get("transform")
                else:
                    print(f"  [{name}] Assembling features (mode={mode})...")
                    if mode == "spherical":
                        pts, raw = _assemble_features_spherical(cfg, name)
                        transform = "spherical"
                    else:
                        pts, raw = _assemble_features_direct(cfg, name)
                        transform = "none"
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  [{name}] {e}")
                continue

            on_sphere = transform in ("spherical", "unit_sphere")
            print(f"  [{name}] Feature matrix: {pts.shape}  "
                  f"(transform={transform}, on_sphere={on_sphere})")

            print(f"  [{name}] Microstate clustering ({micro_meth}, K_micro={K_micro})...")
            dtraj = _cluster_microstates(pts, micro_meth, K_micro)

            print(f"  [{name}] ITS scan (K={K}, max_lag={max_lag})...")
            its_lags, its_ts = _its_scan([dtraj], K, dt_time, max_lag, nsamples=min(20, nsamples))

            entry = {
                "pts_xyz":      pts,
                "raw":          raw,
                "columns":      columns,
                "meta":         meta,
                "transform":    transform,
                "on_sphere":    on_sphere,
                "dtraj_micro":  dtraj,
                "its_lags":     its_lags,
                "its_ts":       its_ts,
                "dt_time":      dt_time,
                "K":            K,
            }

            if lag > 0:
                print(f"  [{name}] Fitting HMM (lag={lag})...")
                hmm_result = _fit_hmm([dtraj], K, lag, nsamples, dt_time, pts,
                                      on_sphere=on_sphere)
                entry["hmm"] = hmm_result
            else:
                print(f"  [{name}] lag=0, skipping HMM fit (ITS scan only).")

            results[name] = entry

        save_per_system(results, outdir, ANALYSIS_KEY)
        return results

    return cached_compute(cache, _run, force_recompute=force)


# ── Plotting functions ────────────────────────────────────────────────────────

def _resolve_params(cfg, stats_cfg=None):
    """Pick the right param dict — stats_cfg takes priority when given."""
    if stats_cfg is not None:
        return get_stats_params(stats_cfg, ANALYSIS_KEY)
    return get_analysis_params(cfg, ANALYSIS_KEY)


def _get_state_labels(cfg, K, stats_cfg=None):
    """Return list of state label strings."""
    params = _resolve_params(cfg, stats_cfg)
    label_map = params.get("state_labels", {})
    # label_map can be {0: "OS1", ...} or null
    if label_map:
        return [str(label_map.get(i, label_map.get(str(i), f"S{i}"))) for i in range(K)]
    return [f"S{i}" for i in range(K)]


def _get_mfpt_order(cfg, K, stats_cfg=None):
    params = _resolve_params(cfg, stats_cfg)
    order = params.get("mfpt_order")
    if order and len(order) == K:
        return list(order)
    return list(range(K))


def _plot_its(its_lags, its_ts, dt_time, K, outpath):
    """Implied timescale plot."""
    if len(its_lags) == 0:
        return

    x = its_lags * dt_time
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for m in range(its_ts.shape[1]):
        ax.plot(x, its_ts[:, m], lw=2, label=f"Mode {m+1}")

    ax.plot(x, x, "k-", lw=2, label="Lag time limit")
    ax.fill_between(x, 0, x, color="grey", alpha=0.15)
    ax.set_yscale("log")

    style_axes(ax, xlabel="Lag time (ns)", ylabel="Implied timescale (ns)",
               title=f"Bayesian HMM ITS (K={K})")
    ax.legend(fontsize=10, frameon=False)
    save_figure(fig, outpath)


def _plot_transition_matrix(T, labels, outpath):
    K = T.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, K * 1.2), max(3.5, K)),
                           constrained_layout=True)
    im = ax.imshow(T, origin="upper", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Probability", shrink=0.8)

    for i in range(K):
        for j in range(K):
            c = "white" if T[i, j] < 0.6 else "black"
            ax.text(j, i, f"{T[i, j]:.3f}", ha="center", va="center", color=c, fontsize=11)

    ax.set_xticks(range(K)); ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=12)
    style_axes(ax, title="Transition matrix")
    save_figure(fig, outpath)


def _plot_mfpt(mfpt, labels, outpath):
    K = mfpt.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, K * 1.2), max(3.5, K)),
                           constrained_layout=True)
    im = ax.imshow(mfpt, origin="upper", cmap="cividis")
    plt.colorbar(im, ax=ax, label="MFPT (ns)", shrink=0.8)

    mx = np.max(mfpt)
    for i in range(K):
        for j in range(K):
            c = "black" if mfpt[i, j] < mx * 0.6 else "white"
            txt = f"{mfpt[i, j]:.1f}" if i != j else "-"
            ax.text(j, i, txt, ha="center", va="center", color=c, fontsize=11)

    ax.set_xticks(range(K)); ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=12)
    style_axes(ax, xlabel="To", ylabel="From", title="Mean First Passage Time (ns)")
    save_figure(fig, outpath)


def _plot_populations(pi, labels, outpath):
    K = len(pi)
    colors = [f"C{i}" for i in range(K)]
    fig, ax = plt.subplots(figsize=(max(4, K * 1.5), 4), constrained_layout=True)
    bars = ax.bar(labels, pi, color=colors, alpha=0.9, edgecolor="black", linewidth=1.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(pi) * 1.3)
    style_axes(ax, ylabel="Stationary probability")
    save_figure(fig, outpath)


def _plot_dwell_times(T, lag, dt_time, labels, outpath):
    K = T.shape[0]
    T_diag = np.clip(np.diag(T), 1e-12, 1.0 - 1e-12)
    lifetimes_ns = -lag * dt_time / np.log(T_diag)
    colors = [f"C{i}" for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(4, K * 1.5), 4), constrained_layout=True)
    bars = ax.bar(labels, lifetimes_ns, color=colors, alpha=0.9,
                  edgecolor="black", linewidth=1.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f} ns",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(lifetimes_ns) * 1.3)
    style_axes(ax, ylabel="Mean dwell time (ns)")
    save_figure(fig, outpath)


def _plot_ck_test(ck, labels, outpath):
    """Chapman-Kolmogorov test: model prediction vs data estimation."""
    K         = ck["prediction"].shape[1]
    test_lags = ck["test_lags"]
    pred      = ck["prediction"]
    est       = ck["estimation"]
    dt_time   = ck["dt_time"]

    x = test_lags * dt_time

    fig, axes = plt.subplots(K, K, figsize=(3 * K, 3 * K),
                             sharex=True, sharey=True, constrained_layout=True)
    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            ax.plot(x, pred[:, i, j], "k-", lw=2, label="Model")
            ax.plot(x, est[:, i, j],  "b--", lw=2, label="Data")
            ax.set_ylim(-0.05, 1.1)
            if i == j:
                ax.set_facecolor("#f7f7f7")
            ax.set_title(f"{labels[i]} -> {labels[j]}", fontsize=10)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(labelsize=9)

    fig.supxlabel("Lag time (ns)", fontsize=14)
    fig.supylabel("Probability", fontsize=14)
    save_figure(fig, outpath)


def _plot_flux_network(flux, T, labels, outpath, min_flux=1e-4):
    """Flux network with directed edges."""
    K = flux.shape[0]
    G = nx.MultiDiGraph()
    for i in range(K):
        G.add_node(i)

    pos = nx.circular_layout(range(K), scale=2.0)
    node_sizes = 800 + 3000 * (flux.sum(axis=1) / (flux.sum(axis=1).max() + 1e-12))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="white",
                           edgecolors="black", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(K)},
                            font_size=14, font_weight="bold", ax=ax)

    max_flux = np.max(flux) + 1e-12
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            f = flux[i, j]
            if f < min_flux:
                continue
            w = 1.0 + 8.0 * (f / max_flux)
            color = "#FF4444" if f >= flux[j, i] else "#4444FF"
            nx.draw_networkx_edges(
                G, pos, edgelist=[(i, j)], width=w,
                edge_color=color, connectionstyle="arc3,rad=0.15",
                arrowsize=20, ax=ax, node_size=node_sizes,
            )
            mid = 0.5 * (np.array(pos[i]) + np.array(pos[j]))
            vec = np.array(pos[j]) - np.array(pos[i])
            perp = np.array([-vec[1], vec[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            label_pos = mid + 0.15 * np.linalg.norm(vec) * perp
            ax.text(label_pos[0], label_pos[1],
                    f"F={f:.2e}\nP={T[i, j]:.2e}",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    ax.axis("off")
    save_figure(fig, outpath)


# ── Top-level plot ────────────────────────────────────────────────────────────

def plot(cfg, results_or_stats_cfg=None, results=None):
    """Generate all kinetics diagnostic plots.

    Supports two call signatures:
      - ``plot(cfg, results)``             — main pipeline
      - ``plot(cfg, stats_cfg, results)``  — stats pipeline
    """
    if results is None:
        # Two-arg form: second positional is the results dict
        results   = results_or_stats_cfg
        stats_cfg = None
    else:
        stats_cfg = results_or_stats_cfg

    outdir = os.path.join(get_output_dir(cfg), ANALYSIS_KEY)

    for name, data in results.items():
        K        = data["K"]
        dt_time  = data["dt_time"]
        labels   = _get_state_labels(cfg, K, stats_cfg=stats_cfg)
        sys_dir  = os.path.join(outdir, name)
        os.makedirs(sys_dir, exist_ok=True)
        prefix   = os.path.join(sys_dir, "")

        # ITS plot (always)
        _plot_its(data["its_lags"], data["its_ts"], dt_time, K,
                  prefix + "its.png")

        if "hmm" not in data:
            continue

        hmm   = data["hmm"]
        order = _get_mfpt_order(cfg, K, stats_cfg=stats_cfg)
        T_ord = hmm["T_sorted"][order][:, order]
        pi_ord = hmm["pi_sorted"][order]
        mfpt_ord = hmm["mfpt_ns"][order][:, order]
        flux_ord = hmm["flux"][order][:, order]
        labels_ord = [labels[i] for i in order]

        _plot_transition_matrix(T_ord, labels_ord, prefix + "T_matrix.png")
        _plot_mfpt(mfpt_ord, labels_ord, prefix + "mfpt.png")
        _plot_populations(pi_ord, labels_ord, prefix + "populations.png")
        _plot_dwell_times(T_ord, hmm["lag"], dt_time, labels_ord,
                          prefix + "dwell_times.png")
        _plot_ck_test(hmm["ck"], labels_ord, prefix + "ck_test.png")
        _plot_flux_network(flux_ord, T_ord, labels_ord,
                           prefix + "flux_network.png")
