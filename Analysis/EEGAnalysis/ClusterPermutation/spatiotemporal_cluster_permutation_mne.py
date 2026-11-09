"""
File: spatiotemporal_cluster_permutation_mne.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-02-12
Description: Run spatio-temporal cluster permutation test with mne python.

Notebook-style version for stepwise execution with ipykernel.
Model: amplitude ~ response_decoded + intensity_level_centered (+ optional interaction)
"""

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats


# %%
# Configuration
DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
RECONDIR = os.getenv("SUBJECTS_DIR", "")

if not DATADIR:
    raise EnvironmentError("DATADIR is not set.")
if not RESDIR:
    raise EnvironmentError("RESDIR is not set.")

EPOCH_SUFFIX = ""  # e.g., "preprestim"
N_PERMUTATIONS = 2000
ALPHA = 0.05
CLUSTER_ALPHA = 0.05
SEED = 42
INCLUDE_INTERACTION = True
SUBJECT_LIMIT = None  # e.g., 5 for quick tests


# %%
# Helper functions

def parse_subjects() -> list[str]:
    raw = os.getenv("SUBJECTS", "").strip()
    if not raw:
        return [str(s) for s in range(13, 53)]
    tokens = raw.replace(",", " ").split()
    return [str(int(t)) for t in tokens]


def parse_modalities() -> list[str]:
    raw = os.getenv("MODALITIES", "elec,mecha").strip()
    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    mapping = {
        "electrical": "elec",
        "mechanical": "mecha",
        "elec": "elec",
        "mecha": "mecha",
    }
    normalized = [mapping[t.lower()] for t in tokens if t.lower() in mapping]
    return normalized if normalized else ["elec", "mecha"]


def resolve_loader_dir() -> Path:
    candidates = [Path.cwd()]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[3])

    for base in candidates:
        candidate = base / "Analysis" / "SourceReconstruction" / "NullingBeamformer"
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not locate Analysis/SourceReconstruction/NullingBeamformer")


def extract_eeg_from_epochs(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, mne.Info, list[str]]:
    """
    Extract EEG channels and metadata from an Epochs object.

    Returns:
        y_st: (n_trials, n_times, n_channels)
        times: (n_times,)
        meta: DataFrame with n_trials rows
        info_eeg: Info for the selected EEG channels
        eeg_ch_names: channel names in data order
    """
    if epochs.metadata is None:
        raise ValueError("Epochs metadata is missing; cannot fit model.")

    epochs_eeg = epochs.copy().pick("eeg")
    if len(epochs_eeg.ch_names) == 0:
        raise ValueError("No EEG channels found after picking eeg channels.")

    data = epochs_eeg.get_data(copy=False)  # (n_trials, n_channels, n_times)
    y_st = np.transpose(data, (0, 2, 1))  # (n_trials, n_times, n_channels)
    times = epochs_eeg.times.copy()
    meta = epochs_eeg.metadata.reset_index(drop=True).copy()

    # Data integrity rule: never force row alignment.
    if y_st.shape[0] != len(meta):
        raise ValueError(
            "Row mismatch between metadata and EEG trials after loading epochs: "
            f"metadata={len(meta)}, eeg_trials={y_st.shape[0]}"
        )

    return y_st, times, meta, epochs_eeg.info.copy(), epochs_eeg.ch_names


def prepare_design(
    events: pd.DataFrame,
    include_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "response_decoded" not in events.columns:
        raise ValueError("Missing required metadata column: response_decoded")

    resp_raw = events["response_decoded"].astype(str).str.strip().str.lower()
    mapping = {"yes": 1.0, "no": 0.0}
    if not resp_raw.isin(mapping).all():
        bad_levels = sorted(resp_raw[~resp_raw.isin(mapping)].unique())
        raise ValueError(
            "response_decoded contains unsupported levels. "
            f"Expected Yes/No, got: {bad_levels}"
        )
    response = resp_raw.map(mapping).to_numpy(dtype=float)

    if "intensity_level_centered" in events.columns:
        intensity = pd.to_numeric(events["intensity_level_centered"], errors="coerce").to_numpy(dtype=float)
    elif "intensity_level" in events.columns:
        raw_intensity = pd.to_numeric(events["intensity_level"], errors="coerce").to_numpy(dtype=float)
        intensity = raw_intensity - np.nanmean(raw_intensity)
    else:
        raise ValueError("Need either 'intensity_level_centered' or 'intensity_level' in metadata.")

    if np.isnan(intensity).any():
        raise ValueError("Found NaN in intensity predictor after conversion.")

    columns = [np.ones_like(response), response, intensity]
    predictor_names = ["Intercept", "response_decoded", "intensity_level_centered"]

    if include_interaction:
        columns.append(response * intensity)
        predictor_names.append("response_x_intensity")

    design = np.column_stack(columns)
    return design, response, predictor_names


def fit_time_channel_ols(
    y_st: np.ndarray,
    design: np.ndarray,
    predictor_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Fit OLS over all time*channel features.

    y_st: (n_trials, n_times, n_channels)
    Returns per effect arrays shaped (n_times, n_channels).
    """
    n_trials, n_times, n_channels = y_st.shape
    y = y_st.reshape(n_trials, n_times * n_channels)

    xtx_inv = np.linalg.pinv(design.T @ design)
    betas = xtx_inv @ design.T @ y

    resid = y - design @ betas
    dof = design.shape[0] - design.shape[1]
    if dof <= 0:
        raise ValueError("Not enough trials for OLS degrees of freedom.")

    sigma2 = np.sum(resid ** 2, axis=0) / dof

    out: dict[str, dict[str, np.ndarray]] = {}
    for idx, name in enumerate(predictor_names):
        if name == "Intercept":
            continue

        se = np.sqrt(np.maximum(sigma2, np.finfo(float).tiny) * xtx_inv[idx, idx])
        beta = betas[idx, :]
        t_val = beta / se
        p_val = 2.0 * stats.t.sf(np.abs(t_val), df=dof)

        out[name] = {
            "beta": beta.reshape(n_times, n_channels),
            "t": t_val.reshape(n_times, n_channels),
            "p": p_val.reshape(n_times, n_channels),
        }

    return out


def cluster_test_for_effect(
    y_st: np.ndarray,
    target: np.ndarray,
    nuisance: np.ndarray,
    adjacency,
    n_permutations: int,
    cluster_alpha: float,
    seed: int,
):
    """
    Spatio-temporal cluster test for one effect using residualization.
    """
    pz = nuisance @ np.linalg.pinv(nuisance)

    target_res = target - pz @ target
    y_res = y_st - (pz @ y_st.reshape(y_st.shape[0], -1)).reshape(y_st.shape)

    if np.allclose(target_res, 0.0):
        raise ValueError("Target predictor is fully explained by nuisance predictors.")

    effect_samples = target_res[:, None, None] * y_res  # (n_trials, n_times, n_channels)
    df = effect_samples.shape[0] - 1
    threshold = stats.t.ppf(1.0 - cluster_alpha / 2.0, df=df)

    t_obs, clusters, cluster_p_values, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
        effect_samples,
        adjacency=adjacency,
        threshold=threshold,
        tail=0,
        n_permutations=n_permutations,
        seed=seed,
        out_type="mask",
        verbose=False,
    )

    return t_obs, clusters, cluster_p_values, h0


def run_spatiotemporal_tests(
    y_st: np.ndarray,
    design: np.ndarray,
    predictor_names: list[str],
    adjacency,
    n_permutations: int,
    cluster_alpha: float,
    seed: int,
):
    out = {}
    for idx, name in enumerate(predictor_names):
        if name == "Intercept":
            continue

        nuisance_idx = [j for j in range(design.shape[1]) if j != idx]
        nuisance = design[:, nuisance_idx]
        target = design[:, idx]

        t_obs, clusters, cluster_p_values, h0 = cluster_test_for_effect(
            y_st=y_st,
            target=target,
            nuisance=nuisance,
            adjacency=adjacency,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            seed=seed,
        )

        out[name] = {
            "t_obs": t_obs,
            "clusters": clusters,
            "cluster_p_values": cluster_p_values,
            "h0": h0,
        }

    return out


def build_cluster_summary(
    times: np.ndarray,
    ch_names: list[str],
    cluster_results,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for effect_name, res in cluster_results.items():
        t_obs = res["t_obs"]
        clusters = res["clusters"]
        pvals = res["cluster_p_values"]

        for c_idx, (mask, p_val) in enumerate(zip(clusters, pvals)):
            if p_val >= alpha:
                continue

            ti, ci = np.where(mask)
            if ti.size == 0:
                continue

            abs_t = np.abs(t_obs)
            peak_flat = np.argmax(abs_t * mask)
            peak_ti, peak_ci = np.unravel_index(peak_flat, t_obs.shape)

            rows.append(
                {
                    "effect": effect_name,
                    "cluster_id": c_idx,
                    "p_cluster": float(p_val),
                    "time_start_s": float(times[ti.min()]),
                    "time_end_s": float(times[ti.max()]),
                    "n_timepoints": int(np.unique(ti).size),
                    "n_channels": int(np.unique(ci).size),
                    "n_points": int(mask.sum()),
                    "peak_t": float(t_obs[peak_ti, peak_ci]),
                    "peak_time_s": float(times[peak_ti]),
                    "peak_channel": ch_names[peak_ci],
                }
            )

    return pd.DataFrame(rows)


def plot_effect(
    effect_name: str,
    times: np.ndarray,
    info_eeg: mne.Info,
    ch_names: list[str],
    t_obs: np.ndarray,
    clusters,
    cluster_p_values: np.ndarray,
    alpha: float,
    out_file: Path,
):
    # combined significant mask
    sig_mask = np.zeros_like(t_obs, dtype=bool)
    for mask, p_val in zip(clusters, cluster_p_values):
        if p_val < alpha:
            sig_mask |= mask

    sig_count_per_time = sig_mask.sum(axis=1)
    peak_time_idx = int(np.argmax(np.abs(t_obs).max(axis=1)))
    peak_t = t_obs[peak_time_idx, :]
    peak_mask = sig_mask[peak_time_idx, :]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    ax = axes[0]
    ax.plot(times, sig_count_per_time, color="black")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("# significant channels")
    ax.set_title(f"{effect_name}: significant sensors over time")

    ax = axes[1]
    im = ax.imshow(
        t_obs.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, len(ch_names)],
        cmap="RdBu_r",
        vmin=-np.nanmax(np.abs(t_obs)),
        vmax=np.nanmax(np.abs(t_obs)),
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("EEG channel index")
    ax.set_title(f"{effect_name}: t-map (time x channel)")
    fig.colorbar(im, ax=ax, label="t")

    ax = axes[2]
    mne.viz.plot_topomap(
        peak_t,
        info_eeg,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        contours=0,
        mask=peak_mask,
        mask_params=dict(markersize=6, markerfacecolor="k", markeredgecolor="k"),
    )
    ax.set_title(f"{effect_name}: topomap at t={times[peak_time_idx]:.3f} s")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# %%
# Load DataLoader
loader_dir = resolve_loader_dir()
if str(loader_dir) not in sys.path:
    sys.path.insert(0, str(loader_dir))

from data_loading_helper import DataLoader  # noqa: E402

# DataLoader.load_epochs duplicates general_helpers.load_clean_epochs and so does NOT pick up
# the subject exclusion; apply it explicitly in the loop below.
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from general_helpers import is_subject_excluded  # noqa: E402


# %%
# Data loading
subjects = parse_subjects()
if SUBJECT_LIMIT is not None:
    subjects = subjects[:SUBJECT_LIMIT]
modalities = parse_modalities()

all_y = []
all_meta = []
times_ref = None
info_ref = None
ch_names_ref = None

for sub_id in subjects:
    for modality in modalities:
        # Sessions on Pre-Processing/exclued_subs.json (low post-cue ERP SNR in O1/O2).
        if is_subject_excluded(sub_id, modality):
            print(f"Skipping sub-{sub_id} {modality}: on the exclusion list.")
            continue

        print("loading sub", sub_id, modality)

        dl = DataLoader(DATADIR, RECONDIR, sub_id, modality)
        try:
            epochs_clean, _ = dl.load_epochs(suffix=EPOCH_SUFFIX, exclude_analysis=True)

            if len(epochs_clean) == 0:
                print("no epochs left after cleaning, skipping")
                continue

            if "response_decoded" in epochs_clean.metadata.columns:
                epochs_clean = epochs_clean["response_decoded!='nan'"]

        except FileNotFoundError:
            print(f"Epoch file not found for sub-{sub_id} {modality}, skipping...")
            continue

        y_st, times, meta, info_eeg, ch_names = extract_eeg_from_epochs(epochs_clean)

        if times_ref is None:
            times_ref = times
            info_ref = info_eeg
            ch_names_ref = ch_names
        else:
            if times_ref.shape != times.shape or not np.allclose(times_ref, times, atol=1e-9):
                raise ValueError(f"Time axis mismatch for sub-{sub_id} {modality}.")
            if ch_names_ref != ch_names:
                raise ValueError(f"Channel order mismatch for sub-{sub_id} {modality}.")

        all_y.append(y_st)
        all_meta.append(meta)

if not all_y:
    raise FileNotFoundError("No epochs loaded. Check DATADIR, SUBJECTS, MODALITIES, and EPOCH_SUFFIX.")

y_st = np.concatenate(all_y, axis=0)
events = pd.concat(all_meta, ignore_index=True)

if y_st.shape[0] != len(events):
    raise RuntimeError(
        "Global mismatch between metadata and EEG trials: "
        f"metadata={len(events)}, eeg_trials={y_st.shape[0]}"
    )

print(f"Subjects included: {len(subjects)} | Modalities: {modalities}")
print(f"Data shape (trials, times, channels): {y_st.shape}")
print(f"EEG channels: {len(ch_names_ref)}")


# %%
# Design + adjacency
design, response, predictor_names = prepare_design(
    events=events,
    include_interaction=INCLUDE_INTERACTION,
)

adjacency, adj_ch_names = mne.channels.find_ch_adjacency(info_ref, ch_type="eeg")
if list(adj_ch_names) != list(ch_names_ref):
    raise ValueError("Adjacency channel order does not match data channel order.")

print("Adjacency matrix shape:", adjacency.shape)
print("Predictors:", predictor_names)


# %%
# OLS + spatio-temporal cluster tests
ols_results = fit_time_channel_ols(y_st=y_st, design=design, predictor_names=predictor_names)
cluster_results = run_spatiotemporal_tests(
    y_st=y_st,
    design=design,
    predictor_names=predictor_names,
    adjacency=adjacency,
    n_permutations=N_PERMUTATIONS,
    cluster_alpha=CLUSTER_ALPHA,
    seed=SEED,
)

effects = [name for name in predictor_names if name != "Intercept"]

for effect_name in effects:
    pvals = cluster_results[effect_name]["cluster_p_values"]
    sig_idx = np.where(pvals < ALPHA)[0]
    print(f"\nEffect: {effect_name}")
    print(f"Significant clusters (p < {ALPHA}): {len(sig_idx)}")


# %%
# Save outputs + plots
outdir = Path(RESDIR) / "Figures" / "Main" / "eeg_results" / "cluster_permutation_spatiotemporal"
outdir.mkdir(parents=True, exist_ok=True)

# cluster summary
cluster_summary = build_cluster_summary(
    times=times_ref,
    ch_names=ch_names_ref,
    cluster_results=cluster_results,
    alpha=ALPHA,
)
summary_file = outdir / "spatiotemporal_cluster_summary.tsv"
cluster_summary.to_csv(summary_file, sep="\t", index=False)
print("Saved cluster summary:", summary_file)

# time x channel maps for each effect
for effect_name in effects:
    effect_file = outdir / f"spatiotemporal_{effect_name}_channel-time-topo.png"
    plot_effect(
        effect_name=effect_name,
        times=times_ref,
        info_eeg=info_ref,
        ch_names=ch_names_ref,
        t_obs=cluster_results[effect_name]["t_obs"],
        clusters=cluster_results[effect_name]["clusters"],
        cluster_p_values=cluster_results[effect_name]["cluster_p_values"],
        alpha=ALPHA,
        out_file=effect_file,
    )
    print("Saved figure:", effect_file)

# full beta/t/p maps
for effect_name in effects:
    df_map = pd.DataFrame(
        {
            "time_s": np.repeat(times_ref, len(ch_names_ref)),
            "channel": np.tile(ch_names_ref, len(times_ref)),
            "beta": ols_results[effect_name]["beta"].reshape(-1),
            "t": ols_results[effect_name]["t"].reshape(-1),
            "p": ols_results[effect_name]["p"].reshape(-1),
        }
    )
    map_file = outdir / f"spatiotemporal_{effect_name}_beta_t_p.tsv"
    df_map.to_csv(map_file, sep="\t", index=False)
    print("Saved map:", map_file)
