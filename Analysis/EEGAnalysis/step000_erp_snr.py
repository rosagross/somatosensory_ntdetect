"""
File: step000_erp_snr.py
Author: Rosa Grossmann
Description:
- Compute the signal-to-noise ratio (SNR) of the ERP traces for both modalities.
- The "noise" reference is the pre-cue baseline window; the "signal" is the ERP in two
  later windows, giving TWO SNR values per modality/channel:
    * post-cue  (anticipatory activity): (-0.4, -0.1) s
    * post-stim (evoked response)      : ( 0.0,  0.2) s
- Baseline and noise windows are modality specific and separate (cue at -0.43 mecha,
  -0.4 elec). Each ERP is re-baselined to the 100 ms before the cue:
    * baseline: mecha (-0.53, -0.43) s, elec (-0.5, -0.4) s
  The SNR noise reference is the whole pre-cue stretch up to the cue:
    * noise:    mecha (-0.7, -0.43) s,  elec (-0.7, -0.4) s
- Two SNR metrics per window: RMS amplitude ratio and variance (power) ratio, both in dB.
- Subject-first: per-subject ERP -> per-subject SNR -> group statistics.
- ALSO computes a trial-by-trial SNR (noise vs post-cue only): for each single trial,
  the RMS/variance ratio of the post-cue window over the noise window. This is a within-
  trial amplitude ratio (not phase-locked signal vs noise), so ~0 dB is expected.
"""

# %%
import os
import sys
import argparse
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from plotting_style import figure_style
import eeg_analysis_helpers as ah

colors, _ = figure_style()

DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
WDDIR = os.getenv("WDDIR")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import load_clean_epochs

# ---- Analysis parameters -------------------------------------------------------------
modalities = ["elec", "mecha"]
channels = ["CP4", "C6", "O1", "O2"]

l_freq_orig = 0.1
l_freq = 1
h_freq = 40
file_suffix_epochs = "_preprestim"

# Windows (seconds). Modality specific because the cue time differs (-0.43 mecha, -0.4 elec).
_cue_time_by_modality = {"mecha": -0.43, "elec": -0.4}
# Baseline: 100 ms before the cue, applied to each ERP with apply_baseline.
BASELINE_WIN = {"mecha": (-0.53, -0.43), "elec": (-0.5, -0.4)}
# Noise reference for the SNR denominator: the whole pre-cue stretch up to the cue.
NOISE_WIN = {"mecha": (-0.7, -0.43), "elec": (-0.7, -0.4)}
SIGNAL_WINS = {
    "postcue": (-0.4, -0.1),   # anticipatory activity between cue and stimulus
    "poststim": (0.0, 0.2),    # somatosensory evoked response
}
# A subject/channel counts as "above" when the signal RMS is at least 1.5x the pre-cue
# baseline-noise RMS. Applied to the amplitude (rms) metric only -- the variance metric
# drops the window mean, which is exactly the slow shift the post-cue window is about.
# NOTE: this is a stricter reference than the one-sample tests in summarise(), which
# test against 0 dB (ratio 1.0).
SNR_THRESHOLD_RATIO = 1.5
SNR_THRESHOLD_DB = 20.0 * np.log10(SNR_THRESHOLD_RATIO)   # 3.52 dB

# Standard voltage ERP (step001_ERPvisualization/step008 use a surface Laplacian; here we
# keep raw voltage so the SNR is interpretable as amplitude above baseline noise).
APPLY_LAPLACIAN = False

# Modality-specific colours for plotting.
palette_modality = {"elec": colors["electrical"], "mecha": colors["mechanical"]}

# Output directories.
stats_dir = os.path.join(RESDIR, "Statistics", "eeg", "erp_snr")
fig_dir = os.path.join(RESDIR, "Figures", "Main", "eeg_results", "erp_snr")
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


def parse_args():
    """Parse --sub (integer IDs, default 13-52) and --modalities (default both)."""
    parser = argparse.ArgumentParser(description="Compute ERP SNR per modality.")
    parser.add_argument("--sub", type=int, nargs="+", default=list(range(13, 53)),
                        help="Subject IDs as integers (e.g. --sub 13 14 15).")
    parser.add_argument("--modalities", nargs="+", default=modalities,
                        help="Modalities to process (elec mecha).")
    args, _ = parser.parse_known_args()
    return args.sub, args.modalities


def window_mask(times, win):
    """Boolean mask for a (tmin, tmax) window; crash if it selects nothing (no fallback)."""
    tmin, tmax = win
    mask = (times >= tmin) & (times <= tmax)
    if not np.any(mask):
        raise ValueError(f"Window {win} does not overlap the ERP time axis "
                         f"({times[0]:.3f}..{times[-1]:.3f} s).")
    return mask


def rms(x):
    """Root-mean-square of a 1-D signal."""
    return float(np.sqrt(np.mean(np.square(x))))


# %%
# ---- Per-subject SNR computation -----------------------------------------------------
def compute_snr():
    subjects, mods = parse_args()
    rows = []
    # Cache subject-level ERP traces for the diagnostic grand-average figure.
    erp_traces = {mod: {ch: [] for ch in channels} for mod in mods}
    erp_times = None

    for modality in mods:
        for subID in subjects:
            # Faulty recordings: subject 23 (swapped ground/reference) and 36 (extreme force),
            # only for the mechanical session.
            if subID in (23, 36) and modality == "mecha":
                print(f"Skipping sub-0{subID} {modality} (faulty recording).")
                continue

            sub_dir = os.path.join(
                DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", f"ses-{modality}"
            )
            base_name = (
                f"sub-0{subID}_ses-{modality}_task-NT_"
                f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}"
            )
            epoch_fif = os.path.join(sub_dir, f"{base_name}_epochs{file_suffix_epochs}.fif")
            if not os.path.exists(epoch_fif):
                print(f"epoch file missing, skipping sub-0{subID} {modality}")
                continue

            # apply_subject_exclusion=False: this script PRODUCES the exclusion list, so it
            # must keep measuring the subjects on it - otherwise the criterion becomes
            # circular and unreproducible on the next run.
            epochs, _ = load_clean_epochs(epoch_fif, exclude_analysis=True,
                                          apply_subject_exclusion=False)
            # Match the band pipeline: drop the intensity==0 (catch) trials if present.
            if "intensity_level" in epochs.metadata.columns:
                epochs = epochs[epochs.metadata["intensity_level"] != 0]
            if len(epochs) == 0:
                print(f"no clean trials for sub-0{subID} {modality}, skipping")
                continue

            if APPLY_LAPLACIAN:
                epochs = mne.preprocessing.compute_current_source_density(
                    epochs.copy().pick("eeg")
                )
            # Re-baseline each ERP to the 100 ms before the cue (modality specific).
            epochs.apply_baseline(BASELINE_WIN[modality])

            evoked = epochs.copy().pick(channels).average()
            erp = evoked.get_data()          # (n_channels, n_times)
            times = evoked.times
            if erp_times is None:
                erp_times = times
            n_trials = len(epochs)

            noise_mask = window_mask(times, NOISE_WIN[modality])
            for ch_idx, ch_name in enumerate(evoked.ch_names):
                trace = erp[ch_idx]
                erp_traces[modality][ch_name].append(trace)
                noise_rms = rms(trace[noise_mask])
                noise_var = float(np.var(trace[noise_mask]))
                for win_label, win in SIGNAL_WINS.items():
                    sig_mask = window_mask(times, win)
                    sig_rms = rms(trace[sig_mask])
                    sig_var = float(np.var(trace[sig_mask]))
                    snr_rms = sig_rms / noise_rms
                    snr_var = sig_var / noise_var
                    rows.append(dict(
                        subject=subID,
                        modality=modality,
                        channel=ch_name,
                        window=win_label,
                        snr_rms=snr_rms,
                        snr_rms_db=20.0 * np.log10(snr_rms),
                        snr_var=snr_var,
                        snr_var_db=10.0 * np.log10(snr_var),
                        n_trials=n_trials,
                    ))

    df = pd.DataFrame(rows)
    return df, erp_traces, erp_times


# %%
# ---- Trial-by-trial SNR (noise vs post-cue) ------------------------------------------
def compute_trial_snr():
    """
    Single-trial SNR: post-cue vs noise window, per trial.

    For every trial, the RMS (and variance) in the post-cue window is divided by the
    RMS (and variance) in the noise window of the SAME trial. This is a within-trial
    amplitude ratio, not phase-locked signal vs noise, so ~0 dB is expected. Restricted
    to noise vs post-cue (no post-stim). Returns a long DataFrame, one row per
    trial x channel. Reloads the epochs (a second load after compute_snr) - fine for a
    diagnostic script.
    """
    subjects, mods = parse_args()
    postcue_win = SIGNAL_WINS["postcue"]
    rows = []

    for modality in mods:
        for subID in subjects:
            if subID in (23, 36) and modality == "mecha":
                print(f"Skipping sub-0{subID} {modality} (faulty recording).")
                continue

            sub_dir = os.path.join(
                DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", f"ses-{modality}"
            )
            base_name = (
                f"sub-0{subID}_ses-{modality}_task-NT_"
                f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}"
            )
            epoch_fif = os.path.join(sub_dir, f"{base_name}_epochs{file_suffix_epochs}.fif")
            if not os.path.exists(epoch_fif):
                print(f"epoch file missing, skipping sub-0{subID} {modality}")
                continue

            # See compute_snr(): this script must not exclude the subjects it is measuring.
            epochs, _ = load_clean_epochs(epoch_fif, exclude_analysis=True,
                                          apply_subject_exclusion=False)
            if "intensity_level" in epochs.metadata.columns:
                epochs = epochs[epochs.metadata["intensity_level"] != 0]
            if len(epochs) == 0:
                print(f"no clean trials for sub-0{subID} {modality}, skipping")
                continue

            if APPLY_LAPLACIAN:
                epochs = mne.preprocessing.compute_current_source_density(
                    epochs.copy().pick("eeg")
                )
            # Baseline every trial to the 100 ms before the cue (modality specific).
            epochs.apply_baseline(BASELINE_WIN[modality])

            picked = epochs.copy().pick(channels)
            data = picked.get_data()          # (n_trials, n_channels, n_times)
            times = picked.times
            noise_mask = window_mask(times, NOISE_WIN[modality])
            postcue_mask = window_mask(times, postcue_win)

            for ch_idx, ch_name in enumerate(picked.ch_names):
                seg_n = data[:, ch_idx, noise_mask]      # (n_trials, n_noise)
                seg_s = data[:, ch_idx, postcue_mask]
                noise_rms = np.sqrt(np.mean(np.square(seg_n), axis=1))
                sig_rms = np.sqrt(np.mean(np.square(seg_s), axis=1))
                noise_var = seg_n.var(axis=1)
                sig_var = seg_s.var(axis=1)
                snr_rms = sig_rms / noise_rms
                snr_var = sig_var / noise_var
                for tr in range(data.shape[0]):
                    rows.append(dict(
                        subject=subID,
                        modality=modality,
                        channel=ch_name,
                        trial=tr,
                        snr_rms=snr_rms[tr],
                        snr_rms_db=20.0 * np.log10(snr_rms[tr]),
                        snr_var=snr_var[tr],
                        snr_var_db=10.0 * np.log10(snr_var[tr]),
                    ))

    return pd.DataFrame(rows)


def aggregate_trial_snr(trial_df):
    """Subject-first reduction: mean/median/SEM across trials per subject/modality/channel."""
    def sem(x):
        return float(stats.sem(x)) if len(x) > 1 else 0.0

    g = trial_df.groupby(["subject", "modality", "channel"])
    agg = g.agg(
        n_trials=("trial", "size"),
        mean_snr_rms_db=("snr_rms_db", "mean"),
        median_snr_rms_db=("snr_rms_db", "median"),
        sem_snr_rms_db=("snr_rms_db", sem),
        mean_snr_var_db=("snr_var_db", "mean"),
        median_snr_var_db=("snr_var_db", "median"),
        sem_snr_var_db=("snr_var_db", sem),
    ).reset_index()
    return agg.sort_values(["modality", "channel", "subject"]).reset_index(drop=True)


# %%
# ---- Group statistics ----------------------------------------------------------------
def summarise(df):
    """Group means/SEM, one-sample tests vs 0 dB, and paired elec-vs-mecha tests."""
    metric_db = {"rms": "snr_rms_db", "var": "snr_var_db"}
    summary_rows = []
    p_onesample = []            # collect for a single Bonferroni family

    for metric, db_col in metric_db.items():
        for ch_name in channels:
            for win_label in SIGNAL_WINS:
                for modality in df["modality"].unique():
                    vals = df[(df.channel == ch_name) & (df.window == win_label)
                              & (df.modality == modality)][db_col].to_numpy()
                    vals = vals[np.isfinite(vals)]
                    n = len(vals)
                    mean = float(np.mean(vals)) if n else np.nan
                    sem = float(stats.sem(vals)) if n > 1 else 0.0
                    # Is the signal significantly above the noise (SNR > 0 dB)?
                    if n > 1:
                        _, p_gt0 = stats.ttest_1samp(vals, 0.0)
                    else:
                        p_gt0 = np.nan
                    summary_rows.append(dict(
                        metric=metric, channel=ch_name, window=win_label,
                        modality=modality, n=n, mean_db=mean, sem_db=sem,
                        p_vs0=p_gt0,
                    ))
                    p_onesample.append((len(summary_rows) - 1, p_gt0))

    # Paired elec-vs-mecha comparison per (metric, channel, window), aligned by subject.
    paired_rows = []
    for metric, db_col in metric_db.items():
        for ch_name in channels:
            for win_label in SIGNAL_WINS:
                sub = df[(df.channel == ch_name) & (df.window == win_label)]
                wide = sub.pivot_table(index="subject", columns="modality", values=db_col)
                if {"elec", "mecha"}.issubset(wide.columns):
                    wide = wide.dropna(subset=["elec", "mecha"])
                    if len(wide) > 1:
                        _, p_pair = stats.ttest_rel(wide["elec"], wide["mecha"])
                    else:
                        p_pair = np.nan
                    n_pair = len(wide)
                else:
                    p_pair, n_pair = np.nan, 0
                paired_rows.append(dict(
                    metric=metric, channel=ch_name, window=win_label,
                    n_paired=n_pair, p_elec_vs_mecha=p_pair,
                ))

    summary_df = pd.DataFrame(summary_rows)
    # Bonferroni-correct the family of one-sample (vs 0 dB) tests.
    idxs = [i for i, p in p_onesample if np.isfinite(p)]
    pvals = [p for i, p in p_onesample if np.isfinite(p)]
    summary_df["p_vs0_bonf"] = np.nan
    if pvals:
        corrected = multipletests(pvals, method="bonferroni")[1]
        summary_df.loc[idxs, "p_vs0_bonf"] = corrected
    paired_df = pd.DataFrame(paired_rows)
    return summary_df, paired_df


# %%
# ---- Per-subject threshold table ------------------------------------------------------
def above_threshold_table(df, threshold_ratio=SNR_THRESHOLD_RATIO):
    """
    Which subjects clear the RMS SNR threshold, per window.

    Pure reshape of the per-subject `df` from compute_snr(): one row per
    subject x modality x channel, with both signal windows side by side and a
    boolean per window. Window names come from SIGNAL_WINS so renaming a window
    cannot silently produce all-NaN columns.

    The criterion is a linear amplitude ratio (signal RMS / noise RMS), compared
    against the linear `snr_rms_<win>` columns so no dB round-trip is involved.
    """
    win_labels = list(SIGNAL_WINS.keys())

    wide = df.pivot_table(
        index=["subject", "modality", "channel", "n_trials"],
        columns="window",
        values=["snr_rms", "snr_rms_db"],
    )
    # Flatten the ("snr_rms_db", "postcue") MultiIndex to "snr_rms_db_postcue".
    wide.columns = [f"{metric}_{win}" for metric, win in wide.columns]
    wide = wide.reset_index()

    db_cols = [f"snr_rms_db_{win}" for win in win_labels]
    # The booleans below are derived from the linear ratio columns, so those are the ones
    # that must be present and complete.
    ratio_cols = [f"snr_rms_{win}" for win in win_labels]
    missing = [c for c in ratio_cols if c not in wide.columns]
    if missing:
        raise ValueError(
            f"Windows {missing} are absent from the SNR table. Expected one column per "
            f"entry of SIGNAL_WINS ({win_labels})."
        )

    # compute_snr() always fills every window for a subject/channel it processed, so a
    # NaN here means the pivot lost rows - crash rather than turn it into a False.
    incomplete = wide[wide[ratio_cols].isna().any(axis=1)]
    if len(incomplete):
        raise ValueError(
            f"{len(incomplete)} subject/modality/channel rows are missing at least one "
            f"signal window:\n{incomplete[['subject', 'modality', 'channel'] + ratio_cols]}"
        )

    above_cols = []
    for win in win_labels:
        col = f"above_{win}"
        wide[col] = wide[f"snr_rms_{win}"] > threshold_ratio
        above_cols.append(col)

    wide["above_both"] = wide[above_cols].all(axis=1)
    wide["above_either"] = wide[above_cols].any(axis=1)
    # Keep the file self-documenting: record the criterion in both units.
    wide["threshold_ratio"] = threshold_ratio
    wide["threshold_db"] = 20.0 * np.log10(threshold_ratio)

    ordered = (
        ["subject", "modality", "channel", "n_trials"]
        + ratio_cols
        + db_cols
        + above_cols
        + ["above_both", "above_either", "threshold_ratio", "threshold_db"]
    )
    return wide[ordered].sort_values(["modality", "channel", "subject"]).reset_index(drop=True)


def below_threshold_table(above_df):
    """
    Long-format list of every subject/modality/channel/window that FAILS the criterion.

    One row per failure (not per subject), so a subject that fails both windows or several
    channels appears once per failing combination. Derived purely from the booleans in
    `above_df`, so it can never disagree with the threshold table.
    """
    rows = []
    for win in SIGNAL_WINS:
        fail = above_df[~above_df[f"above_{win}"]]
        rows.append(pd.DataFrame(dict(
            subject=fail["subject"].astype(int),
            modality=fail["modality"],
            channel=fail["channel"],
            window=win,
            n_trials=fail["n_trials"],
            snr_rms=fail[f"snr_rms_{win}"],
            snr_rms_db=fail[f"snr_rms_db_{win}"],
            threshold_ratio=fail["threshold_ratio"],
            threshold_db=fail["threshold_db"],
        )))
    below = pd.concat(rows, ignore_index=True)
    return below.sort_values(["modality", "channel", "window", "subject"]).reset_index(drop=True)


def print_above_threshold_counts(above_df):
    """Console summary: how many subjects clear the threshold, per modality/channel/window."""
    print(f"\nSubjects above {above_df['threshold_ratio'].iloc[0]:.2f}x RMS SNR "
          f"({above_df['threshold_db'].iloc[0]:.2f} dB):")
    for modality in above_df["modality"].unique():
        for ch_name in above_df["channel"].unique():
            sub = above_df[(above_df.modality == modality) & (above_df.channel == ch_name)]
            if not len(sub):
                continue
            counts = "  ".join(
                f"{win}: {int(sub[f'above_{win}'].sum())}/{len(sub)}" for win in SIGNAL_WINS
            )
            print(f"  {modality:<6s} {ch_name:<4s}  {counts}")
            # List the subjects that fail, so they can be looked at individually.
            for win in SIGNAL_WINS:
                below = sorted(sub.loc[~sub[f"above_{win}"], "subject"].astype(int))
                if below:
                    print(f"      below ({win}): {', '.join(str(s) for s in below)}")


# %%
# ---- Plotting ------------------------------------------------------------------------
def stars(p):
    """Significance stars for a p-value."""
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_snr_bars(df, summary_df, paired_df, metric):
    """Bar figure of SNR (dB) by window x modality, one row per channel."""
    db_col = {"rms": "snr_rms_db", "var": "snr_var_db"}[metric]
    win_labels = list(SIGNAL_WINS.keys())
    mods = [m for m in modalities if m in df["modality"].unique()]

    fig, axes = plt.subplots(
        nrows=len(channels), ncols=1,
        figsize=(1.8 + 2.4 * len(win_labels), 4.2 * len(channels)),
        squeeze=False,
    )
    x = np.arange(len(win_labels))
    bar_w = 0.8 / max(len(mods), 1)

    for row_idx, ch_name in enumerate(channels):
        ax = axes[row_idx, 0]
        for m_idx, modality in enumerate(mods):
            means, sems = [], []
            for win_label in win_labels:
                s = summary_df[(summary_df.metric == metric) & (summary_df.channel == ch_name)
                               & (summary_df.window == win_label)
                               & (summary_df.modality == modality)]
                means.append(s["mean_db"].values[0] if len(s) else np.nan)
                sems.append(s["sem_db"].values[0] if len(s) else 0.0)
            offset = (m_idx - (len(mods) - 1) / 2) * bar_w
            ax.bar(x + offset, means, yerr=sems, width=bar_w,
                   color=palette_modality.get(modality, "0.5"),
                   capsize=4, label=modality, zorder=2)
            # Per-subject scatter.
            for w_idx, win_label in enumerate(win_labels):
                vals = df[(df.channel == ch_name) & (df.window == win_label)
                          & (df.modality == modality)][db_col].to_numpy()
                vals = vals[np.isfinite(vals)]
                ax.scatter(np.full_like(vals, x[w_idx] + offset), vals,
                           color="black", s=8, alpha=0.5, zorder=3)
            # Annotate each bar with the linear ratio (×) matching the plotted dB height,
            # plus the vs-0 dB significance star (Bonferroni corrected).
            # dB -> linear: amplitude (rms) uses /20, power (var) uses /10.
            db_to_linear = 20.0 if metric == "rms" else 10.0
            for w_idx, win_label in enumerate(win_labels):
                s = summary_df[(summary_df.metric == metric) & (summary_df.channel == ch_name)
                               & (summary_df.window == win_label)
                               & (summary_df.modality == modality)]
                if not len(s):
                    continue
                m, sem = means[w_idx], sems[w_idx]
                sign = 1.0 if m >= 0 else -1.0
                ratio = 10.0 ** (m / db_to_linear)   # linear × ratio for the plotted bar
                ax.text(x[w_idx] + offset, m + sign * sem,
                        f"{ratio:.2f}×\n{stars(s['p_vs0_bonf'].values[0])}",
                        ha="center", va="bottom" if sign > 0 else "top", fontsize=6.5)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(win_labels)
        ax.set_ylabel(f"SNR ({'amplitude' if metric == 'rms' else 'power'}) [dB]")
        # Annotate the paired elec-vs-mecha test in the title.
        p_lines = []
        for win_label in win_labels:
            pr = paired_df[(paired_df.metric == metric) & (paired_df.channel == ch_name)
                           & (paired_df.window == win_label)]
            p_val = pr["p_elec_vs_mecha"].values[0] if len(pr) else np.nan
            p_lines.append(f"{win_label}: elec vs mecha p={p_val:.3g}")
        ax.set_title(f"{ch_name}\n" + "  |  ".join(p_lines), fontsize=9)
        if row_idx == 0:
            ax.legend(frameon=False)
        sns.despine(trim=True, ax=ax)

    fig.suptitle(f"ERP SNR ({metric}) vs pre-cue baseline noise", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(fig_dir, f"erp_snr_{metric}_bars_{file_suffix_epochs}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagnostic_erp(erp_traces, erp_times):
    """Grand-average ERP per modality/channel with the SNR windows shaded."""
    if erp_times is None:
        print("No ERP data to plot for diagnostics.")
        return
    mods = [m for m in modalities if any(len(erp_traces[m][ch]) for ch in channels)]
    fig, axes = plt.subplots(
        nrows=len(channels), ncols=1,
        figsize=(9, 3.4 * len(channels)), squeeze=False, sharex=True,
    )
    for row_idx, ch_name in enumerate(channels):
        ax = axes[row_idx, 0]
        # Shade the signal windows (shared across modalities).
        ax.axvspan(*SIGNAL_WINS["postcue"], color=colors["unperceived"], alpha=0.12,
                   label="post-cue")
        ax.axvspan(*SIGNAL_WINS["poststim"], color=colors["perceived"], alpha=0.12,
                   label="post-stim")
        # Noise window is modality specific (ends at the cue): shade one per modality.
        for modality in mods:
            ax.axvspan(*NOISE_WIN[modality], color=palette_modality.get(modality, "0.5"),
                       alpha=0.08, label=f"noise {modality}")
        for modality in mods:
            stack = erp_traces[modality][ch_name]
            if not stack:
                continue
            ga = np.mean(np.stack(stack, axis=0), axis=0)  # subject-first grand average
            # These FIFs store data in µV (channel unit is mislabelled as V), so
            # get_data() already returns µV — do NOT rescale by 1e6.
            ax.plot(erp_times, ga, color=palette_modality.get(modality, "0.5"),
                    label=modality)
        # Mark both cue times so the per-modality noise edges are visible.
        for modality in mods:
            ax.axvline(_cue_time_by_modality[modality], color=palette_modality.get(modality, "0.5"),
                       linestyle=":", linewidth=1, alpha=0.7)
        ah.add_event_lines(ax, erp_times, cue_time=-0.4)
        ax.set_ylabel(f"{ch_name}\nERP (µV)")
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=7, ncol=2)
        sns.despine(trim=True, ax=ax)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("Grand-average ERP with SNR windows shaded", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(fig_dir, f"erp_snr_diagnostic_erp_{file_suffix_epochs}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_trial_snr(trial_df, agg_df):
    """
    Distribution of the within-trial post-cue/noise SNR (rms, dB).

    Subject-first: one point per subject (mean across its trials); the bar is the
    group mean +- SEM across subjects. ~0 dB is expected (both windows are ongoing
    pre-stimulus EEG), so the 0 dB line is the reference, not the threshold. The
    dotted line marks the 1.5x amplitude criterion used for the averaged ERPs, shown
    for scale only - single trials are not expected to reach it.
    """
    mods = [m for m in modalities if m in trial_df["modality"].unique()]
    fig, axes = plt.subplots(
        nrows=len(channels), ncols=1,
        figsize=(1.8 + 2.2 * max(len(mods), 1), 4.2 * len(channels)),
        squeeze=False,
    )
    x = np.arange(len(mods))

    for row_idx, ch_name in enumerate(channels):
        ax = axes[row_idx, 0]
        means, sems = [], []
        for modality in mods:
            subj_means = agg_df[(agg_df.channel == ch_name) & (agg_df.modality == modality)][
                "mean_snr_rms_db"].to_numpy()
            subj_means = subj_means[np.isfinite(subj_means)]
            means.append(float(np.mean(subj_means)) if len(subj_means) else np.nan)
            sems.append(float(stats.sem(subj_means)) if len(subj_means) > 1 else 0.0)

        ax.bar(x, means, yerr=sems, width=0.6,
               color=[palette_modality.get(m, "0.5") for m in mods],
               capsize=4, zorder=2)
        # Per-subject scatter (mean across that subject's trials).
        for m_idx, modality in enumerate(mods):
            subj_means = agg_df[(agg_df.channel == ch_name) & (agg_df.modality == modality)][
                "mean_snr_rms_db"].to_numpy()
            subj_means = subj_means[np.isfinite(subj_means)]
            ax.scatter(np.full_like(subj_means, x[m_idx]), subj_means,
                       color="black", s=10, alpha=0.5, zorder=3)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axhline(SNR_THRESHOLD_DB, color="0.5", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(mods)
        ax.set_ylabel(f"{ch_name}\ntrial SNR (rms) [dB]")
        sns.despine(trim=True, ax=ax)

    fig.suptitle(
        "Trial-by-trial SNR: within-trial post-cue / noise RMS ratio\n"
        "(per-subject mean across trials; ~0 dB expected, both windows are ongoing EEG)",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(fig_dir, f"erp_snr_trialwise_postcue{file_suffix_epochs}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# %%
if __name__ == "__main__":
    df, erp_traces, erp_times = compute_snr()
    if df.empty:
        raise SystemExit("No SNR rows computed — check data paths and subject list.")

    persub_csv = os.path.join(stats_dir, f"erp_snr_persubject{file_suffix_epochs}.csv")
    df.to_csv(persub_csv, index=False)
    print(f"Saved per-subject SNR: {persub_csv}  ({len(df)} rows)")

    above_df = above_threshold_table(df)
    above_csv = os.path.join(stats_dir, f"erp_snr_above_threshold{file_suffix_epochs}.csv")
    above_df.to_csv(above_csv, index=False)
    print(f"Saved threshold table: {above_csv}  ({len(above_df)} rows)")

    below_df = below_threshold_table(above_df)
    below_csv = os.path.join(stats_dir, f"erp_snr_below_threshold{file_suffix_epochs}.csv")
    below_df.to_csv(below_csv, index=False)
    print(f"Saved below-threshold list: {below_csv}  ({len(below_df)} rows)")

    print_above_threshold_counts(above_df)

    summary_df, paired_df = summarise(df)
    summary_csv = os.path.join(stats_dir, f"erp_snr_summary{file_suffix_epochs}.csv")
    paired_csv = os.path.join(stats_dir, f"erp_snr_modality_comparison{file_suffix_epochs}.csv")
    summary_df.to_csv(summary_csv, index=False)
    paired_df.to_csv(paired_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    print(f"Saved modality comparison: {paired_csv}")

    for metric in ("rms", "var"):
        plot_snr_bars(df, summary_df, paired_df, metric)
    plot_diagnostic_erp(erp_traces, erp_times)

    # Trial-by-trial SNR (noise vs post-cue). Reloads the epochs.
    trial_df = compute_trial_snr()
    if not trial_df.empty:
        trial_csv = os.path.join(stats_dir, f"erp_snr_trialwise{file_suffix_epochs}.csv")
        trial_df.to_csv(trial_csv, index=False)
        print(f"Saved trial-wise SNR: {trial_csv}  ({len(trial_df)} rows)")

        agg_df = aggregate_trial_snr(trial_df)
        agg_csv = os.path.join(stats_dir, f"erp_snr_trialwise_persubject{file_suffix_epochs}.csv")
        agg_df.to_csv(agg_csv, index=False)
        print(f"Saved trial-wise per-subject: {agg_csv}  ({len(agg_df)} rows)")

        plot_trial_snr(trial_df, agg_df)
    else:
        print("No trial-wise SNR rows computed.")
    print("Done.")
