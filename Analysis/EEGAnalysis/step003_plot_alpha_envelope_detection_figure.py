"""
File: step003_plot_alpha_envelope_detection_figure.py
Author: Rosa Grossmann
Date: 2026-08-11
Description:
Publication figure for the pre-stimulus alpha envelope / detection analysis, plus the
statistics table behind it. Three panels:

  A) Detection rate over envelope bins per channel, with the quadratic (inverted-U) test.
  B) Pre-stimulus envelope by detection response (No/Yes), hue = modality. The text in each
     panel gives the FDR-corrected p of a response x modality linear mixed model (both main
     effects and the interaction).
  C) Yes-No envelope topography per modality, marking the peak channel and latency.

"""

# %%
import argparse
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.signal import hilbert as scipy_hilbert
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from plotting_style import figure_style
import eeg_analysis_helpers as ah

# %%
# ---------------------------------------------------------------- configuration

DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
WDDIR = os.getenv("WDDIR")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

# Imported after WDDIR is on the path: this is the cheap pre-load exclusion check, and it
# is the only thing standing between us and silently re-admitting the SNR-excluded
# sessions -- see the note on the subject gate below.
from general_helpers import is_subject_excluded

FILE_SUFFIX_EPOCHS = "_preprestim_corrected"
FILE_SUFFIX_META = "_preprestim"

ALL_SUBJECTS = list(range(13, 53))
MODALITIES = ["elec", "mecha"]
CHANNELS = ["CP4", "C4", "O2"]
BANDS = ["alpha"]
N_BINS = 10

# Lower edge of each band, used ONLY to size the Hilbert reflect padding (3 cycles of the
# slowest component). It is never used to filter: the *_{band}.fif files are already
# band-pass filtered by step002_extract_freq_bands.py, and for alpha the passband is the
# subject's individual CP4 peak +/- 2 Hz. Re-filtering or re-deriving the peak here would
# silently disagree with the saved data.
BAND_FMIN = {"delta": 2, "alpha": 8, "beta": 13}

# Tolerance (s) when cropping the full-epoch envelope to a window, so a boundary sample
# is not lost to float rounding of the time axis.
TIME_EPS = 1e-6

# Binning/envelope window, identical for both modalities (kept as a per-modality dict).
# It ends at -0.1 s because the zero-phase FIR of the filter-first epochs lets the
# post-stimulus P50 leak backwards into the last ~100 ms before stimulus onset.
BIN_WINDOW = {"elec": (-0.4, -0.1), "mecha": (-0.4, -0.1)}
# topography window
TOPO_WINDOW = dict(tmin=-0.4, tmax=-0.1, mean_win=(-0.4, -0.1))

# Sessions with a faulty mechanical recording setup
# due to inconsistent cue-stimulus triggers 
FAULTY_SESSIONS = {("mecha", 23), ("mecha", 36)} 

MODALITY_LABELS = {"elec": "Electrical", "mecha": "Mechanical"}

# Fixed effects of the panel B mixed model and the term each one is read from.
LMM_EFFECTS = {
    "response": "response_c",
    "modality": "modality_c",
    "response:modality": "response_c:modality_c",
}
# Label kept as "late" so rows line up with the window column of the stats table.
WINDOW_LABEL = "late"

colors, _ = figure_style()
PALETTE = {"elec": colors["electrical"], "mecha": colors["mechanical"]}

FIGUREDIR = os.path.join(WDDIR, "Writing", "img")
STATSDIR = os.path.join(RESDIR, "Statistics", "eeg", "envelope_bin_detection")


def running_in_kernel():
    """
    True when the script is executed as '# %%' cells in Jupyter/VS Code.

    ipykernel appends its own '--f=<connection file>' to sys.argv, which argparse would
    reject, so in that case the CLI is skipped and the defaults above are used.
    """
    return os.path.basename(sys.argv[0]).startswith("ipykernel_launcher")


def parse_args(argv=None):
    """CLI. ah.parse_subjects() only knows --sub, so the parser is built here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", nargs="+", default=BANDS, choices=sorted(BAND_FMIN),
                        help="Frequency bands to process (default: alpha).")
    parser.add_argument("--sub", nargs="+", type=int, default=ALL_SUBJECTS,
                        help="Subject IDs (default: 13-52).")
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES,
                        help="Modalities to process (default: both).")
    if argv is None:
        argv = [] if running_in_kernel() else sys.argv[1:]
    # Parsed strictly, so a typo on the command line is still an error.
    return parser.parse_args(argv)


# %%
# ---------------------------------------------------------------- analysis helpers


def hilbert_envelope(data, sfreq, fmin):
    """
    Instantaneous amplitude via pad -> Hilbert -> trim.

    Reflect-padding by three cycles of the slowest band component keeps the edge
    transients of the Hilbert transform out of the window of interest. It is applied to
    the full filter-first epoch (-0.8 to 0.1 s); cropping to the analysis window
    happens afterwards.
    """
    n_pad = int(3 * sfreq / fmin)
    padded = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode="reflect")
    return np.abs(scipy_hilbert(padded, axis=-1)[:, :, n_pad:-n_pad])


def fit_quadratic_coefficients(x_vals, y_mat):
    """
    Fit a quadratic per subject and return the quadratic terms.

    x_vals : 1-D array of bin indices (same length as y_mat columns)
    y_mat  : (n_subjects, n_bins) matrix with NaNs for bins a subject has no trials in
    """
    x_vals = np.asarray(x_vals, dtype=float)
    a_vals = []
    for row in y_mat:
        row = np.asarray(row, dtype=float)
        mask = np.isfinite(row)
        if np.sum(mask) < 3:
            continue
        a_vals.append(np.polyfit(x_vals[mask], row[mask], deg=2)[0])
    return np.asarray(a_vals, dtype=float)


def one_sided_ttest_less(values):
    """One-sided t-test for mean(values) < 0, i.e. an inverted-U quadratic term."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    t_stat, p_two = stats.ttest_1samp(values, 0.0)
    if np.isnan(t_stat) or np.isnan(p_two):
        return t_stat, np.nan
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    return t_stat, float(p_one)


def paired_ttest_yes_no(yes_vals, no_vals):
    """
    Paired t-test of the Yes vs No subject means, keeping the t statistic.

    Returns the same keys even when there is too little data, so the caller can always
    put the cell into the stats table.
    """
    yes_vals = np.asarray(yes_vals, dtype=float)
    no_vals = np.asarray(no_vals, dtype=float)
    # The two lists are appended in lockstep, so pairing is by position; drop a pair
    # whenever either side is non-finite to keep them aligned.
    paired = np.isfinite(yes_vals) & np.isfinite(no_vals)
    yes_vals, no_vals = yes_vals[paired], no_vals[paired]
    if yes_vals.size < 2:
        return {"t": np.nan, "p": np.nan, "n": int(yes_vals.size),
                "mean_yes": np.nan, "mean_no": np.nan, "mean_diff": np.nan}
    t_stat, p_val = stats.ttest_rel(yes_vals, no_vals)
    return {
        "t": float(t_stat),
        "p": float(p_val),
        "n": int(yes_vals.size),
        "mean_yes": float(np.mean(yes_vals)),
        "mean_no": float(np.mean(no_vals)),
        "mean_diff": float(np.mean(yes_vals - no_vals)),
    }


# %%
# ---------------------------------------------------------------- data collection


def collect_band_data(band, subjects, modalities):
    """
    Walk the subjects once per modality and gather everything the three panels need.

    Returns per-subject quantities only; all averaging and testing happens downstream.
    """
    bin_rates = {mod: {ch: [] for ch in CHANNELS} for mod in modalities}
    env_yes = {mod: {ch: [] for ch in CHANNELS} for mod in modalities}
    env_no = {mod: {ch: [] for ch in CHANNELS} for mod in modalities}
    # Subject of every env_yes / env_no entry (appended in lockstep); the mixed model needs it.
    env_subjects = {mod: {ch: [] for ch in CHANNELS} for mod in modalities}
    topo_yes = {mod: [] for mod in modalities}
    topo_no = {mod: [] for mod in modalities}
    # Captured from the first session processed and asserted identical for every later one.
    topo_info, topo_channel_names, topo_times = None, None, None

    fmin = BAND_FMIN[band]

    for modality in modalities:
        tmin, tmax = BIN_WINDOW[modality]

        for subID in subjects:
            if (modality, subID) in FAULTY_SESSIONS:
                print(f"[skip] sub-0{subID} ses-{modality}: faulty recording setup")
                continue
            
            if is_subject_excluded(subID, modality):
                print(f"[skip] sub-0{subID} ses-{modality}: on the SNR exclusion list")
                continue

            sub_dir = os.path.join(
                DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", f"ses-{modality}"
            )
            base_name = (
                f"sub-0{subID}_ses-{modality}_task-NT_"
                f"lfreqori-{0.1}_lfreq-{1}_hfreq-{40}"
            )
            epochs, metadata = ah.load_band_epochs_and_metadata(
                sub_dir, base_name, f"{band}", subID, modality,
                FILE_SUFFIX_EPOCHS, FILE_SUFFIX_META,
            )

            if epochs is None:
                continue
            if len(epochs) != len(metadata):
                raise ValueError(
                    f"Epochs ({len(epochs)}) and metadata rows ({len(metadata)}) differ "
                    f"for sub-0{subID} ses-{modality} band {band}."
                )

            response = metadata["response_decoded"].to_numpy()
            yes_mask, no_mask = response == "Yes", response == "No"
            if not (np.any(yes_mask) and np.any(no_mask)):
                print(f"[skip] sub-0{subID} ses-{modality}: needs both Yes and No trials")
                continue
            detected = (response == "Yes").astype(float)
    
            epochs_lap = ah.apply_surface_laplacian(epochs)
            envelope_full = hilbert_envelope(
                epochs_lap.get_data(), epochs_lap.info["sfreq"], fmin
            )
            times_full = epochs_lap.times
            lap_ch_names = list(epochs_lap.ch_names)

            # --- Panels A and B: the three picked channels over the binning window ---
            tmask = (times_full >= tmin - TIME_EPS) & (times_full <= tmax + TIME_EPS)
            if not np.any(tmask):
                raise ValueError(
                    f"Window ({tmin}, {tmax}) does not overlap the epoch time axis for "
                    f"sub-0{subID} ses-{modality}."
                )

            for ch in CHANNELS:
                ch_idx = lap_ch_names.index(ch)
                trial_power = envelope_full[:, ch_idx, :][:, tmask].mean(axis=1)

                # Equiprobable bins within this subject, then detection rate per bin.
                bin_idx = pd.qcut(trial_power, q=N_BINS, labels=False, duplicates="drop")
                subj_df = pd.DataFrame(
                    {"bin": np.asarray(bin_idx, dtype=float), "detected": detected}
                ).dropna()
                if subj_df.empty:
                    continue
                subj_rates = np.full(N_BINS, np.nan, dtype=float)
                for b in range(N_BINS):
                    vals = subj_df.loc[subj_df["bin"] == b, "detected"].to_numpy()
                    if vals.size > 0:
                        subj_rates[b] = np.mean(vals)
                bin_rates[modality][ch].append(subj_rates)

                env_yes[modality][ch].append(float(trial_power[yes_mask].mean()))
                env_no[modality][ch].append(float(trial_power[no_mask].mean()))
                env_subjects[modality][ch].append(int(subID))

            # --- Panel C: all channels over the wider topography window ---
            topo_mask = ((times_full >= TOPO_WINDOW["tmin"] - TIME_EPS)
                         & (times_full <= TOPO_WINDOW["tmax"] + TIME_EPS))
            if not np.any(topo_mask):
                raise ValueError(
                    f"Topography window ({TOPO_WINDOW['tmin']}, {TOPO_WINDOW['tmax']}) does "
                    f"not overlap the epoch time axis for sub-0{subID} ses-{modality}."
                )
            sub_topo_times = times_full[topo_mask]
            if topo_channel_names is None:
                topo_channel_names = lap_ch_names
                topo_info = epochs_lap.info.copy()
                topo_times = sub_topo_times.copy()
            elif lap_ch_names != topo_channel_names:
                raise ValueError(
                    "EEG channel mismatch in the topography summary: expected "
                    f"{len(topo_channel_names)} channels, got {len(lap_ch_names)} "
                    f"for sub-0{subID} ses-{modality} band {band}."
                )
            elif (sub_topo_times.shape != topo_times.shape
                  or not np.allclose(sub_topo_times, topo_times)):
                raise ValueError(
                    "Time axis mismatch in the topography summary for "
                    f"sub-0{subID} ses-{modality} band {band}."
                )

            topo_envelope = envelope_full[:, :, topo_mask]
            topo_yes[modality].append(topo_envelope[yes_mask].mean(axis=0))
            topo_no[modality].append(topo_envelope[no_mask].mean(axis=0))

        print(f"[{band}] {modality}: {len(topo_yes[modality])} subjects included")

    return dict(
        bin_rates=bin_rates, env_yes=env_yes, env_no=env_no, env_subjects=env_subjects,
        topo_yes=topo_yes, topo_no=topo_no,
        topo_info=topo_info, topo_channel_names=topo_channel_names, topo_times=topo_times,
    )


# %%
# ---------------------------------------------------------------- statistics


def compute_statistics(data, modalities):
    """
    Group-level statistics for all three panels.

    Benjamini-Hochberg FDR correction is applied across the tested channels within each
    modality (n = len(CHANNELS)). It happens here rather than while plotting, because the
    stars need every channel's p-value before anything is drawn.
    """
    x_bins = np.arange(1, N_BINS + 1, dtype=float)
    detection = {ch: {} for ch in CHANNELS}   # group mean/sem of the binned curves
    quad_stats = {ch: {} for ch in CHANNELS}
    bar_stats = {ch: {} for ch in CHANNELS}

    for modality in modalities:
        for ch in CHANNELS:
            subject_rates = data["bin_rates"][modality][ch]
            if len(subject_rates) == 0:
                detection[ch][modality] = None
                quad_stats[ch][modality] = None
            else:
                rates = np.stack(subject_rates, axis=0)
                sem = np.array([
                    stats.sem(rates[~np.isnan(rates[:, b]), b])
                    if np.sum(~np.isnan(rates[:, b])) > 1 else 0.0
                    for b in range(N_BINS)
                ])
                detection[ch][modality] = {"mean": np.nanmean(rates, axis=0), "sem": sem,
                                           "n": rates.shape[0]}

                a_vals = fit_quadratic_coefficients(x_bins, rates)
                t_stat, p_val = one_sided_ttest_less(a_vals)
                quad_stats[ch][modality] = {
                    "a_mean": float(np.nanmean(a_vals)) if a_vals.size else np.nan,
                    "t": t_stat, "p": p_val, "n": int(a_vals.size),
                }

            bar_stats[ch][modality] = paired_ttest_yes_no(
                data["env_yes"][modality][ch], data["env_no"][modality][ch]
            )

        # One correction family per modality: the channels tested in it.
        bar_corrected = ah.fdr_correct_pvals(
            [bar_stats[ch][modality]["p"] for ch in CHANNELS]
        )
        quad_corrected = ah.fdr_correct_pvals(
            [quad_stats[ch][modality]["p"] if quad_stats[ch][modality] else np.nan
             for ch in CHANNELS]
        )
        for ch_idx, ch in enumerate(CHANNELS):
            bar_stats[ch][modality]["p_fdr"] = float(bar_corrected[ch_idx])
            if quad_stats[ch][modality] is not None:
                quad_stats[ch][modality]["p_fdr"] = float(quad_corrected[ch_idx])

    topo_summaries = {
        modality: ah.build_yes_minus_no_topography_summary(
            data["topo_yes"][modality], data["topo_no"][modality],
            data["topo_times"], TOPO_WINDOW["mean_win"], data["topo_channel_names"],
        )
        for modality in modalities
    }

    return detection, quad_stats, bar_stats, topo_summaries


def run_modality_lmm(data):
    """
    Linear mixed model (response x modality) on the subject-level pre-stimulus envelope, per channel.

    envelope ~ response_c * modality_c + (1 | subject). Both predictors are centred (+-0.5:
    Yes and mecha = +0.5), so the main effects are averaged over the other factor. The random
    subject intercept handles the pairing of the two sessions of a subject, so subjects with
    a single session still enter. Each effect's p is FDR-corrected across the channels (one
    family per effect). Returns {channel: dict(n_subjects, n_obs, effects, ...)}.
    """
    missing = [m for m in MODALITIES if m not in data["env_yes"]]
    if missing:
        raise ValueError(f"LMM needs both modalities, missing: {missing}")

    lmm = {}
    for ch in CHANNELS:
        rows = []
        for m in MODALITIES:
            ids = data["env_subjects"][m][ch]
            for resp, values in (("No", data["env_no"][m][ch]), ("Yes", data["env_yes"][m][ch])):
                rows += [dict(subject=s, envelope=v,
                              response_c=0.5 if resp == "Yes" else -0.5,
                              modality_c=0.5 if m == "mecha" else -0.5)
                         for s, v in zip(ids, values)]
        df = pd.DataFrame(rows)
        df = df[np.isfinite(df["envelope"])]
        n_subjects = df["subject"].nunique()
        n_per_modality = {m: df.loc[df["modality_c"] == (0.5 if m == "mecha" else -0.5),
                                    "subject"].nunique() for m in MODALITIES}
        print(f"  LMM {ch}: {n_subjects} subjects ({n_per_modality['elec']} elec, "
              f"{n_per_modality['mecha']} mecha), {len(df)} observations")

        res = smf.mixedlm("envelope ~ response_c * modality_c", df,
                          groups=df["subject"]).fit(reml=True)
        # A silently non-converged fit would put wrong numbers into the figure.
        if not res.converged:
            raise RuntimeError(f"LMM {ch}: mixedlm did not converge.")

        effects = {}
        for eff, term in LMM_EFFECTS.items():
            effects[eff] = dict(beta=float(res.params[term]), se=float(res.bse[term]),
                                z=float(res.tvalues[term]), p=float(res.pvalues[term]),
                                p_fdr=np.nan)
        lmm[ch] = dict(n_subjects=n_subjects, n_elec=n_per_modality["elec"],
                       n_mecha=n_per_modality["mecha"], n_obs=len(df), effects=effects,
                       random_intercept_var=float(res.cov_re.iloc[0, 0]),
                       resid_var=float(res.scale))

    # FDR over channels, one family per effect.
    for eff in LMM_EFFECTS:
        corrected = ah.fdr_correct_pvals([lmm[ch]["effects"][eff]["p"] for ch in CHANNELS])
        for ch, p_corr in zip(CHANNELS, corrected):
            lmm[ch]["effects"][eff]["p_fdr"] = float(p_corr)
    return lmm


def build_lmm_table(band, lmm):
    """One row per channel x effect of the panel B mixed model."""
    rows = []
    for ch in CHANNELS:
        res = lmm[ch]
        for eff, e in res["effects"].items():
            rows.append({
                "band": band, "window": WINDOW_LABEL, "channel": ch,
                "formula": "envelope ~ response_c * modality_c + (1 | subject)",
                "effect": eff, "beta": e["beta"], "se": e["se"], "z": e["z"],
                "p_raw": e["p"], "p_fdr": e["p_fdr"],
                "fdr_family": f"{len(CHANNELS)} channels, per effect",
                "n_subjects": res["n_subjects"], "n_elec": res["n_elec"],
                "n_mecha": res["n_mecha"], "n_obs": res["n_obs"],
                "random_intercept_var": res["random_intercept_var"],
                "resid_var": res["resid_var"],
            })
    return pd.DataFrame(rows)


def format_p(p_val):
    """p-value text for the figure, e.g. 'p_fdr = 0.0078' or 'p_fdr < .001'."""
    if not np.isfinite(p_val):
        return "p_fdr n/a"
    if p_val < 0.001:
        return "p_fdr < .001"
    return f"p_fdr = {p_val:.3g}"


def build_stats_table(band, quad_stats, bar_stats, topo_summaries, modalities, lmm=None):
    """Tidy one-row-per-test table; the topography rows carry the peak channel/latency."""
    rows = []
    for ch in CHANNELS:
        for modality in modalities:
            quad = quad_stats[ch][modality]
            if quad is not None:
                rows.append({
                    "band": band, "window": WINDOW_LABEL, "channel": ch,
                    "modality": modality, "analysis": "quadratic_detection_rate",
                    "n_subjects": quad["n"], "statistic": quad["t"],
                    "p_raw": quad["p"], "p_fdr": quad["p_fdr"],
                    "a_mean": quad["a_mean"],
                    "mean_yes": np.nan, "mean_no": np.nan, "mean_diff": np.nan,
                    "peak_channel": "", "peak_time_ms": np.nan,
                })
            bar = bar_stats[ch][modality]
            rows.append({
                "band": band, "window": WINDOW_LABEL, "channel": ch,
                "modality": modality, "analysis": "envelope_yes_no_ttest",
                "n_subjects": bar["n"], "statistic": bar["t"],
                "p_raw": bar["p"], "p_fdr": bar["p_fdr"], "a_mean": np.nan,
                "mean_yes": bar["mean_yes"], "mean_no": bar["mean_no"],
                "mean_diff": bar["mean_diff"],
                "peak_channel": "", "peak_time_ms": np.nan,
            })

    # Panel B's mixed model: statistic holds the Wald z, mean_diff the coefficient (beta) on
    # the +-0.5 centred predictors.
    if lmm is not None:
        for ch in CHANNELS:
            res = lmm[ch]
            for eff, e in res["effects"].items():
                rows.append({
                    "band": band, "window": WINDOW_LABEL, "channel": ch,
                    "modality": "both", "analysis": f"lmm_{eff.replace(':', '_x_')}",
                    "n_subjects": res["n_subjects"], "statistic": e["z"],
                    "p_raw": e["p"], "p_fdr": e["p_fdr"], "a_mean": np.nan,
                    "mean_yes": np.nan, "mean_no": np.nan, "mean_diff": e["beta"],
                    "peak_channel": "", "peak_time_ms": np.nan,
                })

    # Panel C has no inferential test; the row records where the peak was taken from, so
    # the latency quoted in the manuscript is reproducible from a file.
    for modality in modalities:
        summary = topo_summaries[modality]
        if not summary.get("has_data", False):
            continue
        rows.append({
            "band": band, "window": "topo", "channel": "all",
            "modality": modality, "analysis": "yes_minus_no_topography_peak",
            "n_subjects": np.nan, "statistic": np.nan,
            "p_raw": np.nan, "p_fdr": np.nan, "a_mean": np.nan,
            "mean_yes": np.nan, "mean_no": np.nan, "mean_diff": summary["peak_value"],
            "peak_channel": summary["peak_channel"],
            "peak_time_ms": summary["peak_time"] * 1000,
        })

    return pd.DataFrame(rows)


def significance_label(p_val):
    """Star for a p-value; the caller decides whether that p is corrected."""
    if not np.isfinite(p_val):
        return None
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


# %%
# ---------------------------------------------------------------- figure


def plot_figure(band, detection, quad_stats, bar_stats, topo_summaries, data, modalities,
                lmm=None):
    """Assemble the three-panel figure and return it."""
    n_channels = len(CHANNELS)
    x = np.arange(1, N_BINS + 1)
    bar_offsets = {"elec": -0.2, "mecha": 0.2}
    bar_width = 0.35
    x_response = {"No": 0, "Yes": 1}

    # Channels are columns and the three panels are rows: that keeps the figure wider than
    # tall, so it can be printed at full text width and still leave room for body text on
    # the manuscript page.
    fig = plt.figure(figsize=(9, 7))
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[1, 1, 1.1], hspace=0.55)
    gs_a = outer[0].subgridspec(1, n_channels, wspace=0.32)
    gs_b = outer[1].subgridspec(1, n_channels, wspace=0.32)
    # Panel C columns: [margin, topo, topo, gap, colourbar, margin]. Topographies are square, so
    # topo columns are only as wide as the heads; margins centre the block, the gap sets the
    # distance between the heads and the colourbar.
    gs_c = outer[2].subgridspec(1, len(modalities) + 4, wspace=0.0,
                                width_ratios=[1] + [1.2] * len(modalities) + [0.3, 0.1, 1.6])

    bin_axes = [fig.add_subplot(gs_a[0, c]) for c in range(n_channels)]
    bar_axes = [fig.add_subplot(gs_b[0, c]) for c in range(n_channels)]
    topo_axes = [fig.add_subplot(gs_c[0, 1 + i]) for i in range(len(modalities))]
    cbar_holder = fig.add_subplot(gs_c[0, len(modalities) + 2])
    cbar_holder.axis("off")

    for col_idx, ch in enumerate(CHANNELS):
        # --- Panel A: detection rate over envelope bins ---
        ax = bin_axes[col_idx]
        quad_lines = []
        for modality in modalities:
            current = detection[ch].get(modality)
            if current is None:
                continue
            ax.plot(x, current["mean"], color=PALETTE[modality], marker="o",
                    label=f"{MODALITY_LABELS[modality]} (n={current['n']})" if col_idx == 0 else None)
            ax.fill_between(x, current["mean"] - current["sem"],
                            current["mean"] + current["sem"],
                            color=PALETTE[modality], alpha=0.2)
            quad = quad_stats[ch].get(modality)
            if quad is not None and np.isfinite(quad["p"]):
                # p is the raw one-sided value, p_fdr is FDR-corrected across channels.
                quad_lines.append(
                    f"{modality}: a={quad['a_mean']:.3g}, "
                    f"p={quad['p']:.3g}, p_fdr={quad['p_fdr']:.3g}"
                )
        if quad_lines:
            ax.text(0.02, 0.02, "\n".join(quad_lines), transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=6)
        # Bold channel name on top of the column; each column is one channel.
        ax.set_title(ch, fontsize=12, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Detection rate", fontsize=10)
        ax.set_xticks(x)
        ax.tick_params(labelsize=8)
        # Panel A is a single row now, so every column carries its own x-axis label.
        ax.set_xlabel(f"{band.title()} envelope bin", fontsize=10)
        sns.despine(ax=ax)

        # --- Panel B: envelope by detection response ---
        ax = bar_axes[col_idx]
        bar_top = {}
        envelope_by_response = {"Yes": data["env_yes"], "No": data["env_no"]}
        for modality in modalities:
            for response, x_base in x_response.items():
                vals = np.array(envelope_by_response[response][modality][ch], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                x_pos = x_base + bar_offsets[modality]
                mean, sem = np.nanmean(vals), stats.sem(vals)
                ax.bar(x_pos, mean, width=bar_width, color=PALETTE[modality], alpha=0.85)
                ax.errorbar(x_pos, mean, yerr=sem, fmt="none",
                            color="black", capsize=3, linewidth=1)
                bar_top[(modality, response)] = (x_pos, mean + sem)

        bar_max = max((top for _, top in bar_top.values()), default=0.0)
        if lmm is not None:
            # Mixed model p-values (FDR-corrected across channels) replace the per-modality
            # significance brackets; the interaction is on the last line.
            effects = lmm[ch]["effects"]
            ax.text(0.5, 0.98,
                    f"Response: {format_p(effects['response']['p_fdr'])}\n"
                    f"Modality: {format_p(effects['modality']['p_fdr'])}\n"
                    f"Response x Modality: {format_p(effects['response:modality']['p_fdr'])}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6)
        if bar_max > 0:
            # Headroom so the text above the bars does not overlap them.
            ax.set_ylim(0, bar_max * 1.45)
        # No channel title here: the columns line up with panel A, which already names them,
        # and a second title row would collide with panel A's x-axis label.
        if col_idx == 0:
            ax.set_ylabel(f"{band.title()} envelope (µV/cm²)", fontsize=10)
        ax.set_xticks(list(x_response.values()))
        ax.set_xticklabels(list(x_response.keys()))
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    # --- Panel C: Yes-No envelope topographies ---
    drawn = [s for s in topo_summaries.values() if s.get("has_data", False)]
    image = None
    if drawn:
        # One symmetric colour scale across modalities so the panels are comparable.
        vmax = max(np.max(np.abs(s["topo_values"])) for s in drawn)
        vmax = 1e-12 if np.isclose(vmax, 0.0) else vmax
        channel_index = {n: i for i, n in enumerate(data["topo_channel_names"])}
        for col_idx, modality in enumerate(modalities):
            ax = topo_axes[col_idx]
            summary = topo_summaries[modality]
            if not summary.get("has_data", False):
                ax.set_title(f"{MODALITY_LABELS[modality]}\nno data")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            peak_mask = np.zeros(len(data["topo_channel_names"]), dtype=bool)
            peak_mask[channel_index[summary["peak_channel"]]] = True
            # Both panels share vlim, so either image can drive the colourbar.
            image, _ = mne.viz.plot_topomap(
                summary["topo_values"], data["topo_info"], axes=ax, show=False,
                cmap="RdBu_r", vlim=(-vmax, vmax), contours=6, sensors=True,
                mask=peak_mask,
                mask_params=dict(marker="o", markerfacecolor="none",
                                 markeredgecolor="crimson", linewidth=2, markersize=10),
                outlines="head",
            )
            ax.set_title(
                f"{MODALITY_LABELS[modality]}\n"
                f"peak {summary['peak_channel']} at {summary['peak_time'] * 1000:.0f} ms",
                fontsize=10,
            )
    else:
        for ax in topo_axes:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    if image is not None:
        # One vertical colourbar shared by both head plots, placed to their right.
        cax = cbar_holder.inset_axes([0, 0.1, 1, 0.65])
        cbar = fig.colorbar(image, cax=cax, orientation="vertical")
        cbar.set_label(f"Yes - No \n{band} envelope", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    handles, labels = bin_axes[0].get_legend_handles_labels()
    if handles:
        # Anchored to the header band so it no longer sits on top of the first panel.
        fig.legend(handles, labels, frameon=False, loc="upper center",
                   ncol=len(modalities), fontsize=10, bbox_to_anchor=(0.5, 0.995))

    # Panel letters, anchored to each block after the colourbar has taken its space, so
    # they stay put if the channel list changes.
    fig.canvas.draw()
    # All letters share the x position of panel A's first axis.
    letter_x = max(bin_axes[0].get_position().x0 - 0.06, 0.005)
    for letter, ax_ref in zip("ABC", [bin_axes[0], bar_axes[0], topo_axes[0]]):
        pos = ax_ref.get_position()
        fig.text(letter_x, min(pos.y1 + 0.055, 0.995), letter,
                 fontsize=16, fontweight="bold", va="top", ha="left")

    return fig


# %%
# ---------------------------------------------------------------- main


def main():
    """
    Run every requested band and return the intermediates per band, so that after
    running the file as cells you can poke at e.g. results["alpha"]["quad_stats"].
    """
    args = parse_args()
    os.makedirs(FIGUREDIR, exist_ok=True)
    os.makedirs(STATSDIR, exist_ok=True)
    results = {}

    for band in args.bands:
        print(f"\n===== {band} =====")
        data = collect_band_data(band, args.sub, args.modalities)
        if data["topo_channel_names"] is None:
            raise RuntimeError(
                f"No subject contributed any data for band {band}; nothing to plot."
            )

        detection, quad_stats, bar_stats, topo_summaries = compute_statistics(
            data, args.modalities
        )

        # The mixed model needs both modalities; with a single one it is not defined.
        lmm = run_modality_lmm(data) if set(args.modalities) == set(MODALITIES) else None

        table = build_stats_table(band, quad_stats, bar_stats, topo_summaries,
                                  args.modalities, lmm)
        stats_csv = os.path.join(
            STATSDIR, f"manuscript_envelope_detection_stats_{band}.csv"
        )
        table.to_csv(stats_csv, index=False)
        print(f"\n=== {band} envelope statistics "
              f"(p_fdr: Benjamini-Hochberg FDR over {len(CHANNELS)} channels "
              f"within modality) ===")
        print(table.to_string(index=False))
        print(f"written to {stats_csv}")

        if lmm is not None:
            lmm_table = build_lmm_table(band, lmm)
            lmm_csv = os.path.join(
                STATSDIR, f"manuscript_envelope_detection_modality_lmm_{band}.csv"
            )
            lmm_table.to_csv(lmm_csv, index=False)
            print(f"\n=== {band} panel B mixed model (p_fdr: Benjamini-Hochberg over "
                  f"{len(CHANNELS)} channels, per effect) ===")
            print(lmm_table.to_string(index=False))
            print(f"written to {lmm_csv}")

        fig = plot_figure(band, detection, quad_stats, bar_stats, topo_summaries,
                          data, args.modalities, lmm)
        out_path = os.path.join(FIGUREDIR, "Fig03_alpha_amplitude")
        fig.savefig(f"{out_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
        # plt.show() without arguments: plt.show(fig) only works on the inline backend.
        plt.show()
        print(f"Saved figure: {out_path}.pdf / .png")

        results[band] = dict(data=data, detection=detection, quad_stats=quad_stats,
                             bar_stats=bar_stats, topo_summaries=topo_summaries,
                             lmm=lmm, table=table, fig=fig)

    return results


if __name__ == "__main__":
    results = main()
