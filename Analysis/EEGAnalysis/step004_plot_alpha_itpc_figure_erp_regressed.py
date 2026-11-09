"""
File: step004_plot_alpha_itpc_figure_erp_regressed.py
Author: Rosa Grossmann
Date: 2026-11-02
Description:
Pre-stimulus alpha dITPC manuscript panel (2x2) computed on the dITPC with the broadband ERP
regressed out. dITPC is the debiased squared ITPC (PPC), (n * R^2 - 1) / (n - 1), with R the
resultant vector length over the n trials of one response.

ERP removal, per subject, modality and channel: the ERP comes from the same epochs without the
band-pass (surface Laplacian, channel pick, pre-cue baseline, mean over the trials of one
response) and enters as |ERP(t)|. One OLS fit dITPC = a + b * |ERP| over the Yes and No time
points stacked (first samples left out of the fit as Hilbert-edge contaminated) gives a slope b
shared by both responses. The cleaned trace is
    dITPC_clean(t) = dITPC(t) - b * (|ERP|(t) - mean(|ERP|)),
which removes the ERP term but keeps the dITPC level.

The panel is drawn on the cleaned dITPC:
  A) top left     dITPC block for CP4: rows = elec / mecha, columns = timecourse, post-cue mean,
                  pre- vs post-cue delta.
  B) top right    the same block for CPz.
  C) bottom left  subject level circular-mean phase at -0.10 s for CP4, one polar plot per
                  modality, Perceived and Unperceived pooled, Rayleigh test uncorrected. Phase
                  does not enter the regression, so this panel is the same as without ERP removal.
                  Grey dots mark subjects whose own trial-level Rayleigh test is not significant;
                  the count of significant subjects is written under the modality name.
  D) bottom right post-cue dITPC response x modality interaction, one plot per channel, from the
                  linear mixed model itpc ~ response * modality + (1 | subject) with centred
                  (+-0.5) predictors; p FDR-corrected across the two channels per effect.

Sessions on Pre-Processing/exclued_subs.json and the faulty sessions are skipped. The figure
is written to the results folder and, as Fig03_alpha_itpc (.svg / .png), to Writing/img.
"""

# %%
import argparse
import os
import sys
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.signal import hilbert as scipy_hilbert
import statsmodels.formula.api as smf

from plotting_style import figure_style
import eeg_analysis_helpers as ah

colors, _ = figure_style()

DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
RECONDIR = os.getenv("SUBJECTS_DIR")
WDDIR = os.getenv("WDDIR")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

# Imported after WDDIR is on the path. The band .fif files exist for every subject, so without
# this gate the SNR-excluded sessions would quietly rejoin the analysis.
from general_helpers import apply_baseline_epochs, is_subject_excluded

# %%
# ---------------------------------------------------------------- configuration

FILE_SUFFIX_EPOCHS = "_preprestim_corrected"
FILE_SUFFIX_META = "_preprestim"

ALL_SUBJECTS = list(range(13, 53))
MODALITIES = ["elec", "mecha"]
# Only the two channels shown in the manuscript figure. They are also the FDR family.
CHANNELS = ["CP4", "CPz"]
BAND = "alpha"
BAND_FMIN = {"alpha": 8, "beta": 13}

L_FREQ_ORIG = 0.1
L_FREQ = 1
H_FREQ = 40

# Analysis/display window for the ITPC timecourse.
WIN_TMIN, WIN_TMAX = -0.75, -0.1

# Time point at which the subject-level phase is read out for the polar panel.
PHASE_READOUT_TIME = -0.1

# Per-subject check of panel C: Rayleigh test over the TRIALS of one subject at the readout
# time. A subject whose p is below this threshold has a phase that is consistent across trials,
# so its circular mean on the polar plot is meaningful. Uncorrected - a descriptive reliability
# check, so about 5% of subjects would pass by chance under a uniform phase.
PHASE_SUBJECT_ALPHA = 0.05

# IMPORTANT, why the Hilbert runs on the FULL epoch and the crop happens afterwards:
# np.pad(mode="reflect") mirrors about the last real sample without repeating it, so the
# padded signal is exactly even-symmetric about that sample. The Hilbert transform of a
# function that is even about a point is odd about the same point, hence zero there. The
# imaginary part of the analytic signal at the final sample therefore vanishes and the
# phase collapses to atan2(0, real) = 0 (real > 0) or +-pi (real < 0). Cropping first and
# padding second pins every trial to 0 deg or 180 deg at the crop edge. Running the Hilbert
# on the whole epoch moves that degenerate sample to the epoch edge, safely outside every
# window we read from. The ITPC arrays are cropped to WIN_TMAX; the phase array keeps
# everything the epoch has after WIN_TMIN, so the readout stays far from the Hilbert edges.

# Post-cue averaging window for the bar plot (identical for both modalities; only the
# cue marker below is modality specific).
POSTCUE_WIN = {"mecha": (-0.436, -0.136), "elec": (-0.4, -0.1)}
# Pre-cue averaging window: -0.75 s to -0.45 s, identical for both modalities and ending
# before either cue. Used to test whether the Yes-No ITPC difference only emerges after the cue.
PRECUE_WIN = {"mecha": (-0.736, -0.436), "elec": (-0.7, -0.4)}
CUE_TIME = {"mecha": -0.436, "elec": -0.4}

# First samples of the Hilbert timecourse are edge-contaminated.
N_EDGE_SAMPLES = 5

# Sessions with a faulty recording setup (sub 23: swapped ground and reference electrode,
# sub 36: exceptionally high force values).
# the two electrical session have a wrong trigger time (10ms longer cue-stim window)
FAULTY_SESSIONS = {("mecha", 23), ("mecha", 36), ("elec", 19), ("elec", 28)}

PALETTE = {"No": colors["unperceived"], "Yes": colors["perceived"]}
# Panel C's top row pools the two responses, so it must NOT wear a response colour - a pooled
# distribution is a property of the modality, not of a response.
PALETTE_MOD = {"elec": colors["electrical"], "mecha": colors["mechanical"]}
# Modality names as printed in the figure.
MODALITY_NAME = {"elec": "Electrical", "mecha": "Mechanical"}

RESPONSES = ("No", "Yes")
# One pre-cue baseline for the broadband ERP, both modalities.
BASELINE_BY_MOD = {"elec": (-0.55, -0.45), "mecha": (-0.55, -0.45)}

FIGUREDIR = os.path.join(
    RESDIR, "Figures", "Main", "eeg_results", "frequency_analysis",
    f"window_{FILE_SUFFIX_EPOCHS}",
)
STATSDIR = os.path.join(RESDIR, "Statistics", "eeg", "itpc")
MANUSCRIPT_IMGDIR = os.path.join(WDDIR, "Writing", "img")
MANUSCRIPT_STEM = "Fig03_alpha_itpc"


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
    parser.add_argument("--sub", nargs="+", type=int, default=ALL_SUBJECTS,
                        help="Subject IDs (default: 13-52).")
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES,
                        help="Modalities to process (default: both).")
    parser.add_argument("--band", default=BAND, choices=sorted(BAND_FMIN),
                        help="Frequency band, loaded as '{band}' (default: alpha).")
    if argv is None:
        argv = [] if running_in_kernel() else sys.argv[1:]
    # Parsed strictly, so a typo on the command line is still an error.
    return parser.parse_args(argv)

# %%
# ---------------------------------------------------------------- data collection


def phase_readout_index(times_phase):
    """
    Sample of the phase time axis at PHASE_READOUT_TIME. Guards against reading the degenerate
    first/last sample of the Hilbert output: there the phase can only be 0 or 180 deg (see the
    note next to WIN_TMAX).
    """
    idx = int(np.argmin(np.abs(times_phase - PHASE_READOUT_TIME)))
    if idx < N_EDGE_SAMPLES or idx > len(times_phase) - 1 - N_EDGE_SAMPLES:
        raise ValueError(
            f"Phase readout at {PHASE_READOUT_TIME} s resolves to sample {idx} of "
            f"{len(times_phase)}, within {N_EDGE_SAMPLES} samples of the Hilbert edge. "
            f"Extend the crop so the readout is not edge-contaminated."
        )
    return idx


def compute_subject_itpc_and_phase(epochs_ff, band, subID, modality):
    """
    Per-subject ITPC and mean-phase timecourses from filter-first epochs.

    Returns (times_itpc, times_phase, per-channel dict) or None if the subject has
    no usable Yes/No split. Steps: surface Laplacian, pick channels, reflect-pad,
    Hilbert on the FULL epoch, trim padding, crop to the analysis windows.
    The crop deliberately happens after the Hilbert (see the note next to WIN_TMAX).
    """
    epochs_lap = ah.apply_surface_laplacian(epochs_ff)
    epochs_picks = epochs_lap.copy().pick(CHANNELS)
    times_full = epochs_picks.times

    # The epoch must extend past the readout time, otherwise the degenerate edge sample
    # would land on the readout and the figure would be silently wrong.
    if times_full[-1] < PHASE_READOUT_TIME:
        raise ValueError(
            f"sub-0{subID} {modality} {band}: epoch ends at {times_full[-1]:.3f} s but "
            f"phase is read out at {PHASE_READOUT_TIME:.3f} s. The Hilbert edge artifact "
            f"would contaminate the readout."
        )

    perceived_mask = (epochs_picks.metadata["response_decoded"] == "Yes").to_numpy()
    unperceived_mask = (epochs_picks.metadata["response_decoded"] == "No").to_numpy()
    if perceived_mask.sum() == 0 or unperceived_mask.sum() == 0:
        print(f"[skip] sub-0{subID} ses-{modality}: needs both Yes and No trials")
        return None

    data = epochs_picks.get_data()  # (n_trials, n_channels, n_times)
    # Pad with reflected signal to reduce Hilbert edge artifacts.
    # Rule of thumb: at least 3 cycles of the lowest frequency in the band.
    fmin = BAND_FMIN[band]
    n_pad = int(3 * epochs_picks.info["sfreq"] / fmin)
    data_padded = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode="reflect")
    analytic = scipy_hilbert(data_padded, axis=-1)[:, :, n_pad:-n_pad]
    phase_rad = np.angle(analytic)

    # Crop the phase array only now that the analytic signal exists.
    itpc_mask = (times_full >= WIN_TMIN) & (times_full <= WIN_TMAX)
    # Keep the phase out to the end of the epoch: the readout time must stay several
    # samples away from the Hilbert edge, which now sits at the epoch boundary.
    phase_mask = times_full >= WIN_TMIN
    phase_itpc = phase_rad[:, :, itpc_mask]
    phase_readout = phase_rad[:, :, phase_mask]
    readout_idx = phase_readout_index(times_full[phase_mask])

    # ITPC across trials, independently at each channel and time point.
    #itpc = np.abs(np.mean(np.exp(1j * phase_itpc), axis=0))

    # compute trial numbers for debiased squared ITPC (PPC).
    n_trials_yes = int(perceived_mask.sum())
    n_trials_no = int(unperceived_mask.sum())
    if n_trials_yes < 2 or n_trials_no < 2:
        raise ValueError(
            f"sub-0{subID} {modality} {band}: debiased ITPC needs at least 2 trials per "
            f"condition, got {n_trials_yes} Yes and {n_trials_no} No."
        )

    per_channel = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        # Within-subject phase consistency: Rayleigh test over all (Yes + No) trials of this
        # subject at the readout sample.
        _, _, _, _, trial_p = ah.rayleigh_test(phase_readout[:, ch_idx, readout_idx])
        if not np.isfinite(trial_p):
            raise ValueError(f"sub-0{subID} {modality} {ch_name}: within-subject Rayleigh p is not finite.")

        itpc_yes_value = np.abs(np.mean(np.exp(1j * phase_itpc[perceived_mask, ch_idx, :]), axis=0))
        itpc_no_value = np.abs(np.mean(np.exp(1j * phase_itpc[unperceived_mask, ch_idx, :]), axis=0))
        ppc_yes_value = (n_trials_yes * itpc_yes_value**2 - 1) / (n_trials_yes - 1)
        ppc_no_value = (n_trials_no * itpc_no_value**2 - 1) / (n_trials_no - 1)

        per_channel[ch_name] = dict(
            ppc_yes=ppc_yes_value,
            ppc_no=ppc_no_value,
            itpc_yes=ppc_yes_value, #itpc_yes_value,
            itpc_no=ppc_no_value, #itpc_no_value,
            phase_yes=ah.circular_mean(phase_readout[perceived_mask, ch_idx, :], axis=0),
            phase_no=ah.circular_mean(phase_readout[unperceived_mask, ch_idx, :], axis=0),
            # Pooled over ALL trials, which here means exactly Yes + No: the loader already
            # restricts the epochs to response_decoded in {"Yes", "No"}
            # (eeg_analysis_helpers.load_band_epochs_and_metadata). Deliberately NOT the mean
            # of phase_yes and phase_no - the two conditions have unequal trial counts, so
            # averaging their means would weight a 200-trial condition like a 400-trial one.
            phase_all=ah.circular_mean(phase_readout[:, ch_idx, :], axis=0),
            phase_trial_p=trial_p,
        )
    return times_full[itpc_mask], times_full[phase_mask], per_channel


# %%
# ---------------------------------------------------------------- statistics


def summarise_channel(ch_store, times, times_phase, modality):
    """
    Extract data to plot from subject data matrix.
    - display timecourse (mean +- SEM)
    - post-cue bar means, pre-/post-cue deltas
    - raw paired statistics (FDR correction is applied afterwards).
    """

    itpc_yes_mat = ch_store["itpc_yes"]
    itpc_no_mat = ch_store["itpc_no"]

    # Timecourse: drop the edge-contaminated first samples, keep everything up to WIN_TMAX.
    disp_slice = slice(N_EDGE_SAMPLES, None)
    times_display = times[disp_slice]
    tc_yes = itpc_yes_mat[:, disp_slice]
    tc_no = itpc_no_mat[:, disp_slice]

    n_subjects = tc_yes.shape[0]
    yes_sem = stats.sem(tc_yes, axis=0) if n_subjects > 1 else np.zeros(tc_yes.shape[1])
    no_sem = stats.sem(tc_no, axis=0) if n_subjects > 1 else np.zeros(tc_no.shape[1])

    def window_mean(mat, win):
        """Per-subject mean ITPC over a time window (inclusive of both bounds)."""
        t0 = int(np.argmin(np.abs(times - win[0])))
        t1 = int(np.argmin(np.abs(times - win[1]))) + 1
        return mat[:, t0:t1].mean(axis=1)

    postcue_win = POSTCUE_WIN[modality]
    precue_win = PRECUE_WIN[modality]

    mean_yes_bar = window_mean(itpc_yes_mat, postcue_win)
    mean_no_bar = window_mean(itpc_no_mat, postcue_win)
    mean_yes_pre = window_mean(itpc_yes_mat, precue_win)
    mean_no_pre = window_mean(itpc_no_mat, precue_win)

    if n_subjects > 1:
        t_bar, p_bar = stats.ttest_rel(mean_yes_bar, mean_no_bar)
        diff_bar = mean_yes_bar - mean_no_bar
        d_bar = diff_bar.mean() / diff_bar.std(ddof=1) if diff_bar.std(ddof=1) > 0 else np.nan
    else:
        t_bar, p_bar, d_bar = np.nan, np.nan, np.nan

    # Delta-of-deltas: is the Yes-No ITPC difference larger post-cue than pre-cue?
    diff_pre = mean_yes_pre - mean_no_pre
    diff_post = mean_yes_bar - mean_no_bar
    if n_subjects > 1:
        t_delta, p_delta = stats.ttest_rel(diff_post, diff_pre)
        dd = diff_post - diff_pre
        d_delta = dd.mean() / dd.std(ddof=1) if dd.std(ddof=1) > 0 else np.nan
    else:
        t_delta, p_delta, d_delta = np.nan, np.nan, np.nan

    # Subject-level phase at the readout time, for the polar panel.
    idx = phase_readout_index(times_phase)
    phase_at = dict(
        yes=ch_store["phase_yes"][:, idx],
        no=ch_store["phase_no"][:, idx],
        all=ch_store["phase_all"][:, idx],
    )

    # Rayleigh test for a preferred phase, one per distribution drawn in panel C. This is a
    # SUBJECT-level test: the input is one circular mean per subject, so it asks whether
    # subjects share a common preferred angle, not whether an individual subject is phase
    # locked (that quantity is the ITPC, which panels A and B already show).
    # These tests are reported uncorrected, so p_fdr stays NaN - the file-wide convention that
    # a NaN p_fdr means "no correction was applied to this row".
    phase_rayleigh = {}
    for key in ("all", "yes", "no"):
        n_r, mean_angle_r, R_r, z_r, p_r = ah.rayleigh_test(phase_at[key])
        phase_rayleigh[key] = dict(n=n_r, mean_angle=mean_angle_r, R=R_r, z=z_r,
                                   p=p_r, p_fdr=np.nan)

    return dict(
        times_display=times_display,
        yes_tc_mean=tc_yes.mean(axis=0),
        no_tc_mean=tc_no.mean(axis=0),
        yes_tc_sem=yes_sem,
        no_tc_sem=no_sem,
        mean_itpc_yes_bar=mean_yes_bar,
        mean_itpc_no_bar=mean_no_bar,
        diff_pre=diff_pre,
        diff_post=diff_post,
        precue_win=precue_win,
        postcue_win=postcue_win,
        p_bar=p_bar,
        p_bar_uncorr=p_bar,
        t_bar=t_bar,
        cohens_d_bar=d_bar,
        p_delta=p_delta,
        p_delta_uncorr=p_delta,
        t_delta=t_delta,
        cohens_d_delta=d_delta,
        phase_at=phase_at,
        phase_rayleigh=phase_rayleigh,
        phase_subject_p=ch_store["phase_trial_p"],
        phase_subject_sig=ch_store["phase_trial_p"] < PHASE_SUBJECT_ALPHA,
        subject_ids=ch_store["subject_ids"],
        n_subjects=n_subjects,
    )


# Fixed effects of the mixed model, in the names of the centred predictors.
LMM_EFFECTS = {
    "response": "response_c",
    "modality": "modality_c",
    "response:modality": "response_c:modality_c",
}


def run_modality_lmm(summary):
    """
    Linear mixed model (response x modality) on post-cue mean ITPC, per channel.

    itpc ~ response_c * modality_c + (1 | subject). Both predictors are centred (+-0.5:
    Yes and mecha = +0.5), so the main effects are averaged over the other factor. The random
    subject intercept handles the pairing of the two sessions, which lets every subject of
    panels A and B in, including those with a single session. Each effect's p is
    FDR-corrected across the plotted channels (one family per effect).
    Returns {channel: dict(n_subjects, n_both, n_obs, effects, wide, cell_means, cell_sems)},
    where wide[modality][response] is aligned to subject_ids[modality].
    """
    missing = [m for m in MODALITIES if m not in summary]
    if missing:
        raise ValueError(f"LMM needs both modalities, missing: {missing}")

    lmm = {}
    for ch in CHANNELS:
        ids = {m: list(summary[m][ch]["subject_ids"]) for m in MODALITIES}
        wide = {m: {"No": np.asarray(summary[m][ch]["mean_itpc_no_bar"]),
                    "Yes": np.asarray(summary[m][ch]["mean_itpc_yes_bar"])}
                for m in MODALITIES}
        rows = []
        for m in MODALITIES:
            for resp in ("No", "Yes"):
                rows += [dict(subject=s, itpc=v,
                              response_c=0.5 if resp == "Yes" else -0.5,
                              modality_c=0.5 if m == "mecha" else -0.5)
                         for s, v in zip(ids[m], wide[m][resp])]
        df = pd.DataFrame(rows)
        n_subjects = df["subject"].nunique()
        n_both = len(set(ids["elec"]) & set(ids["mecha"]))
        print(f"  LMM {ch}: {n_subjects} subjects ({len(ids['elec'])} elec, "
              f"{len(ids['mecha'])} mecha, {n_both} with both), {len(df)} observations")

        model = smf.mixedlm("itpc ~ response_c * modality_c", df, groups=df["subject"])
        res = model.fit(reml=True)
        # A silently non-converged fit would put wrong numbers into the figure.
        if not res.converged:
            raise RuntimeError(f"LMM {ch}: mixedlm did not converge.")

        effects = {}
        for eff, term in LMM_EFFECTS.items():
            effects[eff] = dict(beta=float(res.params[term]), se=float(res.bse[term]),
                                z=float(res.tvalues[term]), p=float(res.pvalues[term]),
                                p_fdr=np.nan)

        lmm[ch] = dict(
            n_subjects=n_subjects, n_both=n_both, n_obs=len(df), subject_ids=ids,
            random_intercept_var=float(res.cov_re.iloc[0, 0]), resid_var=float(res.scale),
            effects=effects, wide=wide,
            cell_means={m: {r: float(np.mean(wide[m][r])) for r in ("No", "Yes")}
                        for m in MODALITIES},
            cell_sems={m: {r: float(stats.sem(wide[m][r])) for r in ("No", "Yes")}
                       for m in MODALITIES},
        )

    # FDR over channels, one family per effect.
    for eff in LMM_EFFECTS:
        corrected = ah.fdr_correct_pvals([lmm[ch]["effects"][eff]["p"] for ch in CHANNELS])
        for ch, p_corr in zip(CHANNELS, corrected):
            lmm[ch]["effects"][eff]["p_fdr"] = p_corr
    return lmm


def build_lmm_table(band, lmm):
    """One row per channel x effect of the panel D mixed model, with everything needed to report it."""
    rows = []
    for ch in CHANNELS:
        res = lmm[ch]
        for eff, e in res["effects"].items():
            rows.append({
                "band": band,
                "channel": ch,
                "formula": "itpc ~ response_c * modality_c + (1 | subject)",
                "window": f"post {POSTCUE_WIN['elec'][0]} to {POSTCUE_WIN['elec'][1]} s",
                "effect": eff,
                "beta": e["beta"],
                "se": e["se"],
                "z": e["z"],
                "p_raw": e["p"],
                "p_fdr": e["p_fdr"],
                "significance_fdr": significance_label(e["p_fdr"]),
                "fdr_family": f"{len(CHANNELS)} channels, per effect",
                "n_subjects": res["n_subjects"],
                "n_elec": len(res["subject_ids"]["elec"]),
                "n_mecha": len(res["subject_ids"]["mecha"]),
                "n_both": res["n_both"],
                "n_obs": res["n_obs"],
                "random_intercept_var": res["random_intercept_var"],
                "resid_var": res["resid_var"],
                **{f"mean_{m}_{r}": res["cell_means"][m][r]
                   for m in MODALITIES for r in ("No", "Yes")},
            })
    return pd.DataFrame(rows)


def significance_label(p_val):
    """Star for a p-value; the caller decides whether that p is corrected."""
    if not np.isfinite(p_val):
        return "n/a"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


def format_p(p_val):
    """p-value text for a figure title, e.g. 'p_fdr = 0.0078' or 'p_fdr < .001'."""
    if not np.isfinite(p_val):
        return "p_fdr n/a"
    if p_val < 0.001:
        return "p_fdr < .001"
    return f"p_fdr = {p_val:.3g}"


def build_stats_table(band, summary, modalities, lmm=None):
    """Long-format statistics table: one row per modality x channel x analysis."""
    rows = []
    for modality in modalities:
        for ch in CHANNELS:
            data = summary[modality][ch]
            precue_win = data["precue_win"]
            postcue_win = data["postcue_win"]

            mean_yes = float(np.mean(data["mean_itpc_yes_bar"]))
            mean_no = float(np.mean(data["mean_itpc_no_bar"]))
            rows.append({
                "band": band,
                "modality": modality,
                "channel": ch,
                "analysis": "postcue_itpc_yes_vs_no",
                "n_subjects": data["n_subjects"],
                "window": f"post {postcue_win[0]} to {postcue_win[1]} s",
                "t": data["t_bar"],
                "p_raw": data["p_bar_uncorr"],
                "p_fdr": data["p_bar"],
                "cohens_d": data["cohens_d_bar"],
                "mean_yes": mean_yes,
                "mean_no": mean_no,
                "mean_diff": mean_yes - mean_no,
            })

            # Delta-of-deltas: the mean_yes/mean_no columns hold the post-cue and pre-cue
            # Yes-No differences respectively.
            mean_post = float(np.mean(data["diff_post"]))
            mean_pre = float(np.mean(data["diff_pre"]))
            rows.append({
                "band": band,
                "modality": modality,
                "channel": ch,
                "analysis": "precue_vs_postcue_delta",
                "n_subjects": data["n_subjects"],
                "window": (
                    f"pre {precue_win[0]} to {precue_win[1]} s vs "
                    f"post {postcue_win[0]} to {postcue_win[1]} s"
                ),
                "t": data["t_delta"],
                "p_raw": data["p_delta_uncorr"],
                "p_fdr": data["p_delta"],
                "cohens_d": data["cohens_d_delta"],
                "mean_yes": mean_post,
                "mean_no": mean_pre,
                "mean_diff": mean_post - mean_pre,
            })

            # Panel C's Rayleigh tests, so every number printed on that panel is recoverable
            # from this table. These tests are uncorrected, so their p_fdr is always NaN.
            for key in ("all", "yes", "no"):
                ray = data["phase_rayleigh"][key]
                rows.append({
                    "band": band,
                    "modality": modality,
                    "channel": ch,
                    "analysis": f"phase_rayleigh_{key}",
                    "n_subjects": ray["n"],
                    "window": f"phase at {PHASE_READOUT_TIME} s",
                    "t": ray["z"],           # z is the Rayleigh statistic, not a t
                    "p_raw": ray["p"],
                    "p_fdr": ray["p_fdr"],
                    "cohens_d": np.nan,
                    "mean_yes": np.degrees(ray["mean_angle"]),   # mean phase, degrees
                    "mean_no": ray["R"],                         # resultant length
                    "mean_diff": np.nan,
                })

            # Within-subject phase consistency: mean_yes = subjects with a significant trial-level
            # Rayleigh test, mean_no = their fraction. Uncorrected, so the p columns stay NaN.
            n_sig = int(np.sum(data["phase_subject_sig"]))
            rows.append({
                "band": band,
                "modality": modality,
                "channel": ch,
                "analysis": "phase_rayleigh_within_subject",
                "n_subjects": len(data["phase_subject_sig"]),
                "window": f"phase at {PHASE_READOUT_TIME} s, p < {PHASE_SUBJECT_ALPHA}",
                "t": np.nan,
                "p_raw": np.nan,
                "p_fdr": np.nan,
                "cohens_d": np.nan,
                "mean_yes": n_sig,
                "mean_no": n_sig / len(data["phase_subject_sig"]),
                "mean_diff": np.nan,
            })

    # Panel D's mixed model: t holds the Wald z, mean_diff the coefficient (beta), cohens_d
    # stays NaN. Coefficients are on the +-0.5 centred predictors.
    if lmm is not None:
        for ch in CHANNELS:
            res = lmm[ch]
            for eff, e in res["effects"].items():
                rows.append({
                    "band": band,
                    "modality": "both",
                    "channel": ch,
                    "analysis": f"lmm_{eff.replace(':', '_x_')}",
                    "n_subjects": res["n_subjects"],
                    "window": f"post {POSTCUE_WIN['elec'][0]} to {POSTCUE_WIN['elec'][1]} s",
                    "t": e["z"],                       # Wald z, not a t
                    "p_raw": e["p"],
                    "p_fdr": e["p_fdr"],
                    "cohens_d": np.nan,
                    "mean_yes": np.nan,
                    "mean_no": np.nan,
                    "mean_diff": e["beta"],
                })
    return pd.DataFrame(rows)


# %%
# ---------------------------------------------------------------- panel drawing


def plot_itpc_timecourse(ax, data, ch_name, modality, is_first_column):
    """dITPC timecourse (mean +- SEM across subjects) with the averaging windows shaded."""
    times = data["times_display"]
    for cond, mean_tc, sem_tc in [
        ("No", data["no_tc_mean"], data["no_tc_sem"]),
        ("Yes", data["yes_tc_mean"], data["yes_tc_sem"]),
    ]:
        ax.plot(times, mean_tc, color=PALETTE[cond], label=cond)
        ax.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc,
                        color=PALETTE[cond], alpha=0.2)

    ah.add_event_lines(ax, times, cue_time=CUE_TIME[modality])
    # Shade the two averaging windows feeding the pre-/post-cue means.
    ax.axvspan(*data["precue_win"], color="gray", hatch="/", alpha=0.08, zorder=0, label="pre-cue")
    ax.axvspan(*data["postcue_win"], color="gray", alpha=0.20, zorder=0, label="post-cue")
    ax.set_title(ch_name, fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(
        f"{MODALITY_NAME[modality]} (n={data['n_subjects']})\ndITPC" if is_first_column else "dITPC",
        fontsize=11,
    )
    ax.tick_params(labelsize=10)
    # No per-axis legend: the handles are collected once for a figure level legend, which
    # keeps the traces free of an overlapping legend box.
    sns.despine(trim=True, ax=ax)


def plot_paired_bar(ax, values_no, values_yes, labels, colors_pair, ylabel, title):
    """Paired bar plot: one grey connector line and two dots per subject."""
    n_pair = min(len(values_no), len(values_yes))
    for subj_idx in range(n_pair):
        ax.plot([0, 1], [values_no[subj_idx], values_yes[subj_idx]],
                color="0.7", linewidth=1, zorder=1)
    means = [np.mean(values_no), np.mean(values_yes)]
    sems = [
        stats.sem(values_no) if len(values_no) > 1 else 0.0,
        stats.sem(values_yes) if len(values_yes) > 1 else 0.0,
    ]
    ax.bar([0, 1], means, yerr=sems, color=colors_pair, capsize=4, width=0.7, zorder=2)
    ax.scatter(np.zeros_like(values_no), values_no, color="black", s=12, zorder=3)
    ax.scatter(np.ones_like(values_yes), values_yes, color="black", s=12, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=10)
    sns.despine(trim=True, ax=ax)


def plot_precue_postcue_delta(ax, data):
    """
    Paired plot of the Yes-No dITPC difference in the pre-cue vs post-cue window.

    Each subject contributes one (pre, post) pair; a significant post > pre shift supports
    the idea that the dITPC difference is driven by the cue.
    """
    p_val = data["p_delta"]
    plot_paired_bar(
        ax,
        data["diff_pre"], data["diff_post"],
        ["pre-cue", "post-cue"],
        ["0.6", "0.8"],
        "Yes - No dITPC",
        # Star on the first line, the p value underneath it.
        f"{significance_label(p_val)}\np_fdr={p_val:.3g}",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)


def plot_subject_phase_polar(ax, series, title, n_bins=18, subject_sig=None):
    """
    Polar rose of one circular-mean phase per subject, for ONE or TWO overlaid distributions.

    series : list of (phases, color, label, rayleigh) tuples. `rayleigh` is the dict built by
    summarise_channel (n / mean_angle / R / z / p), or None to draw no statistics.
    Two series overlay cleanly here because each is drawn as a step OUTLINE with a faint fill;
    the opaque bars this used to draw would hide whichever distribution went down first.

    subject_sig : optional boolean array, one entry per subject of the FIRST series; True where
    that subject's trial-level Rayleigh test is significant. Dots of the other subjects are
    drawn grey and the count k/N is written under the header. Only valid with a single series.
    """
    if subject_sig is not None:
        subject_sig = np.asarray(subject_sig, dtype=bool)
        if len(series) != 1 or len(subject_sig) != len(series[0][0]):
            raise ValueError("subject_sig needs one series with one flag per subject.")
    series = [s for s in series if len(s[0]) > 0]
    if not series:
        ax.set_title(f"{title}\nno data", fontsize=10)
        return

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts_per_series = [np.histogram(phases, bins=bins)[0] for phases, _, _, _ in series]
    r_bars = max(1, int(max(c.max() for c in counts_per_series)))
    # One subject ring per series, stacked outwards so the dots of two overlaid distributions
    # never collide.
    ring_radii = [r_bars * (1.16 + 0.14 * i) for i in range(len(series))]
    r_outer = ring_radii[-1] * 1.05

    for (phases, color, _label, rayleigh), counts, r_ring in zip(
        series, counts_per_series, ring_radii
    ):
        # Step outline. The path starts and ends at radius 0 so the polygon closes THROUGH THE
        # ORIGIN; closing it last-point-to-first would put a +pi -> -pi segment in the path,
        # which is a backwards sweep across the whole circle rather than the no-op it looks
        # like. matplotlib curves each segment in the polar projection, so consecutive points
        # at a constant radius come out as arcs.
        theta_step = np.repeat(bins, 2)[1:-1]
        r_step = np.repeat(counts, 2)
        theta_closed = np.concatenate([[bins[0]], theta_step, [bins[-1]]])
        r_closed = np.concatenate([[0], r_step, [0]])
        ax.plot(theta_closed, r_closed, color=color, linewidth=1.6, zorder=3)
        ax.fill(theta_closed, r_closed, color=color, alpha=0.13, zorder=2)

        # Subject scatter ring, one dot per subject.
        if subject_sig is None:
            ax.scatter(phases, np.full(len(phases), r_ring), s=8, color=color,
                       zorder=5, clip_on=False)
        else:
            # Subjects without a consistent trial-level phase are greyed out.
            ax.scatter(phases[~subject_sig], np.full(int((~subject_sig).sum()), r_ring), s=8,
                       color="0.75", zorder=5, clip_on=False)
            ax.scatter(phases[subject_sig], np.full(int(subject_sig.sum()), r_ring), s=8,
                       color=color, zorder=6, clip_on=False)

        # Group mean resultant vector. The angle is the group mean phase; the length is R on
        # its OWN 0-1 scale, where R = 1 spans the full radius. Deliberately not scaled to the
        # count axis, which would make a weakly concentrated distribution look strongly
        # directional. Drawn twice, a fat white arrow under a thin coloured one, because an
        # arrow in the series' own colour lying on its own rose is otherwise invisible.
        mean_angle = float(np.angle(np.mean(np.exp(1j * phases))))
        R = float(np.abs(np.mean(np.exp(1j * phases))))
        if rayleigh is not None and np.isfinite(rayleigh["mean_angle"]):
            mean_angle, R = rayleigh["mean_angle"], rayleigh["R"]
        for arrow_color, lw, mut, z in (("white", 4.5, 18, 6), (color, 2.0, 12, 7)):
            ax.annotate(
                "", xy=(mean_angle, R * r_outer), xytext=(mean_angle, 0), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=lw,
                                mutation_scale=mut),
            )

    ax.set_ylim(0, r_outer)
    # Counts are integers, and at this panel size only a few radial labels stay readable.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.tick_params(labelsize=8, pad=0)

    # One statistics line per series, with the uncorrected Rayleigh p - these tests are not
    # part of an FDR family (see collect_and_summarise). With a single series the label is
    # left out, the header already says what is shown.
    stat_lines = []
    if subject_sig is not None:
        stat_lines.append(
            f"within-subject Rayleigh p<{PHASE_SUBJECT_ALPHA:g}: "
            f"{int(subject_sig.sum())}/{len(subject_sig)} subjects"
        )
    for phases, _color, label, rayleigh in series:
        prefix = f"{label}: " if len(series) > 1 else ""
        if rayleigh is None:
            stat_lines.append(f"{prefix}n={len(phases)}")
            continue
        stat_lines.append(
            f"{prefix}z={rayleigh['z']:.2f}, p={rayleigh['p']:.3g} "
            f"{significance_label(rayleigh['p'])}, R={rayleigh['R']:.2f}"
        )
    # Statistics as the axes title, the header in a slightly larger font directly above it.
    stats_title = ax.set_title("\n".join(stat_lines), fontsize=8, pad=12)
    ax.annotate(title, xy=(0.5, 1.0), xycoords=stats_title, xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=10)


def plot_modality_interaction(ax, res, ch_name):
    """
    Response x modality interaction plot for one channel (panel D).

    Thin grey lines connect each subject's No and Yes post-cue dITPC within a modality; the
    coloured markers are the cell mean +- SEM, joined across modalities per response. Every
    subject with that session is drawn, matching panels A and B. The tested effects are not
    marked in the plot; they are in the mixed model table written next to the figure.
    """
    x_pos = {m: i for i, m in enumerate(MODALITIES)}
    offset = {"No": -0.12, "Yes": 0.12}
    for m in MODALITIES:
        for v_no, v_yes in zip(res["wide"][m]["No"], res["wide"][m]["Yes"]):
            ax.plot([x_pos[m] + offset["No"], x_pos[m] + offset["Yes"]], [v_no, v_yes],
                    color="0.8", linewidth=0.8, zorder=1)
    for resp in ("No", "Yes"):
        means = [res["cell_means"][m][resp] for m in MODALITIES]
        sems = [res["cell_sems"][m][resp] for m in MODALITIES]
        xs = [x_pos[m] + offset[resp] for m in MODALITIES]
        ax.errorbar(xs, means, yerr=sems, color=PALETTE[resp], marker="o", markersize=7,
                    linewidth=2, capsize=4, zorder=3, label=resp)

    # Limits must cover every subject line.
    all_vals = np.concatenate([res["wide"][m][r] for m in MODALITIES for r in ("No", "Yes")])
    y_min, y_max = all_vals.min(), all_vals.max()
    pad = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([f"{MODALITY_NAME[m]} (n={len(res['subject_ids'][m])})"
                        for m in MODALITIES])
    ax.set_xlim(-0.5, len(MODALITIES) - 0.5)
    ax.set_ylabel("Post-cue dITPC", fontsize=11)
    ax.tick_params(labelsize=10)
    # FDR-corrected p of each tested effect under the channel name (no stars).
    eff = res["effects"]
    ax.set_title(
        f"{ch_name}\nResponse: {format_p(eff['response']['p_fdr'])}\n"
        f"Modality: {format_p(eff['modality']['p_fdr'])}\n"
        f"Response x Modality: {format_p(eff['response:modality']['p_fdr'])}",
        fontsize=10,
    )
    sns.despine(trim=True, ax=ax)


# %%
# ---------------------------------------------------------------- figure assembly


def plot_figure(summary, modalities, lmm):
    """Assemble the manuscript panel (2x2); return the figure."""
    fig = plt.figure(figsize=(13, 11))
    outer = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.22,
                     height_ratios=[1.0, 0.95])

    first_axes = {}

    # --- Panels A and B: one dITPC block per channel, rows = modality ---
    for ch_idx, ch_name in enumerate(CHANNELS):
        gs_block = outer[0, ch_idx].subgridspec(len(modalities), 3, hspace=0.65, wspace=0.55)
        for row_idx, modality in enumerate(modalities):
            data = summary[modality][ch_name]

            ax_tc = fig.add_subplot(gs_block[row_idx, 0])
            ax_bar = fig.add_subplot(gs_block[row_idx, 1])
            ax_delta = fig.add_subplot(gs_block[row_idx, 2])
            if row_idx == 0:
                first_axes[ch_name] = ax_tc

            plot_itpc_timecourse(ax_tc, data, ch_name, modality,
                                 is_first_column=(ch_idx == 0))
            p_bar = data["p_bar"]
            plot_paired_bar(
                ax_bar,
                data["mean_itpc_no_bar"], data["mean_itpc_yes_bar"],
                ["No", "Yes"],
                [PALETTE["No"], PALETTE["Yes"]],
                "Post-cue dITPC",
                # Star on the first line, the p value underneath it.
                f"{significance_label(p_bar)}\np_fdr={p_bar:.3g}",
            )
            plot_precue_postcue_delta(ax_delta, data)

    # --- Panel C: subject level phase at the readout time, CP4 only ---
    # One polar axes per modality, showing the two responses POOLED. No response contrast is
    # drawn here at all - panels A and B already carry that, as dITPC. The Rayleigh test of
    # each axes is reported uncorrected (see collect_and_summarise).
    polar_channel = CHANNELS[0]
    # One row, so the axes fill the quadrant vertically. Polar axes keep a square aspect, so
    # they are sized by the column width and leave some whitespace above and below. The wspace
    # stays generous because the subject ring sticks out beyond the axes.
    gs_polar = outer[1, 0].subgridspec(1, len(modalities), wspace=0.35)
    for col_idx, modality in enumerate(modalities):
        data = summary[modality][polar_channel]
        ax = fig.add_subplot(gs_polar[0, col_idx], projection="polar")
        if col_idx == 0:
            first_axes["polar"] = ax
        plot_subject_phase_polar(
            ax,
            [(data["phase_at"]["all"], PALETTE_MOD[modality], "Pooled",
              data["phase_rayleigh"]["all"])],
            MODALITY_NAME[modality],
            subject_sig=data["phase_subject_sig"],
        )
    polar_axes = [ax for ax in fig.axes if ax.name == "polar"]

    # --- Panel D: response x modality interaction, one plot per channel, stacked ---
    gs_lmm = outer[1, 1].subgridspec(len(CHANNELS), 1, hspace=1.1)
    for ch_idx, ch_name in enumerate(CHANNELS):
        ax_d = fig.add_subplot(gs_lmm[ch_idx, 0])
        if ch_idx == 0:
            first_axes["lmm"] = ax_d
        plot_modality_interaction(ax_d, lmm[ch_name], ch_name)

    # Shared legend just below the first row of panels (no figure title).
    handles, labels = first_axes[CHANNELS[0]].get_legend_handles_labels()
    legend_y = outer[0, 0].get_position(fig).y0 - 0.025
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, legend_y),
               ncol=len(labels), frameon=False, fontsize=11)

    # Panel letters, anchored to each block once the layout is final.
    fig.canvas.draw()
    letter_anchors = [
        ("A", first_axes[CHANNELS[0]]),
        ("B", first_axes[CHANNELS[1]]),
        ("C", first_axes["polar"]),
        ("D", first_axes["lmm"]),
    ]
    letter_xy = {}
    for letter, ax_ref in letter_anchors:
        pos = ax_ref.get_position()
        letter_xy[letter] = (max(pos.x0 - 0.055, 0.007), min(pos.y1 + 0.035, 0.995))
    # C shares its column with A and its row with D, so it takes A's x and D's y.
    letter_xy["C"] = (letter_xy["A"][0], letter_xy["D"][1])
    for letter, (x, y) in letter_xy.items():
        fig.text(x, y, letter, fontsize=18, fontweight="bold", va="top", ha="left")

    # Panel C title, on the line of its letter and centred over the polar axes.
    polar_x0 = min(ax.get_position().x0 for ax in polar_axes)
    polar_x1 = max(ax.get_position().x1 for ax in polar_axes)
    fig.text((polar_x0 + polar_x1) / 2, letter_xy["C"][1],
             f"Alpha phase at {PHASE_READOUT_TIME * 1000:.0f} ms {polar_channel}",
             fontsize=11, va="top", ha="center")

    return fig


# %%
# ---------------------------------------------------------------- ERP regression


def compute_subject_abs_erp(broadband_fif, modality, response_alpha, times_itpc, subID):
    """
    |ERP| per response and channel from the broadband epochs of one session.

    Steps: clean-trial mask -> CSD -> pick CHANNELS -> pre-cue baseline -> mean over the trials
    of one response -> absolute value -> crop to the dITPC time axis. The trial order must equal
    that of the alpha epochs the dITPC was computed from, and so must the time axis.
    Returns {channel: {"No": array, "Yes": array}}.
    """
    epochs = mne.read_epochs(broadband_fif, preload=True, verbose=False)
    if epochs.metadata is None:
        raise ValueError(f"Missing metadata in {broadband_fif}.")
    keep = (
        (epochs.metadata["analysis_excluded"] == False)  # noqa: E712 (pandas mask)
        & (epochs.metadata["drop_epoch"] == False)  # noqa: E712
        & (epochs.metadata["response_decoded"].isin(["Yes", "No"]))
    )
    epochs = epochs[keep.to_numpy()]

    response_bb = epochs.metadata["response_decoded"].to_numpy()
    if len(response_bb) != len(response_alpha) or not np.array_equal(response_bb, response_alpha):
        raise ValueError(
            f"sub-0{subID} {modality}: broadband epochs ({len(response_bb)}) and alpha epochs "
            f"({len(response_alpha)}) differ in trial count or response order."
        )

    epochs_csd = ah.apply_surface_laplacian(epochs).pick(CHANNELS)
    apply_baseline_epochs(epochs_csd, broadband_fif, baseline_window=BASELINE_BY_MOD[modality])

    times = epochs_csd.times
    tmask = (times >= WIN_TMIN) & (times <= WIN_TMAX)
    if tmask.sum() != len(times_itpc) or not np.allclose(times[tmask], times_itpc):
        raise ValueError(
            f"sub-0{subID} {modality}: ERP time axis ({tmask.sum()} samples) does not match "
            f"the dITPC time axis ({len(times_itpc)} samples)."
        )

    data = epochs_csd.get_data(copy=True)[:, :, tmask]  # (n_trials, n_channels, n_times)
    abs_erp = {}
    for ch_idx, ch in enumerate(CHANNELS):
        abs_erp[ch] = {
            resp: np.abs(data[response_bb == resp, ch_idx, :].mean(axis=0)) for resp in RESPONSES
        }
    return abs_erp


def regress_out_erp(itpc_no, itpc_yes, erp_no, erp_yes):
    """
    Shared-slope removal of |ERP| from dITPC, per subject (row). All inputs (n_subjects, n_times).

    Returns (clean_no, clean_yes, slope, r2) with slope and r2 of shape (n_subjects,).
    """
    fit = slice(N_EDGE_SAMPLES, None)
    clean_no = np.empty_like(itpc_no)
    clean_yes = np.empty_like(itpc_yes)
    slope = np.empty(itpc_no.shape[0])
    r2 = np.empty(itpc_no.shape[0])
    for s in range(itpc_no.shape[0]):
        x = np.concatenate([erp_no[s, fit], erp_yes[s, fit]])
        y = np.concatenate([itpc_no[s, fit], itpc_yes[s, fit]])
        x_mean = x.mean()
        var_x = np.sum((x - x_mean) ** 2)
        if var_x == 0:
            raise ValueError(f"Subject row {s}: |ERP| is constant, the regression is undefined.")
        slope[s] = np.sum((x - x_mean) * (y - y.mean())) / var_x
        r2[s] = np.corrcoef(x, y)[0, 1] ** 2
        clean_no[s] = itpc_no[s] - slope[s] * (erp_no[s] - x_mean)
        clean_yes[s] = itpc_yes[s] - slope[s] * (erp_yes[s] - x_mean)
    if not (np.all(np.isfinite(clean_no)) and np.all(np.isfinite(clean_yes))
            and np.all(np.isfinite(r2))):
        raise ValueError("Non-finite value after regressing out |ERP|.")
    return clean_no, clean_yes, slope, r2


# %%
# ---------------------------------------------------------------- data collection


def collect_erp_regressed_data(band, modality, subjects):
    """
    Loop over subjects of one modality and return (times_itpc, times_phase, store).

    store[ch] has the keys the dITPC summary expects (itpc_yes, itpc_no, phase_yes, phase_no,
    phase_all, subject_ids), with itpc_yes / itpc_no the dITPC after removing |ERP|.
    """
    itpc_tr = {ch: {r: [] for r in RESPONSES} for ch in CHANNELS}
    erp_tr = {ch: {r: [] for r in RESPONSES} for ch in CHANNELS}
    phase_tr = {ch: {k: [] for k in ("yes", "no", "all")} for ch in CHANNELS}
    trial_p_tr = {ch: [] for ch in CHANNELS}
    subject_ids = []
    times = None
    times_phase = None

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
            f"lfreqori-{L_FREQ_ORIG}_lfreq-{L_FREQ}_hfreq-{H_FREQ}"
        )
        epochs_ff, _ = ah.load_band_epochs_and_metadata(
            sub_dir, base_name, f"{band}", subID, modality,
            FILE_SUFFIX_EPOCHS, FILE_SUFFIX_META,
        )
        if epochs_ff is None:
            continue
        broadband_fif = os.path.join(
            sub_dir, f"{base_name}_epochs{FILE_SUFFIX_EPOCHS}.fif"
        )
        if not os.path.exists(broadband_fif):
            raise FileNotFoundError(f"Broadband epochs not found: {broadband_fif}")

        result = compute_subject_itpc_and_phase(epochs_ff, band, subID, modality)
        if result is None:
            continue
        subject_times, subject_times_phase, per_channel = result

        # All subjects must share the same time axes; anything else is a preprocessing
        # inconsistency and must not be silently reconciled.
        if times is None:
            times = subject_times.copy()
            times_phase = subject_times_phase.copy()
        elif (not np.allclose(times, subject_times)
              or not np.allclose(times_phase, subject_times_phase)):
            raise ValueError(f"Time axis mismatch for sub-0{subID} {modality} {band}.")

        abs_erp = compute_subject_abs_erp(
            broadband_fif, modality, epochs_ff.metadata["response_decoded"].to_numpy(),
            subject_times, subID,
        )
        for ch in CHANNELS:
            itpc_tr[ch]["No"].append(per_channel[ch]["itpc_no"])
            itpc_tr[ch]["Yes"].append(per_channel[ch]["itpc_yes"])
            for resp in RESPONSES:
                erp_tr[ch][resp].append(abs_erp[ch][resp])
            for key in ("yes", "no", "all"):
                phase_tr[ch][key].append(per_channel[ch][f"phase_{key}"])
            trial_p_tr[ch].append(per_channel[ch]["phase_trial_p"])
        subject_ids.append(subID)

    if times is None:
        raise RuntimeError(f"No subject contributed {band} data for modality {modality}.")

    store = {}
    for ch in CHANNELS:
        clean_no, clean_yes, slope, r2 = regress_out_erp(
            np.stack(itpc_tr[ch]["No"]), np.stack(itpc_tr[ch]["Yes"]),
            np.stack(erp_tr[ch]["No"]), np.stack(erp_tr[ch]["Yes"]),
        )
        print(f"  {modality} {ch}: mean slope {slope.mean():.4f}, mean R^2 {r2.mean():.3f}")
        store[ch] = dict(
            itpc_yes=clean_yes, itpc_no=clean_no,
            phase_yes=np.stack(phase_tr[ch]["yes"]),
            phase_no=np.stack(phase_tr[ch]["no"]),
            phase_all=np.stack(phase_tr[ch]["all"]),
            phase_trial_p=np.asarray(trial_p_tr[ch]),
            subject_ids=list(subject_ids),
        )
    print(f"  {modality}: {len(subject_ids)} subjects contributed")
    return times, times_phase, store


def collect_and_summarise(band, subjects, modalities):
    """Per modality: collect the ERP-regressed data, summarise it and FDR-correct the dITPC
    tests across the plotted channels. The Rayleigh tests of panel C stay uncorrected."""
    summary = {}
    for modality in modalities:
        print(f"\n===== {band} / {modality} (|ERP| regressed out) =====")
        times, times_phase, store = collect_erp_regressed_data(band, modality, subjects)
        channel_data = {
            ch: summarise_channel(store[ch], times, times_phase, modality)
            for ch in CHANNELS
        }
        # FDR (Benjamini-Hochberg) across the plotted channels, one family per modality
        # and per test.
        for p_key in ("p_bar", "p_delta"):
            corrected = ah.fdr_correct_pvals([channel_data[ch][p_key] for ch in CHANNELS])
            for ch, p_corr in zip(CHANNELS, corrected):
                channel_data[ch][p_key] = p_corr
                channel_data[ch]["n_tests"] = len(CHANNELS)
        summary[modality] = channel_data
    return summary


# %%
# ---------------------------------------------------------------- main


def main():
    args = parse_args()
    band = args.band
    os.makedirs(FIGUREDIR, exist_ok=True)
    os.makedirs(STATSDIR, exist_ok=True)
    os.makedirs(MANUSCRIPT_IMGDIR, exist_ok=True)

    # Panel D needs both sessions; with a single modality the mixed model is not defined.
    if set(args.modalities) != set(MODALITIES):
        raise ValueError(
            f"Panel D (response x modality) needs both modalities {MODALITIES}, "
            f"got {args.modalities}."
        )
    summary = collect_and_summarise(band, args.sub, args.modalities)
    lmm = run_modality_lmm(summary)

    table = build_stats_table(band, summary, args.modalities, lmm)
    stats_csv = os.path.join(STATSDIR, f"manuscript_itpc_erp_regressed_stats_{band}.csv")
    table.to_csv(stats_csv, index=False)
    print(f"\n=== {band} dITPC statistics, |ERP| regressed out (p_fdr: Benjamini-Hochberg over "
          f"{len(CHANNELS)} channels within modality; the phase_rayleigh rows are uncorrected) ===")
    print(table.to_string(index=False))
    print(f"written to {stats_csv}")

    lmm_table = build_lmm_table(band, lmm)
    lmm_csv = os.path.join(STATSDIR, f"manuscript_itpc_erp_regressed_modality_lmm_{band}.csv")
    lmm_table.to_csv(lmm_csv, index=False)
    print(f"\n=== {band} panel D mixed model, |ERP| regressed out ===")
    print(lmm_table.to_string(index=False))
    print(f"written to {lmm_csv}")

    fig = plot_figure(summary, args.modalities, lmm)
    out_path = os.path.join(FIGUREDIR, f"manuscript_itpc_erp_regressed_{band}")
    fig.savefig(f"{out_path}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    manuscript_stem = os.path.join(MANUSCRIPT_IMGDIR, MANUSCRIPT_STEM)
    fig.savefig(f"{manuscript_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{manuscript_stem}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}.pdf / .png")
    print(f"Saved manuscript figure: {manuscript_stem}.svg / .png")

    return dict(summary=summary, lmm=lmm, table=table, fig=fig)


if __name__ == "__main__":
    results = main()

# %%

