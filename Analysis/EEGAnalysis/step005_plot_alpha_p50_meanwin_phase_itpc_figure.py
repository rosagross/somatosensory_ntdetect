"""
File: step005_plot_alpha_p50_meanwin_phase_itpc_figure.py
Author: Rosa Grossmann
Date: 2026-09-24
Description:
Publication figure on pre-stimulus alpha and the P50 at CP4. The per-trial P50 in panels (a) and
(b) is the MEAN over the modality's whole P50_WIN. Rows are the modality - ELECTRICAL on top,
MECHANICAL underneath - so the reader compares modalities vertically and panels horizontally:

  a) P50 amplitude at CP4 across N_BINS within-session pre-stimulus alpha-AMPLITUDE bins, for
     all trials and for perceived and unperceived trials separately.

  b) P50 amplitude at CP4 across N_PHASE_BINS fixed-width pre-stimulus alpha-PHASE bins, drawn
     as bars on a polar axis. The bar colour encodes the bin's P50 amplitude, so the plotted
     quantity is readable without tracing the radius.

  c) Scalp topographies of pre-stimulus alpha and of the P50, four per modality (2 x 2 block:
     ELECTRICAL block on top, MECHANICAL below). Top row: all clean trials (perceived and
     unperceived pooled); bottom row: perceived MINUS unperceived (Yes - No). Per session:
       - alpha: trial-average Hilbert envelope at TOPO_ALPHA_TIME (-100 ms);
       - P50: signed trial-average broadband signal at the fixed latency TOPO_P50_TIME
         (elec 42 ms, mecha 62 ms), after the pre-cue baseline. This is a single-sample readout,
         unlike the window mean of panels (a) and (b), because a topography is a snapshot at one
         latency.
     Both maps are surface-Laplacian (CSD) transformed on the full montage, so they share one
     spatial filter and one unit (uV/cm^2). The grand averages are shown; CP4 and C6 are circled.
     No statistics are drawn on the panel. Two TANOVAs are written to
     manuscript_fig07_p50meanwin_topo_tanova.csv: (1) the pooled alpha map vs the pooled P50 map,
     (2) the Yes - No alpha difference map vs the Yes - No P50 difference map. Each map is
     centred over electrodes and divided by its global field power (GFP), the statistic is the
     GFP of the grand-mean difference map (global dissimilarity, 0-2), and the null distribution
     comes from TANOVA_N_PERMUTATIONS random within-session swaps of the alpha/P50 labels (sign
     flips of each session's difference map, seed SEED). The per-session spatial Pearson r
     between the two maps of each test is written alongside.

  d) Interpretation aid for panel (b), not a result: per modality, a COSINE_FREQ_HZ cosine whose
     phase at PHASE_READOUT_TIME equals the average phase of the panel (b) bin with the HIGHEST
     P50 (the bin's circular mean over sessions of the per-session mean phase), drawn in the
     modality colour. Dots mark the phase each wave reaches at the readout and at stimulus onset;
     near the P50 each wave's peak closest to COSINE_P50_MS is marked with its time in ms.

MEASUREMENT CONVENTIONS. Every quantity is computed here from the epoch files on disk.

  * THE P50 IN PANELS (a) AND (b) IS THE MEAN OVER THE MODALITY'S P50_WIN (elec 34-56 ms, mecha
    44-66 ms). Each trial is averaged over the full window, with no peak search at the session or
    the trial level. The window is fixed in advance, so it cannot be circular with respect to
    Yes/No or alpha. A per-trial argmax would be biased, because the max of noisy samples scales
    with the trial's noise level, which the pre-stimulus alpha amplitude largely reflects.

  * PRE-STIMULUS WINDOWS STOP AT -0.1 s. The alpha band-pass is zero-phase, so its impulse
    response is two-sided and it ran before the crop with post-stimulus data in its input.
    Post-stimulus signal therefore leaks BACKWARDS into anything called "pre-stimulus". Ending
    at -0.1 s roughly halves that leak (2.86% of the P50 over (-0.40, 0.0) vs 1.50% over
    (-0.40, -0.10)). It is reduced, NOT eliminated, and it runs in exactly the direction
    panels (a) and (b) test.

  * SPATIAL FILTERS. The alpha epochs get a surface Laplacian (CSD), so alpha quantities are in
    uV/cm^2. The broadband P50 of panels (a) and (b) stays in the recorded (average-reference)
    space. Panel (c) applies the CSD to both of its maps.

  * BASELINE is the pre-cue window BASELINE_BY_MOD (-0.55..-0.45 s), not (-0.1, 0). The
    (-0.1, 0) window sits INSIDE the pre-stimulus alpha window, so baselining there would induce
    a spurious alpha->P50 relationship.

  * UNITS: the preprocessed FIFs carry FIFF_UNIT_V but the sample values are numerically
    microvolts. No 1e6 scaling is applied; instead the plausible microvolt range is checked per
    session so a file that really is in volts crashes loudly.

STATISTICS ON THE FIGURE. Panel (a) carries the subject-level slope across bins (one straight
line fitted per subject, then a one-sample t-test on the slopes); the p is uncorrected, since
only one channel is plotted. Panels (b) and (c) are descriptive; panel (c)'s two TANOVAs are in their CSV.

SEPARATE FIGURE. Alongside the composite the script writes a standalone
`p50meanwin_by_intensity_modality` figure: the same CP4 P50 as panel (a), but binned by the
STIMULUS INTENSITY LEVEL instead of by pre-stimulus alpha, as bars grouped by modality in the
project's standard modality colours. Trials are pooled over Yes/No on purpose - the question
there is about the stimulus, not the report - and the figure is purely descriptive (mean and SEM
across sessions, no test annotated). Only intensity levels 2-6 appear: `analysis_excluded`
already removes the catch trials (level 0) and the two outermost levels (1 and 7), which were
presented half as often.

Run:
    python step005_plot_alpha_p50_meanwin_phase_itpc_figure.py
    python step005_plot_alpha_p50_meanwin_phase_itpc_figure.py --sub 13 14 15 16
"""

# %%
import argparse
import os
import pickle
import sys

import matplotlib
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import hilbert as scipy_hilbert

DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
WDDIR = os.getenv("WDDIR")
if not DATADIR or not RESDIR or not WDDIR:
    raise EnvironmentError("DATADIR, RESDIR and WDDIR must be set (source bat/start.sh).")


def running_in_kernel():
    """
    True when the script is executed as '# %%' cells in Jupyter/VS Code.

    ipykernel appends its own '--f=<connection file>' to sys.argv, which argparse would reject,
    so in that case the CLI is skipped and the defaults below are used.
    """
    return os.path.basename(sys.argv[0]).startswith("ipykernel_launcher")


# Batch runs must not need a display; interactive cell runs keep whatever backend is active.
if not running_in_kernel():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# plotting_style is a per-directory copy and eeg_analysis_helpers lives next to this file, so
# both are imported BEFORE WDDIR joins sys.path 
from plotting_style import figure_style  # noqa: E402
import eeg_analysis_helpers as ah  # noqa: E402

if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)
from general_helpers import apply_baseline_epochs, is_subject_excluded  # noqa: E402

# %%
# ---------------------------------------------------------------- configuration
# Every constant below is copied verbatim from step019_prestim_alpha_p50_cnv.py. Changing one
# here silently desynchronises this figure from the analysis it is meant to display.

FILE_SUFFIX_EPOCHS = "_preprestim_corrected"
FILE_SUFFIX_META = "_preprestim"
L_FREQ_ORIG, L_FREQ, H_FREQ = 0.1, 1, 40

BAND = "alpha"          # the file on disk is f"{BAND}"
# Lower band edge, used ONLY to size the Hilbert reflect padding (3 cycles of the slowest
# component). It is never used to filter: the saved epochs are already band-passed at the
# subject's individual alpha peak +/- 2 Hz.
ALPHA_FMIN = 8

ALL_SUBJECTS = list(range(13, 53))
MODALITIES = ["elec", "mecha"]

# Panels (a), (b) and (d) live at CP4.
P50_CHANNEL = "CP4"

# Cue -> stimulus minus the last 100 ms (see the leak note in the docstring). The mechanical
# stimulator has a ~30 ms onset delay, so its cue sits at -0.43 s and the electrical one at
# -0.40 s (project-wide convention).
PRESTIM_WIN = {"elec": (-0.40, -0.10), "mecha": (-0.40, -0.10)}
# Pre-cue baseline: ends exactly at the cue, so both ERP and CNV are deflections from the
# pre-cue level. Deliberately overrides infer_baseline_window_from_suffix().
BASELINE_BY_MOD = {"elec": (-0.55, -0.45), "mecha": (-0.55, -0.45)}

# Window the per-trial P50 is averaged over, per modality. The mechanical window sits 5 ms
# later because of the stimulator's onset delay.
P50_WIN = {"elec": (0.034, 0.056), "mecha": (0.044, 0.066)}

N_BINS = 8              # equiprobable alpha-AMPLITUDE bins, panel (a)
N_PHASE_BINS = 16       # fixed-width alpha-PHASE bins, panel (b)
# Fixed timepoint at which the pre-stimulus alpha phase is read out as a scalar. Same anchor as
# the end of PRESTIM_WIN, for the same back-leak reason.
PHASE_READOUT_TIME = -0.1

# Panel (c). Readout times of the topographies: alpha 100 ms before PRESTIM_WIN's end, clear of
# the backward filter leak; P50 at a fixed latency per modality.
TOPO_ALPHA_TIME = -0.1
TOPO_P50_TIME = {"elec": 0.042, "mecha": 0.062}
# Channels circled on the topographies.
TOPO_HIGHLIGHT = ["CP4", "C6"]
# TANOVAs on the panel (c) maps (CSV only): within-session label swaps and the test level.
TANOVA_N_PERMUTATIONS = 5000
TANOVA_ALPHA = 0.05
SEED = 42

# Panel (d), illustration only. 10 Hz is a round value for display, not the subjects' individual
# alpha peak. One wave per modality is anchored at PHASE_READOUT_TIME at the average phase of the
# panel (b) bin with the highest P50.
COSINE_FREQ_HZ = 10
COSINE_WIN_MS = (-125, 150)
COSINE_HIGHLIGHT_WIDTH_MS = 4
# Plain event markers, (time in ms, name). 50 ms is an illustrative P50 time.
COSINE_P50_MS = 50
COSINE_MARKERS_MS = [(0, "stimulus onset"), (COSINE_P50_MS, "P50")]

# Unusable recordings, not merely low-SNR: sub-23 mecha had ground and reference swapped,
# sub-36 mecha had exceptionally high force values.
FAULTY_SESSIONS = {("mecha", 23), ("mecha", 36), ("elec", 19), ("elec", 28)}


# The outermost samples of a Hilbert envelope are degenerate (the reflect padding makes the
# signal even-symmetric about the last real sample), so every readout is kept this far clear of
# the epoch boundary.
N_EDGE_SAMPLES = 5
PLAUSIBLE_UV_STD = (0.1, 200.0)

MODALITY_LABELS = {"elec": "Electrical", "mecha": "Mechanical"}

colors, _ = figure_style()
PALETTE_RESP = {"Yes": colors["perceived"], "No": colors["unperceived"]}
# Project-wide modality colours. Used by the separate intensity figure only.
PALETTE_MODALITY = {"elec": colors["electrical"], "mecha": colors["mechanical"]}

# Panel (b) colour scale. Diverging and centred on zero because the phase-bin P50 means
# genuinely cross zero; a sequential map would hide the sign.
PHASE_BAR_CMAP = "coolwarm"

FIGUREDIR = os.path.join(WDDIR, "Writing", "img")
STATSDIR = os.path.join(RESDIR, "Statistics", "eeg", "prestim_alpha_p50_cnv")
MANUSCRIPT_STEM = "Fig07"
FIGURE_STEM = "Fig07"
INTENSITY_FIGURE_STEM = "p50meanwin_by_intensity_modality"
PUB_RC = {
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "font.size": 16,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.3,
}

# Font sizes set per artist rather than through the rc context, kept together so the whole
# figure can be re-scaled from one place.
FS_PANEL_TITLE = 13     # titles over the panel (c) topographies (name + readout time)
FS_PANEL_STAT = 14      # the slope line above panel (a)
FS_LEGEND = 13
FS_POLAR_TICK = 13
FS_CBAR_LABEL = 14
FS_CBAR_TICK = 13
FS_ROW_LABEL = 18       # "Electrical (n = 36)" down the left edge
FS_PANEL_LETTER = 21
FS_COSINE_LABEL = 14    # event names and phase labels in panel (d)

# The whole figure, and how much of its height the panel (d) strip gets relative to one modality
# row. The height was raised from 8.8 in when panel (d) was added, so rows a-c keep their size.
FIGSIZE = (18.0, 16.0)
D_ROW_RATIO = 0.5
D_ROW_HSPACE = 0.42


def parse_args(argv=None):
    """CLI. Parsed strictly, so a typo on the command line is still an error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", nargs="+", type=int, default=ALL_SUBJECTS,
                        help="Subject IDs (default: 13-52).")
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES,
                        help="Modalities to process (default: both).")
    parser.add_argument("--n-bins", type=int, default=N_BINS,
                        help=f"Alpha-amplitude quantile bins for panel a (default: {N_BINS}).")
    parser.add_argument("--n-phase-bins", type=int, default=N_PHASE_BINS,
                        help=f"Fixed-width alpha-phase bins for panel b "
                             f"(default: {N_PHASE_BINS}).")
    parser.add_argument("--cache-file", default=None,
                        help="Optional pickle of the collected session tables. When the file "
                             "exists it is read INSTEAD of reloading the epochs, so the figure "
                             "can be iterated quickly; when it does not, the full loop runs and "
                             "writes it. Opt-in only - there is no implicit cache.")
    if argv is None:
        argv = [] if running_in_kernel() else sys.argv[1:]
    return parser.parse_args(argv)


# %%
# ---------------------------------------------------------------- helpers
# window_slice / hilbert_analytic / assert_microvolt_scale / check_epoch_index_match /
# fit_linear_slopes / ttest_1samp_finite / sig_stars are copied from step019 rather than
# imported: they are module-level there, and importing step019 would run its whole analysis.


def window_slice(times, win, label):
    """
    Inclusive index slice for a time window. Raises if the window is not contained in `times` - a silently clipped
    window would change what is being measured.
    """
    if win[0] < times[0] - 1e-9 or win[1] > times[-1] + 1e-9:
        raise ValueError(
            f"{label} window {win} is not contained in the available time axis "
            f"[{times[0]:.3f}, {times[-1]:.3f}]."
        )
    t0 = int(np.argmin(np.abs(times - win[0])))
    t1 = int(np.argmin(np.abs(times - win[1]))) + 1
    return slice(t0, t1)


def hilbert_analytic(data, sfreq, fmin):
    """
    Analytic signal via pad -> Hilbert -> trim, on data shaped (n_trials, n_ch, n_times).

    Reflect-padding by three cycles of the slowest band component keeps the Hilbert edge
    transients out of the window of interest. It must run on the FULL epoch before any crop:
    np.pad(mode="reflect") makes the padded signal even-symmetric about the last real sample, so
    cropping first would put a degenerate sample right at the readout edge.

    Returned complex so that the magnitude (envelope, panel a) and the angle (phase, panels b
    and c) come off ONE Hilbert rather than two.
    """
    n_pad = int(3 * sfreq / fmin)
    padded = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode="reflect")
    return scipy_hilbert(padded, axis=-1)[:, :, n_pad:-n_pad]


def assert_microvolt_scale(data, label):
    """
    Verify the amplitude scale really is microvolt-like before it is reported as uV.

    Must be handed EEG channels only: the epoch files also carry CNAP, ECG, RESP and EOG, whose
    values are in other units (CNAP alone has a std in the thousands).
    """
    data_std = float(np.std(data))
    if not (PLAUSIBLE_UV_STD[0] <= data_std <= PLAUSIBLE_UV_STD[1]):
        raise ValueError(
            f"{label}: standard deviation is {data_std:.3e}, outside the plausible microvolt "
            f"range {PLAUSIBLE_UV_STD}. The file's amplitude scale is not what this script "
            f"assumes - check the preprocessing units."
        )


def check_epoch_index_match(meta_a, meta_b, label_a, label_b, subID, modality):
    """
    Assert two metadata frames describe exactly the same trials, in the same order.

    Trial COUNT alone is not enough: the two measures come from two different files for the same
    session and are paired by position, so a reordering would silently mispair them.
    """
    idx_a = meta_a["epoch_index"].to_numpy()
    idx_b = meta_b["epoch_index"].to_numpy()
    if idx_a.shape != idx_b.shape or not np.array_equal(idx_a, idx_b):
        raise ValueError(
            f"sub-0{subID} ses-{modality}: {label_a} ({len(idx_a)} trials) and {label_b} "
            f"({len(idx_b)} trials) do not describe the same trials. Investigate upstream "
            f"preprocessing - do not reconcile here."
        )


def fit_linear_slopes(x_vals, y_mat):
    """
    Fit a straight line per subject across bins and return the slopes.

    x_vals : 1-D array of bin indices
    y_mat  : (n_subjects, n_bins), NaN where a subject has no trials in a bin
    """
    x_vals = np.asarray(x_vals, dtype=float)
    slopes = []
    for row in np.asarray(y_mat, dtype=float):
        mask = np.isfinite(row)
        if np.sum(mask) < 2:
            continue
        slopes.append(np.polyfit(x_vals[mask], row[mask], deg=1)[0])
    return np.asarray(slopes, dtype=float)


def ttest_1samp_finite(values, popmean=0.0):
    """Two-sided one-sample t-test on the finite entries; returns (t, p, n)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan, int(values.size)
    t_stat, p_val = stats.ttest_1samp(values, popmean)
    return float(t_stat), float(p_val), int(values.size)


def sig_stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def format_p(p):
    """Compact p for an annotation: '<.001' below the printable resolution, else 3 decimals."""
    if not np.isfinite(p):
        return "n/a"
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def load_broadband_epochs(epoch_fif):
    """
    Broadband epochs with the clean-trial mask applied. NOT baselined here: the ERP and the CNV
    need the baseline applied at different points in their chains (the CNV is low-passed first,
    so that its baseline is taken on the filtered signal).

    load_clean_epochs() is deliberately NOT used: it hard-codes the suffix-inferred
    (-0.8, -0.7) baseline, which is not the window this analysis needs.
    """
    epochs = mne.read_epochs(epoch_fif, preload=True, verbose=False)
    if epochs.metadata is None:
        raise ValueError(f"Missing metadata in {epoch_fif}.")
    keep = (
        (epochs.metadata["analysis_excluded"] == False)  # noqa: E712 (pandas mask)
        & (epochs.metadata["drop_epoch"] == False)  # noqa: E712
        & (epochs.metadata["response_decoded"].isin(["Yes", "No"]))
    )
    epochs = epochs[keep.to_numpy()]
    if len(epochs) == 0:
        return None
    return epochs


def sample_index(times, t, label):
    """Index of time t, which must lie on the sample grid."""
    i_t = int(np.argmin(np.abs(times - t)))
    if abs(times[i_t] - t) > 0.5 * (times[1] - times[0]):
        raise ValueError(f"{label}: time {t}s is not on the sample grid.")
    return i_t


def normalize_topo(maps):
    """Centre each map (rows) over electrodes and divide by its root sum of squares."""
    centred = maps - maps.mean(axis=1, keepdims=True)
    norm = np.sqrt((centred ** 2).sum(axis=1, keepdims=True))
    if (norm == 0).any():
        raise ValueError("A topography is flat after centring.")
    return centred / norm


def gfp_normalize_topo(maps):
    """Centre each map over electrodes and divide by its GFP (sqrt of the mean square)."""
    centred = maps - maps.mean(axis=1, keepdims=True)
    gfp = np.sqrt((centred ** 2).mean(axis=1, keepdims=True))
    if (gfp == 0).any():
        raise ValueError("A topography is flat after centring.")
    return centred / gfp


def tanova_paired(a_norm, b_norm, n_perm, rng):
    """
    Paired TANOVA. Statistic: GFP of the grand-mean difference map. Null: within-session label
    swaps, which for a paired difference are random sign flips of each session's row. The GFP
    is taken AFTER averaging - a per-session GFP ignores the sign, so it would not change under
    the swaps at all.
    """
    diff_ab = a_norm - b_norm                                     # (n_sessions, n_electrodes)
    n_s = diff_ab.shape[0]
    if n_s < 2:
        raise ValueError("TANOVA needs at least two sessions.")
    def gfp(m):
        return np.sqrt((m ** 2).mean(axis=-1))
    observed = float(gfp(diff_ab.mean(axis=0)))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n_s))
    null = gfp(signs @ diff_ab / n_s)                             # (n_perm,)
    p = (1 + int((null >= observed).sum())) / (1 + n_perm)
    return {"observed": observed, "null": null, "p": p,
            "null_p95": float(np.percentile(null, 95))}


# %%
# ---------------------------------------------------------------- data collection


def collect_sessions(subjects, modalities, n_bins):
    """
    One pass over every session.

    df_trials : one row per TRIAL, carrying the pre-stimulus alpha envelope, the alpha phase
                scalar, the P50 and the within-session alpha-amplitude bin at P50_CHANNEL.
                Feeds panels (a), (b) and (d).
    topo      : dict with one row per SESSION in `rows` (sub, modality, snr_excluded, alpha_map,
                p50_map over the full EEG montage, both CSD) and the montage `info`. Feeds
                panel (c).

    Both carry `snr_excluded`; the caller drops those sessions for the figure.
    """
    trial_rows = []
    topo_rows = []
    topo_info = None

    for modality in modalities:
        for subID in subjects:
            if (modality, subID) in FAULTY_SESSIONS:
                print(f"[skip] sub-0{subID} ses-{modality}: faulty recording setup")
                continue

            sub_dir = os.path.join(
                DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", f"ses-{modality}"
            )
            base_name = (
                f"sub-0{subID}_ses-{modality}_task-NT_"
                f"lfreqori-{L_FREQ_ORIG}_lfreq-{L_FREQ}_hfreq-{H_FREQ}"
            )
            broadband_fif = os.path.join(sub_dir, f"{base_name}_epochs{FILE_SUFFIX_EPOCHS}.fif")

            epochs_alpha, meta_alpha = ah.load_band_epochs_and_metadata(
                sub_dir, base_name, f"{BAND}", subID, modality,
                FILE_SUFFIX_EPOCHS, FILE_SUFFIX_META,
            )
            if epochs_alpha is None:
                continue
            if not os.path.exists(broadband_fif):
                print(f"Broadband epochs not found: {broadband_fif}")
                continue

            # exclued_subs.json (post-cue O1/O2 SNR) is TAGGED, not skipped: the tag travels
            # into the CSV so a sensitivity run is a one-line filter, but the figure plots the
            # primary set only.
            snr_excluded = bool(is_subject_excluded(subID, modality))

            session_label = f"sub-0{subID} ses-{modality}"
            epochs_broadband = load_broadband_epochs(broadband_fif)
            if epochs_broadband is None:
                print(f"[skip] {session_label}: no valid broadband trials")
                continue

            meta_alpha = meta_alpha.reset_index(drop=True)
            meta_broadband = epochs_broadband.metadata.reset_index(drop=True)
            check_epoch_index_match(
                epochs_alpha.metadata.reset_index(drop=True), meta_alpha,
                "alpha epochs", "alpha metadata TSV", subID, modality,
            )
            check_epoch_index_match(
                meta_alpha, meta_broadband, "alpha epochs", "broadband epochs", subID, modality,
            )

            response = meta_alpha["response_decoded"].to_numpy()
            yes_mask, no_mask = response == "Yes", response == "No"
            if yes_mask.sum() == 0 or no_mask.sum() == 0:
                print(f"[skip] {session_label}: needs both Yes and No trials")
                continue

            if P50_CHANNEL not in epochs_alpha.ch_names:
                raise ValueError(
                    f"{session_label}: channel {P50_CHANNEL} absent from the alpha epochs."
                )

            print(f"Processing {session_label} "
                  f"({yes_mask.sum()} Yes / {no_mask.sum()} No trials)")

            # --- alpha path: CSD on the full montage, then pick, then ONE Hilbert ---
            # The surface Laplacian must run before the channel is picked (it needs the whole
            # montage) and it changes the alpha units to uV/cm^2. The full-montage CSD alpha is
            # kept for panel (c).
            alpha_picks = [P50_CHANNEL]
            epochs_alpha_csd = ah.apply_surface_laplacian(epochs_alpha)
            epochs_alpha_pick = epochs_alpha_csd.copy().pick(alpha_picks)
            if list(epochs_alpha_pick.ch_names) != alpha_picks:
                raise ValueError(
                    f"{session_label}: alpha pick returned {epochs_alpha_pick.ch_names}, "
                    f"expected {alpha_picks} in that order."
                )
            alpha_times = epochs_alpha_pick.times.copy()
            analytic = hilbert_analytic(
                epochs_alpha_pick.get_data(copy=True),
                epochs_alpha_pick.info["sfreq"], ALPHA_FMIN,
            )  # complex, (n_trials, n_channels, n_times)
            envelope = np.abs(analytic)
            alpha_phase = np.angle(analytic)
            i_env = alpha_picks.index(P50_CHANNEL)

            # Pre-stimulus alpha AMPLITUDE, panel (a)'s binning variable.
            prestim_sl = window_slice(alpha_times, PRESTIM_WIN[modality], "pre-stimulus")
            trial_alpha = envelope[:, i_env, prestim_sl].mean(axis=1)

            # Alpha PHASE as a scalar at PHASE_READOUT_TIME, panel (b)'s binning variable.
            # The outermost Hilbert samples are degenerate (reflect padding makes the signal
            # even-symmetric about the last real sample, so the imaginary part vanishes and the
            # phase collapses to exactly 0 or 180 deg), hence the edge guard.
            phase_idx = int(np.argmin(np.abs(alpha_times - PHASE_READOUT_TIME)))
            if phase_idx < N_EDGE_SAMPLES or phase_idx > len(alpha_times) - 1 - N_EDGE_SAMPLES:
                raise ValueError(
                    f"{session_label}: phase readout at {PHASE_READOUT_TIME}s resolves to "
                    f"sample {phase_idx} of {len(alpha_times)}, within {N_EDGE_SAMPLES} "
                    f"samples of the degenerate Hilbert edge."
                )
            trial_phase = alpha_phase[:, i_env, phase_idx]

            # --- broadband path: P50 as the window mean over P50_WIN, no CSD ---
            epochs_erp = epochs_broadband.copy().pick([P50_CHANNEL])
            apply_baseline_epochs(
                epochs_erp, broadband_fif, baseline_window=BASELINE_BY_MOD[modality]
            )
            erp_times = epochs_erp.times.copy()
            erp_data = epochs_erp.get_data(copy=True)[:, 0, :]   # (n_trials, n_times)
            assert_microvolt_scale(erp_data, f"{session_label} broadband {P50_CHANNEL}")

            # Per-trial mean over the modality's whole P50 window. No peak search.
            # window_slice raises if the window is not inside the epoch.
            p50_sl = window_slice(erp_times, P50_WIN[modality], "P50 mean window")
            trial_p50 = erp_data[:, p50_sl].mean(axis=1)

            # --- panel (c): alpha-envelope and P50 maps over the full montage, both CSD ---
            # All clean trials pooled (perceived and unperceived).
            i_alpha_topo = sample_index(alpha_times, TOPO_ALPHA_TIME,
                                        f"{session_label} alpha topography")
            if (i_alpha_topo < N_EDGE_SAMPLES
                    or i_alpha_topo > len(alpha_times) - 1 - N_EDGE_SAMPLES):
                raise ValueError(
                    f"{session_label}: alpha topography time {TOPO_ALPHA_TIME}s is within "
                    f"{N_EDGE_SAMPLES} samples of the degenerate Hilbert edge."
                )
            env_full = np.abs(hilbert_analytic(
                epochs_alpha_csd.get_data(copy=True), epochs_alpha_csd.info["sfreq"], ALPHA_FMIN,
            ))                                                    # (n_trials, n_eeg, n_times)
            alpha_map = env_full[:, :, i_alpha_topo].mean(axis=0)
            alpha_map_yes = env_full[yes_mask, :, i_alpha_topo].mean(axis=0)
            alpha_map_no = env_full[no_mask, :, i_alpha_topo].mean(axis=0)

            # Baseline on the recorded signal, then the same CSD as the alpha. The CSD is a
            # linear spatial transform, so its order relative to the baseline does not matter.
            epochs_p50_topo = epochs_broadband.copy().pick("eeg")
            apply_baseline_epochs(
                epochs_p50_topo, broadband_fif, baseline_window=BASELINE_BY_MOD[modality]
            )
            assert_microvolt_scale(epochs_p50_topo.get_data(copy=True),
                                   f"{session_label} broadband EEG")
            epochs_p50_topo = ah.apply_surface_laplacian(epochs_p50_topo)
            i_p50_topo = sample_index(epochs_p50_topo.times, TOPO_P50_TIME[modality],
                                      f"{session_label} P50 topography")
            p50_topo_samples = epochs_p50_topo.get_data(copy=True)[:, :, i_p50_topo]
            p50_map = p50_topo_samples.mean(axis=0)
            p50_map_yes = p50_topo_samples[yes_mask].mean(axis=0)
            p50_map_no = p50_topo_samples[no_mask].mean(axis=0)

            if list(epochs_alpha_csd.ch_names) != list(epochs_p50_topo.ch_names):
                raise ValueError(
                    f"{session_label}: alpha and broadband EEG channel lists differ; the maps "
                    f"are compared electrode by electrode."
                )
            if topo_info is None:
                topo_info = epochs_p50_topo.info.copy()
            elif list(epochs_p50_topo.ch_names) != list(topo_info.ch_names):
                raise ValueError(
                    f"{session_label}: EEG channel list differs from the other sessions; the "
                    f"topographies need one common montage."
                )
            topo_rows.append({
                "sub": int(subID), "modality": modality, "snr_excluded": snr_excluded,
                "alpha_map": alpha_map, "p50_map": p50_map,
                "alpha_map_yes": alpha_map_yes, "alpha_map_no": alpha_map_no,
                "p50_map_yes": p50_map_yes, "p50_map_no": p50_map_no,
            })

            # --- accumulate ---
            session_df = pd.DataFrame({
                "sub": int(subID),
                "modality": modality,
                "response": response,
                "alpha_env": trial_alpha,
                "alpha_phase": trial_phase,
                "p50": trial_p50,
                # Stimulus intensity of the same trial. meta_alpha was just asserted
                # trial-identical to the broadband epochs, so this lines up with `p50` by
                # position. Feeds the separate intensity figure.
                "intensity_level": meta_alpha["intensity_level"].to_numpy(),
                "snr_excluded": snr_excluded,
            })
            # Equiprobable bins WITHIN this session, so between-subject differences in overall
            # alpha power cannot drive the binning.
            session_df["alpha_bin"] = pd.qcut(
                session_df["alpha_env"], q=n_bins, labels=False, duplicates="drop"
            )
            trial_rows.append(session_df)

    if not trial_rows:
        raise RuntimeError("No usable sessions found - nothing to plot.")
    return (
        pd.concat(trial_rows, ignore_index=True),
        {"rows": topo_rows, "info": topo_info},
    )


def load_or_collect(args):
    """
    Collect the session tables, optionally via the opt-in pickle cache.

    The cache stores the subject/modality/bin settings it was built with and refuses to serve a
    file built for a different selection - a stale cache silently plotting the wrong subject set
    is exactly the kind of quiet wrongness this project forbids.
    """
    key = {
        "sub": sorted(args.sub),
        "modalities": list(args.modalities),
        "n_bins": args.n_bins,
        "suffix": FILE_SUFFIX_EPOCHS,
        "p50_channel": P50_CHANNEL,
        "topo_alpha_time": TOPO_ALPHA_TIME,
        "topo_p50_time": dict(TOPO_P50_TIME),
        # The P50 amplitude depends on the averaging window.
        "p50_win": {m: list(w) for m, w in P50_WIN.items()},
        # Keeps a step021 (peak-readout) pickle from being loaded here.
        "p50_measure": "window_mean",
        # Bumped whenever df_trials gains or loses a column, so a pickle written by an older
        # version of this script is rejected here instead of failing deep inside a plot.
        # Schema 5: topo rows also carry the Yes and No maps.
        "trial_schema": 5,
    }
    if args.cache_file and os.path.exists(args.cache_file):
        with open(args.cache_file, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") != key:
            raise ValueError(
                f"Cache {args.cache_file} was built for a different selection\n"
                f"  cached:    {cached.get('key')}\n"
                f"  requested: {key}\n"
                f"Delete the file or pass a different --cache-file."
            )
        print(f"[cache] loaded session tables from {args.cache_file}")
        return cached["df_trials"], cached["topo"]

    df_trials, topo = collect_sessions(args.sub, args.modalities, args.n_bins)
    if args.cache_file:
        with open(args.cache_file, "wb") as fh:
            pickle.dump({"key": key, "df_trials": df_trials, "topo": topo}, fh)
        print(f"[cache] wrote session tables to {args.cache_file}")
    return df_trials, topo


# %%
# ---------------------------------------------------------------- aggregation

RESPONSE_SETS = [(["Yes", "No"], "all"), (["Yes"], "Yes"), (["No"], "No")]


def aggregate_alpha_bins(df_trials, modality, n_bins):
    """
    Panel (a): subject x bin matrices of mean P50, one per response set.

    Aggregation is SUBJECT-FIRST - each session's own trials are averaged within bin, and only
    then are the per-session bin values averaged across sessions - so every session counts once
    regardless of its trial count. NaN marks a bin a subject has no trials in; pd.qcut with
    duplicates="drop" can leave fewer than n_bins bins in a session with tied envelope values.

    Returns {response_set_label: (matrix, subject_ids)}.
    """
    df_mod = df_trials[df_trials["modality"] == modality]
    out = {}
    for responses, label in RESPONSE_SETS:
        sel = df_mod[df_mod["response"].isin(responses)]
        grouped = sel.groupby(["sub", "alpha_bin"])["p50"].mean()
        subs_here = sorted(sel["sub"].unique())
        mat = np.full((len(subs_here), n_bins), np.nan)
        for row_idx, subID in enumerate(subs_here):
            sub_sel = grouped.loc[subID]
            mat[row_idx, sub_sel.index.to_numpy().astype(int)] = sub_sel.to_numpy()
        out[label] = (mat, subs_here)
    return out


def aggregate_intensity_p50(df_trials):
    """
    Mean P50 per SESSION x stimulus intensity level - the input the intensity figure is drawn
    from.

    Subject-first, like every other aggregation in this script: a session's trials are averaged
    within intensity level first, so a session that happened to collect more trials at one level
    does not weight the grand mean more heavily. Handing the resulting one-row-per-session frame
    to seaborn is what makes its error bar an across-SESSION SEM rather than an across-trial one.

    Trials are pooled over Yes/No on purpose: the question here is whether the P50 tracks the
    stimulus, not whether it tracks the report.

    Only levels 2-6 survive - `analysis_excluded` upstream already drops the catch trials
    (level 0) and the two outermost levels (1 and 7).
    """
    return (
        df_trials
        .groupby(["modality", "sub", "intensity_level"], as_index=False)["p50"]
        .mean()
    )


def aggregate_phase_bins(df_trials, modality, n_phase_bins):
    """
    Panel (b): grand-average P50 per fixed-width alpha-phase bin.

    Phase is circular, so the bins are equal-WIDTH over the full -pi..pi circle rather than the
    equiprobable pd.qcut bins used for the amplitude in panel (a): a bin index must mean the
    same angular range in every session for the bins to be combinable at all.

    Aggregation is subject-first, exactly as in panel (a): each session's trials are binned and
    averaged within session (circular mean for the phase, arithmetic mean for the P50), and only
    then are the per-session bin values combined - circular mean of the session phases,
    arithmetic mean of the session P50s, SEM across sessions.

    Returns a DataFrame with one row per bin.
    """
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    df_mod = df_trials[df_trials["modality"] == modality]

    session_rows = []
    for subID, sub_df in df_mod.groupby("sub"):
        phases = sub_df["alpha_phase"].to_numpy()
        p50s = sub_df["p50"].to_numpy()
        # edges[1:-1] drops the outer edges so every trial lands in 0..n_phase_bins-1.
        bin_idx = np.digitize(phases, edges[1:-1])
        for b in range(n_phase_bins):
            sel = bin_idx == b
            if not sel.any():
                continue
            session_rows.append({
                "sub": int(subID), "bin": b,
                "bin_phase_mean": float(ah.circular_mean(phases[sel])),
                "p50_mean": float(p50s[sel].mean()),
                "n_trials": int(sel.sum()),
            })
    df_sess = pd.DataFrame(session_rows)

    grand_rows = []
    for b, grp in df_sess.groupby("bin"):
        vals = grp["p50_mean"].to_numpy()
        grand_rows.append({
            "bin": int(b),
            "bin_phase_grand": float(ah.circular_mean(grp["bin_phase_mean"].to_numpy())),
            "p50_grand": float(vals.mean()),
            "p50_sem": float(stats.sem(vals)) if vals.size > 1 else np.nan,
            "n_sessions": int(grp["sub"].nunique()),
            "n_trials_total": int(grp["n_trials"].sum()),
        })
    return pd.DataFrame(grand_rows).sort_values("bin").reset_index(drop=True)


def bin_mean_sem(mat):
    """Across-subject mean and SEM per bin, ignoring the NaN cells."""
    mean_curve = np.nanmean(mat, axis=0)
    sem_curve = np.array([
        stats.sem(mat[np.isfinite(mat[:, b]), b]) if np.isfinite(mat[:, b]).sum() > 1
        else np.nan
        for b in range(mat.shape[1])
    ])
    return mean_curve, sem_curve


# %%
# ---------------------------------------------------------------- panels


def plot_panel_a(ax, curves, x_bins, stat_row, show_ylabel, show_legend):
    """
    P50 across pre-stimulus alpha-amplitude bins: 'all' plus the Yes/No split.

    The grey line is a straight fit to the plotted 'all' MEAN curve and is cosmetic only; the
    statistic in the title is the subject-level slope test, which is a different (and correct)
    quantity - it fits one line per subject and then tests those slopes against zero.
    """
    for label, style in [("all", "-"), ("Yes", "--"), ("No", ":")]:
        mat = curves[label][0]
        if mat.shape[0] == 0:
            continue
        mean_curve, sem_curve = bin_mean_sem(mat)
        color = "black" if label == "all" else PALETTE_RESP[label]
        ax.errorbar(x_bins, mean_curve, yerr=sem_curve, marker="o", markersize=3.5,
                    linestyle=style, color=color, capsize=2.5, linewidth=1.2,
                    label={"all": "all", "Yes": "perceived", "No": "unperceived"}[label])

    mean_all, _ = bin_mean_sem(curves["all"][0])
    trend = np.polyfit(x_bins, mean_all, 1)
    ax.plot(x_bins, np.polyval(trend, x_bins), color="grey", linestyle="-", linewidth=0.8,
            zorder=0)

    # p is UNCORRECTED: with only CP4 plotted there is no channel family left to FDR over.
    ax.set_title(
        f"slope = {stat_row['slope_mean']:.3f}, t = {stat_row['t']:.2f}, "
        f"p {format_p(stat_row['p'])} {sig_stars(stat_row['p'])}",
        fontsize=FS_PANEL_STAT,
    )
    ax.set_xticks(x_bins)
    ax.set_xlabel("Alpha amplitude bin (low $\\rightarrow$ high)")
    if show_ylabel:
        ax.set_ylabel(f"P50 at {P50_CHANNEL}, window mean (µV)")
    if show_legend:
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left")


def plot_panel_b(ax, df_grand, n_phase_bins, r_floor, r_top, norm, cmap):
    """
    P50 by alpha-phase bin, as polar bars coloured by the P50 itself.

    Negative bin means are the reason for r_floor. On a polar axis a negative radius plots
    THROUGH the origin at the opposite angle, which would silently move a bin to the wrong side
    of the circle. Anchoring every bar at a common floor below the global minimum keeps the
    geometry honest; the dashed circle marks where the P50 changes sign, and the colour (a
    diverging map centred on zero) carries the sign independently of the radius.

    r_floor / r_top / norm are shared across modalities so the two panels are comparable.
    """
    width = 2 * np.pi / n_phase_bins
    theta = df_grand["bin_phase_grand"].to_numpy()
    radius = df_grand["p50_grand"].to_numpy()
    sem = df_grand["p50_sem"].to_numpy()

    ax.bar(theta, radius - r_floor, width=width, bottom=r_floor,
           color=cmap(norm(radius)), edgecolor="white", linewidth=0.4, zorder=1)
    # SEM across sessions at the bar tip. fmt="none" draws the whisker without a marker.
    ax.errorbar(theta, radius, yerr=sem, fmt="none", ecolor="0.25", elinewidth=0.7,
                capsize=0, zorder=2)

    # Zero circle: the sign change. Drawn above the bars, because on a long bar the sign
    # boundary is exactly the thing a reader has to find.
    circle = np.linspace(0, 2 * np.pi, 361)
    ax.plot(circle, np.zeros_like(circle), color="0.35", linestyle="--", linewidth=0.7,
            zorder=4)

    # A small hole at the centre keeps the bars from converging to a point.
    ax.set_rorigin(r_floor - 0.20 * (r_top - r_floor))
    ax.set_ylim(r_floor, r_top)
    # Force 0 into the radial ticks so the dashed sign-change circle is labelled; without it
    # matplotlib's locator can skip straight past the one radius the reader needs. MaxNLocator
    # keeps the remaining ticks on round numbers.
    rticks = [0.0] + [t for t in MaxNLocator(3).tick_values(0.0, r_top) if t > 0]
    ax.set_yticks([t for t in sorted(set(rticks)) if r_floor < t <= r_top])
    # Panel (a) states its quantity in its axis labels; the polar axis has no x label to do
    # that, so it is named in a title instead. The readout time comes from the constant so the
    # title cannot drift away from what was measured.
    ax.set_title(
        f"P50 amplitude by\nalpha phase at {PHASE_READOUT_TIME * 1000:.0f} ms",
        fontsize=FS_PANEL_TITLE, pad=14,
    )
    ax.set_rlabel_position(112.5)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"])
    ax.tick_params(labelsize=FS_POLAR_TICK, pad=0)
    ax.grid(color="0.85", linewidth=0.5)
    # The radial labels sit over the bars. set_axisbelow(False) is what actually lifts the
    # tick/grid layer above the bar patches - a per-Text zorder does not cross that boundary -
    # and the white backing box keeps them legible where a bar runs underneath.
    ax.set_axisbelow(False)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))


def plot_panel_c(ax, values, info, cmap, vlim, highlight_mask):
    """One grand-average topography of panel (c); returns the image for the shared colourbar."""
    image, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=6,
        sensors=True, outlines="head", mask=highlight_mask,
        mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="black",
                         markeredgewidth=1.4, markersize=7, linestyle="none"),
    )
    return image


def topo_statistics(topo_rows, modalities):
    """
    Grand-average panel (c) maps and the two TANOVAs per modality.

    Returns (maps, tanova_df): maps[modality] holds the grand-average pooled alpha and P50 maps,
    the Yes - No difference maps of both, and the session count; tanova_df has one row per
    modality and test for the CSV.
    """
    rng = np.random.default_rng(SEED)
    maps, rows = {}, []
    for modality in modalities:
        sel = [r for r in topo_rows if r["modality"] == modality]
        if len(sel) < 2:
            raise ValueError(f"Panel (c): only {len(sel)} {modality} sessions.")
        stack = lambda key: np.stack([r[key] for r in sel])          # (n_sessions, n_eeg)
        alpha_maps, p50_maps = stack("alpha_map"), stack("p50_map")
        # Perceived minus unperceived, per session.
        alpha_dmaps = stack("alpha_map_yes") - stack("alpha_map_no")
        p50_dmaps = stack("p50_map_yes") - stack("p50_map_no")
        maps[modality] = {
            "alpha": alpha_maps.mean(axis=0), "p50": p50_maps.mean(axis=0),
            "alpha_diff": alpha_dmaps.mean(axis=0), "p50_diff": p50_dmaps.mean(axis=0),
            "n": len(sel),
        }

        tests = [
            ("pooled alpha vs P50", alpha_maps, p50_maps),
            ("Yes-No difference maps: alpha vs P50", alpha_dmaps, p50_dmaps),
        ]
        for label, a_maps, b_maps in tests:
            # Spatial correlation per session: the dot product of the two normalized maps.
            z_spatial = np.arctanh((normalize_topo(a_maps) * normalize_topo(b_maps)).sum(axis=1))
            t_r, p_r = stats.ttest_1samp(z_spatial, 0.0)
            tanova = tanova_paired(gfp_normalize_topo(a_maps), gfp_normalize_topo(b_maps),
                                   TANOVA_N_PERMUTATIONS, rng)
            rows.append({
                "modality": modality, "effect": label, "n_sessions": len(sel),
                "alpha_time_s": TOPO_ALPHA_TIME, "p50_time_s": TOPO_P50_TIME[modality],
                "observed_diss": tanova["observed"], "null_mean": float(tanova["null"].mean()),
                "null_p95": tanova["null_p95"], "p": tanova["p"],
                "n_permutations": TANOVA_N_PERMUTATIONS, "seed": SEED,
                "alpha_level": TANOVA_ALPHA, "significant": bool(tanova["p"] < TANOVA_ALPHA),
                "spatial_r_mean": float(np.tanh(z_spatial.mean())),
                "spatial_r_t": float(t_r), "spatial_r_p": float(p_r),
            })
    return maps, pd.DataFrame(rows)


def plot_panel_d(ax, ax_legend, anchors):
    """
    Interpretation aid for panel (b): what a given phase at PHASE_READOUT_TIME means for the
    phase at stimulus onset and at the P50.

    One COSINE_FREQ_HZ cosine per modality whose phase at the readout time is anchors[modality]
    (the average phase of that modality's highest-P50 bin in panel b), drawn in the modality
    colour. Cosine convention, as np.angle(hilbert(x)) returns it: 0 deg = peak, +-180 deg =
    trough, because the real part of the analytic signal is A*cos(phase).

    The phase is pinned at the readout time, not at 0 ms. At 10 Hz the two are the same wave
    (one period is 100 ms), but only the readout anchor stays right if COSINE_FREQ_HZ changes.
    """
    period_ms = 1000.0 / COSINE_FREQ_HZ
    t_readout_ms = PHASE_READOUT_TIME * 1000.0

    def phase_at(t_ms, phase_at_readout_deg):
        # Unwrapped phase, linear in time, with phase(t_readout_ms) = phase_at_readout_deg.
        return phase_at_readout_deg + 360.0 * (np.asarray(t_ms, dtype=float) - t_readout_ms) / period_ms

    # (phase at readout, colour, line style, legend label)
    curves = [
        (phase_deg, PALETTE_MODALITY[modality], "-",
         f"{MODALITY_LABELS[modality]}: phase of\nhighest-P50 bin ({phase_deg:.0f}°)")
        for modality, phase_deg in anchors.items()
    ]
    t = np.linspace(COSINE_WIN_MS[0], COSINE_WIN_MS[1], 2000)
    ax.axhline(0.0, color="0.85", linewidth=0.7, zorder=0)
    handles = []
    for phase0, color, style, label in curves:
        (line,) = ax.plot(t, np.cos(np.deg2rad(phase_at(t, phase0))), color=color,
                          linestyle=style, linewidth=1.8, zorder=3, label=label)
        handles.append(line)

    # The readout time is the one panel (b) bins on, so it alone is shaded.
    ax.axvspan(t_readout_ms - COSINE_HIGHLIGHT_WIDTH_MS / 2,
               t_readout_ms + COSINE_HIGHLIGHT_WIDTH_MS / 2,
               color="0.6", alpha=0.25, linewidth=0, zorder=1)

    # Label stacking below needs data units in points: axes height in points / y range.
    ylim = (-1.3, 1.3)
    fig_h_pts = ax.figure.get_figheight() * 72.0
    pts_per_unit = ax.get_position().height * fig_h_pts / (ylim[1] - ylim[0])
    label_sep_pts = 1.15 * FS_COSINE_LABEL

    markers = [(t_readout_ms, "alpha phase readout", "-")]
    markers += [(float(t_ms), name, ":") for t_ms, name in COSINE_MARKERS_MS]
    # Event names sit just above the axes: x in data (ms), y in axes coordinates.
    trans_top = blended_transform_factory(ax.transData, ax.transAxes)
    for t_ms, name, style in markers:
        ax.axvline(t_ms, color="0.3", linestyle=style, linewidth=1.0, zorder=2)
        ax.text(t_ms, 1.03, f"{name} ({t_ms:.0f} ms)", transform=trans_top, ha="center",
                va="bottom", fontsize=FS_COSINE_LABEL, color="0.2")

        if t_ms == COSINE_P50_MS:
            # At the P50: mark each wave's PEAK (phase 0 deg) closest to the P50 time, labelled
            # with its time just below the peak. The peaks fall within a few ms of each other,
            # so the labels step down one line per peak (in time order) to stay apart.
            peaks = []
            for phase0, color, _, _ in curves:
                # phase_at(t) = 0 (mod 360)  <=>  t = t_readout + (360 k - phase0) / 360 * period
                k = np.round((phase0 + 360.0 * (t_ms - t_readout_ms) / period_ms) / 360.0)
                t_peak = t_readout_ms + (360.0 * k - phase0) / 360.0 * period_ms
                peaks.append((float(t_peak), color))
            peaks.sort()
            for rank, (t_peak, color) in enumerate(peaks):
                ax.plot(t_peak, 1.0, "o", color=color, markersize=5, zorder=4)
                ax.annotate(
                    f"{t_peak:.0f} ms", xy=(t_peak, 1.0),
                    xytext=(0, -(rank + 1) * label_sep_pts), textcoords="offset points",
                    ha="center", va="center", fontsize=FS_COSINE_LABEL, color=color,
                    fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
                )
            continue

        # One dot per wave where it crosses the marker, labelled to its right with that wave's
        # phase wrapped to [0, 360). The labels are stacked bottom-up with at least one label
        # height between neighbours, so dots that sit close together still get readable labels.
        ys = [float(np.cos(np.deg2rad(phase_at(t_ms, c[0])))) for c in curves]
        order = np.argsort(ys)
        dot_pts = [ys[i] * pts_per_unit for i in order]
        label_pts = []
        for y_pt in dot_pts:
            label_pts.append(y_pt if not label_pts else max(y_pt, label_pts[-1] + label_sep_pts))
        # Centre the stack on the dots it labels, so it does not drift upwards only, then keep it
        # inside the y range so no label runs into the tick labels or the event names above.
        shift = (np.mean(dot_pts) - np.mean(label_pts))
        lo_pt = ylim[0] * pts_per_unit + 0.6 * label_sep_pts
        hi_pt = ylim[1] * pts_per_unit - 0.6 * label_sep_pts
        shift = max(shift, lo_pt - label_pts[0])
        shift = min(shift, hi_pt - label_pts[-1])
        for idx, y_pt, lab_pt in zip(order, dot_pts, label_pts):
            phase0, color, _, _ = curves[idx]
            ax.plot(t_ms, ys[idx], "o", color=color, markersize=5, zorder=4)
            ax.annotate(
                f"{float(phase_at(t_ms, phase0)) % 360.0:.0f}°", xy=(t_ms, ys[idx]),
                xytext=(7, lab_pt + shift - y_pt), textcoords="offset points", ha="left",
                va="center", fontsize=FS_COSINE_LABEL, color=color,
                fontweight="bold", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
            )

    ax.set_xlim(*COSINE_WIN_MS)
    ax.set_ylim(*ylim)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks(np.arange(COSINE_WIN_MS[0], COSINE_WIN_MS[1] + 1, 25))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # The legend has its own narrow axes so it neither covers the waves nor widens the figure.
    ax_legend.axis("off")
    ax_legend.legend(handles=handles, loc="center left", frameon=False, fontsize=FS_LEGEND,
                     title=f"{COSINE_FREQ_HZ} Hz cosine", title_fontsize=FS_LEGEND)


# %%
# ---------------------------------------------------------------- figure assembly


def build_figure(df_trials, topo, modalities, n_bins, n_phase_bins):
    """
    Assemble the whole figure and return (fig, stats_table, tanova_table).

    Rows are the modality (electrical on top); columns are a | b | c, where c is a 2 x 2 block
    (pooled alpha, P50 / perceived - unperceived alpha, P50), with panel (d) as a full-width
    strip underneath.
    """
    x_bins = np.arange(1, n_bins + 1, dtype=float)
    stat_rows = []

    # --- precompute everything, so the shared scales can be set before anything is drawn ---
    alpha_curves = {}
    phase_grand = {}
    panel_a_stats = {}
    for modality in modalities:
        alpha_curves[modality] = aggregate_alpha_bins(df_trials, modality, n_bins)
        phase_grand[modality] = aggregate_phase_bins(df_trials, modality, n_phase_bins)

        for _, label in RESPONSE_SETS:
            mat = alpha_curves[modality][label][0]
            slopes = fit_linear_slopes(x_bins, mat)
            t_stat, p_val, n_subs = ttest_1samp_finite(slopes)
            row = {
                "panel": "a", "modality": modality, "channel": P50_CHANNEL,
                "response_set": label, "n": n_subs, "n_bins": n_bins,
                "slope_mean": float(np.mean(slopes)) if slopes.size else np.nan,
                "slope_sem": float(stats.sem(slopes)) if slopes.size > 1 else np.nan,
                "t": t_stat, "p": p_val,
            }
            panel_a_stats[(modality, label)] = row
            stat_rows.append(row)

    # Panel (d)'s anchors: per modality, the average phase of the panel (b) bin with the highest
    # P50 - i.e. the tallest bar in panel (b).
    anchors = {}
    for modality in modalities:
        pg = phase_grand[modality]
        peak = pg.loc[pg["p50_grand"].idxmax()]
        anchors[modality] = float(np.degrees(peak["bin_phase_grand"]) % 360.0)
        print(f"[panel d] {modality}: highest-P50 bin {int(peak['bin'])}, phase "
              f"{anchors[modality]:.1f} deg, P50 {peak['p50_grand']:.3f} uV, "
              f"n = {int(peak['n_sessions'])} sessions")
        stat_rows.append({
            "panel": "d", "modality": modality, "channel": P50_CHANNEL, "response_set": "all",
            "n": int(peak["n_sessions"]), "peak_bin": int(peak["bin"]),
            "peak_phase_deg": anchors[modality], "peak_p50": float(peak["p50_grand"]),
        })

    df_stats = pd.DataFrame(stat_rows)

    # Panel (c): grand-average maps and the TANOVAs (CSV only).
    topo_maps, df_tanova = topo_statistics(topo["rows"], modalities)
    topo_info = topo["info"]
    highlight_mask = np.isin(topo_info.ch_names, TOPO_HIGHLIGHT)
    if highlight_mask.sum() != len(TOPO_HIGHLIGHT):
        raise ValueError(f"Panel (c): {TOPO_HIGHLIGHT} not all in the montage.")
    # One scale per map type, shared across modalities so the blocks are comparable. The pooled
    # alpha envelope is positive, so it gets a sequential map from 0; the rest are signed.
    sym = lambda key: max(np.abs(topo_maps[m][key]).max() for m in modalities)
    alpha_max = max(topo_maps[m]["alpha"].max() for m in modalities)
    # (key, cmap, vlim, title, colourbar label); topo_grid[r][c] is the map at block row r, col c.
    topo_grid = [
        [("alpha", "Reds", (0.0, alpha_max), "Alpha amplitude", "µV/cm²"),
         ("p50", "RdBu_r", (-sym("p50"), sym("p50")), "P50", "µV/cm²")],
        [("alpha_diff", "RdBu_r", (-sym("alpha_diff"), sym("alpha_diff")),
          "Alpha, Yes - No", "µV/cm²"),
         ("p50_diff", "RdBu_r", (-sym("p50_diff"), sym("p50_diff")),
          "P50, Yes - No", "µV/cm²")],
    ]

    # Panel (b) shares one radial range and one colour normalisation across modalities, so a
    # bar height means the same thing in both rows.
    all_grand = np.concatenate([phase_grand[m]["p50_grand"].to_numpy() for m in modalities])
    all_sem = np.concatenate([phase_grand[m]["p50_sem"].to_numpy() for m in modalities])
    g_min, g_max = float(np.nanmin(all_grand)), float(np.nanmax(all_grand + all_sem))
    span = g_max - g_min
    r_floor = g_min - 0.15 * span
    r_top = g_max + 0.05 * span
    cmap = plt.get_cmap(PHASE_BAR_CMAP)
    if g_min < 0.0 < g_max:
        norm = TwoSlopeNorm(vmin=g_min, vcenter=0.0, vmax=g_max)
    else:
        # The bin means did not cross zero in this run, so a zero-centred diverging map would
        # use only half its range. Fall back to a plain linear norm and say so.
        print(f"[panel b] bin means do not cross zero ({g_min:.3f}..{g_max:.3f}); "
              f"using a linear colour norm instead of a zero-centred one.")
        norm = Normalize(vmin=g_min, vmax=g_max)

    # --- draw ---
    fig = plt.figure(figsize=FIGSIZE)
    # Panel (d) is a short full-width strip under the modality rows. It is a separate outer row,
    # so the a/b/c grid nested above it keeps its own spacing.
    gs_main = GridSpec(
        len(modalities) + 1, 1, figure=fig,
        height_ratios=[1.0] * len(modalities) + [D_ROW_RATIO], hspace=D_ROW_HSPACE,
    )
    # Two column GROUPS, (a)+(b) and the three (c) topographies. Both the polar axis and the
    # topomaps are equal-aspect and height-limited, so they do not fill their cells; the ratios
    # and wspaces below were tuned (by measuring the drawn extents) so the visible gaps a-b and
    # b-c come out about equal.
    gs_outer = gs_main[:len(modalities), 0].subgridspec(
        len(modalities), 2, width_ratios=[2.0, 1.40], hspace=0.40, wspace=0.10,
    )

    axes_a, axes_b = [], []
    for row_idx, modality in enumerate(modalities):
        n_sessions = df_trials[df_trials["modality"] == modality]["sub"].nunique()

        # The polar axis is square and height-limited, so its cell is narrower than panel (a)'s:
        # any extra cell width would only become dead space around the circle.
        gs_ab = gs_outer[row_idx, 0].subgridspec(1, 2, width_ratios=[1.10, 0.90], wspace=0.27)
        # The topographies are equal-aspect and height-limited, so their cells are wider than
        # the heads; the colourbars are inset axes placed against each head, not grid columns.
        gs_c = gs_outer[row_idx, 1].subgridspec(2, 2, wspace=0.34, hspace=0.50)

        # --- a ---
        ax_a = fig.add_subplot(gs_ab[0, 0])
        plot_panel_a(
            ax_a, alpha_curves[modality], x_bins,
            panel_a_stats[(modality, "all")],
            show_ylabel=True, show_legend=(row_idx == 0),
        )
        axes_a.append(ax_a)
        # Far enough left to clear panel (a)'s own y label, which sits at roughly -0.25 in
        # axes coordinates once the tick labels are at this font size.
        ax_a.text(-0.44, 0.5, f"{MODALITY_LABELS[modality]}\n(n = {n_sessions})",
                  transform=ax_a.transAxes, rotation=90, va="center", ha="center",
                  fontsize=FS_ROW_LABEL, fontweight="bold")

        # --- b ---
        ax_b = fig.add_subplot(gs_ab[0, 1], projection="polar")
        plot_panel_b(ax_b, phase_grand[modality], n_phase_bins, r_floor, r_top, norm, cmap)
        axes_b.append(ax_b)

        # --- c, 2 x 2 topographies: pooled alpha, P50 (top); perceived - unperceived (bottom) ---
        # Each map's title carries its readout time; the scale is shared across modalities.
        readout = {"alpha": TOPO_ALPHA_TIME, "alpha_diff": TOPO_ALPHA_TIME,
                   "p50": TOPO_P50_TIME[modality], "p50_diff": TOPO_P50_TIME[modality]}
        row_c = []
        for r_idx, grid_row in enumerate(topo_grid):
            for c_idx, (key, cmap_c, vlim_c, title, cbar_label) in enumerate(grid_row):
                ax_c = fig.add_subplot(gs_c[r_idx, c_idx])
                image = plot_panel_c(ax_c, topo_maps[modality][key], topo_info, cmap_c,
                                     vlim_c, highlight_mask)
                ax_c.set_title(f"{title}\n{readout[key] * 1000:.0f} ms", fontsize=FS_PANEL_TITLE)
                # Every map has its own vertical colourbar directly to its right.
                cax = ax_c.inset_axes([1.04, 0.12, 0.055, 0.76])
                cbar_c = fig.colorbar(image, cax=cax, orientation="vertical")
                cbar_c.set_label(cbar_label, fontsize=FS_CBAR_LABEL - 2)
                cbar_c.ax.yaxis.set_major_locator(MaxNLocator(3))
                cbar_c.ax.tick_params(labelsize=FS_CBAR_TICK - 2)
                cbar_c.outline.set_linewidth(0.6)
                row_c.append(ax_c)

        if row_idx == 0:
            # All three letters sit on ONE line. The x of each comes from its own axes, but
            # the y comes from panel (a)'s for all of them: a polar axes shrinks its bbox to the
            # square holding the circle, so 1.16 of ax_b is a different height on the page than
            # 1.16 of ax_a and the letters would step up and down across the row.
            for ax, letter, dx in [(ax_a, "A", -0.30), (ax_b, "B", -0.14),
                                   (row_c[0], "C", -0.12)]:
                trans = blended_transform_factory(ax.transAxes, ax_a.transAxes)
                ax.text(dx, 1.16, letter, transform=trans, fontsize=FS_PANEL_LETTER,
                        fontweight="bold", va="bottom", ha="left")

    # Panel (a) is the same quantity in both rows, so it shares a y axis too.
    a_lo = min(a.get_ylim()[0] for a in axes_a)
    a_hi = max(a.get_ylim()[1] for a in axes_a)
    for a in axes_a:
        a.set_ylim(a_lo, a_hi)
        sns.despine(trim=True, ax=a)

    # One colourbar for both polar panels, which share the normalisation. Attached to those two
    # axes rather than given a GridSpec column of its own, so it stays with panel (b) and is
    # sized relative to it. Horizontal: a polar panel is as wide as it is tall, so a bar under
    # the column reads along the same axis the eye already travels.
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes_b, orientation="horizontal", shrink=0.55,
                        fraction=0.045, pad=0.04, aspect=22)
    cbar.set_label(f"P50 at {P50_CHANNEL}, window mean (µV)", fontsize=FS_CBAR_LABEL)
    # A horizontal bar is narrow enough that the default locator drops to two ticks, and
    # TwoSlopeNorm makes the mapping piecewise-linear, so the zero point has to be named
    # explicitly - it is where the bars change sign.
    cticks = sorted(set([0.0] + list(MaxNLocator(3).tick_values(g_min, g_max))))
    cbar.set_ticks([t for t in cticks if g_min <= t <= g_max])
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    cbar.outline.set_linewidth(0.6)

    # --- d, full-width illustration strip; the narrow right column holds its legend ---
    gs_d = gs_main[len(modalities), 0].subgridspec(1, 2, width_ratios=[4.6, 1.0], wspace=0.02)
    ax_d = fig.add_subplot(gs_d[0, 0])
    ax_d_legend = fig.add_subplot(gs_d[0, 1])
    plot_panel_d(ax_d, ax_d_legend, anchors)
    # Letter "d" at the same page x as letter "a" (dx = -0.30 of the top panel (a) width).
    pos_a = axes_a[0].get_position()
    trans_d = blended_transform_factory(fig.transFigure, ax_d.transAxes)
    ax_d.text(pos_a.x0 - 0.30 * pos_a.width, 1.03, "D", transform=trans_d,
              fontsize=FS_PANEL_LETTER, fontweight="bold", va="bottom", ha="left")

    return fig, df_stats, df_tanova


def build_intensity_figure(df_session_means, modalities):
    """
    Separate figure: P50 at P50_CHANNEL against the STIMULUS intensity level, bars grouped by
    modality.

    Descriptive only. Bar height is the mean over sessions of each session's own mean P50 in that
    intensity level; the whisker is the SEM over sessions, which is what seaborn computes because
    `df_session_means` carries exactly one row per session x level.
    """
    levels = sorted(df_session_means["intensity_level"].unique())
    # Never hard-code 2-6: which levels survive analysis_excluded is a property of the data.
    hue_order = [m for m in modalities if m in set(df_session_means["modality"])]
    n_by_mod = df_session_means.groupby("modality")["sub"].nunique()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.barplot(
        data=df_session_means, x="intensity_level", y="p50", hue="modality",
        order=levels, hue_order=hue_order, palette=PALETTE_MODALITY,
        errorbar="se", capsize=0.12, err_kws=dict(linewidth=1.0), ax=ax,
    )

    # The P50 can sit either side of the pre-cue baseline, so mark where the sign changes.
    ax.axhline(0.0, color="0.6", linewidth=0.7, zorder=0)
    ax.set_xlabel("Stimulus intensity level (low $\\rightarrow$ high)")
    ax.set_ylabel(f"P50 at {P50_CHANNEL}, window mean (µV)")

    # Legend entries carry the session count, matching the "(n = 36)" row labels of Fig07.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [f"{MODALITY_LABELS[m]} (n = {int(n_by_mod[m])})" for m in labels],
        frameon=False, fontsize=FS_LEGEND, loc="best",
    )
    sns.despine(trim=True, ax=ax)
    return fig


# %%
# ---------------------------------------------------------------- main


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(FIGUREDIR, exist_ok=True)
    os.makedirs(STATSDIR, exist_ok=True)

    df_trials, topo = load_or_collect(args)

    # The primary set drops the SNR-tagged sessions.
    n_tagged = df_trials.loc[df_trials["snr_excluded"]].groupby(["sub", "modality"]).ngroups
    print(f"[snr] {n_tagged} sessions tagged by exclued_subs.json -> dropped from the figure")
    df_trials = df_trials[~df_trials["snr_excluded"]].reset_index(drop=True)
    topo = {"rows": [r for r in topo["rows"] if not r["snr_excluded"]], "info": topo["info"]}

    with plt.rc_context(PUB_RC):
        fig, df_stats, df_tanova = build_figure(
            df_trials, topo, args.modalities, args.n_bins, args.n_phase_bins
        )
        # Separate figure, same session set (the SNR-tagged sessions were dropped above).
        df_intensity = aggregate_intensity_p50(df_trials)
        fig_intensity = build_intensity_figure(df_intensity, args.modalities)

    stats_csv = os.path.join(
        STATSDIR, f"manuscript_{MANUSCRIPT_STEM.lower()}_p50meanwin_stats.csv"
    )
    df_stats.to_csv(stats_csv, index=False)
    print(f"\n=== {MANUSCRIPT_STEM} statistics ===")
    print(df_stats.to_string(index=False))
    print(f"written to {stats_csv}")

    tanova_csv = os.path.join(
        STATSDIR, f"manuscript_{MANUSCRIPT_STEM.lower()}_p50meanwin_topo_tanova.csv"
    )
    df_tanova.to_csv(tanova_csv, index=False)
    print("\n=== Panel (c) TANOVAs (CSV only) ===")
    print(df_tanova.to_string(index=False))
    print(f"written to {tanova_csv}")

    out_path = os.path.join(FIGUREDIR, FIGURE_STEM)
    fig.savefig(f"{out_path}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}.pdf / .png")

    # --- separate figure: P50 by stimulus intensity level ---
    intensity_csv = os.path.join(STATSDIR, f"{INTENSITY_FIGURE_STEM}_session_means.csv")
    df_intensity.to_csv(intensity_csv, index=False)
    print("\n=== P50 by intensity level (mean over sessions) ===")
    print(
        df_intensity.groupby(["modality", "intensity_level"])["p50"]
        .agg(["mean", "sem", "count"]).to_string()
    )
    print(f"session means written to {intensity_csv}")

    intensity_path = os.path.join(FIGUREDIR, INTENSITY_FIGURE_STEM)
    fig_intensity.savefig(f"{intensity_path}.pdf", dpi=300, bbox_inches="tight")
    fig_intensity.savefig(f"{intensity_path}.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure: {intensity_path}.pdf / .png")

    return dict(
        df_trials=df_trials, topo=topo, df_stats=df_stats, df_tanova=df_tanova, fig=fig,
        df_intensity=df_intensity, fig_intensity=fig_intensity,
    )


if __name__ == "__main__":
    results = main()

# %%
