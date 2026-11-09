"""
File: step008_cnv_alpha.py
Author: Rosa Grossmann
Date: 2026-09-25
Description:
Publication figure on the negative preparation component at CPz and its relation to pre-stimulus
alpha phase locking at CP4. Rows are the modality - ELECTRICAL on top, MECHANICAL underneath:

  a) The preparation negativity at CPz over time, perceived vs unperceived. Per session, the
     trials of one response are averaged; the curves are the mean +/- SEM of those session
     averages. Descriptive only - no test is run on these curves.

  b) Per-session alpha dITPC at CP4 against the preparation negativity at CPz, perceived and
     unperceived in separate panels. Only the Pearson r and p are annotated; the Spearman rho
     and a BH-FDR correction across the four tests (2 modalities x 2 responses) go to the CSV.

     DIRECTION: the dITPC is on the x axis and the preparation component's amplitude on the y,
     because the question is whether pre-stimulus phase locking affects that amplitude - not the
     reverse. Pearson's r and its p are symmetric in x and y, so the annotated numbers are the
     same either way round; only the fitted line (the regression of y on x) depends on it.

MEASURES

  * PREPARATION NEGATIVITY. Broadband epochs at CPz, NO surface Laplacian (the CSD is a spatial
    high-pass and would remove the broad slow potential). Chain per trial: low-pass at
    CNV_H_FREQ = 6 Hz (zero-phase FIR, 4 Hz transition band so the filter fits the short epoch)
    -> baseline on the pre-cue window BASELINE_BY_MOD -> crop to CNV_CROP. The baseline is taken
    on the low-passed signal, so the component is a deflection from the pre-cue level of the same
    signal that is read out. For panel (b) the amplitude is the most negative sample of the
    per-response session average inside CNV_PEAK_WIN.
    Variable and CSV column names use the `cnv_*` spelling for this component.

  * ALPHA dITPC. The alpha epochs are band-passed at each subject's individual alpha peak +/- 2 Hz
    and get a surface Laplacian on the full montage before CP4 is picked (alpha quantities are
    therefore in uV/cm^2). The phase comes from a reflect-padded Hilbert transform on the full
    epoch. The dITPC is the debiased squared ITPC (PPC), (n*R^2 - 1)/(n - 1) with R = |mean(exp(i*phase))|
    over the n trials of one response (can be slightly negative). The broadband |ERP| is then
    regressed out as in the ERP-regressed alpha dITPC figure (Fig03): |ERP| is taken from
    the broadband epochs at CP4 (surface Laplacian, baseline BASELINE_BY_MOD, mean over the trials
    of one response), one OLS slope shared by Yes and No is fitted per session on the dITPC over
    -0.75 .. -0.1 s, and dITPC_clean = dITPC - slope * (|ERP| - mean(|ERP|)). The value per
    session is the MAXIMUM of the cleaned curve inside ITPC_PEAK_WIN (-0.4 .. -0.1 s). The
    window stops at -0.1 s because the zero-phase alpha filter leaks post-stimulus signal
    backwards in time. The uncleaned peak, slope and R^2 are kept in the session table.

  * UNITS: the preprocessed FIFs carry FIFF_UNIT_V but the sample values are numerically
    microvolts. No 1e6 scaling is applied; the plausible microvolt range is checked per session.

Run:
    python step008_cnv_alpha.py
    python step008_cnv_alpha.py --sub 13 14 15 16
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
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# plotting_style is a per-directory copy and eeg_analysis_helpers lives next to this file, so
# both are imported BEFORE WDDIR joins sys.path.
from plotting_style import figure_style  # noqa: E402
import eeg_analysis_helpers as ah  # noqa: E402

if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)
from general_helpers import apply_baseline_epochs, is_subject_excluded  # noqa: E402

# %%
# ---------------------------------------------------------------- configuration

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

ITPC_CHANNEL = "CP4"
CNV_CHANNEL = "CPz"

# Pre-cue baseline: ends before the cue, so the component is a deflection from the pre-cue level.
BASELINE_BY_MOD = {"elec": (-0.55, -0.45), "mecha": (-0.55, -0.45)}
# Cue onset per modality. The mechanical stimulator has a ~30 ms onset delay, so its cue sits
# 30 ms earlier relative to the (delayed) stimulus.
CUE_TIME = {"elec": -0.40, "mecha": -0.43}

# dITPC is the MAX of the per-session dITPC curve inside this window.
ITPC_PEAK_WIN = (-0.4, -0.1)

# Preparation negativity: low-pass, then the most negative sample of the per-response session
# average in CNV_PEAK_WIN.
CNV_H_FREQ = 6.0
CNV_H_TRANS = 4.0
CNV_PEAK_WIN = (-0.2, 0.1)
CNV_CROP = (-0.6, 0.25)

# Unusable recordings, not merely low-SNR: sub-23 mecha had ground and reference swapped,
# sub-36 mecha had exceptionally high force values, and the two electrical sessions elec-19 and
# elec-28 have a wrong trigger time (10 ms longer cue-stimulus window).
FAULTY_SESSIONS = {("mecha", 23), ("mecha", 36), ("elec", 19), ("elec", 28)}

# The outermost samples of a Hilbert transform are degenerate (reflect padding makes the signal
# even-symmetric about the last real sample), so every readout is kept this far from the edge.
N_EDGE_SAMPLES = 5

# dITPC / ERP-regression window. The Hilbert runs on the full epoch, the phase is cropped to this
# window afterwards.
WIN_TMIN, WIN_TMAX = -0.75, -0.1
RESPONSES = ("No", "Yes")
PLAUSIBLE_UV_STD = (0.1, 200.0)

MODALITY_LABELS = {"elec": "Electrical", "mecha": "Mechanical"}
RESPONSE_LABELS = {"Yes": "perceived", "No": "unperceived"}

colors, _ = figure_style()
PALETTE_RESP = {"Yes": colors["perceived"], "No": colors["unperceived"]}

FIGUREDIR = os.path.join(WDDIR, "Writing", "img")
STATSDIR = os.path.join(RESDIR, "Statistics", "eeg", "cnv_alpha")
FIGURE_STEM = "Fig08_cnv_alpha"

# figure_style()'s 20 pt axis labels are sized for a single large panel, so the figure is built
# inside this rc context instead.
PUB_RC = {
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.size": 13,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.3,
}
FS_PANEL_TITLE = 12
FS_PANEL_STAT = 11
FS_LEGEND = 10
FS_ROW_LABEL = 15
FS_PANEL_LETTER = 18
FIGSIZE = (15.0, 8.8)


def parse_args(argv=None):
    """CLI. Parsed strictly, so a typo on the command line is still an error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", nargs="+", type=int, default=ALL_SUBJECTS,
                        help="Subject IDs (default: 13-52).")
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES,
                        help="Modalities to process (default: both).")
    parser.add_argument("--cache-file", default=None,
                        help="Optional pickle of the collected session tables. When the file "
                             "exists it is read INSTEAD of reloading the epochs; when it does "
                             "not, the full loop runs and writes it. Opt-in only.")
    if argv is None:
        argv = [] if running_in_kernel() else sys.argv[1:]
    return parser.parse_args(argv)


# %%
# ---------------------------------------------------------------- helpers


def window_slice(times, win, label):
    """
    Inclusive index slice for a time window. Raises if the window is not contained in `times` -
    a silently clipped window would change what is being measured.
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
    transients out of the window of interest. It runs on the FULL epoch before any crop.
    """
    n_pad = int(3 * sfreq / fmin)
    padded = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode="reflect")
    return scipy_hilbert(padded, axis=-1)[:, :, n_pad:-n_pad]


def assert_microvolt_scale(data, label):
    """
    Verify the amplitude scale really is microvolt-like before it is reported as uV. Must be
    handed EEG channels only (CNAP, ECG, RESP and EOG are in other units).
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
    Assert two metadata frames describe exactly the same trials, in the same order. The alpha
    and broadband measures come from two files and are paired by position.
    """
    idx_a = meta_a["epoch_index"].to_numpy()
    idx_b = meta_b["epoch_index"].to_numpy()
    if idx_a.shape != idx_b.shape or not np.array_equal(idx_a, idx_b):
        raise ValueError(
            f"sub-0{subID} ses-{modality}: {label_a} ({len(idx_a)} trials) and {label_b} "
            f"({len(idx_b)} trials) do not describe the same trials. Investigate upstream "
            f"preprocessing - do not reconcile here."
        )


def compute_subject_abs_erp(broadband_fif, modality, response_alpha, times_itpc, subID):
    """
    |ERP| per response at ITPC_CHANNEL from the broadband epochs of one session.

    Steps: clean-trial mask -> CSD -> pick ITPC_CHANNEL -> pre-cue baseline -> mean over the
    trials of one response -> absolute value -> crop to the dITPC time axis. The trial order must
    equal that of the alpha epochs the dITPC was computed from, and so must the time axis.
    Returns {"No": array, "Yes": array}.
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

    epochs_csd = ah.apply_surface_laplacian(epochs).pick([ITPC_CHANNEL])
    apply_baseline_epochs(epochs_csd, broadband_fif,
                          baseline_window=BASELINE_BY_MOD[modality])

    times = epochs_csd.times
    tmask = (times >= WIN_TMIN) & (times <= WIN_TMAX)
    if tmask.sum() != len(times_itpc) or not np.allclose(times[tmask], times_itpc):
        raise ValueError(
            f"sub-0{subID} {modality}: ERP time axis ({tmask.sum()} samples) does not match "
            f"the dITPC time axis ({len(times_itpc)} samples)."
        )

    data = epochs_csd.get_data(copy=True)[:, 0, tmask]  # (n_trials, n_times)
    return {resp: np.abs(data[response_bb == resp].mean(axis=0)) for resp in RESPONSES}


def regress_out_erp(itpc_no, itpc_yes, erp_no, erp_yes):
    """
    Shared-slope removal of |ERP| from dITPC, per row. All inputs (n_rows, n_times).

    Returns (clean_no, clean_yes, slope, r2) with slope and r2 of shape (n_rows,).
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
            raise ValueError(f"Row {s}: |ERP| is constant, the regression is undefined.")
        slope[s] = np.sum((x - x_mean) * (y - y.mean())) / var_x
        r2[s] = np.corrcoef(x, y)[0, 1] ** 2
        clean_no[s] = itpc_no[s] - slope[s] * (erp_no[s] - x_mean)
        clean_yes[s] = itpc_yes[s] - slope[s] * (erp_yes[s] - x_mean)
    if not (np.all(np.isfinite(clean_no)) and np.all(np.isfinite(clean_yes))
            and np.all(np.isfinite(r2))):
        raise ValueError("Non-finite value after regressing out |ERP|.")
    return clean_no, clean_yes, slope, r2


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
    Broadband epochs with the clean-trial mask applied. NOT baselined here: the preparation
    component is low-passed first, so that its baseline is taken on the filtered signal.
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


def compute_cnv_timecourse(epochs_broadband, epoch_fif, modality, label):
    """
    CPz preparation negativity per trial: low-pass -> pre-cue baseline -> crop. No surface
    Laplacian. Returns (times, data) with data shaped (n_trials, n_times).
    """
    epochs_cnv = (
        epochs_broadband.copy()
        .pick([CNV_CHANNEL])
        .filter(
            l_freq=None, h_freq=CNV_H_FREQ, h_trans_bandwidth=CNV_H_TRANS,
            picks="eeg", method="fir", pad="reflect_limited", phase="zero", verbose=False,
        )
    )
    apply_baseline_epochs(epochs_cnv, epoch_fif, baseline_window=BASELINE_BY_MOD[modality])
    epochs_cnv.crop(tmin=CNV_CROP[0], tmax=CNV_CROP[1])
    data = epochs_cnv.get_data(copy=True)[:, 0, :]
    assert_microvolt_scale(data, f"{label} CNV at {CNV_CHANNEL}")
    return epochs_cnv.times.copy(), data


# %%
# ---------------------------------------------------------------- data collection


def collect_sessions(subjects, modalities):
    """
    One pass over every session.

    df_sessions : one row per SESSION x RESPONSE with the alpha dITPC peak at ITPC_CHANNEL and the
                  preparation-negativity peak at CNV_CHANNEL (panel b).
    df_cnv_tc   : long frame, one row per SESSION x RESPONSE x TIME with the session-average
                  preparation negativity (panel a).

    Both carry `snr_excluded`; the caller drops those sessions for the figure.
    """
    session_rows = []
    tc_frames = []
    ref_times = None

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

            if ITPC_CHANNEL not in epochs_alpha.ch_names:
                raise ValueError(f"{session_label}: channel {ITPC_CHANNEL} absent from the alpha epochs.")
            if CNV_CHANNEL not in epochs_broadband.ch_names:
                raise ValueError(
                    f"{session_label}: channel {CNV_CHANNEL} absent from the broadband epochs."
                )

            print(f"Processing {session_label} "
                  f"({yes_mask.sum()} Yes / {no_mask.sum()} No trials)")

            # --- alpha path: CSD on the full montage, then pick, then one Hilbert ---
            epochs_alpha_pick = ah.apply_surface_laplacian(epochs_alpha).pick([ITPC_CHANNEL])
            alpha_times = epochs_alpha_pick.times.copy()
            alpha_phase = np.angle(hilbert_analytic(
                epochs_alpha_pick.get_data(copy=True),
                epochs_alpha_pick.info["sfreq"], ALPHA_FMIN,
            ))[:, 0, :]                                              # (n_trials, n_times)

            # Crop the phase to the dITPC window of the ERP regression (Hilbert already ran on
            # the full epoch). |ERP| is returned on exactly this time axis.
            win_mask = (alpha_times >= WIN_TMIN) & (alpha_times <= WIN_TMAX)
            win_times = alpha_times[win_mask]
            alpha_phase = alpha_phase[:, win_mask]
            itpc_sl = window_slice(win_times, ITPC_PEAK_WIN, "dITPC peak")
            if itpc_sl.start < N_EDGE_SAMPLES:
                raise ValueError(
                    f"{session_label}: dITPC window {ITPC_PEAK_WIN} starts at sample "
                    f"{itpc_sl.start} of the cropped axis, within {N_EDGE_SAMPLES} samples "
                    f"of the Hilbert edge."
                )

            # |ERP| at ITPC_CHANNEL per response (CSD, pre-cue baseline BASELINE_BY_MOD), as in Fig03.
            abs_erp = compute_subject_abs_erp(
                broadband_fif, modality, response, win_times, subID,
            )

            # --- broadband path: preparation negativity at CPz ---
            cnv_times, cnv_data = compute_cnv_timecourse(
                epochs_broadband, broadband_fif, modality, session_label
            )
            if ref_times is None:
                ref_times = cnv_times
            elif cnv_times.shape != ref_times.shape or not np.allclose(cnv_times, ref_times):
                raise ValueError(
                    f"{session_label}: CNV time axis differs from the first session's; the "
                    f"timecourses cannot be averaged across sessions."
                )
            cnv_sl = window_slice(cnv_times, CNV_PEAK_WIN, "CNV peak")

            # Debiased squared ITPC (PPC): (n * R^2 - 1) / (n - 1), R the resultant vector
            # length. Plain R is biased upward at low trial counts and Yes and No differ in
            # count. The value can be slightly negative.
            itpc_curves = {}
            for resp, mask in [("Yes", yes_mask), ("No", no_mask)]:
                n_resp = int(mask.sum())
                if n_resp < 2:
                    raise ValueError(
                        f"{session_label}: debiased ITPC needs at least 2 {resp} trials, got {n_resp}."
                    )
                itpc_r = np.abs(np.mean(np.exp(1j * alpha_phase[mask]), axis=0))
                itpc_curves[resp] = (n_resp * itpc_r ** 2 - 1) / (n_resp - 1)

            # Remove |ERP| from the dITPC: one OLS slope shared by Yes and No (this session).
            clean_no, clean_yes, slope, r2 = regress_out_erp(
                itpc_curves["No"][None, :], itpc_curves["Yes"][None, :],
                abs_erp["No"][None, :], abs_erp["Yes"][None, :],
            )
            clean_curves = {"Yes": clean_yes[0], "No": clean_no[0]}

            for resp, mask in [("Yes", yes_mask), ("No", no_mask)]:
                itpc_seg = clean_curves[resp][itpc_sl]
                cnv_avg = cnv_data[mask].mean(axis=0)
                cnv_seg = cnv_avg[cnv_sl]
                session_rows.append({
                    "sub": int(subID),
                    "modality": modality,
                    "response": resp,
                    "n_trials": int(mask.sum()),
                    "itpc_peak": float(itpc_seg.max()),
                    "itpc_peak_time": float(win_times[itpc_sl][int(itpc_seg.argmax())]),
                    "itpc_peak_raw": float(itpc_curves[resp][itpc_sl].max()),
                    "erp_slope": float(slope[0]),
                    "erp_r2": float(r2[0]),
                    # The component is a NEGATIVE peak, hence argmin.
                    "cnv_amp_uv": float(cnv_seg.min()),
                    "cnv_peak_time": float(cnv_times[cnv_sl][int(cnv_seg.argmin())]),
                    "snr_excluded": snr_excluded,
                })
                tc_frames.append(pd.DataFrame({
                    "sub": int(subID), "modality": modality, "response": resp,
                    "time": cnv_times, "cnv_uv": cnv_avg, "snr_excluded": snr_excluded,
                }))

    if not session_rows:
        raise RuntimeError("No usable sessions found - nothing to plot.")
    return pd.DataFrame(session_rows), pd.concat(tc_frames, ignore_index=True)


def load_or_collect(args):
    """Collect the session tables, optionally via the opt-in pickle cache (key-checked)."""
    key = {
        "sub": sorted(args.sub),
        "modalities": list(args.modalities),
        "suffix": FILE_SUFFIX_EPOCHS,
        "itpc_channel": ITPC_CHANNEL,
        "cnv_channel": CNV_CHANNEL,
        "cnv_h_freq": CNV_H_FREQ,
        "schema": 3,
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
        return cached["df_sessions"], cached["df_cnv_tc"]

    df_sessions, df_cnv_tc = collect_sessions(args.sub, args.modalities)
    if args.cache_file:
        with open(args.cache_file, "wb") as fh:
            pickle.dump({"key": key, "df_sessions": df_sessions, "df_cnv_tc": df_cnv_tc}, fh)
        print(f"[cache] wrote session tables to {args.cache_file}")
    return df_sessions, df_cnv_tc


# %%
# ---------------------------------------------------------------- panels


def plot_panel_a(ax, df_tc_mod, modality, show_legend):
    """
    Preparation negativity over time, perceived vs unperceived: mean +/- SEM over the
    per-session averages. Descriptive only.
    """
    ax.axvspan(*BASELINE_BY_MOD[modality], color="0.9", zorder=0, linewidth=0)
    ax.axhline(0.0, color="0.6", linewidth=0.7, zorder=1)
    ax.axvline(CUE_TIME[modality], color="0.3", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(0.0, color="0.3", linestyle="-", linewidth=1.0, zorder=1)
    for resp in ["Yes", "No"]:
        mat = (
            df_tc_mod[df_tc_mod["response"] == resp]
            .pivot(index="sub", columns="time", values="cnv_uv")
        )
        times = mat.columns.to_numpy(dtype=float)
        mean_tc = mat.mean(axis=0).to_numpy()
        sem_tc = stats.sem(mat.to_numpy(), axis=0)
        ax.plot(times, mean_tc, color=PALETTE_RESP[resp], linewidth=2.0,
                label=RESPONSE_LABELS[resp], zorder=3)
        ax.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc, color=PALETTE_RESP[resp],
                        alpha=0.15, linewidth=0, zorder=2)
    # Event names just above the axes: x in data (s), y in axes coordinates.
    trans_top = blended_transform_factory(ax.transData, ax.transAxes)
    for t_ev, name in [(CUE_TIME[modality], "cue"), (0.0, "stimulus")]:
        ax.text(t_ev, 1.01, name, transform=trans_top, ha="center", va="bottom",
                fontsize=FS_LEGEND, color="0.2")
    ax.set_xlim(CNV_CROP)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Preparation negativity\nat {CNV_CHANNEL} (µV)")
    if show_legend:
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower left")


def plot_panel_b(ax, panel_df, response, r_val, p_val, show_ylabel):
    """
    Per-session alpha dITPC against the preparation negativity, one response condition. Only the
    Pearson r and p are annotated; Spearman and the BH-FDR go to the CSV.
    """
    sns.regplot(
        x=panel_df["itpc_peak"].to_numpy(dtype=float),
        y=panel_df["cnv_amp_uv"].to_numpy(dtype=float),
        ax=ax, ci=None, color=PALETTE_RESP[response],
        scatter_kws=dict(s=18, alpha=0.85, linewidths=0),
        line_kws=dict(lw=1.5),
    )
    # The stars mark the UNCORRECTED p printed beside them; the BH-FDR is in the CSV.
    ax.text(0.04, 0.96, f"r = {r_val:+.2f}\np {format_p(p_val)} {sig_stars(p_val)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=FS_PANEL_STAT)
    ax.set_title(RESPONSE_LABELS[response], fontsize=FS_PANEL_TITLE)
    ax.set_xlabel(f"alpha dITPC (ERP regressed)\nat {ITPC_CHANNEL}")
    ax.set_ylabel(f"Preparation negativity at {CNV_CHANNEL} (µV)" if show_ylabel else "")


# %%
# ---------------------------------------------------------------- figure assembly


def build_figure(df_sessions, df_cnv_tc, modalities):
    """Assemble the figure (rows = modality; a | b-perceived | b-unperceived) and the stats."""
    stat_rows = []
    corr = {}
    for modality in modalities:
        for response in ["Yes", "No"]:
            sel = df_sessions[
                (df_sessions["modality"] == modality) & (df_sessions["response"] == response)
            ]
            if len(sel) < 3:
                raise ValueError(
                    f"{modality} {response}: only {len(sel)} sessions, too few to correlate."
                )
            x = sel["cnv_amp_uv"].to_numpy(dtype=float)
            y = sel["itpc_peak"].to_numpy(dtype=float)
            r_val, p_val = stats.pearsonr(x, y)
            rho, p_rho = stats.spearmanr(x, y)
            corr[(modality, response)] = (float(r_val), float(p_val), sel)
            stat_rows.append({
                "panel": "b", "modality": modality, "channel": ITPC_CHANNEL,
                "response_set": response, "n": int(len(sel)),
                "measure_x": "itpc_peak", "measure_y": "cnv_amp_uv",
                "pearson_r": float(r_val), "pearson_p": float(p_val),
                "spearman_rho": float(rho), "spearman_p": float(p_rho),
            })
    df_stats = pd.DataFrame(stat_rows)
    # BH-FDR over the four correlations. Written to the CSV only.
    df_stats["pearson_p_fdr"] = ah.fdr_correct_pvals(df_stats["pearson_p"].to_numpy())
    df_stats["n_fdr"] = len(df_stats)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(len(modalities), 2, width_ratios=[1.15, 2.0], hspace=0.55, wspace=0.28)
    axes_a, axes_b_all = [], []
    for row_idx, modality in enumerate(modalities):
        n_sessions = df_sessions[df_sessions["modality"] == modality]["sub"].nunique()

        ax_a = fig.add_subplot(gs[row_idx, 0])
        plot_panel_a(ax_a, df_cnv_tc[df_cnv_tc["modality"] == modality], modality,
                     show_legend=(row_idx == 0))
        axes_a.append(ax_a)
        ax_a.text(-0.40, 0.5, f"{MODALITY_LABELS[modality]}\n(n = {n_sessions})",
                  transform=ax_a.transAxes, rotation=90, va="center", ha="center",
                  fontsize=FS_ROW_LABEL, fontweight="bold")

        gs_b = gs[row_idx, 1].subgridspec(1, 2, wspace=0.38)
        row_b = []
        for col_idx, response in enumerate(["Yes", "No"]):
            ax_b = fig.add_subplot(gs_b[0, col_idx])
            r_val, p_val, sel = corr[(modality, response)]
            plot_panel_b(ax_b, sel, response, r_val, p_val, show_ylabel=(col_idx == 0))
            row_b.append(ax_b)
        # Yes and No are only comparable on shared axes.
        x_lo = min(a.get_xlim()[0] for a in row_b)
        x_hi = max(a.get_xlim()[1] for a in row_b)
        y_lo = min(a.get_ylim()[0] for a in row_b)
        y_hi = max(a.get_ylim()[1] for a in row_b)
        for a in row_b:
            a.set_xlim(x_lo, x_hi)
            a.set_ylim(y_lo, y_hi)
            sns.despine(trim=True, ax=a)
        axes_b_all.append(row_b)

        if row_idx == 0:
            # Both letters on one line: y from panel (a)'s axes, x from each panel's own axes.
            for ax, letter, dx in [(ax_a, "A", -0.30), (row_b[0], "B", -0.30)]:
                trans = blended_transform_factory(ax.transAxes, ax_a.transAxes)
                ax.text(dx, 1.12, letter, transform=trans, fontsize=FS_PANEL_LETTER,
                        fontweight="bold", va="bottom", ha="left")

    # Panel (a) is the same quantity in both rows, so it shares a y axis.
    a_lo = min(a.get_ylim()[0] for a in axes_a)
    a_hi = max(a.get_ylim()[1] for a in axes_a)
    for a in axes_a:
        a.set_ylim(a_lo, a_hi)
        sns.despine(ax=a)

    return fig, df_stats


# %%
# ---------------------------------------------------------------- main


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(FIGUREDIR, exist_ok=True)
    os.makedirs(STATSDIR, exist_ok=True)

    df_sessions, df_cnv_tc = load_or_collect(args)

    n_tagged = df_sessions.loc[df_sessions["snr_excluded"]].groupby(["sub", "modality"]).ngroups
    print(f"[snr] {n_tagged} sessions tagged by exclued_subs.json -> dropped from the figure")
    df_sessions = df_sessions[~df_sessions["snr_excluded"]].reset_index(drop=True)
    df_cnv_tc = df_cnv_tc[~df_cnv_tc["snr_excluded"]].reset_index(drop=True)

    with plt.rc_context(PUB_RC):
        fig, df_stats = build_figure(df_sessions, df_cnv_tc, args.modalities)

    stats_csv = os.path.join(STATSDIR, "cnv_alpha_itpc_stats.csv")
    df_stats.to_csv(stats_csv, index=False)
    print("\n=== dITPC vs preparation negativity ===")
    print(df_stats.to_string(index=False))
    print(f"written to {stats_csv}")

    tc_csv = os.path.join(STATSDIR, "cnv_timecourse_session.csv")
    df_cnv_tc.to_csv(tc_csv, index=False)
    print(f"written to {tc_csv}")

    out_path = os.path.join(FIGUREDIR, FIGURE_STEM)
    fig.savefig(f"{out_path}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}.pdf / .png")

    return dict(df_sessions=df_sessions, df_cnv_tc=df_cnv_tc, df_stats=df_stats, fig=fig)


if __name__ == "__main__":
    results = main()

# %%
