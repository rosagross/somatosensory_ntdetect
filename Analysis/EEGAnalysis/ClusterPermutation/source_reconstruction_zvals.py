"""
File: source_reconstruction_zvals.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-08-27
Description: 
Source reconstruction of the z-value topography maps of selected model
coefficients of cluster-permutation runs. The coefficients are listed in
SOURCE_ROWS and may come from different run directories (runs split up per
coefficient), but all runs must share the baseline, which must also equal the
baseline applied here. Per coefficient, the latencies are the peak |z| of its
strongest significant clusters (at most `max_peaks`), plus optional extra
latencies that are not peaks. Significance is only used as a label on the
figures, never as a filter.

Steps:
- load BEM, transform, forward model per subject
- load noise covariance matrix from ERP data (and, for LCMV, the data covariance)
- create inverste operator / beamformer filters
- apply inverse operator on group z-vals
- morph to average source space (computed individual morph before)
- after running the above steps for all subjects, average to a single stc
- render brain snapshots (both hemispheres, lateral + medial) at selected
  latencies, in the style of plot_cluster_results_parcellated.py

The inverse method is selectable with the INVERSE_METHOD environment variable,
either "mne" (minimum-norm, lambda2 = 1/9; MNE_METHOD selects dSPM or plain MNE) or
"lcmv" (beamformer, using the cached data covariance). Every output -- cached stcs and
figures alike -- carries an "inv-<inverse_name>" tag, e.g.
"inv-mne_loose-0_depth-0.8_method-dSPM" or "inv-lcmv_wnunit-noise-gain_reg-0.05" (the
naming of step002_inverse_solution_multisub_alpha_lcmv.py), so the reconstructions live
side by side on disk.

The manuscript panel (detection response and modality at the peak latencies of
their sensor clusters) is no longer built here -- it reads the cached group stcs
written below and lives in plot_source_zmap_manuscript_panel.py.
"""

# %%
import os
import re
import sys
import textwrap
import matplotlib.pyplot as plt
import mne
from mne.beamformer import apply_lcmv, make_lcmv, apply_lcmv_epochs # , make_nulling_beamformer, apply_nulling
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_brain_colorbar
import numpy as np
import matplotlib.image as mpimg
import pickle
import seaborn as sns
from plotting_style import figure_style
colors, dpi = figure_style()

# %%
# setup dirs
DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
RECONDIR = os.getenv('SUBJECTS_DIR')
WDDIR = os.getenv('WDDIR')
src_file_fsaverage = f'{RECONDIR}/fsaverage_data/bem/fsaverage-ico-5-src.fif'

# import general helpers
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import apply_baseline_epochs

cluster_results_dir = os.path.join(RESDIR, "Statistics", "eeg", "cluster_permutation", "cluster_mass", "runs")

_BL = "mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4"
RUN_DETECTION = f"20261020-205919__{_BL}"
RUN_MODALITY = f"20261028-100155__{_BL}__coef-modality"
RUN_MODALITY_X_DETECTION = f"20261028-144658__{_BL}__coef-modality_response"

# (run ID, coefficient folder, max_peaks, extra non-peak latencies in s or None).
# The numeric prefix of the coefficient folder depends on the run, so it is part of
# the configuration. Each coefficient is reconstructed at the peak |z| latencies of
# its `max_peaks` strongest significant clusters, plus the extra latencies. These
# match the topomaps of plot_cluster_results_topoplot_panel_joint_peaks.py.
SOURCE_ROWS = [
    (RUN_DETECTION, "01__response_decoded_Yes", 5, [0.0, 0.050]),
    (RUN_MODALITY, "01__modality_Mechanical", 5, None),
    (RUN_MODALITY_X_DETECTION, "01__modality_Mechanical_response_decoded_Yes", 1, None),
]

suffix = "_preprestim_corrected"
l_freq_orig = 0.1
l_freq = 1
h_freq = 40
noise_tmin, noise_tmax = -0.55, -0.45
tmin_baseline = -0.55
tmax_baseline = -0.45

# All runs must share one baseline, and it must be the baseline applied in this script
# (epochs and noise covariance), otherwise the z-maps and the inverse do not belong together.
_baselines = {}
for _run_id, _folder, _n, _extra in SOURCE_ROWS:
    _m = re.search(r"__bl-(.+?)__f-", _run_id)
    if _m is None:
        raise ValueError(f"no baseline tag (__bl-...__f-) in run ID {_run_id}")
    _baselines[_run_id] = _m.group(1)
if len(set(_baselines.values())) != 1:
    raise ValueError(
        "SOURCE_ROWS mix runs with different baselines: "
        + "; ".join(f"{b} <- {r}" for r, b in _baselines.items())
    )
BASELINE_TAG = next(iter(_baselines.values()))
_bl_numbers = re.search(r"(-?\d+\.?\d*)_(-?\d+\.?\d*)$", BASELINE_TAG)
if _bl_numbers is None or not np.allclose(
    [float(_bl_numbers.group(1)), float(_bl_numbers.group(2))], [tmin_baseline, tmax_baseline]
):
    raise ValueError(
        f"run baseline {BASELINE_TAG} differs from the baseline applied here "
        f"({tmin_baseline}, {tmax_baseline})."
    )
print(f"[config] all {len(SOURCE_ROWS)} run(s) share baseline {BASELINE_TAG}")

subID_both = [15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
subID_mecha = [30, 32, 40, 16, 17, 18, 19, 24, 26, 27, 28, 29, 34, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52] 
subID_elec = [15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52] 

# The cluster run is a combined LMM over both modalities, so the z-map is projected
# through every available subject/session pair and the resulting stcs are averaged.
sessions = [(s, "mecha") for s in subID_mecha] + [(s, "elec") for s in subID_elec]

# Smoke-test escape hatch: SESSION_LIMIT=2 runs only the first two sessions.
_session_limit = os.getenv("SESSION_LIMIT", "")
if _session_limit:
    sessions = sessions[: int(_session_limit)]
    print(f"[config] SESSION_LIMIT={_session_limit} -> {len(sessions)} session(s)")

# Re-use cached group stcs unless RECOMPUTE=1.
RECOMPUTE = os.getenv("RECOMPUTE", "0") == "1"

# Inverse method: "mne" (minimum-norm, lambda2 = 1/9) or "lcmv" (beamformer). Both run
# on the same fixed-orientation forward and produce one scalar time course per vertex,
# so everything downstream (morph, average, plotting) is method-agnostic. The tag goes
# into every output name so the two reconstructions never overwrite each other.
INVERSE_METHOD = "mne"
if INVERSE_METHOD not in ("mne", "lcmv"):
    raise ValueError(f"INVERSE_METHOD must be 'mne' or 'lcmv', got {INVERSE_METHOD!r}")

# LCMV only: window of the cached data covariance, and the filter settings used by every
# other LCMV call in Analysis/SourceReconstruction/.
data_tmin, data_tmax = -0.4, 0.3
LCMV_REG = 0.05
LCMV_WEIGHT_NORM = "unit-noise-gain"

# MNE only: "dSPM" or "MNE" -- same settings as alpha_source_common.py. dSPM runs on the
# free-orientation forward with depth weighting; plain MNE runs on the fixed-orientation
# forward (make_inverse_operator refuses depth weighting there, so MNE_DEPTH must be None).
MNE_METHOD = "MNE"
MNE_LOOSE = 0
MNE_DEPTH = None #0.8
if MNE_METHOD not in ("dSPM", "MNE"):
    raise ValueError(f"MNE_METHOD must be 'dSPM' or 'MNE', got {MNE_METHOD!r}")

# Same naming scheme as step002_inverse_solution_multisub_alpha_lcmv.py (inverse_name), so
# the settings of a reconstruction can be read off its file names.
if INVERSE_METHOD == "lcmv":
    inverse_name = f"{INVERSE_METHOD}_wn{LCMV_WEIGHT_NORM}_reg-{LCMV_REG}"
    INV_LABEL = "LCMV"
else:
    inverse_name = f"{INVERSE_METHOD}_loose-{MNE_LOOSE}_depth-{MNE_DEPTH}_method-{MNE_METHOD}"
    INV_LABEL = MNE_METHOD
INV_TAG = f"inv-{inverse_name}"
print(f"[config] inverse method: {inverse_name}")

# Figures of all selected coefficients go into one folder named by the shared baseline. The
# cached group stcs stay in one stc folder per run, where the other scripts look for them.
figuredir = os.path.join(
    RESDIR, "Figures", "Main", "eeg_results", "source_reconstruction",
    "zvals_clusterperm", f"selected__bl-{BASELINE_TAG}",
)
os.makedirs(figuredir, exist_ok=True)


def stc_dir_for(run_id):
    path = os.path.join(
        RESDIR, "Figures", "Main", "eeg_results", "source_reconstruction",
        "zvals_clusterperm", run_id, "stc",
    )
    os.makedirs(path, exist_ok=True)
    return path


COLORMAP = "mne_analyze"


# %%
# -----------------------------------------------------------------------------
# Read the z-value map of every selected coefficient and pick its latencies.
# cluster_display holds the cluster id where a channel/time point belongs to a
# significant cluster and NaN everywhere else. A coefficient is identified by
# (run ID, coefficient folder), since folder names repeat across runs.
# -----------------------------------------------------------------------------
def cluster_peaks(z_vals, cluster_display):
    """Per significant cluster: (cluster id, peak |z|, time index of the peak)."""
    ids = cluster_display[~np.isnan(cluster_display)]
    ids = np.unique(ids[~np.isclose(ids, 0.0)])
    peaks = []
    for cluster_id in ids:
        mask = np.isclose(cluster_display, cluster_id)
        coords = np.argwhere(mask)
        k = int(np.argmax(np.abs(z_vals[mask])))
        peaks.append((int(cluster_id), float(np.abs(z_vals[mask][k])), int(coords[k][1])))
    return peaks


zmaps = {}
for run_id, folder, max_peaks, extra_times in SOURCE_ROWS:
    key = (run_id, folder)
    if key in zmaps:
        raise ValueError(f"{folder} of run {run_id} listed twice in SOURCE_ROWS.")
    coeff_dir = os.path.join(cluster_results_dir, run_id, "coefficients", folder)
    cluster_file = os.path.join(coeff_dir, "cluster_results.npz")
    channels_file = os.path.join(coeff_dir, "channels.txt")
    if not os.path.isfile(cluster_file):
        raise FileNotFoundError(f"cluster_results.npz not found: {cluster_file}")
    if not os.path.isfile(channels_file):
        raise FileNotFoundError(f"channels.txt not found: {channels_file}")

    data = np.load(cluster_file)
    with open(channels_file) as f:
        channels = [line.strip() for line in f if line.strip()]
    z_vals = data["z_vals"]
    times = np.asarray(data["times"], dtype=float)
    if z_vals.shape[0] != len(channels):
        raise ValueError(
            f"{folder}: z_vals has {z_vals.shape[0]} rows but channels.txt "
            f"lists {len(channels)} channels."
        )

    peaks = cluster_peaks(z_vals, data["cluster_display"])
    if not peaks:
        raise ValueError(f"{folder} ({run_id}): no significant cluster, so there is no peak to reconstruct.")
    strongest = sorted(peaks, key=lambda pk: pk[1], reverse=True)[:max_peaks]
    peak_idx = np.unique([pk[2] for pk in strongest])

    # Extra latencies are non-peaks unless one lands on a peak sample, which stays a peak.
    extra_idx = np.array([], dtype=int)
    if extra_times:
        extra_times = np.asarray(extra_times, dtype=float)
        extra_idx = np.array([int(np.argmin(np.abs(times - t))) for t in extra_times], dtype=int)
        step = np.median(np.diff(times))
        off = np.abs(times[extra_idx] - extra_times) > step / 2
        if off.any():
            raise ValueError(
                f"{folder}: extra latencies {np.round(extra_times[off] * 1000, 1)} ms are outside the "
                f"time axis or do not land on a sample (step {step * 1000:.1f} ms)."
            )
    snap_idx = np.unique(np.concatenate([peak_idx, extra_idx]).astype(int))
    is_peak = np.isin(snap_idx, peak_idx)

    # pvals_clustermass is a full channel x time map of cluster-mass p-values; its
    # minimum is the smallest cluster p of this coefficient and annotates the figures.
    p_min = float(np.nanmin(data["pvals_clustermass"]))

    zmaps[key] = {
        "z_vals": z_vals,
        "times": times,
        "channels": channels,
        "p_min": p_min,
        "snap_idx": snap_idx,
        "is_peak": is_peak,
    }
    print(f"[coef] {folder} ({run_id[:15]}): {len(peaks)} significant cluster(s), p_min={p_min:.4f}; "
          "latencies (ms, * = peak): "
          + ", ".join(f"{times[i] * 1000:.0f}{'*' if pk else ''}" for i, pk in zip(snap_idx, is_peak)))

coefficient_names = list(zmaps)   # keys, in SOURCE_ROWS order = row order of the summary grid

# All coefficients must share the channel order and the time axis; the inverse
# operator is built once per session against exactly this channel list.
perm_channels = zmaps[coefficient_names[0]]["channels"]
zmap_times = zmaps[coefficient_names[0]]["times"]
for key, entry in zmaps.items():
    if entry["channels"] != perm_channels:
        raise ValueError(f"{key[1]} ({key[0]}): channels.txt differs from the other coefficients.")
    if not np.array_equal(entry["times"], zmap_times):
        raise ValueError(f"{key[1]} ({key[0]}): time axis differs from the other coefficients.")
print(f"[coef] {len(perm_channels)} channels, times {zmap_times[0]:.3f}..{zmap_times[-1]:.3f} s "
      f"({len(zmap_times)} samples)")


# %%
# -----------------------------------------------------------------------------
# Group stcs: the cache is per coefficient, so adding coefficients to a run only
# costs the inverse of the new ones -- the ones already on disk are re-used.
# -----------------------------------------------------------------------------
def stc_cache_path(key):
    # The method tag sits before "-mean" so that the "-lh.stc"/"-rh.stc" suffix
    # mne.read_source_estimate expects stays at the very end of the name.
    run_id, folder = key
    return os.path.join(stc_dir_for(run_id), f"{folder}__{INV_TAG}-mean")


def stc_is_cached(key):
    stem = stc_cache_path(key)
    return os.path.isfile(stem + "-lh.stc") and os.path.isfile(stem + "-rh.stc")


stc_mean = {}
if not RECOMPUTE:
    for coefficient_name in coefficient_names:
        if stc_is_cached(coefficient_name):
            stc_mean[coefficient_name] = mne.read_source_estimate(stc_cache_path(coefficient_name))
    print(f"[cache] loaded {len(stc_mean)}/{len(coefficient_names)} group stc(s) "
          f"(set RECOMPUTE=1 to redo)")

missing_coefficients = [c for c in coefficient_names if c not in stc_mean]
print(f"[cache] {len(missing_coefficients)} coefficient(s) to compute: "
      f"{[k[1] for k in missing_coefficients]}")


# %%
# we want to morphe each subject results to the average brain
# fsaverage source space (higher density for stats)
if missing_coefficients:
    fsaverage_src = mne.read_source_spaces(src_file_fsaverage)

    # The morph only depends on the anatomy, so it is computed once per subject and
    # re-used for both sessions of that subject.
    morphs = {}
    for subID in sorted({s for s, _ in sessions}):
        src_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-ico-5-src.fif')
        sub_src = mne.read_source_spaces(src_file)

        morphs[subID] = mne.compute_source_morph(
            sub_src,
            subject_from=f"sub-0{subID}",
            subject_to="fsaverage",
            src_to=fsaverage_src,
            subjects_dir=RECONDIR,
        )
        print(f"[morph] sub-0{subID} done")


# %%
# -----------------------------------------------------------------------------
# Per session: build one inverse operator, then invert every coefficient's z-map.
# -----------------------------------------------------------------------------
if missing_coefficients:
    # Running sum instead of keeping every session's stc: one fsaverage ico-5 stc is
    # ~68 MB (20484 vertices x n_times), so holding 56 sessions x 7 coefficients would
    # need tens of GB. Only the sum and the session count are kept per coefficient.
    stc_sum = {}
    n_sessions_used = {coefficient_name: 0 for coefficient_name in missing_coefficients}

    for subID, modality in sessions:

        sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')

        # load forward, bem, transform
        trans_file = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}', f'sub-0{subID}_ses-{modality}_task-NT_epochs-trans.fif')
        bem_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-5120-5120-5120-bem-sol.fif')
        fwd_file = os.path.join(sub_dir, f'sub-0{subID}_ico-5_ses-{modality}_fwd.fif')
        fwd_vector = mne.read_forward_solution(fwd_file)
        fwd_fixed = mne.convert_forward_solution(
            fwd_vector, surf_ori=True, force_fixed=True, use_cps=True
        )


        # check if trans file exists
        if trans_file is None or not os.path.exists(trans_file):
            print(f"Trans file for subject {subID} modality {modality} not found, skipping...")
            raise RuntimeError('NO TRANSFILE FOUND')

        # read epochs 
        all_epochs_filename = f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{suffix}"
        epoch_fif = os.path.join(sub_dir, f"{all_epochs_filename}.fif")

        if epoch_fif is None or not os.path.exists(epoch_fif):
            print(f"Epoch file for subject {subID} modality {modality} not found, skipping...")

        epochs = mne.read_epochs(epoch_fif, preload=True)
        apply_baseline_epochs(epochs, epoch_fif, baseline_window=(tmin_baseline, tmax_baseline))
        epochs_clean = epochs["analysis_excluded==False"]   
        epochs_clean = epochs_clean["drop_epoch==False"]   
        if len(epochs_clean) == 0:
            print(f"sub {subID} ses {modality} no epochs left after cleaning, skipping")
            continue

        evoked_clean = epochs_clean.average()

        # load covariance matrix
        noise_cov_reg_file = os.path.join(sub_dir, f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_noise-cov-baseline-{noise_tmin}to{noise_tmax}-covreg{suffix}.fif")
        noise_cov_reg = mne.read_cov(noise_cov_reg_file)

        # Fix the channel order: the Julia run writes z_vals rows in the order of
        # channels.txt (alphabetical, EEG only), while the epoch FIF carries the
        # acquisition order plus the non-EEG channels (CNAP/ECG/RESP/VEOG). Picking
        # by name gives an info whose rows line up with z_vals one-to-one.
        evoked_z_ref = evoked_clean.copy().pick(perm_channels)
        if evoked_z_ref.ch_names != perm_channels:
            raise RuntimeError(
                f"sub-0{subID} ses-{modality}: channel order not aligned after pick "
                f"(got {evoked_z_ref.ch_names[:5]}..., expected {perm_channels[:5]}...)."
            )
        print(f"[align] sub-0{subID} ses-{modality}: channel order aligned, "
              f"{len(evoked_z_ref.ch_names)} channels, {evoked_z_ref.ch_names[0]} ... {evoked_z_ref.ch_names[-1]}")

        # handle reference (has to be average reference, which we already applied before,
        # but for some reason it says custom reference) -- same as in the step002 scripts.
        # The copied info carries the projector over to the z-value evoked below.
        evoked_z_ref.set_eeg_reference(ref_channels="average", projection=True, verbose=False)

        # build the inverse operator / the spatial filters on the same channel
        # selection as the z-maps -- once per session, re-used for every coefficient.
        if INVERSE_METHOD == "mne":
            # Same forward/orientation handling as step002: dSPM on the free-orientation
            # forward (loose=0 collapses it to the cortical normal), MNE on the fixed one.
            if MNE_METHOD == "dSPM":
                fwd_mne, fixed = fwd_vector, "auto"
            else:
                fwd_mne, fixed = fwd_fixed, True
            inverse_operator = make_inverse_operator(
                evoked_z_ref.info, fwd_mne, noise_cov_reg,
                loose=MNE_LOOSE, depth=MNE_DEPTH, fixed=fixed,
            )
        else:
            # The beamformer needs the data covariance on top of the noise covariance.
            # It is a cached file per session; if it is missing, crash rather than
            # silently reconstructing from a different window.
            cov_reg_data_file = os.path.join(
                sub_dir,
                f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}"
                f"_hfreq-{h_freq}_noise-cov-data-{data_tmin}to{data_tmax}-covreg{suffix}.fif",
            )
            if not os.path.exists(cov_reg_data_file):
                raise FileNotFoundError(
                    f"sub-0{subID} ses-{modality}: data covariance for the LCMV filter not "
                    f"found: {cov_reg_data_file}"
                )
            data_cov_reg = mne.read_cov(cov_reg_data_file)
            # pick_ori=None on the fixed-orientation forward gives one scalar time course
            # per vertex, i.e. the same stc shape the MNE branch produces.
            filters = make_lcmv(
                evoked_z_ref.info,
                fwd_fixed,
                data_cov_reg,
                reg=LCMV_REG,
                noise_cov=noise_cov_reg,
                pick_ori=None,
                weight_norm=LCMV_WEIGHT_NORM,
                rank=None,
            )

        for coefficient_name in missing_coefficients:
            # construct evoked using z-values; tmin comes from the cluster run, not
            # from the epochs (the epochs start at -0.9 s, the z-map at -0.53 s).
            z_evoked = mne.EvokedArray(
                zmaps[coefficient_name]["z_vals"],
                evoked_z_ref.info.copy(),
                tmin=float(zmap_times[0]),
                nave=1,
            )

            if INVERSE_METHOD == "mne":
                stc_stat = apply_inverse(
                    z_evoked,
                    inverse_operator,
                    lambda2=1 / 9,
                    method=MNE_METHOD,
                )
            else:
                stc_stat = apply_lcmv(z_evoked, filters)

            stc_morphed = morphs[subID].apply(stc_stat)
            # accumulate in place; the first session seeds the sum
            if coefficient_name in stc_sum:
                stc_sum[coefficient_name].data += stc_morphed.data
            else:
                stc_sum[coefficient_name] = stc_morphed
            n_sessions_used[coefficient_name] += 1

        print(f"[inverse] sub-0{subID} ses-{modality}: {len(missing_coefficients)} coefficient(s) "
              f"done ({inverse_name})")


# %%
# average the stcs to a single mean stc per coefficient and cache it
if missing_coefficients:
    for coefficient_name in missing_coefficients:
        n_used = n_sessions_used[coefficient_name]
        if n_used == 0:
            raise RuntimeError(f"{coefficient_name[1]}: no stc to average.")
        stc_mean[coefficient_name] = stc_sum[coefficient_name] / n_used
        stc_mean[coefficient_name].save(stc_cache_path(coefficient_name), overwrite=True)
        print(f"[average] {coefficient_name[1]}: mean over {n_used} session(s) "
              f"-> {stc_cache_path(coefficient_name)}")

_not_reconstructed = [c for c in coefficient_names if c not in stc_mean]
if _not_reconstructed:
    raise RuntimeError(f"No group stc available for: {[k[1] for k in _not_reconstructed]}")

for coefficient_name, stc in stc_mean.items():
    print(f"[check] {coefficient_name[1]}: times {stc.times[0]:.3f}..{stc.times[-1]:.3f} s, "
          f"{len(stc.times)} samples, {stc.data.shape[0]} vertices")
    if not np.isclose(stc.times[0], zmap_times[0]) or len(stc.times) != len(zmap_times):
        raise RuntimeError(
            f"{coefficient_name[1]}: stc time axis {stc.times[0]}..{stc.times[-1]} "
            f"({len(stc.times)} samples) does not match the z-map."
        )


# %% 
# -----------------------------------------------------------------------------
# Plotting (mirrors plot_cluster_results_parcellated.py)
# -----------------------------------------------------------------------------
def _crop_white(img, threshold=250, pad=4):
    non_white = np.any(img < threshold, axis=2)
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - pad)
    rmax = min(img.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img.shape[1] - 1, cmax + pad)
    return img[rmin:rmax + 1, cmin:cmax + 1, :]


def _pad_to_height(img, target_h, fill=255):
    if img.shape[0] >= target_h:
        return img
    pad = np.full((target_h - img.shape[0], img.shape[1], img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)


def _pad_to_width(img, target_w, fill=255):
    if img.shape[1] >= target_w:
        return img
    pad = np.full((img.shape[0], target_w - img.shape[1], img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([img, pad], axis=1)


COEFF_NAME_OVERRIDES = {
    "intensity_level_centered": "intensity level",
    "response_decoded_Yes": "detection response",
    "modality_Mechanical": "modality",
    "intensity_level_centered_response_decoded_Yes": "intensity level x detection response",
    "modality_Mechanical_response_decoded_Yes": "modality x detection response",
    "modality_Mechanical_intensity_level_centered": "modality x intensity level",
    "modality_Mechanical_intensity_level_centered_response_decoded_Yes":
        "modality x intensity level x detection response",
}


def format_coefficient_name(name: str) -> str:
    cleaned = re.sub(r"^\d+[_-]*", "", name)
    if cleaned in COEFF_NAME_OVERRIDES:
        return COEFF_NAME_OVERRIDES[cleaned]
    for suffix_ in ("_centered", "_scaled", "_z"):
        if cleaned.endswith(suffix_):
            cleaned = cleaned[: -len(suffix_)]
            break
    cleaned = cleaned.replace("_", " ").strip()
    return " ".join(cleaned.split())


def significance_tag(key) -> str:
    """Short tag with the smallest cluster-mass p of the coefficient."""
    return f"p = {zmaps[key]['p_min']:.3f}"


def coefficient_label(key, width: int = 22) -> str:
    """Grid row label: wrapped display name, significance tag on the last line."""
    wrapped = "\n".join(textwrap.wrap(format_coefficient_name(key[1]), width=width))
    return f"{wrapped}\n({significance_tag(key)})"


def render_brain_snapshot(stc, t_idx, clim, colormap):
    """One composite screenshot: both hemispheres, lateral + medial views."""
    brain = stc.plot(
        subject="fsaverage",
        subjects_dir=RECONDIR,
        hemi="split",
        views=["lateral", "medial"],
        colorbar=False,
        clim=clim,
        colormap=colormap,
        background="white",
        time_viewer=False,
        show_traces=False,
        size=(1000, 900),
        initial_time=float(stc.times[t_idx]),
    )
    img = brain.screenshot()
    brain.close()
    return _crop_white(img)


# The reconstructed values are MNE source amplitudes of a z-topography, so their
# unit is arbitrary and one symmetric scale shared by all coefficients and
# latencies keeps the panels comparable.
#
# Two renderings are produced from the same stcs:
#  - "signed": the fixed-orientation estimate as-is (mne_analyze, pos_lims). The
#    source sign follows the cortical normal, so opposing sulcal banks flip sign
#    and the maps look striped -- kept for consistency with the parcel figures.
#  - "unsigned": |estimate| on a sequential map. The sign carries no interpretable
#    meaning here (the direction of the effect is already in the sensor z-map), so
#    this is the readable version.
_snap_vals = {c: stc_mean[c].data[:, zmaps[c]["snap_idx"]].ravel() for c in coefficient_names}


def _pooled_p995(names):
    return float(np.percentile(np.abs(np.concatenate([_snap_vals[c] for c in names])), 99.5))


# One shared scale for the per-coefficient figures and the summary grid, pooled over
# all coefficients. The non-significant coefficients are genuinely much weaker than the
# significant ones, so their panels stay faint -- that is the intended comparison.
V_MAX = _pooled_p995(coefficient_names)

DISPLAY_MODES = {
    "signed": dict(
        colormap="mne_analyze",
        clim=dict(kind="value", pos_lims=[0.0, V_MAX / 2.0, V_MAX]),
        label="source estimate of z-map (a.u.)",
        title="source-reconstructed z-map",
        transform=lambda stc: stc,
    ),
    "unsigned": dict(
        colormap="hot",
        clim=dict(kind="value", lims=[0.0, V_MAX / 2.0, V_MAX]),
        label="|source estimate| (a.u.)",
        title="source-reconstructed |z-map|",
        transform=lambda stc: stc.__class__(
            np.abs(stc.data), vertices=stc.vertices, tmin=stc.tmin, tstep=stc.tstep,
            subject=stc.subject,
        ),
    ),
}

print(f"[plot] shared colour scale: {V_MAX:.3g} "
      f"(99.5th percentile over all {len(coefficient_names)} coefficients)")
for _c in coefficient_names:
    print(f"[plot]   {_c[1]}: own 99.5th percentile = "
          f"{np.percentile(np.abs(_snap_vals[_c]), 99.5):.3g}")


for mode_name, mode in DISPLAY_MODES.items():
    coeff_plot_data = []

    for coefficient_name in coefficient_names:
        stc = mode["transform"](stc_mean[coefficient_name])
        snap_idx = zmaps[coefficient_name]["snap_idx"]
        is_peak = zmaps[coefficient_name]["is_peak"]
        panels = [render_brain_snapshot(stc, ti, mode["clim"], mode["colormap"]) for ti in snap_idx]

        # Pad panels to a common size so they tile cleanly.
        row_h = max(p.shape[0] for p in panels)
        col_w = max(p.shape[1] for p in panels)
        panels = [_pad_to_width(_pad_to_height(p, row_h), col_w) for p in panels]

        n_cols = len(panels)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.4), squeeze=False)
        for ax, ti, panel, peak in zip(axes[0], snap_idx, panels, is_peak):
            ax.imshow(panel)
            # Bold title = cluster peak, regular title = latency chosen without a peak.
            ax.set_title(f"{zmap_times[ti] * 1000:.0f} ms", fontsize=12,
                         fontweight="bold" if peak else "normal")
            ax.axis("off")
        fig.suptitle(
            f"{format_coefficient_name(coefficient_name[1])} ({significance_tag(coefficient_name)})"
            f" — {mode['title']} ({INV_LABEL})",
            y=1.02,
        )

        fig.subplots_adjust(left=0.02, right=0.87, bottom=0.05, top=0.90, wspace=0.02)
        cax = fig.add_axes([0.89, 0.15, 0.012, 0.7])
        plot_brain_colorbar(cax, mode["clim"], colormap=mode["colormap"], transparent=False,
                            orientation="vertical", label=mode["label"])

        out_pdf = os.path.join(
            figuredir,
            f"source_zmap__run-{coefficient_name[0][:15]}__coef-{coefficient_name[1]}"
            f"__{INV_TAG}__{mode_name}__snapshots.pdf",
        )
        # pad_inches keeps the colorbar label inside the cropped bounding box.
        fig.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.4)
        plt.close(fig)
        print(f"[{mode_name}][{coefficient_name[1]}] saved {out_pdf}")

        coeff_plot_data.append({
            "name": coefficient_name,
            "display_name": coefficient_label(coefficient_name),
            "panels": panels,
            "snap_idx": snap_idx,
            "is_peak": is_peak,
        })

    # -------------------------------------------------------------------------
    # All-coefficients grid: rows = coefficients, cols = snapshot times.
    # -------------------------------------------------------------------------
    n_rows = len(coeff_plot_data)
    n_cols = max(len(c["panels"]) for c in coeff_plot_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.7 * n_rows), squeeze=False)

    for row_idx, entry in enumerate(coeff_plot_data):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(entry["panels"]):
                continue
            ax.imshow(entry["panels"][col_idx])
            # Rows have their own latencies, so every panel carries its own title.
            ax.set_title(f"{zmap_times[entry['snap_idx'][col_idx]] * 1000:.0f} ms", fontsize=11,
                         fontweight="bold" if entry["is_peak"][col_idx] else "normal")
        # Row label on the left.
        axes[row_idx, 0].set_ylabel(entry["display_name"], rotation=90, fontsize=10, labelpad=10)
        axes[row_idx, 0].axis("on")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        for spine in axes[row_idx, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"Source-reconstructed cluster-mass z-maps ({INV_LABEL}) — {mode_name} "
        f"(bold = cluster peak), all coefficients on one scale",
        y=1.0,
    )
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.06, top=0.94, wspace=0.02, hspace=0.15)
    cax = fig.add_axes([0.91, 0.2, 0.010, 0.6])
    plot_brain_colorbar(cax, mode["clim"], colormap=mode["colormap"], transparent=False,
                        orientation="vertical", label=mode["label"])

    out_all = os.path.join(
        figuredir, f"source_zmap__all_coefficients__{INV_TAG}__{mode_name}__snapshots.pdf"
    )
    fig.savefig(out_all, dpi=200, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"[{mode_name}] saved all-coefficients grid: {out_all}")

# %%
