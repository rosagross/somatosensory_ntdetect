######################################
# Per-session alpha-band source reconstruction.
#
# - Uses the extracted alpha-band epochs from step007_extract_freq_bands.py
# - Runs the inverse method set in alpha_source_common.py (MNE or fixed-orientation LCMV)
# - Computes the trial-mean alpha envelope and debiased ITPC for Yes and No trials
# - Regresses the source-space |ERP| out of the ITPC, per vertex (see regress_out_erp_source)
# - Morphs them to fsaverage and saves one source estimate per condition key
#
# One (modality, subject) session per call: select it with --session-index (SLURM array
# task) or with --sub / --modalities. --count prints the number of sessions.
######################################

# %%
import argparse
import os
import mne
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_epochs, make_lcmv
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs
from scipy.signal import hilbert as scipy_hilbert

from alpha_source_common import (
    RECONDIR, preproc_suffix, alpha_suffix, analysis_tmin, analysis_tmax, data_cov_tmin, data_cov_tmax,
    noise_tmin, noise_tmax, _hilbert_fmin, h_freq, l_freq, l_freq_orig,
    inverse_method, reg, pick_ori, weight_norm, mne_method, loose, depth,
    fsaverage_stc_stem, modalities, session_dir, session_list,
)
from general_helpers import apply_baseline_epochs  # WDDIR is put on sys.path by alpha_source_common

src_file_fsaverage = os.path.join(RECONDIR, "fsaverage_data", "bem", "fsaverage-ico-5-src.fif")
fsaverage_src = mne.read_source_spaces(src_file_fsaverage)

morphs = {}


def build_alpha_epoch_path(sub_dir, subid, modality):
    base = (
        f"sub-0{subid}_ses-{modality}_task-NT_"
        f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}"
    )
    return os.path.join(sub_dir, f"{base}_epochs{alpha_suffix}.fif")


def ensure_eeg_reference(evoked):
    evoked.set_eeg_reference(ref_channels="average", projection=True, verbose=False)
    return evoked


def load_alpha_epochs(alpha_epoch_fif):
    if not os.path.exists(alpha_epoch_fif):
        return None
    epochs = mne.read_epochs(alpha_epoch_fif, preload=True, verbose=False)
    epochs = epochs["analysis_excluded==False"]
    epochs = epochs["drop_epoch==False"]
    # No crop here — full epoch is kept so surrounding data serves as natural
    # padding when computing the Hilbert transform after beamforming.
    if len(epochs) == 0:
        return None
    return epochs


# Pre-cue baseline of the broadband ERP used for the ITPC regression (one window for both modalities).
ERP_BASELINE = (-0.85, -0.75)
# The first samples of the dITPC time axis are Hilbert-edge contaminated and are left out of the fit
# (the cleaned trace is still returned for every sample).
N_EDGE_SAMPLES = 5


def build_broadband_epoch_path(sub_dir, subid, modality):
    base = (
        f"sub-0{subid}_ses-{modality}_task-NT_"
        f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}"
    )
    return os.path.join(sub_dir, f"{base}_epochs{preproc_suffix}.fif")


def load_broadband_epochs(broadband_fif, alpha_epochs):
    """
    Broadband (not band-passed) epochs of the same session, with the same trial selection as the
    alpha epochs. Trial count, response order and channels must match the alpha epochs exactly.
    """
    if not os.path.exists(broadband_fif):
        raise FileNotFoundError(f"Broadband epochs not found: {broadband_fif}")
    bb = mne.read_epochs(broadband_fif, preload=True, verbose=False)
    bb = bb["analysis_excluded==False"]
    bb = bb["drop_epoch==False"]
    resp_bb = bb.metadata["response_decoded"].to_numpy()
    resp_alpha = alpha_epochs.metadata["response_decoded"].to_numpy()
    if len(resp_bb) != len(resp_alpha) or not np.array_equal(resp_bb, resp_alpha):
        raise ValueError(
            f"Broadband epochs ({len(resp_bb)}) and alpha epochs ({len(resp_alpha)}) differ in "
            f"trial count or response order: {broadband_fif}"
        )
    if bb.ch_names != alpha_epochs.ch_names:
        raise ValueError(f"Channel names of broadband and alpha epochs differ: {broadband_fif}")
    # Baseline before the reference; both are linear, so the order does not change the result.
    apply_baseline_epochs(bb, broadband_fif, baseline_window=ERP_BASELINE)
    return ensure_eeg_reference(bb)


def abs_erp_source(bb_subset, apply_to_evoked, times_ref):
    """
    |ERP| in source space for one response: trial-mean broadband evoked, projected with the same
    spatial filter / inverse as the alpha data, absolute value, cropped to the analysis window.
    Shape (n_sources, n_times); the time axis must equal times_ref (the ITPC time axis).
    """
    stc = apply_to_evoked(bb_subset.average())
    t_mask = (stc.times >= analysis_tmin) & (stc.times <= analysis_tmax)
    if t_mask.sum() != len(times_ref) or not np.allclose(stc.times[t_mask], times_ref):
        raise ValueError(
            f"ERP time axis ({int(t_mask.sum())} samples) does not match the ITPC time axis "
            f"({len(times_ref)} samples)."
        )
    return np.abs(stc.data[:, t_mask])


def regress_out_erp_source(itpc_no, itpc_yes, erp_no, erp_yes):
    """
    Remove |ERP| from the debiased ITPC, independently for every source (row). All inputs have
    shape (n_sources, n_times).

    Per source one OLS fit  ITPC = a + b * |ERP|  over the Yes and No time points stacked, so b is
    shared by both responses and a Yes-No ITPC difference only survives to the extent that
    |ERP| cannot explain it. The cleaned trace is
        ITPC_clean(t) = ITPC(t) - b * (|ERP|(t) - mean(|ERP|)),
    which keeps the ITPC level (a mean-zero residual would erase the Yes-No level difference).
    The mean is taken over the fitted samples.
    """
    fit = slice(N_EDGE_SAMPLES, None)
    x = np.concatenate([erp_no[:, fit], erp_yes[:, fit]], axis=1)
    y = np.concatenate([itpc_no[:, fit], itpc_yes[:, fit]], axis=1)
    x_mean = x.mean(axis=1, keepdims=True)
    var_x = np.sum((x - x_mean) ** 2, axis=1)
    if np.any(var_x == 0):
        raise ValueError(f"|ERP| is constant at {int(np.sum(var_x == 0))} sources, regression undefined.")
    slope = np.sum((x - x_mean) * (y - y.mean(axis=1, keepdims=True)), axis=1) / var_x
    clean_no = itpc_no - slope[:, None] * (erp_no - x_mean)
    clean_yes = itpc_yes - slope[:, None] * (erp_yes - x_mean)
    if not (np.all(np.isfinite(clean_no)) and np.all(np.isfinite(clean_yes))):
        raise ValueError("Non-finite value after regressing out |ERP|.")
    return clean_no, clean_yes


def compute_alpha_covariances(epochs):
    # Compute the data and noise covariance directly from the alpha-filtered epochs so
    # the LCMV spatial filter is tuned to the alpha rhythm rather than to the broadband
    # (ERP / low-frequency / 1-f) structure. Covariance is estimated on the real-valued
    # alpha-filtered time series (pre-Hilbert), which is exactly what these epochs hold.
    # method/rank match the project convention (helper_functions_eeg.compute_cov) and
    # make_lcmv(rank=None) below.
    data_cov = mne.compute_covariance(
        epochs, tmin=data_cov_tmin, tmax=data_cov_tmax, method="auto", rank=None, verbose=False
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=noise_tmin, tmax=noise_tmax, method="auto", rank=None, verbose=False
    )
    return data_cov, noise_cov


def get_morph(subid):
    if subid in morphs:
        return morphs[subid]
    src_file = os.path.join(RECONDIR, f"sub-0{subid}", "bem", f"sub-0{subid}-ico-5-src.fif")
    sub_src = mne.read_source_spaces(src_file)
    morphs[subid] = mne.compute_source_morph(
        sub_src,
        subject_from=f"sub-0{subid}",
        subject_to="fsaverage",
        src_to=fsaverage_src,
        subjects_dir=RECONDIR,
    )
    return morphs[subid]

def debiased_itpc(phase_acc, n):
    """
    Debiased squared ITPC (PPC) from the summed unit phase vectors of n trials.

    Same definition as the sensor-space ITPC in step008_7: (n * R^2 - 1) / (n - 1), with R
    the resultant vector length. Plain R is biased upward at low trial counts, and Yes and
    No have unequal counts. The value can be slightly negative.
    """
    if n < 2:
        raise ValueError(f"Debiased ITPC needs at least 2 trials, got {n}.")
    itpc = np.abs(phase_acc / n)
    return (n * itpc ** 2 - 1) / (n - 1)


def accumulate_envelope_itpc_lcmv(ep_subset, filters):
    """
    Beamform full epoch, mirror-pad → Hilbert → trim, crop to analysis window.

    Returns the trial-mean envelope and the debiased squared ITPC (see debiased_itpc).
    """
    env_acc   = None
    phase_acc = None
    t_mask    = None
    n = 0
    for stc_ep in apply_lcmv_epochs(
        ep_subset, filters, return_generator=True, verbose=False
    ):
        if t_mask is None:
            t_mask = (stc_ep.times >= analysis_tmin) & (stc_ep.times <= analysis_tmax)
            n_src_local = stc_ep.data.shape[0]
            n_times_win = int(t_mask.sum())
            env_acc   = np.zeros((n_src_local, n_times_win), dtype=np.float64)
            phase_acc = np.zeros((n_src_local, n_times_win), dtype=np.complex128)
        _sfreq = round(1.0 / stc_ep.tstep)
        _n_pad = int(3 * _sfreq / _hilbert_fmin)
        _padded = np.pad(stc_ep.data, ((0, 0), (_n_pad, _n_pad)), mode='reflect')
        analytic = scipy_hilbert(_padded, axis=-1)[:, _n_pad:-_n_pad][:, t_mask]
        env_acc   += np.abs(analytic)
        phase_acc += np.exp(1j * np.angle(analytic))
        n += 1
    return env_acc / n, debiased_itpc(phase_acc, n)

def accumulate_envelope_itpc_mne(ep_subset, inverse_operator, lambda2, mne_method):
    """
    Apply the inverse to the full epoch, mirror-pad → Hilbert → trim, crop to analysis window.

    Returns the trial-mean envelope and the debiased squared ITPC (see debiased_itpc).
    """
    env_acc   = None
    phase_acc = None
    t_mask    = None
    n = 0
    if mne_method == "dSPM": 
        pick_ori = None #'normal'
    else:
        pick_ori = None

    for stc_ep in apply_inverse_epochs(ep_subset, inverse_operator, lambda2, method=mne_method, pick_ori=pick_ori,
                                        return_generator=True, verbose=False):
        if t_mask is None:
            t_mask = (stc_ep.times >= analysis_tmin) & (stc_ep.times <= analysis_tmax)
            n_src_local = stc_ep.data.shape[0]
            n_times_win = int(t_mask.sum())
            env_acc   = np.zeros((n_src_local, n_times_win), dtype=np.float64)
            phase_acc = np.zeros((n_src_local, n_times_win), dtype=np.complex128)
        _sfreq = round(1.0 / stc_ep.tstep)
        _n_pad = int(3 * _sfreq / _hilbert_fmin)
        _padded = np.pad(stc_ep.data, ((0, 0), (_n_pad, _n_pad)), mode='reflect')
        analytic = scipy_hilbert(_padded, axis=-1)[:, _n_pad:-_n_pad][:, t_mask]
        env_acc   += np.abs(analytic)
        phase_acc += np.exp(1j * np.angle(analytic))
        n += 1
    return env_acc / n, debiased_itpc(phase_acc, n)


def process_session(modality, subid):
    """Source reconstruction and fsaverage morph for one (modality, subject) session."""
    sub_dir = session_dir(subid, modality)

    trans_file = os.path.join(
        sub_dir,
        f"sub-0{subid}_ses-{modality}_task-NT_epochs-trans.fif",
    )
    bem_file = os.path.join(RECONDIR, f"sub-0{subid}", "bem", f"sub-0{subid}-5120-5120-5120-bem-sol.fif")
    src_file = os.path.join(RECONDIR, f"sub-0{subid}", "bem", f"sub-0{subid}-ico-5-src.fif")
    alpha_epoch_fif = build_alpha_epoch_path(sub_dir, subid, modality)

    if not os.path.exists(trans_file):
        print(f"Trans file for subject {subid} modality {modality} not found, skipping...")
        return
    if not os.path.exists(bem_file):
        print(f"BEM file for subject {subid} modality {modality} not found, skipping...")
        return
    if not os.path.exists(src_file):
        print(f"Source space file for subject {subid} modality {modality} not found, skipping...")
        return

    epochs = load_alpha_epochs(alpha_epoch_fif)
    if epochs is None:
        print(f"Alpha epochs not found or empty for subject {subid} modality {modality}, skipping...")
        return

    if len(epochs) == 0:
        print(f"No alpha epochs left after cleaning for subject {subid} modality {modality}, skipping...")
        return

    broadband_epochs = load_broadband_epochs(
        build_broadband_epoch_path(sub_dir, subid, modality), epochs
    )

    epochs = ensure_eeg_reference(epochs)
    evoked_both = ensure_eeg_reference(epochs.average())

    # Compute alpha-band covariances from the cleaned, average-referenced alpha epochs
    # (all trials, before the Yes/No split), so the beamformer is tuned to the alpha band.
    data_cov, noise_cov = compute_alpha_covariances(epochs)

    fwd_file = os.path.join(sub_dir, f"sub-0{subid}_ico-5_ses-{modality}_fwd.fif")
    fwd = mne.read_forward_solution(fwd_file, verbose=False)

    # Filter to signal trials (no catch) and split by detection response.
    # Use full-range epochs so the beamformed signal covers the whole epoch —
    # surrounding data outside the analysis window absorbs Hilbert edge transients.
    if "intensity_level" in epochs.metadata.columns:
        epochs_valid = epochs["intensity_level != 0"]
    else:
        epochs_valid = epochs.copy()
    yes_epochs = epochs_valid["response_decoded == 'Yes'"]
    no_epochs  = epochs_valid["response_decoded == 'No'"]
    # Same trial selection on the broadband epochs (identical metadata order, checked on load).
    if "intensity_level" in broadband_epochs.metadata.columns:
        bb_valid = broadband_epochs["intensity_level != 0"]
    else:
        bb_valid = broadband_epochs.copy()
    yes_bb = bb_valid["response_decoded == 'Yes'"]
    no_bb  = bb_valid["response_decoded == 'No'"]
    if len(yes_bb) != len(yes_epochs) or len(no_bb) != len(no_epochs):
        raise ValueError(
            f"sub-0{subid} {modality}: broadband Yes/No counts ({len(yes_bb)}/{len(no_bb)}) differ "
            f"from alpha ({len(yes_epochs)}/{len(no_epochs)})."
        )

    if len(yes_epochs) == 0 or len(no_epochs) == 0:
        print(f"Missing Yes/No alpha epochs for sub-0{subid} {modality}, skipping...")
        return

    fwd_fixed = mne.convert_forward_solution(
        fwd,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
    )

    if inverse_method == "lcmv":

        filters = make_lcmv(
            evoked_both.info,
            fwd_fixed,
            data_cov,
            reg=reg,
            noise_cov=noise_cov,
            pick_ori=pick_ori,
            weight_norm=weight_norm,
            rank=None,
        )

        # Template STC: provides vertices/tmin/tstep for constructing new STC objects.
        stc_template = apply_lcmv(evoked_both, filters)


        env_data_yes, itpc_data_yes = accumulate_envelope_itpc_lcmv(yes_epochs, filters)
        env_data_no,  itpc_data_no  = accumulate_envelope_itpc_lcmv(no_epochs, filters)

        t_ref = stc_template.times[(stc_template.times >= analysis_tmin) & (stc_template.times <= analysis_tmax)]
        erp_fn = lambda ev: apply_lcmv(ev, filters)
        itpc_data_no, itpc_data_yes = regress_out_erp_source(
            itpc_data_no, itpc_data_yes,
            abs_erp_source(no_bb, erp_fn, t_ref), abs_erp_source(yes_bb, erp_fn, t_ref),
        )


    elif inverse_method == "mne":
        # Create inverse operator

        if mne_method == "dSPM":
            fwd_mne = fwd
            fixed = 'auto'
        else:
            fwd_mne = fwd_fixed
            fixed = True

        inverse_operator = make_inverse_operator(
            evoked_both.info, fwd_mne, noise_cov, loose=loose, depth=depth, fixed=fixed
        )

        # placeholder for creating new stc from itpc/envelope values later
        stc_template = apply_inverse(evoked_both, inverse_operator)


        snr = 3.0
        lambda2 = 1.0 / snr ** 2

        env_data_yes, itpc_data_yes = accumulate_envelope_itpc_mne(yes_epochs, inverse_operator, lambda2, mne_method)
        env_data_no,  itpc_data_no  = accumulate_envelope_itpc_mne(no_epochs, inverse_operator, lambda2, mne_method)

        t_ref = stc_template.times[(stc_template.times >= analysis_tmin) & (stc_template.times <= analysis_tmax)]
        erp_fn = lambda ev: apply_inverse(ev, inverse_operator, lambda2, method=mne_method, pick_ori=None, verbose=False)
        itpc_data_no, itpc_data_yes = regress_out_erp_source(
            itpc_data_no, itpc_data_yes,
            abs_erp_source(no_bb, erp_fn, t_ref), abs_erp_source(yes_bb, erp_fn, t_ref),
        )


    def _make_stc(data):
        return mne.SourceEstimate(
            data,
            vertices=stc_template.vertices,
            tmin=analysis_tmin,
            tstep=stc_template.tstep,
            subject=f"sub-0{subid}",
        )

    stc_envelope_yes = _make_stc(env_data_yes)
    stc_envelope_no  = _make_stc(env_data_no)
    stc_itpc_yes     = _make_stc(itpc_data_yes)
    stc_itpc_no      = _make_stc(itpc_data_no)

    # Morph to fsaverage.
    morph = get_morph(subid)
    stc_fs = {
        "envelope_yes": morph.apply(stc_envelope_yes),
        "envelope_no":  morph.apply(stc_envelope_no),
        "itpc_yes":     morph.apply(stc_itpc_yes),
        "itpc_no":      morph.apply(stc_itpc_no),
    }

    # Save individual morphed STCs for the group stage and downstream cluster permutation stats.
    for key, stc_morphed in stc_fs.items():
        stc_morphed.save(fsaverage_stc_stem(subid, modality, key), overwrite=True)

    print("subject", subid, modality)


def select_sessions(args):
    sessions = session_list()
    if args.session_index is not None:
        if args.sub or args.modalities:
            raise ValueError("--session-index cannot be combined with --sub or --modalities.")
        if not 0 <= args.session_index < len(sessions):
            raise ValueError(f"--session-index {args.session_index} out of range for {len(sessions)} sessions.")
        return [sessions[args.session_index]]
    if args.sub:
        sessions = [s for s in sessions if s[1] in args.sub]
    if args.modalities:
        sessions = [s for s in sessions if s[0] in args.modalities]
    if not sessions:
        raise ValueError("No sessions selected (check --sub / --modalities and the exclusion list).")
    return sessions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-session alpha-band source reconstruction.")
    parser.add_argument("--session-index", type=int, default=None,
                        help="Index into the ordered session list (SLURM array task id).")
    parser.add_argument("--sub", nargs="+", type=int, default=None, help="Subject ids, e.g. 15 16.")
    parser.add_argument("--modalities", nargs="+", choices=modalities, default=None)
    parser.add_argument("--count", action="store_true", help="Print the number of sessions and exit.")
    args = parser.parse_args()

    if args.count:
        print(len(session_list()))
    else:
        for modality, subid in select_sessions(args):
            process_session(modality, subid)

# %%
