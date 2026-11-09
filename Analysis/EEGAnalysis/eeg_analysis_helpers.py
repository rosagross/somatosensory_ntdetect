import argparse
import numpy as np
import os
import mne 
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# Order matters: check longer suffix first to avoid substring conflicts.
_BASELINE_SUFFIX_MAP = [
    ("_preprestim", (-0.8, -0.7)),
    ("preprestim", (-0.8, -0.7)),
    ("_prestim", (-0.5, -0.43)),
    ("prestim", (-0.5, -0.43)),
]
_DEFAULT_BASELINE = (-0.1, 0.0)

def parse_subjects():
    """
    Parse command line arguments to get subject IDs.
    If no subject IDs are provided, process all subjects from 13 to 52.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub",
        type=str,
        nargs="+",
        help="Subject IDs to process (e.g., '01 02 15' for subjects 1, 2, and 15)",
        required=False,
    )
    args = parser.parse_args()
    if args.sub:
        subIDs = args.sub
    else:
        # process all subjects from subject 13 (no pilot subjects)
        subIDs = map(str, range(13, 53))

    return subIDs


def load_band_epochs_and_metadata(sub_dir, base_name, band_name, subID, modality, file_suffix_epochs, file_suffix_meta):
    """
    Load band-pass epochs and metadata, apply filters to match corrected epochs,
    and enforce strict alignment.
    """
    epoch_fif = os.path.join(
        sub_dir, f"{base_name}_epochs{file_suffix_epochs}_{band_name}.fif"
    )
    meta_tsv = os.path.join(sub_dir, f"{base_name}_allmetadata{file_suffix_meta}.tsv")

    if not os.path.exists(epoch_fif):
        print(f"Band epochs not found: {epoch_fif}")
        return None, None
    if not os.path.exists(meta_tsv):
        print(f"Metadata not found: {meta_tsv}")
        return None, None

    epochs = mne.read_epochs(epoch_fif, preload=True, verbose=False)
    metadata_df = pd.read_csv(meta_tsv, sep="\t")

    # Match corrected epochs: drop bad epochs, analysis-excluded, and catch trials.
    metadata_df = metadata_df[
        (metadata_df["drop_epoch"] == False)
        & (metadata_df["analysis_excluded"] == False)
        & (metadata_df["intensity_level"] != 0)
        & (metadata_df["response_decoded"] != np.nan)
    ].reset_index(drop=True)

    # Remove trials with missing response.
    metadata_df = metadata_df[metadata_df["response_decoded"].notna()]
    epochs = epochs['response_decoded=="Yes" or response_decoded=="No"']

    if len(epochs) != len(metadata_df):
        raise ValueError(
            f"Metadata rows ({len(metadata_df)}) and epochs ({len(epochs)}) differ "
            f"for subject {subID} modality {modality} band {band_name}."
        )

    return epochs, metadata_df


def circular_mean(angles, axis=None):
    """
    Circular mean for phase angles (radians).
    """
    return np.angle(np.mean(np.exp(1j * angles), axis=axis))


def circular_std(angles, axis=None):
    """
    Circular standard deviation for phase angles (radians).
    """
    R = np.abs(np.mean(np.exp(1j * angles), axis=axis))
    return np.sqrt(-2 * np.log(np.clip(R, 1e-8, 1.0)))


def rayleigh_test(angles_rad):
    """
    Rayleigh test of circular uniformity.

    The null is a uniform distribution on the circle; a small p means the angles cluster around
    some preferred direction. Returns (n, mean_angle, R, z, p) so that callers get the descriptive
    quantities (the mean angle and the resultant length R, which is what a mean-vector arrow is
    drawn from) off the same call as the test statistic.

    angles_rad : 1-D array of angles in radians. Non-finite entries are dropped.

    R is the mean resultant length in [0, 1] (0 = perfectly uniform, 1 = all angles identical) and
    z = n * R**2 is the Rayleigh statistic. The p-value is the standard finite-sample expansion
    used in circular statistics texts (Zar), which is markedly more accurate than exp(-z) at the
    session counts this project works with. Fewer than two angles gives NaNs rather than a
    fabricated result.
    """
    angles_rad = np.asarray(angles_rad, dtype=float).ravel()
    angles_rad = angles_rad[np.isfinite(angles_rad)]
    n = int(angles_rad.size)
    if n < 2:
        return n, np.nan, np.nan, np.nan, np.nan

    resultant = np.mean(np.exp(1j * angles_rad))
    mean_angle = float(np.angle(resultant))
    R = float(np.abs(resultant))
    z = n * (R ** 2)
    p = np.exp(-z) * (
        1
        + (2 * z - z**2) / (4 * n)
        - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    )
    return n, mean_angle, R, float(z), float(np.clip(p, 0.0, 1.0))


# MNE's CSD returns V/m^2 for data in V. The FIFs hold microvolts (despite FIFF_UNIT_V), so the
# raw CSD output is uV/m^2; 1 m^2 = 1e4 cm^2, so uV/cm^2 = uV/m^2 * 1e-4.
CSD_UVM2_TO_UVCM2 = 1e-4


def apply_surface_laplacian(epochs):
    """
    Compute a surface Laplacian (CSD) on EEG channels before channel-wise metrics.

    This reduces volume-conduction effects and makes each channel signal more local.
    Expects EEG data in microvolts and returns the CSD in uV/cm^2 (see CSD_UVM2_TO_UVCM2).
    """
    eeg_epochs = epochs.copy().pick("eeg")
    if len(eeg_epochs.ch_names) < 2:
        raise ValueError("Need at least 2 EEG channels to compute Laplacian (CSD).")
    csd_epochs = mne.preprocessing.compute_current_source_density(eeg_epochs)
    # After the CSD the channel type is "csd", not "eeg".
    csd_epochs.apply_function(lambda x: x * CSD_UVM2_TO_UVCM2, picks="csd", channel_wise=False)
    return csd_epochs


def bonferroni_correct_pvals(pvals):
    """Bonferroni-correct the finite p-values in a vector."""
    pvals_arr = np.asarray(pvals, dtype=float)
    corrected = np.full_like(pvals_arr, np.nan, dtype=float)
    finite_mask = np.isfinite(pvals_arr)
    if np.any(finite_mask):
        corrected[finite_mask] = multipletests(pvals_arr[finite_mask], method="bonferroni")[1]
    return corrected


def fdr_correct_pvals(pvals):
    """Benjamini-Hochberg FDR-correct the finite p-values in a vector."""
    pvals_arr = np.asarray(pvals, dtype=float)
    corrected = np.full_like(pvals_arr, np.nan, dtype=float)
    finite_mask = np.isfinite(pvals_arr)
    if np.any(finite_mask):
        corrected[finite_mask] = multipletests(pvals_arr[finite_mask], method="fdr_bh")[1]
    return corrected


def build_yes_minus_no_topography_summary(
    yes_subject_maps,
    no_subject_maps,
    times_vec,
    mean_win,
    topo_channel_names,
):
    """
    Build a subject-first yes-minus-no topography summary for one panel.

    yes/no_subject_maps : lists of (n_channels, n_times) per-subject mean envelope maps.
    The difference is averaged over subjects first, then the strongest absolute deflection
    inside mean_win picks the channel and latency the topography is drawn at.
    """
    if times_vec is None or len(times_vec) == 0:
        return {"has_data": False}
    if topo_channel_names is None:
        return {"has_data": False}
    if len(yes_subject_maps) == 0 or len(no_subject_maps) == 0:
        return {"has_data": False}

    yes_stack = np.stack(yes_subject_maps, axis=0)
    no_stack = np.stack(no_subject_maps, axis=0)
    if yes_stack.shape != no_stack.shape:
        raise ValueError(
            "Yes/no Hilbert envelope stacks have mismatched shapes: "
            f"yes={yes_stack.shape}, no={no_stack.shape}"
        )

    diff_mean = np.mean(yes_stack - no_stack, axis=0)
    tmin, tmax = mean_win
    time_mask = (times_vec >= tmin) & (times_vec <= tmax)
    if not np.any(time_mask):
        raise ValueError(f"Mean window {mean_win} does not overlap the time axis.")

    diff_window = diff_mean[:, time_mask]
    peak_flat_idx = int(np.argmax(np.abs(diff_window)))
    peak_ch_idx, peak_time_rel_idx = np.unravel_index(peak_flat_idx, diff_window.shape)
    peak_time = float(times_vec[time_mask][peak_time_rel_idx])
    peak_time_idx = int(np.argmin(np.abs(times_vec - peak_time)))
    peak_channel = topo_channel_names[peak_ch_idx]

    return {
        "has_data": True,
        "topo_values": diff_mean[:, peak_time_idx],
        "peak_time": peak_time,
        "peak_channel": peak_channel,
        "peak_channel_idx": peak_ch_idx,
        "peak_value": float(diff_mean[peak_ch_idx, peak_time_idx]),
        "mean_win": mean_win,
        "times": times_vec,
    }


def annotate_significance(ax, x1, x2, y, text, line_height=0.02):
    """
    Add a simple significance bar with text between two x positions.
    """
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], color="black")
    ax.text((x1 + x2) / 2, y + line_height, text, ha="center", va="bottom")


def add_event_lines(ax, times, cue_time=-0.4):
    """
    Add baseline and stimulus markers if their timepoints fall within the plotted range.
    cue_time controls the visual cue marker (default -0.4 s).
    """
    tmin = float(np.min(times))
    tmax = float(np.max(times))

    def in_range(t):
        return tmin <= t <= tmax

    # Baseline start/stop.
    if in_range(-0.75):
        ax.axvline(-0.75, color="gray", linestyle="--", linewidth=1, alpha=0.4)

    # Visual cue stimulus onset.
    if in_range(cue_time):
        ax.axvline(cue_time, color="black", linestyle="-", linewidth=1)

    # Somatosensory stimulus onset.
    if in_range(0.0):
        ax.axvline(0.0, color="black", linestyle="-", linewidth=1)


def compute_within_subject_ci(cond_a, cond_b, confidence=0.95):
    """
    Compute within-subject CI (Cousineau 2005) with Morey (2008) correction.
    Inputs are arrays shaped (n_subjects, n_times).
    """
    n_subjects = cond_a.shape[0]
    n_conditions = 2

    # Subject mean across conditions at each time point.
    subj_mean = (cond_a + cond_b) / n_conditions
    grand_mean = subj_mean.mean(axis=0, keepdims=True)

    # Normalize to remove between-subject variance.
    cond_a_norm = cond_a - subj_mean + grand_mean
    cond_b_norm = cond_b - subj_mean + grand_mean

    # Morey correction for within-subject CI.
    correction = np.sqrt(n_conditions / (n_conditions - 1))
    sem_a = cond_a_norm.std(axis=0, ddof=1) / np.sqrt(n_subjects)
    sem_b = cond_b_norm.std(axis=0, ddof=1) / np.sqrt(n_subjects)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n_subjects - 1)

    ci_a = sem_a * t_crit * correction
    ci_b = sem_b * t_crit * correction
    return ci_a, ci_b


def compute_subject_prestim_amplitudes(
    epochs,
    metadata_df,
    stim_time,
    prestim_tmin,
    prestim_tmax,
    return_trial_phase=False,
):
    """
    Compute subject-level prestim amplitudes using Hilbert envelope.
    Returns:
      - response_df: mean amplitude per response_decoded and channel
      - intensity_df: mean amplitude per intensity_level, response_decoded, channel
      - response_phase_df: mean phase at stimulus onset per response_decoded and channel
      - response_itpc_df: phase-locking (ITPC/PLV) at stimulus onset per response_decoded and channel
      - trial_phase_df (optional): trial-level phase at stimulus onset per response/channel
    """
    times = epochs.times
    channel_names = epochs.ch_names

    # Compute analytic amplitude (envelope) for each trial/channel/time.
    # Use MNE's apply_hilbert to keep metadata and picks handling consistent.
    envelope_epochs = epochs.copy().apply_hilbert(envelope=True)
    envelope = envelope_epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Compute analytic signal for phase extraction at stimulus onset.
    analytic_epochs = epochs.copy().apply_hilbert(envelope=False)
    analytic = analytic_epochs.get_data()  # complex, (n_epochs, n_channels, n_times)
    stim_idx = int(np.argmin(np.abs(times - stim_time)))

    # Prestim window indices.
    tmask = (times >= prestim_tmin) & (times <= prestim_tmax)
    if not np.any(tmask):
        raise ValueError("Prestim window does not overlap epochs time range.")

    # Average across time window per trial/channel -> scalar per trial/channel.
    trial_amp = envelope[:, :, tmask].mean(axis=2)  # (n_trials, n_channels)
    stim_phase = np.angle(analytic[:, :, stim_idx])  # (n_trials, n_channels)

    # Build per-trial, per-channel DataFrame for grouping.
    n_trials, n_channels = trial_amp.shape
    trial_df = pd.concat([metadata_df] * n_channels, ignore_index=True)
    trial_df["channel"] = np.repeat(channel_names, n_trials)
    trial_df["prestim_amp"] = trial_amp.T.reshape(-1)
    trial_df["stimonset_phase"] = stim_phase.T.reshape(-1)

    # Subject-level mean by response_decoded.
    response_df = (
        trial_df.groupby(["response_decoded", "channel"], as_index=False)["prestim_amp"]
        .mean()
        .rename(columns={"prestim_amp": "mean_amp"})
    )

    # Subject-level mean by intensity and response.
    intensity_df = (
        trial_df.groupby(
            ["intensity_level", "response_decoded", "channel"], as_index=False
        )["prestim_amp"]
        .mean()
        .rename(columns={"prestim_amp": "mean_amp"})
    )

    # Subject-level mean phase at stimulus onset by response.
    response_phase_df = (
        trial_df.groupby(["response_decoded", "channel"], as_index=False)["stimonset_phase"]
        .apply(lambda x: circular_mean(x.values))
        .rename(columns={"stimonset_phase": "mean_phase"})
    )

    # Subject-level ITPC/PLV at stimulus onset by response and channel.
    def itpc(phases):
        return np.abs(np.mean(np.exp(1j * phases)))

    response_itpc_df = (
        trial_df.groupby(["response_decoded", "channel"], as_index=False)["stimonset_phase"]
        .apply(lambda x: itpc(x.values))
        .rename(columns={"stimonset_phase": "itpc"})
    )

    if return_trial_phase:
        trial_phase_df = trial_df[["response_decoded", "channel", "stimonset_phase"]].copy()
        return response_df, intensity_df, response_phase_df, response_itpc_df, trial_phase_df

    return response_df, intensity_df, response_phase_df, response_itpc_df


def compute_itpc_timecourse(epochs, metadata_df):
    """
    Compute ITPC/PLV time courses per condition and channel (subject-level).
    Returns:
      - itpc_yes: (n_channels, n_times)
      - itpc_no: (n_channels, n_times)
    """
    analytic = epochs.copy().apply_hilbert(envelope=False).get_data()
    phases = np.angle(analytic)  # (n_trials, n_channels, n_times)
    perceived_mask = metadata_df["response_decoded"] == "Yes"
    unperceived_mask = metadata_df["response_decoded"] == "No"
    if perceived_mask.sum() == 0 or unperceived_mask.sum() == 0:
        return None, None

    itpc_yes = np.abs(np.mean(np.exp(1j * phases[perceived_mask.to_numpy()]), axis=0))
    itpc_no = np.abs(np.mean(np.exp(1j * phases[unperceived_mask.to_numpy()]), axis=0))
    return itpc_yes, itpc_no

def perform_pca_analysis(epochs, subID, modality):
    """
    Perform PCA analysis on topographies at specific time points (55ms, 80ms, 140ms)
    and save plots of the first 5 important components with subject ID and modality.

    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs object containing the EEG data
    subID : int
        Subject ID
    modality : str
        Modality name (e.g., 'mecha', 'elect')
    """
    # Increase font sizes for all plots created in this function.
    # Store previous values so plotting outside this function is unchanged.
    old_rc_params = {
        'font.size': plt.rcParams['font.size'],
        'axes.titlesize': plt.rcParams['axes.titlesize'],
        'axes.labelsize': plt.rcParams['axes.labelsize'],
        'xtick.labelsize': plt.rcParams['xtick.labelsize'],
        'ytick.labelsize': plt.rcParams['ytick.labelsize'],
        'legend.fontsize': plt.rcParams['legend.fontsize'],
        'figure.titlesize': plt.rcParams['figure.titlesize'],
    }
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 17,
    })

    # Define time points for PCA analysis (in seconds)
    time_points = [0.05, 0.080, 0.140]  # 55ms, 80ms, 140ms

    # Create directory for individual subject results
    subject_resdir = os.path.join(RESDIR, 'Figures', 'Main', 'eeg_results', 'pca_topomaps', f'sub-0{subID}', f'ses-{modality}')
    if not os.path.exists(subject_resdir):
        os.makedirs(subject_resdir)

    print(f"Performing PCA analysis for subject {subID}, modality {modality}")

    # Get channel names and positions for topomap plotting
    ch_names = epochs.ch_names
    info = epochs.info
    epochs._data *= 1e-6

    epochs = epochs["(response_decoded == 'Yes')"]

    # Compute average once so we can show the original topomap before PCA.
    evoked_original = epochs.average()

    fig = evoked_original.plot_joint(
    times=time_points, title=f'All Trials {modality}', show=False)
    fig.savefig(os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_topomaps_alltrials{suffix}.png'), dpi=300)


    # Store PCA results for each time point
    pca_results_per_timepoint = {}

    # Create figure for original topomap + PCA components
    fig, axes = plt.subplots(len(time_points), 6, figsize=(15, 8))
    fig.suptitle(f'Original + PCA Components - Subject {subID}, Modality {modality}', fontsize=14, fontweight='bold')
    
    # Process each time point independently
    for time_idx, time_point in enumerate(time_points):

        # Get data at specific time point
        # Find the closest sample to the desired time point
        sample_idx = np.argmin(np.abs(epochs.times - time_point))
        time_actual = epochs.times[sample_idx]

        print(f"  Processing time point {time_point*1000:.0f}ms (actual: {time_actual*1000:.1f}ms)")

        # Extract topography data for all trials at this time point
        # Shape: (n_epochs, n_channels)
        data_at_time = epochs.get_data(picks='eeg')[:, :, sample_idx]  # Shape: (n_epochs, n_channels)

        # Standardize the data (n_trials x n_channels)
        #scaler = StandardScaler()
        #data_scaled = scaler.fit_transform(data_at_time)

        # Perform PCA on trials x channels data
        pca = PCA(n_components=5)
        pca_result = pca.fit_transform(data_at_time)

        # Store results for this time point
        pca_results_per_timepoint[f'time_{time_point*1000:.0f}ms'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'transformed_data': pca_result,
            'time_actual': time_actual
        }

        # Plot original (pre-PCA) topomap for this time point.
        evoked_original.plot_topomap(
            times=time_actual,
            axes=axes[time_idx][0],
            show=False,
            colorbar=False,
            outlines='head',
            contours=0,
        )
        axes[time_idx][0].set_title(f'Original\n{time_actual*1000:.1f} ms', fontsize=10, fontweight='bold')
        # Use the topomap image itself as mappable so colorbar colors match.
        if len(axes[time_idx][0].images) > 0:
            original_mappable = axes[time_idx][0].images[0]
        elif len(axes[time_idx][0].collections) > 0:
            original_mappable = axes[time_idx][0].collections[0]
        else:
            original_mappable = None

        # Plot the first 5 PCA components as topomaps for this time point
        # Use a shared symmetric scale so one colorbar is meaningful across PCs.
        comp_vmax = np.max(np.abs(pca.components_[:5]))
        comp_vlim = (-comp_vmax, comp_vmax)
        components_mappable = None
        for i in range(5):
            # Create evoked object for topomap plotting
            pca_component = pca.components_[i]  # Shape: (n_channels,)

            # Create a dummy evoked object for plotting
            # Need to add a time dimension: (n_channels,) -> (1, n_channels)
            evoked_data = pca_component.reshape(1, -1).T  # Shape: (1, n_channels)

            eeg_ch_names = epochs.copy().pick('eeg').ch_names
            info_eeg = mne.pick_info(epochs.info, mne.pick_types(epochs.info, eeg=True))
            evoked_pca = mne.EvokedArray(evoked_data, info_eeg, tmin=0.0)
            
            mne.viz.plot_topomap(
                pca.components_[i],           # 1D array directly
                info_eeg, 
                axes=axes[time_idx][i+1],
                show=False,
                vlim=comp_vlim,
                contours=6,
                outlines='head',
                cmap='RdBu_r'
            )

            # Plot topomap
            """
            evoked_pca.plot_topomap(
                times=0.0,
                axes=axes[time_idx][i + 1],
                show=False,
                colorbar=False,
                outlines='head',
                contours=6,
                #vlim=comp_vlim,
            )
            """
            # Keep one component mappable for the shared PCA colorbar.
            if components_mappable is None:
                if len(axes[time_idx][i + 1].images) > 0:
                    components_mappable = axes[time_idx][i + 1].images[0]
                elif len(axes[time_idx][i + 1].collections) > 0:
                    components_mappable = axes[time_idx][i + 1].collections[0]

            # Add title with component info
            explained_var = pca.explained_variance_ratio_[i] * 100
            axes[time_idx][i + 1].set_title(f'PC {i+1}\n{explained_var:.1f}% variance', fontsize=10, fontweight='bold')


        # Add two shared colorbars for this timepoint figure:
        # one for the original topomap and one for the PCA components.
        
        #plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.93])

        # Place original colorbar directly next to the original topomap axis.
        ax0_pos = axes[time_idx][0].get_position()
        ax_last_pos = axes[time_idx][-1].get_position()
        cbar_width = 0.010
        gap = 0.006
        cbar_ax_orig = fig.add_axes([ax0_pos.x1 + gap, ax0_pos.y0, cbar_width, ax0_pos.height])
        cbar_ax_comp = fig.add_axes([ax_last_pos.x1 + gap, ax_last_pos.y0, cbar_width, ax_last_pos.height])
        if original_mappable is not None:
            fig.colorbar(original_mappable, cax=cbar_ax_orig)
            cbar_ax_orig.set_title('Orig', fontsize=8, pad=4)
        if components_mappable is not None:
            fig.colorbar(components_mappable, cax=cbar_ax_comp)
            cbar_ax_comp.set_title('PCs', fontsize=8, pad=4)

        plt.subplots_adjust(bottom=0.15, top=0.85)
        print(f"PC0 mappable norm: {axes[time_idx][1].images[0].norm.vmin:.3f} to {axes[time_idx][1].images[0].norm.vmax:.3f}")
    
    
    # Add subject and modality info to the figure
    fig.text(0.5, 0.02, f'Subject: {subID} | Modality: {modality}',
            ha='center', fontsize=12, style='italic')

    # Save the figure
    filename = f'sub-0{subID}_ses-{modality}_PCA_components_topomaps_unscaled.png'
    filepath = os.path.join(subject_resdir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close()

    print(f"PCA analysis completed for subject {subID}, modality {modality}")
    print(f"Saved to: {filepath}")

    # Save PCA results for potential future analysis
    # Save results for each time point
    for time_key, results in pca_results_per_timepoint.items():
        np.save(os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_PCA_results_{time_key}_unscaled.npy'), results)

    # Also save a summary file
    summary_results = {
        'time_points': time_points,
        'pca_results_per_timepoint': pca_results_per_timepoint
    }
    np.save(os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_PCA_summary_results_unscaled.npy'), summary_results)

    n_components = 10
    pca = PCA(n_components=n_components)

    # Fit on all timepoints concatenated
    data_orig = evoked_original.get_data(picks='eeg')
    time_mask = evoked_original.times >= -0.1
    data = data_orig[:, time_mask]
    times = evoked_original.times[time_mask]
    n_channels, n_times = data.shape
    data_2d = data.reshape(n_times, n_channels)  # (n_epochs*n_times, n_channels)
    pca.fit(data_2d)

    # Get scores at each timepoint
    scores_over_time = np.zeros((n_components, n_times))
    for t in range(n_times):
        data_t = data[:, t].reshape(-1, 1).T                    # (n_epochs, n_channels)
        scores_t = pca.transform(data_t)          # (n_epochs, n_components)
        scores_over_time[:, t] = scores_t.T[:,0]

    fig, axes = plt.subplots(n_components, 1, figsize=(12, n_components * 2), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(times, scores_over_time[i], color=f'C{i}')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')  # stimulus onset
        ax.set_ylabel(f'PC{i+1}')
        ax.set_xlabel('Time (s)')

    plt.suptitle('PCA component scores over time (averaged over epochs)', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_topomaps_timepattern{suffix}.png'), dpi=300)
    plt.close()



    fig = plt.figure(figsize=(16, n_components * 2))

    for i in range(n_components):
        # Two columns: topomap on left, timecourse on right
        ax_topo = fig.add_subplot(n_components, 4, i * 4 + 1)        # narrow left
        ax_time = fig.add_subplot(n_components, 4, (i * 4 + 2, i * 4 + 4))  # wide right

        # --- Topomap ---
        component_topo = pca.components_[i]  # (n_channels,)
        mne.viz.plot_topomap(
            component_topo,
            info,
            axes=ax_topo,
            show=False,
            contours=4
        )
        ax_topo.set_title(f'PC{i+1}', fontsize=9)

        # --- Timecourse ---
        ax_time.plot(times, scores_over_time[i], color=f'C{i}')
        ax_time.axhline(0, color='k', linewidth=0.5)
        ax_time.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_time.set_ylabel(f'PC{i+1}')
        if i == n_components - 1:
            ax_time.set_xlabel('Time (s)')

    plt.suptitle('PCA components: topography & timecourse (averaged over epochs)', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_topomaps_timepattern{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Compute ICA separately for each timepoint and plot 5 components per timepoint.
    ica_epochs = epochs.copy().pick('eeg')
    n_ica_plot = 5
    fig, axes = plt.subplots(len(time_points), n_ica_plot, figsize=(3 * n_ica_plot, 3 * len(time_points)))
    if len(time_points) == 1:
        axes = np.array([axes])

    for row_idx, tp in enumerate(time_points):
        # Fit ICA on the single sample nearest to this target timepoint.
        sample_idx = np.argmin(np.abs(ica_epochs.times - tp))
        actual_time = float(ica_epochs.times[sample_idx])
        ica_data_tp = ica_epochs.get_data()[:, :, sample_idx:sample_idx + 1]
        ica_epochs_tp = mne.EpochsArray(
            data=ica_data_tp,
            info=ica_epochs.info.copy(),
            events=ica_epochs.events.copy(),
            event_id=ica_epochs.event_id.copy(),
            tmin=actual_time,
            verbose=False
        )

        # Keep component count valid for available samples/channels.
        n_samples_for_ica = ica_data_tp.shape[0] * ica_data_tp.shape[2]
        ica_n_components = min(15, len(ica_epochs_tp.ch_names), n_samples_for_ica - 1)
        if ica_n_components < 1:
            raise ValueError(f"Not enough samples for ICA at {actual_time*1000:.1f} ms")

        ica = mne.preprocessing.ICA(
            n_components=ica_n_components,
            method='fastica',
            random_state=42,
            max_iter='auto'
        )
        ica.fit(ica_epochs_tp, verbose=False)

        # Save ICA matrices per timepoint.
        time_label = f"{int(np.round(actual_time * 1000))}ms"
        np.save(
            os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_ICA_mixing_matrix_{time_label}.npy'),
            ica.mixing_matrix_
        )
        np.save(
            os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_ICA_unmixing_matrix_{time_label}.npy'),
            ica.unmixing_matrix_
        )

        # Plot first 5 components for this timepoint row.
        ica_components = ica.get_components()
        n_available = min(n_ica_plot, ica_components.shape[1])
        for col_idx in range(n_ica_plot):
            ax = axes[row_idx, col_idx]
            if col_idx < n_available:
                # Explained variance for this ICA component on EEG channels.
                var_ratio = ica.get_explained_variance_ratio(
                    ica_epochs_tp,
                    components=[col_idx],
                    ch_type='eeg'
                )['eeg']
                mne.viz.plot_topomap(
                    ica_components[:, col_idx],
                    ica_epochs_tp.info,
                    axes=ax,
                    show=False,
                    contours=4,
                    cmap='RdBu_r'
                )
                ax.set_title(f'IC{col_idx+1}\n{var_ratio*100:.1f}% var', fontsize=14)
            else:
                ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(f'{actual_time*1000:.1f} ms', fontsize=14)

    plt.suptitle('ICA Components by Time Point (5 per timepoint)', y=1.02, fontsize=18)
    plt.tight_layout()
    fig.savefig(
        os.path.join(subject_resdir, f'sub-0{subID}_ses-{modality}_ICA_components_topomaps_by_timepoint{suffix}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


    # Restore previous global matplotlib style settings.
    plt.rcParams.update(old_rc_params)

    return summary_results
