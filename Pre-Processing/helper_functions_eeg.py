import os
import argparse
import re
import numpy as np
import mne
import pandas as pd
import datetime as dt
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from mne.viz import plot_topomap

def parse_subjects():
    """
    Parse command line arguments to get subject IDs.
    If no subject IDs are provided, process all subjects from 13 to 52.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject_id",
        type=str,
        nargs="+",
        help="Subject IDs to process (e.g., '01 02 15' for subjects 1, 2, and 15)",
        required=False,
    )
    parser.add_argument(
        "--mod",
        type=str,
        nargs="+",
        help="Stimulation Modality (Electrical or Mechanical)",
        required=False,
    )
    args = parser.parse_args()

    # determine modalities
    if args.mod:
        modalities = args.mod
    else:
        modalities = ['Electrical', 'Mechanical']

    # determine subject IDs
    if args.subject_id:
        subIDs = args.subject_id
    else:
        # process all subjects from subject 13 (no pilot subjects)
        subIDs = map(str, range(13, 53))

    return subIDs, modalities


def assign_modality(modality):
    if (modality=='Electrical') or (modality=='elec'):
        ses_label = 'elec' 
        stim_trigger = 'A - Stimulation'
    elif (modality=='Mechanical') or (modality=='mecha'):
        ses_label = 'mecha' 
        stim_trigger = 'B - Stimulation'
    
    return ses_label, stim_trigger

def natural_session_key(s):
    """ Split the string (name of the EEG session) by digits and non-digits,
    then convert digit parts to int. This helps to sort the sessions in the order they were recorded.""" 
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def check_exceptions(modality, block, run, subID, ses, EEG_data_dir):
    # Exeptions when the block ID was not entered correctly during the experiment 
    exception1 = '2025-01-09T105547_session2.set'
    exception2 = '2025-01-09T120435_session2.set'
    exception3 = '2025-01-09T120435_session6.set'
    exception4 = '2025-01-09T120435_session4.set'
    exception5 = '2025-04-28T161444_session5.set'
    exception8 = '2025-06-17T153147_session2.set'
    exception9 = '2025-06-19T102831_session3.set'
    exception10 = '2025-05-19T100801_session5.set'

    # there is something wrong with this file in sub 17 electrical (block 2)
    #if ses == '/data/p_02989/shared_workspace/rawdir/sub-017/ID17_Electrical/converted_EEG/NT/2025-02-13T164813_session2.set':
    #    continue

    if subID == '09':
        if (modality == 'Mechanical') & (int(block) == 0):
            block = 2
            run = 1
    elif (ses==os.path.join(EEG_data_dir, exception1)) & (int(block) == 1):
        block = 2
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception2)) & (modality == 'Electrical'):
        block = 5
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception3)) & (modality == 'Electrical'):
        block = 7
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception4)) & (modality == 'Electrical'):
        block = 6
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception5)) & (modality == 'Mechanical'):
        block = 6
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception8)) & (modality == 'Mechanical'):
        block = 4
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception9)) & (modality == 'Mechanical'):
        block = 1
        run = 1
    elif (ses==os.path.join(EEG_data_dir, exception10)) & (modality == 'Mechanical'):
        block = 2
        run = 1

    return block, run


def reject_chan_psd(raw):
    """ Reject bad channels based on their power spectral density (PSD) in the high-frequency range (70-100 Hz)."""
    channel_names = np.asarray(raw.info['ch_names'])
    eeg_chans = np.where(~np.isin(channel_names,['ECG','RESP', 'CNAP', 'VEOG']))[0]

    bads = []
    psds_overall, _ = mne.time_frequency.psd_array_welch(raw.get_data()[eeg_chans], sfreq=raw.info['sfreq'], fmin=70, fmax=100, average='mean')
    psds_mean = np.mean(psds_overall, 0)
    psds_std = np.std(psds_overall, 0)
    formula = np.mean(psds_mean) + 2*np.mean(psds_std)

    for i, channel in enumerate(raw.get_data()[eeg_chans]):
        channel_psd, _ = mne.time_frequency.psd_array_welch(channel, sfreq=raw.info['sfreq'], fmin=70, fmax=100, average='mean')
        channel_psd_mean = np.mean(channel_psd)

        # check if the channels' mean psd is higher than the overal psd + 3*std
        if channel_psd_mean > formula:
            bads.append(channel_names[i])
    print('bad channels:', bads)

    return bads


def interpolate_bads(raw, bads):
    """ Interpolate bad channels """
    print('bad channels:', bads)
    raw.info['bads'] = bads
    raw.interpolate_bads()
    raw.set_eeg_reference('average', projection=True)  #compute the reference
    raw.apply_proj()



def get_block_run(events_from_annot, event_dict, ses, EEG_data_dir, subID, modality):
    '''
    Extracts block and run number from events 
    '''

    event_dict_numeric = {k: v for k, v in event_dict.items() if k.isdigit()}
    trigger_list = list(map(int,list(event_dict_numeric.keys())))
    trigger_array = np.array(trigger_list)
    
    # start trigger was not recorded in block 8 of subject 52 electrical and 17 mechanical
    exception_sub17_5 = os.path.join(EEG_data_dir, '2025-02-19T112012_session4.set')
    exception_sub17_7 = os.path.join(EEG_data_dir, '2025-02-19T121905_session1.set')
    exception_sub52 = os.path.join(EEG_data_dir, '2025-06-27T132126_session18.set')
    if ses == exception_sub52:
        init_trigger = None
        startstop_trigger = '53'
        block = 8
        run = 1
    # in electrical sessions, IDs 44, all >= 46 (except ID47) have the trigger 72 as start trigger (I don't know why..)
    elif ((subID in ["44", "46", "35"]) or (subID >= "48")) and modality=='Electrical':
        init_trigger = 72
        second_trigger = events_from_annot[1,2]
        startstop_trigger = list(event_dict_numeric.keys())[list(event_dict_numeric.values()).index(second_trigger)]
        run = 1 # run is always 1 for those subjects
        block = None
        if (startstop_trigger == '112') and (subID == "35"):
            block = 1
        if (startstop_trigger == '122') and (subID == "35"):
            block = 2
        elif startstop_trigger == '111':
            block = 1
        elif startstop_trigger == '121':
            block = 2
        elif startstop_trigger == '3':
            block = 3
        elif (int(startstop_trigger) > 10) & (int(startstop_trigger) < 60):
            block = sum(int(digit) for digit in startstop_trigger)
    
    # in electrical sessions, ID 38 and 39 have trigger 8 as start trigger (I don't know why..)
    elif subID in ['38','39'] and modality=='Electrical':
        init_trigger = 8
        second_trigger = events_from_annot[1,2]
        startstop_trigger = list(event_dict_numeric.keys())[list(event_dict_numeric.values()).index(second_trigger)]
        run = 1 # run is always 1 for those subjects
        block = None
        if startstop_trigger == '15':
            block = 1
        elif startstop_trigger == '25':
            block = 2
        elif startstop_trigger == '3':
            block = 3
        elif startstop_trigger == '13':
            block = 4
        elif startstop_trigger == '23':
            block = 5
        elif startstop_trigger == '1':
            block = 6
        elif startstop_trigger == '11':
            block = 7
        elif startstop_trigger == '21':
            block = 8

    else:
        init_trigger = 200
        startstop_trigger = trigger_array[(trigger_array<200)&(trigger_array>=100)][0]
        startstop_trigger = str(startstop_trigger)
        block = startstop_trigger[-2]
        run = startstop_trigger[-1]


    # make sure that this is a main block and no threshold assessment!
    if (init_trigger not in trigger_list) & (ses not in [exception_sub52, exception_sub17_5, exception_sub17_7]):
        print('NO MAIN BLOCK!')
        raise Exception("No Main Block! Please check the session number!") 


    # in the best case there are 198 triggers, but if it less it should still be processed
    # but if it is less than 50, something is wrong
    if len(events_from_annot) < 50:
        raise Exception("Dataset not complete") 

    return block, run, startstop_trigger


def handle_startstop_exceptions(annotations, trigger_code):
    start_trigger = annotations.iloc[1]
    assert start_trigger['description'] == trigger_code, "No valid start trigger!!"
    annotations.at[annotations.index[1], 'trigger_meaning'] = 'startstop'

    stop_trigger = annotations.iloc[-1]
    assert stop_trigger['description'] == trigger_code, "No valid start trigger!!"
    annotations.at[annotations.index[-1], 'trigger_meaning'] = 'startstop'

    return annotations


def create_physio_df(picks, raw_physio):

    # make one file with both resp and ecg data (use the uncropped data!!)
    physio_data, times_samples_ecg = raw_physio.get_data(
        picks=picks, return_times=True
    )
    # save it as compresses tabular file
    physio_data_df = pd.DataFrame(physio_data.T)

    return physio_data_df, times_samples_ecg

def read_scale_data(scale_file_path):
    """
    Read scale data from csv.
    Renames columns to fit the rest of the analysis.
    """
    # Read the data (and rename columns)
    scale_data = pd.read_csv(scale_file_path)
    if "Timestamp" in scale_data.columns:
        scale_data.rename(
            columns={"Weight (g)": "force_value"}, inplace=True
        )
        scale_data.rename(columns={"Timestamp": "timestamp"}, inplace=True)
        
    scale_data["timestamp"] = pd.to_datetime(scale_data["timestamp"])

    return scale_data


def exclude_invalid_triggers(scale_data, trigger_scale):
    """ Exclude triggers in the beginning of the recording that were just testing triggers.
    The can be recognized, since they are very close to each other."""

    iti = np.diff(
        trigger_scale["timestamp"]
    )/ np.timedelta64(1, 's') # inter-trigger intervals in secs
    threshold = 1.0
    idx = np.where(iti <= threshold)[0]

    if idx.size:
        last_invalid = trigger_scale.iloc[idx[-1]]["total_secs"]
        scale_data = scale_data[scale_data["total_secs"] > last_invalid]
        trigger_scale = scale_data[scale_data["trigger_code"] == 1]

    return scale_data, trigger_scale


def merge_trigger_df(stim_annot, trigger_scale, tolerance_delay):
    """Merge the triggers from the scale with the triggers from the EEG data.
    How to merge them depends on the number of triggers in each data set."""

    if len(stim_annot) == len(trigger_scale):
        # if we have as many triggers in the force data as we got epochs (best case)
        # we assign the nearest trigger to the nearest epoch
        trigger_diff_df = pd.DataFrame()
        trigger_diff_df['total_secs_x'] = stim_annot['total_secs'].reset_index(drop=True)
        trigger_diff_df['timestamp'] = stim_annot['timestamp'].reset_index(drop=True)
        trigger_diff_df['total_secs_y'] = trigger_scale['total_secs'].reset_index(drop=True)

    else:
        merge_direction = 'backward'
    
        # match trigger timings
        # it does not really matter if we merge on timestamp or total_secs
        # both capture the delay
        trigger_diff_df = pd.merge_asof(
            stim_annot[["total_secs", "timestamp"]],
            trigger_scale[["total_secs", "timestamp"]],
            left_on="timestamp",
            right_on="timestamp",
            direction=merge_direction,
            tolerance=tolerance_delay,
                        )

    # if not all force triggers were recorded, ignore those for the difference computation
    trigger_diff_df = trigger_diff_df.dropna(subset=["total_secs_y"])

    return trigger_diff_df


def annotate_fig(dataname, x=-2):
    """ Write on the figure with which data it was generated, the date and the script name."""
    plt.annotate(dataname, xy=(x, 0.3), xycoords='axes fraction', fontsize=8, ha='center')
    plt.annotate(dt.datetime.now(), xy=(x, 0.2), xycoords='axes fraction', fontsize=8, ha='center')
    plt.annotate(f"generated in {os.path.basename(__file__)}", xy=(x, 0.1), xycoords='axes fraction', fontsize=8, ha='center')


def compute_prestim_oscillations(epochs):

    eeg_picks = mne.pick_types(epochs.info, eeg=True)
    # crop epochs to pre-stimulus interval -600 to -50 ms
    prestim_epochs = epochs.copy().pick(eeg_picks)
    prestim_epochs.crop(tmin=-0.6, tmax=-0.05)

    sfreq = prestim_epochs.info['sfreq']
    n_samples = prestim_epochs._data.shape[2]
    data = prestim_epochs._data
    yf = fft(prestim_epochs._data, axis=2) # 301 samples
    power = (np.abs(yf) ** 2) # Normalize by number of samples

    freqs = fftfreq(n_samples, 1 / sfreq)

    # extract positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    power_pos = power[:, :, pos_mask]

    # Convert power to PSD by dividing by frequency bin width (df):
    df = sfreq / n_samples
    psd = power_pos / df  # PSD units: (uV^2)/Hz

    # Select bands
    alpha_mask = (freqs_pos > 8) & (freqs_pos < 14)
    beta_mask = (freqs_pos > 15) & (freqs_pos < 26)

    alpha_power = np.mean(psd[:, :, alpha_mask], axis=2)
    beta_power = np.mean(psd[:, :, beta_mask], axis=2)

    return alpha_power, beta_power


def plot_prestim_oscillations(mean_psd, info, figure_name):
    """Plot Presimulus Oscillations (alpha or beta).
    
    Args:
        mean_psd (type): array of len channels.
        info (Info): info object for eeg channel location infos.
        figuredir (str): path to save the figure
    """
    
    fig, ax = plt.subplots()
    
    im, _ = mne.viz.plot_topomap(mean_psd, info, axes=ax, show=False)

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean PSD')

    plt.savefig(figure_name)


##########################################
# Functions for covariance matrix analysis
##########################################

def compute_cov(epochs, data_window, noise_window):
    """Compute covariance matrix for noise and data, regularized."""
    tmin, tmax = noise_window 
    noise_cov_baseline = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax)
    noise_cov_reg = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method="auto", rank=None)
    
    # Compute data covariance matrix
    tmin, tmax = data_window
    data_cov_reg = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method="auto", rank=None)
    
    return noise_cov_baseline, noise_cov_reg, data_cov_reg


def compute_svd(cov_matrix):
    """
    Compute SVD of covariance matrix.
    
    Parameters
    ----------
    cov_matrix : mne.Covariance
        MNE covariance object
        
    Returns
    -------
    U : ndarray
        Left singular vectors (eigenvectors)
    S : ndarray
        Singular values (square root of eigenvalues)
    Vh : ndarray
        Right singular vectors (conjugate transpose of U for symmetric matrix)
    """
    # Extract covariance data
    if isinstance(cov_matrix, mne.Covariance):
        cov_data = cov_matrix.data
    else:
        cov_data = cov_matrix
    
    # Compute SVD
    U, S, Vh = linalg.svd(cov_data)
    
    # For symmetric matrices, eigenvalues are S^2
    eigenvalues = S ** 2
    
    print(f"SVD Decomposition:")
    print(f"  Matrix shape: {cov_data.shape}")
    print(f"  Number of singular values: {len(S)}")
    print(f"  Largest singular value: {S[0]:.6e}")
    print(f"  Smallest singular value: {S[-1]:.6e}")
    print(f"  Condition number: {S[0]/S[-1]:.2e}")
    
    return U, S, Vh


def rank_evaluation(cov_matrix, threshold=0.01, save_figures=False, figuredir=None, add_on=None, suffix=''):
    """
    Compute and output effective rank of the matrix.
    
    Parameters
    ----------
    cov_matrix : mne.Covariance or ndarray
        Covariance matrix
    threshold : float
        Relative threshold for effective rank (default: 1% of max singular value)
        
    Returns
    -------
    effective_rank : int
        Effective rank based on threshold
    """
    # Compute SVD
    U, S, Vh = compute_svd(cov_matrix)
    
    # Compute effective rank (number of singular values above threshold)
    threshold_abs = threshold * S[0]
    effective_rank = np.sum(S > threshold_abs)
    
    # Compute variance explained
    variance_explained = np.cumsum(S**2) / np.sum(S**2)
    rank_95 = np.searchsorted(variance_explained, 0.95) + 1
    rank_99 = np.searchsorted(variance_explained, 0.99) + 1
    
    # Plot singular value spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Singular values
    ax = axes[0]
    ax.plot(S, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=threshold_abs, color='r', linestyle='--', 
               label=f'Threshold ({threshold*100:.0f}% of max)')
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Cumulative variance explained
    ax = axes[1]
    ax.plot(variance_explained * 100, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=95, color='g', linestyle='--', label='95% variance')
    ax.axhline(y=99, color='orange', linestyle='--', label='99% variance')
    ax.axvline(x=rank_95-1, color='g', linestyle=':', alpha=0.5)
    ax.axvline(x=rank_99-1, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(figuredir, f'rank_evaluation_{add_on}{suffix}.png'))
    plt.close()
    
    # Print results
    print(f"\nRank Evaluation:")
    print(f"  Full rank (n_channels): {len(S)}")
    print(f"  Effective rank ({threshold*100:.0f}% threshold): {effective_rank}")
    print(f"  Rank for 95% variance: {rank_95}")
    print(f"  Rank for 99% variance: {rank_99}")
    print(f"  Variance explained by first component: {variance_explained[0]*100:.2f}%")
    print(f"  Variance explained by first 5 components: {variance_explained[4]*100:.2f}%")
    print(f"  Variance explained by first 10 components: {variance_explained[9]*100:.2f}%")
    
    return effective_rank, S


def eigenvector_evaluation(cov_matrix, epochs, n=5, figsize=(15, 10), save_figures=False, figuredir=None, suffix=''):
    """
    Investigate the eigenvectors. Plot the topography of the first n eigenvectors.
    
    Parameters
    ----------
    cov_matrix : mne.Covariance
        MNE covariance object
    epochs : mne.Epochs
        Epochs object (needed for channel positions)
    n : int
        Number of eigenvectors to plot
    figsize : tuple
        Figure size
    """
    # Compute SVD
    U, S, Vh = compute_svd(cov_matrix)
    
    # Get channel info
    info = epochs.info
    ch_names = cov_matrix.ch_names if hasattr(cov_matrix, 'ch_names') else info['ch_names']
    
    # Create figure
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    # Variance explained
    variance_explained = (S**2) / np.sum(S**2) * 100
    
    # Plot topographies
    for i in range(n):
        ax = axes[i]
        
        # Get eigenvector (spatial pattern)
        eigenvector = U[:, i]
        
        # Plot topomap
        im, _ = plot_topomap(
            eigenvector, 
            info, 
            axes=ax,
            show=False,
            contours=6,
            cmap='RdBu_r',
            sensors=True
        )
        
        ax.set_title(f'Component {i+1}\n'
                    f'Var: {variance_explained[i]:.2f}%\n'
                    f'SV: {S[i]:.2e}',
                    fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Eigenvector Weight', rotation=270, labelpad=20, fontsize=12)
    plt.suptitle('Eigenvector Topographies (Spatial Patterns)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(figuredir, f'eigenvector_evaluation{suffix}.png'))
    plt.show()
    
    # Print dominant channels for each component
    print(f"\nDominant channels for first {n} components:")
    for i in range(n):
        eigenvector = U[:, i]
        top_indices = np.argsort(np.abs(eigenvector))[-5:][::-1]
        print(f"\n  Component {i+1} (variance: {variance_explained[i]:.2f}%):")
        for idx in top_indices:
            print(f"    {ch_names[idx]}: {eigenvector[idx]:.4f}")
    
    return U, S


def plot_correlation_strength(cov_matrix, epochs, visual_channels=None, somatosensory_channels=None, save_figures=False, figuredir=None, suffix=''):
    """
    Plot the strength of correlation between visual and somatosensory channels.
    Also print the 2 strongest correlated channels.
    
    Parameters
    ----------
    cov_matrix : mne.Covariance
        MNE covariance object
    epochs : mne.Epochs
        Epochs object
    visual_channels : list
        List of visual channel names (e.g., ['O1', 'O2', 'Oz'])
    somatosensory_channels : list
        List of somatosensory channel names (e.g., ['C3', 'C4', 'Cz'])
    """
    # Default channel groups if not specified
    if visual_channels is None:
        visual_channels = ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8']
    if somatosensory_channels is None:
        somatosensory_channels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6']
    
    # Get covariance data and channel names
    if isinstance(cov_matrix, mne.Covariance):
        cov_data = cov_matrix.data
        ch_names = cov_matrix.ch_names
    else:
        cov_data = cov_matrix
        ch_names = epochs.info['ch_names']
    
    # Convert to correlation matrix
    std_dev = np.sqrt(np.diag(cov_data))
    corr_matrix = cov_data / np.outer(std_dev, std_dev)
    
    # Find indices of channel groups
    visual_idx = [i for i, ch in enumerate(ch_names) if ch in visual_channels]
    somato_idx = [i for i, ch in enumerate(ch_names) if ch in somatosensory_channels]
    
    print(f"\nChannel Groups:")
    print(f"  Visual channels found: {[ch_names[i] for i in visual_idx]}")
    print(f"  Somatosensory channels found: {[ch_names[i] for i in somato_idx]}")
    
    # Extract cross-correlation block
    if len(visual_idx) > 0 and len(somato_idx) > 0:
        cross_corr = corr_matrix[np.ix_(visual_idx, somato_idx)]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 5))
        
        # Plot 1: Full correlation matrix
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(ch_names)))
        ax1.set_yticks(range(len(ch_names)))
        ax1.set_xticklabels(ch_names, rotation=90, fontsize=8)
        ax1.set_yticklabels(ch_names, fontsize=8)
        ax1.set_title('Full Correlation Matrix', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Correlation')
        
        # Plot 2: Cross-correlation (Visual vs Somatosensory)
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(cross_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax2.set_xticks(range(len(somato_idx)))
        ax2.set_yticks(range(len(visual_idx)))
        ax2.set_xticklabels([ch_names[i] for i in somato_idx], rotation=90)
        ax2.set_yticklabels([ch_names[i] for i in visual_idx])
        ax2.set_xlabel('Somatosensory Channels')
        ax2.set_ylabel('Visual Channels')
        ax2.set_title('Visual-Somatosensory Cross-Correlation', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Correlation')
        
        # Plot 3: Histogram of correlations
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(cross_corr.flatten(), bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='k', linestyle='--', linewidth=2)
        ax3.set_xlabel('Correlation Coefficient')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Cross-Correlations', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(figuredir, f'correlation_strength{suffix}.png'))
        plt.close()
        
        # Find strongest correlations
        abs_cross_corr = np.abs(cross_corr)
        flat_idx = np.argsort(abs_cross_corr.flatten())[-10:][::-1]
        row_idx, col_idx = np.unravel_index(flat_idx, cross_corr.shape)
        
        print(f"\nTop 10 strongest correlations (Visual-Somatosensory):")
        for i, (r, c) in enumerate(zip(row_idx, col_idx)):
            vis_ch = ch_names[visual_idx[r]]
            som_ch = ch_names[somato_idx[c]]
            corr_val = cross_corr[r, c]
            print(f"  {i+1}. {vis_ch} <-> {som_ch}: {corr_val:+.4f}")
        
        # Statistics
        print(f"\nCross-correlation statistics:")
        print(f"  Mean absolute correlation: {np.mean(np.abs(cross_corr)):.4f}")
        print(f"  Max absolute correlation: {np.max(np.abs(cross_corr)):.4f}")
        print(f"  Std of correlations: {np.std(cross_corr):.4f}")
    
    else:
        print("Warning: Not enough channels found in specified groups")
    
    return corr_matrix


def compare_cov(cov_data, cov_noise, epochs, n_components=5, save_figures=False, figuredir=None, suffix=''):
    """
    Compare data covariance matrix to the noise covariance matrix. 
    Are there overlaps in patterns?
    
    Parameters
    ----------
    cov_data : mne.Covariance
        Data covariance matrix
    cov_noise : mne.Covariance
        Noise covariance matrix
    epochs : mne.Epochs
        Epochs object
    n_components : int
        Number of components to compare
    """
    print("\n" + "="*60)
    print("COMPARING DATA VS NOISE COVARIANCE MATRICES")
    print("="*60)
    
    # Compute SVD for both
    print("\nData Covariance:")
    U_data, S_data, _ = compute_svd(cov_data)
    
    print("\nNoise Covariance:")
    U_noise, S_noise, _ = compute_svd(cov_noise)
    
    # Get covariance matrices
    if isinstance(cov_data, mne.Covariance):
        cov_data_mat = cov_data.data
        cov_noise_mat = cov_noise.data
    else:
        cov_data_mat = cov_data
        cov_noise_mat = cov_noise
    
    # Compare singular value spectra
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Singular value comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(S_data, 'o-', label='Data', linewidth=2, markersize=6)
    ax1.plot(S_noise, 's-', label='Noise', linewidth=2, markersize=6)
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Value Spectra', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio of singular values
    ax2 = plt.subplot(2, 3, 2)
    ratio = S_data / S_noise
    ax2.plot(ratio, 'o-', linewidth=2, markersize=6, color='purple')
    ax2.axhline(y=1, color='k', linestyle='--', label='Equal strength')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Ratio (Data/Noise)')
    ax2.set_title('Signal-to-Noise Ratio per Component', fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance explained comparison
    ax3 = plt.subplot(2, 3, 3)
    var_data = np.cumsum(S_data**2) / np.sum(S_data**2) * 100
    var_noise = np.cumsum(S_noise**2) / np.sum(S_noise**2) * 100
    ax3.plot(var_data, 'o-', label='Data', linewidth=2, markersize=6)
    ax3.plot(var_noise, 's-', label='Noise', linewidth=2, markersize=6)
    ax3.axhline(y=95, color='g', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Component Index')
    ax3.set_ylabel('Cumulative Variance Explained (%)')
    ax3.set_title('Cumulative Variance', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Compare topographies of first few components
    info = epochs.info
    for i in range(min(n_components, 3)):
        ax = plt.subplot(2, 3, 4 + i)
        
        # Correlation between eigenvectors
        corr = np.abs(np.dot(U_data[:, i], U_noise[:, i]))
        
        # Plot difference in patterns
        diff_pattern = U_data[:, i] - U_noise[:, i]
        
        plot_topomap(
            diff_pattern,
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap='RdBu_r',
            sensors=True
        )
        
        ax.set_title(f'Component {i+1} Difference\n'
                    f'Pattern Correlation: {corr:.3f}',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(figuredir, f'compare_cov{suffix}.png'))
    plt.show()
    
    # Calculate pattern correlations
    print(f"\nPattern Similarity (first {n_components} components):")
    print("  Component | Correlation | Interpretation")
    print("  " + "-"*50)
    
    for i in range(n_components):
        corr = np.abs(np.dot(U_data[:, i], U_noise[:, i]))
        interpretation = "High overlap" if corr > 0.7 else "Moderate overlap" if corr > 0.4 else "Low overlap"
        print(f"  {i+1:>9} | {corr:>11.3f} | {interpretation}")
    
    # Compute Frobenius norm difference
    frobenius_diff = np.linalg.norm(cov_data_mat - cov_noise_mat, 'fro')
    frobenius_data = np.linalg.norm(cov_data_mat, 'fro')
    frobenius_noise = np.linalg.norm(cov_noise_mat, 'fro')
    
    print(f"\nMatrix-level comparison:")
    print(f"  Frobenius norm (Data): {frobenius_data:.2e}")
    print(f"  Frobenius norm (Noise): {frobenius_noise:.2e}")
    print(f"  Frobenius norm (Difference): {frobenius_diff:.2e}")
    print(f"  Relative difference: {frobenius_diff/frobenius_data*100:.2f}%")
    
    # Check if data has signal above noise
    signal_above_noise = np.sum(ratio > 1.5)
    print(f"\nSignal detection:")
    print(f"  Components with >1.5x noise level: {signal_above_noise}/{len(ratio)}")
    print(f"  Strongest signal-to-noise ratio: {np.max(ratio):.2f}")
    
    return ratio


def analyze_covariance_matrices(epochs, data_cov, noise_cov, figure_dir, save_figures=True, suffix=''):
    """
    Complete analysis pipeline for covariance matrices.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    data_cov
    noise_cov
    save_figures : bool
        Whether to save figures
    figure_dir : str
        Directory to save figures
    """
 
    if suffix == 'preprestim':
        suffix = '_preprestim'

    # Rank evaluation
    print("\n" + "="*60)
    print("RANK EVALUATION - DATA COVARIANCE")
    print("="*60)
    eff_rank_data, S_data = rank_evaluation(data_cov, save_figures=save_figures, figuredir=figure_dir, add_on="DATA", suffix=suffix)
    
    print("\n" + "="*60)
    print("RANK EVALUATION - NOISE COVARIANCE")
    print("="*60)
    eff_rank_noise, S_noise = rank_evaluation(noise_cov, save_figures=save_figures, figuredir=figure_dir, add_on="NOISE", suffix=suffix)
    
    # Eigenvector evaluation
    print("\n" + "="*60)
    print("EIGENVECTOR TOPOGRAPHIES - DATA COVARIANCE")
    print("="*60)
    U_data, S_data = eigenvector_evaluation(data_cov, epochs, n=10, save_figures=save_figures, figuredir=figure_dir, suffix=suffix)
    
    # Correlation strength
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    corr_matrix = plot_correlation_strength(data_cov, epochs, save_figures=save_figures, figuredir=figure_dir, suffix=suffix)
    
    # Compare data vs noise
    print("\n" + "="*60)
    print("DATA VS NOISE COMPARISON")
    print("="*60)
    snr_ratio = compare_cov(data_cov, noise_cov, epochs, n_components=10, save_figures=save_figures, figuredir=figure_dir, suffix=suffix)
    
    return {
        'noise_cov': noise_cov,
        'data_cov': data_cov,
        'eff_rank_data': eff_rank_data,
        'eff_rank_noise': eff_rank_noise,
        'snr_ratio': snr_ratio,
        'corr_matrix': corr_matrix
    }