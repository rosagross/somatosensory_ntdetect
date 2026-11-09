# ########################################################################
# COMPUTE NOISE COVARIANCE MATRIX
# - compute how the sensors relate to each other 
# - We might have rank deficient data because we applied ICA while pre-processing 
# - rank-deficient means we have less ranks (seen in the plot after cov var computation) than channels
# - we can use regularization to fix this
##########################################################################

# %%
import os
import sys
import mne
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions_eeg import analyze_covariance_matrices

# Ensure repo root is on sys.path so general_helpers can be imported reliably.
WDDIR = os.getenv('WDDIR')
if not WDDIR:
    raise EnvironmentError("WDDIR is not set. Please export WDDIR to the repo root.")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import parse_cli_args, apply_baseline_epochs

# %%

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')

# specify subjects and modality to load
subjects = np.arange(13, 53)
l_freq, h_freq, suffix, modalities, _, epoch_length = parse_cli_args(
    description=(
        "Compute noise covariance matrices. Optional arguments override "
        "frequency cutoffs, file suffix, modalities, and epoch length."
    ),
    default_l_freq=1,
    default_h_freq=40,
    default_file_suffix="_preprestim_corrected",
    default_meta_suffix="_preprestim",
    default_modalities=["mecha", "elec"],
    default_epoch_length=651,
)
l_freq_orig = 0.1
baseline = [(-0.55, -0.45), (-0.4, 0.3)]

# %%

for modality in modalities:
    
    for subID in subjects:
        print(f"Processing subject {subID} modality {modality}...")
        
        sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')
        figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'eeg_results', 'cov_matrix', f'sub-0{subID}', f'ses-{modality}')
        if not os.path.exists(figuredir):
            os.makedirs(figuredir)
        
        # read in fif epoch 
        all_epochs_filename = (
            f"sub-0{subID}_ses-{modality}_task-NT_"
            f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{suffix}"
        )
        epoch_fif = os.path.join(sub_dir, f"{all_epochs_filename}.fif")
        
        if epoch_fif is None or not os.path.exists(epoch_fif):
            print(f"Epoch file for subject {subID} modality {modality} not found, skipping...")
            continue

        epochs = mne.read_epochs(epoch_fif, preload=True)
        
        if epochs._data.shape[2]!=epoch_length:
            raise ValueError('unexpected epoch length:', epochs._data.shape[2])

        if epochs is None:
            #print(f"Epoch file for subject {subID} modality {modality} could not be loaded, skipping...")
            raise ValueError(f"Epoch file for subject {subID} modality {modality} could not be loaded, skipping...")
    
        # exclude bad epochs 
        #epochs = epochs[epochs.metadata['plot_excluded'] == False]
        epochs = epochs[epochs.metadata['analysis_excluded'] == False]
        epochs = epochs[epochs.metadata['drop_epoch'] == False]

        # check if there are still epochs to work with after exclusion 
        if epochs._data.size == 0:
            print('No data left! Sub:', subID, modality)
            continue

        # compute noise covariance from baseline period
        # also compute regularized noise covariance
        if baseline:
            noise_tmin, noise_tmax = baseline[0]
            data_tmin, data_tmax = baseline[1]
        else:
            if suffix == '_preprestim':
                # Baseline window is pre-prestim; data window is still pre-stimulus.
                noise_tmin, noise_tmax = -0.8, -0.7
                data_tmin, data_tmax = -0.7, -0.4
            elif '_prestim' in suffix:
                # Baseline window is pre-stim; data window is the visual response.
                noise_tmin, noise_tmax = -0.5, -0.4
                data_tmin, data_tmax = -0.4, 0
            elif (suffix == ''):
                # Baseline window is immediate pre-stim; data window is early post-stim.
                noise_tmin, noise_tmax = -0.1, 0
                data_tmin, data_tmax = 0, 0.3

        # apply baseline
        apply_baseline_epochs(epochs, epoch_fif, baseline_window=(noise_tmin, noise_tmax))

        # Compute covariance using the chosen time windows.
        noise_cov_baseline = mne.compute_covariance(epochs, tmin=noise_tmin, tmax=noise_tmax)
        noise_cov_reg = mne.compute_covariance(
            epochs, tmin=noise_tmin, tmax=noise_tmax, method="auto", rank=None
        )
        data_cov_reg = mne.compute_covariance(
            epochs, tmin=data_tmin, tmax=data_tmax, method="auto", rank=None
        )

        cov_reg_data_fname = os.path.join(
            sub_dir,
            f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_noise-cov-data-{data_tmin}to{data_tmax}-covreg{suffix}.fif",
        )
        mne.write_cov(cov_reg_data_fname, data_cov_reg, overwrite=True)

        evoked = epochs.average()
        #evoked.plot_white(noise_cov_reg, time_unit='s')

        # save noise and data covariance matrices
        cov_baseline_fname = os.path.join(
            sub_dir,
            f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_noise-cov-baseline-{noise_tmin}to{noise_tmax}-cov{suffix}.fif",
        )
        cov_reg_baseline_fname = os.path.join(
            sub_dir,
            f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_noise-cov-baseline-{noise_tmin}to{noise_tmax}-covreg{suffix}.fif",
        )

        mne.write_cov(cov_baseline_fname, noise_cov_baseline, overwrite=True)
        mne.write_cov(cov_reg_baseline_fname, noise_cov_reg, overwrite=True)

        # Inspect covariance marix
        #analyze_covariance_matrices(epochs, data_cov_reg, noise_cov_reg, save_figures=True, figure_dir=figuredir, suffix=suffix)

# %%
