######################################
# We compute the forward solution.
# Make sure you already computed:
# - covariance matrix
# - Segmentation with Freesurfer
# - BEM 
#  
# Before you run this code: 
# - check if you have the following files:
#   - transform file (COR.fif)
#   - BEM file (bem/...-bem-sol.fif)
#   - source file (bem/-src.fif)
#
# Summary of forward model computation:
# - N sources (dipoles)
# - M sensors (EEG electrodes)
# - G = M x N (lead field matrix) --> this is the output of this script
######################################

# %%
import os
import sys
import mne  
import numpy as np 
import helper_functions_sourcerec as hfs

# Ensure repo root is on sys.path so general_helpers can be imported reliably.
WDDIR = os.getenv('WDDIR')
if not WDDIR:
    raise EnvironmentError("WDDIR is not set. Please export WDDIR to the repo root.")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import parse_cli_args, is_subject_excluded

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
RECONDIR = os.getenv('SUBJECTS_DIR')

# this actually does not matter because only the info not the data is used
l_freq, h_freq, suffix, modalities, _, _ = parse_cli_args(
    description=(
        "Compute forward solutions. Optional arguments override "
        "frequency cutoffs, file suffix, and modalities."
    ),
    default_l_freq=1.0,
    default_h_freq=40.0,
    default_file_suffix="_preprestim",
    default_modalities=["mecha", "elec"],
)

# subejcts that have the coordinate system transform file ready for both sessions
subID_both = np.arange(13,53)
l_freq_orig = 0.1

# %%
for subID in subID_both:
    for modality in modalities:

        # Sessions on Pre-Processing/exclued_subs.json (low post-cue ERP SNR in O1/O2).
        if is_subject_excluded(subID, modality):
            print(f"Skipping sub-0{subID} {modality}: on the exclusion list.")
            continue

        # load necessary files
        trans_file = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}', f'sub-0{subID}_ses-{modality}_task-NT_epochs-trans.fif')
        #trans_file = os.path.join(RECONDIR, f'sub-0{subID}', 'mri', 'T1-neuromag', 'sets', f'ses-{modality}', f'COR-ses-{modality}.fif')
        bem_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-5120-5120-5120-bem-sol.fif')
        src_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-ico-5-src.fif')

        # check if trans file exists
        if trans_file is None or not os.path.exists(trans_file):
            print(f"Trans file for subject {subID} modality {modality} not found, skipping...")
            continue

        # read in BEM meshes and the computed source space 
        if bem_file is None or not os.path.exists(bem_file):
            print(f"BEM file for subject {subID} modality {modality} not found, skipping...")
            continue
        bem = mne.read_bem_solution(bem_file)
        src = mne.read_source_spaces(src_file)

        # read epochs 
        sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')

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
        hfs.apply_baseline_epochs(epochs, epoch_fif)
        epochs_clean = epochs["analysis_excluded==False"]
        epochs_clean = epochs_clean["drop_epoch==False"]   

        if len(epochs_clean) == 0:
            print("no epochs left after cleaning, skipping")
            continue
            
        evoked_clean = epochs_clean.average()

        # compute forward solution
        fwd_file = os.path.join(sub_dir, f'sub-0{subID}_ico-5_ses-{modality}_fwd.fif')
        fwd_solution = mne.make_forward_solution(evoked_clean.info, trans=trans_file, src=src, bem=bem, eeg=True)
        mne.write_forward_solution(fwd_file, fwd_solution, overwrite=True)

# %%
