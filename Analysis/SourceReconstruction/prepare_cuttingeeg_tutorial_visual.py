"""
File: prepare_cuttingeeg_tutorial.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-09-07
Description: Prepare dataset for CuttingEEG 2026 tutorial.

- morph to average
- epochs: shorten them to prestimulus, cue evoked
- forward model
"""

import mne
import os

output_folder = "/data/tu_grossmannr/Conferences/CuttingEEG2026/cuttingeeg-2026-sourcerec-tutorial/data_visual/"
subID_mecha = [30, 32, 40, 16, 17, 18, 19, 24] #, 26, 27, 28, 29, 34, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52] #
modality = 'mecha'
suffix = '_preprestim_corrected' # could be "_prestim" / "preprestim"
pick_ori = 'fixed' #None #"max-power"
avg_mode = 'mean' # options are: 'max', 'mean', 'mean_flip', 'pca_flip', None, and 'auto'
tmin_baseline = -0.1
tmax_baseline = 0
l_freq_orig = 0.1
l_freq = 1
h_freq = 40
noise_tmin, noise_tmax = -0.5, -0.43
data_tmin, data_tmax = -0.4, 0.3 
cue_time = -0.43  # visual cue onset relative to the tactile stimulus (mecha: 30 ms stimulator delay)

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
RECONDIR = os.getenv('SUBJECTS_DIR')
WDDIR = os.getenv('WDDIR')
src_file_fsaverage = f'{RECONDIR}/fsaverage_data/bem/fsaverage-ico-5-src.fif'


def sub_out_dir(subID):
    """BIDS-style output folder for one subject in the tutorial dataset."""
    out_dir = os.path.join(output_folder, f'sub-0{subID}', f'ses-{modality}', 'eeg')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# %%

# we want to morphe each subject results to the average brain
# fsaverage source space (higher density for stats)
fsaverage_src = mne.read_source_spaces(src_file_fsaverage)

# Load fsaverage morph per subject
morphs = {}
for subID in subID_mecha:
    src_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-ico-5-src.fif')
    sub_src = mne.read_source_spaces(src_file)
    
    #morphs[subID] = mne.compute_source_morph(
    #    sub_src, 'fsaverage', spacing=fsaverage_src,
    #    subjects_dir=RECONDIR, zoom=0.8  # Smooth morph
    #)
    morphs[subID] = mne.compute_source_morph(
    sub_src,
    subject_from=f"sub-0{subID}",
    subject_to="fsaverage",
    src_to=fsaverage_src,
    subjects_dir=RECONDIR,
    )

    # save the morphed data in the tutorial folder
    morph_file = os.path.join(
        sub_out_dir(subID), f'sub-0{subID}_ses-{modality}_task-NT_morph.h5'
    )
    morphs[subID].save(morph_file, overwrite=True)
    print(f"saved morph: {morph_file}")


# %%

# load epochs
for subID in subID_mecha:

    sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')
    all_epochs_filename = (
        f"sub-0{subID}_ses-{modality}_task-NT_"
        f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{suffix}"
    )
    epoch_fif = os.path.join(sub_dir, f"{all_epochs_filename}.fif")
    if not os.path.exists(epoch_fif):
        raise FileNotFoundError(f"Epoch file missing: {epoch_fif}")

    epochs = mne.read_epochs(epoch_fif, preload=True)

    # keep only clean trials with a Yes/No response
    epochs = epochs["analysis_excluded==False"]
    epochs = epochs["drop_epoch==False"]
    epochs = epochs[epochs.metadata['response_decoded'].isin(['No', 'Yes'])]

    # the tutorial only needs the scalp electrodes (drops ECG, VEOG, CNAP, RESP)
    epochs.pick('eeg')

    # The preprocessed files are tagged FIFF_UNIT_V, but the values are numerically microvolts
    # (the units survive the EEGLAB import in step001_convert_ses_to_set.m without a rescale).
    # Convert to real volts so the tutorial data are in MNE standard units. Check the assumption
    # instead of silently rescaling: microvolt EEG sits around a few units of std.
    eeg_std = epochs.get_data(picks='eeg').std()
    if not 0.1 < eeg_std < 200:
        raise ValueError(
            f"sub-0{subID}: EEG std {eeg_std:.3g} is outside the expected microvolt range, "
            "so the data are not in the assumed units"
        )
    epochs.apply_function(lambda x: x * 1e-6, picks='eeg')

    # cut them, align them to the pre-stim cue
    # keep everything before the tactile stimulus, then shift so the cue sits at t = 0
    epochs.crop(tmin=-0.9, tmax=0.0)
    epochs.shift_time(-cue_time, relative=True)
    # (-0.1, 0) after the shift is the mecha pre-cue window (-0.53, -0.43) in the original timeline
    epochs.apply_baseline((tmin_baseline, tmax_baseline))
    # data are already average referenced; store the projector so the inverse can use it
    epochs.set_eeg_reference(ref_channels="average", projection=True)
    # EEG-only recording, so there is no device->head transform in the file, but writing a
    # forward solution requires one. For EEG it is the identity (as in the existing project
    # forward solutions).
    if epochs.info['dev_head_t'] is None:
        epochs.info['dev_head_t'] = mne.transforms.Transform('meg', 'head')

    # rename and save in tutoiral folder according to BIDS 
    epochs_file = os.path.join(
        sub_out_dir(subID), f'sub-0{subID}_ses-{modality}_task-NT_epo.fif'
    )
    epochs.save(epochs_file, overwrite=True)
    print(f"saved epochs: {epochs_file} ({len(epochs)} trials, {epochs.tmin} to {epochs.tmax} s)")


# %% 

# precompute forward model using cut eeg epochs
for subID in subID_mecha:

    out_dir = sub_out_dir(subID)
    epochs_file = os.path.join(out_dir, f'sub-0{subID}_ses-{modality}_task-NT_epo.fif')
    epochs = mne.read_epochs(epochs_file, preload=False)

    trans_file = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}', f'sub-0{subID}_ses-{modality}_task-NT_epochs-trans.fif')
    bem_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-5120-5120-5120-bem-sol.fif')
    src_file = os.path.join(RECONDIR, f'sub-0{subID}', 'bem', f'sub-0{subID}-ico-5-src.fif')
    for f in (trans_file, bem_file, src_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file for sub-0{subID}: {f}")

    bem = mne.read_bem_solution(bem_file)
    src = mne.read_source_spaces(src_file)

    # info of the cut epochs, so the leadfield matches the tutorial data exactly
    fwd_solution = mne.make_forward_solution(epochs.info, trans=trans_file, src=src, bem=bem, eeg=True)
    # fixed orientation (one dipole per vertex, normal to the cortex): what the tutorial uses
    fwd_solution = mne.convert_forward_solution(
        fwd_solution, surf_ori=True, force_fixed=True, use_cps=True
    )
    # write_forward_solution stores the *original* (free-orientation) leadfield kept in the
    # private '_orig_*' fields and re-converts on read, so the conversion above alone would not
    # end up in the file. Point those fields at the fixed leadfield to actually ship a fixed
    # forward (a third of the leadfield size, and no conversion step in the tutorial).
    fwd_solution['_orig_sol'] = fwd_solution['sol']['data'].copy()
    fwd_solution['_orig_source_ori'] = mne.io.constants.FIFF.FIFFV_MNE_FIXED_ORI

    # save forward model in tutorial folder
    fwd_file = os.path.join(out_dir, f'sub-0{subID}_ses-{modality}_task-NT_fwd.fif')
    mne.write_forward_solution(fwd_file, fwd_solution, overwrite=True)
    print(f"saved forward solution: {fwd_file}")
