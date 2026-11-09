"""
File: step002_extract_freq_bands.py
Author: Rosa Grossmann
Date: 2026-03-19
Description:
- Load prestimulus-corrected epochs and save band-pass filtered epochs per subject and modality.
- double check metadata/epoch alignment.
"""

# %%
import argparse
import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so general_helpers can be imported reliably.
WDDIR = os.getenv('WDDIR')
if not WDDIR:
    raise EnvironmentError("WDDIR is not set. Please export WDDIR to the repo root.")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import parse_cli_args #, apply_baseline_epochs

RESDIR = os.getenv("RESDIR")
DATADIR = os.getenv("DATADIR")

# Processing parameters
# --sub selects subjects (e.g. one SLURM array task per subject); default is all.
# parse_cli_args uses parse_known_args, so both parsers coexist.
_sub_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_sub_parser.add_argument("--sub", nargs="+", type=int, default=None)
_sub_parser.add_argument(
    "--subtract-erp", action="store_true",
    help="Subtract the condition-wise (Yes/No) mean ERP before cropping and filtering "
         "the filter-first epochs (default: off).",
)
_sub_args, _ = _sub_parser.parse_known_args()
subjects = np.arange(13, 53)
if _sub_args.sub is not None:
    invalid = [s for s in _sub_args.sub if s not in subjects]
    if invalid:
        raise ValueError(f"Subject ids {invalid} outside the valid range 13-52.")
    subjects = np.array(_sub_args.sub)
l_freq, h_freq, file_suffix_epochs, modalities, file_suffix_meta, _ = parse_cli_args(
    description=(
        "Extract frequency bands from prestimulus epochs. Optional arguments "
        "override frequency cutoffs, file suffixes, and modalities."
    ),
    default_l_freq=1,
    default_h_freq=40,
    default_file_suffix="_preprestim_corrected",
    default_modalities=["elec", "mecha"],
    default_meta_suffix="_preprestim",
)

# Bands (Hz)
bands = {
    "alpha": (8, 12)
}

l_freq_orig = 0.1
prestim_crop_tmin = -0.8  # used only for alpha-peak PSD, not for the saved filtered signal
prestim_crop_tmax = -0.1 # used only for alpha-peak PSD, not for the saved filtered signal
filter_crop_tmax = 0 # use for e

# %%

def load_and_validate_epochs(epoch_fif, meta_tsv, subID, modality):
    """
    Load epochs and metadata, apply metadata filters to match cleaned epochs,
    and enforce strict alignment.
    """
    epochs = mne.read_epochs(epoch_fif, preload=True, verbose=False)
    metadata_df = pd.read_csv(meta_tsv, sep="\t")
    
    epochs = epochs[epochs.metadata['analysis_excluded'] == False]
    epochs = epochs[epochs.metadata['drop_epoch'] == False]

    metadata_df = metadata_df[
        (metadata_df["drop_epoch"] == False)
        & (metadata_df["analysis_excluded"] == False)
    ].reset_index(drop=True)

    # Match corrected epochs: drop bad epochs, analysis-excluded, and catch trials.
    if len(epochs) != len(metadata_df):
        raise ValueError(
            f"Metadata rows ({len(metadata_df)}) and epochs ({len(epochs)}) differ "
            f"for subject {subID} modality {modality}."
        )

    return epochs


# %%
for modality in modalities:
    for subID in subjects:
        # exclude subject 23 mechanical (swapped ground and reference electrode)
        # exclude mecha 36 (force values were exceptionally high!)
        if ((subID == 23) or (subID == 36)) and modality == "mecha":
            print(f"Skipping subject {subID} modality {modality} due to faulty recording setup.")
            continue

        sub_dir = os.path.join(
            DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", f"ses-{modality}"
        )

        base_name = (
            f"sub-0{subID}_ses-{modality}_task-NT_"
            f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}"
        )
        epoch_fif = os.path.join(sub_dir, f"{base_name}_epochs{file_suffix_epochs}.fif")
        meta_tsv = os.path.join(sub_dir, f"{base_name}_allmetadata{file_suffix_meta}.tsv")

        if not os.path.exists(epoch_fif):
            print(f"Epoch file not found for subject {subID} modality {modality}, skipping...")
            continue
        if not os.path.exists(meta_tsv):
            print(f"Metadata file not found for subject {subID} modality {modality}, skipping...")
            continue

        print(f"Processing subject {subID}, modality {modality}")
        epochs = load_and_validate_epochs(epoch_fif, meta_tsv, subID, modality)
        
        # check if there are still epochs to work with after exclusion 
        if epochs._data.size == 0:
            print('No data left! Sub:', subID, modality)
            continue
        
        # set up figure dir
        figuredir = os.path.join(RESDIR, "Figures", "Main", "eeg_results", "frequency_analysis")
        figuredir_suffix = os.path.join(figuredir, f"window_{file_suffix_epochs}", "single")
        if not os.path.exists(figuredir_suffix):
            os.makedirs(figuredir_suffix)

        # crop and pad epochs
        epochs_orig = epochs.copy()  # keep full epochs for filter-first variant
        epochs = epochs.copy().crop(tmin=prestim_crop_tmin, tmax=prestim_crop_tmax)
        # Mirror-pad each epoch by one epoch length on both sides to absorb filter edge transients.
        n_pad = epochs._data.shape[-1]
        padded = np.pad(epochs._data, ((0, 0), (0, 0), (n_pad, n_pad)), mode='reflect')
        new_tmin = epochs.tmin - n_pad / epochs.info['sfreq']
        epochs = mne.EpochsArray(
            padded, epochs.info, tmin=new_tmin, metadata=epochs.metadata, verbose=False
        )
        

        # Compute PSD (Welch method) on the prestim window only.
        epochs_psd = epochs.copy()
        picks = ["CP4", "C6", "CP3", "O2", "CPz"]
        epochs_picked = epochs_psd.copy().pick(picks)
        psd = epochs_picked.compute_psd(method='welch', fmin=1, fmax=40)
        psd_data = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)
        freqs = psd.freqs
        psd_mean = psd_data.mean(axis=0)
        psd_mean_db = 10 * np.log10(psd_mean)
        alpha_mask = (freqs >= 8) & (freqs <= 14)
        alpha_freqs = freqs[alpha_mask]
        alpha_peaks = {}

        for i, ch_name in enumerate(psd.ch_names):
            ch_power = psd_mean[i, alpha_mask]
            
            peak_freq = alpha_freqs[np.argmax(ch_power)]
            peak_power = np.max(ch_power)
            
            alpha_peaks[ch_name] = (peak_freq, peak_power)

        print(alpha_peaks)


        plt.figure()

        for i, ch in enumerate(psd.ch_names):
            line = plt.plot(alpha_freqs, psd_mean_db[i][alpha_mask], label=ch)
            peak_freq, peak_power = alpha_peaks[ch]
    
            # plot vertical line at peak frequency
            plt.axvline(peak_freq, 
                        linestyle='--', 
                        color=line[0].get_color(), 
                        alpha=0.8)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title(f"PSD - Subject {subID} ({modality})")
        plt.legend()
        plt.grid()
        filename = f"sub-{subID}_{modality}_PSD.png"
        plt.savefig(os.path.join(figuredir_suffix, filename), dpi=300, bbox_inches='tight')
        plt.close()

        # use the alpha peak freq of CP4 as window middle point
        fmin_cp4 = alpha_peaks["CP4"][0] -2
        fmax_cp4 = alpha_peaks["CP4"][0] +2

        # Save band-pass filtered epochs per band and per analysis window.
        for band_name, (fmin, fmax) in bands.items():
            
            if band_name=="alpha":
                fmin = fmin_cp4
                fmax = fmax_cp4
            
            
            # Filter the full epoch — surrounding data absorbs edge transients.
            # No crop here: step008_1 handles all window cropping.
            band_epochs = epochs.copy().filter(
                l_freq=fmin,
                h_freq=fmax,
                picks="eeg",
                method="fir",
                pad="reflect_limited",
                phase="zero",
                verbose=False,
            )

            band_out = os.path.join(
                sub_dir,
                f"{base_name}_epochs{file_suffix_epochs}_{band_name}.fif",
            )
            band_epochs.save(band_out, overwrite=True)
            print(f"  Saved {band_name} band epochs: {band_out}")

        # --- crop-first variant: crop, pad, filter ---
        epochs_orig = epochs_orig.crop(tmin=prestim_crop_tmin, tmax=filter_crop_tmax)
        n_pad_orig = epochs_orig._data.shape[-1]
        padded_orig = np.pad(epochs_orig, ((0, 0), (0, 0), (n_pad_orig, n_pad_orig)), mode='reflect')
        new_tmin_orig = epochs_orig.tmin - n_pad_orig / epochs_orig.info['sfreq']
        epochs_orig_padded = mne.EpochsArray(
            padded_orig, epochs_orig.info, tmin=new_tmin_orig, metadata=epochs_orig.metadata, verbose=False
        )

        for band_name, (fmin, fmax) in bands.items():
            if band_name == "alpha":
                fmin = fmin_cp4
                fmax = fmax_cp4

            band_epochs_ff = epochs_orig_padded.copy().filter(
                l_freq=fmin, h_freq=fmax, picks="eeg",
                method="fir", pad="reflect_limited", phase="zero", verbose=False,
            ).crop(tmin=prestim_crop_tmin, tmax=filter_crop_tmax)

            band_out_ff = os.path.join(
                sub_dir,
                f"{base_name}_epochs{file_suffix_epochs}_{band_name}.fif",
            )
            band_epochs_ff.save(band_out_ff, overwrite=True)
            print(f"  Saved {band_name} band epochs (filter-first): {band_out_ff}")

# %%
