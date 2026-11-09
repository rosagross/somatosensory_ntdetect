"""
File: step010_cnap_group_average.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-03-13
Description:
- Load combined CNAP epochs (.fif) per subject/session
- Exclude trials with intensity_level == 0
- Plot and save one grand-average CNAP per session type
"""

# %%
import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from helper_functions_eeg import assign_modality

DATADIR = os.getenv("DATADIR")
WDDIR = os.getenv("WDDIR")

# Read in preprocessing parameters
with open(os.path.join(WDDIR, "Pre-Processing", "preproc_parameter.json"), "r") as json_file:
    params = json.load(json_file)

session_type = params["session_type"]

modalities = ["Electrical"]  # "Electrical" 
subIDs = np.arange(13, 53)

# %%
for modality in modalities:
    ses_label, _ = assign_modality(modality)

    epoch_files = []
    for subID in subIDs:
        subject_label = f"sub-0{subID}"
        epoch_file = os.path.join(
            DATADIR,
            "derivatives",
            "cnap-preproc",
            subject_label,
            f"ses-{ses_label}",
            f"{subject_label}_ses-{ses_label}_task-{session_type}_cnap-epochs.fif",
        )
        if os.path.exists(epoch_file):
            epoch_files.append(epoch_file)

    if not epoch_files:
        print(f"No CNAP epoch files found for ses-{ses_label}")
        continue

    all_epoch_data = []
    times = None
    all_intensity_levels = []

    for epoch_fif in epoch_files:
        epochs = mne.read_epochs(epoch_fif, preload=True, verbose="ERROR")
        # high-pass filter at 150 Hz for CNAP (CNAP is stored as misc)
        epochs.filter(l_freq=300, h_freq=None, method="iir", picks="misc", verbose="ERROR")

        if epochs.metadata is None:
            raise ValueError(f"Missing metadata in {epoch_fif}")

        if "intensity_level" not in epochs.metadata.columns:
            if "trigger_meaning" not in epochs.metadata.columns:
                raise ValueError(
                    f"Missing intensity_level and trigger_meaning metadata in {epoch_fif}"
                )

            # Derive intensity_level from trigger_meaning when it is not present.
            # Mechanical: intensity row comes AFTER the stimulation row.
            # Electrical: intensity row comes BEFORE the stimulation row.
            trigger_meaning = epochs.metadata["trigger_meaning"].astype(str).to_list()
            intensity_level = np.zeros(len(trigger_meaning), dtype=int)

            modality_key = modality.strip().lower()
            if modality_key.startswith("mech"):
                intensity_offset = 1
            elif modality_key.startswith("elec"):
                intensity_offset = -1
            else:
                raise ValueError(
                    f"Unknown modality '{modality}' for intensity parsing in {epoch_fif}"
                )

            for i, meaning in enumerate(trigger_meaning):
                if "Stimulation" not in meaning:
                    continue

                idx = i + intensity_offset
                if idx < 0 or idx >= len(trigger_meaning):
                    intensity_level[i] = 0
                    continue

                neighbor = trigger_meaning[idx]
                if "IntensLevel" in neighbor:
                    try:
                        intensity_level[i] = int(neighbor.replace("IntensLevel", ""))
                    except ValueError:
                        intensity_level[i] = 0
                else:
                    intensity_level[i] = 0

            epochs.metadata = epochs.metadata.copy()
            epochs.metadata["intensity_level"] = intensity_level


        # --- CNAP noise rejection (method 1 + 2) ---
        # Method 1: Baseline peak-to-peak (pre-stim window)
        # Method 2: Baseline HF RMS (500-1000 Hz band if available)
        # Use robust thresholds within each subject/session file.
        times = epochs.times
        baseline_mask = (times >= -0.02) & (times <= -0.002)
        if baseline_mask.sum() < 2:
            raise ValueError(
                f"Baseline window too small for CNAP QC in {epoch_fif} "
                f"(found {baseline_mask.sum()} samples)."
            )

        # CNAP is stored as a single misc channel; keep shape (n_epochs, n_times)
        data = epochs.get_data(picks="misc")[:, 0, :]
        baseline = data[:, baseline_mask]
        ptp_baseline = np.ptp(baseline, axis=1)

        sfreq = epochs.info["sfreq"]
        nyquist = sfreq / 2.0
        hf_low = 500.0
        hf_high = min(1000.0, nyquist - 1.0)
        if hf_high > hf_low:
            data_hf = mne.filter.filter_data(
                data,
                sfreq=sfreq,
                l_freq=hf_low,
                h_freq=hf_high,
                verbose="ERROR",
            )
            baseline_hf = data_hf[:, baseline_mask]
            rms_hf = np.sqrt(np.mean(baseline_hf**2, axis=1))
        else:
            # If sampling is too low for the intended HF band, fall back to broadband baseline RMS.
            rms_hf = np.sqrt(np.mean(baseline**2, axis=1))

        def robust_threshold(values, k=6.0):
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            # If MAD collapses to 0, avoid rejecting everything by using a tiny epsilon.
            if mad == 0:
                return med + 1e-12
            return med + k * mad

        ptp_thr = robust_threshold(ptp_baseline, k=6.0)
        rms_thr = robust_threshold(rms_hf, k=6.0)
        bad_cnap = (ptp_baseline > ptp_thr) | (rms_hf > rms_thr)

        epochs.metadata = epochs.metadata.copy()
        epochs.metadata["bad_cnap"] = bad_cnap

        mask = (epochs.metadata["intensity_level"] != 0) & (~epochs.metadata["bad_cnap"])
        epochs = epochs[mask]

        if len(epochs) == 0:
            continue

        data = epochs.get_data()  # shape: (n_epochs, 1, n_times)
        all_epoch_data.append(data)
        times = epochs.times
        all_intensity_levels.append(epochs.metadata["intensity_level"].to_numpy())

    if not all_epoch_data:
        print(f"No valid CNAP epochs after filtering for ses-{ses_label}")
        continue

    all_epochs = np.concatenate(all_epoch_data, axis=0)
    avg_cnap = np.nanmean(all_epochs, axis=0).squeeze()

    output_dir = os.path.join(DATADIR, "derivatives", "cnap-preproc", "group")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.plot(times * 1000, avg_cnap, color="black", linewidth=1)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Time (ms)")
    plt.ylabel("CNAP (uV)")
    plt.title(f"CNAP grand average ses-{ses_label}")
    plt.tight_layout()

    fig_name = f"group_ses-{ses_label}_task-{session_type}_cnap-average.png"
    plt.savefig(os.path.join(output_dir, fig_name), dpi=200)
    plt.show()

    # plot each intensity separately
    if all_intensity_levels:
        all_intensity = np.concatenate(all_intensity_levels, axis=0)
        intensity_levels = sorted([lvl for lvl in np.unique(all_intensity) if lvl != 0])

        plt.figure(figsize=(6, 3))
        for lvl in intensity_levels:
            lvl_mask = all_intensity == lvl
            avg_lvl = np.nanmean(all_epochs[lvl_mask, :, :], axis=0).squeeze()
            plt.plot(times * 1000, avg_lvl, linewidth=1, label=f"Level {lvl}")

        plt.axvline(0, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Time (ms)")
        plt.ylabel("CNAP (uV)")
        plt.title(f"CNAP grand average by intensity ses-{ses_label}")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()

        fig_name = f"group_ses-{ses_label}_task-{session_type}_cnap-average_by-intensity.png"
        plt.savefig(os.path.join(output_dir, fig_name), dpi=200)
        plt.show()

# %%
