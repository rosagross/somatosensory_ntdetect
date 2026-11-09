"""
File: step009_cnap_extraction.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-03-12
Description:
- Load CNAP data produced in step004
- Read event files *_events_cnaponly.csv
- Extract CNAP epochs around events (-50 to 50 ms)
- Save epochs in derivatives folder
- Plot and save the average CNAP signal (one plot per subject/session)
"""

# %%
import os
import json

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from helper_functions_eeg import assign_modality


DATADIR = os.getenv("DATADIR")
WDDIR = os.getenv("WDDIR")

# Read in preprocessing parameters
with open(os.path.join(WDDIR, "Pre-Processing", "preproc_parameter.json"), "r") as json_file:
    params = json.load(json_file)

session_type = params["session_type"]

# Subjects and modalities to process
subIDs = np.arange(13, 53)
modalities = ["Mechanical", "Electrical"]  # add "Electrical" if needed

# Epoch window in seconds
tmin = -0.01
tmax = 0.03

# %%
for modality in modalities:
    ses_label, _ = assign_modality(modality)

    for subID in subIDs:
        subID = str(subID)
        subject_label = f"sub-0{subID}"

        input_dir = os.path.join(DATADIR, subject_label, f"ses-{ses_label}", "eeg")
        output_dir = os.path.join(
            DATADIR, "derivatives", "cnap-preproc", subject_label, f"ses-{ses_label}"
        )
        os.makedirs(output_dir, exist_ok=True)

        all_epoch_data = []
        all_epoch_times = None
        all_epoch_metadata = []

        # process all blocks if present
        for block in range(1, 9):
            cnap_file = os.path.join(
                input_dir,
                f"{subject_label}_ses-{ses_label}_task-{session_type}_run-{block}_cnap.tsv.gz",
            )
            cnap_json = os.path.join(
                input_dir,
                f"{subject_label}_ses-{ses_label}_task-{session_type}_run-{block}_cnap.json",
            )
            events_file = os.path.join(
                input_dir,
                f"{subject_label}_ses-{ses_label}_task-{session_type}_run-{block}_events_cnaponly.csv",
            )

            if not (os.path.exists(cnap_file) and os.path.exists(cnap_json) and os.path.exists(events_file)):
                continue

            # load CNAP data and metadata
            with open(cnap_json, "r", encoding="utf-8") as f:
                cnap_meta = json.load(f)
            sfreq = float(cnap_meta["SamplingFrequency"])

            cnap_df = pd.read_csv(cnap_file, sep="\t", header=None, compression="gzip")
            cnap_df.columns = ["cnap", "timestamp", "total_secs"]
            cnap_signal = cnap_df["cnap"].to_numpy()

            # load events
            events_df = pd.read_csv(events_file)
            event_times = events_df["total_secs"].to_numpy()

            # convert event times to sample indices
            event_samples = np.round(event_times * sfreq).astype(int)

            pre_samp = int(round(abs(tmin) * sfreq))
            post_samp = int(round(abs(tmax) * sfreq))
            n_times = pre_samp + post_samp + 1
            epoch_times = np.arange(-pre_samp, post_samp + 1) / sfreq

            # extract epochs
            epochs = np.zeros((len(event_samples), n_times), dtype=float)
            for i, center in enumerate(event_samples):
                start = center - pre_samp
                stop = center + post_samp + 1

                # data integrity: fail fast if any epoch cannot be extracted
                if start < 0 or stop > len(cnap_signal):
                    print('CNAP epoch out of bounds for {subject_label}, ses-{ses_label}, run-{block}')
                    continue
                    #raise ValueError(
                    #    f"CNAP epoch out of bounds for {subject_label}, ses-{ses_label}, run-{block}"
                    #)


                epochs[i, :] = cnap_signal[start:stop]

            # collect for overall average plot
            all_epoch_data.append(epochs)
            all_epoch_times = epoch_times
            all_epoch_metadata.append(events_df)

        # save combined epochs and plot average (only once per subject/session)
        if all_epoch_data:
            all_epochs = np.concatenate(all_epoch_data, axis=0)
            all_metadata = pd.concat(all_epoch_metadata, ignore_index=True)

            info = mne.create_info(["CNAP"], sfreq, ch_types=["misc"])
            epochs_mne = mne.EpochsArray(
                all_epochs[:, np.newaxis, :], info, tmin=tmin, metadata=all_metadata
            )
            epochs_name = (
                f"{subject_label}_ses-{ses_label}_task-{session_type}_cnap-epochs.fif"
            )
            epochs_mne.save(os.path.join(output_dir, epochs_name), overwrite=True)

            avg_cnap = np.nanmean(all_epochs, axis=0)

            plt.figure(figsize=(6, 3))
            plt.plot(all_epoch_times * 1000, avg_cnap, color="black", linewidth=1)
            plt.axvline(0, color="red", linestyle="--", linewidth=1)
            plt.xlabel("Time (ms)")
            plt.ylabel("CNAP (uV)")
            plt.title(f"{subject_label} ses-{ses_label} CNAP average")
            plt.tight_layout()

            fig_name = f"{subject_label}_ses-{ses_label}_task-{session_type}_cnap-average.png"
            plt.savefig(os.path.join(output_dir, fig_name), dpi=200)
            plt.close()

# %%
