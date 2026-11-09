"""
File: step001_forceresp_coherence.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-08-20
Description: Reading in data from the scale that measures finger pressure during the experiment. 

In this script we compute the coherence and correlation between respiration and force activity. 

"""


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import xml.etree.ElementTree as ET
import statsmodels.api as sm
from scipy import signal, interpolate
from scipy.signal import resample, correlate, correlation_lags, coherence, welch, csd
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr, zscore
from statsmodels.nonparametric.smoothers_lowess import lowess
from analysis_helper_functions import *

# %%
# set directory paths
WDDIR = os.getenv("WDDIR")
RAWDIR = os.getenv("RAWDIR")
DATADIR = os.getenv("DATADIR")
data_dir = os.path.join(RAWDIR, "FingerForceData")
RESDIR = os.getenv("RESDIR")
fig_dir = os.path.join(RESDIR, "Figures", "Main", "force_results", "single")

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# read parameters
with open(os.path.join(WDDIR, 'Pre-Processing', 'preproc_parameter.json'), 'r') as json_file:
    params = json.load(json_file)

# %%
# some subjects have a trigger, others don't
subswithtrigger = params["scale"]["subs_with_trigger"].keys()  # force trigger available for those

# if no arguments are used when starting script in commandline
# all subjects from 13 to 52 will be processed
# use --help to see options
subjects = parse_subjects()
ubjects = map(str, range(31, 32))

samplingdiff_info = []
trigger_diffs = []
processing_notes = []

#check_force_availability(subjects)

# %%

# Loop over subjects and blocks
print('subjects', subjects)
for subID in subjects:
    
    # first check if subject has triggers, if no, skip the subject
    subswithtrigger = params["scale"]["subs_with_trigger"].keys()
    if subID not in subswithtrigger:
        print(f"Subject {subID} has no trigger data")
        blocks = range(1, 9)
    else:
        blocks = params["scale"]["subs_with_trigger"][subID]
        if len(blocks) == 0:
            blocks = range(1, 9)
        print("Blocks to be processed:", blocks)

    sub_dir = os.path.join(f"sub-0{subID}", "ses-mecha", "eeg")
    print("Subject:", subID)

    # make directory for the figures
    fig_dir_sub = os.path.join(fig_dir, f"sub-0{subID}")
    if not os.path.exists(fig_dir_sub):
        os.makedirs(fig_dir_sub)

    # load behavioural data
    derivatives_dir = os.path.join(
        DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", "ses-mecha"
    )
    derivatives_dir_path = os.path.join(DATADIR, derivatives_dir, "force_resp_data")
    if not os.path.exists(derivatives_dir_path):
        os.makedirs(derivatives_dir_path)

    all_freq_spec_df = pd.DataFrame()

    for block in blocks:
        print("Block:", block)

        #######################################################
        # 1. Load Data
        #######################################################

        # Construct the file names
        # folder to read ecg & resp data from
        output_dir = os.path.join(DATADIR, f"sub-0{subID}")
        ses_dir = os.path.join(output_dir, "ses-mecha", "eeg")

        # load ecg and resp data
        physio_data_path = os.path.join(
                ses_dir, f"sub-0{subID}_ses-mecha_task-NT_run-{block}_physio.tsv.gz"
            )
        physio_meta_path = os.path.join(
                ses_dir, f"sub-0{subID}_ses-mecha_task-NT_run-{block}_physio.json"
            )

        if not os.path.exists(physio_data_path):
            print(f"No physio data for sub{subID} block {block}")
            continue

        # read in the physio data (ecg, etc.)
        physio_data, physio_metadata = read_physio_data(physio_data_path, physio_meta_path)
        if physio_data is None:
            raise ValueError(f"No valid physio data for sub{subID} block {block}")
            
        # load event annotations from eeg 
        annotations = pd.read_csv(
            os.path.join(
                ses_dir, f"sub-0{subID}_ses-mecha_task-NT_run-{block}_events.csv"
            )
        )
        # convert timestamp into datetime format
        annotations["timestamp"] = pd.to_datetime(annotations["timestamp"])
        stim_annot = annotations[
                        annotations["trigger_meaning"] == "Stimulation"
                    ]

        #########################################
        # 4. Compute Frequency Spectra and Coherence
        #########################################

        # transform the signal to have a zero mean (otherwise we will have a huge peak at 0 Hz!)
        y_uniform = physio_data["force"]

        # check if there is really some force data!
        if np.all(np.isnan(y_uniform)):
            print(f"All NaN values in force data for sub{subID} block {block}")
            continue
            #raise Exception(f"All NaN values in force data for sub{subID} block {block}")

        t_uniform = np.linspace(
            0, physio_data["total_secs"].max(), len(physio_data["total_secs"])
        )
        y_uniform_zeromean = physio_data["force_zeromean"]

        plt.figure(figsize=(10, 6))
        plt.plot(t_uniform, y_uniform_zeromean)
        plt.ylabel("Force (g)")
        plt.xlabel("Time (sec)")
        plt.savefig(
            os.path.join(
                fig_dir_sub, f"signal_nonzeromean_overview_sub{subID}_block{block}.pdf"
            )
        )
        #plt.show()

        # if there is not enoguh force data (more than 60% NaN), skip the subject
        nan_ratio = np.sum(np.isnan(y_uniform_zeromean)) / len(y_uniform_zeromean)
        if nan_ratio > 0.6:
            print(f"More than 60% NaN values in force data for sub{subID} block {block}, skipping block.")
            continue

        ###############  FREQUENCY SPECTRA  ##################
        # compute frequency spectrum of the force signal, respiration and hearbeat

        # .. force
        fs = 1 / 0.01
        window_samples = 50 * fs
        y = np.asarray(y_uniform_zeromean, dtype=float)
        y = y[np.isfinite(y)]
        xf_scale, yf_scale = welch(y, fs=fs, nperseg=window_samples)
        mask = xf_scale <= 2
        xf_scale = xf_scale[mask]
        yf_scale = yf_scale[mask]
        mask = xf_scale > 0
        xf_scale = xf_scale[mask]
        yf_scale = yf_scale[mask]
        freq_spec_df = pd.DataFrame()
        freq_spec_df["freq_force"] = yf_scale
        freq_spec_df["block"] = block

        # .. ecg
        time_diff_phys = np.diff(physio_data["total_secs"])
        fs_ecg = np.round(1 / np.mean(time_diff_phys), 5)
        print("sampling rate ECG:", fs_ecg)
        window_samples = 50 * fs_ecg
        xf_ecg, yf_ecg = welch(physio_data["ecg"].values, fs=fs_ecg, nperseg=window_samples)
        mask = xf_ecg <= 2
        xf_ecg = xf_ecg[mask]
        yf_ecg = yf_ecg[mask]
        mask = xf_ecg > 0
        xf_ecg = xf_ecg[mask]
        yf_ecg = yf_ecg[mask]
        freq_spec_df["freq_ecg"] = yf_ecg

        # .. resp
        # first upsample the respiration data
        fs_resp = np.round(1 / np.mean(time_diff_phys), 5)
        print("sampling rate RESP:", fs_resp)
        window_samples = 50 * fs_resp
        xf_resp, yf_resp = welch(
            physio_data["resp"].values, fs=fs_resp, nperseg=window_samples
        )
        mask = xf_resp <= 2
        xf_resp = xf_resp[mask]
        yf_resp = yf_resp[mask]
        mask = xf_resp > 0
        xf_resp = xf_resp[mask]
        yf_resp = yf_resp[mask]
        freq_spec_df["freq_resp"] = yf_resp

        ############   COHERENCE #############
        # compute the coherence between respiration and force
        coh_window = (
            len(physio_data["force"]) / fs
        ) * 15  # 15 sec window, which is around 5 cycles of respiration

        # check if there are periods on Nans in the data, then deal with the secions without nans
        if nan_ratio > 0:
            print('compute coherence in segments without nans')
            valid = np.isfinite(physio_data["force"].values) & np.isfinite(physio_data["resp"].values)
            # get the parts of the data dont have nans
            segments = []
            start = None
            for i, v in enumerate(valid):
                if v:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        segments.append((start, i))
                        start = None
            if start is not None:
                segments.append((start, len(valid)))

            # compute coherence for each segment and average
            coherences = []
            delays = []
            max_freqs = []
            max_coherences = []
            for (start, end) in segments:
                if end - start < coh_window:
                    continue
                f_coh_seg, y_coh_seg, delay, freq_max, max_coh = compute_coherence_and_delay(
                    physio_data["force"].values[start:end],
                    physio_data["resp"].values[start:end],
                    fs,
                    coh_window
                )
                coherences.append(y_coh_seg)
                delays.append(delay)
                max_freqs.append(freq_max)
                max_coherences.append(max_coh)

            if len(coherences) == 0:
                print(f"No valid segments for coherence computation for sub{subID} block {block}")
                continue
            y_coherence = np.mean(coherences, axis=0)
            f_coherence = f_coh_seg
            delay = np.mean(delays)
            freq_max = np.mean(max_freqs)
            max_coh = np.mean(max_coherences)

        else:
            f_coherence, y_coherence, delay, freq_max, max_coh = compute_coherence_and_delay(physio_data["force"].values,
                                                                                    physio_data["resp"].values,
                                                                                    fs,
                                                                                    coh_window)
        
        # save info in dataframe
        coherence_df = pd.DataFrame()
        coherence_df["coherence_y"] = y_coherence
        coherence_df["coherence_x"] = f_coherence
        coherence_df["block"] = block
        coherence_df["sub"] = subID
        coherence_df["delay_max_respfreq"] = delay
        coherence_df["max_coh_respfreq"] = freq_max
        coherence_df["max_coh_value"] = max_coh

        # try to load corresponding scale trigger data
        trigger_scale_path = os.path.join(
                ses_dir, f"sub-0{subID}_ses-mecha_task-NT_run-{block}_forcetrigger.csv"
            )

        if not os.path.exists(trigger_scale_path):
            print(f"No trigger data for sub{subID} block {block}")
            if subID in subswithtrigger:
                raise Exception("No trigger data available BUT SHOULD HAVE TRIGGER DATA!!")

            # Note down that we don't have force triggers for this dataset
            coherence_df['force_trigger_exists'] = 'No'

        else:
            coherence_df['force_trigger_exists'] = 'Yes'
            # and load trigger data (for plotting)
            trigger_scale = pd.read_csv(trigger_scale_path)
            trigger_scale["timestamp"] = pd.to_datetime(trigger_scale["timestamp"])
            trigger_scale["timestamp_corrected"] = pd.to_datetime(trigger_scale["timestamp_corrected"])


        # save coherence info
        coherence_df.to_csv(
            os.path.join(
                derivatives_dir_path, f"sub-0{subID}_ses-mecha_task-NT_run-{block}_coherence_values.csv"
            ),
            index=False,
        )

        #############   REMOVE RESPIRATION PROJECTION ##############
        #
        # disentangle the signals (with the formular from thomas)
        # projecting the respiration signal from the force signal
        # formula:
        # force_vector - (force_vector * direction_of_respiration) * direction_of_respiration
        # ... which is the same as
        # force_vector - ((force_vector * resp_vector)/ length_resp^2) * resp_vector

        # zero-mean data and compute respiration projection
        force_data = physio_data["force"] - np.nanmean(physio_data["force"])
        resp_data = physio_data["resp"] - np.mean(physio_data["resp"])
        valid = np.isfinite(force_data) & np.isfinite(resp_data)

        # compute projection and take care of nan values!!
        proj = np.full_like(resp_data, 0)
        if valid.sum() >= 1:
            num = np.dot(force_data[valid], resp_data[valid])
            den = np.dot(resp_data[valid], resp_data[valid])
            if den == 0:
                coeff = np.nan  # resp is (near) zero in valid region
            else:
                coeff = num / den
            proj[valid] = coeff * resp_data[valid]

        projection_resp = proj
        force_without_resp = force_data - projection_resp

        # compute the frequency spectrum of the resulting signal
        window_samples = 50 * fs
        force_without_resp_nonan = np.asarray(y_uniform_zeromean, dtype=float)
        force_without_resp_nonan = force_without_resp_nonan[
            np.isfinite(force_without_resp_nonan)
        ]
        xf_force_proj, yf_force_proj = welch(
            force_without_resp_nonan, fs=fs, nperseg=window_samples
        )
        mask = xf_force_proj <= 2
        xf_force_proj = xf_force_proj[mask]
        yf_force_proj = yf_force_proj[mask]
        mask = xf_force_proj > 0
        xf_force_proj = xf_force_proj[mask]
        yf_force_proj = yf_force_proj[mask]
        freq_spec_df["freq_force-proj"] = yf_force_proj
        all_freq_spec_df = pd.concat((all_freq_spec_df, freq_spec_df))

        # Plot spectra
        print("plot spectra")
        fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 12))
        ax_force = axs[0]
        ax_resp = axs[1]
        ax_coherence = axs[2]
        ax_proj = axs[3]
        ax_ecg = axs[4]

        # plot freq spectra of all signals
        plot_power_spectrum(ax_resp, xf_resp, yf_resp, "Power Spectrum Respiration")
        plot_power_spectrum(ax_force, xf_scale, yf_scale, "Power Spectrum Force")
        plot_power_spectrum(ax_ecg, xf_ecg, yf_ecg, "Power Spectrum ECG")
        plot_power_spectrum(ax_proj, xf_force_proj, yf_force_proj, "Force - Resp Projection")
        
        # plot coherence
        ax_coherence.plot(f_coherence, y_coherence)
        ax_coherence.set_ylabel("Coherence")
        ax_coherence.set_title("Coherence between force and respiration")
        ax_coherence.set_xlabel("Frequency (Hz)")


        plt.tight_layout()
        sns.despine(trim=True)
        fig.savefig(
            os.path.join(fig_dir_sub, f"powerspectra_sub{subID}_block{block}.pdf")
        )
        #plt.show()

        # plot signal in time
        if subID in subswithtrigger:
            plot_forcesignal(
                subID, stim_annot, physio_data, 15, 45, block, trigger_scale, fig_dir_sub
            )

        else:
            plot_forcesignal(
                subID, stim_annot, physio_data, 20, 50, block, fig_dir_sub=fig_dir_sub
            )

        ############ Cross correlation ################
        #
        # normalize the data
        resp_norm = zscore(resp_data)
        scale_norm = zscore(force_data)

        # compute cross corr.
        cross_corr = correlate(resp_norm, scale_norm, mode="full", method="auto")
        lags = correlation_lags(len(resp_norm), len(scale_norm), mode="full") / fs_resp

        # Find best lag
        best_lag_sec = lags[np.argmax(cross_corr)]
        best_corr_value = np.max(cross_corr)

        plt.figure(figsize=(10, 6))
        plt.plot(lags, cross_corr, label="Cross-correlation", color="lightblue")
        plt.axvline(
            x=best_lag_sec,
            color="r",
            linestyle="--",
            label=f"Best lag between signals = {best_lag_sec:.3f} s",
        )
        plt.axhline(y=0, color="black", linewidth=0.8)
        plt.title(
            "Cross-correlation between respiratory signal and finger force", fontsize=14
        )
        plt.xlabel("Lag (seconds)", fontsize=12)
        plt.ylabel("Cross-correlation coefficient", fontsize=12)
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        sns.despine(trim=True)
        plt.savefig(
            os.path.join(fig_dir_sub, f"crosscorr_lags_sub{subID}_block{block}.pdf")
        )
        #plt.show()

        # plot result of projection in the time domain
        sample_start = 1000
        sample_stop = 5000
        t_samples = physio_data["total_secs"]
        # plt.plot(t_samples[sample_start:sample_stop], resp_data[sample_start:sample_stop], label='resp')
        plt.figure(figsize=(10, 6))
        plt.plot(
            t_samples[sample_start:sample_stop],
            force_data[sample_start:sample_stop],
            label="force",
        )
        plt.plot(
            t_samples[sample_start:sample_stop],
            projection_resp[sample_start:sample_stop],
            label="resp. projection vector",
        )
        plt.plot(
            t_samples[sample_start:sample_stop],
            force_without_resp[sample_start:sample_stop],
            label="force_without_resp",
        )

        plt.ylabel("Force (g)")
        plt.xlabel("time (s)")
        plt.legend()
        plt.savefig(
            os.path.join(
                fig_dir_sub,
                f"timecourse_respiration_projection_sub{subID}_block{block}.pdf",
            )
        )
        #plt.show()

        # TODO: if the subject was already processed, a new columns will be added... make sure that 
        # the old one will be overwrtten instead of adding a new one
        # add computed projection to physio dataframe
        if len(physio_data.columns) == 10:
            physio_data = physio_data.drop(columns=['resp_projection', 'force_without_resp'])

        physio_data['resp_projection'] = projection_resp
        physio_data['force_without_resp'] = force_without_resp

        physio_data.to_csv(
                    physio_data_path,
                index=False,
                header=False,
                sep="\t",
                compression="gzip",
            )

        # Add new entries about respiration projection
        if len(physio_metadata['Columns']) == 8:
            physio_metadata['Columns'].append('resp_projection') 
            physio_metadata['Columns'].append('force_without_resp') 
        
        physio_metadata["resp_projection"] = {
            "Description": "Projection of respiration signal in the force signal",
            "Units": "no unit",
        }
        physio_metadata["force_without_resp"] = {
            "Description": "Force signal with subtracted projection of respiration signal in the original force signal",
            "Units": "gramm",
        }

        # Save updated JSON back
        with open(physio_meta_path, "w", encoding="utf-8") as f:
            json.dump(physio_metadata, f, indent=4)
        
# %%
