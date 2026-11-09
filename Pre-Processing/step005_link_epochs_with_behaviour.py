"""
File: step005_link_epochs_with_behaviour.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-02-26
Description: Load the cleaned epochs, behavioural data (responses), and the notes about which 
epochs to drop and what the absolute intensity of stimulation was. Link all this information to the epoch trials and concatenate the epoch of all blocks.
Also the sigmoids are fitted blockwise and bad ones (r2<0.6) are excluded.

Output:
- _metadata.tsv : metadata of all trials (also missing triggers and dropped epochs)
- _metasummary.tsv
- _epochs.fif
- _epochs.tsv 

"""

# %%
import matplotlib.pyplot as plt
import os
import sys
import mne
import json
import re
from pathlib import Path
import pickle
import numpy as np
import seaborn as sns
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import pandas as pd
import scipy
import h5py
from mne.io import RawArray
from statsmodels.formula.api import ols
import statsmodels.api as sm
from helper_functions_behav import fit_curves, bin_responses
from helper_functions_eeg import parse_subjects, assign_modality, compute_prestim_oscillations, plot_prestim_oscillations
from plotting_style import figure_style 
colors, _ = figure_style()

RAWDIR = os.getenv("RAWDIR")
DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv('RESDIR')
WDDIR = os.getenv("WDDIR")
if not WDDIR:
    raise EnvironmentError("WDDIR is not set. Please export WDDIR to the repo root.")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import parse_cli_args
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'eeg_results')
figuredir_behav = os.path.join(RESDIR, 'Figures', 'Main', 'beh_results', 'single')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)
if not os.path.exists(figuredir_behav):
    os.makedirs(figuredir_behav)
session_type = "NT"  # make sure that this matches the session number

# read parameter
with open(os.path.join(WDDIR, 'Pre-Processing', 'preproc_parameter.json'), 'r') as json_file:
    params = json.load(json_file)

# if no arguments are used when starting script in commandline
# all subjects from 13 to 52 will be processed
# use --help to see options
#subIDs, modalities = parse_subjects()
subIDs = np.arange(13, 53)
# The frequencies in the filenames refer to the band the ICA was fitted on in step004_1
# (the saved epochs themselves are not filtered to it), hence the "ICA" marker in the names.
# Defaults therefore come from the ICA section of preproc_parameter.json.
l_freq_ica, h_freq_ica, file_suffix, modalities, _, _ = parse_cli_args(
    description=(
        "Link epochs with behavioural data. Optional arguments override "
        "frequency cutoffs, file suffix, and modalities."
    ),
    default_l_freq=params["ICA"]["l_freq"],
    default_h_freq=params["ICA"]["h_freq"],
    default_file_suffix="_preprestim",
    default_modalities=["Electrical"],
)

# frequency
l_freq_orig = 0.1

# frequency of the epochs that were used to detect bad epochs
# plotting colors for the different blocks
intensity_value = 'intensity_level'  # or 'intensity_norm' (this is used in the sigmoidal fitting process)


# %%
for subID in subIDs:
    subID = '0' + str(subID) if int(subID)<10 else str(subID)

    for m, modality in zip([0,2], modalities):
        if modality=='elec':
            modality_fullname = "Electrical"
        if modality=='mecha':
            modality_fullname = "Mechanical"
        
        print(m, modality)
        results = []

        # set session label and stimulation trigger based on modality
        ses_label, _ = assign_modality(modality)

        run = 1
        sub_dir = os.path.join(DATADIR, f"sub-0{subID}", f"ses-{ses_label}", "eeg")

        output_dir = os.path.join(
            DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", f"ses-{ses_label}"
        )
        output_path = os.path.join(DATADIR, output_dir)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        response_path = os.path.join(
            DATADIR, f"sub-0{subID}", f"ses-{ses_label}", "eeg"
        )

        all_epochs = []
        all_metadata_df = pd.DataFrame()

        # extract the block numbers from the filenames
        blocks = range(1, 9)

        for block in blocks:
            if block == 0:
                run = 0
            else:
                run = 1

            block_path = sub_dir
            epochs_file_name = (
                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_"
                f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{file_suffix}.fif"
            )
            dropEpochs_file_name = (
                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_"
                f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_badEpochs{file_suffix}.csv"
            )
            annot_file_name = f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_events{file_suffix}.csv"
            response_file_name = (
                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_beh.tsv"
            )

            if os.path.exists(
                os.path.join(block_path, epochs_file_name)
            ) & os.path.exists(os.path.join(response_path, response_file_name)):
                # Load behavioral data
                session_df = pd.read_csv(
                    os.path.join(response_path, response_file_name),
                    index_col=False,
                    sep="\t",
                )

                # 1. Load cleaned epochs and annotations
                epochs = mne.read_epochs(
                    os.path.join(block_path, epochs_file_name), preload=True
                )

                # 2. load annotations and bad epoch indices
                annotations_df = pd.read_csv(
                    os.path.join(block_path, annot_file_name), index_col=False
                )
                annotations_df = annotations_df[annotations_df["block"] == block]
                annotations_df = annotations_df.drop(["block", "run"], axis=1)

                try:
                    bad_epochs_df = pd.read_csv(
                        os.path.join(block_path, dropEpochs_file_name), index_col=False
                    )
                    trials_to_drop = list(bad_epochs_df["0"])
                except pd.errors.EmptyDataError:
                    print("no epochs to remove")
                    trials_to_drop = [None]

                # 3. Map response and intensity level to each epoch

                # Filter annotations for 'stimulation' events
                conditions = [
                    "Catch",
                    "IntensLevel1",
                    "IntensLevel2",
                    "IntensLevel3",
                    "IntensLevel4",
                    "IntensLevel5",
                    "IntensLevel6",
                    "IntensLevel7",
                ]
                stim_condition_df = annotations_df[
                    annotations_df["trigger_meaning"].isin(
                        [
                            "Catch",
                            "IntensLevel1",
                            "IntensLevel2",
                            "IntensLevel3",
                            "IntensLevel4",
                            "IntensLevel5",
                            "IntensLevel6",
                            "IntensLevel7",
                            "Stimulation",
                        ]
                    )
                ].reset_index(drop=True)
                missing_trials = []
                trialIDs = []
                stim_condition_df["trialID"] = None
                stim_condition_df["trigger_condition"] = None
                correct_missing_trial = 0
                trial_ID_counter = 0

                for i, row in stim_condition_df.iterrows():
                    trigger_meaning = row["trigger_meaning"]

                    if trigger_meaning not in conditions:
                        continue


                    # in the electrical condition the condition trigger comes before the stimulus, in the mechanical it comes after
                    # electrical starts with the condition trigger, therefore for the first trial we need to
                    # check if the next trigger is a stimulation
                    stim_idx = i + 1 if modality == "elec" else i - 1

                    # Check if the stimulation trigger is missing
                    if (
                        stim_condition_df.iloc[stim_idx]["trigger_meaning"]
                        != "Stimulation"
                    ):
                        raise RuntimeError("Trial missing")

                        missing_trials.append([trigger_meaning, trialID + 1])
                        correct_missing_trial += 1
                        trialID = (
                            (i - 1 + correct_missing_trial) / 2
                            if modality == "mecha"
                            else (i + correct_missing_trial) / 2
                        )
                        print("TRIAL MISSING!")
                        raise RuntimeError("Trial missing")

                        # Update the DataFrame for missing trial
                        stim_condition_df.at[i, "trigger_condition"] = trigger_meaning
                        stim_condition_df.at[i, "trigger_missing"] = True
                        stim_condition_df.at[i, "trialID"] = trialID
                        stim_condition_df.at[
                            i, "trialID_skipmissing"
                        ] = trial_ID_counter
                    else:
                        # Update the DataFrame for present trial
                        trialID = (
                            (i + correct_missing_trial) / 2
                            if modality == "elec"
                            else (i - 1 + correct_missing_trial) / 2
                        )

                        # Update the current and next/previous row based on modality
                        stim_condition_df.at[i, "trigger_condition"] = trigger_meaning
                        stim_condition_df.at[
                            stim_idx, "trigger_condition"
                        ] = trigger_meaning
                        stim_condition_df.at[i, "trialID"] = trialID
                        stim_condition_df.at[stim_idx, "trialID"] = trialID
                        stim_condition_df.at[
                            i, "trialID_skipmissing"
                        ] = trial_ID_counter
                        stim_condition_df.at[
                            stim_idx, "trialID_skipmissing"
                        ] = trial_ID_counter
                        stim_condition_df.at[i, "trigger_missing"] = False
                        stim_condition_df.at[stim_idx, "trigger_missing"] = False

                        # increase trial counter (so that we can track the ID of the actually recorded epochs)
                        # this might sometimes differ from the trials of the behaviour since sometimes triggers are not recorded..
                        trial_ID_counter += 1

                # now only pick the stimulation rows
                stimulation_annotations = stim_condition_df[
                    stim_condition_df["trigger_meaning"] == "Stimulation"
                ].reset_index(drop=True)

                # add a column for epochs to drop
                stimulation_annotations["drop_epoch"] = stimulation_annotations[
                    "trialID_skipmissing"
                ].isin(trials_to_drop)

                metadata_df = pd.merge(
                    stimulation_annotations, session_df, on="trialID", how="inner"
                )
                metadata_df = metadata_df[
                    [
                        "onset",
                        "total_secs",
                        "sub",
                        "modality",
                        "block",
                        "trialID",
                        "intensity_level",
                        "actual_intensity",
                        "response_decoded",
                        "trigger_missing",
                        "trialID_skipmissing",
                        "drop_epoch",
                        "force",
                        "force_zeromean",
                        "force_quantile",
                        "response_time",
                        "force_trigger_exists"
                    ]
                ]

                # normalise actual intensities
                grouped = (
                    metadata_df.groupby(
                        [
                            "modality",
                            "sub",
                            "block",
                            "intensity_level",
                            "actual_intensity",
                            "response_decoded",
                        ]
                    )
                    .size()
                    .unstack(
                        fill_value=0
                    )  # sets 'response_decoded' as name for the index column..
                    .reset_index()
                )

                grouped["intensity_norm"] = grouped.groupby(
                    ["sub", "modality", "block"]
                )["actual_intensity"].transform(lambda x: x / x.iloc[4])

                d = {
                    key: value
                    for key, value in zip(
                        grouped["intensity_level"], grouped["intensity_norm"]
                    )
                }
                metadata_df["intensity_norm"] = metadata_df["intensity_level"].replace(
                    d
                )

                # TODO: for subject 17 Mechanical session we need to add the first trial manually, since recording was started too late
                # also, the trials need to be shifted entirely, by 1! Since the first one was just not recorded
                metadata_df["sub"] = subID
                metadata_df["modality"] = modality_fullname
                
                epochs.metadata = metadata_df[metadata_df["trigger_missing"] == False]
                # projection must be the same so just in case I apply the average reference if not done yet
                epochs.set_eeg_reference("average", ch_type="eeg")
                all_epochs.append(epochs)
                all_metadata_df = pd.concat((all_metadata_df, metadata_df))

            else:
                print("No data for block:", block, "subject:", subID, "modality:", modality)

        # Concatenate epochs and metadata from all blocks
        # crop to the same time window
        common_tmin = params["epochs"]["tmin"]
        common_tmax = params["epochs"]["tmax"]
        all_epochs = [e.crop(tmin=common_tmin, tmax=common_tmax) for e in all_epochs]
        try:
            epochs_blocks = mne.concatenate_epochs(all_epochs)
        except IndexError:
            # for some subjects there is no data at all
            if (subID == '38' and modality == 'mecha') or (subID == '31' and modality == 'elec') or (subID == '25' and modality == 'mecha'):
                continue
            else:
                raise ValueError("No data available to concatenate: 'all_epochs' list is empty for this subject.")


        ######################################################
        # CREATE METADATA SUMMARY  
        # ####################################################

        # Evaluate how many trials I have per block and how much of those are yes responses
        # group by block and intensity level, calculate trial counts and percentage of 'yes' responses
        all_metadata_df["response_decoded"] = all_metadata_df[
            "response_decoded"
        ].astype("str")
        meta_summary = (
            all_metadata_df.groupby(["block", "intensity_level", "actual_intensity", "intensity_norm"])
            .agg(
                ntrials=("trialID", "count"),
                yes_responses=("response_decoded", lambda x: (x == "Yes").sum()),
                no_responses=("response_decoded", lambda x: (x == "No").sum()),
                nan_responses=("response_decoded", lambda x: (x == "nan").sum()),
            )
            .reset_index()
        )

        # now calculate percentage of 'yes' responses BEFORE we drop the bad epochs and missing triggers
        meta_summary["percent_yes_before_trialdrop"] = (
            meta_summary["yes_responses"] / meta_summary["ntrials"]
        ) * 100

        cleaned_metadata_df = all_metadata_df[
            (~all_metadata_df["drop_epoch"]) & (~all_metadata_df["trigger_missing"])
        ].reset_index(drop=True)

        # count the trials again and update summary
        meta_summary_cleaned = (
            cleaned_metadata_df.groupby(["block", "intensity_level"])
            .agg(
                ntrials=("trialID", "count"),
                yes_responses=("response_decoded", lambda x: (x == "Yes").sum()),
                no_responses=("response_decoded", lambda x: (x == "No").sum()),
                nan_responses=("response_decoded", lambda x: (x == "nan").sum()),
            )
            .reset_index()
        )
        meta_summary["ntrials_after_trialdrop"] = meta_summary_cleaned["ntrials"]
        meta_summary["yes_responses_after_trialdrop"] = meta_summary_cleaned[
            "yes_responses"
        ]
        meta_summary["no_responses_after_trialdrop"] = meta_summary_cleaned[
            "yes_responses"
        ]
        meta_summary["percent_yes_after_trialdrop"] = (
            meta_summary_cleaned["yes_responses"] / meta_summary_cleaned["ntrials"]
        ) * 100
        meta_summary["modality"] = modality_fullname

        ###################
        # FIT SIGMOIDS AND LINEAR MODELS TO BEHAVIOURAL DATA (blockwise)
        ###################
        all_blocks_fitting = pd.DataFrame()
        for block in range(1,9):
            df = meta_summary[(meta_summary['block']==block) & (meta_summary['intensity_level']>0)]

            # extract intensites and reponses
            intensities_block = np.array(df['intensity_level'])
            proportions_block = np.array(df['percent_yes_before_trialdrop'])

            # Fit the logistic function to the data
            initial_guess_sigm = [4, np.mean(intensities_block)]

            # check if there are enough response points to fit (above 0)
            if len(proportions_block[proportions_block > 0])<3:
                print('Not enough response points to fit a curve')
                [r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept, x_fit, y_fit] = [np.nan]*8
            else:
                [r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept, x_fit, y_fit] = fit_curves(intensities_block, proportions_block, initial_guess_sigm)

            # safe the fits in a dataframe
            fitting_df = pd.DataFrame()
            fitting_df['intensity_level'] = x_fit
            fitting_df['percent_yes_fitted'] = y_fit
            fitting_df['sub'] = subID
            fitting_df['modality'] = modality_fullname
            fitting_df['block'] = block
            fitting_df['r2_sigm'] = r_squared
            fitting_df['k_sigm'] = k_sigm
            fitting_df['x0_sigm'] = x0_sigm
            fitting_df['r2_linear'] = r_squared_lin
            fitting_df['k_linear'] = slope
            fitting_df['intercept_linear'] = intercept

            fitting_df.to_csv(os.path.join(output_path, f'sub-0{subID}_ses-{ses_label}_task-NT_run-{block}_behavfits.tsv'), index=False, sep='\t') 
            all_blocks_fitting = pd.concat((all_blocks_fitting, fitting_df), ignore_index=True) 

        ################# APPLY EXCLUSION CRITERIA BASED ON SIGMOID FITS #################

        # put an epoch index to be able to match them with the epoch data later
        all_metadata_df['epoch_index'] = all_metadata_df.index

        # merge information about exclusions back to metadata
        meta_summary = meta_summary.merge(
            all_blocks_fitting[['block', 'intensity_level', 'percent_yes_fitted']],
            on=['block', 'intensity_level'],
            how='left'
        )
        
        # exclude trials with false positive responses higher than 30%
        false_positives = meta_summary[(meta_summary[intensity_value]==0) & (meta_summary['percent_yes_before_trialdrop']>30)] 
        false_positives['analysis_excluded'] = True
        try:
            all_metadata_df.drop(columns=['analysis_excluded'], inplace=True)
        except:
            pass

        all_metadata_df = all_metadata_df.merge(false_positives[['block', intensity_value, 'analysis_excluded']], on=['block', intensity_value], how='left')
        meta_summary['analysis_excluded'] = False
        meta_summary['analysis_excluded'] = np.where(meta_summary['block'].isin(false_positives['block']), True, meta_summary['analysis_excluded'])

        # exclude blocks based on the sigmoid fit r2 threshold of 0.6, (and a middle point between 2 and 6?)
        df_fits_invalid = all_blocks_fitting[(all_blocks_fitting['r2_sigm']<0.6)] #|(all_blocks_fitting['x0_sigm']<2)|(all_blocks_fitting['x0_sigm']>6)|(all_blocks_fitting['x0_sigm']<2)]
        all_metadata_df['analysis_excluded'] = np.where(all_metadata_df['block'].isin(df_fits_invalid['block']), True, all_metadata_df['analysis_excluded'])
        meta_summary['analysis_excluded'] = np.where(meta_summary['block'].isin(df_fits_invalid['block']), True, meta_summary['analysis_excluded'])
        
        # exclude intensity levels 0, 1, 7
        meta_summary['analysis_excluded'] = np.where((meta_summary[intensity_value]==0)|(meta_summary[intensity_value]==1)|(meta_summary[intensity_value]==7), True, meta_summary['analysis_excluded'])
        all_metadata_df['analysis_excluded'] = np.where((all_metadata_df[intensity_value]==0)|(all_metadata_df[intensity_value]==1)|(all_metadata_df[intensity_value]==7), True, all_metadata_df['analysis_excluded'])
        all_metadata_df['analysis_excluded'] = all_metadata_df['analysis_excluded'].fillna(False)

        # for plotting, we want to exclude intensities for which the response was less than 20% and more than 80%
        boarder_intensities = ((meta_summary['intensity_level'] < 2)|(meta_summary['intensity_level']>6))
        invalid_responses = (meta_summary['percent_yes_before_trialdrop']>80)|(meta_summary['percent_yes_before_trialdrop']<20)
        invalid_plotting_trials = meta_summary[invalid_responses | boarder_intensities]
        invalid_plotting_trials['plot_excluded'] = True
        # drop any previously existing (and saved) column to avoid duplication
        try:
            all_metadata_df.drop(columns=['plot_excluded'], inplace=True)
        except:
            pass

        all_metadata_df = all_metadata_df.merge(invalid_plotting_trials[['block', intensity_value, 'plot_excluded']], on=['block', intensity_value], how='left')
        all_metadata_df['plot_excluded'] = all_metadata_df['plot_excluded'].fillna(False)
        meta_summary['plot_excluded'] = False
        meta_summary['plot_excluded'] = np.where(meta_summary['block'].isin(invalid_plotting_trials['block']), True, meta_summary['plot_excluded'])
        
        # convert to µV
        epochs_blocks._data *= 1e6
        epochs_blocks.metadata = all_metadata_df  # Assign the combined metadata

        # group yes response percentage into 8 bins
        for index, row in meta_summary.iterrows():
            meta_summary.at[index, 'percent_yes_bin'] = bin_responses(row, 'percent_yes_before_trialdrop')
            meta_summary.at[index, 'percent_yes_fitted_bin'] = bin_responses(row, 'percent_yes_fitted')

        # merge information from meta summary back to the metadata of all trials
        all_metadata_df = all_metadata_df.merge(
            meta_summary[['block', intensity_value, 'percent_yes_before_trialdrop', 'percent_yes_fitted', 'percent_yes_bin', 'percent_yes_fitted_bin']],
            on=['block', intensity_value],
            how='left',
        )
        
        # save the metasummary
        meta_summary.to_csv(
            os.path.join(
                output_path,
                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_metasummary{file_suffix}.tsv",
            ),
            index=False,
            sep="\t",
        )  # processed (overview of how many trials end up after removing all bad epochs)


        ######################################
        # COMPUTE PRESTIMULUS OSCILLATION POWER (ALPHA & BETA)
        alpha_power, beta_power = compute_prestim_oscillations(epochs_blocks)

        # mean power over trials (for plotting)
        trialsmean_alpha = np.mean(alpha_power, axis=0)
        trialsmean_beta = np.mean(beta_power, axis=0)
        info = epochs_blocks.info

        # Plotting topomap with prestim alpha
        prestim_dir = os.path.join(figuredir, 'single_prestim_oscillations')
        if not os.path.exists(prestim_dir):
            os.makedirs(prestim_dir)
        figure_name_alpha = os.path.join(prestim_dir, f'alpha_sub-0{subID}_ses-{ses_label}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}.png')
        figure_name_beta = os.path.join(prestim_dir, f'beta_sub-0{subID}_ses-{ses_label}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}.png')

        plot_prestim_oscillations(trialsmean_alpha, info, figure_name_alpha)
        plot_prestim_oscillations(trialsmean_beta, info, figure_name_beta)
        ####################################
        
        #################
        # drop bad epochs
        epochs_blocks = epochs_blocks[~epochs_blocks.metadata["drop_epoch"]]
        all_epochs_filename = f"sub-0{subID}_ses-{ses_label}_task-{session_type}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{file_suffix}"
        print('filename', all_epochs_filename)
        # save the epochs combined for all blocks and metadata
        epochs_blocks.save(
            os.path.join(
                output_path,
                f"{all_epochs_filename}.fif",
            ),
            overwrite=True,
        )


        # Get the data (n_epochs x n_channels x n_times)
        data = (
            epochs_blocks.get_data()
        )  # Raw data: n_epochs x n_channels x n_times

        # drop bad trials 
        cleaned_metadata_df = all_metadata_df[
            (~all_metadata_df["drop_epoch"]) & (~all_metadata_df["trigger_missing"])
        ].reset_index(drop=True)
        
        # double check if data and saved metadata have the same amount of trials!
        assert data.shape[0] == cleaned_metadata_df.shape[0], "Mismatch in meta data and eeg epochs!"

        # save cleaned metadata
        cleaned_metadata_df.to_csv(
            os.path.join(
                output_path,
                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_allmetadata{file_suffix}.tsv",

            ),
            index=False,
            sep="\t",
        )

        # Save channel data in hdf5 file (one channel per group)
        channel_names = epochs_blocks.ch_names
        times = epochs_blocks.times

        with h5py.File(os.path.join(output_path, f"{all_epochs_filename}.h5"), "w") as f:
            for ch_idx, ch_name in enumerate(channel_names):
                # Create a group for the channel
                grp = f.create_group(ch_name)
                
                # Save channel data (trials x timepoints)
                grp.create_dataset(
                    "data",
                    data=data[:, ch_idx, :],
                    compression="gzip",
                    compression_opts=9,
                )
                
                grp.create_dataset("times", data=times)
                
                # Store attributes for the group
                grp.attrs["channel_name"] = ch_name
                grp.attrs["n_trials"] = data.shape[0]
                grp.attrs["n_timepoints"] = data.shape[2]

        print('Finished subject', subID, modality)

# %%
# plot response percentage for each block and intensity
response_percentage_measure = 'percent_yes_fitted'  # or 'percent_yes_before_trialdrop'
plt.figure(figsize=(10,6))
test_df = all_metadata_df[all_metadata_df['block']==1]
test_df_intensity = test_df[test_df['intensity_level']==4]
sns.lineplot(all_metadata_df, x='intensity_level', y=response_percentage_measure, hue='block', marker='o')
# %%
# plot how many datapoints per yes percentage bin 
response_percentage_measure = 'percent_yes_fitted_bin'  # or 'percent_yes_bin'
plt.figure(figsize=(8,5))
percentages = all_metadata_df[response_percentage_measure].unique().tolist()
sorted_percentages = sorted(percentages, key=lambda x: float(x.split('-')[0]))
sns.countplot(all_metadata_df, x=response_percentage_measure, hue='block', order=sorted_percentages)
sns.despine()
plt.xlabel('Binned % yes responses')
plt.ylabel('Number of trials')
plt.title(f'Subject {subID} {modality} - Trials per % yes response bin')
plt.legend(title='Block')
plt.tight_layout()
plt.savefig(os.path.join(figuredir_behav, f'sub-0{subID}_ses-{ses_label}_task-{session_type}_{modality}_trials_per_{response_percentage_measure}_bin.png'), dpi=300)
# %%
