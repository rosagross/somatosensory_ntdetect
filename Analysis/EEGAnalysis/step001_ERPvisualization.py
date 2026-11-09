# geretate file doc 
"""
Script to visualize ERP data for all subjects and conditions.
- Plotting grand average ERP graphs
"""


# %%
import matplotlib.pyplot as plt
import os
import sys
import gc
import mne
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from scipy.stats import sem
import argparse
from scipy.optimize import curve_fit
from mne.io import RawArray
from plotting_style import figure_style
from analysis_plotting_functions import plot_single_ERP, plot_grand_ERP
colors, _ = figure_style()

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
WDDIR = os.getenv('WDDIR')

# Ensure repo root is on sys.path so general_helpers can be imported reliably.
if not WDDIR:
    raise EnvironmentError("WDDIR is not set. Please export WDDIR to the repo root.")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import load_clean_epochs, apply_baseline_epochs
 
# %%

# load data for plotting for a specific channel
subjects = np.arange(13, 53)

channel_names = ['CP4', 'C4', 'C6']
measure_option = 0
measure_dict = {
    'intensity_level': [2, 3, 4, 5, 6],
    'percent_yes_fitted_bin': ['20-35%', '35-50%', '50-65%', '65-80%', '80-95%']
}

intensity_measure = list(measure_dict.keys())[measure_option]
intensity_levels = measure_dict[list(measure_dict.keys())[measure_option]]

"""# parse channels from commandline 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--channels",
    type=str,
    nargs="+",
    help="channels to plot (0:C4, 1:C6, 2:CP4, 3:Pz)",
    required=False,
)
args = parser.parse_args()
if args.channels:
    channel_ids = args.channels
else: 
    channel_ids = ["1"]"""

channel_ids = ["0", "1", "2", "3", "4"]  #["0", "1", "2", "3", "4", "5", "6", "7", "8"]  # "0", "1", "2", "3", "4", 

palette = {
    'No': colors['unperceived'], 
    'Yes': colors['perceived']
}
palette_modality = {
    'elec': colors['electrical'], 
    'mecha': colors['mechanical']
}

min_epochs = 0 # minimum number of epochs as exclusion criterium

modalities = ['elec', 'mecha']
file_suffix = '_preprestim_corrected'
file_suffix_meta = '_preprestim'
sfreq = 500
tmin = -0.55
tmax = 0.3
timewindow = tmax - tmin
datapoints = int(timewindow*sfreq)
l_freq_orig = 0.1
l_freq = 1
h_freq = 40
tmin_baseline = -0.55
tmax_baseline = -0.45
baseline_window = (tmin_baseline, tmax_baseline)

# create dirs based on suffix
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'eeg_results', 'ERPvisualisation', f'{file_suffix}_baseline{tmin_baseline}-{tmax_baseline}')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)

statsdir = os.path.join(RESDIR, 'Statistics', 'eeg', 'erps', f'{file_suffix}_baseline{tmin_baseline}-{tmax_baseline}')
if not os.path.exists(statsdir):
    os.makedirs(statsdir)


def apply_surface_laplacian(epochs):
    """
    Compute a surface Laplacian (CSD) on EEG channels before channel-wise metrics.

    This reduces volume-conduction effects and makes each channel signal more local.
    """
    eeg_epochs = epochs.copy().pick("eeg")
    if len(eeg_epochs.ch_names) < 2:
        raise ValueError("Need at least 2 EEG channels to compute Laplacian (CSD).")
    return mne.preprocessing.compute_current_source_density(eeg_epochs)

# %%
all_epochs = {}
df = pd.DataFrame()

# Create subject and trial tracking dataframes
subject_trial_counts = pd.DataFrame(columns=['subject', 'modality', 'total_trials'])

for channel_id in channel_ids:
    figure, axis = plt.subplots(figsize=(10, 5))
    channel = channel_names[int(channel_id)]
    
    for modality in modalities:
        count_subs = 0 
        
        all_perceived = np.array(np.empty((datapoints, 0))) # x timepoints
        all_unperceived = np.array(np.empty((datapoints, 0)))

        allsubs_intensities = np.array(np.empty((datapoints, 0)))
        allsubs_intensities_perceived = np.array(np.empty((datapoints, 0)))
        allsubs_intensities_unperceived = np.array(np.empty((datapoints, 0)))

        for subID in subjects:

            # exclude subject 23 mechanical (swapped ground and reference electrode)
            # exclude mecha 36 (force values were exceptionally high!)
            if ((subID == 23) or (subID == 36)) and modality == 'mecha':
                print(f"Skipping subject {subID} modality {modality} due to faulty recording setup.")
                continue 
                 
            sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')

            # read in epochs (FIF only)
            all_epochs_filename = f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{file_suffix}"
            meta_filename = f"sub-0{subID}_ses-{modality}_task-NT_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_allmetadata{file_suffix_meta}.tsv"
            epoch_fif = os.path.join(sub_dir, f"{all_epochs_filename}.fif")
            print('subject', subID)
            
            if epoch_fif is None or not os.path.exists(epoch_fif):
                print(f"Epoch file for subject {subID} modality {modality} not found, skipping...")
                continue

            # Load epochs from FIF and extract a single channel.
            # This keeps data handling consistent with MNE structures.
            epochs = mne.read_epochs(epoch_fif, preload=True, verbose=False)

            print('len epochs', len(epochs))
            
            epochs = epochs[epochs.metadata['analysis_excluded'] == False]
            epochs = epochs[epochs.metadata['drop_epoch'] == False]
            epochs = epochs[((epochs.metadata['response_decoded'] == 'No') | (epochs.metadata['response_decoded'] == 'Yes'))]

            # if now no trials left, skip subject
            if len(epochs) == 0:
                print(f"No valid trials left for subject {subID} modality {modality}, skipping...")
                continue
            elif len(epochs) < min_epochs: 
                print("subject does not have enough trials")
                continue
            else:
                count_subs += 1

            # apply baseline
            apply_baseline_epochs(epochs, epoch_fif, baseline_window=baseline_window)
            
            # apply laplacian
            #epochs = apply_surface_laplacian(epochs)

            time_steps = epochs.times  # in seconds
            window_tmin = np.where(time_steps >= tmin)[0][0]
            window_tmax = np.where(time_steps <= tmax)[0][-1]
            time_steps = time_steps[window_tmin:window_tmax]
            print("time steps", time_steps.shape)

            # shape: (epochs x timepoints)
            channel_data = epochs.copy().pick(channel).get_data()[:, 0, :]
            channel_data = channel_data[:, window_tmin:window_tmax]
            print("channel data", channel_data.shape)

            # read metadata 
            metadata_df = pd.read_csv(os.path.join(sub_dir, meta_filename), sep='\t')
            print(len(metadata_df), "trials subject", subID, "modality", modality)
            
            # remove trials of bad epochs from metadata
            metadata_df = metadata_df[metadata_df['drop_epoch'] == False]
            
            print(len(metadata_df), "trials subject", subID, "modality", modality)

            # remove trials excluded for plotting
            #metadata_df = metadata_df[metadata_df['plot_excluded'] == False]
            metadata_df = metadata_df[metadata_df['analysis_excluded'] == False].reset_index(drop=True)
            
            # remove trials with nan response
            metadata_df = metadata_df[metadata_df['response_decoded'].notna()]
            
            print("channel data after exclusion", channel_data.shape)

            # make sure that the epoch numbers match (fail fast per data integrity rule)
            if channel_data.shape[0] != metadata_df.shape[0]:
                raise ValueError(
                    f"Epoch/metadata mismatch for subject {subID} modality {modality}: "
                    f"{channel_data.shape[0]} epochs in FIF vs {metadata_df.shape[0]} metadata rows."
                )

            # now we need to reset the index of metadata_df, since we dropped some trials
            metadata_df = metadata_df.reset_index(drop=True)
            perceived_trials = metadata_df[metadata_df['response_decoded'] == 'Yes'].index
            unperceived_trials = metadata_df[metadata_df['response_decoded'] == 'No'].index
            perceived_mean = np.mean(channel_data[perceived_trials, :].T, axis=1)
            unperceived_mean = np.mean(channel_data[unperceived_trials, :].T, axis=1)
            print(len(perceived_trials)+len(unperceived_trials), "trials")

            # Count trials for this subject and modality
            total_trials = len(metadata_df)

            
            # Add to subject trial counts dataframe
            new_row = {
                'subject': f'sub-0{subID}',
                'modality': modality,
                'total_trials': total_trials
            }
            subject_trial_counts = pd.concat([subject_trial_counts, pd.DataFrame([new_row])], ignore_index=True)

            # plot single subject ERP
            plot_single_ERP(time_steps, np.mean(channel_data.T, axis=1), unperceived_mean, subID, channel, modality)

            # concat data from subject 
            all_perceived = np.append(
                all_perceived,
                #perceived_mean.reshape(-1, 1),
                channel_data[perceived_trials, :].T,
                axis=1
            )

            all_unperceived = np.append(
                all_unperceived,
                #unperceived_mean.reshape(-1, 1),
                channel_data[unperceived_trials, :].T,
                axis=1
            ) 

            # per block compute ERP per intensity level in a certain window
            allblocks_intensities = np.array(np.empty((datapoints, 0)))
            allblocks_intensities_perceived = np.array(np.empty((datapoints, 0)))
            allblocks_intensities_unperceived = np.array(np.empty((datapoints, 0)))
            
            for block in metadata_df['block'].unique():
                metadata_df_block = metadata_df[metadata_df['block'] == block]
                
                # compute ERP per intensity level
                perceived_intensity_mean = {}
                unperceived_intensity_mean = {}
                all_intensity_mean = {}

                for intensity in intensity_levels:
                    trialIDs_perceived_intensity = metadata_df_block[(metadata_df_block['response_decoded'] == 'Yes') & (metadata_df_block[intensity_measure] == intensity)].index
                    perceived_intensity_mean[intensity] = np.mean(channel_data[trialIDs_perceived_intensity, :].T, axis=1)
                    trialIDs_unperceived_intensity = metadata_df_block[(metadata_df_block['response_decoded'] == 'No') & (metadata_df_block[intensity_measure] == intensity)].index
                    unperceived_intensity_mean[intensity] = np.mean(channel_data[trialIDs_unperceived_intensity, :].T, axis=1)
                    trialIDs_all_intensity = [*trialIDs_unperceived_intensity, *trialIDs_perceived_intensity] # metadata_df[metadata_df['intensity_level'] == intensity].index
                    all_intensity_mean[intensity] = np.mean(channel_data[trialIDs_all_intensity, :].T, axis=1)

                    timewindows = {
                                        '50-60ms': (0.050, 0.060),
                                        '70-80ms': (0.07, 0.08),
                                        '140-150ms': (0.14, 0.15)
                                    }

                    # save which timepoint was used
                    timepoints = []
                    for window in timewindows:

                        # window indices
                        windowstart, windowend = timewindows[window]
                        windowstart_idx = np.argmin(np.abs(time_steps - windowstart))
                        windowend_idx = np.argmin(np.abs(time_steps - windowend))
                        
                        # build dataframe for chosen timepoint
                        timepoint = np.round((windowstart + windowend)/2, 3)
                        timepoint_idx = np.argmin(np.abs(time_steps - timepoint))
                        timepoint = time_steps[timepoint_idx]
                        timepoints.append(timepoint)

                        for perceived, condition_trials in zip(['No', 'Yes'],[trialIDs_unperceived_intensity, trialIDs_perceived_intensity]):

                            erp_data = {
                                'subID': subID,
                                'modality': modality,
                                'block': block,
                                'channel': channel,
                                'timepoint': timepoint,
                                intensity_measure: intensity,
                                #'trialID': metadata_df.loc[condition_trials, 'trialID'].values,
                                'amplitude': channel_data[condition_trials, :].T[timepoint_idx],
                                'response_decoded': perceived,
                                'window': window,
                                'window_amplitude': np.nanmean(channel_data[condition_trials, :].T[windowstart_idx: windowend_idx], axis=0),
                                'force': metadata_df['force'][condition_trials],
                                'force_quantile': metadata_df['force_quantile'][condition_trials]
                            }

                            df = pd.concat([df, pd.DataFrame(erp_data)], ignore_index=True)
                
                # append all intensities to numpy array
                allblocks_intensities = np.append(
                    allblocks_intensities,
                    all_intensity_mean
                )
                allblocks_intensities_perceived = np.append(
                    allblocks_intensities_perceived,
                    perceived_intensity_mean
                )
                allblocks_intensities_unperceived = np.append(
                    allblocks_intensities_unperceived,
                    unperceived_intensity_mean
                )

                # per block: compute grand average per intensity level across subjects
                block_mean_per_intensity = {}
                block_mean_per_intensity_perceived = {}
                block_mean_per_intensity_unperceived = {}
                for intensity_key in intensity_levels:
                    values_stack = np.array([block_dict[intensity_key] for block_dict in allblocks_intensities])  # shape: (37, 500)
                    trajectory_means = np.nanmean(values_stack, axis=0)
                    block_mean_per_intensity[intensity_key] = trajectory_means

                    values_stack_perceived = np.array([block_dict[intensity_key] for block_dict in allblocks_intensities_perceived])  # shape: (37, 500)
                    trajectory_means_perceived = np.nanmean(values_stack_perceived, axis=0)
                    block_mean_per_intensity_perceived[intensity_key] = trajectory_means_perceived

                    values_stack_unperceived = np.array([block_dict[intensity_key] for block_dict in allblocks_intensities_unperceived])  # shape: (37, 500)
                    trajectory_means_unperceived = np.nanmean(values_stack_unperceived, axis=0)
                    block_mean_per_intensity_unperceived[intensity_key] = trajectory_means_unperceived


            # append all intensities to numpy array
            allsubs_intensities = np.append(
                allsubs_intensities,
                block_mean_per_intensity
            )
            allsubs_intensities_perceived = np.append(
                allsubs_intensities_perceived,
                block_mean_per_intensity_perceived
            )
            allsubs_intensities_unperceived = np.append(
                allsubs_intensities_unperceived,
                block_mean_per_intensity_unperceived
            )
            

        # compute grand average per intensity level across subjects
        mean_per_intensity = {}
        mean_per_intensity_perceived = {}
        mean_per_intensity_unperceived = {}
        for intensity_key in intensity_levels:
            values_stack = np.array([subject_dict[intensity_key] for subject_dict in allsubs_intensities])  # shape: (37, 500)
            trajectory_means = np.nanmean(values_stack, axis=0)
            mean_per_intensity[intensity_key] = trajectory_means

            values_stack_perceived = np.array([subject_dict[intensity_key] for subject_dict in allsubs_intensities_perceived])  # shape: (37, 500)
            trajectory_means_perceived = np.nanmean(values_stack_perceived, axis=0)
            mean_per_intensity_perceived[intensity_key] = trajectory_means_perceived

            values_stack_unperceived = np.array([subject_dict[intensity_key] for subject_dict in allsubs_intensities_unperceived])  # shape: (37, 500)
            trajectory_means_unperceived = np.nanmean(values_stack_unperceived, axis=0)
            mean_per_intensity_unperceived[intensity_key] = trajectory_means_unperceived


        # grand average across subjects
        plot_grand_ERP(all_unperceived, all_perceived, time_steps, tmin, tmax, modality, channel)
        print(count_subs, 'subjects in', modality)

        # plot the grand average (both modalities in one plot)
        modality_label = 'Electrical' if modality == 'elec' else 'Mechanical'

        timewindow = (time_steps >= tmin) & (time_steps <= tmax)
        time_steps = time_steps[timewindow]
        all_unperceived = all_unperceived[timewindow, :]
        all_perceived = all_perceived[timewindow, :]    
        mean_vals_unperceived = np.mean(all_unperceived, axis=1)  
        std_vals_unperceived = np.std(all_unperceived, axis=1)    
        mean_vals_perceived = np.mean(all_perceived, axis=1)  
        std_vals_perceived = np.std(all_perceived, axis=1)    
        sem_vals_unperceived = sem(all_unperceived, axis=1)    
        sem_vals_perceived = sem(all_perceived, axis=1)    
        n_perceived = all_perceived.shape[1]
        ci95_perceived = 1.96 * std_vals_perceived / np.sqrt(n_perceived)
        n_unperceived = all_unperceived.shape[1]
        ci95_unperceived = 1.96 * std_vals_unperceived / np.sqrt(n_unperceived)

        sns.lineplot(x=time_steps, y=mean_vals_perceived, label=f'{modality} perceived', color=palette_modality[modality], ax=axis)
        sns.lineplot(x=time_steps, y=mean_vals_unperceived, label=f'{modality} unperceived', color=palette_modality[modality], linestyle='--', ax=axis)
        #axis.fill_between(time_steps, mean_vals_unperceived - ci95_unperceived, mean_vals_unperceived + ci95_unperceived, alpha=0.3, label='', color=palette_modality[modality])
        #axis.fill_between(time_steps, mean_vals_perceived - ci95_perceived, mean_vals_perceived + ci95_perceived, alpha=0.3, label='', color=palette_modality[modality])
        
        # add vertical line at -0.4s (cue onset) and 0s (simulus onset)
        if modality == 'mecha':
            if time_steps[0] < -0.4:
                axis.axvline(x=-0.4, color='red', linestyle='-')
            if time_steps[0] < 0:
                axis.axvline(x=0, color='black', linestyle='-')
        
        axis.legend()
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('µV')
        #axis.set_xticks(fontsize=12)
        #axis.set_yticks(fontsize=12)
        axis.set_title(f'{channel} - Mechanical AND Electrical', fontsize=14)
        sns.despine(trim=True, ax=axis)
        
    
        # plot grand average ERP per intensity level
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(25, 5))

        for intensity in intensity_levels:
            mean_all = np.mean([mean_per_intensity_perceived[intensity],mean_per_intensity_unperceived[intensity]], axis=0)
            sns.lineplot(x=time_steps, y=mean_per_intensity_perceived[intensity], label=f'{intensity}', ax=ax[0])
            sns.lineplot(x=time_steps, y=mean_per_intensity_unperceived[intensity], label=f'{intensity}', ax=ax[1])
            sns.lineplot(x=time_steps, y=mean_per_intensity[intensity], label=f'{intensity}', ax=ax[2])
            if time_steps[0] < 0:
                ax[0].axvline(x=0, color='k', linestyle='--')
                ax[1].axvline(x=0, color='k', linestyle='--')
                ax[2].axvline(x=0, color='k', linestyle='--')
            
            if time_steps[0] < timepoint:
                ax[0].axvline(x=timepoint, color='red', linestyle='--')
                ax[1].axvline(x=timepoint, color='red', linestyle='--')
                ax[2].axvline(x=timepoint, color='red', linestyle='--')

            ax[0].set_title('Perceived')
            ax[1].set_title('Unperceived')    
            ax[2].set_title('All trials')   

            # check mean amplitude at timepoint
            timepoint_idx = np.argmin(np.abs(time_steps - timepoint)) 
            print(f"{mean_all[timepoint_idx]:.4f} µV intensity {intensity} time {timepoint*1000:.0f}")

        plt.legend(title='Intensity level')
        sns.despine(trim=True)
        plt.title(f'Grand Average ERP by intensity level - channel={channel} modality={modality}')
        
        # save figure
        plt.savefig(os.path.join(figuredir, f'Grand_Average_ERP_by_{intensity_measure}_channel-{channel}-{modality}{file_suffix}.pdf'))
        plt.savefig(os.path.join(figuredir, f'Grand_Average_ERP_by_{intensity_measure}_channel-{channel}-{modality}{file_suffix}.png'))

        # Save subject and trial counts to CSV
        subfile = os.path.join(statsdir, f'subject_trial_counts_{modality}_step001_ERPvisualization.csv')
        subject_trial_counts.to_csv(subfile, index=False)
        print("Subject and trial counts saved to:", subfile)
        print("Script completed successfully!")

    for window in ['50-60ms', '70-80ms', '140-150ms']:

        # plot barplot of amplitudes at given timepoint per intensity level
        fig, ax = plt.subplots(figsize=(10, 6))
        df_channel = df[(df['channel'] == channel)]
        df_window = df_channel[df_channel['window'] == window]
        # check mean for intensity levels
        print(df_channel.groupby(intensity_measure)['amplitude'].mean())

        sns.barplot(data=df_window, y='window_amplitude', x=intensity_measure, hue='modality', palette=palette_modality)
        plt.ylabel('ERP Amplitude (µV)')
        plt.xlabel(f'Initensity ({intensity_measure})')
        #plt.xticklabels(fontsize=12)
        plt.title(f'ERP amplitudes at {window} by intensity level \n- channel={channel}')

        sns.despine(trim=True)
        # save figure
        plt.savefig(os.path.join(figuredir, f'barplot_ERP_amplitude_window-{window}_by_{intensity_measure}_channel{channel}{file_suffix}.pdf'))
        plt.savefig(os.path.join(figuredir, f'barplot_ERP_amplitude_window-{window}_by_{intensity_measure}_channel{channel}{file_suffix}.png'))
    
    for timepoint in timepoints:
        # plot barplot of amplitudes at given timepoint per intensity level
        fig, ax = plt.subplots(figsize=(10, 6))
        df_channel = df[(df['channel'] == channel)]
        df_window = df_channel[df_channel['timepoint'] == timepoint]
        # check mean for intensity levels
        print(df_channel.groupby(intensity_measure)['amplitude'].mean())

        sns.barplot(data=df_window, y='amplitude', x=intensity_measure, hue='modality', palette=palette_modality)
        plt.ylabel('ERP Amplitude (µV)')
        plt.xlabel(f'Initensity ({intensity_measure})')
        #plt.xticklabels(fontsize=12)
        plt.title(f'ERP amplitudes at {timepoint} by intensity level \n- channel={channel}')

        sns.despine(trim=True)
        # save figure
        plt.savefig(os.path.join(figuredir, f'barplot_ERP_amplitude_timepoint-{timepoint}_by_{intensity_measure}_channel{channel}{file_suffix}.pdf'))
        plt.savefig(os.path.join(figuredir, f'barplot_ERP_amplitude_timepoint-{timepoint}_by_{intensity_measure}_channel{channel}{file_suffix}.png'))

    figure.show()
    figure.savefig(os.path.join(figuredir, f'Grand_Average_ERP_channel-{channel}-both_modalities{file_suffix}.pdf'), dpi=300)
    figure.savefig(os.path.join(figuredir, f'Grand_Average_ERP_channel-{channel}-both_modalities{file_suffix}.png'), dpi=300)

# save dataframe for statistical anaylysis
df.to_csv(os.path.join(statsdir, f'ERP_amplitude_perWindow_by_{intensity_measure}{file_suffix}.csv'), index=False)

# save separate CSV files for each channel and time window for R analysis
unique_channels = df['channel'].unique()
unique_windows = df['window'].unique()

for channel_selected in unique_channels:
    for window_selected in unique_windows:
        # filter data for this channel and window
        df_channel_window = df[(df['channel'] == channel_selected) & (df['window'] == window_selected)]
        
        # save with R-compatible naming format
        filename = f'ERP_amplitude_{channel_selected}_{window_selected}.csv'
        filepath = os.path.join(statsdir, filename)
        df_channel_window.to_csv(filepath, index=False)
        print(f"Saved {filename} with {len(df_channel_window)} rows")

# %%
# Implement bar plot showing ERP amplitude by intensity level with response as hue
# for each timepoint, modality, and channel separately

def create_barplot_erp_amplitude(df, figuredir, file_suffix, intensity_measure):
    """
    Create bar plots showing ERP amplitude by intensity level with response as hue
    for each timepoint, modality, and channel separately.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ERP amplitude data
    figuredir : str
        Directory to save figures
    file_suffix : str
        File suffix for saving figures
    intensity_measure : str
        Intensity measure used ('intensity_level' or 'percent_yes_fitted_bin')
    """
    
    # Get unique combinations of timepoints, modalities, and channels
    unique_timepoints = df['timepoint'].unique()
    unique_modalities = df['modality'].unique()
    unique_channels = df['channel'].unique()
    
    print(f"Creating bar plots for {len(unique_timepoints)} timepoints, "
          f"{len(unique_modalities)} modalities, and {len(unique_channels)} channels")
    
    # Create directory for bar plots
    barplot_dir = os.path.join(figuredir, 'barplots')
    if not os.path.exists(barplot_dir):
        os.makedirs(barplot_dir)
    
    plot_count = 0
    
    for timepoint in unique_timepoints:
        for modality in unique_modalities:
            for channel in unique_channels:
                
                # Filter data for this specific combination
                mask = (
                    (df['timepoint'] == timepoint) & 
                    (df['modality'] == modality) & 
                    (df['channel'] == channel)
                )
                df_subset = df[mask].copy()
                
                # Skip if no data for this combination
                if len(df_subset) == 0:
                    continue
                
                # Create the bar plot
                plt.figure(figsize=(12, 6))
                
                # Create bar plot with intensity_level on x-axis, amplitude on y-axis, and response as hue
                ax = sns.barplot(
                    data=df_subset,
                    x=intensity_measure,
                    y='amplitude',
                    hue='response_decoded',
                    palette=palette,
                    errorbar='se'  # Standard error
                )
                
                # Customize the plot
                modality_label = 'Electrical' if modality == 'elec' else 'Mechanical'
                plt.title(f'ERP Amplitude by {intensity_measure} - {modality_label} Modality\n'
                         f'Channel {channel}, Timepoint {timepoint*1000:.0f}ms, Ref.: laplacian', fontsize=14, fontweight='bold')
                plt.xlabel(f'{intensity_measure.replace("_", " ").title()}', fontsize=14)
                plt.ylabel('ERP Amplitude (µV)', fontsize=14)
                plt.legend(title='Response', title_fontsize=12, fontsize=11)
                
                # Improve layout
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                sns.despine(trim=True)
                
                # Save the plot
                filename = f'barplot_erp_amplitude_{intensity_measure}_{modality}_{channel}_timepoint{int(timepoint*1000)}ms{file_suffix}.pdf'
                filepath = os.path.join(barplot_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.show()
                
                plot_count += 1
                
                # Print progress
                if plot_count % 10 == 0:
                    print(f"Created {plot_count} bar plots...")

    print(f"Successfully created {plot_count} bar plots in {barplot_dir}")

# Create the bar plots
create_barplot_erp_amplitude(df, figuredir, file_suffix, intensity_measure)

# %%
# to investigate the influence of intensity on erp and interaction with channel
# we split the data into three timewindows

# three-way anova (affect on amplitude)
# 1. timepoint (50-60ms, 70-80ms, 140-150ms)
# 2. channel 
# 3. intensity level


# averages of individual trials per subject 
# dataframe columns: 
# subID, channel, intensity_level, timepoint, amplitude

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('amplitude ~ C(channel) + C(intensity_level) + C(window) + C(channel):C(intensity_level) + C(channel):C(window) + C(intensity_level):C(window)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# save anova table
anova_table.to_csv(os.path.join(statsdir, f'ANOVA_ERP_amplitude_channel_intensity_window.csv'))
