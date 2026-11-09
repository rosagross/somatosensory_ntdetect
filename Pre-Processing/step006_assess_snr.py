"""
File: step005_assess_snr.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-02-25
Description: To get an overview of data quality and preprocessing, plot remaining epochs and response behaviour per sub&block.

"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from helper_functions_eeg import parse_subjects

plt.ioff() # supress plotting in interactive window

# set paths
RAWDIR = os.getenv('RAWDIR')
DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')

# loop through the subjects
#subIDs, modalities = parse_subjects()
subIDs = np.arange(13, 53)
modalities = ['Electrical', 'Mechanical'] 
l_freq = 1
h_freq = 40
l_freq_orig = 0.1
suffix = "_preprestim"

means = []

fig_dir = os.path.join(RESDIR, 'Figures', 'Pilot', 'snr_checks', f'hfreq_{h_freq}')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# %%
total_nr = []
for subID in subIDs:
    subID = '0' + str(subID) if int(subID) < 10 else str(subID)

    for modality in modalities:

        if modality=='Electrical':
            ses_label = 'elec' 
        elif modality == 'Mechanical':
            ses_label = 'mecha' 

        print('sub:', subID, modality)

        # set data path for combined epoch & annotation data 
        session_type = 'NT'
        prepro_dir = os.path.join('derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{ses_label}')
        prepro_path = os.path.join(DATADIR, prepro_dir) 

        # load trial annotations
        meta_file = os.path.join(prepro_path, f'sub-0{subID}_ses-{ses_label}_task-{session_type}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_allmetadata{suffix}.tsv')
        if not os.path.exists(meta_file):
            print('no data for sub', subID)
            continue

        all_metadata_df = pd.read_csv(meta_file, sep='\t')

        # make plot with the following details
        # - x-axis: blocks
        # - y-axis: nr. trials
        # - hue: response (yes/no)

        # Aggregate the data by block
        meta_df = all_metadata_df.groupby('block').agg(
            total_trials=('trialID', 'count'),  # Total trials per block
            trials_with_missing_triggers=('trigger_missing', 'sum'),  # Trials with missing triggers
            trials_after_epoch_drop=('drop_epoch', lambda x: (x == False).sum())  # Trials not dropped
        ).reset_index()

        # Calculate the trials dropped due to epoch drop
        meta_df['trials_dropped'] = meta_df['total_trials'] - meta_df['trials_after_epoch_drop']
        means.append(np.mean(meta_df['trials_dropped']))

        total_nr.append(np.sum(meta_df['total_trials']))

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot the total number of trials
        sns.barplot(x='block', y='total_trials', data=meta_df, color='lightblue', label='Total Trials')

        # Plot the trials after epoch drop (stacked on top of the remaining trials)
        sns.barplot(x='block', y='trials_after_epoch_drop', data=meta_df, color='darkblue', label='Trials After Epoch Drop')

        # Plot the trials with missing triggers (stacked on top of total trials)
        sns.barplot(x='block', y='trials_with_missing_triggers', data=meta_df, color='red', label='Trials with Missing Triggers')

        # Add labels and title
        plt.xlabel('Block')
        plt.ylabel('Number of Trials')
        plt.title(f'Sub{subID} {modality}: Number of Trials by Block with Missing Triggers and Dropped Epochs Highlighted')
        plt.legend()
        sns.despine(trim=True)
        plt.savefig(os.path.join(fig_dir, f'sub{subID}_{modality}_trialnumber_checks_90µVepoch_cutoff.png'))
        #plt.show()

        # load summary data
        meta_summary= pd.read_csv(os.path.join(prepro_path, f'sub-0{subID}_ses-{ses_label}_task-{session_type}_metasummary.tsv'), sep='\t') # processed (overview of how many trials end up after removing all bad epochs)

        #  Plot behavioural

        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=meta_summary,
            x='intensity_level',
            y='percent_yes_before_trialdrop',
            hue='block',
            marker='o'
        )

        # Add labels and title
        sns.despine(trim=True)
        plt.xlabel("Intensity Level")
        plt.ylabel("Proportion of Yes Responses (%)")
        plt.title("Response Proportion by Intensity Level and Block")
        plt.legend(title="Block")
        sns.despine(trim=True)
        plt.savefig(os.path.join(fig_dir, f'sub{subID}_{modality}_responses.png'))
        #plt.show()

        # TODO: also plot the absolute intensity values 
        
# %%
print(f'Mean: {np.mean(means)} nr. of trials dropped due to epoch drop:')
print(np.sum(total_nr))
# %%
