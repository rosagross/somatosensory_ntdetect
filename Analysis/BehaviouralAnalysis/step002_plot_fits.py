"""
File: step001_prep_gamm.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-05-22
Description: Plot the fitted sigmoid curves of the behavioural data. 
Based on the pre-registered r2 threshold value of 0.6 we will exclude the specific block.

"""

# %%
import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from analysis_helper_functions import *
from scipy.optimize import curve_fit
from mne.io import RawArray
from plotting_style import figure_style

colors, _ = figure_style()

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'beh_results')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)

# %%

# if no arguments are used when starting script in commandline
# all subjects from 13 to 52 will be processed
# use --help to see options
subs = np.arange(13, 53) # parse_subjects()

# compute fitted linear and sigmoidal curves
all_fitting_df = pd.DataFrame() 
modalities = ['Electrical', 'Mechanical']
plot = True
intensity_value = 'intensity_level'  # or 'intensity_norm'
file_suffix = ''  # or '_prestim'

# plotting colors
colors_blocks = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

all_subs_metasummary = pd.DataFrame()

for subID in subs:
    fig_index = (int(subID)-int(subs[0]))%5
    # every 5 subjects create a separate figure for better visibility
    if fig_index==0:
        fig, axs = plt.subplots(5,4, figsize=(10,15), sharey=True, sharex=True)
        fig.suptitle('Electrical vs. Mechanical Stimulation')

    subID = '0' + str(subID) if int(subID)<10 else str(subID)

    for m, modality in zip([0,2], modalities):

        results = []
        if modality=='Electrical':
            ses_label = 'elec' 
        elif modality == 'Mechanical':
            ses_label = 'mecha' 

        # define directory
        sub_dir = os.path.join(DATADIR, f'sub-0{subID}', f'ses-{ses_label}', 'eeg')
        output_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{ses_label}')
        output_path = os.path.join(DATADIR, output_dir) 

        meta_file = os.path.join(output_path, f'sub-0{subID}_ses-{ses_label}_task-NT_allmetadata{file_suffix}.tsv')
        if not os.path.exists(meta_file):
            print(f'Metadata file for sub-{subID} ses-{ses_label} not found, skipping subject')
            continue

         # load metadata and metasummary
        df_metasummary = pd.read_csv(os.path.join(output_path, f'sub-0{subID}_ses-{ses_label}_task-NT_metasummary{file_suffix}.tsv'), index_col=False, sep='\t')
        
        for block in range(1,9):
            df = df_metasummary[(df_metasummary['block']==block) & (df_metasummary['intensity_level']>0)]
            df.reset_index(drop=True, inplace=True)

            # load fit 
            fitting_file = os.path.join(output_path, f'sub-0{subID}_ses-{ses_label}_task-NT_run-{block}_behavfits.tsv')
            if not os.path.exists(fitting_file):
                print(f'Fitting file for sub-{subID} ses-{ses_label} block-{block} not found, skipping block')
                continue
            
            fitting_df = pd.read_csv(fitting_file, sep='\t') 
            print(fitting_df[['r2_sigm', 'block']], df['analysis_excluded'].values)
            # clean data
            fitting_df = fitting_df[df['analysis_excluded']==False]
            if fitting_df.empty:
                print(f'No valid fitting data for sub-{subID} ses-{ses_label} block-{block}, skipping block')
                continue

            if plot==True:
                ax = axs[fig_index]
                ax[0].set_ylabel('Yes response proportion (%)', fontsize=16)
                ax[m+1].plot(fitting_df['intensity_level'], fitting_df['percent_yes_fitted'], label=f'{block}', color=colors_blocks[block-1])
                ax[m+1].legend('')
                #ax[m+1].legend(title='Block')
                # set x ticks
                ax[m+1].set_xticks(np.arange(1, 8, 1))
                sns.lineplot(ax=ax[m], data=df, x='intensity_level', y='percent_yes_before_trialdrop', marker='o', hue='block', palette=[colors_blocks[block-1]])
                ax[m].set_ylim(0, 100)
                ax[m].legend('')

                for a in ax:
                    a.set_title(f'Subject {subID}')
                    sns.despine(trim=True, ax=a)
                    if fig_index==4:
                        a.set_xlabel('Intensity')

            # append fitted data for all subjects
            all_fitting_df = pd.concat((all_fitting_df, fitting_df), ignore_index=True)

        
        # save figure and make some annotations
        if (fig_index==4) or (subID==subs[-1]):
            #annotate_fig('task-NT_allmetadata.tsv')
            plt.tight_layout()
            plt.savefig((os.path.join(figuredir, f'SigmoidFitting_{intensity_value}_sub{int(subID)-5}-{subID}_cleaned.pdf')), dpi=300)


        # save in df for later plotting 
        df_metasummary['sub'] = subID
        all_subs_metasummary = pd.concat((all_subs_metasummary, df_metasummary))
        
plt.tight_layout()

# %%
# Plot fitting distribution
numeric_cols = ['r2_sigm', 'k_sigm', 'x0_sigm', 'r2_linear', 'k_linear', 'intercept_linear']

# unique values for sub block and modality
all_fitting_df_plot = all_fitting_df.drop_duplicates(subset=['sub', 'block', 'modality'])

# Convert columns to float (errors='coerce' will turn non-numeric values into NaN)
all_fitting_df_plot[numeric_cols] = all_fitting_df_plot[numeric_cols].apply(pd.to_numeric, errors='coerce')
filtered_fit_df = all_fitting_df_plot[all_fitting_df_plot['r2_sigm']>=0.1]

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# Top row - sigm parameters
sns.histplot(data=filtered_fit_df, x='r2_sigm', ax=axes[0, 0], color='skyblue', kde=True)
axes[0, 0].set_xlabel('R2 sigmoidal fit')
sns.histplot(data=filtered_fit_df, x='k_sigm', ax=axes[0, 1], color='skyblue', kde=True)
axes[0, 1].set_xlabel('steepmness (k) sigmoidal fit')
sns.histplot(data=filtered_fit_df, x='x0_sigm', ax=axes[0, 2], color='skyblue', kde=True)
axes[0, 2].set_xlabel('midpoint (x0) sigmoidal fit')
# Bottom row - linear parameters
sns.histplot(data=filtered_fit_df, x='r2_linear', ax=axes[1, 0], color='salmon', kde=True)
axes[1, 0].set_xlabel('R2 linear fit')
sns.histplot(data=filtered_fit_df, x='k_linear', ax=axes[1, 1], color='salmon', kde=True)
axes[1, 1].set_xlabel('steepmness (k) linear fit')
sns.histplot(data=filtered_fit_df, x='intercept_linear', ax=axes[1, 2], color='salmon', kde=True)
axes[1, 2].set_xlabel('intercept linear fit')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'Fitting_parameters_sigmoid_linear.pdf'))

# %%
# plot average responses
# for demonstation purpuses include intensity 0, 1,7 (but otherwise exclude bad trials/block for other intensities)


filtered_metasummary = all_subs_metasummary[((all_subs_metasummary['analysis_excluded']==False)|(all_subs_metasummary['intensity_level']<2)|(all_subs_metasummary['intensity_level']>6))]
print(len(filtered_metasummary))
# Sum ntrials per subject and modality (across all blocks)
trials_per_sub_modality = filtered_metasummary.groupby(['sub', 'modality'])['ntrials'].sum()

# Find subjects who have >= 600 trials in ALL modalities
valid_subs = trials_per_sub_modality[trials_per_sub_modality >= 600].groupby('sub').size()
valid_subs = valid_subs[valid_subs == filtered_metasummary['modality'].nunique()].index

# Filter the dataframe
filtered_metasummary = filtered_metasummary[filtered_metasummary['sub'].isin(valid_subs)]
print(len(filtered_metasummary))

palette_modality = {
    'Electrical': colors['electrical'], 
    'Mechanical': colors['mechanical']
}

fig = plt.figure(figsize=(5,3))

sns.lineplot(filtered_metasummary, x='intensity_level', y='percent_yes_before_trialdrop', hue='modality', palette=palette_modality)
plt.xlabel('Intensity level')
plt.ylabel('Yes Response %', fontsize=15)
plt.legend(title='', fontsize=13)
plt.hlines(50, 0, 7, 'red', '--')
plt.xticks(np.arange(0,8,1))
sns.despine(trim=True)
plt.savefig(os.path.join(figuredir, 'average_behaviour.pdf'))

# %%
# plot how many blocks per subjects and modality are left
blocks_total = all_subs_metasummary[all_subs_metasummary['analysis_excluded']==False].drop_duplicates(subset=['sub', 'block', 'modality'])
blocks_per_sub = blocks_total[blocks_total['analysis_excluded']==False].groupby(['sub', 'modality']).size().reset_index(name='num_blocks')

fig = plt.figure(figsize=(15,6))
sns.barplot(blocks_per_sub, x='sub', y='num_blocks', hue='modality', palette=palette_modality)
plt.xlabel('Subject ID')    
plt.ylabel('Number of included blocks', fontsize=15)
plt.ylim(0,8)
plt.legend(title='', fontsize=13)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'num_included_blocks_per_subject.pdf'))

# %%
# create a plot that shows that shows the distribution of the intensities 
# within the modalities (histogram) 


# use only non-excluded trials/blocks
intensity_dist_df = all_subs_metasummary[
    all_subs_metasummary['analysis_excluded'] == False
].copy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, modality in zip(axes, modalities):

    modality_df = intensity_dist_df[
        intensity_dist_df['modality'] == modality
    ]

    sns.histplot(
        data=modality_df,
        x='intensity_level',
        weights='ntrials_after_trialdrop',
        discrete=True,
        binwidth=1,
        binrange=(-0.5, 7.5),
        color=palette_modality[modality],
        ax=ax
    )

    ax.set_title(f'{modality} stimulation')
    ax.set_xlabel('Intensity level')
    ax.set_ylabel('Number of trials')
    ax.set_xticks(np.arange(0, 8, 1))
    sns.despine(trim=True, ax=ax)

plt.tight_layout()
plt.savefig(
    os.path.join(figuredir, 'intensity_distribution_per_modality.png'),
    dpi=300
)