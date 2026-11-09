"""
File: step005_beh_plots.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-11-09
Description: Plot some interesting aspects of behavioural data.
- response by grouped force values (grouped in three quantiles)
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
from plotting_style import figure_style

colors, _ = figure_style()

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'beh_results')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)
figuredir_force = os.path.join(RESDIR, 'Figures', 'Main', 'force_results')
if not os.path.exists(figuredir_force):
    os.makedirs(figuredir_force)

# %%   
subs = np.arange(13, 53) # parse_subjects()
modalities = ['Electrical', 'Mechanical']
file_suffix = ''  # or '_prestim'

# collect all data
df_all = pd.DataFrame()

for subID in subs:
    print(f'subject {subID}')
    for modality in modalities:

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
        df_meta = pd.read_csv(meta_file, index_col=False, sep='\t')
        df_metasummary = pd.read_csv(os.path.join(output_path, f'sub-0{subID}_ses-{ses_label}_task-NT_metasummary{file_suffix}.tsv'), index_col=False, sep='\t')

        # append data
        df_all = pd.concat([df_all, df_meta], ignore_index=True)

# extract catch trials and high intensity trials
df_all_catch = df_all[df_all['intensity_level']==0]
df_all_catch_high = df_all[(df_all['intensity_level']==0)|(df_all['intensity_level']==7)]
df_all_catch_high_unsure = df_all[((df_all['intensity_level']==0)&(df_all['response_decoded']=='Yes'))|((df_all['intensity_level']==7)&(df_all['response_decoded']=='No'))]
df_all_high = df_all[df_all['intensity_level']>5]

# remove bad blocks and trials
#df_all = df_all[df_all['analysis_excluded']==False]
#df_all = df_all[(df_all['intensity_level']>1)&(df_all['intensity_level']<7)]
df_all = df_all[df_all['response_decoded'].notna()]

#%%  define color palette
palette = {
    'No': colors['unperceived'], 
    'Yes': colors['perceived']
}
palette_modality = {
    'Electrical': colors['electrical'], 
    'Mechanical': colors['mechanical']
}

# %% 
# Plot response times
fig = plt.figure()
plt.xlabel('Reaction Time (sec)')
sns.histplot(df_all, x='response_time', hue='response_decoded', palette=palette)
sns.despine(fig, trim=True, bottom=False)
plt.savefig(os.path.join(figuredir, 'response_time_distribution.png'), dpi=300)
plt.show()

# %%
# response time per modality

fig = plt.figure()
plt.ylabel('Reaction Time (sec)', fontsize=14)
plt.xlabel('Response', fontsize=14)
sns.barplot(df_all, x='modality', y='response_time', palette=palette_modality)
sns.despine(fig, trim=True, bottom=False)
plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(figuredir, 'response_time_barplot_modality.png'), dpi=300)
plt.show()

#%% Plot response times
df_all['response_time_log'] = np.log(df_all['response_time'])
# compute t-test between response times
response_yes = df_all[df_all['response_decoded']=='Yes']['response_time']
response_no = df_all[df_all['response_decoded']=='No']['response_time']
# log-transform response times
response_yes = np.log(response_yes)
response_no = np.log(response_no)
t_stat, p_val = scipy.stats.ttest_ind(response_yes, response_no)
print(f'T-test between response times for Yes and No responses: t={t_stat:.5f}, p={p_val:.10f}')

fig = plt.figure()
plt.ylabel('Reaction Time (sec)', fontsize=14)
plt.xlabel('Response', fontsize=14)
sns.barplot(df_all, x='response_decoded', y='response_time', palette=palette)
sns.despine(fig, trim=True, bottom=False)
plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(figuredir, 'response_time_distribution_barplot.png'), dpi=300)
plt.show()

# %%
# response times per intensity level

fig = plt.figure()
plt.ylabel('Reaction Time (sec)', fontsize=14)
plt.xlabel('Response', fontsize=14)
sns.barplot(df_all, x='response_decoded', y='response_time', hue='intensity_level')
sns.despine(fig, trim=True, bottom=False)
plt.legend(title='Intensity Level', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(figuredir, 'response_time_distribution_barplot_intensity.png'), dpi=300)
plt.show()


# %%
# Response time across trialID 
plt.figure(figsize=(6,4))
sns.lineplot(data=df_all, x='trialID', y='response_time', hue='response_decoded', palette=palette, legend=False)
plt.title('Response Time across Trials')
plt.ylabel('Response Time (sec)')
plt.xlabel('Trial Number')
#plt.legend(title='Response')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_time_across_trials.png'), dpi=300)
plt.show()

# %%
# Response time across blocks 
plt.figure(figsize=(6,4))
sns.lineplot(data=df_all, x='block', y='response_time', hue='response_decoded', palette=palette)
plt.title('Response Time across Blocks')
plt.ylabel('Response Time (sec)')
plt.xlabel('Block')
plt.legend(title='Response')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_time_across_blocks.png'), dpi=300)
plt.show()

# %%
# response time only for catch trials
df_all_catch = df_all_catch.dropna(subset=['response_decoded'])
plt.figure(figsize=(6,4))
sns.histplot(data=df_all_catch, x='response_time', hue='response_decoded', palette=palette)
plt.title('Response for Catch Trials (Intensity Level 0)')
plt.ylabel('Count')
plt.xlabel('Response')
#plt.legend(title='Response', labels=['No', 'Yes'])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_catch_trials.png'), dpi=300)
plt.show()

# %% response time only for high intensities
df_all_high = df_all_high.dropna(subset=['response_decoded'])
plt.figure(figsize=(6,4))
sns.histplot(data=df_all_high, x='response_time', hue='response_decoded', palette=palette)
plt.title('Response for Intensity Level > 5')
plt.ylabel('Count')
plt.xlabel('Response')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_catch_trials.png'), dpi=300)
plt.show()

# %%
df_all_catch_high = df_all_catch_high.dropna(subset=['response_decoded'])
plt.figure(figsize=(6,4))
sns.histplot(data=df_all_catch_high, x='response_time', hue='intensity_level')
plt.title('Response for Catch Trials versus Intensity Level 7')
plt.ylabel('Count')
plt.xlabel('Response Time')
#plt.legend(title='Intensity', labels=['Catch', '7'])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_catch_trials.png'), dpi=300)
plt.show()

# %%
df_all_catch_high_unsure = df_all_catch_high_unsure.dropna(subset=['response_decoded'])
plt.figure(figsize=(6,4))
sns.histplot(data=df_all_catch_high_unsure, x='response_time', hue='intensity_level')
plt.title('Response for Unsure Trials')
plt.ylabel('Count')
plt.xlabel('Response')
#plt.legend(title='Intensity', labels=['Catch', '7'])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_catch_trials.png'), dpi=300)
plt.show()


# %% Plot response by force quantiles
df_all = df_all[df_all['analysis_excluded']==False]
df_mecha = df_all[df_all['modality']=='Mechanical']
df_mecha = df_mecha.dropna(subset=['force', 'response_decoded'])
subjects = df_mecha['sub'].unique()

plt.figure(figsize=(6,4))
sns.countplot(data=df_mecha, x='force_quantile', hue='response_decoded', palette=palette, order=['Low', 'Medium', 'High'])
plt.title('Response by Force Quantiles') # \n n subjects: '+str(len(subjects)))
plt.ylabel('Count')
plt.xlabel('Force Quantile')
plt.legend(title=' ', loc='right', bbox_to_anchor=(1.3, 0.8))
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_by_force_quantiles.pdf'), dpi=300)


# %% Plot response by force quantiles NO FORCE TRIGGERS
df_mecha = df_all[df_all['modality']=='Mechanical']
df_mecha = df_mecha.dropna(subset=['force', 'response_decoded'])
df_mecha = df_mecha[df_mecha['force_trigger_exists']=='No']
subjects = df_mecha['sub'].unique()
plt.figure(figsize=(6,4))
sns.countplot(data=df_mecha, x='force_quantile', hue='response_decoded', palette=palette, order=['Low', 'Medium', 'High'])
plt.title('Response by Force Quantiles \n n subjects: '+str(len(subjects)))
plt.ylabel('Count')
plt.xlabel('Force Quantile')
plt.legend(title=' ')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_by_force_quantiles_NOTRIGGERS.png'), dpi=300)

# %% Plot response by force quantiles
df_mecha = df_all[df_all['modality']=='Mechanical']
df_mecha = df_mecha.dropna(subset=['force', 'response_decoded'])
df_mecha = df_mecha[df_mecha['force_trigger_exists']=='Yes']
subjects = df_mecha['sub'].unique()

plt.figure(figsize=(6,4))
sns.countplot(data=df_mecha, x='force_quantile', hue='response_decoded', palette=palette, order=['Low', 'Medium', 'High'])
plt.title('Response by Force Quantiles \n n subjects: '+str(len(subjects)))
plt.ylabel('Count')
plt.xlabel('Force Quantile')
plt.legend(title=' ')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'response_by_force_quantiles_WITHTRIGGER.png'), dpi=300)

# %%
# plot number of trials per subject and modality 
trial_counts = df_all.groupby(['sub', 'modality']).size().reset_index(name='trial_count')
plt.figure(figsize=(15,6))
sns.barplot(data=trial_counts, x='sub', y='trial_count', hue='modality', palette=palette_modality)
plt.title('Number of Trials per Subject and Modality')
plt.ylabel('Number of Trials')
plt.xlabel('Subject')
plt.legend(title='Modality', bbox_to_anchor=(1.0, 1), loc='upper left')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'number_of_trials_per_subject_modality.png'), dpi=300)

# %%
# plot the force per subject and per block
df_mecha = df_all[df_all['modality']=='Mechanical']
df_mecha = df_mecha.dropna(subset=['force'])
subjects = df_mecha['sub'].unique()

plt.figure(figsize=(15,6))
sns.barplot(data=df_mecha, x='sub', y='force', color=palette_modality['Mechanical'])
sns.despine(trim=True)
plt.title('Mean Force per Subject and Block')
plt.ylabel('Force (grams)')
plt.xlabel('Subject')
plt.legend(title='Block', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figuredir_force, 'force_per_subject_block.png'), dpi=300)
# %%
# now plot this for every subject
subjects = df_mecha['sub'].unique()
for subID in subjects:
    sub_dir = os.path.join(figuredir_force, 'single', f'sub-0{subID}')
    df_sub = df_mecha[df_mecha['sub']==subID]
    plt.figure(figsize=(8,4))
    sns.barplot(data=df_sub, x='block', y='force', palette='viridis')
    sns.despine(trim=True)
    plt.title(f'Mean Force per Block - Subject {subID}')
    plt.ylabel('Force (grams)')
    plt.xlabel('Block')
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, f'force_per_block_sub-{subID:02d}.png'), dpi=300)
    plt.close()

# %%
# plot the Standard Deviation of force per subject and block
plt.figure(figsize=(15,6))
stds = df_mecha.groupby(['sub', 'block'])['force'].std().reset_index()
sns.barplot(data=stds, x='sub', y='force', hue='block', color=palette_modality['Mechanical'])
sns.despine(trim=True)
plt.title('Standard Deviation of Force per Subject and Block')
plt.ylabel('Force Standard Deviation')
plt.xlabel('Subject')
plt.savefig(os.path.join(figuredir_force, 'force_std_per_subject_block.png'), dpi=300)

# %%
# now plot the Standard Deviation for every subject separately
for subID in subjects:
    sub_dir = os.path.join(figuredir_force, 'single', f'sub-0{subID}')
    df_sub = df_mecha[df_mecha['sub']==subID]
    stds_sub = df_sub.groupby('block')['force'].std().reset_index()
    plt.figure(figsize=(8,4))
    sns.barplot(data=stds_sub, x='block', y='force', palette='viridis')
    sns.despine(trim=True)
    plt.title(f'Standard Deviation of Force per Block - Subject {subID}')
    plt.ylabel('Force Standard Deviation')
    plt.xlabel('Block')
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, f'force_std_per_block_sub-{subID:02d}.png'), dpi=300)
    plt.close()
    
# %%
