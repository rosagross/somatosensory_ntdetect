"""
File: step003_behavioural_analysis_singlesub.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-06-20
Description: In this file the BEHAVIOURAL (!!!) data of the near-threshold detection paradigm is inspected
on a single subject level and exclusion criteria are applied:
- for plotting we use different exclusion criteria than for the analysis. Applying the same critera woul
lead to a skewed picture of the data. That's why we apply the exclusion criteria for plotting here!
    - exclude nan responses
    - exclude intensity level 0, 1 and 7 
    - exclude blocks with a false positive rate of more than 30%

All in all this should lead to the same number of trial when applying those criteria to the EEG meta data (stores same information about trials). 
"""

# %% imports 
import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import scipy
import pandas as pd
import seaborn as sns
from analysis_helper_functions import *
from plotting_style import figure_style
colors, dpi = figure_style()

RAWDIR = os.getenv('RAWDIR')
DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'beh_results')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)

output_folder = os.path.join(DATADIR, 'derivatives', 'beh-preproc')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#%% 
# QUESTIONNAIRE
# read questionnaire file
q_name = 'NTDetect_Questionnaires.csv'
q_df = pd.read_csv(os.path.join(RAWDIR, q_name), header=1)
q_df.columns = ['ID', 'Age', 'Sex', 'Q3', 'Q5', 'Q6', 'Q7', 'Q8', 'comments']

# exclude pilots
q_df_main = q_df[q_df['ID']>12]
count_sex = q_df_main.groupby('Sex')['ID'].count()
print(f'{int(count_sex.iloc[0])} females, {int(count_sex.iloc[1])} males')



# %% 
# ###########  MAIN BLOCK ############## 
# Load data of all subjects and blocks

session_type = 'NT'
modalities = ['Electrical', 'Mechanical']
blocks = np.arange(1,9)
subs = np.arange(9,53)
all_NT_df = pd.DataFrame()

# loop through subjects and blocks
for modality in modalities:
    ses_label = 'elec' if modality=='Electrical' else 'mecha'

    for sub in subs:
        if sub < 10: 
            sub = '0' + str(sub) 
        
        for block in blocks:

            behav_dir = os.path.join(DATADIR, 'sub-0' + str(sub), f'ses-{ses_label}', 'eeg')
            filepath = os.path.join(behav_dir, f'sub-0{sub}_ses-{ses_label}_task-{session_type}_run-{block}_beh.tsv')
            file_badEpochs = os.path.join(behav_dir, f'sub-0{sub}_ses-{ses_label}_task-{session_type}_run-{block}_badEpochs.csv')
            
            if os.path.exists(filepath): 
            
                # load the data
                dataNT_df = pd.read_csv(filepath, sep='\t')
                # attach also info about bad epochs 
                # badEpochs_df = pd.read_csv(file_badEpochs)

                # concatenate
                all_NT_df = pd.concat((all_NT_df, dataNT_df))
            else:
                #print(sub, modality, block)
                #print(f'File not found: {filepath}')
                continue

# %% 

# check how many trials are kept after criterion
# --> group by block, intensity_level and response 
grouped = (
    all_NT_df
    .groupby(['modality', 'sub', 'block', 'intensity_level', 'actual_intensity', 'response_decoded'])
    .size()
    .unstack(fill_value=0) # sets 'response_decoded' as name for the index column..
    .reset_index()
)
grouped.columns.name = None

# normalize (intensity 4 should be 1 after normalisation)
grouped['intensity_norm'] = grouped.groupby(['modality', 'sub', 'block'])['actual_intensity'] \
                            .transform(lambda x: x / x.iloc[4])

# check valitidy per intensity level
data_df_merged = all_NT_df.merge(
    grouped[['modality', 'sub', 'block', 'intensity_level', 'actual_intensity', 'intensity_norm']],
    on=['modality', 'sub', 'block', 'intensity_level', 'actual_intensity']
)

# Keep only valid trials 
filtered_df = data_df_merged[data_df_merged['response_decoded'].notna()]

# remove blocks with a false positive rate of more than 30%
exclude_blocks = grouped[
    grouped['intensity_level'] == 0
].groupby(['modality', 'sub', 'block'])['Yes'].sum()
exclude_blocks = exclude_blocks[exclude_blocks > 5].reset_index()
grouped_filtered = grouped.merge(
    exclude_blocks[['modality', 'sub', 'block']],
    on=['modality', 'sub', 'block'],
    how='left',
    indicator=True
)
grouped_filtered = grouped_filtered[grouped_filtered['_merge'] == 'left_only'].drop(columns='_merge')
grouped_filtered = grouped_filtered[~grouped_filtered['intensity_level'].isin([0, 1, 7])]

# delete the false positive rate also from the trial-wise dataframe (filtered_df)
filtered_df = filtered_df.merge(
    grouped_filtered[['modality', 'sub', 'block', 'intensity_level']],
    on=['modality', 'sub', 'block', 'intensity_level'],
    how='left',
    indicator=True
)
filtered_df = filtered_df[filtered_df['_merge'] == 'both'].drop(columns='_merge')

# %%

# Count remaining datapoints per block
counts = filtered_df.groupby(['modality', 'sub']).size().reset_index(name='n_datapoints')

# Plot how many data points per subject and modality stay
plt.figure(figsize=(10,5))
sns.barplot(data=counts, x='sub', y='n_datapoints', hue='modality')
sns.despine()
plt.axhline(y=600, color='r', linestyle='-')
plt.ylabel('Remaining datapoints')
plt.xlabel('Subject')
plt.title('Remaining datapoints per subject after filtering')
plt.show()

# Compute and plot response proportions
total_counts = (
    filtered_df
    .groupby(['modality', 'sub', 'block', 'intensity_level', 'actual_intensity', 'intensity_norm'])
    .size()
    .reset_index()
)
total_counts.columns.name = None
total_counts.columns = ['modality', 'sub', 'block', 'intensity_level', 'actual_intensity', 'intensity_norm', 'total_count']
total_counts['yes_percent'] = np.array((np.array(grouped_filtered['Yes']) / np.array(total_counts['total_count']) * 100))

# Unfiltered counts
total_counts_unfiltered = (
    data_df_merged
    .groupby(['modality', 'sub', 'block', 'intensity_level', 'actual_intensity', 'intensity_norm'])
    .size()
    .reset_index()
)

total_counts_unfiltered.columns.name = None
total_counts_unfiltered.columns = ['modality', 'sub', 'block', 'intensity_level', 'actual_intensity', 'intensity_norm', 'total_count']
total_counts_unfiltered['yes_percent'] = np.array((np.array(grouped['Yes']) / np.array(total_counts_unfiltered['total_count']) * 100))


#%% Data exclusion 
summary_count = total_counts.groupby(['modality', 'sub'], as_index=False)['total_count'].sum()
valid_mecha = len(summary_count[(summary_count['modality']=='Mechanical')&(summary_count['sub']>12)&(summary_count['total_count']>600)])
valid_elec = len(summary_count[(summary_count['modality']=='Electrical')&(summary_count['sub']>12)&(summary_count['total_count']>600)])
print(f'Valid subjects \nElectrical: {valid_elec}\nMechanical: {valid_mecha}')

#%% Plot the response proportions for all main subjects

intensity_type = 'intensity_level' # 'intensity_level' or 'intensity_norm'

fig, axs = plt.subplots(1,2, figsize=(20,10))
total_counts_mechanical = total_counts[(total_counts['modality']=='Mechanical')&(total_counts['sub']>12)]
ax_mecha = axs[0]
sns.lineplot(ax=ax_mecha, data=total_counts_mechanical, x=intensity_type, y='yes_percent', marker='o', hue='sub')
sns.despine()
ax_mecha.set_ylabel('Yes response proportion (%)')
ax_mecha.set_xlabel('Intensity')
ax_mecha.set_ylim(0, 100)
ax_mecha.set_title('Mechanical Stimulation')

total_counts_electrical = total_counts[(total_counts['modality']=='Electrical')&(total_counts['sub']>12)]
ax_elec = axs[1]
sns.lineplot(ax=ax_elec, data=total_counts_electrical, x=intensity_type, y='yes_percent', marker='o', hue='sub')
sns.despine()
ax_elec.set_ylabel('Yes response proportion (%)')
ax_elec.set_xlabel('Intensity')
ax_elec.set_ylim(0, 100)
ax_mecha.set_title('Electrical Stimulation')

plt.show()

#%%
# PILOT FILTERED
# Plot subjects next to each other 
subs_pilot = np.arange(9,13)

fig, axs = plt.subplots(len(subs_pilot),2, figsize=(10,15), sharey=True)
fig.suptitle('Electrical vs. Mechanical Stimulation')
for sub, ax in zip(subs_pilot, axs):
    
    counts_sub_mechanical = total_counts[(total_counts['modality']=='Mechanical')&(total_counts['sub']==sub)]
    counts_sub_electrical = total_counts[(total_counts['modality']=='Electrical')&(total_counts['sub']==sub)]

    sns.lineplot(ax=ax[0], data=counts_sub_electrical, x='intensity_level', y='yes_percent', marker='o', hue='block')
    sns.lineplot(ax=ax[1], data=counts_sub_mechanical, x='intensity_level', y='yes_percent', marker='o', hue='block')
    sns.despine(trim=True, ax=ax[0])
    sns.despine(trim=True, ax=ax[1])

    ax[0].set_title(f'Subject {sub}')
    ax[1].set_title(f'Subject {sub}')
    ax[0].set_ylabel('Yes response proportion (%)')
    ax[0].set_xlabel('Intensity')
    ax[1].set_xlabel('Intensity')
    ax[0].set_ylim(0, 100)
    ax[1].set_ylim(0, 100)
    ax[0].set_xticks([2,3,4,5,6])
    ax[1].set_xticks([2,3,4,5,6])

plt.tight_layout()
plt.show()

# %% 
# compute fitted linear and sigmoidal curves
subs_tofilter = np.arange(13,53) # choose subjects to look at (13-52 are main subjects)
all_fitting_df = pd.DataFrame() 
results = []
plot = True

for sub in subs_tofilter:
    print('SUB', sub)
    mecha_unfiltered = total_counts_unfiltered[(total_counts_unfiltered['modality']=='Mechanical')&(total_counts_unfiltered['sub']==sub)]
    mecha_unfiltered = mecha_unfiltered[mecha_unfiltered['intensity_level']>0]
    elec_unfiltered = total_counts_unfiltered[(total_counts_unfiltered['modality']=='Electrical')&(total_counts_unfiltered['sub']==sub)]
    elec_unfiltered = elec_unfiltered[elec_unfiltered['intensity_level']>0]
    
    # every 5 subjects create a separate figure for better visibility
    if (sub-subs_tofilter[0])%5==0:
        fig, axs = plt.subplots(5,4, figsize=(10,15), sharey=False)
        fig.suptitle('Electrical vs. Mechanical Stimulation')

    for modality in modalities:
            
        if modality=='Electrical':

            for block in np.unique(elec_unfiltered['block']):
                intensities_block = np.array(elec_unfiltered[elec_unfiltered['block']==block]['intensity_level'])
                proportions_block = np.array(elec_unfiltered[elec_unfiltered['block']==block]['yes_percent'])

                # Fit the logistic function to the data
                initial_guess_sigm = [5, np.mean(intensities_block)] 
                
                # check if there are enough response points to fit (above 0)
                if len(proportions_block[proportions_block > 0])<3:
                    print('Not enough response points to fit a curve')
                    [r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept, x_fit, y_fit] = [np.nan]*8
                else:
                    [r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept, x_fit, y_fit] = fit_curves(intensities_block, proportions_block, initial_guess_sigm)

                # append results to dataframe
                results.append([modality, sub, block, r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept])

                if plot==True:
                    ax = axs[(sub-subs_tofilter[0])%5]
                    ax[1].plot(x_fit, y_fit, label=f'{block}')
                    ax[1].legend(title='Block')
        
        columns = ['modality', 'sub', 'block', 'r2_sigm', 'k_sigm', 'x0_sigm', 'r2_linear', 'k_linear', 'intercept_linear']

        if modality=='Mechanical':

            for block in np.unique(mecha_unfiltered['block']):
                intensities_block = np.array(mecha_unfiltered[mecha_unfiltered['block']==block]['intensity_level'])
                proportions_block = np.array(mecha_unfiltered[mecha_unfiltered['block']==block]['yes_percent'])
                
                # Fit the logistic function to the data
                initial_guess_sigm = [5, np.mean(intensities_block)] 

                # check if there are enoguh response points to fit (above 0)
                if len(proportions_block[proportions_block > 0])<3:
                    print('Not enough response points to fit a curve')
                    [r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept, x_fit, y_fit] = [np.nan]*8
                else:
                    [r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept, x_fit, y_fit] = fit_curves(intensities_block, proportions_block, initial_guess_sigm)

                # append results to dataframe
                results.append([modality, sub, block,r_squared, k_sigm, x0_sigm, r_squared_lin, slope, intercept])

                if plot==True:
                    ax = axs[(sub-subs_tofilter[0])%5]
                    ax[3].plot(x_fit, y_fit, label=f'{block}')
                    ax[3].legend(title='Block')

    all_fitting_df = pd.DataFrame(np.array(results), columns=columns)
    
    if plot==True:
        sns.lineplot(ax=ax[0], data=elec_unfiltered, x='intensity_level', y='yes_percent', marker='o', hue='block')
        sns.lineplot(ax=ax[2], data=mecha_unfiltered, x='intensity_level', y='yes_percent', marker='o', hue='block')

        for a in ax:
            sns.despine(trim=True, ax=a)
            a.set_xlabel('Intensity')
            a.set_ylim(0, 100)
            a.set_title(f'Subject {sub}')

        ax[0].set_ylabel('Yes response proportion (%)')

        if ((sub-subs_tofilter[0])%5==4) or (sub==subs_tofilter[-1]):
            plt.tight_layout()
            plt.savefig((os.path.join(figuredir, f'SigmoidFitting_sub{sub-5}-{sub}.pdf')))

# %%
# Plot fitting distribution
numeric_cols = ['r2_sigm', 'k_sigm', 'x0_sigm', 'r2_linear', 'k_linear', 'intercept_linear']

# Convert columns to float (errors='coerce' will turn non-numeric values into NaN)
all_fitting_df[numeric_cols] = all_fitting_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# plot the R2 distribution for both modalities
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
sns.histplot(data=all_fitting_df, x='r2_sigm', ax=axes[0], color='skyblue', kde=True)
axes[0].set_xlim(0, 1)
axes[0].set_xlabel('R2 linear fit')
sns.histplot(data=all_fitting_df, x='r2_linear', ax=axes[1], color='salmon', kde=True)
axes[1].set_xlim(0, 1)
axes[1].set_xlabel('R2 linear fit')

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'FullR2Distributions.pdf'))
plt.show()


#%%
# fitting before exclusion
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
plt.savefig(os.path.join(figuredir, 'Fitting_before_parameters_sigmoid_linear.pdf'))

# %%
# DATA EXCLUSION BASED ON FITTING 
# ... filter if the fit is below 0.6
# make table with subjects and block that did not pass the criterion
cutoff_value = 0.6
excluded_fit = all_fitting_df[all_fitting_df['r2_sigm']<cutoff_value]
print(f'Excluded subjects based on sigmoidal fit <{cutoff_value}')
print(excluded_fit[['modality', 'sub', 'block', 'r2_sigm']])

# exclusion based on r2 fit
filtered_fit_df = all_fitting_df[all_fitting_df['r2_sigm']>=cutoff_value]

# based on x0
#cutoff_value_max = 6
#cutoff_value_min = 2
#filtered_fit_df = all_fitting_df[all_fitting_df['x0_sigm']<=cutoff_value_max]
#filtered_fit_df = filtered_fit_df[filtered_fit_df['x0_sigm']>=cutoff_value_min]
#filtered_fit_df = filtered_fit_df[filtered_fit_df['r2_sigm']>cutoff_value]


print("filtered mean x0", np.mean(filtered_fit_df['x0_sigm']))
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# Top row - sigm parameters
sns.histplot(data=filtered_fit_df, x='r2_sigm', ax=axes[0, 0], color='skyblue', kde=True)
axes[0, 0].set_xlabel('R2 sigmoidal fit')
axes[0, 0].set_xlim(0.4, 1)
sns.histplot(data=filtered_fit_df, x='k_sigm', ax=axes[0, 1], color='skyblue', kde=True)
axes[0, 1].set_xlabel('steepmness (k) sigmoidal fit')
sns.histplot(data=filtered_fit_df, x='x0_sigm', ax=axes[0, 2], color='skyblue', kde=True)
axes[0, 2].set_xlabel('midpoint (x0) sigmoidal fit')
# Bottom row - linear parameters
sns.histplot(data=filtered_fit_df, x='r2_linear', ax=axes[1, 0], color='salmon', kde=True)
axes[1, 0].set_xlabel('R2 linear fit')
# set xlim to 0.4 to 1
axes[1, 0].set_xlim(0.4, 1)
sns.histplot(data=filtered_fit_df, x='k_linear', ax=axes[1, 1], color='salmon', kde=True)
axes[1, 1].set_xlabel('steepness (k) linear fit')
sns.histplot(data=filtered_fit_df, x='intercept_linear', ax=axes[1, 2], color='salmon', kde=True)
axes[1, 2].set_xlabel('intercept linear fit')

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(figuredir, 'FilteredFittingParameterDistributions.pdf'))
plt.show()

# %%

# Count excluded blocks per subject and modality
counts = (
    excluded_fit
    .groupby(['sub', 'modality'])
    .size()
    .reset_index(name='n_blocks_excluded')
)
counts['exclude'] = counts['n_blocks_excluded'] > 4

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=counts,
    x='sub', y='n_blocks_excluded', hue='modality',
    dodge=True
)

# Add threshold line
plt.axhline(4, color='red', linestyle='--', label='Exclusion threshold (3)')
plt.ylabel("Number of excluded blocks")
plt.xlabel("Subject")
plt.title("Excluded blocks per subject and modality")
plt.legend()
plt.tight_layout()
plt.show()
print('Subjects to exclude based on more than 3 blocks excluded:', len(counts[counts['exclude']==True]))

# %%
# exclude blocks based on fitting
keys = ['sub', 'block', 'modality']
filtered_fit_df['sub'] = pd.to_numeric(filtered_fit_df['sub'])
filtered_fit_df['block'] = pd.to_numeric(filtered_fit_df['block'])
mask = total_counts.set_index(keys).index.isin(filtered_fit_df.set_index(keys).index)
filtered_total_counts = total_counts[mask]
print(len(filtered_total_counts), 'remaining data points after filtering based on fitting')

# exclude subjects based on more than 4 blocks excluded
keys = ['sub', 'modality']
counts['sub'] = pd.to_numeric(counts['sub'])
mask = filtered_total_counts.set_index(keys).index.isin(counts[counts['exclude']==True].set_index(keys).index)
filtered_total_counts = filtered_total_counts[~mask]
print(len(filtered_total_counts), 'remaining data points after filtering based on subjects with too many excluded blocks')

# Save excluded subjects to csv
filtered_total_counts.to_csv(os.path.join(output_folder, 'summary_allsubjects_beh_filtered.csv'))

# Now filter the trial-wise dataframe accordingly
keys = ['sub', 'block', 'modality']
mask = filtered_df.set_index(keys).index.isin(filtered_fit_df.set_index(keys).index)
filtered_df = filtered_df[mask]
print(len(filtered_df), 'remaining trials after filtering based on fitting and subjects with too many excluded blocks')

keys = ['sub', 'modality']
mask = filtered_df.set_index(keys).index.isin(counts[counts['exclude']==True].set_index(keys).index)
filtered_df = filtered_df[~mask]
print(len(filtered_df), 'remaining trials after filtering based on fitting and subjects with too many excluded blocks')
filtered_df.to_csv(os.path.join(output_folder, 'valid_trials_allsubjects_beh.csv'))



# %% 
# MAIN 
# Plot remaining subjects after filtering

subs_main = np.arange(13,52)

fig.suptitle('Electrical vs. Mechanical Stimulation\n\n')
for sub in subs_main:
    print(sub)

    counts_sub_mechanical = filtered_total_counts[(filtered_total_counts['modality']=='Mechanical')&(filtered_total_counts['sub']==sub)]
    counts_sub_electrical = filtered_total_counts[(filtered_total_counts['modality']=='Electrical')&(filtered_total_counts['sub']==sub)]
    
    # every 5 subjects create a separate figure for better visibility
    if (sub-subs_main[0])%5==0:
        fig, axs = plt.subplots(5,2, figsize=(10,15), sharey=True)
        fig.suptitle('Electrical vs. Mechanical Stimulation')
    
    ax = axs[(sub-subs_main[0])%5]
    sns.lineplot(ax=ax[0], data=counts_sub_electrical, x='intensity_norm', y='yes_percent', marker='o', hue='block')
    sns.lineplot(ax=ax[1], data=counts_sub_mechanical, x='intensity_norm', y='yes_percent', marker='o', hue='block')
    sns.despine(trim=False, ax=ax[0])
    sns.despine(trim=False, ax=ax[1])

    ax[0].set_title(f'Subject {sub}')
    ax[1].set_title(f'Subject {sub}')
    ax[0].set_ylabel('Yes response proportion (%)')
    ax[0].set_xlabel('Intensity')
    ax[1].set_xlabel('Intensity')
    #ax[0].set_ylim(0, 100)
    #ax[1].set_ylim(0, 100)
    ax[0].set_xticks([2,3,4,5,6])   
    ax[1].set_xticks([2,3,4,5,6])

    if ((sub-subs_main[0])%5==4) or (sub==subs_main[-1]):
        plt.tight_layout()
        plt.savefig((os.path.join(figuredir, f'filtered_sub{sub-5}-{sub}.pdf')))
        plt.show()

