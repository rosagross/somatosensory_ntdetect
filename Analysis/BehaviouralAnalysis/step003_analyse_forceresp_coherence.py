"""
File: step001_analyse_force.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-10-14
Description: Analyse coherence between respiration and force data.
Read in the previously computed coherence values and delay and plot the results here.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from plotting_style import figure_style

colors, _ = figure_style()


# set directory paths
WDDIR = os.getenv("WDDIR")
RAWDIR = os.getenv("RAWDIR")
DATADIR = os.getenv("DATADIR")
data_dir = os.path.join(RAWDIR, "FingerForceData")
RESDIR = os.getenv("RESDIR")
fig_dir = os.path.join(RESDIR, "Figures", "Main", "force_results", "single")
fig_dir_all = os.path.join(RESDIR, "Figures", "Main", "force_results", "grand")
if not os.path.exists(fig_dir_all):
    os.makedirs(fig_dir_all)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

##########################
# Coherence Analysis
##########################

# 1. How strong is the coherence between respiration and force?
# 2. Is the delay between respiration and force consistent across subjects? 

all_coherence = pd.DataFrame()
subjects = range(13, 53)
for subID in subjects:
    print(f"Processing subject 0{subID}...")
    blocks = range(1, 9)
    for block in blocks:
        derivatives_dir = os.path.join(
            DATADIR, "derivatives", "eeg-preproc", f"sub-0{subID}", "ses-mecha"
        )
        derivatives_dir_path = os.path.join(DATADIR, derivatives_dir, "force_resp_data")

        coherence_path = os.path.join(
                derivatives_dir_path, f"sub-0{subID}_ses-mecha_task-NT_run-{block}_coherence_values.csv"
            )
        if os.path.exists(coherence_path):
            # read in coherence info
            coherence_df = pd.read_csv(coherence_path)
            all_coherence = pd.concat((all_coherence, coherence_df), ignore_index=True)

# only compute for blocks where force trigger exists
#all_coherence = all_coherence[all_coherence['force_trigger_exists']=='Yes']
# %%


# plot frequency with highest coherence 
plt.figure(figsize=(15, 6))
sns.boxplot(x="sub", y="max_coh_respfreq", data=all_coherence)
plt.axhline(y=0.25, color='r', linestyle='--', label='normal heart rate')
plt.ylabel("Frequency with max. coherence (Hz)")
plt.xlabel("Subject")
plt.xticks(rotation=45)
plt.title("Frequency with max. coherence \nbetween respiration and force", fontsize=20)
sns.despine(trim=True)
plt.savefig(os.path.join(fig_dir_all, "frequency_max_coherence_resp_force.png"), dpi=300)
plt.show()

# %%

plt.figure(figsize=(15, 6))
sns.boxplot(x="sub", y="delay_max_respfreq", data=all_coherence)
plt.ylabel("Delay (sec)")
plt.xlabel("Subject")
plt.xticks(rotation=45)
plt.title("Delay between respiration and force \nat frequency with max. coherence", fontsize=20)
sns.despine(trim=True)
plt.savefig(os.path.join(fig_dir_all, "delay_max_coherence_resp_force.png"), dpi=300)
plt.show()
# %%
# plot coherence magnitude per subject and block 
plt.figure(figsize=(15, 6))
sns.boxplot(x="sub", y="max_coh_value", data=all_coherence)
plt.ylabel("Max. coherence value")
plt.xlabel("Subject")
plt.xticks(rotation=45)
plt.title("Max. coherence value \nbetween respiration and force", fontsize=20)
sns.despine(trim=True)
plt.savefig(os.path.join(fig_dir_all, "max_coherence_value_resp_force.png"), dpi=300)
plt.show()

# %%

# plot average response for electrical and mechanical
# excluding bad blocks for analyisis


