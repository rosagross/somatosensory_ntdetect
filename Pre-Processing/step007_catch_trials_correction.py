"""
Script to extract basline of catch trials and subtract it from all trials
to get rid of visual stimulus contamination.
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
import h5py
from scipy.optimize import curve_fit
from mne.io import RawArray
from plotting_style import figure_style

# Ensure repo root is on sys.path so general_helpers can be imported reliably.
WDDIR = os.getenv('WDDIR')
if not WDDIR:
    raise EnvironmentError("WDDIR is not set. Please export WDDIR to the repo root.")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import parse_cli_args
colors, _ = figure_style()

DATADIR = os.getenv('DATADIR')
WDDIR = os.getenv('WDDIR')
RESDIR = os.getenv('RESDIR')
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'eeg_results', 'catch_trials_correction')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)

statsdir = os.path.join(RESDIR, 'Statistics', 'eeg')
if not os.path.exists(statsdir):
    os.makedirs(statsdir)


# %%

# load data for plotting for a specific channel
subjects = np.arange(13, 53)

palette = {
    'No': colors['unperceived'], 
    'Yes': colors['perceived']
}
palette_modality = {
    'elec': colors['electrical'], 
    'mecha': colors['mechanical']
}

l_freq, h_freq, file_suffix, modalities, _, _ = parse_cli_args(
    description=(
        "Catch-trial baseline correction. Optional arguments override "
        "frequency cutoffs, file suffix, and modalities."
    ),
    default_l_freq=1,
    default_h_freq=40,
    default_file_suffix="_preprestim",
    default_meta_suffix="_preprestim",
    default_modalities=["elec", "mecha"] #,"mecha"]
)
sfreq = 500
l_freq_orig = 0.1

# %%
all_epochs = []
all_catch_epochs = []
all_noncatch_epochs = []
all_noncatch_epochs_pre = []

# Create subject and trial tracking dataframes
subject_trial_counts = pd.DataFrame(columns=['subject', 'modality', 'total_trials'])


for modality in modalities:
    count_subs = 0 
    
    for subID in subjects:

        # exclude subject 23 mechanical (swapped ground and reference electrode)
        # exclude mecha 36 (force values were exceptionally high!)
        #if ((subID == 23) or (subID == 36)) and modality == 'mecha':
        #    print(f"Skipping subject {subID} modality {modality} due to faulty recording setup.")
        #    continue 
                
        sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')

        # read in epoch 
        all_epochs_filename = (
            f"sub-0{subID}_ses-{modality}_task-NT_"
            f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{file_suffix}"
        )
        meta_filename = (
            f"sub-0{subID}_ses-{modality}_task-NT_"
            f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_allmetadata{file_suffix}.tsv"
        )
        epoch_fif = os.path.join(sub_dir, f"{all_epochs_filename}.fif")
        print('subject', subID)
        
        if epoch_fif is None or not os.path.exists(epoch_fif):
            print(f"Epoch file for subject {subID} modality {modality} not found, skipping...")
            continue

        # read metadata 
        metadata_df = pd.read_csv(os.path.join(sub_dir, meta_filename), sep='\t')
        print(len(metadata_df), "trials subject", subID, "modality", modality)
        
        # remove trials of bad epochs from metadata
        metadata_df = metadata_df[metadata_df['drop_epoch'] == False].reset_index(drop=True)
        
        # load epochs 
        epochs = mne.read_epochs(epoch_fif, preload=True)
        if len(epochs) != len(metadata_df):
            raise ValueError(
                f"Metadata rows ({len(metadata_df)}) and epochs ({len(epochs)}) differ "
                f"for subject {subID} modality {modality}."
            )

        # Split catch trials and non-catch trials using a boolean mask.
        catch_mask = metadata_df['intensity_level'].to_numpy() == 0
        noncatch_mask = ~catch_mask
        catch_epochs = epochs[catch_mask]
        # Keep a pre-correction copy of non-catch epochs for later plotting.
        noncatch_epochs_pre = epochs[noncatch_mask].copy()
        noncatch_epochs = epochs[noncatch_mask]

        # Compute catch baseline and subtract it from non-catch trials only.
        # This keeps catch trials unchanged while correcting non-catch trials in-place.
        catch_mean = catch_epochs._data.mean(axis=0)
        epochs._data[noncatch_mask] -= catch_mean
        # Re-slice after correction so non-catch epochs reflect the updated data.
        noncatch_epochs = epochs[noncatch_mask]

        # Save corrected epochs (catch + non-catch) to mirror step005.
        # Keep metadata/epoch counts aligned per the data integrity rule.
        if len(epochs) != len(metadata_df):
            raise ValueError(
                f"Metadata rows ({len(metadata_df)}) and corrected epochs ({len(epochs)}) differ "
                f"for subject {subID} modality {modality}."
            )
        
        if len(epochs) > 0:
            # Save the corrected epochs (all trials) in a new fif file
            epochs.save(os.path.join(sub_dir, f"{all_epochs_filename}_corrected.fif"), overwrite=True)
        else:
            print('No data left.. continue with next subject')
            continue
        
        corrected_data = epochs.get_data()
        corrected_channel_names = epochs.ch_names
        corrected_times = epochs.times

        with h5py.File(os.path.join(sub_dir, f"{all_epochs_filename}_corrected.h5"), "w") as f:
            for ch_idx, ch_name in enumerate(corrected_channel_names):
                grp = f.create_group(ch_name)
                grp.create_dataset(
                    "data",
                    data=corrected_data[:, ch_idx, :],
                    compression="gzip",
                    compression_opts=9,
                )
                grp.create_dataset("times", data=corrected_times)
                grp.attrs["channel_name"] = ch_name
                grp.attrs["n_trials"] = corrected_data.shape[0]
                grp.attrs["n_timepoints"] = corrected_data.shape[2]

        all_catch_epochs.append(catch_epochs)
        all_noncatch_epochs.append(noncatch_epochs)
        all_noncatch_epochs_pre.append(noncatch_epochs_pre)
        all_epochs.append(epochs)
    
    all_epochs_fif = mne.concatenate_epochs(all_epochs)

# %%

channels = ["CP4", "O2"]
all_catch_epochs_fif = mne.concatenate_epochs(all_catch_epochs)
all_noncatch_epochs_fif = mne.concatenate_epochs(all_noncatch_epochs)
all_noncatch_epochs_pre_fif = mne.concatenate_epochs(all_noncatch_epochs_pre)

for channel in channels:
    times = all_epochs_fif.times
    mean_all = all_epochs_fif.get_data(picks=[channel]).mean(axis=0).squeeze()
    mean_catch = all_catch_epochs_fif.get_data(picks=[channel]).mean(axis=0).squeeze()
    mean_noncatch = all_noncatch_epochs_fif.get_data(picks=[channel]).mean(axis=0).squeeze()

    modalities = ['Mechanical', 'Electrical']
    for modality in modalities:

        all_epochs_mod = all_epochs_fif[f'modality=="{modality}"']
        all_catch_epochs_mod = all_catch_epochs_fif[f'modality=="{modality}"']
        all_noncatch_epochs_mod = all_noncatch_epochs_fif[f'modality=="{modality}"']
        all_noncatch_epochs_pre_mod = all_noncatch_epochs_pre_fif[f'modality=="{modality}"']
        mean_all = all_epochs_mod.get_data(picks=[channel]).mean(axis=0).squeeze()
        mean_catch = all_catch_epochs_mod.get_data(picks=[channel]).mean(axis=0).squeeze()
        mean_noncatch = all_noncatch_epochs_mod.get_data(picks=[channel]).mean(axis=0).squeeze()
        yes_epochs = all_noncatch_epochs_mod['response_decoded=="Yes"']
        no_epochs = all_noncatch_epochs_mod['response_decoded=="No"']
        yes_epochs_pre = all_noncatch_epochs_pre_mod['response_decoded=="Yes"']
        no_epochs_pre = all_noncatch_epochs_pre_mod['response_decoded=="No"']
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, mean_all, label="All epochs", color=colors["grey"])
        plt.plot(times, mean_catch, label="Catch epochs", color="red")
        plt.plot(times, mean_noncatch, label="Non-catch (baseline corrected)", color="violet")
        if len(yes_epochs) > 0:
            mean_yes = yes_epochs.get_data(picks=[channel]).mean(axis=0).squeeze()
            plt.plot(times, mean_yes, label="Yes (non-catch corrected)", color=palette["Yes"])
        if len(no_epochs) > 0:
            mean_no = no_epochs.get_data(picks=[channel]).mean(axis=0).squeeze()
            plt.plot(times, mean_no, label="No (non-catch corrected)", color=palette["No"])
        if len(yes_epochs_pre) > 0:
            mean_yes_pre = yes_epochs_pre.get_data(picks=[channel]).mean(axis=0).squeeze()
            plt.plot(
                times,
                mean_yes_pre,
                label="Yes (non-catch pre-correction)",
                color=palette["Yes"],
                linestyle="--",
            )
        if len(no_epochs_pre) > 0:
            mean_no_pre = no_epochs_pre.get_data(picks=[channel]).mean(axis=0).squeeze()
            plt.plot(
                times,
                mean_no_pre,
                label="No (non-catch pre-correction)",
                color=palette["No"],
                linestyle="--",
            )
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("µV")
        plt.title(f"{channel} - {modality} - All vs Catch vs Non-catch")
        plt.legend()
        sns.despine(trim=True)
        plt.savefig(os.path.join(figuredir, f"Epoch_Types_{channel}_{modality}.png"), dpi=300)

# %%
# ---------------------------------------------------------------------------
# Publication supplementary figure: effect of the catch-trial correction on the ERP.
# One figure, 2 x 2: rows are channels (A: CP4, B: O2), columns are modalities
# (left mechanical, right electrical). Each panel shows the non-catch ERP before
# and after subtracting the catch-trial mean. Non-catch trials are the only
# ones the correction touches, so catch trials are left out here.
# ---------------------------------------------------------------------------
mod_labels = ['Mechanical', 'Electrical']

# Grand average is built per subject first (one ERP per subject x modality),
# then averaged across subjects, so subjects with more trials do not dominate.
per_sub_erp = {
    channel: {
        mod: {'uncorrected': [], 'corrected': []} for mod in mod_labels
    }
    for channel in channels
}

# The two lists are appended in lockstep in the loop above, so index i is the
# same subject/modality in both.
for ep_pre, ep_post in zip(all_noncatch_epochs_pre, all_noncatch_epochs):
    if len(ep_pre) != len(ep_post):
        raise ValueError(
            f"Pre-correction ({len(ep_pre)}) and corrected ({len(ep_post)}) non-catch "
            "trial counts differ - correction changed the trial set."
        )
    if not np.array_equal(ep_pre.times, ep_post.times):
        raise ValueError("Pre-correction and corrected epochs have different time axes.")

    mod = ep_pre.metadata['modality'].iloc[0]
    if mod not in mod_labels:
        raise ValueError(f"Unexpected modality label in metadata: {mod!r}")

    for channel in channels:
        per_sub_erp[channel][mod]['uncorrected'].append(
            ep_pre.get_data(picks=[channel]).mean(axis=0).squeeze()
        )
        per_sub_erp[channel][mod]['corrected'].append(
            ep_post.get_data(picks=[channel]).mean(axis=0).squeeze()
        )

# %%

start_plot = 200
stop_plot = -150
erp_times = all_noncatch_epochs_pre[0].times[start_plot:stop_plot]

# Modality colours; metadata uses the long labels, palette_modality the short ones.
modality_color = {
    'Mechanical': palette_modality['mecha'],
    'Electrical': palette_modality['elec'],
}
# Uncorrected is dashed/lighter, corrected is solid/full weight - same dashed =
# pre-correction convention as the diagnostic plot above.
condition_style = {
    'uncorrected': dict(linestyle='--', linewidth=1.0, alpha=0.65, label='Uncorrected'),
    'corrected': dict(linestyle='-', linewidth=1.5, alpha=1.0, label='Corrected'),
}

cue_times = {'Mechanical': -0.46, 'Electrical': -0.4}

fig, axes = plt.subplots(
    len(channels), len(mod_labels), figsize=(7, 6), sharex=True, sharey=True, squeeze=False
)

for row, channel in enumerate(channels):
    for col, mod in enumerate(mod_labels):
        ax = axes[row, col]

        n_subs = len(per_sub_erp[channel][mod]['corrected'])
        if n_subs == 0:
            print(f"No non-catch data for {channel} {mod}, hiding panel.")
            ax.set_visible(False)
            continue

        for cond, style in condition_style.items():
            stack = np.stack(per_sub_erp[channel][mod][cond], axis=0)  # (n_subjects, n_times)
            grand_mean = stack.mean(axis=0)
            ax.plot(
                erp_times,
                grand_mean[start_plot:stop_plot],
                color=modality_color[mod],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                label=f"{style['label']} (n={n_subs})",
            )
            # SEM across subjects is undefined for a single subject.
            if n_subs > 1:
                grand_sem = sem(stack, axis=0)
                ax.fill_between(
                    erp_times,
                    grand_mean[start_plot:stop_plot] - grand_sem[start_plot:stop_plot],
                    grand_mean[start_plot:stop_plot] + grand_sem[start_plot:stop_plot],
                    color=modality_color[mod],
                    alpha=0.25 * style['alpha'],
                    linewidth=0,
                )

        ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)  # stimulus onset
        # Cue marker (mechanical: -0.46 s, electrical: -0.4 s), only if it falls
        # inside the plotted epoch window.
        cue_time = cue_times[mod]
        if erp_times[0] <= cue_time <= erp_times[-1]:
            ax.axvline(cue_time, color='grey', linestyle='--', linewidth=0.8)

        ax.set_ylim([-1.5, 1])
        if row == 0:
            ax.set_title(mod)
        if row == len(channels) - 1:
            ax.set_xlabel('Time (s)')
        if col == 0:
            ax.set_ylabel(f'{channel} - Amplitude (µV)')
        ax.legend(loc='best')
        sns.despine(trim=True, ax=ax)

# Panel letters: A = first channel (CP4), B = second channel (O2).
for row, letter in enumerate('AB'):
    axes[row, 0].text(
        -0.28, 1.05, letter, transform=axes[row, 0].transAxes,
        fontsize=12, fontweight='bold', va='bottom', ha='left',
    )

fig.tight_layout()

# Supplementary figure goes straight into the manuscript image folder.
manuscript_imgdir = os.path.join(WDDIR, 'Writing', 'img')
os.makedirs(manuscript_imgdir, exist_ok=True)
figname = "Sup_01" #f"catch_correction_ERP{file_suffix}"
fig.savefig(os.path.join(manuscript_imgdir, f"{figname}.pdf"), dpi=300)
fig.savefig(os.path.join(manuscript_imgdir, f"{figname}.png"), dpi=300)
print("Saved:", os.path.join(manuscript_imgdir, f"{figname}.pdf"))
