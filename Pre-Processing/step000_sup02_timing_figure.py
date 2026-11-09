'''
Supplementary figure Sup_02: trigger timing of the main blocks.

Interval between the pre-stimulus cue trigger and the stimulus trigger.
Panel A mechanical, panel B electrical. Mean and SD (ms) are printed on each panel.
All blocks (.set files) of a session are pooled; the block of every trial is kept in the csv output.
'''

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import mne
import pandas as pd
import seaborn as sns
from plotting_style import figure_style

# %% settings
RAWDIR = os.getenv('RAWDIR')
WDDIR = os.getenv('WDDIR')
RESDIR = os.getenv('RESDIR')
if not RAWDIR:
    raise EnvironmentError("RAWDIR is not set.")
if not RESDIR:
    raise EnvironmentError("RESDIR is not set.")
if not WDDIR:
    raise EnvironmentError("WDDIR is not set.")

colors, _ = figure_style()
palette_modality = {
    'Mechanical': colors['mechanical'],
    'Electrical': colors['electrical'],
}

session_type = 'NT'
modalities = ['Mechanical', 'Electrical']
subjects = np.arange(13, 53)
n_trials_expected = 198

stim_triggers = {'Electrical': 'A - Stimulation', 'Mechanical': 'B - Stimulation'}
# Trigger code 4 = cue onset.
cue_code = '4'

# %% extract trigger intervals
times_df = pd.DataFrame()

for modality in modalities:
    for subID in subjects:
        subID = str(subID)

        EEG_data_dir = os.path.join(
            RAWDIR, 'sub-0' + subID, 'ID' + subID + '_' + modality, 'converted_EEG', session_type
        )
        all_sessions = glob.glob(os.path.join(EEG_data_dir, '*.set'))
        if len(all_sessions) == 0:
            print(f"No EEG .set files found for subject {subID} modality {modality}, skipping...")
            continue
        if len(all_sessions) < 2:
            raise ValueError(
                f"Expected at least 2 .set files in {EEG_data_dir}, found {len(all_sessions)}."
            )

        # Order blocks by recording timestamp and session number from the file name
        # (e.g. 2025-02-13T095322_session8.set); the position in this order is the block id.
        def block_sort_key(path):
            match = re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{6})_session(\d+)\.set', os.path.basename(path))
            if match is None:
                raise ValueError(f"Unexpected .set file name: {path}")
            return match.group(1), int(match.group(2))

        all_sessions = sorted(all_sessions, key=block_sort_key)

        for block, set_file in enumerate(all_sessions, start=1):
            raw = mne.io.read_raw_eeglab(set_file, preload=False)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            sfreq = raw.info['sfreq']

            stim_trigger = stim_triggers[modality]
            if stim_trigger not in event_dict:
                raise KeyError(f"{stim_trigger!r} not found in events of {set_file}.")

            # event_dict maps annotation description -> integer event id
            samples = events_from_annot[:, 0]
            ids = events_from_annot[:, 2]
            trigger_samples = {
                'cue': samples[ids == event_dict[cue_code]],
                'stim': samples[ids == event_dict[stim_trigger]],
            }

            # Skip incomplete blocks.
            incomplete = [
                f"{len(trig)} {key}" for key, trig in trigger_samples.items()
                if len(trig) != n_trials_expected
            ]
            if incomplete:
                print(f"Subject {subID} {modality} block {block} ({os.path.basename(set_file)}) is incomplete "
                      f"({', '.join(incomplete)} triggers, expected {n_trials_expected}), skipping...")
                continue

            block_df = pd.DataFrame({
                'cue_to_stim': (trigger_samples['stim'] - trigger_samples['cue']) / sfreq * 1000,
            })
            block_df['subject'] = int(subID)
            block_df['modality'] = modality
            block_df['block'] = block
            block_df['block_file'] = os.path.basename(set_file)
            times_df = pd.concat([times_df, block_df], ignore_index=True)

# %% statistics
stats = {}
for modality in modalities:
    mod_df = times_df[times_df['modality'] == modality]
    if mod_df.empty:
        raise ValueError(f"No complete sessions found for modality {modality}.")
    values = mod_df['cue_to_stim'].to_numpy()
    stats[modality] = (values.mean(), values.std(ddof=1), len(values))
    print(f"{modality} cue_to_stim: mean = {values.mean():.2f} ms, "
          f"SD = {values.std(ddof=1):.2f} ms, n = {len(values)} trials, "
          f"{mod_df['subject'].nunique()} subject(s), {mod_df.groupby(['subject', 'block']).ngroups} block(s)")

# %% per-subject summary, outlier check and csv export
# Summary per subject and modality (ms), pooled over all blocks. The robust z-score compares
# each subject's mean / SD with the median and MAD across subjects of the same modality.
summary_df = (
    times_df.groupby(['modality', 'subject'])
    .agg(n_blocks=('block', 'nunique'), n_trials=('cue_to_stim', 'size'), mean_ms=('cue_to_stim', 'mean'),
         sd_ms=('cue_to_stim', 'std'), median_ms=('cue_to_stim', 'median'),
         min_ms=('cue_to_stim', 'min'), max_ms=('cue_to_stim', 'max'))
    .reset_index()
)


def robust_z(x):
    mad = np.median(np.abs(x - np.median(x)))
    if mad == 0:
        raise ValueError("MAD across subjects is zero, robust z-score is undefined.")
    return 0.6745 * (x - np.median(x)) / mad


outlier_threshold = 3.5
for stat_col in ['mean_ms', 'sd_ms']:
    summary_df[f'z_{stat_col}'] = summary_df.groupby('modality')[stat_col].transform(robust_z)
summary_df['outlier'] = (
    (summary_df['z_mean_ms'].abs() > outlier_threshold) | (summary_df['z_sd_ms'].abs() > outlier_threshold)
)

# Largest single deviation of any trial from the modality median (ms), per subject and per block.
modality_median = times_df.groupby('modality')['cue_to_stim'].transform('median')
times_df['abs_dev_from_median_ms'] = (times_df['cue_to_stim'] - modality_median).abs()
max_dev = times_df.groupby(['modality', 'subject'])['abs_dev_from_median_ms'].max().rename('max_abs_dev_ms')
summary_df = summary_df.merge(max_dev.reset_index(), on=['modality', 'subject'])

# Per-block summary to trace a flagged subject back to the problematic block. The block mean is
# also given relative to the modality median, and the late trials (> late_trial_ms from the
# modality median) are counted.
late_trial_ms = 5
block_df_summary = (
    times_df.groupby(['modality', 'subject', 'block', 'block_file'])
    .agg(n_trials=('cue_to_stim', 'size'), mean_ms=('cue_to_stim', 'mean'), sd_ms=('cue_to_stim', 'std'),
         min_ms=('cue_to_stim', 'min'), max_ms=('cue_to_stim', 'max'),
         max_abs_dev_ms=('abs_dev_from_median_ms', 'max'),
         n_late_trials=('abs_dev_from_median_ms', lambda x: int((x > late_trial_ms).sum())))
    .reset_index()
)
block_df_summary['mean_dev_from_modality_median_ms'] = (
    block_df_summary['mean_ms'] - block_df_summary['modality'].map(times_df.groupby('modality')['cue_to_stim'].median())
)

# Block with the largest single-trial deviation for each subject.
worst_block = (
    block_df_summary.loc[block_df_summary.groupby(['modality', 'subject'])['max_abs_dev_ms'].idxmax(),
                         ['modality', 'subject', 'block', 'block_file']]
    .rename(columns={'block': 'worst_block', 'block_file': 'worst_block_file'})
)
summary_df = summary_df.merge(worst_block, on=['modality', 'subject'])

print(f"\nSubjects with |robust z| > {outlier_threshold} in mean or SD of the interval:")
flagged = summary_df[summary_df['outlier']]
print(flagged.to_string(index=False) if len(flagged) else "none")
print("\nBlocks of flagged subjects:")
flagged_blocks = block_df_summary.merge(flagged[['modality', 'subject']], on=['modality', 'subject'])
print(flagged_blocks.to_string(index=False) if len(flagged_blocks) else "none")
print("\nTop 5 subjects by largest single-trial deviation from the modality median:")
print(summary_df.sort_values('max_abs_dev_ms', ascending=False).head(5).to_string(index=False))

statsdir = os.path.join(RESDIR, 'Statistics', 'eeg')
os.makedirs(statsdir, exist_ok=True)
trials_csv = os.path.join(statsdir, 'timing_cue_to_stim_trials.csv')
summary_csv = os.path.join(statsdir, 'timing_cue_to_stim_per_subject.csv')
blocks_csv = os.path.join(statsdir, 'timing_cue_to_stim_per_block.csv')
times_df.to_csv(trials_csv, index=False)
summary_df.to_csv(summary_csv, index=False)
block_df_summary.to_csv(blocks_csv, index=False)
print("Saved:", trials_csv)
print("Saved:", summary_csv)
print("Saved:", blocks_csv)

# %% figure
fig, axes = plt.subplots(1, len(modalities), figsize=(7, 3), squeeze=False)
axes = axes[0]

for ax, modality, letter in zip(axes, modalities, 'AB'):
    values = times_df.loc[times_df['modality'] == modality, 'cue_to_stim'].to_numpy()
    mean_ms, std_ms, n_trials = stats[modality]

    ax.hist(values, bins=30, color=palette_modality[modality], alpha=0.8)
    ax.axvline(mean_ms, color='black', linestyle='--', linewidth=0.8)
    ax.text(
        0.97, 0.95, f"mean = {mean_ms:.1f} ms\nSD = {std_ms:.1f} ms\nn = {n_trials}",
        transform=ax.transAxes, ha='right', va='top',
    )
    ax.set_title(modality)
    ax.set_xlabel('Cue to stimulus (ms)')
    ax.text(
        -0.2, 1.05, letter, transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='bottom', ha='left',
    )
    sns.despine(ax=ax)

axes[0].set_ylabel('Trials')

fig.tight_layout()

manuscript_imgdir = os.path.join(WDDIR, 'Writing', 'img')
os.makedirs(manuscript_imgdir, exist_ok=True)
figname = "Sup_02_timing_checks"
fig.savefig(os.path.join(manuscript_imgdir, f"{figname}.pdf"), dpi=300)
fig.savefig(os.path.join(manuscript_imgdir, f"{figname}.png"), dpi=300)
print("Saved:", os.path.join(manuscript_imgdir, f"{figname}.pdf"))
