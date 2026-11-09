"""
File: step007_catch_trial_ERP.py
Author: Rosa Grossmann
Date: 2026-06-01
Description:
- Load broadband epochs (_preprestim_corrected) for all subjects.
- Extract catch trials only (intensity_level == 0).
- Plot grand-average ERP timecourses (perceived vs unperceived) at channels
  CP4, CP3, CPz, C4 in a 2x2 grid, separately per modality.
- Baseline: (-0.8, -0.7) s (inferred automatically from _preprestim suffix).
- Surface Laplacian (CSD) applied before plotting.
"""
# %%
import os
import sys
import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser

DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
WDDIR = os.getenv("WDDIR")
if WDDIR and WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import load_clean_epochs
from plotting_style import figure_style

# --- Parameters ---
file_suffix_epochs = "_preprestim"
l_freq_orig = 0.1
l_freq = 1
h_freq = 40
all_subjects = list(range(13, 53))
modalities = ["elec", "mecha"]
channels = ["CP4", "CP3", "CPz", "C4"]
MIN_TRIALS = 1  # skip a condition for a subject if fewer catch trials than this

colors, dpi = figure_style()

parser = ArgumentParser(description="Catch-trial ERP: perceived vs unperceived")
parser.add_argument("--sub", nargs="+", type=str, default=None,
                    help="Subject IDs to process (e.g. --sub 013 014). Default: all subjects.")
args, _ = parser.parse_known_args()
subjects = [int(s) for s in args.sub] if args.sub else all_subjects

figuredir = os.path.join(RESDIR, "Figures", "Main", "eeg_results", "catch_trial_ERP")
os.makedirs(figuredir, exist_ok=True)

SCALE = 1e6  # MNE stores data in V/m²; multiply to get µV/m²


def apply_surface_laplacian(epochs):
    """Surface Laplacian (CSD) to reduce volume conduction before channel metrics."""
    eeg_epochs = epochs.copy().pick("eeg")
    if len(eeg_epochs.ch_names) < 2:
        raise ValueError("Need at least 2 EEG channels for surface Laplacian.")
    return mne.preprocessing.compute_current_source_density(eeg_epochs)


# Storage: modality → response label → channel → list of (n_times,) subject-level means
subj_data = {
    mod: {label: {ch: [] for ch in channels} for label in ("Yes", "No")}
    for mod in modalities
}
times_ref = None

for modality in modalities:
    for subID in subjects:
        # Faulty recording setup for these subjects 
        if modality == "mecha" and subID in (23, 36):
            print(f"Skipping sub {subID} {modality}: faulty recording.")
            continue

        sub_dir = os.path.join(
            DATADIR, "derivatives", "eeg-preproc",
            f"sub-0{subID}", f"ses-{modality}",
        )
        base_name = (
            f"sub-0{subID}_ses-{modality}_task-NT_"
            f"lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}"
        )
        epoch_fif = os.path.join(sub_dir, f"{base_name}_epochs{file_suffix_epochs}.fif")

        if not os.path.exists(epoch_fif):
            print(f"Not found: {epoch_fif}")
            continue

        # load_clean_epochs applies baseline (-0.8, -0.7) s inferred from the _preprestim suffix,
        # and filters out analysis_excluded, drop_epoch, and non-Yes/No responses.
        epochs, _ = load_clean_epochs(epoch_fif, exclude_analysis=False)

        if "intensity_level" not in epochs.metadata.columns:
            print(f"  Sub {subID} {modality}: missing intensity_level column, skipping.")
            continue

        catch = epochs[epochs.metadata["intensity_level"] == 0]
        if len(catch) == 0:
            print(f"  Sub {subID} {modality}: no catch trials.")
            continue

        catch_csd = apply_surface_laplacian(catch)

        if times_ref is None:
            times_ref = catch_csd.times

        n_yes = (catch_csd.metadata["response_decoded"] == "Yes").sum()
        n_no  = (catch_csd.metadata["response_decoded"] == "No").sum()
        print(f"  Sub {subID} {modality}: {n_yes} Yes / {n_no} No catch trials.")

        for label in ("Yes", "No"):
            mask = (catch_csd.metadata["response_decoded"] == label).to_numpy()
            if mask.sum() < MIN_TRIALS:
                print(f"    → {label}: only {mask.sum()} trials (< {MIN_TRIALS}), skipping condition.")
                continue
            evoked = catch_csd[mask].average()
            for ch_name in channels:
                if ch_name not in evoked.ch_names:
                    continue
                ch_idx = evoked.ch_names.index(ch_name)
                subj_data[modality][label][ch_name].append(evoked.data[ch_idx])

if times_ref is None:
    print("No valid data for any subject. Nothing to plot.")
    sys.exit(0)

# --- Plot: one 2×2 figure per modality ---
for modality in modalities:
    any_data = any(
        len(subj_data[modality][label][ch]) > 0
        for label in ("Yes", "No")
        for ch in channels
    )
    if not any_data:
        print(f"No data for {modality}, skipping figure.")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ch_idx, ch_name in enumerate(channels):
        ax = axes[ch_idx]
        legend_handles = []

        for label, color_key, display_label in [
            ("Yes", "perceived",   "Perceived"),
            ("No",  "unperceived", "Unperceived"),
        ]:
            ch_trials = subj_data[modality][label][ch_name]
            if len(ch_trials) == 0:
                continue
            stack = np.stack(ch_trials, axis=0)  # (n_subjects, n_times)
            grand_mean = stack.mean(axis=0) * SCALE
            grand_sem  = (stack.std(axis=0) / np.sqrt(stack.shape[0])) * SCALE
            n_subs = stack.shape[0]
            line, = ax.plot(
                times_ref, grand_mean,
                color=colors[color_key],
                linewidth=1.5,
                label=f"{display_label} (n={n_subs})",
            )
            ax.fill_between(
                times_ref, grand_mean - grand_sem, grand_mean + grand_sem,
                color=colors[color_key], alpha=0.25,
            )
            legend_handles.append(line)

        cue_line   = ax.axvline(-0.4, color="grey",  linestyle="--", linewidth=0.8)
        stim_line  = ax.axvline( 0.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)

        if ch_idx == 0:
            legend_handles += [cue_line, stim_line]

        ax.set_title(ch_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CSD (µV/m²)")

    # Shared legend built from all unique labels across axes.
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, lbl in zip(h, l):
            if lbl not in all_labels:
                all_handles.append(handle)
                all_labels.append(lbl)
    # Add event line labels manually.
    from matplotlib.lines import Line2D
    all_handles += [
        Line2D([0], [0], color="grey",  linestyle="--", linewidth=0.8),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8),
    ]
    all_labels += ["Cue (-0.4 s)", "Stimulus (0 s)"]

    fig.legend(all_handles, all_labels, loc="upper right", bbox_to_anchor=(1.15, 1.0))

    n_yes_subs = len(subj_data[modality]["Yes"]["CP4"])
    n_no_subs  = len(subj_data[modality]["No"]["CP4"])
    fig.suptitle(
        f"Catch-trial ERP — {modality}  |  "
        f"Perceived: n={n_yes_subs}  Unperceived: n={n_no_subs}"
    )
    fig.tight_layout()

    out_path = os.path.join(figuredir, f"catch_trial_ERP_{modality}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

print("Done.")
