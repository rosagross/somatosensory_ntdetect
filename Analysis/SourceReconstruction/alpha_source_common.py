"""
Shared configuration for the alpha-band source reconstruction scripts.

Holds paths, subject lists, analysis parameters and the output file names of the
per-session fsaverage source estimates and the group averages, so every script that
reads or writes them uses the same definitions.
"""

import os
import sys

location = "mpi"
if location == "laptop":
    WDDIR = "C:\\Users\\gross\\OneDrive - UvA\\Documents\\IMPRS_Leipzig\\MyProject\\Experiment\\Analysis\\LocalCode"
    DATADIR = "C:\\Users\\gross\\OneDrive - UvA\\Documents\\IMPRS_Leipzig\\MyProject\\Experiment\\Analysis\\LocalCode\\data"
    RECONDIR = os.path.join(DATADIR, "freesurfer")
    RESDIR = os.path.join(WDDIR, "output")
else:
    DATADIR = os.getenv("DATADIR")
    RESDIR = os.getenv("RESDIR")
    RECONDIR = os.getenv("SUBJECTS_DIR")
    WDDIR = os.getenv("WDDIR")

for _name, _value in (("DATADIR", DATADIR), ("RESDIR", RESDIR), ("SUBJECTS_DIR", RECONDIR), ("WDDIR", WDDIR)):
    if not _value:
        raise RuntimeError(f"Environment variable {_name} is not set.")

if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import is_subject_excluded

subID_mecha = [30, 32, 40, 16, 17, 18, 19, 24, 26, 27, 28, 29, 34, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
subID_elec = [15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
FAULTY_SESSIONS = {("mecha", 23), ("mecha", 36), ("elec", 19), ("elec", 28)}

modalities = ["elec", "mecha"]
subID_lists = {
    "mecha": subID_mecha,
    "elec": subID_elec,
}

preproc_suffix = "_preprestim_corrected"
alpha_suffix = f"{preproc_suffix}_alpha"  # crop-first alpha epochs from step007
band_freqs = {
    "alpha": (8, 12)
}
_hilbert_fmin = band_freqs["alpha"][0]  # lower bound of the processed band
analysis_tmin, analysis_tmax = -0.75, -0.1  # crop applied to output after Hilbert (matches step008_2)
data_cov_tmin,  data_cov_tmax  = -0.4, -0.1  # clamped to alpha epoch end
noise_tmin,     noise_tmax     = -0.8, -0.45  # ~3.7 alpha cycles; long enough to estimate alpha-band noise cov

inverse_method = "mne"

# LCMV parameter
reg = 0.05
pick_ori = None
weight_norm = "unit-noise-gain"

# MNE parameter
mne_method = "MNE" #"dSPM" 
loose=0
depth=None #0.8

# general parameter
peak_detection_window = (-0.4, -0.1)
l_freq_orig = 0.1
l_freq = 1
h_freq = 40

if inverse_method == "lcmv":
    inverse_name = f"{inverse_method}_wn{weight_norm}_reg-{reg}"
elif inverse_method == "mne":
    inverse_name = f"{inverse_method}_loose-{loose}_depth-{depth}_method-{mne_method}"
else:
    raise ValueError(f"Unsupported inverse_method={inverse_method!r}")

stc_keys = ("envelope_yes", "envelope_no", "itpc_yes", "itpc_no")


def session_list():
    """Ordered (modality, subid) pairs to process, after the fixed skips and the exclusion list."""
    sessions = []
    for modality in modalities:
        for subid in subID_lists[modality]:
            if (modality, subid) in FAULTY_SESSIONS:
                continue
            # Sessions on Pre-Processing/exclued_subs.json (low post-cue ERP SNR in O1/O2).
            if is_subject_excluded(subid, modality):
                continue
            sessions.append((modality, subid))
    return sessions


def session_dir(subid, modality):
    return os.path.join(DATADIR, "derivatives", "eeg-preproc", f"sub-0{subid}", f"ses-{modality}")


def fsaverage_stc_stem(subid, modality, key):
    """Path stem (without -lh/-rh.stc) of a subject's morphed source estimate."""
    return os.path.join(
        session_dir(subid, modality),
        f"sub-0{subid}_ses-{modality}_cond-{key}_{inverse_name}-{pick_ori}_{alpha_suffix}_fsaverage",
    )


def group_stc_stem(modality, key):
    """Path stem (without -lh/-rh.stc) of a group-average source estimate."""
    return os.path.join(
        DATADIR, "derivatives", "eeg-preproc",
        f"group_fsaverage_{modality}_cond-{key}_{inverse_name}-{pick_ori}_{alpha_suffix}",
    )
