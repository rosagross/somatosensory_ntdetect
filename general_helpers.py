import argparse
import json
import os
import re
import mne
import numpy as np

# Order matters: check longer suffix first to avoid substring conflicts.
_BASELINE_SUFFIX_MAP = [
    ("_preprestim", (-0.8, -0.7)),
    ("preprestim", (-0.8, -0.7)),
    ("_prestim", (-0.55, -0.45)),
    ("prestim", (-0.55, -0.45)),
]
_DEFAULT_BASELINE = (-0.1, 0.0)

# Exclusion list lives next to the preprocessing parameters; resolved relative to this file
# so it works regardless of the caller's cwd.
_EXCLUDED_SUBS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Pre-Processing", "exclued_subs.json"
)
_VALID_MODALITIES = {"elec", "mecha"}
# Parsed exclusion lists, keyed by resolved json path (a loop over 40 subjects must not
# re-read and re-parse the file 40 times).
_EXCLUDED_SUBS_CACHE = {}

def parse_subjects():
    """
    Parse command line arguments to get subject IDs.
    If no subject IDs are provided, process all subjects from 13 to 52.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub",
        type=str,
        nargs="+",
        help="Subject IDs to process (e.g., '01 02 15' for subjects 1, 2, and 15)",
        required=False,
    )
    args = parser.parse_args()
    if args.sub:
        subIDs = args.sub
    else:
        # process all subjects from subject 13 (no pilot subjects)
        subIDs = map(str, range(13, 53))

    return subIDs


def infer_baseline_window_from_suffix(name_or_path):
    """
    Infer baseline window from an epochs filename or suffix string.
    Returns (baseline_window, matched_suffix).
    """
    for suffix, window in _BASELINE_SUFFIX_MAP:
        if suffix in name_or_path:
            return window, suffix
    return _DEFAULT_BASELINE, ""


def apply_baseline_epochs(epochs, name_or_path, baseline_window=None):
    """
    Apply baseline correction to an MNE Epochs object using suffix inference.
    """
    if baseline_window==None:
        baseline_window, matched_suffix = infer_baseline_window_from_suffix(name_or_path)
        print(
            f"[baseline] epochs: suffix='{matched_suffix}' window={baseline_window} source={name_or_path}"
        )
    else:
        print(
            f"[baseline] epochs: window={baseline_window} source={name_or_path}"
        )
        matched_suffix = None

    epochs.apply_baseline(baseline_window)
    return baseline_window, matched_suffix

def apply_baseline_array(data, times, name_or_path):
    """
    Apply baseline correction to a NumPy array (epochs-like data).
    data can be shaped (n_trials, n_times) or (n_trials, n_channels, n_times).
    """
    baseline_window, matched_suffix = infer_baseline_window_from_suffix(name_or_path)
    tmask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    if not np.any(tmask):
        raise ValueError(
            f"Baseline window {baseline_window} not in provided times for {name_or_path}."
        )
    print(
        f"[baseline] array: suffix='{matched_suffix}' window={baseline_window} source={name_or_path}"
    )
    baseline = data[..., tmask].mean(axis=-1, keepdims=True)
    return data - baseline, baseline_window, matched_suffix


def normalize_modality_labels(entries):
    """
    Normalize modality labels to the short session labels used in filenames.

    Accepts 'Electrical'/'Mechanical' (step005-style) and 'elec'/'mecha'.
    """
    normalized = []
    for entry in entries:
        parts = [p.strip() for p in entry.split(",") if p.strip()]
        for part in parts:
            lower = part.lower()
            if lower in {"electrical", "elec", "e"}:
                normalized.append("elec")
            elif lower in {"mechanical", "mecha", "m"}:
                normalized.append("mecha")
            else:
                normalized.append(part)
    return normalized


def format_freq_for_filename(freq_value):
    """
    Format frequency values to preserve filename compatibility.
    """
    if float(freq_value).is_integer():
        return str(int(freq_value))
    return str(freq_value)


def parse_cli_args(
    description,
    default_l_freq,
    default_h_freq,
    default_file_suffix,
    default_modalities,
    default_meta_suffix=None,
    default_epoch_length=None,
    argv=None,
):
    """
    Shared CLI parser for preprocessing/analysis scripts.

    Returns:
        l_freq, h_freq, file_suffix, modalities, meta_suffix, epoch_length
    """
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(
        "--l-freq",
        type=float,
        default=default_l_freq,
        help=f"Low cutoff frequency (default: {default_l_freq}).",
    )
    parser.add_argument(
        "--h-freq",
        type=float,
        default=default_h_freq,
        help=f"High cutoff frequency (default: {default_h_freq}).",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default=default_file_suffix,
        help=f"Filename suffix (default: {default_file_suffix}).",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=default_modalities,
        help=(
            "Modalities to process. Accepts Electrical/Mechanical or elec/mecha "
            "(e.g., --modalities Electrical Mechanical)."
        ),
    )
    if default_meta_suffix is not None:
        parser.add_argument(
            "--meta-suffix",
            type=str,
            default=default_meta_suffix,
            help=f"Filename suffix for metadata (default: {default_meta_suffix}).",
        )
    if default_epoch_length is not None:
        parser.add_argument(
            "--epoch-length",
            type=int,
            default=default_epoch_length,
            help=f"Expected epoch length in samples (default: {default_epoch_length}).",
        )

    # Use parse_known_args so Jupyter/interactive kernels don't hijack our args.
    args, _ = parser.parse_known_args(argv)
    modalities = normalize_modality_labels(args.modalities)
    if not modalities:
        modalities = list(default_modalities)

    meta_suffix = getattr(args, "meta_suffix", None)
    epoch_length = getattr(args, "epoch_length", None)
    return args.l_freq, args.h_freq, args.file_suffix, modalities, meta_suffix, epoch_length


def _parse_cli_args(
    description,
    default_l_freq,
    default_h_freq,
    default_file_suffix,
    default_modalities,
    default_meta_suffix=None,
    default_epoch_length=None,
    argv=None,
):
    """
    Backwards-compatible alias for parse_cli_args.
    """
    return parse_cli_args(
        description=description,
        default_l_freq=default_l_freq,
        default_h_freq=default_h_freq,
        default_file_suffix=default_file_suffix,
        default_modalities=default_modalities,
        default_meta_suffix=default_meta_suffix,
        default_epoch_length=default_epoch_length,
        argv=argv,
    )


def load_excluded_subjects(path=None):
    """
    Read the subject exclusion list and return {modality: set of subject IDs}.

    The file is data-driven (currently: post-cue ERP SNR below the 1.5x criterion in O1/O2),
    keyed by modality because a subject can fail in one session and pass in the other.
    Result is cached per path. No fallbacks: a missing file, unparseable JSON, an absent
    'exclude' key or an unknown modality all raise.
    """
    json_path = path or _EXCLUDED_SUBS_JSON
    if json_path in _EXCLUDED_SUBS_CACHE:
        return _EXCLUDED_SUBS_CACHE[json_path]

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Subject exclusion file not found: {json_path}")
    with open(json_path, "r") as json_file:
        payload = json.load(json_file)

    if "exclude" not in payload:
        raise ValueError(f"'exclude' key missing from {json_path}. Found: {sorted(payload)}")

    excluded = {}
    for modality, subs in payload["exclude"].items():
        if modality not in _VALID_MODALITIES:
            raise ValueError(
                f"Unknown modality '{modality}' in {json_path}. "
                f"Expected one of {sorted(_VALID_MODALITIES)}."
            )
        excluded[modality] = {int(s) for s in subs}

    _EXCLUDED_SUBS_CACHE[json_path] = excluded
    return excluded


def parse_sub_modality_from_path(name_or_path):
    """
    Extract (subject_id, modality) from a BIDS-style epochs filename.

    e.g. 'sub-017_ses-mecha_task-NT_..._epochs_preprestim.fif' -> (17, 'mecha').
    Raises if either field is absent - guessing here would silently disable the exclusion.
    """
    basename = os.path.basename(name_or_path)
    sub_match = re.search(r"sub-0*(\d+)", basename)
    ses_match = re.search(r"ses-(elec|mecha)", basename)
    if sub_match is None or ses_match is None:
        raise ValueError(
            f"Cannot parse subject/modality from '{basename}'. Expected a name containing "
            f"'sub-0NN' and 'ses-elec'/'ses-mecha'."
        )
    return int(sub_match.group(1)), ses_match.group(1)


def is_subject_excluded(subject, modality, path=None):
    """
    Whether this subject/modality is on the exclusion list.

    Use this to skip a session before paying for the epoch file read; load_clean_epochs()
    applies the same check itself, but only after loading.
    """
    if modality not in _VALID_MODALITIES:
        raise ValueError(
            f"Unknown modality '{modality}'. Expected one of {sorted(_VALID_MODALITIES)}."
        )
    return int(subject) in load_excluded_subjects(path).get(modality, set())


def load_clean_epochs(epoch_file, exclude_analysis: bool = True,
                      apply_subject_exclusion: bool = True) -> tuple[mne.Epochs, mne.Evoked]:
    """
    Load epochs and compute evoked response.

    apply_subject_exclusion: honour Pre-Processing/exclued_subs.json. Subjects on the list
    come back with zero epochs (and evoked=np.nan), so the usual `if len(epochs) == 0:
    continue` guard skips them. Set False to measure an excluded subject anyway - step016,
    which produces the list, must do this to avoid a circular dependency.
    """

    if not os.path.exists(epoch_file):
        raise FileNotFoundError(f"Epoch file not found: {epoch_file}")

    epochs = mne.read_epochs(epoch_file, preload=True)
    apply_baseline_epochs(epochs, epoch_file)

    if exclude_analysis:
        epochs_clean = epochs["analysis_excluded==False"]
        epochs_clean = epochs_clean["drop_epoch==False"]
        epochs_clean = epochs_clean[((epochs_clean.metadata['response_decoded'] == 'No') | (epochs_clean.metadata['response_decoded'] == 'Yes'))]
    else:
        epochs_clean = epochs

    if apply_subject_exclusion:
        subject, modality = parse_sub_modality_from_path(epoch_file)
        if is_subject_excluded(subject, modality):
            print(f"[exclude] sub-{subject:03d} ses-{modality}: on the exclusion list "
                  f"-> returning 0 epochs")
            # Drop every trial rather than returning None, so callers keep a real Epochs
            # object and the existing `if len(epochs) == 0` guard below handles it. The
            # reason is recorded in drop_log, keeping the exclusion visible downstream.
            epochs_clean.drop(np.arange(len(epochs_clean)), reason="SUBJECT_EXCLUDED")

    if len(epochs_clean) == 0:
        print("no epochs left after cleaning!")
        evoked = np.nan
    else:
        evoked = epochs_clean.average()
    return epochs_clean, evoked