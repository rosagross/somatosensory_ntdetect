# %%
"""
Publication figure: ERP + temporal cluster-permutation significance for CP4 and C6.

Two stacked panels (a = CP4, b = C6). Each panel shows the grand-average ERP of that
channel split by modality x detection response - four traces, modality as colour and
detection as line style - and below the traces one coloured horizontal line per model
coefficient marking the time windows where that coefficient's cluster-mass test is
significant (p < 0.05).

A second figure replaces the ERP traces with the z-value trace of each coefficient, the
statistic the cluster test ran on, keeping the identical significance strip.

Inputs are the temporal (single-channel) runs written by
`unfold_temporal_permutation_clusterMass.jl` under
$RESDIR/Statistics/eeg/cluster_permutation/cluster_mass_temporal/runs/.

The two runs must share the same time window and the same model/baseline configuration,
otherwise the panels would not be comparable: mismatches are a hard error, never a
silently cropped or reconciled figure.

The Julia pipeline renders its ERPs only to PDF, so the ERP traces here are recomputed
from the same HDF5 epoch files and metadata TSVs the permutation run consumed, using the
identical baseline windows and trial filters. The recomputed per-subject trial counts are
checked against the run's own subject_trial_counts.csv, which is what proves the ERP is
averaged over exactly the trial set that was tested.
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# plotting_style is a per-directory copy; import the one next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting_style import figure_style  # noqa: E402

# %%
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Panel order, top to bottom. The FIRST channel listed is the anchor whose run
# configuration every other run is checked against.
PANEL_CHANNELS = ["CP4", "C6"]

# Run id per channel. "latest" picks the newest run folder for that channel whose
# configuration matches the anchor run; an explicit id is used verbatim. The anchor
# channel must be an explicit id.
RUN_IDS = {
    # different baselines
    #"C6": "20260902-091649__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.1_-0.0__f-6dace1beb4__ch-C6",
    #"C6": "20260903-095934__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4__ch-C6",
    #"CP4": "20260902-090055__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.1_-0.0__f-6dace1beb4__ch-CP4",
    #"CP4": "20260904-074701__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4__ch-CP4"
    # with block term
    "C6": "20261021-072249__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-96b492e3cf__ch-C6",
    "CP4": "20261020-234021__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-96b492e3cf__ch-CP4"
}

# (legend label, coefficient folder). Same five coefficients as
# plot_cluster_results_topoplot_panel_joint.py. The numeric folder prefix depends on the
# model formula of the run, so it is configuration rather than something derived.
COEFFICIENTS = [
    ("detection response", "01__response_decoded_Yes"),
    ("intensity level", "02__intensity_level_centered"),
    ("modality", "04__modality_Mechanical"),
    ("intensity level x detection response",
     "03__intensity_level_centered_response_decoded_Yes"),
    ("modality x detection response", "06__modality_Mechanical_response_decoded_Yes"),
    ("modality x intensity level", "05__modality_Mechanical_intensity_level_centered"),
    ("modality x intensity level x detection response",
     "07__modality_Mechanical_intensity_level_centered_response_decoded_Yes"),
]

ALPHA = 0.05

# The four ERP traces: modality x detection response. Colour comes from
# plotting_style.figure_style() (`mechanical` / `electrical`), perceived is solid and
# unperceived dashed, so modality reads as colour and detection as line style.
ERP_TRACES = [
    ("mecha", "Yes", "mechanical, perceived", "mechanical", "-"),
    ("mecha", "No", "mechanical, unperceived", "mechanical", (0, (6, 2))),
    ("elec", "Yes", "electrical, perceived", "electrical", "-"),
    ("elec", "No", "electrical, unperceived", "electrical", (0, (6, 2))),
]

# One colour per coefficient, in the order of COEFFICIENTS. None of these may collide with
# the ERP trace colours, which plotting_style fixes at #D55E00 (mechanical) and #CC78BC
# (electrical). The two most-used rows (detection response, modality) are a colour-blind-
# safe pair from Paul Tol's muted scheme rather than the green/red they replaced.
COEFFICIENT_COLORS = [
    "#44AA99",  # teal      (Paul Tol muted)
    "#9467BD",  # purple
    "#882255",  # wine      (Paul Tol muted)
    "#8C564B",  # brown
    "#56B4E9",  # sky blue  (Okabe-Ito)
    "#6E6E6E",  # grey
    "#1A1A1A",  # near-black
]

# Startup-environment keys that must be identical across the runs being combined.
CONFIG_KEYS = [
    "time_start_resolved",
    "time_stop_resolved",
    "epoch_stem_resolved",
    "metadata_stem_resolved",
    "baseline_tag_resolved",
    "formula_resolved",
    "n_permutations_resolved",
    "subjects_resolved",
    "excluded_pairs_resolved",
]

# Fixed filename part of the preprocessing chain, as built by `build_paths` in
# cluster_permutation_helpers.jl.
FILE_PREFIX_PARAMS = "lfreqori-0.1_lfreq-1_hfreq-40"


# %%
# ---------------------------------------------------------------------------
# Run discovery and configuration checks
# ---------------------------------------------------------------------------
def read_startup_environment(run_dir):
    """Return the run's startup_environment.csv as a plain key -> value dict."""
    path = os.path.join(run_dir, "startup_environment.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"startup_environment.csv not found: {path}")
    table = pd.read_csv(path)
    if list(table.columns) != ["key", "value"]:
        raise ValueError(f"Unexpected columns in {path}: {list(table.columns)}")
    return dict(zip(table["key"].astype(str), table["value"].astype(str)))


def config_signature(env):
    """The subset of the run configuration that has to agree across panels."""
    missing = [k for k in CONFIG_KEYS if k not in env]
    if missing:
        raise ValueError(f"startup_environment.csv is missing keys: {missing}")
    return {k: env[k] for k in CONFIG_KEYS}


def resolve_run_dir(runs_root, channel, run_id, anchor_signature):
    """
    Resolve a channel's run folder.

    An explicit run id is used as given. "latest" scans every `*__ch-<channel>` folder and
    keeps the newest one whose configuration matches `anchor_signature`; if none matches,
    every candidate is listed with the keys that differ, so the wrong-run case is visible
    rather than silently plotted.
    """
    if run_id != "latest":
        run_dir = os.path.join(runs_root, run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run folder for {channel} not found: {run_dir}")
        return run_dir

    if anchor_signature is None:
        raise ValueError(
            f"RUN_IDS['{channel}'] is 'latest' but {channel} is the anchor channel; "
            "the anchor needs an explicit run id."
        )

    suffix = f"__ch-{channel}"
    candidates = sorted(
        (os.path.join(runs_root, name) for name in os.listdir(runs_root)
         if name.endswith(suffix) and os.path.isdir(os.path.join(runs_root, name))),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run folder ending in '{suffix}' under {runs_root}")

    rejected = []
    for candidate in candidates:
        signature = config_signature(read_startup_environment(candidate))
        differing = [k for k in CONFIG_KEYS if signature[k] != anchor_signature[k]]
        if not differing:
            return candidate
        rejected.append((os.path.basename(candidate), differing))

    lines = "\n".join(
        f"  {name}\n    differs in: {', '.join(keys)}" for name, keys in rejected
    )
    raise ValueError(
        f"No run for channel {channel} matches the anchor configuration.\n"
        f"Candidates inspected (newest first):\n{lines}"
    )


# %%
# ---------------------------------------------------------------------------
# Cluster windows
# ---------------------------------------------------------------------------
def load_significant_windows(run_dir, channel, coefficient_folder, label, alpha=ALPHA):
    """
    Significant (p < alpha) cluster windows of one coefficient, plus the run's time axis
    and the tested z-trace.

    Returns (windows, times, z) with `windows` a list of (t_start_s, t_stop_s, pvalue,
    direction) and `z` the estimate/stderror trace the permutation test ran on. A missing
    folder or file means the Julia run has not written that coefficient yet and is a hard
    error.
    """
    coef_dir = os.path.join(run_dir, "coefficients", coefficient_folder)
    if not os.path.isdir(coef_dir):
        raise FileNotFoundError(
            f"[{channel}] coefficient folder '{coefficient_folder}' ({label}) is missing "
            f"under {os.path.join(run_dir, 'coefficients')} - the Julia run has not "
            "written it yet."
        )

    cluster_csv = os.path.join(coef_dir, "temporal_clusters.csv")
    npz_path = os.path.join(coef_dir, "cluster_results.npz")
    for path in (cluster_csv, npz_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[{channel}] {label}: {os.path.basename(path)} missing in {coef_dir} - "
                "the Julia run is still writing this coefficient."
            )

    payload = np.load(npz_path)
    times = payload["times"].astype(float).ravel()
    # z_vals is (n_channels x n_time); these runs are single-channel by construction.
    z_vals = np.atleast_2d(payload["z_vals"].astype(float))
    if z_vals.shape[0] != 1:
        raise ValueError(
            f"[{channel}] {label}: expected a single-channel z-map, got shape {z_vals.shape}."
        )
    z = z_vals[0]
    if z.shape[0] != times.shape[0]:
        raise ValueError(
            f"[{channel}] {label}: z-trace has {z.shape[0]} samples but the time axis has "
            f"{times.shape[0]}."
        )

    table = pd.read_csv(cluster_csv)
    significant = table[table["pvalue"] < alpha]
    windows = [
        (float(r.t_start_s), float(r.t_stop_s), float(r.pvalue), str(r.direction))
        for r in significant.itertuples()
    ]
    return windows, times, z


# %%
# ---------------------------------------------------------------------------
# ERP recomputation, mirroring load_subject_session / plot_erp_helper in
# cluster_permutation_helpers.jl
# ---------------------------------------------------------------------------
def parse_baseline_tag(tag):
    """
    Turn a run's baseline_tag into {modality: (start, stop)}.

    Two shapes are produced by the Julia script:
      "elec-0.5_-0.4-mecha-0.53_-0.43"  -> per-modality windows
      "env-0.53_-0.43" / "auto-0.1_-0.0" -> one window shared by both modalities
    """
    per_modality = {}
    if tag.startswith("elec") and "-mecha" in tag:
        elec_part, mecha_part = tag.split("-mecha", 1)
        per_modality["elec"] = _parse_window(elec_part[len("elec"):])
        per_modality["mecha"] = _parse_window(mecha_part)
    else:
        for prefix in ("env", "auto"):
            if tag.startswith(prefix):
                window = _parse_window(tag[len(prefix):])
                per_modality = {"elec": window, "mecha": window}
                break
    if set(per_modality) != {"elec", "mecha"}:
        raise ValueError(f"Cannot parse baseline tag '{tag}' into per-modality windows.")
    return per_modality


def _parse_window(text):
    """Parse '-0.5_-0.4' (start_stop, both possibly negative) into (start, stop)."""
    start_text, stop_text = text.split("_", 1)
    start, stop = float(start_text), float(stop_text)
    if start >= stop:
        raise ValueError(f"Baseline window '{text}' is not increasing: ({start}, {stop}).")
    return start, stop


def baseline_start_token(baseline_windows):
    """
    Filename token carrying the baseline start(s), e.g. "bl-0.1" when both modalities
    share a start, or "bl-elec-0.5-mecha-0.53" when they differ.
    """
    starts = {modality: window[0] for modality, window in baseline_windows.items()}
    if len(set(starts.values())) == 1:
        return f"bl{next(iter(starts.values())):g}"
    return "bl" + "".join(f"-{modality}{start:g}"
                          for modality, start in sorted(starts.items()))


def signal_space(env):
    """
    (amplitude unit, filename token) for the run's signal space.

    A surface-Laplacian run stores `<stem>_csd` epochs, whose values are in uV/m^2 rather
    than uV. Labelling those axes "uV" would be wrong by three orders of magnitude, and an
    unmarked filename would make a CSD figure indistinguishable from an average-reference one.
    """
    if env["epoch_stem_resolved"].endswith("_csd"):
        return "µV/m²", "_csd"
    return "µV", ""


def parse_excluded_pairs(text):
    """'023/mecha;036/mecha' or '<none>' -> set of (subject, modality) tuples."""
    if text.strip() in ("", "<none>"):
        return set()
    pairs = set()
    for token in text.split(";"):
        subject, modality = token.strip().split("/")
        pairs.add((subject, modality))
    return pairs


def load_channel_trials(datadir, subject, modality, channel, epoch_stem, metadata_stem,
                        baseline_window, time_start, time_stop):
    """
    Baseline-corrected, cleaned trials of one channel for one subject x session.

    Returns (data, times, responses) with `data` trial x time in microvolts, or None when
    the file pair does not exist (the Julia loader skips those blocks the same way).
    """
    base = os.path.join(datadir, "derivatives", "eeg-preproc", f"sub-{subject}", f"ses-{modality}")
    stem = f"sub-{subject}_ses-{modality}_task-NT_{FILE_PREFIX_PARAMS}"
    eeg_file = os.path.join(base, f"{stem}_{epoch_stem}.h5")
    meta_file = os.path.join(base, f"{stem}_{metadata_stem}.tsv")
    if not (os.path.isfile(eeg_file) and os.path.isfile(meta_file)):
        return None

    with h5py.File(eeg_file, "r") as handle:
        if channel not in handle:
            raise KeyError(f"Channel '{channel}' not present in {eeg_file}")
        # HDF5.jl writes time x trial; h5py reads the transpose, so this is trial x time.
        data = np.asarray(handle[channel]["data"], dtype=float)
        times = np.asarray(handle[channel]["times"], dtype=float).ravel()

    meta = pd.read_csv(meta_file, sep="\t")
    if len(meta) != data.shape[0]:
        raise ValueError(
            "Row mismatch between metadata and EEG trials. "
            f"sub-{subject} ses-{modality} n_meta={len(meta)} n_data={data.shape[0]}"
        )

    # Baseline correction before any cropping, per trial, over the modality's window.
    b_start, b_stop = baseline_window
    b_mask = (times >= b_start) & (times <= b_stop)
    if not b_mask.any():
        raise ValueError(
            f"Baseline window ({b_start}, {b_stop}) not covered by the time axis of {eeg_file}."
        )
    data = data - data[:, b_mask].mean(axis=1, keepdims=True)

    # Same trial filter, in the same order, as the Julia loader.
    responses = meta["response_decoded"].astype(str)
    keep = (responses != "nan") & (meta["drop_epoch"] != True) & (meta["analysis_excluded"] != True)  # noqa: E712
    data = data[keep.to_numpy(), :]
    responses = responses[keep.to_numpy()].to_numpy()

    t_mask = (times >= time_start) & (times <= time_stop)
    if not t_mask.any():
        raise ValueError(
            f"Time window ({time_start}, {time_stop}) selects no samples in {eeg_file}."
        )
    return data[:, t_mask], times[t_mask], responses


def compute_erp(datadir, channel, env, run_dir):
    """
    Grand-average ERP of `channel`, split by modality x detection response and pooled over
    all subjects. Returns {(modality, response_level): {mean, sem, n}}.

    Modality comes from the session the trials were loaded from, not from a metadata
    column, so it cannot disagree with the file the data came out of.

    The average is unweighted over trials (not over subjects), matching `plot_erp_helper`
    in cluster_permutation_helpers.jl. Per-subject trial counts are cross-checked against
    the run's own subject_trial_counts.csv.
    """
    subjects = [s.strip() for s in env["subjects_resolved"].split(",") if s.strip()]
    excluded = parse_excluded_pairs(env["excluded_pairs_resolved"])
    baselines = parse_baseline_tag(env["baseline_tag_resolved"])
    epoch_stem = env["epoch_stem_resolved"]
    metadata_stem = env["metadata_stem_resolved"]
    time_start = float(env["time_start_resolved"])
    time_stop = float(env["time_stop_resolved"])
    modalities = [m.strip() for m in env["run_modalities_resolved"].split(",") if m.strip()]

    blocks, labels, trace_modalities, counts = [], [], [], []
    times_ref = None
    for subject in subjects:
        for modality in modalities:
            if (subject, modality) in excluded:
                continue
            loaded = load_channel_trials(
                datadir, subject, modality, channel, epoch_stem, metadata_stem,
                baselines[modality], time_start, time_stop,
            )
            if loaded is None:
                continue
            data, times, responses = loaded
            if times_ref is None:
                times_ref = times
            elif not np.array_equal(times_ref, times):
                raise ValueError(f"Time axis mismatch for sub-{subject} ses-{modality}.")
            blocks.append(data)
            labels.append(responses)
            trace_modalities.append(np.full(data.shape[0], modality, dtype=object))
            counts.append({"subject": f"sub-{subject}", "modality": modality,
                           "total_trials": data.shape[0]})

    if not blocks:
        raise ValueError(f"No EEG blocks loaded for channel {channel}.")

    # The trial set behind the ERP must be exactly the one the permutation test used.
    recomputed = pd.DataFrame(counts)
    reference = pd.read_csv(os.path.join(run_dir, "subject_trial_counts.csv"))
    merged = reference.merge(recomputed, on=["subject", "modality"], how="outer",
                             suffixes=("_run", "_recomputed"))
    mismatch = merged[merged["total_trials_run"] != merged["total_trials_recomputed"]]
    if len(mismatch):
        raise ValueError(
            f"[{channel}] recomputed trial counts differ from the run's "
            f"subject_trial_counts.csv:\n{mismatch.to_string(index=False)}"
        )

    data = np.concatenate(blocks, axis=0)
    responses = np.concatenate(labels, axis=0)
    trace_modality = np.concatenate(trace_modalities, axis=0)

    erp = {}
    for modality, level, _label, _color_key, _linestyle in ERP_TRACES:
        selection = data[(trace_modality == modality) & (responses == level), :]
        if selection.shape[0] == 0:
            raise ValueError(
                f"[{channel}] no trials for modality '{modality}' with "
                f"response_decoded == '{level}'."
            )
        erp[(modality, level)] = {
            "mean": selection.mean(axis=0),
            "sem": selection.std(axis=0, ddof=0) / np.sqrt(selection.shape[0]),
            "n": selection.shape[0],
        }
    return times_ref, erp


# %%
# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def draw_significance_strip(ax, panel_windows, coef_colors, data_low, data_high,
                            headroom=0.12):
    """
    Draw one significance row per coefficient in a band below `data_low`, and return the
    y-limits that fit both the data and the strip.

    The rows are sized relative to the data range so the strip keeps the same visual
    proportions whatever the y-unit of the panel above it.
    """
    span = data_high - data_low
    strip_top = data_low - 0.08 * span
    row_step = 0.055 * span

    for row, (label, windows, _z) in enumerate(panel_windows):
        y = strip_top - (row + 0.5) * row_step
        for t_start, t_stop, _pvalue, _direction in windows:
            ax.plot([t_start, t_stop], [y, y], color=coef_colors[label], linewidth=5,
                    solid_capstyle="butt", zorder=3)

    low = strip_top - len(panel_windows) * row_step - 0.05 * span
    return low, data_high + headroom * span


def draw_erp_panel(ax, letter, channel, times, panel, colors, coef_colors, threshold):
    """One channel panel: ERP +- SEM on top, coefficient significance rows below."""
    erp = panel["erp"]
    for modality, level, label, color_key, linestyle in ERP_TRACES:
        cell = erp[(modality, level)]
        mean, sem = cell["mean"], cell["sem"]
        # The band is kept faint: with n in the thousands the SEM is small, and a solid
        # band in the trace colour would fill in the dashes of the unperceived traces.
        ax.fill_between(times, mean - sem, mean + sem, color=colors[color_key], alpha=0.15,
                        linewidth=0, zorder=1)
        ax.plot(times, mean, color=colors[color_key], linewidth=1.8, linestyle=linestyle,
                dash_capstyle="butt", zorder=2, label=label)

    data_low = min(cell["mean"].min() - cell["sem"].max() for cell in erp.values())
    data_high = max(cell["mean"].max() + cell["sem"].max() for cell in erp.values())

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.axvline(-0.45, color="black", linewidth=1.0, linestyle="-.")

    # Extra headroom so the four-entry trace legend does not sit on top of the data.
    ax.set_ylim(*draw_significance_strip(ax, panel["windows"], coef_colors, data_low, data_high, headroom=0.38))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel(f"Amplitude ({panel.get('amplitude_unit', 'µV')})")
    ax.set_title(f"{letter}   {channel}", loc="left", fontweight="bold")
    sns.despine(ax=ax, trim=False)
    return data_low, data_high


def draw_zvalue_panel(ax, letter, channel, times, panel, colors, coef_colors, threshold):
    """
    Same layout as the ERP panel, but the traces are the tested statistic: one z-trace per
    coefficient, in that coefficient's colour, with the cluster-forming threshold marked.
    """
    for label, _windows, z in panel["windows"]:
        ax.plot(times, z, color=coef_colors[label], linewidth=1.5, label=label)

    stacked = np.vstack([z for _label, _windows, z in panel["windows"]])
    data_low, data_high = float(stacked.min()), float(stacked.max())

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    # The cluster-forming threshold: only excursions beyond it can enter a cluster.
    for sign in (1, -1):
        ax.axhline(sign * threshold, color="black", linewidth=1.0, linestyle=":",
                   alpha=0.6)

    ax.set_ylim(*draw_significance_strip(ax, panel["windows"], coef_colors,
                                         data_low, data_high))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("z-value")
    ax.set_title(f"{letter}   {channel}", loc="left", fontweight="bold")
    sns.despine(ax=ax, trim=False)
    return data_low, data_high


def shared_y_ticks(data_ranges, min_ticks=3, max_bins=9):
    """
    Tick positions that fall inside every panel's data range, so all panels end up with
    the identical y tick labels even though their y-limits differ.

    Ticks are chosen over the intersection of the panels' data ranges; anything outside it
    would be dropped by one panel and not the other, which is exactly the mismatch this
    avoids. The bin count is raised until enough ticks survive.
    """
    low = max(r[0] for r in data_ranges)
    high = min(r[1] for r in data_ranges)
    if not low < high:
        raise ValueError(
            f"Panel data ranges do not overlap (intersection {low} to {high}); "
            "a shared y tick set is not meaningful."
        )

    for nbins in range(3, max_bins + 1):
        candidates = MaxNLocator(nbins=nbins).tick_values(low, high)
        ticks = [t for t in candidates if low <= t <= high]
        if len(ticks) >= min_ticks:
            return ticks
    return ticks


def save_panel_figure(panels, draw_fn, colors, coef_colors, threshold, figuredir, stem,
                      first_axis_legend=False):
    """Build, lay out and save one two-panel figure; returns the paths written."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9), sharex=True)
    axes = np.atleast_1d(axes)
    data_ranges = [
        draw_fn(ax, letter, panel["channel"], panel["times"], panel, colors, coef_colors,
                threshold)
        for ax, panel, letter in zip(axes, panels, "abcdef")
    ]

    # Same y tick labels on every panel; the y-limits stay panel-specific.
    ticks = shared_y_ticks(data_ranges)
    for ax in axes:
        ax.set_yticks(ticks)

    axes[-1].set_xlabel("Time (s)")
    if first_axis_legend:
        axes[0].legend(loc="upper right", frameon=False, fontsize=12, ncol=2,
                       columnspacing=1.4, handlelength=2.6)

    coef_handles = [Line2D([0], [0], color=coef_colors[label], linewidth=5, label=label)
                    for label, _ in COEFFICIENTS]
    legend = fig.legend(handles=coef_handles, loc="lower center", ncol=3, frameon=False,
                        fontsize=14,
                        title=f"Significant cluster (cluster-mass permutation, p < {ALPHA})",
                        title_fontsize=14, bbox_to_anchor=(0.5, 0.0))

    # Measure the rendered legend and reserve exactly that much at the bottom, so adding
    # or removing coefficients cannot make it collide with the x-axis label.
    fig.canvas.draw()
    legend_fraction = legend.get_window_extent().height / fig.bbox.height
    fig.tight_layout(rect=(0.0, min(legend_fraction + 0.04, 0.5), 1.0, 1.0))

    written = []
    for extension in ("pdf", "png"):
        output_file = os.path.join(figuredir, f"{stem}.{extension}")
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        written.append(output_file)
    plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for channel in PANEL_CHANNELS:
        parser.add_argument(f"--run-{channel.lower()}", default=None,
                            help=f"Run id for channel {channel} (overrides RUN_IDS).")
    args = parser.parse_args()

    resdir = os.getenv("RESDIR")
    if not resdir:
        raise EnvironmentError("RESDIR is not set.")
    datadir = os.getenv("DATADIR")
    if not datadir:
        raise EnvironmentError("DATADIR is not set.")

    runs_root = os.path.join(resdir, "Statistics", "eeg", "cluster_permutation",
                             "cluster_mass_temporal", "runs")

    run_ids = dict(RUN_IDS)
    for channel in PANEL_CHANNELS:
        override = getattr(args, f"run_{channel.lower()}")
        if override:
            run_ids[channel] = override

    # ---- resolve runs; the first panel channel anchors the configuration ----
    anchor_channel = PANEL_CHANNELS[0]
    anchor_signature = None
    if run_ids[anchor_channel] == "latest":
        # The anchor cannot match itself, so anchor on the first explicit run instead.
        explicit = [c for c in PANEL_CHANNELS if run_ids[c] != "latest"]
        if not explicit:
            raise ValueError("At least one channel needs an explicit run id in RUN_IDS.")
        anchor_channel = explicit[0]
    anchor_dir = resolve_run_dir(runs_root, anchor_channel, run_ids[anchor_channel], None)
    anchor_env = read_startup_environment(anchor_dir)
    anchor_signature = config_signature(anchor_env)
    print(f"[{anchor_channel}] anchor run: {os.path.basename(anchor_dir)}")

    panels = []
    for channel in PANEL_CHANNELS:
        if channel == anchor_channel:
            run_dir, env = anchor_dir, anchor_env
        else:
            run_dir = resolve_run_dir(runs_root, channel, run_ids[channel], anchor_signature)
            env = read_startup_environment(run_dir)
            differing = [k for k in CONFIG_KEYS
                         if config_signature(env)[k] != anchor_signature[k]]
            if differing:
                raise ValueError(
                    f"Run configuration mismatch between {anchor_channel} "
                    f"({os.path.basename(anchor_dir)}) and {channel} "
                    f"({os.path.basename(run_dir)}); differing keys: {', '.join(differing)}.\n"
                    + "\n".join(
                        f"  {k}: {anchor_channel}={anchor_signature[k]!r}  {channel}={config_signature(env)[k]!r}"
                        for k in differing
                    )
                )
            print(f"[{channel}] run: {os.path.basename(run_dir)}")
        panels.append({"channel": channel, "run_dir": run_dir, "env": env})

    print(f"\nShared time window: {anchor_env['time_start_resolved']} to "
          f"{anchor_env['time_stop_resolved']} s   "
          f"baseline: {anchor_env['baseline_tag_resolved']}   "
          f"permutations: {anchor_env['n_permutations_resolved']}")

    # ---- cluster windows ----
    cluster_times_ref = None
    for panel in panels:
        panel_windows = []
        print(f"\n[{panel['channel']}] significant clusters (p < {ALPHA}):")
        for label, folder in COEFFICIENTS:
            windows, times, z = load_significant_windows(
                panel["run_dir"], panel["channel"], folder, label
            )
            if cluster_times_ref is None:
                cluster_times_ref = times
            elif not np.allclose(cluster_times_ref, times):
                raise ValueError(
                    f"[{panel['channel']}] time axis of '{folder}' differs from the "
                    "other coefficients."
                )
            if windows:
                for t_start, t_stop, pvalue, direction in windows:
                    print(f"  {label:<32s} {direction} {t_start * 1000:7.1f} to "
                          f"{t_stop * 1000:7.1f} ms   p = {pvalue:.4f}")
            else:
                print(f"  {label:<32s} none")
            panel_windows.append((label, windows, z))
        panel["windows"] = panel_windows

    # ---- ERPs ----
    for panel in panels:
        print(f"\n[{panel['channel']}] recomputing ERP from epoch files ...")
        times, erp = compute_erp(datadir, panel["channel"], panel["env"], panel["run_dir"])
        if not np.allclose(times, cluster_times_ref):
            raise ValueError(
                f"[{panel['channel']}] ERP time axis does not match the permutation "
                f"time axis ({len(times)} vs {len(cluster_times_ref)} samples)."
            )
        panel["times"], panel["erp"] = times, erp
        for modality, level, label, _color_key, _linestyle in ERP_TRACES:
            print(f"  trial count {label:<28s} n = {erp[(modality, level)]['n']}")

    # ---- figure ----
    colors, _dpi = figure_style()
    if len(COEFFICIENT_COLORS) < len(COEFFICIENTS):
        raise ValueError("COEFFICIENT_COLORS has fewer entries than COEFFICIENTS.")
    coef_colors = {label: color
                   for (label, _), color in zip(COEFFICIENTS, COEFFICIENT_COLORS)}

    threshold = float(anchor_env["clusterforming_threshold_resolved"])

    figuredir = os.path.join(resdir, "Figures", "Main", "eeg_results", "cluster_permutation")
    os.makedirs(figuredir, exist_ok=True)
    amplitude_unit, csd_token = signal_space(anchor_env)
    for panel in panels:
        panel["amplitude_unit"] = amplitude_unit
    baseline_token = baseline_start_token(parse_baseline_tag(anchor_env["baseline_tag_resolved"]))
    channel_token = "_".join(PANEL_CHANNELS)

    # Two views of the same runs: the ERP the effects sit on, and the statistic the
    # cluster test actually ran on. Both carry the identical significance strip.
    figures = [
        ("temporal_clusters_erp_", draw_erp_panel, True),
        ("temporal_clusters_zvalues_", draw_zvalue_panel, False),
    ]
    for prefix, draw_fn, first_axis_legend in figures:
        stem = f"{prefix}{channel_token}_{baseline_token}{csd_token}"
        for output_file in save_panel_figure(panels, draw_fn, colors, coef_colors,
                                             threshold, figuredir, stem,
                                             first_axis_legend=first_axis_legend):
            print(f"\nSaved {output_file}")


if __name__ == "__main__":
    main()

# %%
