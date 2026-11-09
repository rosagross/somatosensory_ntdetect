# %%
"""
Topomap panel of cluster-permutation results for the joint mecha+elec model,
with topomaps placed on a shared timeline.

One row per coefficient of a joint run that models both modalities together.
Each row names its own run ID (runs may be split up per coefficient), but all
runs must share the same baseline. Each row is plotted at the latencies of the
peak |z| of every significant cluster in its run, so rows may have different
numbers of topomaps. A time axis at the bottom of the figure is shared by all
rows; every topomap sits above its latency on that axis. When latencies in a row
are too close for their topomaps to fit side by side, the topomaps are shifted
sideways just enough to avoid overlap, and a thin leader line connects each one
to a tick at its exact latency.

Rows from bottom (closest to the timeline) to top: detection response,
modality, intensity level, modality x detection response.

Latencies are computed from the loaded cluster results: the MAX_PEAKS clusters
with the largest peak |z| per row, printed per row. Topomaps at a cluster peak
have a bold title; topomaps at other latencies (extra or fixed) have a regular one. A row whose coefficient has no significant cluster
needs fixed latencies in PANEL_ROWS, otherwise the script raises; fixed latencies
on a row that does have clusters also raise.
"""

import os
import re
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mne
import numpy as np

helper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, helper_dir)
WDDIR = os.getenv("WDDIR")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

# %%
# ---------------------------------------------------------------------------
# Configuration: one row per coefficient folder rendered top to bottom. Each
# row names the run it is read from; all runs must share the same baseline.
# ---------------------------------------------------------------------------
_BL = "mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4"
RUN_DETECTION = f"20261020-205919__{_BL}"
RUN_INTENSITY = f"20261028-122831__{_BL}__coef-intensity_level_centered"
RUN_MODALITY = f"20261028-100155__{_BL}__coef-modality"
RUN_MODALITY_X_DETECTION = f"20261028-144658__{_BL}__coef-modality_response"

# Most topomaps per row that are drawn at cluster peaks: the clusters with the
# largest peak |z| are kept, then shown in time order.
MAX_PEAKS = 5

# (row label, run ID, coefficient folder, fixed latencies in seconds or None,
#  extra non-peak latencies in seconds or None). The numeric prefix of the coefficient folder depends on the model formula and
# on the coefficient selection of the run, so it is part of the configuration
# rather than derived. Latencies are the cluster peaks found in the loaded
# results (None). Fixed latencies are only for a coefficient without any
# significant cluster; if the run has a cluster, the script raises. Extra
# latencies are drawn in addition to the peaks and are marked as non-peaks.
#
# Rows are listed top to bottom; the last row sits directly above the timeline.
PANEL_ROWS = [
    ("Modality x Detection Response", RUN_MODALITY_X_DETECTION,
     "01__modality_Mechanical_response_decoded_Yes", None, None),
    # No significant cluster in the recomputed run, so there is no peak to plot.
    ("Intensity Level", RUN_INTENSITY, "01__intensity_level_centered",
     [0.050, 0.060, 0.070], None),
    ("Modality", RUN_MODALITY, "01__modality_Mechanical", None, None),
    ("Detection Response", RUN_DETECTION, "01__response_decoded_Yes", None,
     [0.0, 0.050]),
]

# The baseline tag is part of the run ID ("...__bl-<tag>__f-..."). Rows from
# runs with different baselines are not comparable, so refuse to mix them.
_baseline_by_run = {}
for _label, _run_id, _folder, _fixed, _extra in PANEL_ROWS:
    _match = re.search(r"__bl-(.+?)__f-", _run_id)
    if _match is None:
        raise ValueError(f"{_label}: no baseline tag (__bl-...__f-) in run ID {_run_id}")
    _baseline_by_run[_run_id] = _match.group(1)
if len(set(_baseline_by_run.values())) != 1:
    raise ValueError(
        "PANEL_ROWS mix runs with different baselines: "
        + "; ".join(f"{b} <- {r}" for r, b in _baseline_by_run.items())
    )
BASELINE_TAG = next(iter(_baseline_by_run.values()))
print(f"All {len(_baseline_by_run)} run(s) share baseline {BASELINE_TAG}")

# Shared colour scale for all topomaps so rows stay comparable.
VLIM = (-6, 6)
mask_params = dict(
    marker="o",
    markerfacecolor="none",
    markeredgecolor="black",
    linewidth=0,
)

# %%
RESDIR = os.getenv("RESDIR")
if not RESDIR:
    raise EnvironmentError("RESDIR is not set.")

datadir = os.getenv("DATADIR")
if not datadir:
    raise EnvironmentError("DATADIR is not set.")

run_dir = os.path.join(RESDIR, "Statistics", "eeg", "cluster_permutation", "cluster_mass", "runs")

# Subject 13 mechanical epochs only serve as a sensor-position template; no
# baseline is applied because only the montage in `info` is used.
template_epoch_file = os.path.join(
    datadir,
    "derivatives",
    "eeg-preproc",
    "sub-013",
    "ses-mecha",
    "sub-013_ses-mecha_task-NT_epochs.fif",
)
epochs_template = mne.read_epochs(template_epoch_file, preload=False, verbose="ERROR")
info_cache = {}


def get_info_for_channels(channels):
    channel_key = tuple(channels)
    if channel_key in info_cache:
        return info_cache[channel_key]

    pick_idx = mne.pick_channels(epochs_template.ch_names, include=channels, ordered=True)
    if len(pick_idx) != len(channels):
        raise ValueError(
            "Mismatch between channels.txt and subject 13 mechanical epochs channels: "
            f"channels.txt={len(channels)}, found_in_epochs={len(pick_idx)}"
        )
    info = mne.pick_info(epochs_template.info, pick_idx, copy=True)
    info_cache[channel_key] = info
    return info


def draw_topomap_on_axis(ax, z_vals, cluster_display, info, time_idx):
    ax.clear()
    z_map = z_vals[:, time_idx]
    sig_mask = ~np.isnan(cluster_display[:, time_idx])
    mask = sig_mask[:, np.newaxis]

    im, _ = mne.viz.plot_topomap(
        z_map,
        info,
        cmap="RdBu_r",
        vlim=VLIM,
        contours=0,
        mask=mask,
        mask_params=mask_params,
        axes=ax,
        show=False,
    )
    return im


def inspect_cluster_summary(z_vals, cluster_display, times, channels):
    """Summarize significant clusters from cluster_display."""
    # Keep only significant cluster IDs (NaN = not significant, 0 = no cluster).
    valid_ids = cluster_display[~np.isnan(cluster_display)]
    valid_ids = valid_ids[~np.isclose(valid_ids, 0.0)]
    if valid_ids.size == 0:
        return []

    cluster_summaries = []
    for cluster_id in np.unique(valid_ids):
        cluster_mask = np.isclose(cluster_display, cluster_id)
        if not np.any(cluster_mask):
            continue

        time_indices = np.where(cluster_mask.any(axis=0))[0]
        channel_indices = np.where(cluster_mask.any(axis=1))[0]
        if time_indices.size == 0 or channel_indices.size == 0:
            continue

        cluster_coords = np.argwhere(cluster_mask)
        peak_coord_idx = int(np.argmax(np.abs(z_vals[cluster_mask])))
        peak_ch_idx, peak_t_idx = cluster_coords[peak_coord_idx]
        peak_signed_z = float(z_vals[peak_ch_idx, peak_t_idx])

        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id) if float(cluster_id).is_integer() else float(cluster_id),
                "start_idx": int(time_indices[0]),
                "start_time_s": float(times[time_indices[0]]),
                "end_time_s": float(times[time_indices[-1]]),
                "channels": [channels[i] for i in channel_indices],
                "n_channels": int(channel_indices.size),
                "peak_abs_z": float(np.abs(peak_signed_z)),
                "peak_signed_z": peak_signed_z,
                "peak_time_s": float(times[peak_t_idx]),
                "peak_t_idx": int(peak_t_idx),
                "peak_channel": channels[int(peak_ch_idx)],
            }
        )

    return sorted(cluster_summaries, key=lambda x: (x["start_idx"], x["cluster_id"]))


def spread_positions(desired_x, width, gap, x_min, x_max):
    """Shift box centers sideways so boxes of `width` do not overlap.

    Boxes that would overlap are merged into a group, laid out with a fixed
    spacing of `width + gap`, and the group is centered on the mean of its
    members' desired centers. Groups are then kept inside [x_min, x_max].
    Returns the centers in the order of `desired_x`.
    """
    desired_x = np.asarray(desired_x, dtype=float)
    step = width + gap
    n_boxes = desired_x.size
    if n_boxes * width + (n_boxes - 1) * gap > x_max - x_min:
        raise ValueError(
            f"{n_boxes} boxes of width {width:.4f} do not fit between {x_min:.4f} and {x_max:.4f}."
        )

    order = np.argsort(desired_x)
    sorted_x = desired_x[order]
    # Each group is a list of indices into sorted_x.
    groups = [[i] for i in range(n_boxes)]

    def group_start(group):
        # Left-most center of a group, clamped so the whole group stays inside the limits.
        center = sorted_x[group].mean()
        start = center - (len(group) - 1) * step / 2
        start = max(start, x_min + width / 2)
        start = min(start, x_max - width / 2 - (len(group) - 1) * step)
        return start

    merged = True
    while merged:
        merged = False
        for g in range(len(groups) - 1):
            left_end = group_start(groups[g]) + (len(groups[g]) - 1) * step
            right_start = group_start(groups[g + 1])
            if right_start - left_end < step:
                groups[g] = groups[g] + groups[g + 1]
                del groups[g + 1]
                merged = True
                break

    sorted_centers = np.empty(n_boxes)
    for group in groups:
        start = group_start(group)
        for k, i in enumerate(group):
            sorted_centers[i] = start + k * step

    centers = np.empty(n_boxes)
    centers[order] = sorted_centers
    return centers


# %%
# ---------------------------------------------------------------------------
# Load one cell per panel row. Missing results are a hard error: an empty
# coefficient folder means the Julia job has not finished writing yet.
# ---------------------------------------------------------------------------
panel_cells = []

for row_label, row_run_id, coefficient_folder, fixed_times, extra_times in PANEL_ROWS:
    coeff_dir = os.path.join(run_dir, row_run_id, "coefficients", coefficient_folder)
    cluster_file = os.path.join(coeff_dir, "cluster_results.npz")
    channels_file = os.path.join(coeff_dir, "channels.txt")

    if not os.path.isfile(cluster_file):
        raise FileNotFoundError(f"cluster_results.npz not found for {row_label}: {cluster_file}")
    if not os.path.isfile(channels_file):
        raise FileNotFoundError(f"channels.txt not found for {row_label}: {channels_file}")

    data = np.load(cluster_file)
    z_vals = data["z_vals"]
    cluster_display = data["cluster_display"]
    times = data["times"]

    with open(channels_file) as f:
        channels = [line.strip() for line in f]

    info = get_info_for_channels(channels)

    clusters = inspect_cluster_summary(z_vals, cluster_display, times, channels)
    print(f"\n[{row_label}] run={row_run_id}")
    print(f"[{row_label}] Total significant clusters found: {len(clusters)}")
    for idx, cl in enumerate(clusters, start=1):
        print(
            f"[{row_label}] Cluster {idx} (id={cl['cluster_id']}): "
            f"start={cl['start_time_s']*1000:.1f} ms, end={cl['end_time_s']*1000:.1f} ms, "
            f"channels={cl['n_channels']}, peak |z|={cl['peak_abs_z']:.3f} "
            f"at {cl['peak_time_s']*1000:.1f} ms on {cl['peak_channel']}"
        )

    tolerance = np.median(np.diff(times)) / 2

    def latencies_to_indices(latencies, what):
        """Sample indices of latencies (s); a latency off the time axis is an error."""
        latencies = np.asarray(latencies, dtype=float)
        idx = np.array([int(np.argmin(np.abs(times - t))) for t in latencies], dtype=int)
        off = np.abs(times[idx] - latencies) > tolerance
        if off.any():
            raise ValueError(
                f"{row_label}: {what} {np.round(latencies[off] * 1000, 1)} ms are outside the time "
                f"axis or do not land on a sample ({times[0]*1000:.0f} to {times[-1]*1000:.0f} ms, "
                f"step {np.median(np.diff(times))*1000:.1f} ms)."
            )
        return idx

    if fixed_times is None:
        if not clusters:
            raise ValueError(
                f"{row_label}: no significant cluster in {coeff_dir}, so there is no peak to "
                "plot. Give fixed latencies in PANEL_ROWS or remove the row."
            )
        # Keep the MAX_PEAKS clusters with the largest peak |z|. Clusters that peak
        # on the same sample share a topomap, which shows all of them anyway.
        strongest = sorted(clusters, key=lambda cl: cl["peak_abs_z"], reverse=True)[:MAX_PEAKS]
        peak_indices = np.unique([cl["peak_t_idx"] for cl in strongest])
        dropped = [cl for cl in clusters if cl not in strongest]
        for cl in dropped:
            print(f"[{row_label}] Not plotted (weaker): cluster {cl['cluster_id']:+d}, "
                  f"peak |z|={cl['peak_abs_z']:.2f} at {cl['peak_time_s']*1000:.1f} ms")
    else:
        if clusters:
            raise ValueError(
                f"{row_label}: fixed latencies given, but the run has {len(clusters)} significant "
                "cluster(s). Set the latencies to None to plot the cluster peaks."
            )
        peak_indices = np.array([], dtype=int)
        extra_times = list(fixed_times) + list(extra_times or [])

    # Extra latencies are non-peaks, unless one lands on a peak sample, which stays a peak.
    extra_indices = (latencies_to_indices(extra_times, "extra latencies")
                     if extra_times else np.array([], dtype=int))
    grid_time_indices = np.unique(np.concatenate([peak_indices, extra_indices]).astype(int))
    is_peak = np.isin(grid_time_indices, peak_indices)
    print(f"[{row_label}] Plotting {len(grid_time_indices)} latencies (ms), peaks marked *: "
          + ", ".join(f"{times[i]*1000:.0f}{'*' if pk else ''}"
                      for i, pk in zip(grid_time_indices, is_peak)))

    panel_cells.append(
        {
            "row_label": row_label,
            "z_vals": z_vals,
            "cluster_display": cluster_display,
            "info": info,
            "times": times,
            "grid_time_indices": grid_time_indices,
            "is_peak": is_peak,
        }
    )

# %%
# ---------------------------------------------------------------------------
# Panel figure: one row per coefficient above a shared timeline. Axes are placed
# by hand in figure coordinates so every topomap can sit above its latency.
# Sizes below are in inches and converted to figure fractions.
# ---------------------------------------------------------------------------
n_rows = len(panel_cells)
map_size = 1.2          # topomap width = height
map_gap = 0.08          # minimum horizontal gap between topomaps in a row
plot_left = 0.14
plot_right = 0.90
# The figure widens when the fullest row needs more room than 14 in gives it.
max_maps_in_row = max(len(cell["grid_time_indices"]) for cell in panel_cells)
fig_w = max(14.0, max_maps_in_row * (map_size + map_gap) / (plot_right - plot_left))
tick_to_map = 0.35      # vertical room for the leader lines
baseline_pad = 0.15     # row baseline above the bottom of its band
title_room = 0.35       # latency title above each topomap
band_h = baseline_pad + tick_to_map + map_size + title_room
timeline_y_in = 0.6     # timeline height above the figure bottom (room for tick labels)
top_pad = 0.1
fig_h = timeline_y_in + n_rows * band_h + top_pad

timeline_y = timeline_y_in / fig_h
map_w = map_size / fig_w
map_h = map_size / fig_h
gap_w = map_gap / fig_w

# Timeline range: all plotted latencies plus a margin, in ms.
all_times_ms = np.concatenate(
    [cell["times"][cell["grid_time_indices"]] * 1000 for cell in panel_cells]
)
time_pad_ms = 40.0
xlim_ms = (all_times_ms.min() - time_pad_ms, all_times_ms.max() + time_pad_ms)


def time_to_fig_x(t_ms):
    return plot_left + (np.asarray(t_ms) - xlim_ms[0]) / (xlim_ms[1] - xlim_ms[0]) * (plot_right - plot_left)


fig = plt.figure(figsize=(fig_w, fig_h))

# Timeline: a zero-height axis whose bottom spine is the time axis.
ax_time = fig.add_axes([plot_left, timeline_y, plot_right - plot_left, 1e-6])
ax_time.set_xlim(xlim_ms)
ax_time.set_yticks([])
for side in ("left", "right", "top"):
    ax_time.spines[side].set_visible(False)
ax_time.tick_params(axis="x", labelsize=13)
ax_time.set_xlabel("Time (ms)", fontsize=14)

grey = "0.6"
top_y = timeline_y + n_rows * band_h / fig_h

# Stimulus onset guide through all rows, drawn first so the topomaps cover it.
if xlim_ms[0] <= 0 <= xlim_ms[1]:
    fig.add_artist(Line2D(
        [time_to_fig_x(0.0)] * 2, [timeline_y, top_y],
        transform=fig.transFigure, color=grey, linewidth=0.6, linestyle="--", zorder=0,
    ))

last_im = None
row_map_y = []
for row_idx, cell in enumerate(panel_cells):
    # Row 0 is the top band; the last row sits directly above the timeline.
    band_y0 = timeline_y + (n_rows - 1 - row_idx) * band_h / fig_h
    baseline_y = band_y0 + baseline_pad / fig_h
    map_y0 = baseline_y + tick_to_map / fig_h
    row_map_y.append(map_y0 + map_h / 2)

    fig.add_artist(Line2D(
        [plot_left, plot_right], [baseline_y, baseline_y],
        transform=fig.transFigure, color=grey, linewidth=0.5,
    ))

    row_times_ms = cell["times"][cell["grid_time_indices"]] * 1000
    exact_x = time_to_fig_x(row_times_ms)
    map_x = spread_positions(exact_x, map_w, gap_w, plot_left, plot_right)

    for t_ms, x_exact, x_map, time_idx, peak in zip(
        row_times_ms, exact_x, map_x, cell["grid_time_indices"], cell["is_peak"]
    ):
        # Tick at the exact latency on the row baseline.
        fig.add_artist(Line2D(
            [x_exact, x_exact], [baseline_y - 0.04 / fig_h, baseline_y + 0.04 / fig_h],
            transform=fig.transFigure, color="black", linewidth=0.8,
        ))
        # Leader line from the tick to the bottom center of the (possibly shifted) topomap.
        fig.add_artist(Line2D(
            [x_exact, x_map], [baseline_y, map_y0],
            transform=fig.transFigure, color="black", linewidth=0.5,
        ))
        ax = fig.add_axes([x_map - map_w / 2, map_y0, map_w, map_h])
        last_im = draw_topomap_on_axis(ax, cell["z_vals"], cell["cluster_display"], cell["info"], int(time_idx))
        # Bold title = cluster peak, regular title = latency chosen without a peak.
        ax.set_title(f"{int(round(t_ms))} ms", fontsize=13, pad=4,
                     fontweight="bold" if peak else "normal")

# Row labels to the left of each row, level with the topomaps.
for row_idx, cell in enumerate(panel_cells):
    fig.text(
        plot_left - 0.012,
        row_map_y[row_idx],
        cell["row_label"],
        ha="right",
        va="center",
        fontsize=16,
    )

# Vertical colorbar to the right of the rows (the bottom holds the timeline).
cbar_bottom = row_map_y[-1] - map_h / 2
cbar_top = row_map_y[0] + map_h / 2
cbar_ax = fig.add_axes([plot_right + 0.02, cbar_bottom, 0.012, cbar_top - cbar_bottom])
cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="vertical")
cbar.set_label("z-value", fontsize=14)
cbar.ax.tick_params(labelsize=13)

figuredir = os.path.join(RESDIR, "Figures", "Main", "eeg_results", "cluster_permutation")
os.makedirs(figuredir, exist_ok=True)
output_file = os.path.join(figuredir, "cluster_topomap_panel__joint__mecha-elec__peak_latencies.pdf")
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved combined topomap panel: {output_file}")

# %%
