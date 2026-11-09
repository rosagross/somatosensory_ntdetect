"""
File: plot_source_zmap_manuscript_panel.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-09-09
Description:
Manuscript panel of the source-reconstructed cluster-permutation z-maps of the
selected coefficients (detection response, modality, modality x detection
response), rendered as brain snapshots at the latencies of their sensor clusters.

The coefficients may come from different run directories (runs split up per
coefficient); all runs must share the baseline. Latencies are computed from the
cluster results, as in source_reconstruction_zvals.py and
plot_cluster_results_topoplot_panel_joint_peaks.py: the peak |z| of the
`max_peaks` strongest significant clusters per row, plus optional extra latencies.
Titles at cluster peaks are bold, titles at other latencies are regular.

This script only plots. The group source estimates are produced and cached by
source_reconstruction_zvals.py, one stc folder per run; here they are read back from
<figure root>/<run ID>/stc/<coefficient>__inv-<inverse_name>-mean-{lh,rh}.stc, with
inverse_name e.g. "mne_loose-0_depth-None_method-MNE" or "lcmv_wnunit-noise-gain_reg-0.05".
A missing cache is a hard error. The inverse settings are set in the configuration
block (INVERSE_METHOD "mne" or "lcmv", MNE_METHOD "dSPM" or "MNE") and have to match
the ones the cache was written with; the name is also part of the output filename.
"""

# %%
import os
import re
import sys
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_brain_colorbar
import numpy as np
from plotting_style import figure_style
colors, dpi = figure_style()

# %%
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WDDIR = os.getenv("WDDIR")
if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

_BL = "mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4"
RUN_DETECTION = f"20261020-205919__{_BL}"
RUN_MODALITY = f"20261028-100155__{_BL}__coef-modality"
RUN_MODALITY_X_DETECTION = f"20261028-144658__{_BL}__coef-modality_response"

# (row label, run ID, coefficient folder, max_peaks, extra non-peak latencies in s or None).
# Same selection as SOURCE_ROWS in source_reconstruction_zvals.py, which reconstructed
# these coefficients. The latencies are the peak |z| of the `max_peaks` strongest
# significant clusters plus the extra latencies; nothing is typed in by hand.
PANEL_ROWS = [
    ("detection response", RUN_DETECTION, "01__response_decoded_Yes", 5, [0.0, 0.050]),
    ("modality", RUN_MODALITY, "01__modality_Mechanical", 5, None),
    ("modality x detection response", RUN_MODALITY_X_DETECTION,
     "01__modality_Mechanical_response_decoded_Yes", 1, None),
]

# All runs must share one baseline; rows from different baselines are not comparable.
_baselines = {}
for _label, _run_id, _folder, _n, _extra in PANEL_ROWS:
    _m = re.search(r"__bl-(.+?)__f-", _run_id)
    if _m is None:
        raise ValueError(f"{_label}: no baseline tag (__bl-...__f-) in run ID {_run_id}")
    _baselines[_run_id] = _m.group(1)
if len(set(_baselines.values())) != 1:
    raise ValueError(
        "PANEL_ROWS mix runs with different baselines: "
        + "; ".join(f"{b} <- {r}" for r, b in _baselines.items())
    )
BASELINE_TAG = next(iter(_baselines.values()))
print(f"[config] all {len(PANEL_ROWS)} run(s) share baseline {BASELINE_TAG}")

# The reconstructed values are MNE source amplitudes of a z-topography, so the
# unit is arbitrary and the sign only follows the cortical normal (opposing
# sulcal banks flip). The panel therefore shows |estimate| on a sequential map,
# with one symmetric scale shared by both rows: the 99.5th percentile of the
# pooled absolute values of exactly the panels that are drawn.
COLORMAP = "hot"
CBAR_LABEL = "|source estimate| (a.u.)"
# Width/height in pixels the white bands inside one snapshot are squeezed down
# to: HEMI_GAP_PX between the two hemispheres (columns), VIEW_GAP_PX between the
# lateral and the medial view (rows). The pyvista split view leaves a wide margin
# around every brain, which makes the four brains of one latency drift apart and
# the neighbouring latencies run together.
HEMI_GAP_PX = 10
VIEW_GAP_PX = 10
PERCENTILE = 99.5
TITLE_FONTSIZE = 15   # latency header above each snapshot
ROW_LABEL_FONTSIZE = 17

# Which reconstruction to read: "mne" (minimum-norm, MNE_METHOD = dSPM or MNE) or "lcmv"
# (beamformer). These settings must equal the ones in source_reconstruction_zvals.py,
# which writes one cache per setting; they are encoded in the cache file name.
INVERSE_METHOD = "mne" #os.getenv("INVERSE_METHOD", "mne").lower()
if INVERSE_METHOD not in ("mne", "lcmv"):
    raise ValueError(f"INVERSE_METHOD must be 'mne' or 'lcmv', got {INVERSE_METHOD!r}")
LCMV_REG = 0.05
LCMV_WEIGHT_NORM = "unit-noise-gain"
MNE_METHOD = "MNE"
MNE_LOOSE = 0
MNE_DEPTH = None #0.8
if MNE_METHOD not in ("dSPM", "MNE"):
    raise ValueError(f"MNE_METHOD must be 'dSPM' or 'MNE', got {MNE_METHOD!r}")

# Same naming scheme as step002_inverse_solution_multisub_alpha_lcmv.py (inverse_name).
if INVERSE_METHOD == "lcmv":
    inverse_name = f"{INVERSE_METHOD}_wn{LCMV_WEIGHT_NORM}_reg-{LCMV_REG}"
else:
    inverse_name = f"{INVERSE_METHOD}_loose-{MNE_LOOSE}_depth-{MNE_DEPTH}_method-{MNE_METHOD}"
INV_TAG = f"inv-{inverse_name}"
OUTPUT_STEM = f"source_zmap__manuscript__{INV_TAG}__unsigned__peak_latencies"
print(f"[config] inverse method: {inverse_name}")
MANUSCRIPT_IMGDIR = os.path.join(WDDIR, "Writing", "img")
MANUSCRIPT_STEM = "Fig02_source"

# %%
# ---------------------------------------------------------------------------
# Paths -- same expressions as in source_reconstruction_zvals.py, so this script reads
# the caches that one writes (one stc folder per run, figures in one folder per baseline).
# ---------------------------------------------------------------------------
RESDIR = os.getenv("RESDIR")
if not RESDIR:
    raise EnvironmentError("RESDIR is not set.")
RECONDIR = os.getenv("SUBJECTS_DIR")
if not RECONDIR:
    raise EnvironmentError("SUBJECTS_DIR is not set.")

cluster_results_dir = os.path.join(RESDIR, "Statistics", "eeg", "cluster_permutation", "cluster_mass", "runs")

source_root = os.path.join(
    RESDIR, "Figures", "Main", "eeg_results", "source_reconstruction", "zvals_clusterperm",
)
figuredir = os.path.join(source_root, f"selected__bl-{BASELINE_TAG}")


# %%
# ---------------------------------------------------------------------------
# Helpers (same as in source_reconstruction_zvals.py / plot_cluster_results_parcellated.py)
# ---------------------------------------------------------------------------
def _crop_white(img, threshold=250, pad=4):
    non_white = np.any(img < threshold, axis=2)
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - pad)
    rmax = min(img.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img.shape[1] - 1, cmax + pad)
    return img[rmin:rmax + 1, cmin:cmax + 1, :]


def _squeeze_white_bands(img, axis, max_gap, threshold=250):
    """Shrink the white bands inside a snapshot along one axis.

    axis=1 squeezes the gap between the two hemispheres, axis=0 the gap between
    the lateral and the medial view. Only bands enclosed by content on both sides
    are touched; the outer margins are _crop_white's job. Pixels are dropped from
    the middle of a band so the remaining gap stays centred between the brains.
    """
    non_white = np.any(img < threshold, axis=2)
    # content projected onto `axis`: a line is empty only if it is white in both
    # of the other axis's brains, so squeezing never shifts one brain against the
    # other.
    has_content = np.any(non_white, axis=1 - axis)
    n_lines = img.shape[axis]
    keep = np.ones(n_lines, dtype=bool)

    line_idx = 0
    while line_idx < n_lines:
        if has_content[line_idx]:
            line_idx += 1
            continue
        band_start = line_idx
        while line_idx < n_lines and not has_content[line_idx]:
            line_idx += 1
        band_end = line_idx  # exclusive
        if band_start == 0 or band_end == n_lines:
            continue  # outer margin
        n_drop = (band_end - band_start) - max_gap
        if n_drop > 0:
            band_mid = (band_start + band_end) // 2
            drop_from = band_mid - n_drop // 2
            keep[drop_from:drop_from + n_drop] = False

    return img[keep, :, :] if axis == 0 else img[:, keep, :]


def _pad_to_height(img, target_h, fill=255):
    if img.shape[0] >= target_h:
        return img
    pad = np.full((target_h - img.shape[0], img.shape[1], img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)


def _pad_to_width(img, target_w, fill=255):
    if img.shape[1] >= target_w:
        return img
    pad = np.full((img.shape[0], target_w - img.shape[1], img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([img, pad], axis=1)


COEFF_NAME_OVERRIDES = {
    "intensity_level_centered": "intensity level",
    "response_decoded_Yes": "detection response",
    "modality_Mechanical": "modality",
    "intensity_level_centered_response_decoded_Yes": "intensity level x detection response",
    "modality_Mechanical_response_decoded_Yes": "modality x detection response",
    "modality_Mechanical_intensity_level_centered": "modality x intensity level",
    "modality_Mechanical_intensity_level_centered_response_decoded_Yes":
        "modality x intensity level x detection response",
}


def format_coefficient_name(name: str) -> str:
    cleaned = re.sub(r"^\d+[_-]*", "", name)
    if cleaned in COEFF_NAME_OVERRIDES:
        return COEFF_NAME_OVERRIDES[cleaned]
    for suffix_ in ("_centered", "_scaled", "_z"):
        if cleaned.endswith(suffix_):
            cleaned = cleaned[: -len(suffix_)]
            break
    cleaned = cleaned.replace("_", " ").strip()
    return " ".join(cleaned.split())


def cluster_peaks(z_vals, cluster_display):
    """Per significant cluster: (cluster id, peak |z|, time index of the peak)."""
    ids = cluster_display[~np.isnan(cluster_display)]
    ids = np.unique(ids[~np.isclose(ids, 0.0)])
    peaks = []
    for cluster_id in ids:
        mask = np.isclose(cluster_display, cluster_id)
        coords = np.argwhere(mask)
        k = int(np.argmax(np.abs(z_vals[mask])))
        peaks.append((int(cluster_id), float(np.abs(z_vals[mask][k])), int(coords[k][1])))
    return peaks


def render_brain_snapshot(stc, t_idx, clim, colormap):
    """One composite screenshot: both hemispheres, lateral + medial views."""
    brain = stc.plot(
        subject="fsaverage",
        subjects_dir=RECONDIR,
        hemi="split",
        views=["lateral", "medial"],
        colorbar=False,
        clim=clim,
        colormap=colormap,
        background="white",
        time_viewer=False,
        show_traces=False,
        size=(1000, 900),
        initial_time=float(stc.times[t_idx]),
    )
    img = brain.screenshot()
    brain.close()
    img = _crop_white(img)
    img = _squeeze_white_bands(img, axis=1, max_gap=HEMI_GAP_PX)   # between hemispheres
    img = _squeeze_white_bands(img, axis=0, max_gap=VIEW_GAP_PX)   # between lateral and medial
    return img


def unsigned(stc):
    """|estimate|: the sign of a fixed-orientation MNE solution carries no meaning here."""
    return stc.__class__(
        np.abs(stc.data), vertices=stc.vertices, tmin=stc.tmin, tstep=stc.tstep,
        subject=stc.subject,
    )


# %%
# ---------------------------------------------------------------------------
# Load one cached group stc per row and compute its latencies from the cluster peaks.
# ---------------------------------------------------------------------------
rows = []
for row_label, run_id, coefficient_name, max_peaks, extra_times in PANEL_ROWS:
    cluster_file = os.path.join(
        cluster_results_dir, run_id, "coefficients", coefficient_name, "cluster_results.npz"
    )
    if not os.path.isfile(cluster_file):
        raise FileNotFoundError(f"{row_label}: cluster_results.npz not found: {cluster_file}")

    stc_stem = os.path.join(source_root, run_id, "stc", f"{coefficient_name}__{INV_TAG}-mean")
    if not (os.path.isfile(stc_stem + "-lh.stc") and os.path.isfile(stc_stem + "-rh.stc")):
        raise RuntimeError(
            f"{row_label}: no cached group stc at {stc_stem}-{{lh,rh}}.stc. Run "
            f"source_reconstruction_zvals.py with {inverse_name} for run "
            f"{run_id} first."
        )

    data = np.load(cluster_file)
    times = np.asarray(data["times"], dtype=float)
    p_min = float(np.nanmin(data["pvals_clustermass"]))
    stc = unsigned(mne.read_source_estimate(stc_stem))

    # The stc was inverted from exactly this z-map, so the two time axes must agree;
    # a mismatch means the cache belongs to a different run.
    if len(stc.times) != len(times) or not np.allclose(stc.times, times):
        raise RuntimeError(
            f"{row_label}: cached stc time axis {stc.times[0]:.3f}..{stc.times[-1]:.3f} s "
            f"({len(stc.times)} samples) does not match the z-map "
            f"{times[0]:.3f}..{times[-1]:.3f} s ({len(times)} samples)."
        )

    peaks = cluster_peaks(data["z_vals"], data["cluster_display"])
    if not peaks:
        raise ValueError(f"{row_label} ({run_id}): no significant cluster, so there is no peak to plot.")
    strongest = sorted(peaks, key=lambda pk: pk[1], reverse=True)[:max_peaks]
    peak_idx = np.unique([pk[2] for pk in strongest])

    # Extra latencies are non-peaks unless one lands on a peak sample, which stays a peak.
    extra_idx = np.array([], dtype=int)
    if extra_times:
        extra_times = np.asarray(extra_times, dtype=float)
        extra_idx = np.array([int(np.argmin(np.abs(times - t))) for t in extra_times], dtype=int)
        step = np.median(np.diff(times))
        off = np.abs(times[extra_idx] - extra_times) > step / 2
        if off.any():
            raise ValueError(
                f"{row_label}: extra latencies {np.round(extra_times[off] * 1000, 1)} ms are outside "
                f"the time axis or do not land on a sample (step {step * 1000:.1f} ms)."
            )
    time_indices = np.unique(np.concatenate([peak_idx, extra_idx]).astype(int))
    is_peak = np.isin(time_indices, peak_idx)

    print(f"[{row_label}] {coefficient_name} ({run_id[:15]}): smallest cluster-mass p = {p_min:.4f}, "
          f"{len(peaks)} significant cluster(s); latencies (ms, * = peak): "
          + ", ".join(f"{times[i] * 1000:.0f}{'*' if pk else ''}" for i, pk in zip(time_indices, is_peak)))

    rows.append({
        "row_label": row_label,
        "run_id": run_id,
        "coefficient_name": coefficient_name,
        "stc": stc,
        "times": times,
        "time_indices": time_indices,
        "is_peak": is_peak,
    })


# %%
# ---------------------------------------------------------------------------
# Shared colour scale over exactly the panels that are drawn.
# ---------------------------------------------------------------------------
panel_values = {
    (row["run_id"], row["coefficient_name"]): row["stc"].data[:, row["time_indices"]].ravel()
    for row in rows
}
V_MAX = float(np.percentile(np.concatenate(list(panel_values.values())), PERCENTILE))
CLIM = dict(kind="value", lims=[0.0, V_MAX / 2.0, V_MAX])

print(f"[plot] shared colour scale ({inverse_name}): {V_MAX:.3g} "
      f"({PERCENTILE}th percentile pooled over the plotted panels)")
for (_run, coefficient_name), vals in panel_values.items():
    print(f"[plot]   {coefficient_name}: own {PERCENTILE}th percentile = "
          f"{np.percentile(vals, PERCENTILE):.3g}")


# %%
# ---------------------------------------------------------------------------
# Panel figure: one row per coefficient, as many columns as the longest row.
# Rows do not share a time axis, so every panel carries its own latency title and
# the trailing axes of the shorter rows are switched off.
# ---------------------------------------------------------------------------
for row in rows:
    panels = [
        render_brain_snapshot(row["stc"], int(ti), CLIM, COLORMAP)
        for ti in row["time_indices"]
    ]
    # Pad panels to a common size so they tile cleanly.
    row_h = max(p.shape[0] for p in panels)
    col_w = max(p.shape[1] for p in panels)
    row["panels"] = [_pad_to_width(_pad_to_height(p, row_h), col_w) for p in panels]

n_rows = len(rows)
n_cols = max(len(row["panels"]) for row in rows)

# The grid cell is shaped like the snapshots it holds, so that squeezing the
# brains together above is not undone by matplotlib centring a wide image in a
# square axes. Everything below is laid out in inches and converted to figure
# fractions once, which keeps the header band and the colorbar the same physical
# size no matter how many rows or columns the configuration has.
CELL_W_IN = 3.0    # width of one snapshot axes
COL_GAP_IN = 0.15  # white gap between neighbouring latencies
panel_aspect = float(np.median([row["panels"][0].shape[1] / row["panels"][0].shape[0]
                                for row in rows]))
cell_h_in = CELL_W_IN / panel_aspect
# per row: bold row label on top plus the latency titles (title pad 6 pt, 4 pt
# between label and titles) and a small margin
header_in = (ROW_LABEL_FONTSIZE + 4 + TITLE_FONTSIZE + 6) / 72 + 0.08
cbar_in = 0.7      # below the grid: horizontal colorbar plus its label

fig_w_in = CELL_W_IN * n_cols + COL_GAP_IN * (n_cols - 1)
fig_h_in = n_rows * (cell_h_in + header_in) + cbar_in
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w_in, fig_h_in), squeeze=False)

for row_idx, row in enumerate(rows):
    row_times = row["times"][row["time_indices"]]
    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]
        ax.axis("off")
        if col_idx >= len(row["panels"]):
            continue
        ax.imshow(row["panels"][col_idx])
        # Bold title = cluster peak, regular title = latency chosen without a peak.
        ax.set_title(f"{row_times[col_idx] * 1000:.0f} ms", fontsize=TITLE_FONTSIZE,
                     fontweight="bold" if row["is_peak"][col_idx] else "normal")

# left/right span the full width so every axes is exactly CELL_W_IN wide and
# shaped like its snapshot; bbox_inches="tight" adds the outer margin on save.
fig.subplots_adjust(
    left=0.0,
    right=1.0,
    bottom=cbar_in / fig_h_in,
    top=1 - header_in / fig_h_in,
    wspace=COL_GAP_IN / CELL_W_IN,
    hspace=header_in / cell_h_in,
)

# Row label above its row, left-aligned with the first snapshot and clear of the
# latency titles (which sit TITLE_FONTSIZE plus matplotlib's 6 pt pad above the
# axes box).
title_band_in = (TITLE_FONTSIZE + 6 + 4) / 72
for row_idx, row in enumerate(rows):
    row_pos = axes[row_idx, 0].get_position()
    fig.text(
        row_pos.x0,
        row_pos.y1 + title_band_in / fig_h_in,
        row["row_label"],
        ha="left",
        va="bottom",
        fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold",
    )

# Horizontal colorbar under the grid, centred and at most two snapshots wide --
# stretched across the whole row it reads as a rule under the figure rather than
# as a scale.
grid_left = axes[0, 0].get_position().x0
grid_right = axes[0, -1].get_position().x1
two_cells = axes[0, min(1, n_cols - 1)].get_position().x1 - grid_left
cbar_w = min(two_cells, grid_right - grid_left)
cbar_h_in = 0.14
cbar_top_gap_in = 0.12  # between the lowest brains and the bar
cax = fig.add_axes([(grid_left + grid_right) / 2 - cbar_w / 2,
                    (cbar_in - cbar_top_gap_in - cbar_h_in) / fig_h_in,
                    cbar_w, cbar_h_in / fig_h_in])
plot_brain_colorbar(cax, CLIM, colormap=COLORMAP, transparent=False,
                    orientation="horizontal", label=CBAR_LABEL)

os.makedirs(figuredir, exist_ok=True)
for ext, dpi_ in (("pdf", 200), ("png", 300)):
    out_file = os.path.join(figuredir, f"{OUTPUT_STEM}.{ext}")
    # figure_style() sets savefig.transparent; force an opaque white ground so the
    # panel does not pick up the page colour when included in the manuscript.
    fig.savefig(out_file, dpi=dpi_, bbox_inches="tight", pad_inches=0.05,
                transparent=False, facecolor="white")

    manuscript_stem = os.path.join(MANUSCRIPT_IMGDIR, MANUSCRIPT_STEM)
    fig.savefig(f"{manuscript_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{manuscript_stem}.png", dpi=300, bbox_inches="tight")
    print(f"[manuscript] saved {manuscript_stem}")

plt.close(fig)

# %%
