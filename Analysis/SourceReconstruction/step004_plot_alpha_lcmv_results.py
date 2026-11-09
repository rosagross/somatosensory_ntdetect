######################################
# Standalone plotting script for alpha envelope & ITPC source results.
#
# Loads the group-level STCs saved by step002_inverse_solution_multisub_alpha_lcmv.py
# and regenerates all brain plots without re-running the full computation.
#
# Expected files in $DATADIR/derivatives/eeg-preproc/:
#   group_fsaverage_{modality}_cond-{envelope_yes|envelope_no|itpc_yes|itpc_no}_lcmv-None_preprestim_corrected_alpha-{lh|rh}.stc
######################################

# %%
import os
import sys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mne
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable

from plotting_style import figure_style
from helper_functions_sourcerec import read_average_labels

colors, dpi = figure_style()

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

if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

modalities = ["mecha", "elec"]
preproc_suffix = "_preprestim_corrected"
alpha_suffix = f"{preproc_suffix}_alpha"
alpha_tag = alpha_suffix.lstrip("_")
bf_name = "mne"
pick_ori = None
peak_detection_window = (-0.4, -0.1)

group_outdir = os.path.join(DATADIR, "derivatives", "eeg-preproc")
group_fig_root = os.path.join(
    RESDIR,
    "Figures",
    "Main",
    "eeg_results",
    "source_reconstruction",
    f"alpha_{bf_name}",
    "group")

# %%
# ---- Plot helper functions (identical to step002) ----

def get_global_peak_vertex_per_hemi(stc, time_window=None, mode="abs"):
    if time_window is not None:
        tmin, tmax = time_window
        tidx = np.where((stc.times >= tmin) & (stc.times <= tmax))[0]
    else:
        tidx = np.arange(len(stc.times))

    if len(tidx) == 0:
        raise ValueError("No STC samples found in requested time window.")

    if mode not in ("abs", "pos"):
        raise ValueError(f"Unsupported mode={mode}. Use 'abs' or 'pos'.")

    lh_count = len(stc.vertices[0])
    rh_count = len(stc.vertices[1])
    hemi_slices = {"lh": slice(0, lh_count), "rh": slice(lh_count, lh_count + rh_count)}
    peaks = {}

    for hemi_name, hemi_idx in (("lh", 0), ("rh", 1)):
        hemi_vertices = stc.vertices[hemi_idx]
        if len(hemi_vertices) == 0:
            raise ValueError(f"No vertices available for hemisphere {hemi_name}.")

        data_hemi = stc.data[hemi_slices[hemi_name], :][:, tidx]
        if mode == "abs":
            flat_peak_idx = int(np.argmax(np.abs(data_hemi)))
        else:
            flat_peak_idx = int(np.argmax(data_hemi))

        peak_vert_idx, peak_time_idx = np.unravel_index(flat_peak_idx, data_hemi.shape)
        peaks[hemi_name] = {
            "vertex": int(hemi_vertices[peak_vert_idx]),
            "time": float(stc.times[tidx][peak_time_idx]),
            "value": float(data_hemi[peak_vert_idx, peak_time_idx]),
        }

    return peaks


def plot_peak_vertices_per_hemi(stc, modality, condition, figuredir, suffix):
    if stc is None:
        return

    peaks = get_global_peak_vertex_per_hemi(stc, time_window=peak_detection_window, mode="abs")
    marker_colors = {"lh": "deepskyblue", "rh": "crimson"}

    for peak_hemi in ("lh", "rh"):
        sample_idx = int(np.argmin(np.abs(stc.times - peaks[peak_hemi]["time"])))
        sample_t = float(stc.times[sample_idx])
        sample_t_ms = int(round(sample_t * 1000))

        brain = stc.plot(
            subject="fsaverage",
            subjects_dir=RECONDIR,
            hemi="split",
            views=["lat", "med"],
            time_viewer=False,
            initial_time=sample_t,
            colorbar=True,
            size=(1200, 600),
            background="white",
        )
        for hemi in ("lh", "rh"):
            brain.add_foci(
                peaks[hemi]["vertex"],
                hemi=hemi,
                coords_as_verts=True,
                color=marker_colors[hemi],
                scale_factor=0.45,
            )

        out_file = os.path.join(
            figuredir,
            f"group_fsaverage_ses-{modality}_cond-{condition}_global-peak-hemi-{peak_hemi}_time-{sample_t_ms}ms{suffix}.pdf",
        )
        brain.save_image(out_file)
        brain.close()



def plot_average_combined(stc_mecha, stc_elec, condition, figuredir, suffix, n_top_vertices=0):

    if "envelope" in suffix:
        metric_label = "alpha envelope Yes - No difference"
    elif "itpc" in suffix:
        metric_label = "alpha ITPC Yes - No difference"
    else:
        metric_label = suffix.strip("_")

    # Average over time
    stc_mecha_average = stc_mecha.copy()
    stc_mecha_average.data = np.expand_dims(np.mean(stc_mecha_average.data, axis=1), axis=0).T
    stc_elec_average = stc_elec.copy()
    stc_elec_average.data = np.expand_dims(np.mean(stc_elec_average.data, axis=1), axis=0).T

    colormap = "mne_analyze"
    cmax = np.round(max(np.abs(stc_mecha_average.data).max(), np.abs(stc_elec_average.data).max()), 2)
    clim = dict(kind="value", pos_lims=[0, cmax / 2, cmax])

    # Colors for BA label borders (used both in brain and in the matplotlib legend)
    ba_label_colors = {"BA1": "cyan", "BA3b": "magenta"}
    _, _, soma_labels, soma_names, _, _ = read_average_labels(RECONDIR)

    # Render both brains and crop whitespace before building the figure
    screenshots = []
    for stc in [stc_mecha_average, stc_elec_average]:
        brain = stc.plot(
            subject="fsaverage",
            subjects_dir=RECONDIR,
            hemi="split",
            views=["lat", "med"],
            colorbar=False,
            clim=clim,
            colormap=colormap,
            size=(1600, 800),
            background="white",
            time_viewer=False,
            show_traces=False,
        )

        # Overlay BA1 and BA3b borders with distinct colors
        for label, name in zip(soma_labels, soma_names):
            if "BA1" in name:
                brain.add_label(label, borders=True, color=ba_label_colors["BA1"])
            elif "3b" in name:
                brain.add_label(label, borders=True, color=ba_label_colors["BA3b"])

        # Mark the top-n vertices by absolute value with a green dot
        n_lh = len(stc.vertices[0])
        if n_top_vertices > 0:
            top_verts = np.argsort(np.abs(stc.data[:, 0]))[-n_top_vertices:]
            for global_idx in top_verts:
                if global_idx < n_lh:
                    brain.add_foci(int(stc.vertices[0][global_idx]), hemi="lh",
                                coords_as_verts=True, color="yellow", scale_factor=0.45)
                else:
                    brain.add_foci(int(stc.vertices[1][global_idx - n_lh]), hemi="rh",
                                coords_as_verts=True, color="yellow", scale_factor=0.45)

        img = brain.screenshot()
        brain.close()

        # Split the 2×2 panel grid (lat/med rows, lh/rh cols), crop each panel
        # individually, then reassemble — removes inter-hemisphere whitespace
        h, w = img.shape[:2]
        half_h, half_w = h // 2, w // 2
        lh_lat = _crop_white(img[:half_h, :half_w], threshold=250, pad=8)
        rh_lat = _crop_white(img[:half_h, half_w:], threshold=250, pad=8)
        lh_med = _crop_white(img[half_h:, :half_w], threshold=250, pad=8)
        rh_med = _crop_white(img[half_h:, half_w:], threshold=250, pad=8)

        # Normalise each column to the same width and each row to the same height
        # so lateral and medial panels are perfectly aligned
        col_w_lh = max(lh_lat.shape[1], lh_med.shape[1])
        col_w_rh = max(rh_lat.shape[1], rh_med.shape[1])
        row_h_lat = max(lh_lat.shape[0], rh_lat.shape[0])
        row_h_med = max(lh_med.shape[0], rh_med.shape[0])

        lh_lat = _pad_to_width(_pad_to_height(lh_lat, row_h_lat), col_w_lh)
        rh_lat = _pad_to_width(_pad_to_height(rh_lat, row_h_lat), col_w_rh)
        lh_med = _pad_to_width(_pad_to_height(lh_med, row_h_med), col_w_lh)
        rh_med = _pad_to_width(_pad_to_height(rh_med, row_h_med), col_w_rh)

        lat_row = np.concatenate([lh_lat, rh_lat], axis=1)
        med_row = np.concatenate([lh_med, rh_med], axis=1)
        screenshots.append(np.concatenate([lat_row, med_row], axis=0))

    plt.rcParams.update(
        {
            "ytick.labelsize": "small",
            "xtick.labelsize": "small",
            "axes.labelsize": "small",
            "axes.titlesize": "medium",
            "grid.color": "0.75",
            "grid.linestyle": ":",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f"Perceived-Unperceived - {metric_label}", fontsize=12)

    legend_handles = [
        mpatches.Patch(color=ba_label_colors["BA1"], label="BA1"),
        mpatches.Patch(color=ba_label_colors["BA3b"], label="BA3b"),
    ]

    for i, (ax, screenshot, modality_label) in enumerate(zip(axes, screenshots, ["Mechanical", "Electrical"])):
        ax.imshow(screenshot)
        ax.set_title(modality_label)
        ax.axis("off")
        if i == 0:
            ax.legend(handles=legend_handles, loc="center", fontsize=7, framealpha=0.7)

    # Single shared colorbar on the right side of the figure
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.70, 0.15, 0.02, 0.7])
    mne.viz.plot_brain_colorbar(cax, clim, colormap, label=metric_label)

    fig.tight_layout()
    out_file = os.path.join(
        figuredir,
        f"group_fsaverage_cond-{condition}_combinedaverage{suffix}.pdf",
    )
    fig.savefig(out_file, bbox_inches="tight", dpi=500)
    plt.show(fig)


def plot_average(stc, modality, condition, figuredir, suffix):
    """Render the brain at a fixed timepoint (nearest available sample)."""
    if stc is None:
        return

    # average data 
    stc_average = stc.copy()
    stc_average.data = np.expand_dims(np.mean(stc_average.data, axis=1),axis=0).T

    # before/after results
    fig = plt.figure(figsize=(4, 4))
    axes = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.5)

    colormap = "mne_analyze"
    cmax = np.round(np.max([np.abs(np.min(stc.data)), np.max(stc.data)]),2)
    cmin = 0
    cmid = cmax/2
    clim = dict(kind="value", pos_lims=[cmin, cmid, cmax])

    brain = stc_average.plot(
        subject="fsaverage",
        subjects_dir=RECONDIR,
        hemi="split",
        views="lat",
        colorbar=False,
        clim=clim,
        colormap=colormap,
        size=(1200, 600),
        background="white",
        time_viewer=False,
        show_traces=False,
    )

    screenshot = brain.screenshot()
    brain.close()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

    # Tweak the figure style
    plt.rcParams.update(
        {
            "ytick.labelsize": "small",
            "xtick.labelsize": "small",
            "axes.labelsize": "small",
            "axes.titlesize": "medium",
            "grid.color": "0.75",
            "grid.linestyle": ":",
        }
    )

    axes[0].imshow(cropped_screenshot)
    # add a vertical colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap, label=f"{suffix}")
    axes[0].set_title(f"{condition} {modality}")
    axes[0].axis("off")

    out_file = os.path.join(
        figuredir,
        f"group_fsaverage_ses-{modality}_cond-{condition}_prestimaverage{suffix}.pdf",
    )
    plt.savefig(out_file)



def plot_at_timepoint(stc, modality, condition, figuredir, suffix, t_ms):
    """Render the brain at a fixed timepoint (nearest available sample)."""
    if stc is None:
        return
    t_s = t_ms / 1000.0
    sample_t = float(stc.times[int(np.argmin(np.abs(stc.times - t_s)))])
    actual_ms = int(round(sample_t * 1000))


    # before/after results
    fig = plt.figure(figsize=(4, 4))
    axes = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.5)

    colormap = "mne_analyze"
    cmax = np.round(np.max([np.abs(np.min(stc.data)), np.max(stc.data)]),2)
    cmin = 0
    cmid = cmax/2
    clim = dict(kind="value", pos_lims=[cmin, cmid, cmax])

    brain = stc.plot(
        subject="fsaverage",
        subjects_dir=RECONDIR,
        hemi="split",
        views="lat",
        initial_time=sample_t,
        colorbar=False,
        clim=clim,
        colormap=colormap,
        size=(1200, 600),
        background="white",
        time_viewer=False,
        show_traces=False,
    )

    screenshot = brain.screenshot()
    brain.close()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

    # Tweak the figure style
    plt.rcParams.update(
        {
            "ytick.labelsize": "small",
            "xtick.labelsize": "small",
            "axes.labelsize": "small",
            "axes.titlesize": "medium",
            "grid.color": "0.75",
            "grid.linestyle": ":",
        }
    )

    axes[0].imshow(cropped_screenshot)
    # add a vertical colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap, label=f"{suffix}")
    axes[0].set_title(f"{condition} {modality}")
    axes[0].axis("off")

    out_file = os.path.join(
        figuredir,
        f"group_fsaverage_ses-{modality}_cond-{condition}_time-{actual_ms}ms{suffix}.pdf",
    )
    plt.savefig(out_file)


def _make_mne_style_cmap(data_min, data_max, grey_frac=0.10, n=256):
    """Build a colormap and norm matching MNE's rendering style.

    Positive data: grey from 0 to fmin (grey_frac * max), then hot above.
    Signed data:   blue → grey (±grey_frac zone around zero) → red.

    Returns (mpl_cmap, norm_cb, mne_clim).
    """
    if data_min < 0:
        abs_max = max(abs(data_min), abs(data_max)) or 1.0
        n_grey_half = max(1, int(n * grey_frac / 2))
        center = n // 2
        rdbu = plt.cm.RdBu_r(np.linspace(0, 1, n)).copy()
        rdbu[center - n_grey_half : center + n_grey_half] = [0.75, 0.75, 0.75, 1.0]
        mpl_cmap = mcolors.ListedColormap(rdbu)
        norm_cb  = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        mne_clim = dict(kind="value", lims=[-abs_max, 0, abs_max])
    else:
        fmax_cb = data_max or 1.0
        fmin_cb = fmax_cb * grey_frac
        fmid_cb = fmax_cb * 0.5
        n_grey  = max(1, int(n * grey_frac))
        n_hot   = n - n_grey
        grey_part = np.tile([0.75, 0.75, 0.75, 1.0], (n_grey, 1))
        hot_part  = plt.cm.hot(np.linspace(0, 1, n_hot))
        mpl_cmap  = mcolors.ListedColormap(np.vstack([grey_part, hot_part]))
        norm_cb   = mcolors.Normalize(vmin=0, vmax=fmax_cb)
        mne_clim  = dict(kind="value", lims=[fmin_cb, fmid_cb, fmax_cb])
    return mpl_cmap, norm_cb, mne_clim


def _crop_white(img, threshold=250, pad=4):
    """Crop white margins from a brain panel."""
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


def _pad_to_height(img, target_h, fill=255):
    """Pad image bottom to reach target_h rows."""
    if img.shape[0] >= target_h:
        return img
    pad = np.full((target_h - img.shape[0], img.shape[1], img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)


def _pad_to_width(img, target_w, fill=255):
    """Pad image right to reach target_w columns."""
    if img.shape[1] >= target_w:
        return img
    pad = np.full((img.shape[0], target_w - img.shape[1], img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([img, pad], axis=1)


def plot_brain_timepoints(stc, modality, condition, figuredir, suffix, step_ms=50):
    """Plot source estimate at timepoints every step_ms ms.

    Layout: row 1 = lateral (LH | RH per timepoint), row 2 = medial (LH | RH per timepoint).
    Each hemisphere panel is tight-cropped to minimise whitespace.
    Starts at peak_detection_window[0] (-400 ms).
    """
    if stc is None:
        return

    step_s = step_ms / 1000.0
    t_start = max(stc.times[0], peak_detection_window[0])
    t_end = stc.times[-1]
    targets = np.arange(
        np.ceil(t_start / step_s) * step_s,
        t_end + step_s * 0.5,
        step_s,
    )
    seen = set()
    actual_times = []
    for t in targets:
        idx = int(np.argmin(np.abs(stc.times - t)))
        t_snap = float(stc.times[idx])
        if t_snap not in seen:
            seen.add(t_snap)
            actual_times.append(t_snap)

    # Compute colorbar limits from actual STC data and build clim for stc.plot().
    # For difference STCs (negative values present) use a symmetric diverging colormap.
    data_min = float(stc.data.min())
    data_max = float(stc.data.max())
    has_negative = data_min < 0

    if has_negative:
        abs_max = max(abs(data_min), abs(data_max))
        abs_max = abs_max if abs_max > 0 else 1.0
        fmin_cb, fmax_cb = -abs_max, abs_max
        mpl_cmap = plt.cm.RdBu_r
        mne_clim = dict(kind="value", lims=[-abs_max, 0, abs_max])
    else:
        fmax_cb = data_max if data_max > 0 else 1.0
        fmid_cb = fmax_cb * 0.5
        fmin_cb = fmax_cb * 0.1
        mpl_cmap = plt.cm.hot
        mne_clim = dict(kind="value", lims=[fmin_cb, fmid_cb, fmax_cb])

    norm_cb = mcolors.Normalize(vmin=fmin_cb, vmax=fmax_cb)

    screenshots = []
    for t in actual_times:
        brain = stc.plot(
            subject="fsaverage",
            subjects_dir=RECONDIR,
            hemi="split",
            views=["lat", "med"],
            time_viewer=False,
            initial_time=t,
            colorbar=False,
            clim=mne_clim,
            size=(800, 400),
            background="white",
        )
        img = brain.screenshot()
        brain.close()
        screenshots.append((t, img))

    if not screenshots:
        return

    t_panels = []
    for _t, img in screenshots:
        h, w = img.shape[:2]
        half_h, half_w = h // 2, w // 2
        t_panels.append((
            _crop_white(img[:half_h, :half_w, :]),
            _crop_white(img[:half_h, half_w:, :]),
            _crop_white(img[half_h:, :half_w, :]),
            _crop_white(img[half_h:, half_w:, :]),
        ))

    lat_h = max(max(p[0].shape[0], p[1].shape[0]) for p in t_panels)
    med_h = max(max(p[2].shape[0], p[3].shape[0]) for p in t_panels)

    lat_strips, med_strips, col_widths = [], [], []
    for lh_lat, rh_lat, lh_med, rh_med in t_panels:
        lat_strip = np.concatenate([_pad_to_height(lh_lat, lat_h), _pad_to_height(rh_lat, lat_h)], axis=1)
        med_strip = np.concatenate([_pad_to_height(lh_med, med_h), _pad_to_height(rh_med, med_h)], axis=1)
        col_w = max(lat_strip.shape[1], med_strip.shape[1])
        lat_strip = _pad_to_width(lat_strip, col_w)
        med_strip = _pad_to_width(med_strip, col_w)
        lat_strips.append(lat_strip)
        med_strips.append(med_strip)
        col_widths.append(col_w)

    row_lat = np.concatenate(lat_strips, axis=1)
    row_med = np.concatenate(med_strips, axis=1)
    combined = np.concatenate([row_lat, row_med], axis=0)

    n_times = len(screenshots)
    fig_w = max(4, n_times * 2)
    img_ratio = combined.shape[0] / combined.shape[1]
    fig_img_h = max(2, fig_w * img_ratio)

    fig = plt.figure(figsize=(fig_w, fig_img_h + 0.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[fig_img_h, 0.3], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_cb = fig.add_subplot(gs[1])

    ax.imshow(combined)
    ax.axis("off")

    cumx = 0
    for (t, _), col_w in zip(screenshots, col_widths):
        t_ms = int(round(t * 1000))
        ax.text(
            cumx + col_w / 2,
            -combined.shape[0] * 0.02,
            f"{t_ms} ms",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax.transData,
        )
        cumx += col_w

    ax.text(
        -combined.shape[1] * 0.01,
        lat_h // 2,
        "Lateral",
        ha="right",
        va="center",
        fontsize=8,
        rotation=90,
        transform=ax.transData,
    )
    ax.text(
        -combined.shape[1] * 0.01,
        lat_h + med_h // 2,
        "Medial",
        ha="right",
        va="center",
        fontsize=8,
        rotation=90,
        transform=ax.transData,
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_cb, cmap=mpl_cmap),
        cax=ax_cb,
        orientation='horizontal',
    )
    cbar.set_label('Source amplitude (a.u.)', fontsize=8)
    ax_cb.tick_params(labelsize=7)

    out_file = os.path.join(
        figuredir,
        f"group_fsaverage_ses-{modality}_cond-{condition}_timepoints-{step_ms}ms{suffix}.pdf",
    )
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# %%
# ---- Load saved group STCs and regenerate all plots ----

all_stcs = {}  # {modality: {metric: diff_stc}} collected for combined plot

for modality in modalities:

    def _load(key):
        fname = os.path.join(
            group_outdir,
            f"group_fsaverage_{modality}_cond-{key}_{bf_name}-{pick_ori}_{alpha_tag}",
        )
        if not os.path.exists(fname + "-lh.stc") and not os.path.exists(fname + ".h5"):
            raise FileNotFoundError(
                f"Group STC not found for {modality} / {key}.\n"
                f"Expected: {fname}-lh.stc (or .h5)\n"
                f"Run step002_inverse_solution_multisub_alpha_lcmv.py first."
            )
        return mne.read_source_estimate(fname, subject="fsaverage")

    group_stcs = {k: _load(k) for k in ("envelope_yes", "envelope_no", "itpc_yes", "itpc_no")}
    print(f"Loaded group STCs for {modality}.")

    # Yes − No difference maps.
    stc_envelope_diff = group_stcs["envelope_yes"].copy()
    stc_envelope_diff._data = group_stcs["envelope_yes"].data - group_stcs["envelope_no"].data
    stc_itpc_diff = group_stcs["itpc_yes"].copy()
    stc_itpc_diff._data = group_stcs["itpc_yes"].data - group_stcs["itpc_no"].data

    all_stcs[modality] = {"envelope": stc_envelope_diff, "itpc": stc_itpc_diff}

    # Per-metric figure subdirectories.
    metric_fig_dirs = {}
    for metric in ("envelope", "itpc"):
        d = os.path.join(group_fig_root, metric)
        os.makedirs(d, exist_ok=True)
        metric_fig_dirs[metric] = d

    # Generate peak-vertex and timepoint plots for all conditions.
    plot_pairs = [
        #("envelope", "yes",        group_stcs["envelope_yes"]),
        #("envelope", "no",         group_stcs["envelope_no"]),
        ("envelope", "YesMinusNo", stc_envelope_diff),
        #("itpc",     "yes",        group_stcs["itpc_yes"]),
        #("itpc",     "no",         group_stcs["itpc_no"]),
        ("itpc",     "YesMinusNo", stc_itpc_diff),
    ]
    for metric, cond_label, stc_obj in plot_pairs:
        fig_dir = metric_fig_dirs[metric]
        suffix  = f"_{metric}_{alpha_tag}"
        """plot_peak_vertices_per_hemi(
            stc_obj, modality=modality, condition=cond_label,
            figuredir=fig_dir, suffix=suffix,
        )
        plot_brain_timepoints(
            stc_obj, modality=modality, condition=cond_label,
            figuredir=fig_dir, suffix=suffix,
        )"""
        
        """plot_average(
            stc_obj, modality=modality, condition=cond_label,
            figuredir=fig_dir, suffix=suffix
        )"""
        print("arrived here")
        
        #for t_ms in (-200, -100, -50):
        #    plot_at_timepoint(
        #        stc_obj, modality=modality, condition=cond_label,
        #        figuredir=fig_dir, suffix=suffix, t_ms=t_ms,
        #    )

    print(f"Done plotting {modality}.")

# ---- Combined average: both modalities × both metrics ----
combined_figdir = os.path.join(group_fig_root, "combined")
os.makedirs(combined_figdir, exist_ok=True)
for metric in ("envelope", "itpc"):
    plot_average_combined(
        stc_mecha=all_stcs["mecha"][metric],
        stc_elec=all_stcs["elec"][metric],
        condition="YesMinusNo",
        figuredir=combined_figdir,
        suffix=f"_{metric}_{alpha_tag}",
    )

# %%
