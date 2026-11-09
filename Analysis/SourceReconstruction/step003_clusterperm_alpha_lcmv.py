"""
File: step003_clusterperm_alpha_lcmv.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-10-20
Description: 
Spatio-temporal cluster permutation test for alpha envelope & ITPC source maps.

 Loads the per-subject morphed STCs saved by step002_inverse_solution_multisub_alpha_lcmv.py
 (suffix _fsaverage), computes the per-subject Yes−No difference, and runs a one-sample
 spatio-temporal cluster permutation test across subjects.

 Expected inputs in $DATADIR/derivatives/eeg-preproc/sub-0{subid}/ses-{modality}/:
   sub-0{subid}_ses-{modality}_cond-{metric}_{yes|no}_lcmv-None__preprestim_corrected_alpha_fsaverage-lh.stc

 Outputs in $RESDIR/Statistics/eeg/source_reconstruction/cluster_perm_alpha_lcmv/:
   cluster_alpha_lcmv_ses-{modality}_{metric}_{inverse_name}_preprestim_corrected_alpha.pkl
   cluster_stc_alpha_lcmv_ses-{modality}_{metric}_{inverse_name}_preprestim_corrected_alpha   (STC files)
"""


# %%
import os
import sys
import pickle
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mne.viz import plot_brain_colorbar

from plotting_style import figure_style

colors, dpi = figure_style()

DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
RECONDIR = os.getenv("SUBJECTS_DIR")
WDDIR = os.getenv("WDDIR")

if WDDIR not in sys.path:
    sys.path.insert(0, WDDIR)

from general_helpers import is_subject_excluded

# ---- Subject lists (from step002_inverse_solution_multisub_alpha_lcmv.py) ----
subID_mecha = [30, 32, 40, 16, 17, 18, 19, 24, 26, 27, 28, 29, 34, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
subID_elec  = [15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

modalities     = ["mecha"] # "mecha"
metrics        = ["itpc"] # "envelope"
preproc_suffix = "_preprestim_corrected"
alpha_suffix   = f"{preproc_suffix}_alpha"
alpha_tag      = alpha_suffix  


from alpha_source_common import (
    inverse_method, reg, pick_ori, weight_norm, mne_method, loose, depth, inverse_name,
    FAULTY_SESSIONS,
)


# Permutation parameter
n_perm         = 2000
alpha_thresh   = 0.05    # cluster-forming threshold (per-sample p-value)
tail           = 0       # two-sided: tests whether Yes−No differs from zero

plotting_timepoint = -0.25

src_file_fsaverage = os.path.join(RECONDIR, "fsaverage_data", "bem", "fsaverage-ico-5-src.fif")
fsaverage_src = mne.read_source_spaces(src_file_fsaverage)

cluster_stats_dir = os.path.join(
    RESDIR, "Statistics", "eeg", "source_reconstruction",
    "cluster_perm_alpha_lcmv", alpha_tag.lstrip("_"),
)
cluster_fig_dir = os.path.join(
    RESDIR, "Figures", "Main", "eeg_results", "source_reconstruction",
    f"alpha_{inverse_method}", "cluster_perm",
)
os.makedirs(cluster_stats_dir, exist_ok=True)
os.makedirs(cluster_fig_dir, exist_ok=True)

# %%
def _load_morphed_stc(sub_dir, subid, modality, metric, condition):
    fname = os.path.join(
        sub_dir,
        f"sub-0{subid}_ses-{modality}_cond-{metric}_{condition}"
        f"_{inverse_name}-{pick_ori}_{alpha_tag}_fsaverage",
    )
    if not os.path.exists(fname + "-lh.stc") and not os.path.exists(fname + ".h5"):
        raise FileNotFoundError(
            f"Morphed STC not found for sub-0{subid} {modality} {metric}/{condition}.\n"
            f"Expected: {fname}-lh.stc\n"
            f"Re-run step002_inverse_solution_multisub_alpha_lcmv.py first."
        )
    return mne.read_source_estimate(fname, subject="fsaverage")


def _load_cluster_run(modality, metric, inverse_name=inverse_name,
                      stats_dir=cluster_stats_dir):
    """
    Read back a run saved by the loop below, so the plotting cells can be re-run
    without repeating the permutation test.

    Returns the results dict; the caller unpacks it.
    """
    pkl_path = os.path.join(
        stats_dir,
        f"cluster_alpha_lcmv_ses-{modality}_{metric}_{inverse_name}{alpha_tag}.pkl",
    )
    if not os.path.exists(pkl_path):
        available = sorted(f for f in os.listdir(stats_dir) if f.endswith(".pkl"))
        raise FileNotFoundError(
            f"No saved run at {pkl_path}\n"
            f"Available in {stats_dir}:\n  "
            + ("\n  ".join(available) if available else "(none)")
        )
    with open(pkl_path, "rb") as f:
        run = pickle.load(f)

    # Pickles written before inverse_name was recorded do not have the key, so the
    # check is a no-op for those; anything that does carry it must agree with the
    # name the file was found under.
    if run.get("inverse_name", inverse_name) != inverse_name:
        raise ValueError(
            f"{pkl_path} was written with inverse_name={run['inverse_name']!r} "
            f"but loaded as {inverse_name!r}."
        )

    n_sig = int(np.sum(run["p_vals"] < run["alpha_thresh"]))
    print(f"Loaded {modality}/{metric}: {run['n_subjects']} subjects, "
          f"{n_sig}/{len(run['clusters'])} significant clusters, "
          f"times {run['times'][0]:.3f}–{run['times'][-1]:.3f}s")
    return run


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


def build_tvalue_stc(T_obs, clusters, p_vals, times_ref, alpha_thresh, vertices):
    """
    Masked t-value SourceEstimate: T_obs kept only inside significant clusters, else 0.

    T_obs is (n_times, n_verts); the returned STC is (n_verts, n_times) as MNE expects.
    """
    T_masked = np.zeros_like(T_obs)
    for cluster, pval in zip(clusters, p_vals):
        if pval < alpha_thresh:
            t_idx, v_idx = cluster
            T_masked[t_idx, v_idx] = T_obs[t_idx, v_idx]
    return mne.SourceEstimate(
        data=T_masked.T,
        vertices=vertices,
        tmin=float(times_ref[0]),
        tstep=float(times_ref[1] - times_ref[0]),
        subject="fsaverage",
    )


def plot_cluster_stc(cluster_stc, modality, metric, out_dir, suffix, plot_time, show=False):
    # Use the max ABSOLUTE t-value: clusters can be all-negative (e.g. envelope No>Yes),
    # in which case data.max() would be 0 and pos_lims=[0,0,0] collapses the diverging
    # map to grey/black. abs-max gives a symmetric scale that colours negative clusters.
    fmax = float(np.abs(cluster_stc.data).max())
    if fmax == 0:
        print(f"  No significant clusters to plot for {modality} {metric}.")
        return

    clim = dict(kind="value", pos_lims=[0, fmax / 2, fmax])

    colormap = "mne_analyze"

    # A timepoint is required: snap it to the nearest available sample.
    t_idx = int(np.argmin(np.abs(cluster_stc.times - plot_time)))
    initial_time = float(cluster_stc.times[t_idx])

    screenshots = []
    for view in ("lat", "med"):
        brain = cluster_stc.plot(
            subject="fsaverage",
            subjects_dir=RECONDIR,
            hemi="split",
            views=view,
            colorbar=False,
            clim=clim,
            colormap=colormap,
            size=(1200, 400),
            background="white",
            time_viewer=False,
            show_traces=False,
            time_label=None,   # timepoint is shown in the figure title instead
            initial_time=initial_time,
        )
        img = brain.screenshot()
        brain.close()
        screenshots.append((view, img))

    h, w = screenshots[0][1].shape[:2]
    half_w = w // 2
    panels = []
    for view, img in screenshots:
        lh = _crop_white(img[:, :half_w, :])
        rh = _crop_white(img[:, half_w:, :])
        row_h = max(lh.shape[0], rh.shape[0])
        lh = _pad_to_height(lh, row_h)
        rh = _pad_to_height(rh, row_h)
        panels.append(np.concatenate([lh, rh], axis=1))

    col_w = max(p.shape[1] for p in panels)
    panels = [_pad_to_width(p, col_w) for p in panels]
    combined = np.concatenate(panels, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    fig.suptitle(
        f"Significant clusters — {metric} Yes−No | {modality} | t = {initial_time * 1000:.0f} ms",
        fontsize=11,
    )
    for ax, (view, _), panel in zip(axes, screenshots, panels):
        ax.imshow(panel)
        ax.set_title(view)
        ax.axis("off")

    # Add a colorbar matching the brain rendering (same clim/colormap). The screenshots were
    # taken with colorbar=False, so we build one from the same settings via plot_brain_colorbar.
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    plot_brain_colorbar(
        cax, clim, colormap=colormap, transparent=False,
        orientation="vertical", label="t-value",
    )

    # Encode the plotted timepoint in the filename (ms, sign as m/p so no bare "-").
    t_ms = initial_time * 1000
    t_tag = f"t{'m' if t_ms < 0 else 'p'}{abs(t_ms):.0f}ms"
    out_file = os.path.join(out_dir, f"cluster_stc_{metric}_ses-{modality}_{suffix}_{t_tag}.pdf")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved cluster plot: {out_file}")


# %%
print("Computing adjacency for fsaverage source space.")
adjacency = mne.spatial_src_adjacency(fsaverage_src)

for modality in modalities:
    subIDs = subID_mecha if modality == "mecha" else subID_elec
    # Exclude faulty sessions (setup issues, shifted cue-stimulus interval); same shared
    # FAULTY_SESSIONS set as the session list of step002.
    subIDs = [s for s in subIDs if (modality, s) not in FAULTY_SESSIONS]
    # Sessions on Pre-Processing/exclued_subs.json (low post-cue ERP SNR in O1/O2). Filtered
    # here, where the cohort is defined, rather than inside the loop below.
    dropped = [s for s in subIDs if is_subject_excluded(s, modality)]
    if dropped:
        print(f"{modality}: excluding {dropped} (on the exclusion list).")
    subIDs = [s for s in subIDs if not is_subject_excluded(s, modality)]

    for metric in metrics:
        print(f"\n=== {modality} / {metric} ===")

        subject_diffs = []
        subjects_used = []

        for subid in subIDs:
            sub_dir = os.path.join(
                DATADIR, "derivatives", "eeg-preproc", f"sub-0{subid}", f"ses-{modality}"
            )
            try:
                stc_yes = _load_morphed_stc(sub_dir, subid, modality, metric, "yes")
                stc_no  = _load_morphed_stc(sub_dir, subid, modality, metric, "no")
            except FileNotFoundError as e:
                print(f"  Skipping sub-0{subid}: {e}")
                continue

            diff = stc_yes.data - stc_no.data   # (n_verts, n_times)
            subject_diffs.append(diff.T)          # (n_times, n_verts)
            subjects_used.append(subid)
            print(f"  Loaded sub-0{subid}")

        if len(subject_diffs) < 2:
            raise RuntimeError(
                f"Not enough subjects with data for {modality} {metric} "
                f"(need ≥2, found {len(subject_diffs)})."
            )

        X = np.stack(subject_diffs, axis=0)   # (n_subjects, n_times, n_verts)
        n_subjects = X.shape[0]
        print(f"  Data shape: {X.shape}  (subjects={n_subjects})")

        t_thresh = scipy.stats.t.ppf(1 - alpha_thresh / 2, df=n_subjects - 1)
        print(f"  Cluster-forming t-threshold (p={alpha_thresh}, two-sided, df={n_subjects-1}): {t_thresh:.4f}")

        T_obs, clusters, p_vals, H0 = spatio_temporal_cluster_1samp_test(
            X,
            adjacency=adjacency,
            threshold=t_thresh,
            n_permutations=n_perm,
            tail=tail,
            n_jobs=-1,
            verbose=True,
        )

        n_sig = int(np.sum(p_vals < alpha_thresh))
        print(f"  Significant clusters (p<{alpha_thresh}): {n_sig} of {len(clusters)}")

        times_ref = stc_yes.times  # same for all subjects after morphing

        results = {
            "T_obs":          T_obs,
            "clusters":       clusters,
            "p_vals":         p_vals,
            "H0":             H0,
            "times":          times_ref,
            "n_subjects":     n_subjects,
            "subjects_used":  subjects_used,
            "modality":       modality,
            "metric":         metric,
            "alpha_thresh":   alpha_thresh,
            "tail":           tail,
            "n_perm":         n_perm,
            "t_thresh":       t_thresh,
            "inverse_name":   inverse_name,
        }
        # inverse_name is part of the filename (as it is for the STC below): without it
        # the lcmv and mne branches write to the same path and overwrite each other.
        pkl_path = os.path.join(
            cluster_stats_dir,
            f"cluster_alpha_lcmv_ses-{modality}_{metric}_{inverse_name}{alpha_tag}.pkl",
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"  Saved results: {pkl_path}")

        cluster_stc = summarize_clusters_stc(
            (T_obs, clusters, p_vals, H0),
            tstep=float(times_ref[1] - times_ref[0]),
            vertices=[fsaverage_src[0]["vertno"], fsaverage_src[1]["vertno"]],
            subject="fsaverage",
        )
        stc_out = os.path.join(
            cluster_stats_dir,
            f"cluster_stc_alpha_lcmv_ses-{modality}_{metric}_{inverse_name}{alpha_tag}",
        )
        cluster_stc.save(stc_out, overwrite=True)
        print(f"  Saved cluster STC: {stc_out}")

        # Plot the masked t-value map (not the temporal-extent summary above).
        tval_stc = build_tvalue_stc(
            T_obs, clusters, p_vals, times_ref, alpha_thresh,
            vertices=[fsaverage_src[0]["vertno"], fsaverage_src[1]["vertno"]],
        )
        plot_cluster_stc(
            tval_stc,
            modality=modality,
            metric=metric,
            out_dir=cluster_fig_dir,
            suffix=f"{inverse_name}{alpha_tag}",
            plot_time=plotting_timepoint,   # required: time in seconds, e.g. -0.2
        )

print("\nDone.")

# %%
###########################################
#            Read existing runs
###########################################

# Restores a saved run so the plotting cells below can be re-run without repeating
# the permutation test. Edit these two and re-run this cell.
read_modality, read_metric = "elec", "itpc"

run          = _load_cluster_run(read_modality, read_metric)
T_obs        = run["T_obs"]
clusters     = run["clusters"]
p_vals       = run["p_vals"]
H0           = run["H0"]
times_ref    = run["times"]
alpha_thresh = run["alpha_thresh"]
n_subjects   = run["n_subjects"]
modality     = run["modality"]
metric       = run["metric"]

# The summarize_clusters_stc map saved next to the pickle. It encodes temporal
# extent, not t-values - the plots below use build_tvalue_stc(T_obs, ...) instead.
cluster_stc = mne.read_source_estimate(
    os.path.join(
        cluster_stats_dir,
        f"cluster_stc_alpha_lcmv_ses-{modality}_{metric}_{inverse_name}{alpha_tag}",
    ),
    subject="fsaverage",
)

# %%
# Masked t-value map for the loaded modality/metric, at a fixed timepoint.
plotting_timepoint = -0.1
tval_stc = build_tvalue_stc(
    T_obs, clusters, p_vals, times_ref, alpha_thresh,
    vertices=[fsaverage_src[0]["vertno"], fsaverage_src[1]["vertno"]],
)
print("tval_stc data range:", tval_stc.data.min(), tval_stc.data.max())
print("Non-zero entries:", np.sum(tval_stc.data != 0))

plot_cluster_stc(
    tval_stc,
    modality=modality,
    metric=metric,
    out_dir=cluster_fig_dir,
    suffix=f"{inverse_name}{alpha_tag}",
    plot_time=plotting_timepoint,   # required: time in seconds, e.g. -0.2
    show=True
)


# %%

for i, (cluster, pval) in enumerate(zip(clusters, p_vals)):
    if pval < alpha_thresh:
        t_idx, v_idx = cluster
        print(f"Cluster {i}: p={pval:.4f} | "
              f"verts={len(np.unique(v_idx))} | "
              f"times={len(np.unique(t_idx))} | "
              f"t range={times_ref[t_idx.min()]:.3f}–{times_ref[t_idx.max()]:.3f}s | "
              f"T range={T_obs[t_idx, v_idx].min():.2f}–{T_obs[t_idx, v_idx].max():.2f}")
# %%
