import os 
import sys
import json
import mne
from mne.beamformer import apply_lcmv, make_lcmv #, make_nulling_beamformer, apply_nulling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import pyvista as pv
from sklearn.decomposition import TruncatedSVD
from plotting_style import figure_style
colors, dpi = figure_style()

# Order matters: check longer suffix first to avoid substring conflicts.
_BASELINE_SUFFIX_MAP = [
    ("_preprestim", (-0.8, -0.7)),
    ("preprestim", (-0.8, -0.7)),
    ("_prestim", (-0.5, -0.43)),
    ("prestim", (-0.5, -0.43)),
]
_DEFAULT_BASELINE = (-0.1, 0.0)

# Color mapping for somatosensory label groups used in peak-vertex plots.
AREA_COLORS = {
    "BA1": "#E63946",
    "BA2": "#F4A261",
    "BA3a": "#2A9D8F",
    "BA3b": "#457B9D",
    "S2": "#9B5DE5",
}


def get_exclusive_peak_vertices(stc_list, labels, n_vertices=10, time_window=None):
    """
    Find top-N vertices per label while removing any vertices shared across labels.

    Parameters
    ----------
    stc_list : list of SourceEstimate
        STCs used to compute the selection score (typically yes/no conditions).
    labels : list of Label
        Labels to evaluate.
    n_vertices : int
        Number of vertices to keep per label.
    time_window : tuple[float, float] | None
        Optional (start, stop) time window in seconds.
    """

    stc_avg_data = np.mean([stc.data for stc in stc_list], axis=0)
    ref_stc = stc_list[0]

    if time_window is not None:
        t_mask = (ref_stc.times >= time_window[0]) & (ref_stc.times <= time_window[1])
    else:
        t_mask = np.ones(len(ref_stc.times), dtype=bool)

    label_candidates = {}
    for label in labels:
        hemi_idx = 0 if label.hemi == "lh" else 1
        stc_verts = ref_stc.vertices[hemi_idx]
        offset = len(ref_stc.vertices[0]) if hemi_idx == 1 else 0

        print(len(label.vertices), "vertices in", label.name)

        common_verts, stc_idx_hemi, _ = np.intersect1d(
            stc_verts, label.vertices, return_indices=True
        )
        stc_data_idx = stc_idx_hemi + offset
        peak_act = np.max(np.abs(stc_avg_data[stc_data_idx][:, t_mask]), axis=1)

        label_candidates[label.name] = {
            vert: (data_idx, act)
            for vert, data_idx, act in zip(common_verts, stc_data_idx, peak_act)
        }

    all_verts = [v for cands in label_candidates.values() for v in cands]
    overlapping = {v for v in all_verts if all_verts.count(v) > 1}
    print(f"Found {len(overlapping)} overlapping vertices across labels - excluding all overlaps.")

    label_vertices = {}
    for label in labels:
        cands = label_candidates[label.name]
        exclusive = {
            vert: (data_idx, act)
            for vert, (data_idx, act) in cands.items()
            if vert not in overlapping
        }

        if len(exclusive) == 0:
            print(f"  WARNING: No exclusive vertices for {label.name}!")
            label_vertices[label.name] = np.array([], dtype=int)
            continue

        sorted_verts = sorted(exclusive.items(), key=lambda x: x[1][1], reverse=True)
        top_n = sorted_verts[:n_vertices]
        top_data_idx = np.array([data_idx for _, (data_idx, _) in top_n])
        top_act = [act for _, (_, act) in top_n]

        label_vertices[label.name] = top_data_idx
        print(
            f"  {label.name}: {len(top_data_idx)} vertices, "
            f"peak act range [{min(top_act):.4f}, {max(top_act):.4f}]"
        )

    return label_vertices


def extract_timecourses_from_vertices(stc, label_vertices):
    """Extract one mean time course per label from selected STC row indices."""
    return {
        label_name: np.mean(stc.data[idx], axis=0, keepdims=True)
        if len(idx) > 0
        else np.zeros((1, stc.data.shape[1]))
        for label_name, idx in label_vertices.items()
    }


def _label_to_area(label_name):
    """Map a FreeSurfer label name to an area key used by AREA_COLORS."""
    for area in AREA_COLORS:
        if area.lower() in label_name.lower():
            return area
    return label_name


def _data_indices_to_vertices(stc, data_indices, hemi):
    """Convert STC data-row indices to FreeSurfer vertex IDs for one hemisphere."""
    data_indices = np.asarray(data_indices, dtype=int)
    if data_indices.size == 0:
        return []

    lh_n = len(stc.vertices[0])
    if hemi == "lh":
        valid = data_indices[(data_indices >= 0) & (data_indices < lh_n)]
        return [int(stc.vertices[0][idx]) for idx in valid]

    valid = data_indices[(data_indices >= lh_n) & (data_indices < stc.data.shape[0])]
    local_idx = valid - lh_n
    return [int(stc.vertices[1][idx]) for idx in local_idx]


def get_max_activated_vertices_in_window(
    stc_yes,
    stc_no,
    time_window,
    n_vertices=50,
):
    """
    Compute top-N whole-cortex vertices from the yes-no difference STC within a time window.

    Selection score per vertex is the maximum absolute activation inside `time_window`.
    No temporal averaging is performed.
    """
    if stc_yes.data.shape != stc_no.data.shape:
        raise ValueError("stc_yes and stc_no have different data shapes; cannot compute difference STC.")
    if stc_yes.times.shape != stc_no.times.shape or not np.allclose(stc_yes.times, stc_no.times):
        raise ValueError("stc_yes and stc_no have different time axes; cannot compute difference STC.")

    t_start, t_stop = float(time_window[0]), float(time_window[1])
    if t_start >= t_stop:
        raise ValueError(f"Invalid time_window={time_window}; expected start < stop.")

    stc_diff = stc_yes.copy()
    stc_diff._data = stc_yes.data - stc_no.data

    t_mask = (stc_diff.times >= t_start) & (stc_diff.times <= t_stop)
    if not np.any(t_mask):
        raise ValueError(f"time_window={time_window} not in STC time range.")

    data_window = stc_diff.data[:, t_mask]
    abs_window = np.abs(data_window)

    # Per-vertex score without temporal averaging: strongest absolute sample in the window.
    peak_local_idx = np.argmax(abs_window, axis=1)
    peak_abs = abs_window[np.arange(abs_window.shape[0]), peak_local_idx]
    peak_signed = data_window[np.arange(data_window.shape[0]), peak_local_idx]
    peak_times = stc_diff.times[t_mask][peak_local_idx]

    n_select = int(min(max(1, n_vertices), stc_diff.data.shape[0]))
    top_data_indices = np.argsort(peak_abs)[::-1][:n_select].astype(int)

    lh_n = len(stc_diff.vertices[0])
    lh_mask = top_data_indices < lh_n
    lh_data_idx = top_data_indices[lh_mask]
    rh_data_idx = top_data_indices[~lh_mask]

    lh_vertex_ids = [int(stc_diff.vertices[0][idx]) for idx in lh_data_idx]
    rh_vertex_ids = [int(stc_diff.vertices[1][idx - lh_n]) for idx in rh_data_idx]

    global_best_data_idx = int(top_data_indices[0])
    global_best_time = float(peak_times[global_best_data_idx])

    return {
        "stc_diff": stc_diff,
        "time_window": (t_start, t_stop),
        "top_data_indices": top_data_indices,
        "top_peak_abs": peak_abs[top_data_indices],
        "top_peak_signed": peak_signed[top_data_indices],
        "top_peak_times": peak_times[top_data_indices],
        "global_peak_time": global_best_time,
        "lh_data_indices": lh_data_idx,
        "rh_data_indices": rh_data_idx,
        "lh_vertex_ids": lh_vertex_ids,
        "rh_vertex_ids": rh_vertex_ids,
    }



def plot_max_vertices_screenshot(
    stc_diff,
    selected_data_indices,
    modality,
    out_dir,
    recondir,
    identifier,
    suffix,
    colormap,
    clim,
    plot_time=None,
):
    """
    Plot selected max-activation vertices on a 3D brain screenshot with white background.
    2x2 panel layout (LH/RH x lateral/medial) with shared colorbar.
    """
    if selected_data_indices is None or len(selected_data_indices) == 0:
        raise ValueError("selected_data_indices is empty; nothing to plot.")

    if plot_time is None:
        abs_selected = np.abs(stc_diff.data[np.asarray(selected_data_indices, dtype=int)])
        strongest_flat = int(np.argmax(abs_selected))
        strongest_t_idx = strongest_flat % abs_selected.shape[1]
        plot_time = float(stc_diff.times[strongest_t_idx])
    else:
        plot_time = float(plot_time)

    if plot_time < float(stc_diff.times[0]) or plot_time > float(stc_diff.times[-1]):
        raise ValueError(
            f"plot_time={plot_time} outside STC time range "
            f"[{float(stc_diff.times[0])}, {float(stc_diff.times[-1])}]"
        )

    plot_time_ms = int(round(plot_time * 1000))
    selected_data_indices = np.asarray(selected_data_indices, dtype=int)
    lh_n = len(stc_diff.vertices[0])

    panel_spec = [
        ("lh", "lat", "Left hemisphere (lateral)"),
        ("rh", "lat", "Right hemisphere (lateral)"),
        ("lh", "med", "Left hemisphere (medial)"),
        ("rh", "med", "Right hemisphere (medial)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'Max-activation vertices (yes - no)  |  {modality}  |  t = {plot_time_ms} ms',
        fontsize=14, fontweight='bold', y=0.98,
    )

    print(stc_diff.data.min(), stc_diff.data.max())
    print('clim', clim['lims'][0])    

    for ax, (hemi, view_name, title) in zip(axes.flatten(), panel_spec):
        brain = stc_diff.plot(
            subject="fsaverage",
            subjects_dir=recondir,
            hemi=hemi,
            views=[view_name],
            time_viewer=False,
            initial_time=plot_time,
            colorbar=False,
            size=(900, 700),
            background="w",
            clim=clim,
            colormap=colormap,
            show_traces=False,
        )

        # overlay foci on matching hemisphere only
        for data_idx in selected_data_indices:
            if data_idx < lh_n:
                vert      = int(stc_diff.vertices[0][data_idx])
                vert_hemi = "lh"
            else:
                vert      = int(stc_diff.vertices[1][data_idx - lh_n])
                vert_hemi = "rh"
            if vert_hemi != hemi:
                continue
            brain.add_foci(
                vert,
                hemi=vert_hemi,
                color="black",
                scale_factor=0.35,
                coords_as_verts=True,
            )

        screenshot = brain.screenshot()
        brain.close()

        # crop white borders
        mask = np.any(screenshot < 250, axis=2)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            screenshot = screenshot[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        ax.imshow(screenshot)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    # shared colorbar on the right
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    mne.viz.plot_brain_colorbar(cax, clim, colormap, label="Activation (nAm)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    out_pdf = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_max-vertices_clean_time-{plot_time_ms}ms{suffix}.pdf",
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved max-vertices clean figure: {out_pdf}")
    plt.show()


def plot_selected_area_vertices_cortex_only(
    stc,
    area_vertex_indices,
    area_colors,
    modality,
    out_dir,
    subjects_dir,
    identifier,
    suffix,
    subject=None,
):
    """
    Plot selected vertices on a cortex-only 2x2 brain layout (LH/RH x lateral/medial).

    This helper intentionally shows no activation underlay and only overlays the chosen
    vertices for each area using area-specific colors.
    """
    plot_subject = subject 
    
    if not area_vertex_indices:
        raise ValueError("area_vertex_indices is empty; nothing to plot.")

    panel_spec = [
        ("lh", "lat", "Left hemisphere (lateral)"),
        ("rh", "lat", "Right hemisphere (lateral)"),
        ("lh", "med", "Left hemisphere (medial)"),
        ("rh", "med", "Right hemisphere (medial)"),
    ]
    lh_n = len(stc.vertices[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'Chosen vertices by area  |  {identifier}  |  {modality}',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    for ax, (hemi, view_name, title) in zip(axes.flatten(), panel_spec):
        # Use a plain cortical rendering (white background) and overlay only chosen foci.
        brain = mne.viz.Brain(
            subject=plot_subject,
            hemi=hemi,
            surf="inflated",
            subjects_dir=subjects_dir,
            background="white",
            cortex="low_contrast",
            views=view_name,
            size=(900, 700),
            show=False
        )

        for area_name, data_idx in area_vertex_indices.items():
            color = area_colors.get(area_name, "#111111")
            data_idx = np.asarray(data_idx, dtype=int)
            for idx in data_idx:
                if idx < lh_n:
                    vert = int(stc.vertices[0][idx])
                    vert_hemi = "lh"
                else:
                    vert = int(stc.vertices[1][idx - lh_n])
                    vert_hemi = "rh"
                if vert_hemi != hemi:
                    continue
                brain.add_foci(
                    vert,
                    hemi=vert_hemi,
                    color=color,
                    scale_factor=0.45,
                    coords_as_verts=True,
                )

        screenshot = brain.screenshot()
        brain.close()

        mask = np.any(screenshot < 250, axis=2)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            screenshot = screenshot[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        ax.imshow(screenshot)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    legend_patches = [
        mpatches.Patch(color=area_colors[name], label=name)
        for name in area_vertex_indices.keys()
        if name in area_colors
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=max(1, len(legend_patches)),
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_pdf = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_chosen-vertices-cortex-only{suffix}.pdf",
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved chosen-vertices cortex figure: {out_pdf}")
    plt.close()

def label_to_area(label_name):
    """Map a FreeSurfer label name to an area key for AREA_COLORS."""
    for area in AREA_COLORS:
        if area.lower() in label_name.lower():
            return area
    return label_name


def plot_noninflated_brain_vertices_single_view(
    stc,    
    src_sub,
    area_vertex_indices,
    a3b_label,
    a1_label,
    area_colors,
    modality,
    out_dir,
    subjects_dir,
    identifier,
    suffix,
    view,        # e.g. "lat", "dor", "fro", "par"
    subject=None,
    plotter_window_size=(3000, 2400),
    plotter_scale=2,
):
    """
    Plot chosen vertices with arrows on the non-inflated RH brain surface, single view.

    Parameters
    ----------
    view : str
        One of "lat", "dor", "fro", "par" (or any valid MNE Brain view string).
    plotter_window_size : tuple
        Offscreen framebuffer resolution (width, height) in pixels.
    plotter_scale : int
        Supersampling scale factor passed to plotter.screenshot().
    """

    stc_ref = stc[0]
    lh_n = len(stc_ref.vertices[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f'Chosen vertices by area  |  {identifier}  |  {modality}  |  view: {view}',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    brain = mne.viz.Brain(
        subject=subject,
        subjects_dir=subjects_dir,
        hemi='rh',
        surf='pial',
        background='white',
        views=view,
        alpha=0.15,
        size=(1200, 900),
        show=False,
    )

    plotter = brain._renderer.plotter
    plotter.camera.zoom(0.7) 

    for area_name, vert_idx in area_vertex_indices.items():
        color = area_colors.get(area_name, "#111111")

        vert_idx = np.asarray(vert_idx, dtype=int)
        vert_idx = stc_ref.vertices[1][vert_idx - lh_n]

        rr = src_sub[1]['rr'][vert_idx]
        nn = src_sub[1]['nn'][vert_idx]

        avg_normal = np.mean(nn, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        avg_origin = np.mean(rr, axis=0) * 1000

        origins = rr * 1000
        normals = nn

        point_cloud = pv.PolyData(origins)
        point_cloud['normals'] = normals
        point_cloud.set_active_vectors('normals')
        glyphs = point_cloud.glyph(orient='normals', scale=False, factor=8)
        plotter.add_mesh(glyphs, color=color, render=True)

        avg_cloud = pv.PolyData(avg_origin.reshape(1, 3))
        avg_cloud['normals'] = avg_normal.reshape(1, 3)
        avg_cloud.set_active_vectors('normals')
        avg_glyph = avg_cloud.glyph(orient='normals', scale=False, factor=30)
        plotter.add_mesh(avg_glyph, color='yellow', render=True)

    brain.add_label(a3b_label, color='grey', alpha=0.8)
    brain.add_label(a1_label, color='grey', alpha=0.8)

    plotter.window_size = list(plotter_window_size)
    screenshot = plotter.screenshot(None, return_img=True, scale=plotter_scale)
    brain.close()

    # crop whitespace
    mask = np.any(screenshot < 250, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size > 0 and cols.size > 0:
        screenshot = screenshot[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    ax.imshow(screenshot, interpolation='lanczos')
    ax.axis('off')

    legend_patches = [
        mpatches.Patch(color=area_colors[name], label=name)
        for name in area_vertex_indices.keys()
        if name in area_colors
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=max(1, len(legend_patches)),
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_noninflated_{view}{suffix}.pdf",
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    plt.close()


def plot_noninflated_brain_vertices(
    stc,
    src_sub,
    area_vertex_indices,
    a3b_label, 
    a1_label, 
    area_colors,
    modality,
    out_dir,
    subjects_dir,
    identifier,
    suffix,
    subject=None):
    """
    Plot the chosen vertices with arrows in normal direction on the non-inflated brain surface. 
    Vertices MUST be from right hemisphere!!
    """

    panel_spec = [
        ("rh", "lat", "Right hemisphere (lateral)"),
        ("rh", "dor", "Right hemisphere (dorsal)"),
        ("rh", "fro", "Right hemisphere (frontal)"),
        ("rh", "par", "Right hemisphere (perietal)")
    ]
    
    # to later convert vertex labels to correct hemisphere
    stc_ref = stc[0]
    lh_n = len(stc_ref.vertices[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'Chosen vertices by area  |  {identifier}  |  {modality}',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    for ax, (hemi, view_name, title) in zip(axes.flatten(), panel_spec):

        # --- Brain viewer (RH) ---
        brain = mne.viz.Brain(
            subject=subject,
            subjects_dir=subjects_dir,
            hemi='rh',
            surf='pial',
            background='white',
            views=view_name,
            alpha=0.15,
            size=(900, 700),
            show=False
        )

        plotter = brain._renderer.plotter
        
        for area_name, vert_idx in area_vertex_indices.items():
            color = area_colors.get(area_name, "#111111")

            vert_idx = np.asarray(vert_idx, dtype=int)
            vert_idx = stc_ref.vertices[1][vert_idx - lh_n]

            # --- Extract RH data ---
            rr = src_sub[1]['rr'][vert_idx]    # vertex positions (n_verts, 3) in meters
            nn = src_sub[1]['nn'][vert_idx]    # normal vectors  (n_verts, 3) unit vectors
            #use = src_sub[1]['inuse'].astype(bool)  # active vertices only

            # Compute average normal direction
            avg_normal = np.mean(nn, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)  # normalize

            # Average position (centroid of the patch)
            avg_origin = np.mean(rr, axis=0) * 1000  # in mm

            # Filter to active vertices
            origins = rr * 1000   # convert to mm
            normals = nn        # already unit vectors, scale as needed

            # --- Build glyphs ---
            point_cloud = pv.PolyData(origins)
            point_cloud['normals'] = normals
            point_cloud.set_active_vectors('normals')

            glyphs = point_cloud.glyph(
                orient='normals',
                scale=False,      # normals are unit vectors, use factor to control length
                factor=8          # length in mm — tune this
            )

            plotter.add_mesh(glyphs, color=color, render=True)

            # Average normal — single big arrow at centroid
            avg_cloud = pv.PolyData(avg_origin.reshape(1, 3))
            avg_cloud['normals'] = avg_normal.reshape(1, 3)
            avg_cloud.set_active_vectors('normals')
            avg_glyph = avg_cloud.glyph(orient='normals', scale=False, factor=30)  # bigger
            plotter.add_mesh(avg_glyph, color='yellow', render=True)

            brain.add_label(a3b_label, color='grey', alpha=0.8)
            brain.add_label(a1_label, color='grey', alpha=0.8)


        plotter.window_size = [3000, 2400]

        screenshot = plotter.screenshot(None, return_img=True, scale=1)
        screenshot = brain.screenshot()
        brain.close()

        mask = np.any(screenshot < 250, axis=2)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            screenshot = screenshot[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        ax.imshow(screenshot)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    legend_patches = [
        mpatches.Patch(color=area_colors[name], label=name)
        for name in area_vertex_indices.keys()
        if name in area_colors
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=max(1, len(legend_patches)),
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout() #rect=[0, 0.03, 1, 0.96])
    out_pdf = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_noninflated{suffix}.pdf",
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved chosen-vertices cortex figure: {out_pdf}")
    plt.close()

def plot_peak_vertices_screenshot(
    stc_yes,
    stc_no,
    labels,
    label_vertices,
    peak_window,
    modality,
    out_dir,
    recondir,
    identifier,
    suffix,
    colormap,
    clim
):
    """
    Save 3D screenshots with selected peak vertices highlighted on the mean yes/no STC.
    """
    stc_plot = stc_yes.copy()
    stc_plot._data = (stc_yes.data - stc_no.data) 

    selected_idx = (
        np.unique(
            np.concatenate(
                [np.asarray(idx, dtype=int) for idx in label_vertices.values() if len(idx) > 0]
            )
        )
        if len(label_vertices) > 0
        else np.array([], dtype=int)
    )
    if selected_idx.size == 0:
        print(f"[{modality}] No peak vertices available for 3D screenshot.")
        return

    t_mask = (stc_plot.times >= peak_window[0]) & (stc_plot.times <= peak_window[1])
    if not np.any(t_mask):
        raise ValueError(f"peak_window={peak_window} not in STC time range.")

    mean_abs_tc = np.mean(np.abs(stc_plot.data[selected_idx][:, t_mask]), axis=0)
    peak_t_local = int(np.argmax(mean_abs_tc))
    plot_time = float(stc_plot.times[t_mask][peak_t_local])
    plot_time_ms = int(round(plot_time * 1000))

    brain = stc_plot.plot(
        subject="fsaverage",
        subjects_dir=recondir,
        hemi="split",
        views=["lat", "med"],
        time_viewer=False,
        initial_time=plot_time,
        colorbar=True,
        size=(1400, 700),
        background="w",
        clim=clim,
        colormap=colormap,
        show_traces=False,
    )

    for label in labels:
        label_name = label.name
        if label_name not in label_vertices:
            continue
        area_key = _label_to_area(label_name)
        label_color = AREA_COLORS.get(area_key, "#555555")
        verts = _data_indices_to_vertices(stc_plot, label_vertices[label_name], label.hemi)
        if len(verts) == 0:
            continue
        """for vert in verts:
            brain.add_foci(
                vert,
                hemi=label.hemi,
                color=label_color,
                scale_factor=0.35,
                coords_as_verts=True,
            )"""
        brain.add_label(label, borders=True, color=label_color)

    out_file = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_peak-vertices_3d_time-{plot_time_ms}ms{suffix}.png",
    )
    brain.save_image(out_file)
    screenshot = brain.screenshot()
    brain.close()
    print(f"Saved peak-vertex 3D screenshot: {out_file}")

    brain_rh = stc_plot.plot(
        subject="fsaverage",
        subjects_dir=recondir,
        hemi="rh",
        views=["lat"],
        time_viewer=False,
        initial_time=plot_time,
        colorbar=True,
        size=(900, 700),
        background="w",
        clim=clim,
        colormap=colormap,
        show_traces=False,
    )

    for label in labels:
        if label.hemi != "rh":
            continue
        label_name = label.name
        if label_name not in label_vertices:
            continue
        area_key = _label_to_area(label_name)
        label_color = AREA_COLORS.get(area_key, "#555555")
        verts = _data_indices_to_vertices(stc_plot, label_vertices[label_name], label.hemi)
        if len(verts) == 0:
            continue
        """for vert in verts:
            brain_rh.add_foci(
                vert,
                hemi="rh",
                color=label_color,
                scale_factor=0.35,
                coords_as_verts=True,
            )"""
        brain_rh.add_label(label, borders=True, color=label_color)

    screenshot_brain_rh = brain_rh.screenshot()

    out_file_rh = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_peak-vertices_3d-rh-lateral_time-{plot_time_ms}ms{suffix}.png",
    )
    brain_rh.save_image(out_file_rh)
    brain_rh.close()
    print(f"Saved RH lateral peak-vertex screenshot: {out_file_rh}")

    # create a clean plot using the screenshot
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(
        f'Peak vertices – BA1 & BA3b (yes-no) |  {modality}  |  t = {plot_time_ms} ms',
        fontsize=14, fontweight='bold', y=1.01,
    )

    # crop white borders
    mask = np.any(screenshot_brain_rh < 250, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    cropped_screenshot = screenshot_brain_rh[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    axes.imshow(cropped_screenshot)
    axes.set_title('Right Hemisphere (lateral)', fontsize=12, fontweight='bold', pad=8)
    axes.axis('off')

    # add a vertical colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    mne.viz.plot_brain_colorbar(cax, clim, colormap, label="Activation (nAm)")

    # area colour legend
    legend_patches = [
        mpatches.Patch(color=AREA_COLORS['BA1'],  label='BA1'),
        mpatches.Patch(color=AREA_COLORS['BA3b'], label='BA3b'),
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=2,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    out_file_fig = os.path.join(
        out_dir,
        f'{identifier}_ses-{modality}_peak-vertices_clean_{plot_time_ms}ms{suffix}.pdf',
    )
    fig.savefig(out_file_fig, dpi=300, bbox_inches='tight')
    print(f'Saved clean brain figure: {out_file_fig}')
    plt.show()


def plot_activation_views_screenshot(
    stc_yes,
    stc_no,
    plot_time,
    modality,
    out_dir,
    recondir,
    identifier,
    suffix,
    colormap,
    clim,
):
    """
    Save a clean 2x2 screenshot panel (LH/RH × lateral/medial) of mean yes/no activation.
    The requested time point is provided explicitly via `plot_time` (seconds).
    No vertices or labels are overlaid.
    """
    stc_plot = stc_yes.copy()
    stc_plot._data = stc_yes.data - stc_no.data

    if plot_time < float(stc_plot.times[0]) or plot_time > float(stc_plot.times[-1]):
        raise ValueError(
            f"plot_time={plot_time} outside STC time range "
            f"[{float(stc_plot.times[0])}, {float(stc_plot.times[-1])}]"
        )

    plot_time_ms = int(round(plot_time * 1000))
    panel_spec = [
        ("lh", "lat", "Left hemisphere (lateral)"),
        ("rh", "lat", "Right hemisphere (lateral)"),
        ("lh", "med", "Left hemisphere (medial)"),
        ("rh", "med", "Right hemisphere (medial)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'Activation map (yes - no)  |  {modality}  |  t = {plot_time_ms} ms',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    for ax, (hemi, view_name, title) in zip(axes.flatten(), panel_spec):
        brain = stc_plot.plot(
            subject="fsaverage",
            subjects_dir=recondir,
            hemi=hemi,
            views=[view_name],
            time_viewer=False,
            initial_time=plot_time,
            colorbar=False,
            size=(900, 700),
            background="w",
            clim=clim,
            colormap=colormap,
            show_traces=False,
        )
        screenshot = brain.screenshot()
        brain.close()

        # Crop white borders from each panel so the cortical surface fills the subplot.
        mask = np.any(screenshot < 250, axis=2)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size > 0 and cols.size > 0:
            screenshot = screenshot[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        ax.imshow(screenshot)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    mne.viz.plot_brain_colorbar(cax, clim, colormap, label="Activation (nAm)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    out_file = os.path.join(
        out_dir,
        f"{identifier}_ses-{modality}_activation-views_4panel_time-{plot_time_ms}ms{suffix}.pdf",
    )
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved activation views figure: {out_file}")
    plt.show()


def save_peak_vertices_for_sequential_dipole(
    stc,
    labels,
    label_vertices,
    modality,
    datadir,
    suffix,
):
    """Save selected BA1/BA3b peak vertices for sequential dipole fitting."""
    payload = {}
    for label in labels:
        label_name = label.name
        if label_name not in label_vertices:
            continue
        area_key = _label_to_area(label_name)
        if area_key not in ("BA1", "BA3b"):
            continue

        hemi_key = "LH" if label.hemi == "lh" else "RH"
        out_key = f"{hemi_key}_{area_key}"
        verts = _data_indices_to_vertices(stc, label_vertices[label_name], label.hemi)
        if len(verts) == 0:
            continue

        payload[out_key] = {
            "fit_vertex": int(verts[0]),
            "all_vertices": [int(v) for v in verts],
            "label_name": label_name,
            "hemi": label.hemi,
            "area": area_key,
        }

    out_path = os.path.join(
        datadir,
        "derivatives",
        "eeg-preproc",
        f"group_peak_vertices_for_sequentialdipole_{modality}{suffix}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved peak vertices for sequential dipole script: {out_path}")


def plot_timecourses_per_area(
    lh_tc_yes,
    lh_tc_no,
    rh_tc_yes,
    rh_tc_no,
    somatosensory_labels,
    times,
    time_slice=slice(None),
    modality="",
    figsize=(22, 5),
):
    """
    Plot one panel per somatosensory area with LH/RH and perceived/unperceived traces.
    """
    t_ms = times[time_slice] * 1000

    areas_lh, areas_rh = {}, {}
    for lbl in somatosensory_labels:
        area = _label_to_area(lbl.name)
        if lbl.hemi == "lh":
            areas_lh[area] = lbl.name
        else:
            areas_rh[area] = lbl.name

    all_areas = sorted(set(areas_lh) | set(areas_rh))
    fig, axes = plt.subplots(1, len(all_areas), figsize=figsize, sharey=True)
    if len(all_areas) == 1:
        axes = [axes]

    fig.suptitle(
        f'Somatosensory source time courses – peak vertices  ({"  |  " + modality if modality else ""})',
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, area in zip(axes, all_areas):
        lh_name = areas_lh.get(area)
        rh_name = areas_rh.get(area)

        def _get(tc_dict, name):
            if name is None or name not in tc_dict:
                return None
            return tc_dict[name][0, time_slice]

        lh_yes = _get(lh_tc_yes, lh_name)
        lh_no = _get(lh_tc_no, lh_name)
        rh_yes = _get(rh_tc_yes, rh_name)
        rh_no = _get(rh_tc_no, rh_name)

        if lh_yes is not None:
            ax.plot(t_ms, lh_yes, color=colors["perceived"], ls="--", lw=2.5, label="Perceived LH")
        if rh_yes is not None:
            ax.plot(t_ms, rh_yes, color=colors["perceived"], ls="-", lw=2.5, label="Perceived RH")
        if lh_no is not None:
            ax.plot(t_ms, lh_no, color=colors["unperceived"], ls="--", lw=2.5, label="Unperceived LH")
        if rh_no is not None:
            ax.plot(t_ms, rh_no, color=colors["unperceived"], ls="-", lw=2.5, label="Unperceived RH")

        ax.axvline(0, color="k", ls="--", lw=1.5, alpha=0.6)
        ax.axhline(0, color="k", ls="-", lw=0.8, alpha=0.3)
        ax.set_title(area, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Amplitude (nAm)", fontsize=11)
        ax.set_xlim(t_ms[0], t_ms[-1])

    legend_elements = [
        mpatches.Patch(facecolor=colors["perceived"], alpha=1.0, label="Perceived"),
        mpatches.Patch(facecolor=colors["unperceived"], alpha=1.0, label="Unperceived"),
        plt.Line2D([0], [0], color="k", ls="--", lw=2, label="Left hemi"),
        plt.Line2D([0], [0], color="k", ls="-", lw=2, label="Right hemi"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()
    sns.despine(trim=True)
    return fig


def infer_baseline_window_from_suffix(name_or_path):
    """
    Infer baseline window from an epochs filename or suffix string.
    Returns (baseline_window, matched_suffix).
    """
    for suffix, window in _BASELINE_SUFFIX_MAP:
        if suffix in name_or_path:
            return window, suffix
    return _DEFAULT_BASELINE, ""


def apply_baseline_epochs(epochs, name_or_path):
    """
    Apply baseline correction to an MNE Epochs object using suffix inference.
    """
    baseline_window, matched_suffix = infer_baseline_window_from_suffix(name_or_path)
    print(
        f"[baseline] epochs: suffix='{matched_suffix}' window={baseline_window} source={name_or_path}"
    )
    epochs.apply_baseline(baseline_window)
    return baseline_window, matched_suffix

def read_labels(RECONDIR, subID):

    # read labels
    somatosensory_label_files = [
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'lh.BA1_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'lh.BA2_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'lh.BA3a_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'lh.BA3b_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'rh.BA1_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'rh.BA2_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'rh.BA3a_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'rh.BA3b_exvivo.label'),
    ]

    visual_label_files = [
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'lh.V1_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'lh.V2_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'rh.V1_exvivo.label'),
        os.path.join(RECONDIR, f'sub-0{subID}', 'label', 'rh.V2_exvivo.label'),
    ]

    # Read labels
    somatosensory_labels = [mne.read_label(label_file) for label_file in somatosensory_label_files]
    somatosensory_label_names = [
        'LH BA1',
        'LH BA2',
        'LH BA3a',
        'LH 3b',
        'RH BA1',
        'RH BA2',
        'RH BA3a',
        'RH 3b'
    ]

    visual_labels = [mne.read_label(label_file) for label_file in visual_label_files]
    visual_label_names = [
        'LH V1',
        'LH V2',
        'RH V1',
        'RH V2'
    ]

    return visual_labels, visual_label_names, somatosensory_labels, somatosensory_label_names


def read_average_labels(RECONDIR):

    # read labels
    somatosensory_label_files = [
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.BA1.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.BA2.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.BA3a.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.BA3b.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.BA1.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.BA2.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.BA3a.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.BA3b.label'),
    ]

    visual_label_files = [
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.V1.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.V2.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.V1.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.V2.label'),
    ]

    central_label_files = [
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'lh.BA4a.label'),
        os.path.join(RECONDIR, 'fsaverage_data', 'label', 'rh.BA4a.label')
    ]

    # Read labels
    somatosensory_labels = [mne.read_label(label_file) for label_file in somatosensory_label_files]
    somatosensory_label_names = [
        'LH BA1',
        'LH BA2',
        'LH BA3a',
        'LH 3b',
        'RH BA1',
        'RH BA2',
        'RH BA3a',
        'RH 3b'
    ]

    visual_labels = [mne.read_label(label_file) for label_file in visual_label_files]
    visual_label_names = [
        'LH V1',
        'LH V2',
        'RH V1',
        'RH V2'
    ]

    central_labels = [mne.read_label(label_file) for label_file in central_label_files]
    central_label_names = [
        'LH BA4a',
        'RH BA4a'
    ]

    return visual_labels, visual_label_names, somatosensory_labels, somatosensory_label_names, central_labels, central_label_names



def plot_time_course(stc, src, label_soma, label_vis_rh, label_vis_lh, figuredir):
    """
    Plot time course of V1 and A3b
    """
    # extract timecourse for region labels
    somatosensory_labels_ts = stc.extract_label_time_course(label_soma, src, mode='mean')
    visual_labels_ts_rh = stc.extract_label_time_course(label_vis_rh, src, mode='mean') 
    visual_labels_ts_lh = stc.extract_label_time_course(label_vis_lh, src, mode='mean') 
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1) Somatosensory
    axes[0].plot(stc.times * 1000, somatosensory_labels_ts.T, 
                'r-', linewidth=3, label='S1 RH (BA3b)')
    axes[0].axvline(0, color='k', ls='--', alpha=0.5)
    axes[0].set_ylabel('Amplitude (nAm)')
    axes[0].set_title('Somatosensory (S1 RH)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2) Visual RH & LH
    axes[1].plot(stc.times * 1000, visual_labels_ts_rh.T, 
                'b-', linewidth=3, label='V1 RH')
    axes[1].plot(stc.times * 1000, visual_labels_ts_lh.T, 
                'purple', linewidth=2, label='V1 LH', alpha=0.8)
    axes[1].axvline(0, color='k', ls='--', alpha=0.5)
    axes[1].set_ylabel('Amplitude (nAm)')
    axes[1].set_title('Visual Cortex (V1 bilateral)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Combined Visual signal
    visual_combined = (visual_labels_ts_rh + visual_labels_ts_lh) / 2
    axes[2].plot(stc.times * 1000, somatosensory_labels_ts.T, 
                'r-', linewidth=3, label='S1 RH')
    axes[2].plot(stc.times * 1000, visual_combined.T, 
                'b-', linewidth=3, label='V1 Mean')
    axes[2].axvline(0, color='k', ls='--', alpha=0.5)
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude (nAm)')
    axes[2].set_title('S1 vs V1 Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f'{figuredir}/v1_a3b_timecourses.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_labels(src_file, stc, somatosensory_label_names, somatosensory_labels, visual_label_names, visual_labels, modality, figuredir, subID='average', timewindow=None):
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # extract timecourse for somatosensory region labels
    somatosensory_labels_ts = [stc.extract_label_time_course(label, src=mne.read_source_spaces(src_file), mode='mean') for label in somatosensory_labels]
    combined_somatosensory_ts = np.mean(somatosensory_labels_ts, axis=0)
    visual_labels_ts = [stc.extract_label_time_course(label, src=mne.read_source_spaces(src_file), mode='mean') for label in visual_labels]
    combined_visual_ts = np.mean(visual_labels_ts, axis=0)

    # Create single figure with two subplots sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Plot somatosensory sources
    for ts, label_name in zip(somatosensory_labels_ts, somatosensory_label_names):
        ax1.plot(stc.times[start:stop], ts.T[start:stop], label=label_name)
    ax1.plot(stc.times[start:stop], combined_somatosensory_ts.T[start:stop], label='Average', linewidth=2, color='black')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.set_title('Time Courses for Somatosensory Cortices')

    # Plot visual sources 
    for ts, label_name in zip(visual_labels_ts, visual_label_names):
        ax2.plot(stc.times[start:stop], ts.T[start:stop], label=label_name)
    ax2.plot(stc.times[start:stop], combined_visual_ts.T[start:stop], label='Average', linewidth=2, color='black')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.set_title('Time Courses for Visual Cortices')

    sns.despine(trim=True)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_somatosensory_visual_timecourses_lcmv_cond-Yes.png'))
    plt.show()


def plot_condition_comparison(stc_yes, stc_no, src_file, somatosensory_label_names, somatosensory_labels, visual_label_names, visual_labels, modality, figuredir, subID='average', timewindow=None):
    """
    Plot comparison between Yes and No conditions for each brain area.
    
    Parameters:
    -----------
    stc_yes : SourceEstimate
        Source estimate for 'Yes' condition
    stc_no : SourceEstimate  
        Source estimate for 'No' condition
    src_file : str
        Path to source space file
    somatosensory_label_names : list
        Names of somatosensory labels
    somatosensory_labels : list
        List of somatosensory label objects
    visual_label_names : list
        Names of visual labels
    visual_label_names : list
        List of visual label objects
    modality : str
        Modality (e.g., 'mecha', 'elec')
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    src = mne.read_source_spaces(src_file)
    
    # Extract time courses for both conditions
    # Somatosensory regions
    somatosensory_ts_yes = [stc_yes.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    somatosensory_ts_no = [stc_no.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    
    # Visual regions  
    visual_ts_yes = [stc_yes.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    visual_ts_no = [stc_no.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    
    # Create figure with subplots for each brain area
    n_areas = len(somatosensory_label_names) + len(visual_label_names)
    n_cols = 4  # Arrange in grid
    n_rows = (n_areas + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot somatosensory areas
    for i, (ts_yes, ts_no, label_name) in enumerate(zip(somatosensory_ts_yes, somatosensory_ts_no, somatosensory_label_names)):
        ax = axes_flat[i]
        ax.plot(stc_yes.times[start:stop], ts_yes.T[start:stop], '-', color=colors['perceived'], linewidth=2, label='Yes', alpha=0.8)
        ax.plot(stc_no.times[start:stop], ts_no.T[start:stop], '-', color=colors['unperceived'], linewidth=2, label='No', alpha=0.8)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f'{label_name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot visual areas
    for i, (ts_yes, ts_no, label_name) in enumerate(zip(visual_ts_yes, visual_ts_no, visual_label_names)):
        ax = axes_flat[len(somatosensory_label_names) + i]
        ax.plot(stc_yes.times[start:stop], ts_yes.T[start:stop], '-', color=colors['perceived'], linewidth=2, label='Yes', alpha=0.8)
        ax.plot(stc_no.times[start:stop], ts_no.T[start:stop], '-', color=colors['unperceived'], linewidth=2, label='No', alpha=0.8)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f'{label_name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_areas, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'Condition Comparison: Yes vs No ({identifier}, {modality})', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_condition_comparison_yes_vs_no.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create average plots for each region type
    # Somatosensory average
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    somatosensory_avg_yes = np.mean(somatosensory_ts_yes, axis=0)
    somatosensory_avg_no = np.mean(somatosensory_ts_no, axis=0)
    
    ax.plot(stc_yes.times[start:stop], somatosensory_avg_yes.T[start:stop], '-', color=colors['perceived'], linewidth=3, label='Yes (Somatosensory Avg)', alpha=0.8)
    ax.plot(stc_no.times[start:stop], somatosensory_avg_no.T[start:stop], '-', color=colors['unperceived'], linewidth=3, label='No (Somatosensory Avg)', alpha=0.8)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Somatosensory Regions Average - {identifier} ({modality})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(trim=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_somatosensory_avg_condition_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visual average
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    visual_avg_yes = np.mean(visual_ts_yes, axis=0)
    visual_avg_no = np.mean(visual_ts_no, axis=0)
    
    ax.plot(stc_yes.times[start:stop], visual_avg_yes.T[start:stop], '-', color=colors['perceived'], linewidth=3, label='Yes (Visual Avg)', alpha=0.8)
    ax.plot(stc_no.times[start:stop], visual_avg_no.T[start:stop], '-', color=colors['unperceived'], linewidth=3, label='No (Visual Avg)', alpha=0.8)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Visual Regions Average - {identifier} ({modality})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(trim=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_visual_avg_condition_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_vector_orientations(stc_vector, src_file, somatosensory_label_names, somatosensory_labels, visual_label_names, visual_labels, modality, figuredir, subID='average', timewindow=None):
    """
    Plot the three vector orientations (x, y, z components) for each brain area.
    
    Parameters:
    -----------
    stc_vector : VectorSourceEstimate
        Source estimate with vector orientations (from pick_ori="vector")
    src_file : str
        Path to source space file
    somatosensory_label_names : list
        Names of somatosensory labels
    somatosensory_labels : list
        List of somatosensory label objects
    visual_label_names : list
        Names of visual labels
    visual_labels : list
        List of visual label objects
    modality : str
        Modality (e.g., 'mecha', 'elec')
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to plot (start, stop) in samples
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    src = mne.read_source_spaces(src_file)
    
    # Check if this is a vector source estimate
    if not isinstance(stc_vector, mne.VectorSourceEstimate):
        raise ValueError("Input source estimate must be a VectorSourceEstimate (from pick_ori='vector')")
    
    # Extract time courses for vector components (x, y, z) for both conditions
    # Somatosensory regions
    somatosensory_ts_vector = [stc_vector.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    
    # Visual regions  
    visual_ts_vector = [stc_vector.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    
    # Create figure with subplots for each brain area showing all three orientations
    n_areas = len(somatosensory_label_names) + len(visual_label_names)
    n_cols = 4  # Arrange in grid
    n_rows = (n_areas + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    colors_ori = ['#D55E00', '#009E73', '#0072B2']
    
    # Plot somatosensory areas with vector orientations
    for i, (ts_vector, label_name) in enumerate(zip(somatosensory_ts_vector, somatosensory_label_names)):
        ax = axes_flat[i]
        ts_vector = ts_vector.swapaxes(0, 1)
        
        # ts_vector has shape (n_orientations, n_vertices, n_times)
        # Extract x, y, z components (assuming they correspond to orientations 0, 1, 2)
        if ts_vector.shape[0] >= 3:
            # Average across vertices for each orientation
            x_component = np.mean(ts_vector[0, :, :], axis=0)
            y_component = np.mean(ts_vector[1, :, :], axis=0) 
            z_component = np.mean(ts_vector[2, :, :], axis=0)
            
            ax.plot(stc_vector.times[start:stop], x_component[start:stop], color=colors_ori[0], linewidth=2, label='X (Medial-Lateral)', alpha=0.8)
            ax.plot(stc_vector.times[start:stop], y_component[start:stop], color=colors_ori[1], linewidth=2, label='Y (Anterior-Posterior)', alpha=0.8)
            ax.plot(stc_vector.times[start:stop], z_component[start:stop], color=colors_ori[2], linewidth=2, label='Z (Inferior-Superior)', alpha=0.8)
        else:
            # Fallback if fewer than 3 orientations
            for orient_idx in range(ts_vector.shape[0]):
                component = np.mean(ts_vector[orient_idx, :, :], axis=0)
                ax.plot(stc_vector.times[start:stop], component[start:stop], linewidth=2, 
                       label=f'Orientation {orient_idx}', alpha=0.8)
        
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f'{label_name} - Vector Orientations')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot visual areas with vector orientations
    for i, (ts_vector, label_name) in enumerate(zip(visual_ts_vector, visual_label_names)):
        ax = axes_flat[len(somatosensory_label_names) + i]
        ts_vector = ts_vector.swapaxes(0, 1)
        
        # ts_vector has shape (n_orientations, n_vertices, n_times)
        # Extract x, y, z components (assuming they correspond to orientations 0, 1, 2)
        if ts_vector.shape[0] >= 3:
            # Average across vertices for each orientation
            x_component = np.mean(ts_vector[0, :, :], axis=0)
            y_component = np.mean(ts_vector[1, :, :], axis=0)
            z_component = np.mean(ts_vector[2, :, :], axis=0)
            
            ax.plot(stc_vector.times[start:stop], x_component[start:stop], color=colors_ori[0], linewidth=2, label='X (Medial-Lateral)', alpha=0.8)
            ax.plot(stc_vector.times[start:stop], y_component[start:stop], color=colors_ori[1], linewidth=2, label='Y (Anterior-Posterior)', alpha=0.8)
            ax.plot(stc_vector.times[start:stop], z_component[start:stop], color=colors_ori[2], linewidth=2, label='Z (Inferior-Superior)', alpha=0.8)
        else:
            # Fallback if fewer than 3 orientations
            for orient_idx in range(ts_vector.shape[0]):
                component = np.mean(ts_vector[orient_idx, :, :], axis=0)
                ax.plot(stc_vector.times[start:stop], component[start:stop], linewidth=2,
                       label=f'Orientation {orient_idx}', alpha=0.8)
        
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f'{label_name} - Vector Orientations')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_areas, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'Vector Orientations: X, Y, Z Components ({identifier}, {modality})', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_vector_orientations_xyz.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create average plots for each region type showing vector orientations
    # Somatosensory average
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if len(somatosensory_ts_vector) > 0 and somatosensory_ts_vector[0].shape[0] >= 3:
        # Average across all somatosensory regions for each orientation
        all_x = np.mean([np.mean(ts[:, 0, :], axis=0) for ts in somatosensory_ts_vector], axis=0)
        all_y = np.mean([np.mean(ts[:, 1, :], axis=0) for ts in somatosensory_ts_vector], axis=0)
        all_z = np.mean([np.mean(ts[:, 2, :], axis=0) for ts in somatosensory_ts_vector], axis=0)
        
        ax.plot(stc_vector.times[start:stop], all_x[start:stop], color=colors_ori[0], linewidth=3, label='X (Medial-Lateral)', alpha=0.8)
        ax.plot(stc_vector.times[start:stop], all_y[start:stop], color=colors_ori[1], linewidth=3, label='Y (Anterior-Posterior)', alpha=0.8)
        ax.plot(stc_vector.times[start:stop], all_z[start:stop], color=colors_ori[2], linewidth=3, label='Z (Inferior-Superior)', alpha=0.8)
    else:
        # Fallback for fewer orientations
        for orient_idx in range(somatosensory_ts_vector[0].shape[0]):
            avg_component = np.mean([np.mean(ts[:, orient_idx, :], axis=0) for ts in somatosensory_ts_vector], axis=0)
            ax.plot(stc_vector.times[start:stop], avg_component[start:stop], color=colors_ori[0], linewidth=3,
                   label=f'Orientation {orient_idx}', alpha=0.8)
    
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Somatosensory Regions Average - Vector Orientations - {identifier} ({modality})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(trim=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_somatosensory_avg_vector_orientations.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visual average
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if len(visual_ts_vector) > 0 and visual_ts_vector[0].shape[0] >= 3:
        # Average across all visual regions for each orientation
        all_x = np.mean([np.mean(ts[0, :, :], axis=0) for ts in visual_ts_vector], axis=0)
        all_y = np.mean([np.mean(ts[1, :, :], axis=0) for ts in visual_ts_vector], axis=0)
        all_z = np.mean([np.mean(ts[2, :, :], axis=0) for ts in visual_ts_vector], axis=0)
        
        ax.plot(stc_vector.times[start:stop], all_x[start:stop], color=colors_ori[0], linewidth=3, label='X (Medial-Lateral)', alpha=0.8)
        ax.plot(stc_vector.times[start:stop], all_y[start:stop], color=colors_ori[1], linewidth=3, label='Y (Anterior-Posterior)', alpha=0.8)
        ax.plot(stc_vector.times[start:stop], all_z[start:stop], color=colors_ori[2], linewidth=3, label='Z (Inferior-Superior)', alpha=0.8)
    else:
        # Fallback for fewer orientations
        for orient_idx in range(visual_ts_vector[0].shape[0]):
            avg_component = np.mean([np.mean(ts[:, orient_idx, :], axis=0) for ts in visual_ts_vector], axis=0)
            ax.plot(stc_vector.times[start:stop], avg_component[start:stop], linewidth=3,
                   label=f'Orientation {orient_idx}', alpha=0.8)
    
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Visual Regions Average - Vector Orientations - {identifier} ({modality})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(trim=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figuredir, f'{identifier}_ses-{modality}_visual_avg_vector_orientations.png'), 
                dpi=300, bbox_inches='tight')
    plt.show() 


def plot_vector_svd_components(stc_vector, src_file, labels, label_names, modality, figuredir, subID='average', timewindow=None):
    """
    Perform SVD on vector source reconstruction data for each region/label and plot the first 4 components.
    
    Parameters:
    -----------
    stc_vector : VectorSourceEstimate
        Source estimate with vector orientations (from pick_ori="vector")
    src_file : str
        Path to source space file
    labels : list
        List of label objects for regions of interest
    label_names : list
        Names of the labels
    modality : str
        Modality (e.g., 'mecha', 'elec')
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to analyze (start, stop) in samples
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    src = mne.read_source_spaces(src_file)
    
    # Check if this is a vector source estimate
    if not isinstance(stc_vector, mne.VectorSourceEstimate):
        raise ValueError("Input source estimate must be a VectorSourceEstimate (from pick_ori='vector')")
    
    # Create figure with subplots for each brain area showing SVD components
    n_areas = len(label_names)
    n_cols = 4  # Arrange in grid
    n_rows = (n_areas + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Process each label
    for i, (label, label_name) in enumerate(zip(labels, label_names)):
        
        ax = axes_flat[i]
        
        # Extract time course for this label
        label_vertices = label.get_vertices_used(np.arange(20484))
        ts_vector = stc_vector.data[label_vertices]

        # ts_vector has shape (n_orientations, n_vertices, n_times)
        # Reshape to (n_vertices, n_orientations * n_times) for SVD
        n_vertices, n_orientations, n_times = ts_vector.shape
        data_matrix = ts_vector.reshape(n_vertices, n_orientations * n_times)
        
        # Perform SVD
        u, s, v = np.linalg.svd(data_matrix)

        # Calculate variance explained
        variance_explained = (s**2) / np.sum(s**2)

        # Number of components to analyze
        n_components = 4

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))

        # 1. Scree plot - variance explained
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(variance_explained[:20], 'o-')
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title('Variance explained per Component')
        ax1.grid(True)
        ax1.axvline(x=n_components-1, color='r', linestyle='--', alpha=0.5)

        # 2. Cumulative variance explained
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(np.cumsum(variance_explained[:20]), 'o-')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Cumulative Variance Explained')
        ax2.set_title('Cumulative Variance')
        ax2.grid(True)
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
        ax2.legend()

        # 3-6. Temporal patterns for first 4 components
        for i in range(n_components):
            ax = plt.subplot(3, 2, i+3)
            ax.plot(stc_vector.times, v[i, :])
            ax.set_xlabel('Time Points')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Component {i+1} Temporal Pattern\n({variance_explained[i]*100:.2f}% variance)')
            ax.grid(True)

        plt.suptitle(f'Label: {label_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuredir, f'{label_name}svd_temporal_patterns.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Separate figure for spatial patterns
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i in range(n_components):
            # Reshape spatial pattern back to (n_vertices, n_orientations) if needed
            if data_matrix.shape[0] == n_vertices * n_orientations:
                spatial_pattern = u[:, i].reshape(n_vertices, n_orientations)
                # Plot norm across orientations
                spatial_norm = np.linalg.norm(spatial_pattern, axis=1)
            else:
                spatial_norm = u[:, i]
            
            axes[i].plot(spatial_norm)
            axes[i].set_xlabel('Vertex Index')
            axes[i].set_ylabel('Amplitude')
            axes[i].set_title(f'Component {i+1} Spatial Pattern')
            axes[i].grid(True)

        plt.suptitle(f'Label: {label_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuredir, f'{label_name}svd_spatial_patterns.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary statistics
        print('label', label_name)
        print("=" * 50)
        print("SVD Summary")
        print("=" * 50)
        print(f"Data shape: {data_matrix.shape}")
        print(f"Number of singular values: {len(s)}")
        print(f"\nFirst {n_components} components:")
        for i in range(n_components):
            print(f"  Component {i+1}: {variance_explained[i]*100:.2f}% variance")
        print(f"\nCumulative variance (first {n_components}): {np.sum(variance_explained[:n_components])*100:.2f}%")


def plot_vector_svd_components_extended(stc_vector, src_file, labels, label_names, modality, figuredir, subID='average', timewindow=None):
    """
    Perform SVD on vector source reconstruction data for each region/label and plot the first 4 components.
    
    Parameters:
    -----------
    stc_vector : VectorSourceEstimate
        Source estimate with vector orientations (from pick_ori="vector")
    src_file : str
        Path to source space file
    labels : list
        List of label objects for regions of interest
    label_names : list
        Names of the labels
    modality : str
        Modality (e.g., 'mecha', 'elec')
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to analyze (start, stop) in samples
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    src = mne.read_source_spaces(src_file)
    
    # Check if this is a vector source estimate
    if not isinstance(stc_vector, mne.VectorSourceEstimate):
        raise ValueError("Input source estimate must be a VectorSourceEstimate (from pick_ori='vector')")
    
    # Process each label
    for label_idx, (label, label_name) in enumerate(zip(labels, label_names)):
        
        # Extract time course for this label
        label_vertices = label.get_vertices_used(np.arange(20484))
        ts_vector = stc_vector.data[label_vertices]

        # ts_vector has shape (n_vertices, n_orientations, n_times)
        n_vertices, n_orientations, n_times = ts_vector.shape
        
        print(f"\n{'='*70}")
        print(f"Processing label: {label_name}")
        print(f"Data shape: ({n_vertices} vertices, {n_orientations} orientations, {n_times} time points)")
        print(f"{'='*70}")
        
        # ====================================================================
        # METHOD 1: Reshape to (n_vertices * n_orientations, n_times)
        # This treats each vertex-orientation pair as a separate "spatial" feature
        # Good for: Finding temporal patterns that are consistent across space
        # ====================================================================
        data_spatial_time = ts_vector.reshape(n_vertices * n_orientations, n_times)
        u1, s1, v1 = np.linalg.svd(data_spatial_time, full_matrices=False)
        variance_explained_1 = (s1**2) / np.sum(s1**2)
        
        # ====================================================================
        # METHOD 2: Reshape to (n_vertices, n_orientations * n_times)
        # This treats each vertex as a separate feature with mixed time-orientation
        # Good for: Finding spatial patterns across vertices
        # ====================================================================
        data_vertex_timeorien = ts_vector.reshape(n_vertices, n_orientations * n_times)
        u2, s2, v2 = np.linalg.svd(data_vertex_timeorien, full_matrices=False)
        variance_explained_2 = (s2**2) / np.sum(s2**2)

        n_components = 4

        # ====================================================================
        # FIGURE 1: METHOD 1 - Spatial-Temporal decomposition
        # ====================================================================
        fig1 = plt.figure(figsize=(20, 12))
        
        # 1. Scree plot
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(variance_explained_1[:20], 'o-')
        ax1.set_xlabel('Component', fontsize=11)
        ax1.set_ylabel('Variance Explained', fontsize=11)
        ax1.set_title('Method 1: Variance per Component', fontsize=12, fontweight='bold')
        ax1.grid(True)
        ax1.axvline(x=n_components-1, color='r', linestyle='--', alpha=0.5)

        # 2. Cumulative variance
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(np.cumsum(variance_explained_1[:20]), 'o-')
        ax2.set_xlabel('Component', fontsize=11)
        ax2.set_ylabel('Cumulative Variance', fontsize=11)
        ax2.set_title('Cumulative Variance', fontsize=12, fontweight='bold')
        ax2.grid(True)
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
        ax2.legend(fontsize=11)

        # 3. Comparison of both methods
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(variance_explained_1[:10], 'o-', label='Method 1 (Space-Time)', linewidth=2)
        ax3.plot(variance_explained_2[:10], 's-', label='Method 2 (Vertex-TimeOrient)', linewidth=2)
        ax3.set_xlabel('Component', fontsize=11)
        ax3.set_ylabel('Variance Explained', fontsize=11)
        ax3.set_title('Methods Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True)

        # 4-7. Temporal patterns for first 4 components (Method 1)
        for i in range(n_components):
            ax = plt.subplot(3, 3, i+4)
            # v1[i, :] has shape (n_times,) - pure temporal pattern
            ax.plot(stc_vector.times[start:stop], v1[i, start:stop], linewidth=2)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(f'Component {i+1} Temporal\n({variance_explained_1[i]*100:.2f}% var)', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='k', ls='--', alpha=0.3)

        plt.suptitle(f'Spatial-Temporal patterns, SVD of  (n_vertices * n_orientations, n_times) - {label_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuredir, f'{label_name}_method1_temporal_patterns.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

        # ====================================================================
        # FIGURE 2: METHOD 1 - Spatial patterns
        # ====================================================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
        axes2 = axes2.ravel()

        for i in range(n_components):
            # u1[:, i] has shape (n_vertices * n_orientations,)
            # Reshape back to (n_vertices, n_orientations)
            spatial_pattern = u1[:, i].reshape(n_vertices, n_orientations)
            
            # Plot norm across orientations
            spatial_norm = np.linalg.norm(spatial_pattern, axis=1)
            
            axes2[i].plot(spatial_norm, linewidth=2)
            axes2[i].set_xlabel('Vertex Index', fontsize=11)
            axes2[i].set_ylabel('Amplitude (norm)', fontsize=11)
            axes2[i].set_title(f'Component {i+1} Spatial Pattern\n({variance_explained_1[i]*100:.2f}% var)', 
                             fontsize=12, fontweight='bold')
            axes2[i].grid(True, alpha=0.3)

        plt.suptitle(f'Spatial Patterns (n_vertices * n_orientations, n_times) - {label_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuredir, f'{label_name}_method1_spatial_patterns.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

        # ====================================================================
        # FIGURE 3: METHOD 2 - Temporal-Orientation patterns
        # ====================================================================
        fig3 = plt.figure(figsize=(20, 12))
        
        # Colorblind-friendly palette for orientations
        orient_colors = ['#D55E00', '#009E73', '#0072B2']  # Vermillion, Bluish green, Blue
        orient_labels = ['X (lateral)', 'Y (anterior)', 'Z (superior)']
        
        for i in range(n_components):
            ax = plt.subplot(2, 2, i+1)
            
            # v2[i, :] has shape (n_orientations * n_times,)
            # Reshape to (n_orientations, n_times)
            temporal_orient_pattern = v2[i, :].reshape(n_orientations, n_times)
            
            # Plot each orientation
            for orient in range(n_orientations):
                ax.plot(stc_vector.times[start:stop], 
                       temporal_orient_pattern[orient, start:stop], 
                       color=orient_colors[orient],
                       label=orient_labels[orient], 
                       linewidth=2, 
                       alpha=0.8)
            
            # Also plot the magnitude
            temporal_magnitude = np.linalg.norm(temporal_orient_pattern[:, start:stop], axis=0)
            ax.plot(stc_vector.times[start:stop], temporal_magnitude, 
                   'k--', linewidth=2.5, label='Magnitude', alpha=0.9)
            
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.set_title(f'Component {i+1} Temporal-Orientation\n({variance_explained_2[i]*100:.2f}% var)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='k', ls='--', alpha=0.3)

        plt.suptitle(f'Temporal-Orientation Patterns, SVD of (n_vertices, n_orientations * n_times) - {label_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuredir, f'{label_name}_method2_temporal_orient_patterns.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

        # ====================================================================
        # FIGURE 4: METHOD 2 - Spatial patterns across vertices
        # ====================================================================
        fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))
        axes4 = axes4.ravel()

        for i in range(n_components):
            # u2[:, i] has shape (n_vertices,) - spatial pattern across vertices
            axes4[i].plot(u2[:, i], linewidth=2)
            axes4[i].set_xlabel('Vertex Index', fontsize=11)
            axes4[i].set_ylabel('Amplitude', fontsize=11)
            axes4[i].set_title(f'Component {i+1} Vertex Pattern\n({variance_explained_2[i]*100:.2f}% var)', 
                             fontsize=12, fontweight='bold')
            axes4[i].grid(True, alpha=0.3)

        plt.suptitle(f'Method 2: Spatial (Vertex) Patterns - {label_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuredir, f'{label_name}_method2_spatial_patterns.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

        # ====================================================================
        # Print summary statistics
        # ====================================================================
        print(f"\nLabel: {label_name}")
        print("=" * 70)
        print("METHOD 1: (n_vertices * n_orientations, n_times)")
        print("=" * 70)
        print(f"Data shape: {data_spatial_time.shape}")
        print(f"Number of singular values: {len(s1)}")
        print(f"\nFirst {n_components} components:")
        for i in range(n_components):
            print(f"  Component {i+1}: {variance_explained_1[i]*100:.2f}% variance")
        print(f"Cumulative variance (first {n_components}): {np.sum(variance_explained_1[:n_components])*100:.2f}%")
        
        print("\n" + "=" * 70)
        print("METHOD 2: (n_vertices, n_orientations * n_times)")
        print("=" * 70)
        print(f"Data shape: {data_vertex_timeorien.shape}")
        print(f"Number of singular values: {len(s2)}")
        print(f"\nFirst {n_components} components:")
        for i in range(n_components):
            print(f"  Component {i+1}: {variance_explained_2[i]*100:.2f}% variance")
        print(f"Cumulative variance (first {n_components}): {np.sum(variance_explained_2[:n_components])*100:.2f}%")
        print("=" * 70 + "\n")


def plot_condition_differences(stc_yes, stc_no, src_file, somatosensory_label_names, somatosensory_labels,
                               visual_label_names, visual_labels, modality, figuredir, 
                               subID='average', timewindow=None):
    """
    Plot the difference per condition for left and right hemisphere for visual and somatosensory signals
    (separately but averaged within areas).
    
    Parameters:
    -----------
    stc_yes : SourceEstimate
        Source estimate for 'Yes' condition
    stc_no : SourceEstimate  
        Source estimate for 'No' condition
    src_file : str
        Path to source space file
    visual_labels : list
        List of visual region labels
    visual_label_names : list
        Names of visual labels
    somatosensory_labels : list
        List of somatosensory region labels
    somatosensory_label_names : list
        Names of somatosensory labels
    modality : str
        Modality (e.g., 'mecha', 'elec')
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to analyze (start, stop) in samples
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    src = mne.read_source_spaces(src_file)
    
    # Extract time courses for both conditions
    # Visual regions
    visual_ts_yes = [stc_yes.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    visual_ts_no = [stc_no.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    
    # Somatosensory regions  
    somatosensory_ts_yes = [stc_yes.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    somatosensory_ts_no = [stc_no.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    
    # Separate left and right hemisphere signals
    # Visual: indices 0,1 = LH, 2,3 = RH
    visual_lh_ts_yes = np.mean(visual_ts_yes[0:2], axis=0)  # Average LH V1, V2
    visual_rh_ts_yes = np.mean(visual_ts_yes[2:4], axis=0)  # Average RH V1, V2
    visual_lh_ts_no = np.mean(visual_ts_no[0:2], axis=0)    # Average LH V1, V2
    visual_rh_ts_no = np.mean(visual_ts_no[2:4], axis=0)    # Average RH V1, V2
    
    # Somatosensory: indices 0-3 = LH, 4-7 = RH  
    somatosensory_lh_ts_yes = np.mean(somatosensory_ts_yes[0:4], axis=0)  # Average LH BA1, BA2, BA3a, BA3b
    somatosensory_rh_ts_yes = np.mean(somatosensory_ts_yes[4:8], axis=0)  # Average RH BA1, BA2, BA3a, BA3b
    somatosensory_lh_ts_no = np.mean(somatosensory_ts_no[0:4], axis=0)    # Average LH BA1, BA2, BA3a, BA3b
    somatosensory_rh_ts_no = np.mean(somatosensory_ts_no[4:8], axis=0)    # Average RH BA1, BA2, BA3a, BA3b
    
    # Compute condition differences (Yes - No)
    visual_lh_diff = visual_lh_ts_yes - visual_lh_ts_no
    visual_rh_diff = visual_rh_ts_yes - visual_rh_ts_no
    somatosensory_lh_diff = somatosensory_lh_ts_yes - somatosensory_lh_ts_no
    somatosensory_rh_diff = somatosensory_rh_ts_yes - somatosensory_rh_ts_no
    
    times = stc_yes.times[start:stop]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Perception condition by Hemisphere - {identifier} ({modality})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Visual Cortex (LH vs RH)
    ax1 = axes[0]
    ax1.plot(times * 1000, visual_lh_ts_yes.T[start:stop], 
             linewidth=3, color=colors['perceived'], ls='--', label='Perceived Left Hemi')
    ax1.plot(times * 1000, visual_rh_ts_yes.T[start:stop], 
             linewidth=3, color=colors['perceived'], ls='-', label='Perceived Right Hemi')
    ax1.plot(times * 1000, visual_lh_ts_no.T[start:stop], 
             linewidth=3, color=colors['unperceived'], ls='--', label='Unperceived Left Hemi')
    ax1.plot(times * 1000, visual_rh_ts_no.T[start:stop], 
             linewidth=3, color=colors['unperceived'], ls='-', label='Unperceived Right Hemi')
    ax1.axvline(0, color='k', ls='--', alpha=0.5, linewidth=2)
    ax1.axhline(0, color='k', ls='-', alpha=0.3)
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Amplitude Difference (nAm)', fontsize=12)
    ax1.set_title('Visual Cortex (V1/V2)', fontsize=14, fontweight='bold')
    ax1.set_ylim([-0.1, 1])
    ax1.set_xlim(times[0] * 1000, times[-1] * 1000)
    
    # Plot 2: Somatosensory Cortex (LH vs RH)
    ax2 = axes[1]
    ax2.plot(times * 1000, somatosensory_lh_ts_yes.T[start:stop], 
             linewidth=3, color=colors['perceived'], ls='--', label='Perceived Left Hemi')
    ax2.plot(times * 1000, somatosensory_rh_ts_yes.T[start:stop], 
             linewidth=3, color=colors['perceived'], ls='-', label='Perceived Right Hemi')
    ax2.plot(times * 1000, somatosensory_lh_ts_no.T[start:stop], 
             linewidth=3, color=colors['unperceived'], ls='--', label='Unperceived Left Hemi')
    ax2.plot(times * 1000, somatosensory_rh_ts_no.T[start:stop], 
             linewidth=3, color=colors['unperceived'], ls='-', label='Unperceived Right Hemi')
    ax2.axvline(0, color='k', ls='--', alpha=0.5, linewidth=2)
    ax2.axhline(0, color='k', ls='-', alpha=0.3)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Amplitude Difference (nAm)', fontsize=12)
    ax2.set_title('Somatosensory Cortex (BA1/BA2/BA3a/BA3b)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim([-0.1, 1])
    ax2.set_xlim(times[0] * 1000, times[-1] * 1000)
    sns.despine(trim=True)
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_condition_perhemi.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filename}")
    plt.show()


def plot_condition_differences_stats(stc_yes, stc_no, src_file, visual_labels, visual_label_names, 
                                     somatosensory_labels, somatosensory_label_names, modality, figuredir, 
                                     subID='average', timewindow=None):
    """
    Plot statistical summary of condition differences.
    
    Parameters:
    -----------
    stc_yes : SourceEstimate
        Source estimate for 'Yes' condition
    stc_no : SourceEstimate  
        Source estimate for 'No' condition
    src_file : str
        Path to source space file
    visual_labels : list
        List of visual region labels
    visual_label_names : list
        Names of visual labels
    somatosensory_labels : list
        List of somatosensory region labels
    somatosensory_label_names : list
        Names of somatosensory labels
    modality : str
        Modality (e.g., 'mecha', 'elec')
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to analyze (start, stop) in samples
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    src = mne.read_source_spaces(src_file)
    
    # Extract time courses for both conditions
    # Visual regions
    visual_ts_yes = [stc_yes.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    visual_ts_no = [stc_no.extract_label_time_course(label, src=src, mode='mean') for label in visual_labels]
    
    # Somatosensory regions  
    somatosensory_ts_yes = [stc_yes.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    somatosensory_ts_no = [stc_no.extract_label_time_course(label, src=src, mode='mean') for label in somatosensory_labels]
    
    # Separate left and right hemisphere signals
    # Visual: indices 0,1 = LH, 2,3 = RH
    visual_lh_ts_yes = np.mean(visual_ts_yes[0:2], axis=0)  # Average LH V1, V2
    visual_rh_ts_yes = np.mean(visual_ts_yes[2:4], axis=0)  # Average RH V1, V2
    visual_lh_ts_no = np.mean(visual_ts_no[0:2], axis=0)    # Average LH V1, V2
    visual_rh_ts_no = np.mean(visual_ts_no[2:4], axis=0)    # Average RH V1, V2
    
    # Somatosensory: indices 0-3 = LH, 4-7 = RH  
    somatosensory_lh_ts_yes = np.mean(somatosensory_ts_yes[0:4], axis=0)  # Average LH BA1, BA2, BA3a, BA3b
    somatosensory_rh_ts_yes = np.mean(somatosensory_ts_yes[4:8], axis=0)  # Average RH BA1, BA2, BA3a, BA3b
    somatosensory_lh_ts_no = np.mean(somatosensory_ts_no[0:4], axis=0)    # Average LH BA1, BA2, BA3a, BA3b
    somatosensory_rh_ts_no = np.mean(somatosensory_ts_no[4:8], axis=0)    # Average RH BA1, BA2, BA3a, BA3b
    
    # Compute condition differences (Yes - No)
    visual_lh_diff = visual_lh_ts_yes - visual_lh_ts_no
    visual_rh_diff = visual_rh_ts_yes - visual_rh_ts_no
    somatosensory_lh_diff = somatosensory_lh_ts_yes - somatosensory_lh_ts_no
    somatosensory_rh_diff = somatosensory_rh_ts_yes - somatosensory_rh_ts_no
    
    times = stc_yes.times[start:stop]
    
    # Calculate statistics
    stats = {}
    for key, signal in [('visual_lh', visual_lh_diff), ('visual_rh', visual_rh_diff),
                       ('somatosensory_lh', somatosensory_lh_diff), ('somatosensory_rh', somatosensory_rh_diff)]:
        signal_flat = signal.flatten()
        stats[key] = {
            'mean': np.mean(signal_flat),
            'std': np.std(signal_flat),
            'max': np.max(signal_flat),
            'min': np.min(signal_flat),
            'abs_max': np.max(np.abs(signal_flat))
        }
    
    # Create figure with statistical summary
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    fig.suptitle(f'Statistical Summary - Condition Differences ({identifier}, {modality})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Bar plot of mean differences
    ax1 = axes[0, 0]
    regions = ['Visual LH', 'Visual RH', 'Somatosensory LH', 'Somatosensory RH']
    means = [stats['visual_lh']['mean'], stats['visual_rh']['mean'], 
             stats['somatosensory_lh']['mean'], stats['somatosensory_rh']['mean']]
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax1.bar(regions, means, color=colors_bar, alpha=0.7)
    ax1.set_ylabel('Mean Difference (nAm)', fontsize=12)
    ax1.set_title('Mean Condition Differences by Region', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='k', ls='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Heatmap of correlation between regions
    ax2 = axes[0, 1]
    signals_matrix = np.array([
        visual_lh_diff.flatten(),
        visual_rh_diff.flatten(),
        somatosensory_lh_diff.flatten(),
        somatosensory_rh_diff.flatten()
    ])
    
    correlation_matrix = np.corrcoef(signals_matrix)
    im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.set_yticklabels(regions)
    ax2.set_title('Correlation Matrix - Condition Differences', fontsize=14, fontweight='bold')
    
    # Add correlation values to heatmap
    for i in range(4):
        for j in range(4):
            text = ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax2, label='Correlation')
    
    # 3. Time course of absolute differences
    ax3 = axes[1, 0]
    ax3.plot(times * 1000, np.abs(visual_lh_diff.T), color='#1f77b4', 
             linewidth=2, label='Visual LH', alpha=0.8)
    ax3.plot(times * 1000, np.abs(visual_rh_diff.T), color='#ff7f0e', 
             linewidth=2, label='Visual RH', alpha=0.8)
    ax3.plot(times * 1000, np.abs(somatosensory_lh_diff.T), color='#2ca02c', 
             linewidth=2, label='Somatosensory LH', alpha=0.8)
    ax3.plot(times * 1000, np.abs(somatosensory_rh_diff.T), color='#d62728', 
             linewidth=2, label='Somatosensory RH', alpha=0.8)
    ax3.axvline(0, color='k', ls='--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Absolute Difference (nAm)', fontsize=12)
    ax3.set_title('Absolute Condition Differences Over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(times[0] * 1000, times[-1] * 1000)
    
    # 4. Box plot of difference distributions
    ax4 = axes[1, 1]
    data_for_box = [
        visual_lh_diff.flatten(),
        visual_rh_diff.flatten(),
        somatosensory_lh_diff.flatten(),
        somatosensory_rh_diff.flatten()
    ]
    
    bp = ax4.boxplot(data_for_box, labels=regions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_bar):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Difference Amplitude (nAm)', fontsize=12)
    ax4.set_title('Distribution of Condition Differences', fontsize=14, fontweight='bold')
    ax4.axhline(0, color='k', ls='-', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_condition_differences_stats.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Statistical summary saved to: {filename}")
    plt.show()
    
    # Print numerical summary
    print(f"\n{'='*60}")
    print(f"STATISTICAL SUMMARY - {identifier} ({modality})")
    print(f"{'='*60}")
    print(f"{'Region':<20} {'Mean':<10} {'Std':<10} {'Max':<10} {'Min':<10}")
    print(f"{'-'*60}")
    for key, name in [('visual_lh', 'Visual LH'), ('visual_rh', 'Visual RH'),
                     ('somatosensory_lh', 'Somatosensory LH'), ('somatosensory_rh', 'Somatosensory RH')]:
        s = stats[key]
        print(f"{name:<20} {s['mean']:<10.3f} {s['std']:<10.3f} {s['max']:<10.3f} {s['min']:<10.3f}")
    print(f"{'='*60}")


def plot_label_quadrants(stc, src, label, label_name, figuredir, subID='average', timewindow=None):
    """
    Split a label into 4 quadrants and plot time courses for each quadrant.
    
    Parameters:
    -----------
    stc : SourceEstimate or VectorSourceEstimate
        Source estimate data
    src : SourceSpaces
        Source space information
    label : Label
        Brain region label to split into quadrants
    label_name : str
        Name of the label for plotting
    figuredir : str
        Directory to save figures
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to analyze (start, stop) in samples
    """
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1

    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"

    # Load source space
    if isinstance(src, str):
        src = mne.read_source_spaces(src)
    
    # Get label vertices and their coordinates
    label_vertices = label.get_vertices_used(np.arange(20484))
    
    # Get vertex coordinates from source space
    hemi_idx = 0 if label.hemi == 'lh' else 1
    vertex_coords = src[hemi_idx]['rr'][label_vertices]
    
    print(f"\nProcessing label: {label_name}")
    print(f"Number of vertices: {len(label_vertices)}")
    print(f"Hemisphere: {label.hemi}")
    
    # Calculate center of mass for the label
    center_of_mass = np.mean(vertex_coords, axis=0)
    print(f"Center of mass: {center_of_mass}")
    
    # Split vertices into 4 quadrants based on X and Y coordinates relative to center
    # Quadrant definitions (assuming standard brain coordinate system):
    # Q1: X > center_X, Y > center_Y (Anterior-Lateral)
    # Q2: X < center_X, Y > center_Y (Anterior-Medial) 
    # Q3: X < center_X, Y < center_Y (Posterior-Medial)
    # Q4: X > center_X, Y < center_Y (Posterior-Lateral)
    
    x_coords = vertex_coords[:, 0]
    y_coords = vertex_coords[:, 1]
    
    # Create quadrant masks
    q1_mask = (x_coords > center_of_mass[0]) & (y_coords > center_of_mass[1])  # Anterior-Lateral
    q2_mask = (x_coords < center_of_mass[0]) & (y_coords > center_of_mass[1])  # Anterior-Medial
    q3_mask = (x_coords < center_of_mass[0]) & (y_coords < center_of_mass[1])  # Posterior-Medial
    q4_mask = (x_coords > center_of_mass[0]) & (y_coords < center_of_mass[1])  # Posterior-Lateral
    
    # Get vertices for each quadrant
    q1_vertices = label_vertices[q1_mask]
    q2_vertices = label_vertices[q2_mask]
    q3_vertices = label_vertices[q3_mask]
    q4_vertices = label_vertices[q4_mask]
    
    print(f"Quadrant 1 (Anterior-Lateral): {len(q1_vertices)} vertices")
    print(f"Quadrant 2 (Anterior-Medial): {len(q2_vertices)} vertices")
    print(f"Quadrant 3 (Posterior-Medial): {len(q3_vertices)} vertices")
    print(f"Quadrant 4 (Posterior-Lateral): {len(q4_vertices)} vertices")
    
    # Extract time courses for each quadrant
    if isinstance(stc, mne.VectorSourceEstimate):
        # For vector source estimates
        q1_ts = stc.data[q1_vertices]
        q2_ts = stc.data[q2_vertices]
        q3_ts = stc.data[q3_vertices]
        q4_ts = stc.data[q4_vertices]
        
        # Average across vertices and orientations
        q1_avg = np.mean(np.linalg.norm(q1_ts, axis=1), axis=0)
        q2_avg = np.mean(np.linalg.norm(q2_ts, axis=1), axis=0)
        q3_avg = np.mean(np.linalg.norm(q3_ts, axis=1), axis=0)
        q4_avg = np.mean(np.linalg.norm(q4_ts, axis=1), axis=0)
        
        ylabel = 'Amplitude (nAm)'
    else:
        # For regular source estimates
        q1_ts = stc.data[q1_vertices]
        q2_ts = stc.data[q2_vertices]
        q3_ts = stc.data[q3_vertices]
        q4_ts = stc.data[q4_vertices]
        
        # Average across vertices
        q1_avg = np.mean(q1_ts, axis=0)
        q2_avg = np.mean(q2_ts, axis=0)
        q3_avg = np.mean(q3_ts, axis=0)
        q4_avg = np.mean(q4_ts, axis=0)
        
        ylabel = 'Amplitude (nAm)'
    
    # Create figure with 2x2 grid for quadrants
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{label_name} - Quadrant Analysis', fontsize=16, fontweight='bold')
    
    # Color scheme for quadrants
    colors = ['#D55E00', '#009E73', '#0072B2', '#CC79A7']  # Colorblind-friendly
    quadrant_names = ['Q1: Anterior-Lateral', 'Q2: Anterior-Medial', 
                     'Q3: Posterior-Medial', 'Q4: Posterior-Lateral']
    
    # Plot each quadrant
    quadrants = [(q1_avg, 'Q1\nAnterior-Lateral'), (q2_avg, 'Q2\nAnterior-Medial'),
                (q3_avg, 'Q3\nPosterior-Medial'), (q4_avg, 'Q4\nPosterior-Lateral')]
    
    for i, (quadrant_data, quadrant_label) in enumerate(quadrants):
        ax = axes[i // 2, i % 2]
        
        ax.plot(stc.times[start:stop], quadrant_data[start:stop], 
               color=colors[i], linewidth=3, alpha=0.8)
        ax.set_title(quadrant_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        
        # Add statistics text
        mean_amp = np.mean(np.abs(quadrant_data[start:stop]))
        max_amp = np.max(np.abs(quadrant_data[start:stop]))
        ax.text(0.05, 0.95, f'Mean: {mean_amp:.2f}\nMax: {max_amp:.2f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(figuredir, f'{label_name}_quadrant_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary plot comparing all quadrants
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for i, (quadrant_data, quadrant_label) in enumerate(quadrants):
        ax.plot(stc.times[start:stop], quadrant_data[start:stop], 
               color=colors[i], linewidth=3, label=quadrant_label, alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{label_name} - All Quadrants Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figuredir, f'{label_name}_quadrant_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create spatial visualization of quadrants
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot vertex coordinates colored by quadrant
    scatter = ax.scatter(vertex_coords[:, 0], vertex_coords[:, 1], 
                        c=[colors[0]] * len(q1_vertices) + 
                          [colors[1]] * len(q2_vertices) + 
                          [colors[2]] * len(q3_vertices) + 
                          [colors[3]] * len(q4_vertices),
                        s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add center of mass
    ax.plot(center_of_mass[0], center_of_mass[1], 'k+', markersize=15, linewidth=3, label='Center of Mass')
    
    # Add quadrant labels
    ax.text(center_of_mass[0] + 0.01, center_of_mass[1] + 0.01, 'Q1', fontsize=14, fontweight='bold', ha='left', va='bottom')
    ax.text(center_of_mass[0] - 0.01, center_of_mass[1] + 0.01, 'Q2', fontsize=14, fontweight='bold', ha='right', va='bottom')
    ax.text(center_of_mass[0] - 0.01, center_of_mass[1] - 0.01, 'Q3', fontsize=14, fontweight='bold', ha='right', va='top')
    ax.text(center_of_mass[0] + 0.01, center_of_mass[1] - 0.01, 'Q4', fontsize=14, fontweight='bold', ha='left', va='top')
    
    # Add quadrant boundaries
    ax.axhline(y=center_of_mass[1], color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=center_of_mass[0], color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{label_name} - Spatial Quadrant Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10, label='Q1: Anterior-Lateral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=10, label='Q2: Anterior-Medial'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=10, label='Q3: Posterior-Medial'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], markersize=10, label='Q4: Posterior-Lateral'),
        Line2D([0], [0], marker='+', color='w', markerfacecolor='black', markersize=15, label='Center of Mass')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figuredir, f'{label_name}_quadrant_spatial.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nQuadrant Analysis Summary for {label_name}:")
    print("=" * 60)
    for i, (quadrant_data, quadrant_label) in enumerate(quadrants):
        mean_amp = np.mean(np.abs(quadrant_data[start:stop]))
        max_amp = np.max(np.abs(quadrant_data[start:stop]))
        print(f"{quadrant_label}: Mean={mean_amp:.3f}, Max={max_amp:.3f}")
    
    # Calculate quadrant dominance
    total_activity = np.sum([np.mean(np.abs(q[0][start:stop])) for q in quadrants])
    print(f"\nQuadrant Dominance (% of total activity):")
    for i, (quadrant_data, quadrant_label) in enumerate(quadrants):
        dominance = np.mean(np.abs(quadrant_data[start:stop])) / total_activity * 100
        print(f"{quadrant_label}: {dominance:.1f}%")
    print("=" * 60)


def extract_intensity_level_results(epochs, stc, src, visual_labels, visual_label_names, 
                                   somatosensory_labels, somatosensory_label_names, 
                                   intensity_levels=None):
    """
    Extract reconstructed results per intensity level for visual and somatosensory labels.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object containing the intensity_level metadata
    stc : SourceEstimate
        Source estimate to extract time courses from
    src : SourceSpaces or str
        Source space information
    visual_labels : list
        List of visual region labels
    visual_label_names : list
        Names of visual labels
    somatosensory_labels : list
        List of somatosensory region labels
    somatosensory_label_names : list
        Names of somatosensory labels
    intensity_levels : list, optional
        List of intensity levels to extract. If None, extracts all unique levels from epochs.
    
    Returns:
    --------
    results : dict
        Dictionary containing extracted results organized by intensity level and region type
    """
    
    # Load source space if provided as string
    if isinstance(src, str):
        src = mne.read_source_spaces(src)
    
    # Get unique intensity levels if not provided
    if intensity_levels is None:
        intensity_levels = sorted(epochs.metadata['intensity_level'].unique())
    
    print(f"Extracting results for intensity levels: {intensity_levels}")
    
    results = {
        'intensity_levels': intensity_levels,
        'visual': {},
        'somatosensory': {},
        'metadata': {
            'n_epochs_per_level': {},
            'epochs_per_level': {}
        }
    }
    
    # Process each intensity level
    for intensity in intensity_levels:
        print(f"\nProcessing intensity level: {intensity}")
        
        # Filter epochs for this intensity level
        epochs_level = epochs[epochs.metadata['intensity_level'] == intensity]
        
        if len(epochs_level) == 0:
            print(f"  Warning: No epochs found for intensity level {intensity}")
            continue
            
        print(f"  Number of epochs: {len(epochs_level)}")
        results['metadata']['n_epochs_per_level'][intensity] = len(epochs_level)
        results['metadata']['epochs_per_level'][intensity] = epochs_level
        
        # Average epochs for this intensity level
        evoked_level = epochs_level.average()
        
        # Extract time courses for visual regions
        visual_ts_level = []
        for label in visual_labels:
            ts = stc.extract_label_time_course(label, src=src, mode='mean')
            visual_ts_level.append(ts)
        
        # Extract time courses for somatosensory regions
        somatosensory_ts_level = []
        for label in somatosensory_labels:
            ts = stc.extract_label_time_course(label, src=src, mode='mean')
            somatosensory_ts_level.append(ts)
        
        # Store results
        results['visual'][intensity] = {
            'time_courses': visual_ts_level,
            'average': np.mean(visual_ts_level, axis=0),
            'individual': dict(zip(visual_label_names, visual_ts_level)),
            'times': evoked_level.times
        }
        
        results['somatosensory'][intensity] = {
            'time_courses': somatosensory_ts_level,
            'average': np.mean(somatosensory_ts_level, axis=0),
            'individual': dict(zip(somatosensory_label_names, somatosensory_ts_level)),
            'times': evoked_level.times
        }
    
    print(f"\nSuccessfully extracted results for {len(results['intensity_levels'])} intensity levels")
    return results


def plot_intensity_level_results(results, figuredir, modality, subID='average', 
                                timewindow=None, save_data=True):
    """
    Plot reconstructed results per intensity level for visual and somatosensory labels.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from extract_intensity_level_results
    figuredir : str
        Directory to save figures
    modality : str
        Modality (e.g., 'mecha', 'elec')
    subID : str
        Subject identifier, default 'average'
    timewindow : tuple, optional
        Time window to plot (start, stop) in samples
    save_data : bool
        Whether to save the extracted data as numpy arrays
    """
    
    if subID == 'average':
        identifier = 'fsaverage'
    else:
        identifier = f"sub-0{subID}"
    
    if timewindow:
        start = timewindow[0]
        stop = timewindow[1]
    else:
        start = 0
        stop = -1
    
    intensity_levels = results['intensity_levels']
    print(f"Plotting results for intensity levels: {intensity_levels}")
    
    # Color scheme for different intensity levels
    colors_intensity = plt.cm.viridis(np.linspace(0, 1, len(intensity_levels)))
    
    # ====================================================================
    # FIGURE 1: Visual Regions - Individual Labels
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{identifier} - Visual Regions: Intensity Level Effects ({modality})', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    for i, (label_name, label_data) in enumerate(results['visual']['individual'].items()):
        ax = axes[i]
        
        # Plot each intensity level
        for j, intensity in enumerate(intensity_levels):
            if intensity in results['visual']:
                times = results['visual'][intensity]['times'][start:stop]
                ts = results['visual'][intensity]['individual'][label_name].T[start:stop]
                ax.plot(times, ts, color=colors_intensity[j], linewidth=2.5, 
                       label=f'Intensity {intensity}', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (nAm)', fontsize=12)
        ax.set_title(f'{label_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
    
    plt.tight_layout()
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_visual_intensity_levels.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visual regions plot saved to: {filename}")
    plt.show()
    
    # ====================================================================
    # FIGURE 2: Visual Regions - Average Across Labels
    # ====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for j, intensity in enumerate(intensity_levels):
        if intensity in results['visual']:
            times = results['visual'][intensity]['times'][start:stop]
            avg_ts = results['visual'][intensity]['average'].T[start:stop]
            ax.plot(times, avg_ts, color=colors_intensity[j], linewidth=3, 
                   label=f'Intensity {intensity}', alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Average Amplitude (nAm)', fontsize=12)
    ax.set_title(f'{identifier} - Visual Regions: Average Across Labels ({modality})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    
    plt.tight_layout()
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_visual_intensity_average.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visual average plot saved to: {filename}")
    plt.show()
    
    # ====================================================================
    # FIGURE 3: Somatosensory Regions - Individual Labels
    # ====================================================================
    n_somato_labels = len(results['somatosensory']['individual'])
    n_cols = 4
    n_rows = (n_somato_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'{identifier} - Somatosensory Regions: Intensity Level Effects ({modality})', 
                 fontsize=16, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for i, (label_name, label_data) in enumerate(results['somatosensory']['individual'].items()):
        ax = axes_flat[i]
        
        # Plot each intensity level
        for j, intensity in enumerate(intensity_levels):
            if intensity in results['somatosensory']:
                times = results['somatosensory'][intensity]['times'][start:stop]
                ts = results['somatosensory'][intensity]['individual'][label_name].T[start:stop]
                ax.plot(times, ts, color=colors_intensity[j], linewidth=2.5, 
                       label=f'Intensity {intensity}', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude (nAm)', fontsize=11)
        ax.set_title(f'{label_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', ls='--', alpha=0.5)
    
    # Hide unused subplots
    for i in range(n_somato_labels, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_somatosensory_intensity_levels.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Somatosensory regions plot saved to: {filename}")
    plt.show()
    
    # ====================================================================
    # FIGURE 4: Somatosensory Regions - Average Across Labels
    # ====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for j, intensity in enumerate(intensity_levels):
        if intensity in results['somatosensory']:
            times = results['somatosensory'][intensity]['times'][start:stop]
            avg_ts = results['somatosensory'][intensity]['average'].T[start:stop]
            ax.plot(times, avg_ts, color=colors_intensity[j], linewidth=3, 
                   label=f'Intensity {intensity}', alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Average Amplitude (nAm)', fontsize=12)
    ax.set_title(f'{identifier} - Somatosensory Regions: Average Across Labels ({modality})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    
    plt.tight_layout()
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_somatosensory_intensity_average.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Somatosensory average plot saved to: {filename}")
    plt.show()
    
    # ====================================================================
    # FIGURE 5: Combined Comparison - Visual vs Somatosensory
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{identifier} - Intensity Level Effects: Visual vs Somatosensory ({modality})', 
                 fontsize=16, fontweight='bold')
    
    # Visual average
    ax1 = axes[0]
    for j, intensity in enumerate(intensity_levels):
        if intensity in results['visual']:
            times = results['visual'][intensity]['times'][start:stop]
            avg_ts = results['visual'][intensity]['average'].T[start:stop]
            ax1.plot(times, avg_ts, color=colors_intensity[j], linewidth=3, 
                    label=f'Intensity {intensity}', alpha=0.9)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Average Amplitude (nAm)', fontsize=12)
    ax1.set_title('Visual Regions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='k', ls='--', alpha=0.5)
    
    # Somatosensory average
    ax2 = axes[1]
    for j, intensity in enumerate(intensity_levels):
        if intensity in results['somatosensory']:
            times = results['somatosensory'][intensity]['times'][start:stop]
            avg_ts = results['somatosensory'][intensity]['average'].T[start:stop]
            ax2.plot(times, avg_ts, color=colors_intensity[j], linewidth=3, 
                    label=f'Intensity {intensity}', alpha=0.9)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Average Amplitude (nAm)', fontsize=12)
    ax2.set_title('Somatosensory Regions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='k', ls='--', alpha=0.5)
    
    plt.tight_layout()
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_intensity_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot saved to: {filename}")
    plt.show()
    
    # ====================================================================
    # FIGURE 6: Intensity Response Curves
    # ====================================================================
    # Calculate peak amplitudes for each intensity level
    visual_peaks = []
    somatosensory_peaks = []
    
    for intensity in intensity_levels:
        if intensity in results['visual']:
            avg_ts = results['visual'][intensity]['average'].flatten()
            visual_peaks.append(np.max(np.abs(avg_ts[start:stop])))
        else:
            visual_peaks.append(0)
            
        if intensity in results['somatosensory']:
            avg_ts = results['somatosensory'][intensity]['average'].flatten()
            somatosensory_peaks.append(np.max(np.abs(avg_ts[start:stop])))
        else:
            somatosensory_peaks.append(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(intensity_levels, visual_peaks, 'o-', color='blue', linewidth=3, 
           markersize=8, label='Visual Regions', alpha=0.8)
    ax.plot(intensity_levels, somatosensory_peaks, 's-', color='red', linewidth=3, 
           markersize=8, label='Somatosensory Regions', alpha=0.8)
    
    ax.set_xlabel('Intensity Level', fontsize=12)
    ax.set_ylabel('Peak Amplitude (nAm)', fontsize=12)
    ax.set_title(f'{identifier} - Intensity Response Curves ({modality})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(figuredir, f'{identifier}_ses-{modality}_intensity_response_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Intensity response curves saved to: {filename}")
    plt.show()
    
    # ====================================================================
    # Save extracted data if requested
    # ====================================================================
    if save_data:
        data_dir = os.path.join(figuredir, 'intensity_level_data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save metadata
        metadata_file = os.path.join(data_dir, f'{identifier}_ses-{modality}_intensity_metadata.npy')
        np.save(metadata_file, results['metadata'])
        print(f"Metadata saved to: {metadata_file}")
        
        # Save visual data
        visual_data = {}
        for intensity in intensity_levels:
            if intensity in results['visual']:
                visual_data[intensity] = {
                    'times': results['visual'][intensity]['times'],
                    'average': results['visual'][intensity]['average'],
                    'individual': results['visual'][intensity]['individual']
                }
        
        visual_file = os.path.join(data_dir, f'{identifier}_ses-{modality}_visual_intensity_data.npy')
        np.save(visual_file, visual_data)
        print(f"Visual data saved to: {visual_file}")
        
        # Save somatosensory data
        somatosensory_data = {}
        for intensity in intensity_levels:
            if intensity in results['somatosensory']:
                somatosensory_data[intensity] = {
                    'times': results['somatosensory'][intensity]['times'],
                    'average': results['somatosensory'][intensity]['average'],
                    'individual': results['somatosensory'][intensity]['individual']
                }
        
        somatosensory_file = os.path.join(data_dir, f'{identifier}_ses-{modality}_somatosensory_intensity_data.npy')
        np.save(somatosensory_file, somatosensory_data)
        print(f"Somatosensory data saved to: {somatosensory_file}")
    
    # ====================================================================
    # Print summary statistics
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"INTENSITY LEVEL ANALYSIS SUMMARY - {identifier} ({modality})")
    print(f"{'='*70}")
    print(f"Number of intensity levels: {len(intensity_levels)}")
    print(f"Intensity levels: {intensity_levels}")
    print(f"\nEpochs per intensity level:")
    for intensity, n_epochs in results['metadata']['n_epochs_per_level'].items():
        print(f"  Intensity {intensity}: {n_epochs} epochs")
    
    print(f"\nPeak amplitudes:")
    for i, intensity in enumerate(intensity_levels):
        print(f"  Intensity {intensity}: Visual={visual_peaks[i]:.3f}, Somatosensory={somatosensory_peaks[i]:.3f}")
    
    print(f"{'='*70}")



def run_beamformer(evoked, fwd, data_cov_reg, noise_cov_reg, pick_ori=None, nulling_label=None, null_reduction=None):
    
    # apply LCMV beamformer to evoked data
    evoked.set_eeg_reference(ref_channels="average", projection=True, verbose=False)
    
    if nulling_label is not None:
        filters = make_nulling_beamformer(
            evoked.info, 
            fwd, 
            data_cov_reg, 
            nulling_label, 
            null_reduction=null_reduction,
            weight_norm="unit-noise-gain",
            noise_cov=noise_cov_reg, 
            rank=None
        )
        stc_lcmv = apply_nulling(evoked, filters)

    else:
        filters_lcmv = make_lcmv(
            evoked.info,
            fwd,
            data_cov_reg,
            weight_norm="unit-noise-gain",
            pick_ori=None,
            noise_cov=noise_cov_reg, 
            rank=None
        )
        stc_lcmv = apply_lcmv(evoked, filters_lcmv)

    # normalize
    stc_lcmv_norm = stc_lcmv.copy()
    stc_lcmv_norm.data /= np.max(np.abs(stc_lcmv_norm.data))

    return stc_lcmv 

def get_label_data(rh_somatosensory_labels, src_sub, stc_vector):   
    # Compute and plot SVD with data shape (orientations, n_sources*times)
    print("\nComputing SVD with shape (orientations, n_sources*times)...")

    # Combine all right hemisphere somatosensory labels
    all_rh_vertices = []
    all_rh_indices = []
    for label in rh_somatosensory_labels:
        vertices = label.get_vertices_used(src_sub[1]['vertno'])
        all_rh_vertices.extend(vertices)
        print(len(vertices), 'in area', label.name)

        # find the indices 
        hemi_map = np.searchsorted(src_sub[1]['vertno'], vertices)
        all_rh_indices.extend(hemi_map)

    print(f"found {len(all_rh_indices)} indices")
    print(f"found {len(all_rh_vertices)} vertices")

    # Remove duplicates and sort
    all_rh_vertices = sorted(list(set(all_rh_vertices)))
    print(f"Total RH somatosensory vertices: {len(all_rh_vertices)}")
    all_rh_indices = sorted(list(set(all_rh_indices)))
    print(f"Total RH somatosensory indices: {len(all_rh_indices)}")


    # Extract data for all RH somatosensory vertices
    all_regions_hemi_map = np.searchsorted(src_sub[1]['vertno'], all_rh_vertices)
    ts_vector_rh = stc_vector.data[all_regions_hemi_map]
    n_vertices_rh, n_orientations, n_times = ts_vector_rh.shape

    print(f"Data shape: ({n_vertices_rh} vertices, {n_orientations} orientations, {n_times} times)")

    # Return both data and the exact RH vertex ids used to build it.
    return ts_vector_rh, all_rh_vertices



# %%
def plot_reconstructed_sources_topography_2d(
    stc, fwd, info, figuredir, filename, time_idx=None, title_prefix="Reconstructed source"
):
    """
    Plot 2D scalp topography of reconstructed sources by projecting the STC back
    to sensor space at one time point.

    Parameters
    ----------
    stc : mne.SourceEstimate | mne.VectorSourceEstimate
        Reconstructed source estimate.
    fwd : mne.Forward
        Forward solution matching the source space used for reconstruction.
    info : mne.Info
        Sensor info used to plot topography.
    figuredir : str
        Directory where the figure is saved.
    filename : str
        Figure file name.
    time_idx : int | None
        Time index to plot. If None, uses peak absolute source activity.
    """
    if time_idx is None:
        if stc.data.ndim == 3:
            # Vector estimate: use magnitude to select a representative peak.
            source_mag = np.linalg.norm(stc.data, axis=1)
            time_idx = int(np.argmax(np.max(np.abs(source_mag), axis=0)))
        else:
            time_idx = int(np.argmax(np.max(np.abs(stc.data), axis=0)))

    stc_one = stc.copy().crop(tmin=stc.times[time_idx], tmax=stc.times[time_idx])
    evoked_recon = mne.apply_forward(fwd, stc_one, info, use_cps=True)

    topo_data = evoked_recon.data[:, 0]
    vmax = np.max(np.abs(topo_data))
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        topo_data,
        info,
        axes=ax,
        contours=0,
        show=False,
        vlim=(-vmax, vmax),
        cmap="RdBu_r",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Amplitude")
    ax.set_title(f"{title_prefix} topography ({stc.times[time_idx] * 1e3:.1f} ms)")

    os.makedirs(figuredir, exist_ok=True)
    fig.savefig(os.path.join(figuredir, filename), dpi=300, bbox_inches="tight")
    plt.show()
    #plt.close(fig)


def plot_topography_comparison_2d(
    stc_original,
    stc_reconstructed,
    stc_lcmv,
    fwd,
    info,
    figuredir,
    filename="nullingbf_original_vs_reconstructed_topography_2d.pdf",
    time_idx_original=None,
    time_idx_reconstructed=None,
    time_idx_lcmv=None,
    selected_channel=None,
):
    """
    Plot original simulated, nulling beamformer, and LCMV reconstructed source topographies 
    in one figure, with an additional row showing timecourses for a selected channel.
    
    Parameters
    ----------
    stc_original : mne.SourceEstimate
        Original simulated source estimate.
    stc_reconstructed : mne.SourceEstimate
        Nulling beamformer reconstructed source estimate.
    stc_lcmv : mne.SourceEstimate
        LCMV beamformer reconstructed source estimate.
    fwd : mne.Forward
        Forward solution.
    info : mne.Info
        Sensor info.
    figuredir : str
        Directory to save the figure.
    filename : str
        Output filename.
    time_idx_original : int | None
        Time index for original topography. If None, uses peak activity.
    time_idx_reconstructed : int | None
        Time index for nulling beamformer topography. If None, uses peak activity.
    time_idx_lcmv : int | None
        Time index for LCMV topography. If None, uses peak activity.
    selected_channel : str | None
        Channel name for timecourse plot. If None, uses the channel with 
        maximum activity at the peak timepoint.
    """
    def _get_time_idx(stc, time_idx):
        if time_idx is not None:
            return int(time_idx)
        if stc.data.ndim == 3:
            source_mag = np.linalg.norm(stc.data, axis=1)
            return int(np.argmax(np.max(np.abs(source_mag), axis=0)))
        return int(np.argmax(np.max(np.abs(stc.data), axis=0)))

    def _get_timecourse(stc, fwd, info, channel_name):
        """Extract timecourse for a specific channel."""
        evoked_full = mne.apply_forward(fwd, stc, info, use_cps=True)
        ch_idx = info.ch_names.index(channel_name)
        return evoked_full.times, evoked_full.data[ch_idx, :]

    t_idx_orig = _get_time_idx(stc_original, time_idx_original)
    # add 100ms to original time index to account for the shift in simulated data
    t_idx_orig = t_idx_orig + 50
    t_idx_recon = _get_time_idx(stc_reconstructed, time_idx_reconstructed)
    t_idx_lcmv = _get_time_idx(stc_lcmv, time_idx_lcmv)

    # Apply forward solution to get full timecourse for timecourse plots
    evoked_full_orig = mne.apply_forward(fwd, stc_original, info, use_cps=True)
    evoked_full_recon = mne.apply_forward(fwd, stc_reconstructed, info, use_cps=True)
    evoked_full_lcmv = mne.apply_forward(fwd, stc_lcmv, info, use_cps=True)

    # Get topography at specific timepoints
    topo_orig = evoked_full_orig.data[:, t_idx_orig]
    topo_recon = evoked_full_recon.data[:, t_idx_recon]
    topo_lcmv = evoked_full_lcmv.data[:, t_idx_lcmv]
    vmax_orig = np.max(np.abs(topo_orig))
    vmax_recon = np.max(np.abs(topo_recon))
    vmax_lcmv = np.max(np.abs(topo_lcmv))

    # Determine selected channel if not provided
    if selected_channel is None:
        # Use channel with maximum absolute activity at peak timepoint
        evoked_full_orig = mne.apply_forward(fwd, stc_original, info, use_cps=True)
        max_ch_idx = np.argmax(np.abs(evoked_full_orig.data[:, t_idx_orig]))
        selected_channel = info.ch_names[max_ch_idx]

    # Get timecourses for the selected channel
    times_orig, timecourse_orig = _get_timecourse(stc_original, fwd, info, selected_channel)
    # original time is displayed 100ms too early
    times_orig = times_orig + 0.1
    times_recon, timecourse_recon = _get_timecourse(stc_reconstructed, fwd, info, selected_channel)
    times_lcmv, timecourse_lcmv = _get_timecourse(stc_lcmv, fwd, info, selected_channel)

    # Create figure with 2 rows: topographies (row 1) and timecourses (row 2)
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    
    # Top row: Topographies
    im1, _ = mne.viz.plot_topomap(
        topo_orig,
        info,
        axes=axes[0, 0],
        contours=0,
        show=False,
        vlim=(-vmax_orig, vmax_orig),
        cmap="RdBu_r",
    )
    im2, _ = mne.viz.plot_topomap(
        topo_recon,
        info,
        axes=axes[0, 1],
        contours=0,
        show=False,
        vlim=(-vmax_recon, vmax_recon),
        cmap="RdBu_r",
    )
    im3, _ = mne.viz.plot_topomap(
        topo_lcmv,
        info,
        axes=axes[0, 2],
        contours=0,
        show=False,
        vlim=(-vmax_lcmv, vmax_lcmv),
        cmap="RdBu_r",
    )

    axes[0, 0].set_title(f"Original simulated ({times_orig[t_idx_orig] * 1e3:.1f} ms)", fontsize=16, fontweight='bold')
    axes[0, 1].set_title(
        f"Nulling BF reconstructed ({stc_reconstructed.times[t_idx_recon] * 1e3:.1f} ms)", fontsize=16, fontweight='bold'
    )
    axes[0, 2].set_title(
        f"LCMV reconstructed ({stc_lcmv.times[t_idx_lcmv] * 1e3:.1f} ms)", fontsize=16, fontweight='bold'
    )
    cbar1 = fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    cbar1.set_label("Amplitude", fontsize=14)
    cbar2 = fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    cbar2.set_label("Amplitude", fontsize=14)
    cbar3 = fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    cbar3.set_label("Amplitude", fontsize=14)

    # Bottom row: Timecourses
    axes[1, 0].plot(times_orig * 1000, timecourse_orig, 'b-', linewidth=2, label='Original')
    axes[1, 0].axvline(times_orig[t_idx_orig] * 1000, color='red', linestyle='--', 
                       alpha=0.7, label=f'Peak time ({stc_original.times[t_idx_orig] * 1e3:.1f} ms)')
    axes[1, 0].set_xlabel('Time (ms)', fontsize=14)
    axes[1, 0].set_ylabel('Amplitude', fontsize=14)
    axes[1, 0].set_title(f'Timecourse: {selected_channel} (Original)', fontsize=16, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(labelsize=12)
    
    axes[1, 1].plot(times_recon * 1000, timecourse_recon, 'r-', linewidth=2, label='Nulling BF')
    axes[1, 1].axvline(stc_reconstructed.times[t_idx_recon] * 1000, color='blue', linestyle='--', 
                       alpha=0.7, label=f'Peak time ({stc_reconstructed.times[t_idx_recon] * 1e3:.1f} ms)')
    axes[1, 1].set_xlabel('Time (ms)', fontsize=14)
    axes[1, 1].set_ylabel('Amplitude', fontsize=14)
    axes[1, 1].set_title(f'Timecourse: {selected_channel} (Nulling BF)', fontsize=16, fontweight='bold')
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(labelsize=12)
    
    axes[1, 2].plot(times_lcmv * 1000, timecourse_lcmv, 'g-', linewidth=2, label='LCMV')
    axes[1, 2].axvline(stc_lcmv.times[t_idx_lcmv] * 1000, color='purple', linestyle='--', 
                       alpha=0.7, label=f'Peak time ({stc_lcmv.times[t_idx_lcmv] * 1e3:.1f} ms)')
    axes[1, 2].set_xlabel('Time (ms)', fontsize=14)
    axes[1, 2].set_ylabel('Amplitude', fontsize=14)
    axes[1, 2].set_title(f'Timecourse: {selected_channel} (LCMV)', fontsize=16, fontweight='bold')
    axes[1, 2].legend(fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].tick_params(labelsize=12)

    fig.suptitle(f"2D Scalp Topography & Timecourse: Original vs Nulling BF vs LCMV\nSelected Channel: {selected_channel}", 
                 y=0.98, fontsize=18, fontweight='bold')

    os.makedirs(figuredir, exist_ok=True)
    fig.savefig(os.path.join(figuredir, filename), dpi=300, bbox_inches="tight")
    plt.show()
    #plt.close(fig)


def plot_source_space_topography_comparison_2d(
    stc_original,
    stc_reconstructed,
    src,
    figuredir,
    filename="nullingbf_original_vs_reconstructed_source_space_2d.png",
    time_idx_original=None,
    time_idx_reconstructed=None,
):
    """
    Plot 2D source-space topography (vertex-wise activity) for original vs reconstructed.
    Uses source coordinates from src (not electrode space).
    """
    def _get_time_idx(stc, time_idx):
        if time_idx is not None:
            return int(time_idx)
        if stc.data.ndim == 3:
            source_mag = np.linalg.norm(stc.data, axis=1)
            return int(np.argmax(np.max(np.abs(source_mag), axis=0)))
        return int(np.argmax(np.max(np.abs(stc.data), axis=0)))

    def _get_scalar_data_at_time(stc, time_idx):
        if stc.data.ndim == 3:
            data_t = np.linalg.norm(stc.data, axis=1)[:, time_idx]
        else:
            data_t = stc.data[:, time_idx]
        return data_t

    def _hemi_points_and_values(stc, data_t, src, hemi_idx):
        stc_hemi_vertices = stc.vertices[hemi_idx]
        src_hemi_vertices = src[hemi_idx]["vertno"]
        hemi_coords = src[hemi_idx]["rr"]

        hemi_data = np.zeros(len(src_hemi_vertices))
        offset = len(stc.vertices[0]) if hemi_idx == 1 else 0
        stc_hemi_data = data_t[offset : offset + len(stc_hemi_vertices)]

        hemi_map = np.searchsorted(src_hemi_vertices, stc_hemi_vertices)
        hemi_data[hemi_map] = stc_hemi_data
        coords = hemi_coords[src_hemi_vertices]
        return coords, hemi_data
    
    t_idx_orig = _get_time_idx(stc_original, time_idx_original)
    t_idx_recon = _get_time_idx(stc_reconstructed, time_idx_reconstructed)
    print(t_idx_orig, t_idx_recon)

    data_orig_t = _get_scalar_data_at_time(stc_original, t_idx_orig)
    data_recon_t = _get_scalar_data_at_time(stc_reconstructed, t_idx_recon)

    coords_lh_orig, vals_lh_orig = _hemi_points_and_values(stc_original, data_orig_t, src, 0)
    coords_rh_orig, vals_rh_orig = _hemi_points_and_values(stc_original, data_orig_t, src, 1)
    coords_lh_rec, vals_lh_rec = _hemi_points_and_values(stc_reconstructed, data_recon_t, src, 0)
    coords_rh_rec, vals_rh_rec = _hemi_points_and_values(stc_reconstructed, data_recon_t, src, 1)

    vmax_orig = np.max(np.abs(np.hstack([vals_lh_orig, vals_rh_orig])))
    vmax_recon = np.max(np.abs(np.hstack([vals_lh_rec, vals_rh_rec])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Left: original
    sc1 = axes[0].scatter(
        coords_lh_orig[:, 0],
        coords_lh_orig[:, 1],
        c=vals_lh_orig,
        s=8,
        cmap="RdBu_r",
        vmin=-vmax_orig,
        vmax=vmax_orig,
    )
    axes[0].scatter(
        coords_rh_orig[:, 0],
        coords_rh_orig[:, 1],
        c=vals_rh_orig,
        s=8,
        cmap="RdBu_r",
        vmin=-vmax_orig,
        vmax=vmax_orig,
    )
    axes[0].set_title(f"Original simulated ({stc_original.times[t_idx_orig] * 1e3:.1f} ms)")

    # Right: reconstructed
    sc2 = axes[1].scatter(
        coords_lh_rec[:, 0],
        coords_lh_rec[:, 1],
        c=vals_lh_rec,
        s=8,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    axes[1].scatter(
        coords_rh_rec[:, 0],
        coords_rh_rec[:, 1],
        c=vals_rh_rec,
        s=8,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    axes[1].set_title(
        f"Nulling BF reconstructed ({stc_reconstructed.times[t_idx_recon] * 1e3:.1f} ms)"
    )

    for ax in axes:
        ax.set_xlabel("X (source space)")
        ax.set_ylabel("Y (source space)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    cbar1 = fig.colorbar(sc1, ax=axes[0], shrink=0.8)
    cbar1.set_label("Source amplitude")
    cbar2 = fig.colorbar(sc2, ax=axes[1], shrink=0.8)
    cbar2.set_label("Source amplitude")
    fig.suptitle("2D Source-Space Topography: Original vs Reconstructed", y=0.98)

    os.makedirs(figuredir, exist_ok=True)
    fig.savefig(os.path.join(figuredir, filename), dpi=300, bbox_inches="tight")
    plt.show()
    #plt.close(fig)
