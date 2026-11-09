######################################
# Build a parcel adjacency (neighbours) file for a cortical atlas, in the
# FieldTrip `neighbours` struct-array layout so it can be read unchanged by
# read_fieldtrip_neighbors (Analysis/EEGAnalysis/ClusterPermutation/
# read_adjacency.jl).
#
# Two parcels are neighbours iff their cortical patches share a border, i.e. any
# source-space vertex of one parcel is graph-connected to a vertex of the other
# on the fsaverage cortical mesh. The two hemispheres are disconnected in the
# source-space adjacency, so no cross-hemisphere edges are produced.
#
# The atlas is selected by ATLAS below. Two configurations are supported and
# must stay in sync with the parcellated source-rec scripts (same PARC name and
# same DROP_LABEL_PREFIXES, so the neighbour `label` order matches the HDF5
# parcel-group order by construction):
#   "aparc"        -> Desikan-Killiany (step002_..._parcellated.py)
#   "aparc.a2009s" -> Destrieux, fine-grained (step002_..._parcellated_a2009s.py)
#
# Output:
#   Analysis/EEGAnalysis/ClusterPermutation/<atlas_tag>_parcel_neighb.mat
######################################

# %%
import os
import mne
import numpy as np
from scipy.io import savemat

RECONDIR = os.getenv("SUBJECTS_DIR")
WDDIR = os.getenv("WDDIR")

# Select the atlas here. Keep "aparc" as the default so the existing
# aparc_parcel_neighb.mat is reproduced unchanged.
ATLAS = "aparc.a2009s"  # "aparc" or "aparc.a2009s"

# Per-atlas config: FreeSurfer annot name, non-cortical labels to drop, a
# filename-safe tag, and sanity-check probe parcels (guarded on membership).
ATLAS_CONFIG = {
    "aparc": {
        "drop_prefixes": ("unknown", "corpuscallosum"),
        "tag": "aparc",
        "probes": ("precentral-lh", "postcentral-lh"),
    },
    "aparc.a2009s": {
        "drop_prefixes": ("unknown", "medial_wall"),
        "tag": "aparc_a2009s",
        "probes": ("G_precentral-lh", "G_postcentral-lh"),
    },
}
if ATLAS not in ATLAS_CONFIG:
    raise RuntimeError(f"Unknown ATLAS {ATLAS!r}; expected one of {list(ATLAS_CONFIG)}.")

PARC = ATLAS
DROP_LABEL_PREFIXES = ATLAS_CONFIG[ATLAS]["drop_prefixes"]
PROBE_PARCELS = ATLAS_CONFIG[ATLAS]["probes"]

src_file_fsaverage = os.path.join(RECONDIR, "fsaverage_data", "bem", "fsaverage-ico-5-src.fif")
out_path = os.path.join(
    WDDIR, "Analysis", "EEGAnalysis", "ClusterPermutation",
    f"{ATLAS_CONFIG[ATLAS]['tag']}_parcel_neighb.mat",
)


def _keep_label(label):
    name = label.name.lower()
    return not any(name.startswith(p) for p in DROP_LABEL_PREFIXES)


# %%
fsaverage_src = mne.read_source_spaces(src_file_fsaverage)

# Vertex adjacency over the used source vertices, ordered lh vertno then rh vertno.
adjacency = mne.spatial_src_adjacency(fsaverage_src).tocoo()

lh_vertno = fsaverage_src[0]["vertno"]
rh_vertno = fsaverage_src[1]["vertno"]
n_lh = len(lh_vertno)
n_total = n_lh + len(rh_vertno)

labels = mne.read_labels_from_annot("fsaverage", parc=PARC, subjects_dir=RECONDIR)
labels = [lab for lab in labels if _keep_label(lab)]
label_names = [lab.name for lab in labels]
name_to_idx = {name: i for i, name in enumerate(label_names)}

# Map each source-space position (0..n_total-1) to its parcel index (-1 if none).
vert_label = np.full(n_total, -1, dtype=int)
for li, lab in enumerate(labels):
    if lab.hemi == "lh":
        _, src_pos, _ = np.intersect1d(lh_vertno, lab.vertices, return_indices=True)
        vert_label[src_pos] = li
    else:  # rh
        _, src_pos, _ = np.intersect1d(rh_vertno, lab.vertices, return_indices=True)
        vert_label[n_lh + src_pos] = li

# Collect neighbour pairs from mesh edges that cross a parcel boundary.
neighbours = [set() for _ in label_names]
for i, j in zip(adjacency.row, adjacency.col):
    li, lj = vert_label[i], vert_label[j]
    if li < 0 or lj < 0 or li == lj:
        continue
    neighbours[li].add(lj)
    neighbours[lj].add(li)

# %%
# Build a 1 x N struct array with fields `label` and `neighblabel` (cell arrays),
# matching FieldTrip neighbour files exactly.
n = len(label_names)
struct = np.zeros((1, n), dtype=[("label", "O"), ("neighblabel", "O")])
for i, name in enumerate(label_names):
    struct["label"][0, i] = name
    neigh_names = [label_names[k] for k in sorted(neighbours[i])]
    # neighblabel must be a 1-D *cell* array of char, as in real FieldTrip
    # neighbour files. dtype=object is what makes savemat write a cell array;
    # a numpy "<U" array would instead be written as one padded char matrix,
    # which MAT.jl decodes with trailing spaces and collapses to a bare String
    # when there is exactly one neighbour.
    cell = np.empty((len(neigh_names),), dtype=object)
    cell[:] = neigh_names
    struct["neighblabel"][0, i] = cell

savemat(out_path, {"neighbours": struct})

# %%
# Sanity summary.
n_isolated = sum(1 for s in neighbours if not s)
print(f"Saved {n} parcels -> {out_path}")
print(f"Parcels with no neighbour: {n_isolated}")
for probe in PROBE_PARCELS:
    if probe in name_to_idx:
        idx = name_to_idx[probe]
        print(f"  {probe} neighbours: {sorted(label_names[k] for k in neighbours[idx])}")

# %%
