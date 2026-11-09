"""
File: source_zmap_vertex_shares.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2026-10-28
Description:
Where the strongest sources of a source-reconstructed cluster-permutation z-map lie.

For each requested coefficient (detection response, modality) and latency the group
source estimate is turned into |estimate|, the strongest TOP_FRACTION of all vertices
(both hemispheres pooled) is selected, and two shares are reported:
  - the share of those vertices in each hemisphere (right = contralateral to the
    stimulated hand),
  - the share per Desikan-Killiany label (fsaverage "aparc" annotation).
The numbers are printed and written to one CSV (long format: one row per coefficient,
latency and label).

The group estimates are read from the cache of the z-map source reconstruction,
<figuredir>/stc/<coefficient>__<inverse tag>-mean-{lh,rh}.stc. The inverse tag has to
match the cache that should be described, e.g. "inv-mne_loose-0_depth-None_method-MNE"
(default) or "inv-mne" for the older cache written before the settings were part of the
file name. A missing cache or a latency that does not fall on a sample of the run is a
hard error.

Run:
    python source_zmap_vertex_shares.py
    python source_zmap_vertex_shares.py --inv-tag inv-mne --latencies-ms response:138 modality:96
"""

# %%
import argparse
import os

import mne
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration

# Run IDs as in PANEL_ROWS of plot_source_zmap_manuscript_panel.py (one run per coefficient).
_BL = "mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4"
RUN_DETECTION = f"20261020-205919__{_BL}"
RUN_MODALITY = f"20261028-100155__{_BL}__coef-modality"
RUN_MODALITY_X_DETECTION = f"20261028-144658__{_BL}__coef-modality_response"
DEFAULT_INV_TAG = "inv-mne_loose-0_depth-None_method-MNE"

# Fraction of all vertices (both hemispheres) that counts as "strongest".
TOP_FRACTION = 0.01

# key -> (run ID, coefficient folder, latencies in ms). Latencies are those drawn in the
# manuscript source panel (PANEL_ROWS of plot_source_zmap_manuscript_panel.py): the cluster
# peaks plus 0 and 50 ms for the detection response.
COEFFICIENTS = {
    "response": (RUN_DETECTION, "01__response_decoded_Yes", [-276, 0, 50, 138, 154, 294]),
    "modality": (RUN_MODALITY, "01__modality_Mechanical", [-446, 98, 128, 246, 252]),
    "interaction": (RUN_MODALITY_X_DETECTION, "01__modality_Mechanical_response_decoded_Yes", [296]),
}

# Only labels that hold at least this share of the strongest vertices are printed.
MIN_PRINT_SHARE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inv-tag", default=DEFAULT_INV_TAG,
                        help=f"Inverse tag of the cached stcs (default: {DEFAULT_INV_TAG}).")
    parser.add_argument("--latencies-ms", nargs="+", default=None,
                        help="Override the latencies, e.g. response:138,-18 modality:96 "
                             "(keys: " + ", ".join(COEFFICIENTS) + ").")
    return parser.parse_args()


def resolve_latencies(overrides):
    latencies = {key: list(lat) for key, (_, _, lat) in COEFFICIENTS.items()}
    for item in overrides or []:
        key, _, values = item.partition(":")
        if key not in COEFFICIENTS or not values:
            raise ValueError(f"Cannot parse --latencies-ms entry {item!r}; expected key:ms[,ms...] "
                             f"with key in {list(COEFFICIENTS)}.")
        latencies[key] = [float(v) for v in values.split(",")]
    return latencies


def strongest_vertex_shares(stc, latency_s, labels_by_vertex, n_total):
    """Right-hemisphere share and label shares of the strongest TOP_FRACTION vertices at one latency."""
    sample = int(np.argmin(np.abs(stc.times - latency_s)))
    # A latency that is not on the sample grid would silently snap to another time.
    if abs(stc.times[sample] - latency_s) > 1e-6:
        raise ValueError(
            f"Latency {latency_s * 1000:.1f} ms is not a sample of the run "
            f"(nearest: {stc.times[sample] * 1000:.1f} ms)."
        )
    values = np.abs(stc.data[:, sample])
    n_top = int(np.ceil(TOP_FRACTION * n_total))
    top = np.argsort(values)[::-1][:n_top]
    n_lh = len(stc.vertices[0])
    right_share = float(np.mean(top >= n_lh))
    top_labels = pd.Series(labels_by_vertex[top])
    label_shares = top_labels.value_counts(normalize=True)
    return sample, n_top, right_share, label_shares


# %%
def main():
    args = parse_args()
    latencies = resolve_latencies(args.latencies_ms)

    resdir = os.getenv("RESDIR")
    recondir = os.getenv("SUBJECTS_DIR")
    for name, value in (("RESDIR", resdir), ("SUBJECTS_DIR", recondir)):
        if not value:
            raise EnvironmentError(f"{name} is not set.")

    source_root = os.path.join(
        resdir, "Figures", "Main", "eeg_results", "source_reconstruction", "zvals_clusterperm",
    )

    # Desikan-Killiany label of every fsaverage source vertex, in stc row order
    # (left hemisphere rows first, then right hemisphere rows).
    annot = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=recondir, verbose=False)
    label_of = {}
    for label in annot:
        if label.name.startswith("unknown") or label.name.startswith("???"):
            continue
        for vertex in label.vertices:
            label_of[(label.hemi, int(vertex))] = label.name.rsplit("-", 1)[0]

    rows = []
    for key, (run_id, coefficient_name, _) in COEFFICIENTS.items():
        stem = os.path.join(source_root, run_id, "stc", f"{coefficient_name}__{args.inv_tag}-mean")
        if not (os.path.isfile(stem + "-lh.stc") and os.path.isfile(stem + "-rh.stc")):
            raise FileNotFoundError(f"No cached group stc at {stem}-{{lh,rh}}.stc")
        stc = mne.read_source_estimate(stem)
        n_total = stc.data.shape[0]

        # "no_label" collects vertices outside every parcel (medial wall).
        labels_by_vertex = np.array(
            [label_of.get(("lh", int(v)), "no_label") for v in stc.vertices[0]]
            + [label_of.get(("rh", int(v)), "no_label") for v in stc.vertices[1]]
        )
        hemi_suffix = np.array(["-lh"] * len(stc.vertices[0]) + ["-rh"] * len(stc.vertices[1]))
        labels_by_vertex = np.char.add(labels_by_vertex.astype(str), np.where(
            labels_by_vertex == "no_label", "", hemi_suffix))

        print(f"\n=== {key} ({coefficient_name}), {args.inv_tag}, {n_total} vertices ===")
        for latency_ms in latencies[key]:
            sample, n_top, right_share, label_shares = strongest_vertex_shares(
                stc, latency_ms / 1000.0, labels_by_vertex, n_total
            )
            shown = label_shares[label_shares >= MIN_PRINT_SHARE]
            print(f"{latency_ms:7.0f} ms: strongest {n_top} vertices, "
                  f"{right_share * 100:.0f}% right / {(1 - right_share) * 100:.0f}% left hemisphere; "
                  + ", ".join(f"{name} {share * 100:.0f}%" for name, share in shown.items()))
            for name, share in label_shares.items():
                rows.append({
                    "inv_tag": args.inv_tag, "coefficient": key, "latency_ms": latency_ms,
                    "n_top_vertices": n_top, "right_hemisphere_share": right_share,
                    "left_hemisphere_share": 1.0 - right_share,
                    "label": name, "label_share": float(share),
                })

    out_dir = os.path.join(resdir, "Statistics", "eeg", "source_reconstruction")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"source_zmap_vertex_shares__{args.inv_tag}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwritten to {out_csv}")


if __name__ == "__main__":
    main()

# %%
