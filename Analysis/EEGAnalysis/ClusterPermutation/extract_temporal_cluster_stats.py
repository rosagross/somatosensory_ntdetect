"""
Extract the significant clusters of the single-channel (temporal) cluster-mass permutation
tests at CP4 and C6 and write them as the manuscript table.

Usage (from the working directory with the mne conda env active):
    python Analysis/EEGAnalysis/ClusterPermutation/extract_temporal_cluster_stats.py
    python Analysis/EEGAnalysis/ClusterPermutation/extract_temporal_cluster_stats.py --print

Reads the runs written by unfold_temporal_permutation_clusterMass.jl (one run per channel,
joint model over both modalities, time-only clusters). Writes the booktabs body rows to
Writing/tables/single_channel_clusters.tex, which main.tex \\input's, so none of the numbers
are typed by hand. With --print it prints the same clusters as plain text instead.

Effect size is the z value of the model coefficient at the cluster's peak sample (peak z)
together with the cluster mass (sum of z over the cluster; its sign is the polarity). The runs
store z values only, no coefficient estimates. Extent is the time window of the cluster.

Both runs must share the configuration that makes the channels comparable (time window,
baseline, model formula, permutations, subjects); a mismatch raises. The cluster lists are
checked against the temporal_clusters.csv the Julia run wrote next to every coefficient.
"""
#%%

import argparse
import os

import numpy as np
import pandas as pd

from extract_cluster_stats import latex_num, latex_pvalue, summarise_clusters

RESDIR = os.environ["RESDIR"]
WDDIR = os.environ["WDDIR"]
RUNS_ROOT = os.path.join(
    RESDIR, "Statistics", "eeg", "cluster_permutation", "cluster_mass_temporal", "runs"
)

# Same runs as RUN_IDS in plot_temporal_clusters_erp_panel.py (the ERP traces and the
# significance bars of Fig. 2 A, B come from them). The first channel is the anchor.
RUN_IDS = {
    "CP4": "20261020-234021__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-96b492e3cf__ch-CP4",
    "C6": "20261021-072249__mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-96b492e3cf__ch-C6",
}

# (table label, coefficient folder); the numeric prefix depends on the model formula of the runs.
COEFFICIENTS = [
    ("detection response", "01__response_decoded_Yes"),
    ("intensity level", "02__intensity_level_centered"),
    (r"intensity $\times$ response", "03__intensity_level_centered_response_decoded_Yes"),
    ("modality", "04__modality_Mechanical"),
    (r"modality $\times$ intensity", "05__modality_Mechanical_intensity_level_centered"),
    (r"modality $\times$ response", "06__modality_Mechanical_response_decoded_Yes"),
    (r"modality $\times$ intensity $\times$ response",
     "07__modality_Mechanical_intensity_level_centered_response_decoded_Yes"),
]

ALPHA = 0.05

# Startup-environment keys that must be identical across the runs being combined (same list
# as in plot_temporal_clusters_erp_panel.py).
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

TABLE_DIR = os.path.join(WDDIR, "Writing", "tables")
TABLE_FILE = os.path.join(TABLE_DIR, "single_channel_clusters.tex")
N_COLS = 7   # channel, factor, mass, peak z, peak t, time window, p


def read_config(run_dir):
    path = os.path.join(run_dir, "startup_environment.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"startup_environment.csv not found: {path}")
    table = pd.read_csv(path)
    env = dict(zip(table["key"].astype(str), table["value"].astype(str)))
    missing = [k for k in CONFIG_KEYS if k not in env]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return {k: env[k] for k in CONFIG_KEYS}


def check_runs_comparable():
    configs = {}
    for channel, run_id in RUN_IDS.items():
        run_dir = os.path.join(RUNS_ROOT, run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run folder for {channel} not found: {run_dir}")
        configs[channel] = read_config(run_dir)
    anchor = next(iter(configs))
    for channel, config in configs.items():
        differing = [k for k in CONFIG_KEYS if config[k] != configs[anchor][k]]
        if differing:
            raise ValueError(
                f"Run of {channel} differs from the anchor run of {anchor} in: {differing}"
            )
    return configs[anchor]


def load_clusters(channel, label, folder):
    """Significant clusters of one coefficient: (rows, p_floor) from cluster_results.npz."""
    coef_dir = os.path.join(RUNS_ROOT, RUN_IDS[channel], "coefficients", folder)
    npz_path = os.path.join(coef_dir, "cluster_results.npz")
    csv_path = os.path.join(coef_dir, "temporal_clusters.csv")
    for path in (npz_path, csv_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[{channel}] {label}: {path} missing.")

    d = np.load(npz_path)
    if "pvals_clustermass" not in d:
        raise KeyError(f"{npz_path} has no 'pvals_clustermass'; re-run the Julia job.")
    z = np.atleast_2d(d["z_vals"])
    if z.shape[0] != 1:
        raise ValueError(f"[{channel}] {label}: expected a single-channel z-map, got {z.shape}.")
    p_floor = 1.0 / (d["null_max_stats"].size + 1)
    rows, _, _ = summarise_clusters(
        z, d["cluster_display"], d["pvals_clustermass"], d["times"], [channel]
    )

    # Cross-check against the cluster table the Julia run wrote itself: same significant
    # clusters (by mass), same p-values.
    csv = pd.read_csv(csv_path)
    csv = csv[csv["pvalue"] < ALPHA]
    if len(csv) != len(rows):
        raise ValueError(
            f"[{channel}] {label}: {len(rows)} clusters in cluster_results.npz but {len(csv)} "
            f"significant rows in temporal_clusters.csv."
        )
    mine = sorted((round(r["mass"], 6), round(r["pvalue"], 6)) for r in rows)
    theirs = sorted((round(m, 6), round(p, 6)) for m, p in zip(csv["mass"], csv["pvalue"]))
    if mine != theirs:
        raise ValueError(f"[{channel}] {label}: npz and temporal_clusters.csv disagree: {mine} vs {theirs}")

    # Strongest cluster first.
    return sorted(rows, key=lambda r: -abs(r["mass"])), p_floor


def build_table():
    """{channel: [(label, rows, p_floor), ...]} in table order."""
    return {
        channel: [(label, *load_clusters(channel, label, folder)) for label, folder in COEFFICIENTS]
        for channel in RUN_IDS
    }


def write_latex(table, config):
    lines = [
        "% single-channel temporal cluster-mass permutation results, joint model, baseline "
        + config["baseline_tag_resolved"],
        "% generated by extract_temporal_cluster_stats.py; do not edit by hand",
    ]
    lines += [f"% {channel}: run {run_id}" for channel, run_id in RUN_IDS.items()]

    for ch_idx, (channel, blocks) in enumerate(table.items()):
        if ch_idx > 0:
            lines.append(r"\midrule")
        first_line_of_channel = True
        for blk_idx, (label, rows, p_floor) in enumerate(blocks):
            if blk_idx > 0:
                lines.append(r"\cmidrule(lr){2-%d}" % N_COLS)
            channel_cell = channel if first_line_of_channel else ""
            if not rows:
                lines.append(
                    f"{channel_cell} & {label} & \\multicolumn{{{N_COLS - 2}}}{{c}}"
                    r"{\textit{no significant cluster}} \\"
                )
                first_line_of_channel = False
                continue
            for j, r in enumerate(rows):
                lines.append(
                    f"{channel_cell if j == 0 else ''} & {label if j == 0 else ''} & "
                    f"{latex_num(r['mass'])} & {latex_num(r['peak_z'], 2)} & "
                    f"{latex_num(r['peak_time'])} & "
                    f"{latex_num(r['t_start'])} to {latex_num(r['t_end'])} & "
                    f"{latex_pvalue(r['pvalue'], p_floor)} \\\\"
                )
                if j == 0:
                    first_line_of_channel = False
                channel_cell = ""

    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(TABLE_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TABLE_FILE}")


def print_table(table, config):
    print(f"Joint model, baseline {config['baseline_tag_resolved']}, "
          f"{config['n_permutations_resolved']} permutations")
    for channel, blocks in table.items():
        print(f"\n=== {channel}  ({RUN_IDS[channel]})")
        for label, rows, p_floor in blocks:
            print(f"  {label}: {len(rows)} significant cluster(s)")
            for r in rows:
                p = f"<{p_floor:.4f}" if np.isclose(r["pvalue"], p_floor) else f"{r['pvalue']:.4f}"
                print(f"    mass {r['mass']:9.1f}  peak z {r['peak_z']:7.2f} at {r['peak_time']:5.0f} ms  "
                      f"window {r['t_start']:5.0f} to {r['t_end']:4.0f} ms  p {p}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="print the clusters as text instead of writing the table file")
    args = parser.parse_args()

    config = check_runs_comparable()
    table = build_table()
    if args.print_only:
        print_table(table, config)
    else:
        write_latex(table, config)


if __name__ == "__main__":
    main()
