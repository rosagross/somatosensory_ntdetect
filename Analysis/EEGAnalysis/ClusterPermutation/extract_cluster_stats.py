"""
Extract and print cluster-mass permutation statistics from cluster_results.npz files.

Usage (from the working directory with the mne conda env active):
    python Analysis/EEGAnalysis/ClusterPermutation/extract_cluster_stats.py
    python Analysis/EEGAnalysis/ClusterPermutation/extract_cluster_stats.py --latex

Covers the joint model only, i.e. the runs fitted over both modalities together that
the manuscript reports. The recomputed runs are split up per coefficient, so every
coefficient names its own run (COEFFICIENT_RUNS); all runs must share the baseline,
otherwise the script raises. Without flags it prints a plain-text summary; with --latex
it writes the booktabs body rows of the manuscript table to
Writing/tables/full_channel_clusters.tex, which main.tex \\input's, so those numbers are
never retyped.

Requires runs produced with p-value storage enabled (npz key "pvals_clustermass").
Older runs predate that and are rejected rather than summarised with "p < .05".
"""
#%%

import argparse
import numpy as np
import os
import re

# Runs live under $RESDIR; no fallback, a missing RESDIR is a hard error.
RUNS_ROOT = os.path.join(
    os.environ["RESDIR"], "Statistics", "eeg", "cluster_permutation", "cluster_mass", "runs"
)

# Joint model over both modalities. The recomputed runs are split up per coefficient, so each
# table row names its own run and coefficient folder (the numeric folder prefix depends on the
# run). All runs must share the baseline (checked below). Listed in table order.
_BL = "mod-mecha-elec__ep-epochs_preprestim_corrected__bl-env-0.55_-0.45__f-6dace1beb4"
COEFFICIENT_RUNS = [
    # (table label, run ID, coefficient folder)
    ("detection response", f"20261020-205919__{_BL}", "01__response_decoded_Yes"),
    ("intensity level", f"20261028-122831__{_BL}__coef-intensity_level_centered",
     "01__intensity_level_centered"),
    (r"intensity $\times$ response", f"20261028-122831__{_BL}__coef-intensity_level_centered",
     "02__intensity_level_centered_response_decoded_Yes"),
    ("modality", f"20261028-100155__{_BL}__coef-modality", "01__modality_Mechanical"),
    (r"modality $\times$ intensity", f"20261029-082527__{_BL}__coef-modality_intensity",
     "01__modality_Mechanical_intensity_level_centered"),
    (r"modality $\times$ response", f"20261028-144658__{_BL}__coef-modality_response",
     "01__modality_Mechanical_response_decoded_Yes"),
    (r"modality $\times$ intensity $\times$ response",
     f"20261102-102511__{_BL}__coef-threeway_interaction",
     "01__modality_Mechanical_intensity_level_centered_response_decoded_Yes"),
]

# All runs must share one baseline; rows from different baselines are not comparable.
_baselines = {}
for _label, _run_id, _folder in COEFFICIENT_RUNS:
    _m = re.search(r"__bl-(.+?)__f-", _run_id)
    if _m is None:
        raise ValueError(f"{_label}: no baseline tag (__bl-...__f-) in run ID {_run_id}")
    _baselines[_run_id] = _m.group(1)
if len(set(_baselines.values())) != 1:
    raise ValueError(
        "COEFFICIENT_RUNS mix runs with different baselines: "
        + "; ".join(f"{b} <- {r}" for r, b in _baselines.items())
    )
BASELINE_TAG = next(iter(_baselines.values()))

# The manuscript tables live next to main.tex, which \input's them.
TABLE_DIR = os.path.join(os.environ["WDDIR"], "Writing", "tables")
TABLE_FILE = os.path.join(TABLE_DIR, "full_channel_clusters.tex")


def load_channels(ch_file):
    # Build a list so index i → channel name at position i (0-based row in z_vals).
    # The file may list arbitrary integers in column 0; we just keep the names in order.
    names = []
    with open(ch_file) as f:
        for line in f:
            parts = line.strip().split()   # split on any whitespace
            if len(parts) >= 2:
                names.append(parts[1])
            elif len(parts) == 1:
                names.append(parts[0])
    return names   # index directly with ch_idx


def format_pvalue(p, p_floor):
    # p can never be 0: the permutation test floors it at 1/(n_permutations + 1).
    # A cluster sitting on that floor is reported as an inequality, not as its value.
    if p_floor is not None and np.isclose(p, p_floor):
        return f"<{p_floor:.4f}"
    return f"{p:.4f}"


def summarise_clusters(z, cd, pvals, times, channels):
    ids = np.unique(cd[~np.isnan(cd)])
    neg_ids = sorted([x for x in ids if x < 0], key=lambda x: x)   # most negative first
    pos_ids = sorted([x for x in ids if x > 0], key=lambda x: -x)  # most positive first

    rows = []
    for label, id_list in [("NEG", neg_ids), ("POS", pos_ids)]:
        for cid in id_list:
            mask = cd == cid
            mass = float(z[mask].sum())
            t_idx = np.where(np.any(mask, axis=0))[0]
            t_win = (float(times[t_idx.min()] * 1000), float(times[t_idx.max()] * 1000))
            ch_idx = np.where(np.any(mask, axis=1))[0]
            ch_names = [channels[i] if i < len(channels) else f"ch{i}" for i in ch_idx]
            # Locate the strongest |z| element *inside* the cluster mask, so peak z,
            # peak channel and peak latency all refer to the very same sample.
            # Masked-out entries are pushed to -inf and can never win the argmax.
            flat = np.argmax(np.where(mask, np.abs(z), -np.inf))
            peak_row, peak_col = np.unravel_index(flat, z.shape)
            peak_z = float(z[peak_row, peak_col])
            peak_channel = (channels[peak_row] if peak_row < len(channels)
                            else f"ch{peak_row}")
            peak_time = float(times[peak_col] * 1000)
            # Every sample of a cluster carries that cluster's p-value; if it does not,
            # the cluster labelling and the p-value map disagree and the run is corrupt.
            p_in_cluster = np.unique(pvals[mask])
            if p_in_cluster.size != 1:
                raise ValueError(
                    f"Cluster {int(cid)} has {p_in_cluster.size} distinct p-values "
                    f"({p_in_cluster}); cluster_display and pvals_clustermass disagree."
                )
            rows.append(
                dict(
                    direction=label,
                    cluster_id=int(cid),
                    mass=mass,
                    peak_z=peak_z,
                    peak_channel=peak_channel,
                    peak_time=peak_time,
                    pvalue=float(p_in_cluster[0]),
                    t_start=t_win[0],
                    t_end=t_win[1],
                    n_channels=len(ch_idx),
                    channels=ch_names,
                )
            )
    return rows, neg_ids, pos_ids


def load_coefficient(run_dir, coef):
    """Load one coefficient's npz and return (rows, p_floor, z, times), or None if absent."""
    npz_path = os.path.join(run_dir, "coefficients", coef, "cluster_results.npz")
    if not os.path.exists(npz_path):
        return None

    d = np.load(npz_path)
    z = d["z_vals"]            # shape: (n_channels, n_times)
    cd = d["cluster_display"]  # NaN outside significant clusters
    times = d["times"]         # in seconds

    if "pvals_clustermass" not in d:
        raise KeyError(
            f"{npz_path} has no 'pvals_clustermass'. This run predates p-value "
            f"storage in unfold_cluster_permutation_clusterMass.jl and cannot be "
            f"summarised with exact p-values; re-run it to obtain them."
        )
    pvals = d["pvals_clustermass"]  # per-sample; 1.0 outside any cluster

    # The permutation count is recoverable from the stored null distribution, which
    # gives the smallest attainable p. Older parcel runs stored p but not the null.
    p_floor = 1.0 / (d["null_max_stats"].size + 1) if "null_max_stats" in d else None

    channels = load_channels(os.path.join(run_dir, "coefficients", coef, "channels.txt"))
    rows, neg_ids, pos_ids = summarise_clusters(z, cd, pvals, times, channels)
    return rows, neg_ids, pos_ids, p_floor, z, times


def print_run():
    print(f"\n{'#' * 80}")
    print(f"# JOINT MODEL (both modalities), baseline {BASELINE_TAG}")
    print(f"{'#' * 80}")

    for label, run_id, coef in COEFFICIENT_RUNS:
        run_dir = os.path.join(RUNS_ROOT, run_id)
        loaded = load_coefficient(run_dir, coef)
        if loaded is None:
            raise FileNotFoundError(
                f"File not found: {os.path.join(run_dir, 'coefficients', coef, 'cluster_results.npz')}"
            )
        rows, neg_ids, pos_ids, p_floor, z, times = loaded

        print(f"\n{'=' * 70}")
        print(f"  Coefficient: {coef}  ({label})")
        print(f"  Run: {run_id}")
        print(f"  z range: {z.min():.3f} to {z.max():.3f}")
        print(f"  Time window: {times[0]*1000:.0f} to {times[-1]*1000:.0f} ms")
        if p_floor is not None:
            # p_floor = 1/(n_permutations + 1), so the permutation count is recoverable.
            n_perm = int(round(1.0 / p_floor)) - 1
            print(f"  Permutations: {n_perm}  (smallest possible p = {p_floor:.4f})")
        print(f"  Significant clusters: {len(rows)}  "
              f"({len(neg_ids)} negative, {len(pos_ids)} positive)")

        if len(rows) == 0:
            print("  --> No significant clusters.")
            continue

        print(f"\n  {'Dir':3s}  {'ID':>6s}  {'Mass':>10s}  {'PeakZ':>7s}  {'PkCh':>5s}  "
              f"{'PkTime':>7s}  {'p':>8s}  {'t_start':>8s}  {'t_end':>6s}  {'n_ch':>4s}  Channels")
        print(f"  {'-'*3}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*8}  "
              f"{'-'*8}  {'-'*6}  {'-'*4}  {'-'*40}")
        for r in rows:
            ch_str = ", ".join(r["channels"])
            if len(ch_str) > 60:
                ch_str = ", ".join(r["channels"][:6]) + f" … ({r['n_channels']} total)"
            print(f"  {r['direction']:3s}  {r['cluster_id']:6d}  {r['mass']:10.2f}  "
                  f"{r['peak_z']:7.3f}  {r['peak_channel']:>5s}  {r['peak_time']:6.0f}ms  "
                  f"{format_pvalue(r['pvalue'], p_floor):>8s}  "
                  f"{r['t_start']:7.0f}ms  {r['t_end']:5.0f}ms  "
                  f"{r['n_channels']:4d}  {ch_str}")


def latex_pvalue(p, p_floor):
    """Manuscript convention: leading zero dropped, floored p reported as an inequality."""
    if p_floor is not None and np.isclose(p, p_floor):
        return r"$<.0005$"
    return f"${p:.3f}$".replace("$0.", "$.")


def latex_num(x, decimals=0):
    """Format a signed number for math mode with a proper LaTeX minus sign."""
    return f"${x:.{decimals}f}$"


def write_latex():
    """Write the booktabs body rows of the full-channel table, one block per coefficient.

    Columns: factor, mass, peak z, peak channel, peak latency, time window, channels, p.
    The mass carries the sign (polarity) of the cluster.
    """
    n_cols = 8
    lines = [
        "% cluster-mass permutation results, joint model, baseline " + BASELINE_TAG,
        "% generated by extract_cluster_stats.py --latex; do not edit by hand",
    ]
    for label, run_id, coef in COEFFICIENT_RUNS:
        lines.append(f"% {label}: run {run_id}, {coef}")
    for i, (label, run_id, coef) in enumerate(COEFFICIENT_RUNS):
        run_dir = os.path.join(RUNS_ROOT, run_id)
        loaded = load_coefficient(run_dir, coef)
        if loaded is None:
            raise FileNotFoundError(
                f"Missing cluster_results.npz for coefficient {coef} in {run_dir}"
            )
        rows, _, _, p_floor, _, _ = loaded

        if i > 0:
            lines.append(r"\midrule")

        if len(rows) == 0:
            lines.append(f"{label} & \\multicolumn{{{n_cols - 1}}}{{c}}{{\\textit{{no significant "
                         f"cluster}}}} \\\\")
            continue

        # Strongest cluster first, so the poles of the dominant dipolar pair head the block.
        rows = sorted(rows, key=lambda r: -abs(r["mass"]))
        for j, r in enumerate(rows):
            factor_cell = label if j == 0 else ""
            lines.append(
                f"{factor_cell} & {latex_num(r['mass'])} & "
                f"{latex_num(r['peak_z'], 2)} & {r['peak_channel']} & "
                f"{latex_num(r['peak_time'])} & "
                f"{latex_num(r['t_start'])} to {latex_num(r['t_end'])} & "
                f"{r['n_channels']} & {latex_pvalue(r['pvalue'], p_floor)} \\\\"
            )

    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(TABLE_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TABLE_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true",
                        help="write the booktabs table rows to Writing/tables instead of the plain-text summary")
    args = parser.parse_args()

    if args.latex:
        write_latex()
    else:
        print_run()


if __name__ == "__main__":
    main()
