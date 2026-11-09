# Temporal (single-channel) cluster-mass permutation test on ERP traces.
#
# This is the 1D sibling of unfold_cluster_permutation_clusterMass.jl: same data loading,
# same LMM formulas, same cluster-mass statistic and the same permutation machinery, but the
# analysis is restricted to ONE a-priori channel (default C6, see CHANNELS below).
#
# With a single channel there is no spatial dimension left, so clusters can only grow along
# time. Everything spatial from the reference script (FieldTrip adjacency file, adjacency
# alignment report, channel-ordered heatmaps) is therefore dropped; see the `adjacency`
# definition inside the coefficient loop for how the shared cluster code is reused.
#
# Side benefit: lmm_permutations loops over channels, so one channel costs ~1/64 of the
# spatiotemporal run and high permutation counts become cheap.

# activate Julia environment in this folder
import Pkg
Pkg.activate(@__DIR__)
using CairoMakie
using CSV
using Statistics
using CategoricalArrays
using DataFrames
using HDF5
using Random
using UnfoldStats
using UnfoldMakie
using LinearAlgebra
using UnfoldSim
using NPZ
using UnfoldMixedModels
using MixedModelsPermutations
using ClusterDepth
ext = Base.get_extension(UnfoldStats, :UnfoldStatsMixedModelsPermutationsExt)
lmm_permutations = ext.lmm_permutations
get_lmm_statistic = ext.get_lmm_statistic

# import the shared loading/plotting helpers and the cluster-mass primitives
include(joinpath(@__DIR__, "cluster_permutation_helpers.jl"))
using .ClusterPermutationHelpers
include(joinpath(@__DIR__, "cluster_mass_test.jl"))
# NOTE: read_adjacency.jl is deliberately NOT included -- a one-channel analysis has no
# spatial neighbourhood to read.

function env_value_or_marker(key::AbstractString)
    # Distinguish unset env vars from intentionally empty values in run logs.
    if !haskey(ENV, key)
        return "<unset>"
    end
    value = String(ENV[key])
    return isempty(strip(value)) ? "<set-empty>" : value
end

function string_or_marker(v)
    return isnothing(v) ? "<none>" : string(v)
end

function build_startup_env_snapshot(;
    datadir::AbstractString,
    resdir::AbstractString,
    outroot::AbstractString,
    subjects::AbstractVector{<:AbstractString},
    excluded_pairs::AbstractSet{<:Tuple{AbstractString, AbstractString}},
    run_modes::AbstractVector{<:AbstractString},
    epoch_stem::AbstractString,
    metadata_stem::AbstractString,
    baseline_tag::AbstractString,
    n_permutations::Integer,
    seed::Integer,
    time_start,
    time_stop,
    highpass_freq,
    plot_start,
    plot_stop,
    coef_selector::AbstractString,
    laplacian::Bool,
    run_modalities::AbstractVector{<:AbstractString},
    run_id::AbstractString,
    run_dir::AbstractString,
    formula_string::AbstractString,
    channel::AbstractString,
    threshold::Real,
)
    # Keep a deterministic, explicit whitelist of env values used by this script.
    # Compared with the spatiotemporal script, PLOT_CHANNEL(S) are replaced by CHANNELS:
    # here the analysed channel IS the plotted channel.
    snapshot = Pair{String, String}[
        "JULIA_DEPOT_PATH_raw" => env_value_or_marker("JULIA_DEPOT_PATH"),
        "DATADIR_raw" => env_value_or_marker("DATADIR"),
        "RESDIR_raw" => env_value_or_marker("RESDIR"),
        "MODALITY_raw" => env_value_or_marker("MODALITY"),
        "SUBJECTS_raw" => env_value_or_marker("SUBJECTS"),
        "CHANNELS_raw" => env_value_or_marker("CHANNELS"),
        "CHANNEL_raw" => env_value_or_marker("CHANNEL"),
        "EPOCH_STEM_raw" => env_value_or_marker("EPOCH_STEM"),
        "METADATA_STEM_raw" => env_value_or_marker("METADATA_STEM"),
        "N_PERMUTATIONS_raw" => env_value_or_marker("N_PERMUTATIONS"),
        "RNG_SEED_raw" => env_value_or_marker("RNG_SEED"),
        "TIME_START_raw" => env_value_or_marker("TIME_START"),
        "TIME_STOP_raw" => env_value_or_marker("TIME_STOP"),
        "HIGHPASS_FREQ_raw" => env_value_or_marker("HIGHPASS_FREQ"),
        "BASELINE_START_raw" => env_value_or_marker("BASELINE_START"),
        "BASELINE_STOP_raw" => env_value_or_marker("BASELINE_STOP"),
        "BASELINE_START_ELEC_raw" => env_value_or_marker("BASELINE_START_ELEC"),
        "BASELINE_STOP_ELEC_raw" => env_value_or_marker("BASELINE_STOP_ELEC"),
        "BASELINE_START_MECHA_raw" => env_value_or_marker("BASELINE_START_MECHA"),
        "BASELINE_STOP_MECHA_raw" => env_value_or_marker("BASELINE_STOP_MECHA"),
        "PLOT_START_raw" => env_value_or_marker("PLOT_START"),
        "PLOT_STOP_raw" => env_value_or_marker("PLOT_STOP"),
        "COEF_raw" => env_value_or_marker("COEF"),
        "datadir_resolved" => datadir,
        "resdir_resolved" => isempty(strip(resdir)) ? "<none>" : resdir,
        "outroot_resolved" => outroot,
        "subjects_resolved" => join(subjects, ","),
        # Effective (not requested) exclusions. Sorted so two runs can be diffed
        # directly; "<none>" marks a genuinely empty set rather than a dropped field.
        "excluded_pairs_resolved" => isempty(excluded_pairs) ? "<none>" :
            join(["$(s)/$(m)" for (s, m) in sort(collect(excluded_pairs))], ";"),
        "run_modes_resolved" => join(run_modes, ","),
        "epoch_stem_resolved" => epoch_stem,
        "metadata_stem_resolved" => metadata_stem,
        "baseline_tag_resolved" => baseline_tag,
        "n_permutations_resolved" => string(n_permutations),
        "seed_resolved" => string(seed),
        "time_start_resolved" => string_or_marker(time_start),
        "time_stop_resolved" => string_or_marker(time_stop),
        "highpass_freq_resolved" => string_or_marker(highpass_freq),
        "plot_start_resolved" => string_or_marker(plot_start),
        "plot_stop_resolved" => string_or_marker(plot_stop),
        "coef_selector_resolved" => coef_selector,
        "run_modalities_resolved" => join(run_modalities, ","),
        "run_id_resolved" => run_id,
        "run_dir_resolved" => run_dir,
        "formula_resolved" => formula_string,
        "channel_resolved" => channel,
        "clusterforming_threshold_resolved" => string(threshold),
        "cluster_dimension" => "time-only (single channel, no spatial adjacency)",
        "LAPLACIAN_raw" => env_value_or_marker("LAPLACIAN"),
        "laplacian_resolved" => string(laplacian),
        "signal_space_resolved" => laplacian ?
            "surface Laplacian (spherical-spline CSD), uV/m^2" : "average reference, uV",
    ]
    return snapshot
end

function log_and_save_startup_env_snapshot(run_dir::AbstractString, snapshot::Vector{Pair{String, String}})
    # Persist startup settings next to analysis outputs for reproducible reruns.
    snapshot_file = joinpath(run_dir, "startup_environment.csv")
    snapshot_df = DataFrame(key = String[], value = String[])
    for (key, value) in snapshot
        push!(snapshot_df, (key = key, value = value))
    end
    CSV.write(snapshot_file, snapshot_df)

    # Log the same payload to stdout so parameters are visible in batch logs.
    @info "Startup run configuration" snapshot=Dict(snapshot)
    @info "Saved startup environment snapshot" path=snapshot_file
    return snapshot_file
end

"""
Resolve the single channel this script operates on.

`parse_channels()` (helpers) already reads CHANNELS / CHANNEL and defaults to "C6"; it returns
`nothing` for CHANNELS=all. This script is a strictly one-channel analysis: with more than one
channel the cluster search would either need a spatial adjacency again, or would silently run
several uncorrected tests. Both are rejected here rather than guessed.
"""
function resolve_single_channel()
    requested = parse_channels()
    if isnothing(requested)
        error("CHANNELS=all is not supported by the temporal (single-channel) permutation test. " *
              "Set CHANNELS to exactly one channel, e.g. CHANNELS=C6.")
    end
    if length(requested) != 1
        error("The temporal permutation test analyses exactly one channel, but CHANNELS resolved " *
              "to $(length(requested)): $(join(requested, ", ")). Run the script once per channel, " *
              "or use unfold_cluster_permutation_clusterMass.jl for a spatiotemporal test.")
    end
    return String(requested[1]), Vector{String}(requested)
end

"""
Summarise the observed temporal clusters of one channel into a tidy table.

`cluster_ids_row` uses the convention of spatiotemporal_cluster_pvalues: 0 = no cluster,
+k = k-th positive cluster, -k = k-th negative cluster. In a one-channel search every cluster
is a contiguous run of time samples, so min/max index give its time window directly.

Every sample of a cluster must carry that cluster's p-value; if not, the id map and the p-value
map disagree and the run is corrupt (hard error rather than silently picking one).
"""
function build_temporal_cluster_table(
    times::AbstractVector,
    z_row::AbstractVector,
    cluster_ids_row::AbstractVector{<:Integer},
    pvals_row::AbstractVector,
)
    table = DataFrame(
        cluster_id = Int[],
        direction = String[],
        t_start_s = Float64[],
        t_stop_s = Float64[],
        n_samples = Int[],
        mass = Float64[],
        peak_z = Float64[],
        peak_time_s = Float64[],
        pvalue = Float64[],
    )

    for cid in sort(unique(cluster_ids_row[cluster_ids_row .!= 0]))
        idx = findall(==(cid), cluster_ids_row)
        z_in = z_row[idx]
        peak_pos = argmax(abs.(z_in))
        p_in = unique(pvals_row[idx])
        length(p_in) == 1 || error("Cluster $cid carries $(length(p_in)) distinct p-values " *
                                   "($(p_in)); cluster ids and p-values disagree.")
        push!(table, (
            cluster_id = Int(cid),
            direction = cid > 0 ? "POS" : "NEG",
            t_start_s = float(times[minimum(idx)]),
            t_stop_s = float(times[maximum(idx)]),
            n_samples = length(idx),
            mass = sum(z_in),
            peak_z = z_in[peak_pos],
            peak_time_s = float(times[idx[peak_pos]]),
            pvalue = float(p_in[1]),
        ))
    end

    sort!(table, :t_start_s)
    return table
end

"""
Plot the z-trace of the tested coefficient over time, with the cluster-forming threshold and
the significant temporal clusters shaded.

This replaces the channel x time heatmaps of the spatiotemporal script, which carry no
information when there is only one channel.
"""
function save_temporal_z_trace_plot(
    times::AbstractVector,
    z_row::AbstractVector,
    cluster_table::DataFrame,
    threshold::Real,
    outdir::AbstractString,
    channel::AbstractString,
    coef_name::AbstractString;
    alpha::Float64 = 0.05,
)
    fig = Figure(size = (960, 440))
    ax = Axis(
        fig[1, 1],
        xlabel = "Time (s)",
        ylabel = "z-value",
        title = "Temporal cluster-mass test ($channel, $coef_name)",
    )

    # Shade first so the z-trace stays readable on top of the cluster spans.
    sig = cluster_table[cluster_table.pvalue .< alpha, :]
    for r in eachrow(sig)
        vspan!(ax, r.t_start_s, r.t_stop_s, color = (:gray, 0.25))
    end

    lines!(ax, collect(times), collect(z_row), color = :black, linewidth = 2.0)
    hlines!(ax, [0.0], color = (:black, 0.3), linewidth = 1.0)
    hlines!(ax, [float(threshold), -float(threshold)], color = (:red, 0.6),
            linewidth = 1.2, linestyle = :dash)
    vlines!(ax, [0.0], color = :black, linewidth = 1.2, linestyle = :dash)

    mkpath(outdir)
    safe_coef = replace(coef_name, r"[^A-Za-z0-9._-]+" => "_")
    outfile = joinpath(outdir, "temporal_z_trace_channel-$(channel)_coef-$(safe_coef).png")
    save(outfile, fig)
    @info "Saved temporal z-trace plot" outfile n_significant_clusters=nrow(sig)
    return outfile
end

function main()

    # specify loading parameters
    datadir = get(ENV, "DATADIR", "")

    if isempty(strip(datadir))
        error("DATADIR is not set. Example: export DATADIR=/data/p_02989/shared_workspace/datadir")
    end

    # The one channel this analysis runs on (CHANNELS/CHANNEL, default C6).
    channel, requested_channels = resolve_single_channel()
    @info "Temporal permutation channel resolved" channel

    epoch_stem = get(ENV, "EPOCH_STEM", "epochs")
    # LAPLACIAN=true switches to the CSD cache (compute_csd_epochs.py). The metadata stem is
    # untouched: the surface Laplacian is a spatial transform and changes no trial.
    epoch_stem, laplacian = resolve_epoch_stem_with_laplacian(epoch_stem)
    laplacian && @info "Surface-Laplacian (CSD) enabled" epoch_stem
    metadata_stem = get(ENV, "METADATA_STEM", "allmetadata")
    # Reflect the actually-resolved baseline windows in the run folder name. Resolve both
    # modalities independently (resolve_baseline_window checks per-modality env vars first,
    # then the global override, then the suffix-driven default) so a combined run whose two
    # sessions get different windows is tagged accordingly, instead of silently implying one
    # shared window that isn't what was applied.
    elec_baseline_start, elec_baseline_stop, elec_baseline_source = resolve_baseline_window("elec", epoch_stem)
    mecha_baseline_start, mecha_baseline_stop, mecha_baseline_source = resolve_baseline_window("mecha", epoch_stem)
    baseline_tag = if (elec_baseline_start, elec_baseline_stop) == (mecha_baseline_start, mecha_baseline_stop)
        prefix = elec_baseline_source == :inferred ? "auto" : "env"
        sanitize_token("$(prefix)$(elec_baseline_start)_$(elec_baseline_stop)")
    else
        sanitize_token("elec$(elec_baseline_start)_$(elec_baseline_stop)-mecha$(mecha_baseline_start)_$(mecha_baseline_stop)")
    end
    subjects = parse_subjects()
    # Only MODALITY is supported (MODALITIES is intentionally ignored).
    # - MODALITY=mecha | elec => run single-modality model
    # - MODALITY=both => run combined-modality model
    modality_mode_raw = lowercase(strip(get(ENV, "MODALITY", "both")))
    run_modes = String[]
    if modality_mode_raw in ("mecha", "elec")
        run_modes = [modality_mode_raw]
    elseif modality_mode_raw == "both"
        run_modes = ["combined"]
    else
        error("Invalid MODALITY='$modality_mode_raw'. Expected one of: mecha, elec, both.")
    end
    if ("mecha" in run_modes) && ("elec" in run_modes)
        error("Invalid run configuration: mecha and elec cannot run as separate loops together. Use MODALITY=both for the combined model.")
    end
    resdir = get(ENV, "RESDIR", "")
    # Keep temporal single-channel outputs in their own tree so they never mix with the
    # spatiotemporal cluster_mass runs.
    outroot = isempty(strip(resdir)) ?
        joinpath(pwd(), "cluster_permutation", "cluster_mass_temporal") :
        joinpath(resdir, "Statistics", "eeg", "cluster_permutation", "cluster_mass_temporal")
    n_permutations = parse(Int, get(ENV, "N_PERMUTATIONS", "2000"))
    seed = parse(Int, get(ENV, "RNG_SEED", "1"))
    plot_start_str = strip(get(ENV, "PLOT_START", ""))
    plot_stop_str = strip(get(ENV, "PLOT_STOP", ""))
    plot_start = isempty(plot_start_str) ? nothing : parse(Float64, plot_start_str)
    plot_stop = isempty(plot_stop_str) ? nothing : parse(Float64, plot_stop_str)
    # Coefficient selector to limit runtime:
    # COEF=all | intensity_level_centered | response_decoded | force | modality | interaction | <substring>
    coef_selector_raw = strip(get(ENV, "COEF", "all"))
    coef_selector = isempty(coef_selector_raw) ? "all" : lowercase(coef_selector_raw)

    # Add cropping parameters for analysis
    time_start_str = strip(get(ENV, "TIME_START", "-0.55"))
    time_stop_str = strip(get(ENV, "TIME_STOP", "0.3"))
    time_start = isempty(time_start_str) ? nothing : parse(Float64, time_start_str)
    time_stop = isempty(time_stop_str) ? nothing : parse(Float64, time_stop_str)
    # Optional per-epoch high-pass filter (Hz). Empty/unset => no filtering.
    highpass_freq_str = strip(get(ENV, "HIGHPASS_FREQ", ""))
    highpass_freq = isempty(highpass_freq_str) ? nothing : parse(Float64, highpass_freq_str)
    alpha = 0.05

    # Create subject and trial tracking dataframes
    subject_trial_counts = DataFrame(subject = String[], modality = String[], total_trials = Int[])

    # Mirror exclusion from existing analysis scripts.
    # Sessions with a broken recording (23: swapped ground/reference, 36: extreme force).
    faulty = Set([
        ("023", "mecha"),
        ("036", "mecha"),
    ])
    # The data-driven SNR list in Pre-Processing/exclued_subs.json is deliberately
    # NOT applied here: the 1.5x occipital RMS cutoff is too strict for the
    # sensor-space model. Only genuinely broken recordings are dropped.
    excluded = faulty
    @info "Excluded subject/session pairs" n=length(excluded) pairs=sort(collect(excluded))

    # A missing CSD cache would otherwise be skipped file by file and only surface later as
    # "No EEG blocks loaded", which says nothing about the actual cause.
    if laplacian
        assert_laplacian_cache_exists(datadir, subjects, ["mecha", "elec"], epoch_stem, metadata_stem)
    end

    # run loading loop
    # Run each modality as an independent model/permutation pipeline.
    for ses in run_modes
    @info "Starting modality run" modality=ses channel

    all_blocks = Array{Float64, 3}[]
    all_events = DataFrame()
    times_ref = nothing
    channels_ref = nothing

    if ses == "combined"
        # Combined run loads both sessions per subject and stacks them
        # into one model with modality as a fixed effect.
        for subid in subjects
            for ses_inner in ("mecha", "elec")
                if (subid, ses_inner) in excluded
                    @info "Skipping excluded subject/session" subid ses=ses_inner
                    continue
                end

                loaded = load_subject_session(
                    datadir,
                    subid,
                    ses_inner,
                    requested_channels;
                    epoch_stem = epoch_stem,
                    metadata_stem = metadata_stem,
                    time_start = time_start,
                    time_stop = time_stop,
                    highpass_freq = highpass_freq,
                )
                if isnothing(loaded)
                    @info "Missing files, skipped." subid ses=ses_inner
                    continue
                end

                if isnothing(times_ref)
                    times_ref = loaded.times
                elseif length(times_ref) != length(loaded.times)
                    error("Time axis mismatch for sub-$subid ses-$ses_inner")
                end

                if isnothing(channels_ref)
                    channels_ref = loaded.channels
                elseif loaded.channels != channels_ref
                    error("Channel list mismatch for sub-$subid ses-$ses_inner")
                end

                # Keep trial-wise EEG and metadata aligned while stacking blocks.
                push!(all_blocks, loaded.data)
                append!(all_events, loaded.events; promote = true)
                @info "Loaded block (combined run)" subid ses=ses_inner n_trials=size(loaded.data, 3) n_channels=size(loaded.data, 1)

                # Track subject x session trial counts for downstream QC export.
                total_trials = size(loaded.data, 3)
                push!(subject_trial_counts, (subject = "sub-$subid", modality = ses_inner, total_trials = total_trials))
            end
        end
    else
        for subid in subjects
            if (subid, ses) in excluded
                @info "Skipping excluded subject/session" subid ses
                continue
            end

            loaded = load_subject_session(
                datadir,
                subid,
                ses,
                requested_channels;
                epoch_stem = epoch_stem,
                metadata_stem = metadata_stem,
                time_start = time_start,
                time_stop = time_stop,
                highpass_freq = highpass_freq,
            )
            if isnothing(loaded)
                @info "Missing files, skipped." subid ses
                continue
            end

            if isnothing(times_ref)
                times_ref = loaded.times
            elseif length(times_ref) != length(loaded.times)
                error("Time axis mismatch for sub-$subid ses-$ses")
            end

            if isnothing(channels_ref)
                channels_ref = loaded.channels
            elseif loaded.channels != channels_ref
                error("Channel list mismatch for sub-$subid ses-$ses")
            end

            # Keep trial-wise EEG and metadata aligned while stacking blocks.
            push!(all_blocks, loaded.data)
            append!(all_events, loaded.events; promote = true)
            @info "Loaded block" subid ses n_trials=size(loaded.data, 3) n_channels=size(loaded.data, 1)

            # Count trials for this subject and modality
            total_trials = size(loaded.data, 3)

            # Add to subject trial counts dataframe
            push!(subject_trial_counts, (subject = "sub-$subid", modality = ses, total_trials = total_trials))
        end
    end


    if isempty(all_blocks)
        @warn "No EEG blocks loaded for modality; skipping." modality=ses
        continue
    end

    data_epoched = cat(all_blocks...; dims = 3)  # channel x time x trial
    channels = Vector{String}(channels_ref)
    n_channels = length(channels)
    n_time = size(data_epoched, 2)

    # Guard the core assumption of this script: everything below (z-map shape, 1x1 adjacency,
    # per-time cluster table) is only valid for exactly one channel.
    n_channels == 1 || error("Expected exactly one channel after loading, got $n_channels: $(join(channels, ", "))")
    channels[1] == channel || error("Loaded channel '$(channels[1])' does not match the requested channel '$channel'.")


    # Modality-specific fixed effects:
    # - Mechanical includes force as an additive factor.
    # - Electrical keeps the original response/intensity model.
    model_formula = nothing
    if ses == "mecha"
        hasproperty(all_events, :force) || error("force column is required for modality 'mecha' but was not found.")

        # Drop trials with missing/empty/non-informative force values and keep EEG aligned.
        # This preserves strict row alignment by applying the exact same mask to metadata and EEG.
        n_trials_before = nrow(all_events)
        n_subjects_before = length(unique(all_events.sub))
        subjects_before = Set(string.(unique(all_events.sub)))

        forceq_str = lowercase.(strip.(string.(all_events.force)))
        keep_forceq = .!ismissing.(all_events.force) .&
                        .!(forceq_str .== "") .&
                        .!(forceq_str .== "nan") .&
                        .!(forceq_str .== "missing")

        dropped_trials = sum(.!keep_forceq)
        if dropped_trials > 0
            all_events = all_events[keep_forceq, :]
            data_epoched = data_epoched[:, :, keep_forceq]
        end

        n_trials_after = nrow(all_events)
        n_subjects_after = length(unique(all_events.sub))
        subjects_after = Set(string.(unique(all_events.sub)))
        removed_subjects = sort(collect(setdiff(subjects_before, subjects_after)))

        @info "Mechanical force filtering summary" dropped_trials n_trials_before n_trials_after n_subjects_before n_subjects_after removed_subjects
        n_trials_after > 0 || error("No trials left for modality 'mecha' after dropping missing force values.")
        n_subjects_after > 0 || error("No subjects left for modality 'mecha' after dropping missing force values.")

        # Recompute centered predictors after trial filtering so the
        # design matrix reflects the exact trial set used for model fit.
        all_events.intensity_level_centered = all_events.intensity_level .- mean(all_events.intensity_level)
        all_events.block_centered = all_events.block .- mean(all_events.block)
        # `force` can be Union{Missing,Float64}; after filtering there should
        # be only valid numeric trials, so center with skipmissing explicitly.
        mean_force = mean(skipmissing(all_events.force))
        all_events.force_centered = Float64.(all_events.force) .- mean_force

        # Use additive force effect for mechanical runs.
        model_formula = @formula(0 ~ 1 + intensity_level_centered * response_decoded + force_centered + (1 | sub) + (0 + block_centered | sub))
    elseif ses == "combined"
        # Combined model across mecha+elec: include modality fixed effect.
        all_events.intensity_level_centered = all_events.intensity_level .- mean(all_events.intensity_level)
        all_events.block_centered = all_events.block .- mean(all_events.block)
        all_events.modality = categorical(string.(all_events.modality))
        model_formula = @formula(0 ~ 1 + modality * intensity_level_centered * response_decoded + (1 | sub) + (0 + block_centered | sub))
    else
        # Electrical branch does not include force in the model.
        all_events.intensity_level_centered = all_events.intensity_level .- mean(all_events.intensity_level)
        all_events.block_centered = all_events.block .- mean(all_events.block)
        model_formula = @formula(0 ~ 1 + intensity_level_centered * response_decoded + (1 | sub) + (0 + block_centered | sub))
    end
    formula_string = string(model_formula)


    # Run bookkeeping
    run_modalities = ses == "combined" ? ["mecha", "elec"] : [ses]
    run_id, run_dir, runs_root, modality_tag, formula_id = make_run_paths(
        outroot,
        run_modalities,
        formula_string;
        baseline_tag = baseline_tag,
        epoch_tag = epoch_stem,
    )
    # The channel is the defining property of this analysis, so it goes in the folder name.
    run_id = "$(run_id)__ch-$(sanitize_token(channel))"
    if coef_selector != "all"
        # Make the selected coefficient visible directly in the run folder name.
        run_id = "$(run_id)__coef-$(sanitize_token(coef_selector))"
    end
    run_dir = joinpath(runs_root, run_id)
    mkpath(run_dir)
    @info "Run output directory" run_id run_dir modality=ses channel

    # Cluster-forming threshold on |z| (same value as the spatiotemporal script).
    threshold = 1.96

    startup_snapshot = build_startup_env_snapshot(
        datadir = datadir,
        resdir = resdir,
        outroot = outroot,
        subjects = subjects,
        excluded_pairs = excluded,
        run_modes = run_modes,
        epoch_stem = epoch_stem,
        metadata_stem = metadata_stem,
        baseline_tag = baseline_tag,
        n_permutations = n_permutations,
        seed = seed,
        time_start = time_start,
        time_stop = time_stop,
        highpass_freq = highpass_freq,
        plot_start = plot_start,
        plot_stop = plot_stop,
        coef_selector = coef_selector,
        laplacian = laplacian,
        run_modalities = run_modalities,
        run_id = run_id,
        run_dir = run_dir,
        formula_string = formula_string,
        channel = channel,
        threshold = threshold,
    )
    log_and_save_startup_env_snapshot(run_dir, startup_snapshot)

    # Trial counts per subject x session for this run.
    subject_trial_counts_file = joinpath(run_dir, "subject_trial_counts.csv")
    CSV.write(subject_trial_counts_file, subject_trial_counts)
    @info "Saved subject and trial counts" path=subject_trial_counts_file

    # ERP of the analysed channel, split by response (descriptive preview).
    data_2d = Matrix(data_epoched[1, :, :])  # time x trial
    plot_erp_helper(times_ref, data_2d, all_events, run_dir, channel)

    # fit model
    m = fit(
        UnfoldModel,
        [Any => (model_formula, times_ref)],
        all_events,
        data_epoched,
    )

    @info "Model fit complete" modality=ses n_trials=size(data_epoched, 3) n_timepoints=size(data_epoched, 2)
    target_coefs, coef_names = resolve_target_coefficients(m)
    if ses == "mecha"
        # Mechanical model: also test force-related coefficient(s).
        target_coefs = append_factor_coefficients(target_coefs, coef_names, "force")
        # `force` is modeled as a centered numeric predictor in this script.
        # Append explicit force matches to ensure it is selectable via COEF=force.
        force_coefs = [c for c in coef_names if occursin("force", lowercase(c))]
        for c in force_coefs
            if !(c in target_coefs)
                push!(target_coefs, c)
            end
        end
    elseif ses == "combined"
        # Combined model: test modality coefficient(s) as in unfold_cluster_permutation.jl.
        target_coefs = append_factor_coefficients(target_coefs, coef_names, "modality")
    end

    @info "All coeffs:" target_coefs
    # Optional COEF filtering to avoid long runtimes when only one effect is needed.
    if coef_selector != "all"
        lowered = lowercase.(target_coefs)
        keep = falses(length(target_coefs))
        if coef_selector == "intensity_response"
            # Restrict to the single intended interaction term only.
            # Expected coefficient label:
            # "intensity_level_centered & response_decoded: Yes"
            interaction_exact = "intensity_level_centered & response_decoded: yes"
            keep .= lowered .== interaction_exact
        elseif coef_selector == "modality_response"
            interaction_exact = "modality: mechanical & response_decoded: yes"
            keep .= lowered .== interaction_exact
        elseif coef_selector == "modality_intensity"
            interaction_exact = "modality: mechanical & intensity_level_centered"
            keep .= lowered .== interaction_exact
        elseif coef_selector == "threeway_interaction"
            interaction_exact = "modality: mechanical & intensity_level_centered & response_decoded: yes"
            keep .= lowered .== interaction_exact
        elseif coef_selector in ("response_decoded", "response")
            keep .= occursin.("response_decoded", lowered)
        elseif coef_selector in ("intensity_level_centered", "intensity")
            keep .= occursin.("intensity_level_centered", lowered)
        elseif coef_selector in ("force", "force_centered")
            keep .= occursin.("force", lowered)
        elseif coef_selector == "modality"
            keep .= occursin.("modality", lowered)
        else
            # Fallback: treat COEF as a case-insensitive substring.
            keep .= occursin.(coef_selector, lowered)
        end

        selected_coefs = target_coefs[keep]
        isempty(selected_coefs) && error("COEF='$coef_selector_raw' did not match any target coefficient. Available: $(join(target_coefs, ", "))")
        target_coefs = selected_coefs
    end
    @info "Target coefficients resolved" modality=ses target=target_coefs available=coef_names

    time_selection = 1:length(times_ref)

    # A single channel has no spatial neighbours to grow a cluster into. A 1x1 all-false
    # adjacency therefore turns find_connected_clusters_2d into a purely temporal cluster
    # search (its spatial loop finds nothing, its temporal loop still links adjacent samples),
    # while keeping the mass statistic and the max-cluster null identical to the
    # spatiotemporal run in unfold_cluster_permutation_clusterMass.jl.
    adjacency = falses(1, 1)

    # Extract coefficients
    coefs = coeftable(m)

    # Run permutation + save artifacts for each selected coefficient.
    for (k, coef_name) in enumerate(target_coefs)
        coefficient, _ = coefficient_index_from_name(m, coef_name)
        @info "Running permutation for coefficient" modality=ses channel coef_name coefficient current=k total=length(target_coefs)
        # Use a unique directory per coefficient so repeated loop iterations
        # never overwrite plots/results from earlier coefficients.
        coef_tag = "$(lpad(string(k), 2, '0'))__$(sanitize_token(coef_name))"
        coef_dir = joinpath(run_dir, "coefficients", coef_tag)
        mkpath(coef_dir)
        @info "Coefficient output directory" coef_name coef_dir

        println("Generating permutation (npermut=$(n_permutations)) distribution...")
        perm_start_ns = time_ns()
        permuted = lmm_permutations(MersenneTwister(42), m, data_epoched, coefficient;
            n_permutations = n_permutations,
            lmm_statistic = :z,
            time_selection = time_selection
        )
        perm_elapsed_s = (time_ns() - perm_start_ns) / 1e9
        @info "Permutation procedure finished" coef_name permutation_seconds=round(perm_elapsed_s; digits=3) n_permutations threshold

        # Check uncorrected effects before running permutation test
        # Show available columns
        println("Available columns: $(names(coefs))")
        coef = filter(r -> r.coefname == coef_name, coefs)
        println("Rows for response: $(nrow(coef))")

        # Check how many unique channels and times we have
        unique_channels = sort(unique(coef.channel))
        unique_times = sort(unique(coef.time))
        println("Unique channels: $(length(unique_channels))")
        println("Unique times: $(length(unique_times))")

        # Compute z-values from estimate/stderror, handling missing values
        # Replace Nothing with NaN for computation
        estimate_vec = [isnothing(x) ? NaN : Float64(x) for x in coef.estimate]
        stderror_vec = [isnothing(x) ? NaN : Float64(x) for x in coef.stderror]
        z_vec = estimate_vec ./ stderror_vec

        coef[!, :z] = z_vec

        # Average across groups if there are multiple entries per channel x time
        z_summary = combine(
            groupby(coef, [:channel, :time]),
            :z => (x -> mean(filter(!isnan, x))) => :z
        )
        sort!(z_summary, [:channel, :time])

        # Replace NaN with 0 for visualization
        z_summary.z = replace(z_summary.z, NaN => 0.0)

        # Reshape z-values to channels x times (here: 1 x n_time)
        n_ch_coef = length(unique_channels)
        n_t_coef = length(unique_times)
        z_vals = reshape(z_summary.z, n_t_coef, n_ch_coef)'  # (channels x times)
        size(z_vals) == (1, n_time) || error("Expected a (1, $n_time) z-map for the single-channel " *
                                             "test, got $(size(z_vals)).")

        # Check for significant z-values (|z| > 2.0)
        n_sig_uncorrected = sum(abs.(z_vals) .> 2.0)
        max_z = maximum(abs.(z_vals))
        peak_t = argmax(abs.(vec(z_vals)))

        println("\n=== Uncorrected Effects (before permutation) ===")
        println("Channel: $channel")
        println("Max |z|: $(round(max_z, digits=2))")
        println("Peak location: t = $(round(unique_times[peak_t], digits=3))s")
        println("Samples with |z| > 2.0: $(n_sig_uncorrected) / $(length(z_vals))")
        println("Samples with |z| > 1.96: $(sum(abs.(z_vals) .> 1.96)) (uncorrected p < 0.05)")

        # Cluster-mass permutation test, temporal only.
        observed = Matrix(z_vals)

        # Cluster formation on the OBSERVED map, logged before the permutation
        # p-values are computed, so the run log records how the effect groups.
        _, pos_stats = find_connected_clusters_2d(observed, adjacency, threshold)
        _, neg_stats = find_connected_clusters_2d(-observed, adjacency, threshold)
        obs_stats = [pos_stats; neg_stats]
        if isempty(obs_stats)
            println("[$coef_name] observed clusters: none above threshold $threshold")
        else
            top = sort(obs_stats; rev = true)[1:min(end, 5)]
            println("[$coef_name] observed clusters: n=$(length(obs_stats))  " *
                    "max_mass=$(round(maximum(obs_stats); digits = 1))  top5=$(round.(top; digits = 1))")
        end

        pvals_clustermass, cluster_ids, null_max_stats = spatiotemporal_cluster_pvalues(MersenneTwister(1), observed, permuted, adjacency, threshold)

        # Only show significant clusters (p < 0.05); everything else -> NaN
        cluster_display = Float64.(cluster_ids)
        cluster_display[(pvals_clustermass .>= alpha) .| (cluster_ids .== 0)] .= NaN

        # Tidy per-cluster summary (time window, mass, peak z, exact p).
        z_row = vec(observed)
        cluster_ids_row = vec(cluster_ids)
        pvals_row = vec(pvals_clustermass)
        cluster_table = build_temporal_cluster_table(times_ref, z_row, cluster_ids_row, pvals_row)

        println("\n=== Temporal clusters ($channel, $coef_name) ===")
        if nrow(cluster_table) == 0
            println("  no cluster above threshold $threshold")
        else
            for r in eachrow(cluster_table)
                marker = r.pvalue < alpha ? "*" : " "
                println("  $marker cluster $(r.cluster_id) [$(r.direction)]: " *
                        "$(round(r.t_start_s * 1000, digits = 1)) to $(round(r.t_stop_s * 1000, digits = 1)) ms  " *
                        "n=$(r.n_samples)  mass=$(round(r.mass, digits = 1))  " *
                        "peak z=$(round(r.peak_z, digits = 2)) at $(round(r.peak_time_s * 1000, digits = 1)) ms  " *
                        "p=$(round(r.pvalue, digits = 5))")
            end
            println("  (smallest attainable p = $(round(1 / (n_permutations + 1), digits = 5)))")
        end

        cluster_csv = joinpath(coef_dir, "temporal_clusters.csv")
        CSV.write(cluster_csv, cluster_table)
        @info "Saved temporal cluster table" path=cluster_csv n_clusters=nrow(cluster_table)

        # z-trace figure with threshold lines and significant cluster spans.
        save_temporal_z_trace_plot(times_ref, z_row, cluster_table, threshold, coef_dir,
                                   channel, coef_name; alpha = alpha)

        # ERP by response with the significant cluster windows shaded.
        save_permutation_plot(
            times_ref,
            pvals_row,
            data_2d,
            all_events,
            coef_dir,
            channel,
            coef_name;
            alpha = alpha,
            plot_start = plot_start,
            plot_stop = plot_stop,
        )

        # Per-sample p-values (single channel -> tidy time/pvalue table).
        result = make_pvalue_result_df(times_ref, pvals_clustermass, channels)
        save_pvalue_csv(result, coef_dir, channel, coef_name)
        save_permutation_timing_csv(coef_dir, channel, coef_name, perm_elapsed_s,
                                    n_permutations, threshold, seed)

        # Same NPZ layout as the spatiotemporal run (arrays are 1 x n_time here), so
        # extract_cluster_stats.py reads these runs unchanged.
        results_npz_path = joinpath(coef_dir, "cluster_results.npz")
        npzwrite(results_npz_path, Dict(
            "z_vals" => Matrix(observed),
            "cluster_display" => cluster_display,
            "cluster_ids" => Float64.(cluster_ids),      # signed: +k pos, -k neg, 0 = none
            "pvals_clustermass" => pvals_clustermass,    # per-sample; 1.0 outside any cluster
            "null_max_stats" => null_max_stats,          # max cluster mass per permutation
            "times" => collect(times_ref)
        ))
        @info "Saved numeric results" path=results_npz_path

        # Save channel name next to the NPZ payload for reproducible reloads.
        channels_txt_path = joinpath(coef_dir, "channels.txt")
        open(channels_txt_path, "w") do f
            for ch in channels
                println(f, ch)
            end
        end
        @info "Saved channel list" path=channels_txt_path
    end
    end
end

main()
