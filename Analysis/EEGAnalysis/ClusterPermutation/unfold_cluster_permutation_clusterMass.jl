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
using MAT
using LinearAlgebra
using UnfoldSim
using NPZ
using UnfoldMixedModels
using MixedModelsPermutations
using ClusterDepth
ext = Base.get_extension(UnfoldStats, :UnfoldStatsMixedModelsPermutationsExt)
lmm_permutations = ext.lmm_permutations
get_lmm_statistic = ext.get_lmm_statistic

# import the new module 
include(joinpath(@__DIR__, "cluster_permutation_helpers.jl"))
using .ClusterPermutationHelpers
include(joinpath(@__DIR__, "read_adjacency.jl"))
include(joinpath(@__DIR__, "cluster_mass_test.jl")) 

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
    run_modalities::AbstractVector{<:AbstractString},
    run_id::AbstractString,
    run_dir::AbstractString,
    formula_string::AbstractString,
    plot_channel::AbstractString,
    plot_channels::AbstractVector{<:AbstractString},
)
    # Keep a deterministic, explicit whitelist of env values used by this script.
    snapshot = Pair{String, String}[
        "JULIA_DEPOT_PATH_raw" => env_value_or_marker("JULIA_DEPOT_PATH"),
        "DATADIR_raw" => env_value_or_marker("DATADIR"),
        "RESDIR_raw" => env_value_or_marker("RESDIR"),
        "MODALITY_raw" => env_value_or_marker("MODALITY"),
        "SUBJECTS_raw" => env_value_or_marker("SUBJECTS"),
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
        "PLOT_CHANNEL_raw" => env_value_or_marker("PLOT_CHANNEL"),
        "PLOT_CHANNELS_raw" => env_value_or_marker("PLOT_CHANNELS"),
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
        "plot_channel_resolved" => plot_channel,
        "plot_channels_resolved" => join(plot_channels, ","),
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
Log the spatial adjacency actually handed to the cluster test, in the data's own
channel order.

The rows of the adjacency must line up with the rows of the z-map: `adjacency[i, j]`
has to describe the neighbourhood of `data_channels[i]`, where `data_channels` is
the loader's (sorted) channel list from
cluster_permutation_helpers.jl::list_channels_in_file. If they disagree, the
cluster search connects channels that are not scalp neighbours.

`data_channels` : channel names in z-map row order (from the loader)
`adj_labels`    : channel names the adjacency was built in (2nd return of read_fieldtrip_neighbors)
`adjacency`     : the matrix passed to the cluster test
"""
function report_adjacency_alignment(data_channels, adj_labels, adjacency)
    println("\n================ ADJACENCY ALIGNMENT ================")
    println("data rows (z_vals row order) n=$(length(data_channels)): $(join(data_channels[1:min(end, 10)], ", ")) ...")
    println("adjacency rows               n=$(length(adj_labels)): $(join(adj_labels[1:min(end, 10)], ", ")) ...")
    n = min(length(data_channels), length(adj_labels))
    mismatched = [i for i in 1:n if data_channels[i] != adj_labels[i]]
    println("rows where data channel != adjacency channel: $(length(mismatched)) / $n")
    println("edges: $(sum(adjacency) ÷ 2) | mean degree: $(round(sum(adjacency) / n; digits = 2))")

    # Spot-check a few neighbourhoods, labelled with the DATA names, because that
    # is how the cluster search walks them.
    for probe in ("Cz", "C4", "CP4")
        i = findfirst(==(probe), data_channels)
        isnothing(i) && continue
        println("neighbours of $probe (row $i): $(join(data_channels[findall(@view adjacency[i, :])], ", "))")
    end
    println("=====================================================\n")
    return nothing
end

function main()

    # Channel neighbour definition. The adjacency itself is built further down,
    # once the loader has told us the channel order of the data: the matrix rows
    # must line up with the rows of the z-map the cluster search walks, and the
    # loader returns channels sorted, not in any hard-coded layout order.
    subject_id = "013"
    custom_file = "sub-" * subject_id * "_ses-mecha_task-NT_channel_adjacency.mat"


    # specify loading parameters
    datadir = get(ENV, "DATADIR", "")

    if isempty(strip(datadir))
        error("DATADIR is not set. Example: export DATADIR=/data/p_02989/shared_workspace/datadir")
    end

    epoch_stem = get(ENV, "EPOCH_STEM", "epochs")
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
    # Keep cluster-mass outputs isolated under a dedicated subfolder so
    # make_run_paths(...,"runs") resolves to cluster_permutation/cluster_mass/runs.
    outroot = isempty(strip(resdir)) ?
        joinpath(pwd(), "cluster_permutation", "cluster_mass") :
        joinpath(resdir, "Statistics", "eeg", "cluster_permutation", "cluster_mass")
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
    # Collect per-run/per-coefficient traces for end-of-script elec/mecha/combined figures.
    comparison_store = Dict{String, Dict{String, NamedTuple}}()

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

    # channel reorder for later plotting
    channels_ordered = [
    # Frontal
    "Fp1","AF7","AF3","F7","F5","F3","F1",
    "FPz","AFz","Fz",
    "Fp2","AF4","AF8","F2","F4","F6","F8",

    # Frontocentral
    "FT9","FT7","FC5","FC3","FC1",
    "FC2","FC4","FC6","FT8","FT10",

    # Central (Somatosensory)
    "T7","C5","C3","C1",
    "Cz",
    "C2","C4","C6","T8",

    # Centroparietal
    "TP7","CP5","CP3","CP1",
    "CPz",
    "CP2","CP4","CP6","TP8",

    # Parietal
    "P7","P5","P3","P1",
    "Pz",
    "P2","P4","P6","P8",

    # Parieto-occipital
    "PO7","PO3",
    "PO4","PO8",

    # Occipital
    "O1","O2",

    # Remaining lateral frontal (if you want them at the end)
    "F9","F10"
    ]


    # run loading loop
    # Run each modality as an independent model/permutation pipeline.
    for ses in run_modes
    @info "Starting modality run" modality=ses

    all_blocks = Array{Float64, 3}[]
    all_events = DataFrame()
    times_ref = nothing
    channels_ref = nothing
    requested_channels = nothing

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

    # Build the spatial adjacency in the DATA's channel order, so adjacency[i, j]
    # describes the neighbourhood of the channel in row i of the z-map. Anything
    # else makes the cluster search walk a scrambled scalp graph.
    adjacency, matched_labels = read_fieldtrip_neighbors(
        joinpath(@__DIR__, custom_file);
        channel_order = channels,
    )
    matched_labels == channels || error("Adjacency channel order does not match loaded channels.")
    # Channels missing from the neighbour file would silently get no neighbours,
    # i.e. they could only ever cluster with themselves across time.
    isolated = [c for (i, c) in enumerate(channels) if !any(@view adjacency[i, :])]
    isempty(isolated) || error("Channels without any neighbour in $custom_file: $(join(isolated, ", ")).")
    report_adjacency_alignment(channels, matched_labels, adjacency)



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
        model_formula = @formula(0 ~ 1 + modality * intensity_level_centered * response_decoded + (1 | sub)) # + (0 + block_centered | sub))
    else
        # Electrical branch does not include force in the model.
        all_events.intensity_level_centered = all_events.intensity_level .- mean(all_events.intensity_level)
        all_events.block_centered = all_events.block .- mean(all_events.block)
        model_formula = @formula(0 ~ 1 + intensity_level_centered * response_decoded + (1 | sub) + (0 + block_centered | sub))
    end
    formula_string = string(model_formula)


    # PLOT 
    run_modalities = ses == "combined" ? ["mecha", "elec"] : [ses]
    run_id, run_dir, runs_root, modality_tag, formula_id = make_run_paths(
        outroot,
        run_modalities,
        formula_string;
        baseline_tag = baseline_tag,
        epoch_tag = epoch_stem,
    )
    if coef_selector != "all"
        # Make the selected coefficient visible directly in the run folder name.
        run_id = "$(run_id)__coef-$(sanitize_token(coef_selector))"
        run_dir = joinpath(runs_root, run_id)
    end
    mkpath(run_dir)
    @info "Run output directory" run_id run_dir modality=ses

    # Keep PLOT_CHANNEL as the backwards-compatible default target.
    plot_channel_default = channels[9]
    plot_channel = String(strip(get(ENV, "PLOT_CHANNEL", plot_channel_default)))
    plot_channel in channels || error("PLOT_CHANNEL '$plot_channel' not found in analyzed channels: $(join(channels, ", "))")

    # PLOT_CHANNELS allows plotting multiple channels in one figure.
    plot_channels = parse_plot_channels(channels, plot_channel)
    @info "Plot channel selection" n_selected=length(plot_channels) selected=plot_channels

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
        run_modalities = run_modalities,
        run_id = run_id,
        run_dir = run_dir,
        formula_string = formula_string,
        plot_channel = plot_channel,
        plot_channels = plot_channels,
    )
    log_and_save_startup_env_snapshot(run_dir, startup_snapshot)

    # Save a plain ERP preview for the first selected channel.
    preview_channel = plot_channels[1]
    preview_idx = findfirst(==(preview_channel), channels)
    data_preview = Matrix(data_epoched[preview_idx, :, :])  # time x trial

    plot_erp_helper(times_ref, data_preview, all_events, run_dir, preview_channel)

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

    threshold = 1.96
    time_selection = 1:length(times_ref)
    
    # Extract coefficients
    coefs = coeftable(m)

    # if we want to use contrasts we need to define it a bit differently
    

    # Run permutation + save artifacts for each selected coefficient.
    for (k, coef_name) in enumerate(target_coefs)
        coefficient, _ = coefficient_index_from_name(m, coef_name)
        @info "Running permutation for coefficient" modality=ses coef_name coefficient current=k total=length(target_coefs)
        # Use a unique directory per coefficient so repeated loop iterations
        # never overwrite plots/results from earlier coefficients.
        coef_tag = "$(lpad(string(k), 2, '0'))__$(sanitize_token(coef_name))"
        coef_dir = joinpath(run_dir, "coefficients", coef_tag)
        mkpath(coef_dir)
        @info "Coefficient output directory" coef_name coef_dir

        println("Generating permutation (npermut=$(n_permutations)) distribution...")
        permuted = lmm_permutations(MersenneTwister(42), m, data_epoched, coefficient; 
            n_permutations = n_permutations,
            lmm_statistic = :z,
            time_selection = time_selection
        )

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

        # Average across groups if there are multiple entries per channel×time
        z_summary = combine(
            groupby(coef, [:channel, :time]),
            :z => (x -> mean(filter(!isnan, x))) => :z
        )
        sort!(z_summary, [:channel, :time])

        # Replace NaN with 0 for visualization
        z_summary.z = replace(z_summary.z, NaN => 0.0)

        # Reshape z-values to channels × times
        n_ch_coef = length(unique_channels)
        n_t_coef = length(unique_times)
        z_vals = reshape(z_summary.z, n_t_coef, n_ch_coef)'  # (channels × times)

        # Check for significant z-values (|z| > 2.0)
        n_sig_uncorrected = sum(abs.(z_vals) .> 2.0)
        max_z = maximum(abs.(z_vals))
        peak_idx = argmax(abs.(z_vals))
        peak_ch, peak_t = Tuple(CartesianIndices(z_vals)[peak_idx])

        println("\n=== Uncorrected Effects (before permutation) ===")
        println("Max |z|: $(round(max_z, digits=2))")
        println("Peak location: channel $(channels[peak_ch]) at t = $(round(unique_times[peak_t], digits=3))s")
        println("Samples with |z| > 2.0: $(n_sig_uncorrected) / $(length(z_vals))")
        println("Samples with |z| > 1.96: $(sum(abs.(z_vals) .> 1.96)) (uncorrected p < 0.05)")

        # Quick visualization of z-values
        fig_z = Figure(size = (800, 400))
        ax_z = Axis(fig_z[1, 1],
            title = "Uncorrected z-values for stimtype effect",
            xlabel = "Time (s)",
            ylabel = "Channel")
        hm = heatmap!(ax_z, collect(unique_times), 1:n_ch_coef, z_vals', colormap = :RdBu, colorrange = (-4, 4))
        Colorbar(fig_z[1, 2], hm, label = "z-value")
        fig_z

        # Save this diagnostic figure in the run-specific folder.
        fig_z_png = joinpath(coef_dir, "uncorrected_z_values.png")
        save(fig_z_png, fig_z)
        @info "Saved figure" path=fig_z_png

        # 3. Run cluster-mass permutation test
        observed = z_vals

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

        # Only show significant clusters (p < 0.05); everything else → NaN (lightgray)
        cluster_display = Float64.(cluster_ids)
        cluster_display[(pvals_clustermass .>= 0.05) .| (cluster_ids .== 0)] .= NaN

        sig_ids = sort(unique(cluster_ids[pvals_clustermass .< 0.05]))
        println("Significant cluster IDs: $sig_ids")

        # Log the exact p and mass per significant cluster so the run log alone is
        # enough to report the statistics; every sample of a cluster shares its p.
        for cid in sig_ids
            mask = cluster_ids .== cid
            p_cid = first(pvals_clustermass[mask])
            println("  cluster $cid: p = $(round(p_cid, digits = 5))  " *
                    "mass = $(round(sum(z_vals[mask]), digits = 1))  n_samples = $(sum(mask))")
        end

        n_clusters = maximum(abs.(cluster_ids))
        crange = n_clusters == 0 ? (-1.0, 1.0) : (-Float64(n_clusters), Float64(n_clusters))
        perm = indexin(channels_ordered, channels)

        z_vals_sorted = z_vals[perm, :]
        cluster_display_sorted = cluster_display[perm, :]


        sig_ids = sort(unique(cluster_ids[pvals_clustermass .< 0.05]))
        println("Significant cluster IDs: $sig_ids")

        n_clusters = maximum(abs.(cluster_ids))
        crange = n_clusters == 0 ? (-1.0, 1.0) : (-Float64(n_clusters), Float64(n_clusters))

        # 4. Visualize
        fig = Figure(size = (1600, 800))

        # Panel 1: Observed z-values with threshold contours
        ax1 = Axis(fig[1, 1], title = "Observed z-values", xlabel = "Time (s)", ylabel = "Channel",
                yticks = (1:n_channels, channels_ordered))
        hm1 = heatmap!(ax1, collect(times_ref), 1:n_channels, z_vals_sorted', colormap = :RdBu, colorrange = (-6, 6))
        Colorbar(fig[1, 2], hm1, label = "z-value")
        contour!(ax1, collect(times_ref), 1:n_channels, z_vals_sorted', levels = [threshold],  color = (:red,  0.7), linewidth = 1.5)
        contour!(ax1, collect(times_ref), 1:n_channels, z_vals_sorted', levels = [-threshold], color = (:blue, 0.7), linewidth = 1.5)

        # Panel 2: Significant cluster labels only (warm = positive, cool = negative, gray = non-sig / no cluster)
        ax2 = Axis(fig[1, 3], title = "Significant clusters (p < 0.05)",
                xlabel = "Time (s)", ylabel = "Channel", yticks = (1:n_channels, channels_ordered))
        hm2 = heatmap!(ax2, collect(times_ref), 1:n_channels, cluster_display_sorted',
                    colormap = :RdBu, colorrange = crange, nan_color = :lightgray)
        Colorbar(fig[1, 4], hm2, label = "cluster ID")

        fig

        # Save the main cluster visualization in the run-specific folder.
        fig_cluster_png = joinpath(coef_dir, "cluster_significance_overview.png")
        save(fig_cluster_png, fig)
        @info "Saved figure" path=fig_cluster_png

        # Save numeric arrays in run_dir so outputs are not written to the working directory.
        results_npz_path = joinpath(coef_dir, "cluster_results.npz")
        # cluster_display is masked at p < 0.05, so the exact p-values would be lost if
        # only that were stored. Keep the unmasked ids, the per-sample p-values and the
        # permutation null so downstream analysis can report exact p and re-threshold.
        npzwrite(results_npz_path, Dict(
            "z_vals" => z_vals,
            "cluster_display" => cluster_display,
            "cluster_ids" => Float64.(cluster_ids),      # signed: +k pos, -k neg, 0 = none
            "pvals_clustermass" => pvals_clustermass,    # per-sample; 1.0 outside any cluster
            "null_max_stats" => null_max_stats,          # max cluster mass per permutation
            "times" => collect(times_ref)
        ))
        @info "Saved numeric results" path=results_npz_path

        # Save channel names next to the NPZ payload for reproducible reloads.
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
