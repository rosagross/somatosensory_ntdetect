module ClusterPermutationHelpers

using CSV
using CairoMakie
using CategoricalArrays
using DataFrames
using DSP
using HDF5
using JSON
using Random
using Statistics
using StatsModels
using Dates
using SHA
using UnfoldMixedModels
using MixedModelsPermutations, ClusterDepth
using UnfoldStats
using UnfoldMakie
using Distributions

export parse_subjects,
       load_excluded_pairs,
       parse_modalities,
       parse_channels,
       list_channels_in_file,
       non_eeg_channel_set,
       resolve_channels_for_file,
       as_bool,
       normalize_sub!,
       normalize_modality!,
       build_paths,
       resolve_epoch_stem_with_laplacian,
       assert_laplacian_cache_exists,
       crop_data_to_time_window,
       highpass_filter_epochs,
       resolve_baseline_window,
       load_subject_session,
       coefficient_index_from_name,
       resolve_target_coefficients,
       append_factor_coefficients,
       plot_erp_helper,
       save_erp_plot,
       significant_time_clusters,
       parse_plot_channels,
       pvals_channel_time,
       save_permutation_plot_multichannel,
       make_pvalue_result_df,
       save_permutation_plot,
       draw_clustered_erp_panel!,
       save_cross_modality_comparison_plots,
       save_pvalue_csv,
       save_permutation_timing_csv,
       sanitize_token,
       make_run_paths,
       save_run_manifest_and_index

"""
Parse subject list from ENV["SUBJECTS"].
Accepted format: "013,014,015" or "13 14 15".
Default: 013:052
"""
function parse_subjects()
    raw = get(ENV, "SUBJECTS", "")
    # padding string with 0 in the beginning (e.g. 015 for sub 15)
    if isempty(strip(raw))
        return [lpad(string(s), 3, '0') for s in 13:52]
    end
    tokens = split(replace(strip(raw), "," => " "))
    return [lpad(string(parse(Int, t)), 3, '0') for t in tokens]
end

# Exclusion list is shared with the Python pipeline; resolved relative to this file
# (Analysis/EEGAnalysis/ClusterPermutation -> three levels up is the repo root) so it does
# not depend on WDDIR being exported.
const EXCLUDED_SUBS_JSON = normpath(joinpath(@__DIR__, "..", "..", "..",
                                             "Pre-Processing", "exclued_subs.json"))

"""
Read Pre-Processing/exclued_subs.json and return a Set of (subid, modality) pairs.

Subject IDs are zero-padded to 3 characters so they compare equal to parse_subjects()
output - an unpadded "30" would silently never match and the exclusion would be a no-op.
Errors (rather than returning an empty set) if the file or the "exclude" key is missing,
so a broken path cannot quietly disable the exclusion.
"""
function load_excluded_pairs(path::AbstractString = EXCLUDED_SUBS_JSON)
    isfile(path) || error("Subject exclusion file not found: $path")
    payload = JSON.parsefile(path)
    haskey(payload, "exclude") || error("'exclude' key missing from $path")
    pairs = Set{Tuple{String,String}}()
    for (modality, subs) in payload["exclude"]
        modality in ("elec", "mecha") || error("Unknown modality '$modality' in $path")
        for s in subs
            push!(pairs, (lpad(string(s), 3, '0'), String(modality)))
        end
    end
    return pairs
end

"""
Parse modality list from ENV["MODALITIES"].
Default: ["mecha", "elec"]
"""
function parse_modalities()
    raw = get(ENV, "MODALITIES", "mecha,elec")
    tokens = split(replace(strip(raw), "," => " "))
    return [strip(m) for m in tokens if !isempty(strip(m))]
end

"""
Parse channel selection from ENV.
- CHANNELS=all -> all channels found in each EEG file.
- CHANNELS="C6,Cz" -> specific subset.
- Fallback to CHANNEL for backward compatibility.
"""
function parse_channels()
    raw = get(ENV, "CHANNELS", get(ENV, "CHANNEL", "C6"))
    s = lowercase(strip(raw))
    if s == "all"
        return nothing
    end
    tokens = split(replace(strip(raw), "," => " "))
    channels = [String(strip(ch)) for ch in tokens if !isempty(strip(ch))]
    isempty(channels) && error("No valid channels parsed from CHANNELS/CHANNEL.")
    return channels
end

function list_channels_in_file(eeg_file::String)
    return h5open(eeg_file, "r") do h5
        chs = String[]
        for name in keys(h5)
            name_str = String(name)
            if haskey(h5, "$name_str/data") && haskey(h5, "$name_str/times")
                push!(chs, name_str)
            end
        end
        sort!(chs)
        return chs
    end
end

function non_eeg_channel_set()
    raw = get(ENV, "NON_EEG_CHANNELS", "ECG,RESP,CNAP,VEOG")
    toks = split(replace(strip(raw), "," => " "))
    return Set([String(strip(t)) for t in toks if !isempty(strip(t))])
end

function resolve_channels_for_file(eeg_file::String, requested_channels::Union{Nothing, AbstractVector{<:AbstractString}})
    available = list_channels_in_file(eeg_file)
    isempty(available) && error("No channels with /data and /times found in $eeg_file")
    if isnothing(requested_channels)
        non_eeg = non_eeg_channel_set()
        eeg_only = [ch for ch in available if !(ch in non_eeg)]
        isempty(eeg_only) && error("No EEG channels left after excluding NON_EEG_CHANNELS=$(join(collect(non_eeg), ","))")
        if length(eeg_only) != length(available)
            @info "Excluding non-EEG channels from CHANNELS=all" excluded=sort(collect(intersect(Set(available), non_eeg))) n_total=length(available) n_eeg=length(eeg_only)
        end
        return eeg_only
    end
    missing = setdiff(requested_channels, available)
    isempty(missing) || error("Requested channels not found in $eeg_file: $(join(missing, ", "))")
    return requested_channels
end

function as_bool(v)
    if v isa Bool
        return v
    elseif ismissing(v)
        return false
    elseif v isa Integer
        return v != 0
    elseif v isa AbstractString
        s = lowercase(strip(v))
        return s in ("1", "true", "t", "yes", "y")
    end
    return false
end

function normalize_sub!(df::DataFrame, subid::String)
    if hasproperty(df, :sub)
        df.sub = lpad.(string.(df.sub), 3, '0')
    else
        df.sub = fill(subid, nrow(df))
    end
    return df
end

function normalize_modality!(df::DataFrame, ses::AbstractString)
    if !hasproperty(df, :modality)
        label = ses == "mecha" ? "Mechanical" : (ses == "elec" ? "Electrical" : ses)
        df.modality = fill(label, nrow(df))
    end
    return df
end

function build_paths(datadir::String, subid::String, ses::AbstractString, epoch_stem::String, metadata_stem::String)
    base = joinpath(datadir, "derivatives", "eeg-preproc", "sub-$subid", "ses-$ses")
    eeg_file = joinpath(base, "sub-$(subid)_ses-$(ses)_task-NT_lfreqori-0.1_lfreq-1_hfreq-40_$(epoch_stem).h5")
    meta_file = joinpath(base, "sub-$(subid)_ses-$(ses)_task-NT_lfreqori-0.1_lfreq-1_hfreq-40_$(metadata_stem).tsv")
    return eeg_file, meta_file
end

"""
Resolve the epoch stem, switching to the surface-Laplacian (CSD) cache when LAPLACIAN is set.

Returns `(stem, laplacian_enabled)`. LAPLACIAN=true appends `_csd`, which selects the files
written by compute_csd_epochs.py.

The CSD is precomputed rather than applied here on purpose: it is a spatial transform over the
whole montage, and these scripts deliberately load a single channel. Only the metadata stem is
left alone, because the CSD does not change the trial set.

Note for interpretation: the CSD is a spatial high-pass, so slow potentials (the CNV in the
cue-stimulus foreperiod) are strongly attenuated. The cluster statistic is a z-value and so is
insensitive to the ~1000x amplitude change, but ERP amplitudes from a CSD run are in uV/m^2.
"""
function resolve_epoch_stem_with_laplacian(base_stem::AbstractString)
    raw = lowercase(strip(get(ENV, "LAPLACIAN", "false")))
    if raw in ("1", "true", "yes", "on")
        return (endswith(base_stem, "_csd") ? String(base_stem) : base_stem * "_csd"), true
    elseif raw in ("", "0", "false", "no", "off")
        return String(base_stem), false
    end
    error("Invalid LAPLACIAN='$raw'. Expected true or false.")
end

"""
Fail early when LAPLACIAN=true but the CSD cache has not been generated.

Without this the missing files would simply be skipped by `load_subject_session`, and the run
would die much later with the unhelpful "No EEG blocks loaded".
"""
function assert_laplacian_cache_exists(
    datadir::AbstractString,
    subjects::AbstractVector{<:AbstractString},
    modalities::AbstractVector{<:AbstractString},
    epoch_stem::AbstractString,
    metadata_stem::AbstractString,
)
    found = 0
    for subid in subjects, ses in modalities
        eeg_file, _ = build_paths(String(datadir), String(subid), ses, String(epoch_stem), String(metadata_stem))
        isfile(eeg_file) && (found += 1)
    end
    found > 0 || error(
        "LAPLACIAN=true but no '$(epoch_stem).h5' file exists for any requested subject.\n" *
        "Generate the CSD cache first, e.g.:\n" *
        "  DATADIR=\$DATADIR WDDIR=\$WDDIR python " *
        "Analysis/EEGAnalysis/ClusterPermutation/compute_csd_epochs.py " *
        "--epoch-stem $(replace(epoch_stem, r"_csd$" => ""))"
    )
    @info "Surface-Laplacian cache found" epoch_stem sessions_with_cache=found
    return found
end

function infer_baseline_window(path::AbstractString)
    # Prefer more specific suffixes to avoid substring collisions.
    if occursin("_preprestim", path) || occursin("preprestim", path)
        return (-0.8, -0.7)
    elseif occursin("_prestim", path) || occursin("prestim", path)
        return (-0.5, -0.43)
    else
        return (-0.1, 0.0)
    end
end

"""
Resolve the baseline window to use for one modality ("mecha" or "elec").

The mechanical and electrical stimulators are not time-aligned (the mechanical actuator
has a ~30 ms onset delay), so the project convention uses different pre-cue windows per
modality: elec = (-0.50, -0.40) s, mecha = (-0.53, -0.43) s. A single combined
(MODALITY=both) run loads both sessions through the same code path, so the window must be
picked per call rather than once globally -- otherwise one modality silently gets the
other's window.

Resolution order:
1. Per-modality override `BASELINE_START_<MODALITY>`/`BASELINE_STOP_<MODALITY>` (e.g.
   `BASELINE_START_ELEC`/`BASELINE_STOP_ELEC`), when both are set. This is the only way to
   give mecha and elec sessions different windows within one combined run.
2. Global override `BASELINE_START`/`BASELINE_STOP` (both set), applied to every modality
   alike -- kept for backward compatibility with existing single-modality run scripts.
3. Suffix-driven default from `infer_baseline_window(path)`.

Returns `(baseline_start, baseline_stop, source)` where `source` is one of
`:per_modality`, `:global_override`, `:inferred`, so callers can log/tag provenance.
"""
function resolve_baseline_window(modality::AbstractString, path::AbstractString)
    modality_key = uppercase(modality)
    per_modality_start = strip(get(ENV, "BASELINE_START_$(modality_key)", ""))
    per_modality_stop = strip(get(ENV, "BASELINE_STOP_$(modality_key)", ""))
    if !isempty(per_modality_start) && !isempty(per_modality_stop)
        return parse(Float64, per_modality_start), parse(Float64, per_modality_stop), :per_modality
    end

    global_start = strip(get(ENV, "BASELINE_START", ""))
    global_stop = strip(get(ENV, "BASELINE_STOP", ""))
    if !isempty(global_start) && !isempty(global_stop)
        return parse(Float64, global_start), parse(Float64, global_stop), :global_override
    end

    baseline_start, baseline_stop = infer_baseline_window(path)
    return baseline_start, baseline_stop, :inferred
end

function apply_baseline_to_data(data_3d::Array{Float64, 3}, times::AbstractVector, eeg_file::AbstractString, modality::AbstractString)
    """
    Baseline-correct EEG data (channel x time x trial) using the window resolved for
    `modality` by resolve_baseline_window (per-modality env override, global env override,
    or suffix-driven default, in that order).
    """
    baseline_start, baseline_stop, source = resolve_baseline_window(modality, eeg_file)
    @info "Applying baseline correction" eeg_file modality baseline_start baseline_stop source
    mask = (times .>= baseline_start) .& (times .<= baseline_stop)
    any(mask) || error("Baseline window not in times for $eeg_file (window=($baseline_start, $baseline_stop))")

    # Compute per-channel, per-trial baseline and subtract it across time.
    baseline = mean(data_3d[:, mask, :], dims=2)
    corrected = data_3d .- baseline
    return corrected, times
end

function highpass_filter_epochs(data_3d::Array{Float64, 3}, times::AbstractVector, hp_freq::Real)
    """
    Zero-phase high-pass filter each epoch (channel x time x trial).

    - Sampling rate is inferred from `times` (data is ~500 Hz), not hardcoded.
    - Uses a 4th-order Butterworth high-pass applied with `filtfilt` (zero phase).
    - Each channel x trial time series is mirror-padded by one epoch length on both
      sides (reflect padding, matching `step007_extract_freq_bands.py`) before
      filtering, then cropped back to the original window to absorb edge transients.
    """
    hp_freq > 0 || error("HIGHPASS_FREQ must be > 0 (got $hp_freq).")

    n_channels, n_time, n_trials = size(data_3d)
    n_time > 1 || error("Need at least 2 time samples to high-pass filter (got $n_time).")

    # Infer sampling rate from the (assumed uniform) time axis.
    dt = mean(diff(collect(times)))
    dt > 0 || error("Non-increasing time axis; cannot infer sampling rate for filtering.")
    fs = 1 / dt

    hp_freq < fs / 2 || error("HIGHPASS_FREQ=$hp_freq must be below Nyquist ($(fs / 2) Hz).")

    # Design the zero-phase high-pass filter once and reuse it for every trace.
    # DSP 0.8: the cutoff is given in Hz and `fs` is passed to `digitalfilter`,
    # which normalizes the frequency internally.
    hp = digitalfilter(Highpass(hp_freq), Butterworth(4); fs = fs)

    # Reflect-pad on each side (mode="reflect": edge sample not repeated), matching
    # step007_extract_freq_bands.py. A single reflection supports at most n_time-1
    # samples; that already far exceeds a 6 Hz high-pass edge transient, so cap there.
    n_pad = n_time - 1
    @info "High-pass filtering epochs" hp_freq fs n_pad n_time n_channels n_trials

    filtered = similar(data_3d)
    pad_idx_left = (n_pad + 1):-1:2          # samples 2..n_pad+1 reversed
    pad_idx_right = (n_time - 1):-1:(n_time - n_pad)  # samples n_time-1..n_time-n_pad reversed
    for tr in 1:n_trials
        for ch in 1:n_channels
            sig = @view data_3d[ch, :, tr]
            padded = vcat(sig[pad_idx_left], collect(sig), sig[pad_idx_right])
            filt = filtfilt(hp, padded)
            filtered[ch, :, tr] = filt[(n_pad + 1):(n_pad + n_time)]
        end
    end
    return filtered
end

function crop_data_to_time_window(data_3d::Array{Float64, 3}, times::AbstractVector, time_start::Union{Nothing, Real}, time_stop::Union{Nothing, Real})
    """
    Crop the EEG data to the specified time window.
    
    Parameters:
    - data_3d: channel x time x trial array
    - times: time vector corresponding to the time dimension
    - time_start: start time in seconds (inclusive), or nothing for no cropping
    - time_stop: stop time in seconds (inclusive), or nothing for no cropping
    
    Returns:
    - cropped_data: cropped data array
    - cropped_times: cropped time vector
    """
    if isnothing(time_start) && isnothing(time_stop)
        return data_3d, times
    end
    
    # Create mask for time window
    mask = trues(length(times))
    if !isnothing(time_start)
        mask .&= times .>= float(time_start)
    end
    if !isnothing(time_stop)
        mask .&= times .<= float(time_stop)
    end
    
    if !any(mask)
        error("No time points found in the specified time window (start=$time_start, stop=$time_stop)")
    end
    
    # Apply cropping
    cropped_data = data_3d[:, mask, :]
    cropped_times = times[mask]
    
    @info "Data cropped to time window" original_length=length(times) cropped_length=length(cropped_times) time_start time_stop
    
    return cropped_data, cropped_times
end

function load_subject_session(
    datadir::String,
    subid::String,
    ses::AbstractString,
    requested_channels::Union{Nothing, AbstractVector{<:AbstractString}};
    epoch_stem::String = "epochs",
    metadata_stem::String = "allmetadata",
    time_start::Union{Nothing, Real} = nothing,
    time_stop::Union{Nothing, Real} = nothing,
    highpass_freq::Union{Nothing, Real} = nothing,
)
    eeg_file, meta_file = build_paths(datadir, subid, ses, epoch_stem, metadata_stem)
    if !isfile(eeg_file) || !isfile(meta_file)
        return nothing
    end

    channels = resolve_channels_for_file(eeg_file, requested_channels)
    loaded_data = h5open(eeg_file, "r") do h5
        data_channels = Matrix{Float64}[]
        times_ref = nothing

        for ch in channels
            mat = Matrix{Float64}(read(h5["$ch/data"]))  # time x trial
            t = vec(read(h5["$ch/times"]))
            if isnothing(times_ref)
                times_ref = t
            elseif length(times_ref) != length(t) || any(times_ref .!= t)
                error("Time axis mismatch across channels in $eeg_file (channel=$ch).")
            end
            push!(data_channels, mat)
        end

        n_channels = length(data_channels)
        n_time = size(data_channels[1], 1)
        n_trials = size(data_channels[1], 2)
        for (i, mat) in enumerate(data_channels)
            size(mat, 1) == n_time || error("Time-length mismatch across channels in $eeg_file at index $i")
            size(mat, 2) == n_trials || error("Trial-count mismatch across channels in $eeg_file at index $i")
        end

        data_3d = Array{Float64}(undef, n_channels, n_time, n_trials)  # channel x time x trial
        for i in 1:n_channels
            data_3d[i, :, :] = data_channels[i]
        end
        return (data_3d = data_3d, times = times_ref)
    end
    # Apply baseline before any downstream analysis. `ses` selects the modality-specific
    # window (see resolve_baseline_window) so combined runs don't misapply one modality's
    # baseline to the other.
    corrected_data, corrected_times = apply_baseline_to_data(
        loaded_data.data_3d, loaded_data.times, eeg_file, ses
    )
    loaded_data = (data_3d = corrected_data, times = corrected_times)
    meta = CSV.read(meta_file, DataFrame; delim = '\t')

    # check if trials are the same length
    n_trials = size(loaded_data.data_3d, 3)
    if nrow(meta) != n_trials
        error("Row mismatch between metadata and EEG trials." * " subid=$subid ses=$ses n_meta=$(nrow(meta)) n_data=$n_trials")
    end

    # delete trials with nan responses
    meta_res = meta[string.(meta.response_decoded) .!= "nan", :]
    data_res = loaded_data.data_3d[:, :, string.(meta.response_decoded) .!= "nan"]

    # delete bad epochs (should be gone already but just to be sure)
    meta_drop = meta_res[meta_res.drop_epoch .!= true, :]
    data_drop = data_res[:, :, meta_res.drop_epoch .!= true]

    # delete bad fits & extreme intensities
    meta_clean = meta_drop[meta_drop.analysis_excluded .!= true, :]
    data_clean = data_drop[:, :, meta_drop.analysis_excluded .!= true]

    n_trials = size(data_clean, 3)
    if nrow(meta_clean) != n_trials
        error("After cleaning mismatch!! Row mismatch between metadata and EEG trials." *
              " subid=$subid ses=$ses n_meta=$(nrow(meta_clean)) n_data=$n_trials")
    end

    # High-pass filter the full-length (pre-crop) epochs so edge padding uses the
    # complete epoch; no-filter path leaves data unchanged.
    data_to_crop = isnothing(highpass_freq) ? data_clean :
        highpass_filter_epochs(data_clean, loaded_data.times, highpass_freq)

    # Apply time cropping if specified
    data_cropped, times_cropped = crop_data_to_time_window(data_to_crop, loaded_data.times, time_start, time_stop)

    meta_clean.intensity_level_cat = categorical(string.(meta_clean.intensity_level))
    meta_clean.modality = categorical(string.(meta_clean.modality))
    meta_clean.response_decoded = categorical(string.(meta_clean.response_decoded))
    meta_clean.sub = categorical(string.(meta_clean.sub))

    return (data = data_cropped, times = times_cropped, events = meta_clean, channels = channels)
end

function coefficient_index_from_name(model, coef_name::String)
    ct = DataFrame(coeftable(model))
    names_in_order = unique(ct.coefname)
    idx = findfirst(==(coef_name), names_in_order)
    if isnothing(idx)
        error("Coefficient '$coef_name' not found. Available coefficients: $(join(names_in_order, ", "))")
    end
    return idx, names_in_order
end

"""
Resolve one or more coefficient names from ENV / defaults.

Defaults target the three effects of interest:
1) response main effect
2) intensity_level_centered main effect
3) interaction intensity_level_centered × response_decoded

Matching strategy:
- first try exact match
- then fallback to substring match (for robust interaction naming)
"""
function resolve_target_coefficients(model)
    ct = DataFrame(coeftable(model))
    names_in_order = unique(string.(ct.coefname))

    # Optional override: comma-separated list in COEF_NAMES.
    raw = strip(get(ENV, "COEF_NAMES", ""))
    requested = String[]
    if !isempty(raw)
        toks = split(replace(raw, "," => " "))
        requested = [String(strip(t)) for t in toks if !isempty(strip(t))]
    else
        requested = [
            "response_decoded: Yes",
            "intensity_level_centered",
            "intensity_level_centered & response_decoded: Yes",
        ]
    end

    resolved = String[]
    for req in requested
        idx_exact = findfirst(==(req), names_in_order)
        if !isnothing(idx_exact)
            push!(resolved, names_in_order[idx_exact])
            continue
        end

        # Robust fallback for coefficient label formatting differences.
        candidates = [nm for nm in names_in_order if occursin(req, nm)]
        if length(candidates) == 1
            push!(resolved, candidates[1])
            continue
        end

        # Extra fallback for the interaction term.
        if req == "intensity_level_centered & response_decoded: Yes"
            candidates = [
                nm for nm in names_in_order
                if occursin("intensity_level_centered", nm) &&
                   occursin("response_decoded", nm)
            ]
            if length(candidates) == 1
                push!(resolved, candidates[1])
                continue
            end
        end

        error("Could not resolve coefficient '$req'. Available coefficients: $(join(names_in_order, ", "))")
    end

    # Keep order but remove duplicates.
    unique_resolved = String[]
    for nm in resolved
        nm in unique_resolved || push!(unique_resolved, nm)
    end
    return unique_resolved, names_in_order
end

"""
Append every coefficient that belongs to a requested factor.

For categorical predictors, MixedModels expands one factor into multiple
coefficient names (e.g., one per non-reference level). This helper adds all
matching coefficient labels so each level gets its own permutation test/plot.
"""
function append_factor_coefficients(
    target_coefs::Vector{String},
    all_coef_names::Vector{String},
    factor_name::String,
)
    factor_terms = [nm for nm in all_coef_names if occursin(factor_name, nm)]
    if isempty(factor_terms)
        @warn "Requested factor terms were not found in model coefficients." factor_name available=all_coef_names
        return target_coefs
    end

    for nm in factor_terms
        nm in target_coefs || push!(target_coefs, nm)
    end
    return target_coefs
end

function plot_erp_helper(
    times::AbstractVector,
    data_2d::AbstractMatrix,
    events::DataFrame,
    outdir::String,
    channel::String,
)
    if !hasproperty(events, :response_decoded)
        @warn "response_decoded column missing; ERP plot skipped."
        return nothing
    end

    labels = string.(events.response_decoded)
    no_mask = labels .== "No"
    yes_mask = labels .== "Yes"

    if !any(no_mask) || !any(yes_mask)
        @warn "Need both response_decoded levels No and Yes for ERP plot; plot skipped." n_no=sum(no_mask) n_yes=sum(yes_mask)
        return nothing
    end

    # Count trials for unperceived (No) and perceived (Yes) conditions
    n_unperceived = sum(no_mask)
    n_perceived = sum(yes_mask)

    erp_no  = vec(mean(data_2d[:, no_mask],  dims = 2))
    erp_yes = vec(mean(data_2d[:, yes_mask], dims = 2))
    se_no  = vec(std(data_2d[:, no_mask],  dims = 2)) ./ sqrt(sum(no_mask))
    se_yes = vec(std(data_2d[:, yes_mask], dims = 2)) ./ sqrt(sum(yes_mask))

    
    # Build results DataFrame in the format UnfoldMakie expects
    start_ms = -50   # ms
    stop_ms  = 250    # ms

    start = argmin(abs.(times .- start_ms / 1000))
    stop  = argmin(abs.(times .- stop_ms  / 1000))
    times_cropped = times[start:stop]
    results = DataFrame(
        time     = vcat(times_cropped, times_cropped),
        estimate = vcat(erp_no[start:stop], erp_yes[start:stop]),
        coefname = vcat(fill("No", length(times_cropped)), fill("Yes", length(times_cropped))),
        )
    results.stderror = vcat(se_no[start:stop], se_yes[start:stop])

    fig = Figure(size = (900, 520))
    plot_erp!(
        fig,
        results,
        mapping = (; color = :coefname => "Response"),
        axis    = (; xlabel = "Time (s)", ylabel = "Amplitude",
                     title  = "ERP by response_decoded ($channel) - Unperceived: $n_unperceived, Perceived: $n_perceived"),
        stderror = true,
    )

    # add a vertical line for stim onset
    ax = fig.content[1]  # grab the axis that plot_erp! created
    vlines!(ax, [0.0], color = :black, linewidth = 1.5, linestyle = :dash)

    # save figure
    mkpath(outdir)
    filename = "erp_response_decoded_channel-$(channel).pdf"
    outfile = joinpath(outdir, filename)
    try
        save(outfile, fig)
        @info "Saved ERP plot" outfile
    catch err
        fallback = joinpath(pwd(), filename)
        @warn "Failed to save ERP plot to RESDIR-derived path, using current directory instead." outdir error=sprint(showerror, err) fallback
        save(fallback, fig)
        outfile = fallback
        @info "Saved ERP plot" outfile
    end
    return outfile
end


function save_erp_plot(
    times::AbstractVector,
    data_2d::AbstractMatrix,
    events::DataFrame,
    outdir::String,
    channel::String,
)
    if !hasproperty(events, :response_decoded)
        @warn "response_decoded column missing; ERP plot skipped."
        return nothing
    end

    labels = string.(events.response_decoded)
    no_mask = labels .== "No"
    yes_mask = labels .== "Yes"

    if !any(no_mask) || !any(yes_mask)
        @warn "Need both response_decoded levels No and Yes for ERP plot; plot skipped." n_no=sum(no_mask) n_yes=sum(yes_mask)
        return nothing
    end

    erp_no = vec(mean(data_2d[:, no_mask], dims = 2))
    erp_yes = vec(mean(data_2d[:, yes_mask], dims = 2))

    fig = Figure(size = (900, 520))
    ax = Axis(
        fig[1, 1],
        xlabel = "Time (s)",
        ylabel = "Amplitude",
        title = "ERP by response_decoded ($channel)",
    )
    lines!(ax, times, erp_no, label = "response_decoded = No", linewidth = 2.0)
    lines!(ax, times, erp_yes, label = "response_decoded = Yes", linewidth = 2.0)
    #axislegend(ax; position = :rt)

    mkpath(outdir)
    filename = "erp_response_decoded_channel-$(channel).pdf"
    outfile = joinpath(outdir, filename)
    try
        save(outfile, fig)
        @info "Saved ERP plot" outfile
    catch err
        fallback = joinpath(pwd(), filename)
        @warn "Failed to save ERP plot to RESDIR-derived path, using current directory instead." outdir error=sprint(showerror, err) fallback
        save(fallback, fig)
        outfile = fallback
        @info "Saved ERP plot" outfile
    end
    return outfile
end

function significant_time_clusters(times::AbstractVector, pvals::AbstractVector; alpha::Float64 = 0.05)
    mask = pvals .< alpha
    clusters = Tuple{Float64, Float64}[]
    start_idx = nothing

    for i in eachindex(mask)
        if mask[i] && isnothing(start_idx)
            start_idx = i
        elseif !mask[i] && !isnothing(start_idx)
            push!(clusters, (times[start_idx], times[i - 1]))
            start_idx = nothing
        end
    end
    if !isnothing(start_idx)
        push!(clusters, (times[start_idx], times[lastindex(times)]))
    end

    return clusters
end

"""
Parse which channels should be plotted in the final ERP+cluster figure.

ENV usage:
- PLOT_CHANNELS="" (default): use fallback_plot_channel only
- PLOT_CHANNELS="C6,Cz,Pz": plot selected subset
- PLOT_CHANNELS="all": plot every analyzed channel
"""
function parse_plot_channels(available_channels::Vector{String}, fallback_plot_channel::String)
    raw = strip(get(ENV, "PLOT_CHANNELS", ""))
    if isempty(raw)
        return [fallback_plot_channel]
    end

    if lowercase(raw) == "all"
        return available_channels
    end

    tokens = split(replace(raw, "," => " "))
    requested = [String(strip(ch)) for ch in tokens if !isempty(strip(ch))]
    isempty(requested) && error("PLOT_CHANNELS was provided but no valid channel names were parsed.")

    missing = setdiff(requested, available_channels)
    isempty(missing) || error("PLOT_CHANNELS contains channels not in analyzed data: $(join(missing, ", "))")

    # Keep requested order but remove duplicates.
    unique_requested = String[]
    for ch in requested
        ch in unique_requested || push!(unique_requested, ch)
    end
    return unique_requested
end

function pvals_channel_time(pvals, n_channels::Int, n_time::Int)
    if pvals isa AbstractVector
        n_channels == 1 || error("Vector pvals only valid for single-channel analysis.")
        length(pvals) == n_time || error("Vector pvals length ($(length(pvals))) must equal n_time ($n_time).")
        return reshape(vec(pvals), 1, n_time)
    end
    ndims(pvals) == 2 || error("Expected pvals to be a vector or 2D array, got ndims=$(ndims(pvals)).")
    if size(pvals, 1) == n_channels && size(pvals, 2) == n_time
        return Matrix(pvals)
    elseif size(pvals, 1) == n_time && size(pvals, 2) == n_channels
        return permutedims(pvals)
    else
        error("Unexpected pvals shape $(size(pvals)); expected ($n_channels, $n_time) or ($n_time, $n_channels).")
    end
end

"""
Create one figure with several channel panels.

Each panel contains:
- ERP No/Yes lines with standard-error ribbons
- cluster shading derived from channel-specific p-values
"""
function save_permutation_plot_multichannel(
    times::AbstractVector,
    pvals_ct::AbstractMatrix,              # channel x time
    data_epoched::Array{Float64, 3},       # channel x time x trial
    events::DataFrame,
    outdir::String,
    plot_channels::Vector{String},
    all_channels::Vector{String},
    coef_name::String;
    alpha::Float64 = 0.05,
    plot_start::Union{Nothing, Real} = nothing,
    plot_stop::Union{Nothing, Real} = nothing,
)
    size(pvals_ct, 2) == length(times) || error("pvals_ct time dimension must match times length.")
    size(data_epoched, 2) == length(times) || error("data_epoched time dimension must match times length.")
    size(data_epoched, 3) == nrow(events) || error("data_epoched trial dimension must match number of event rows.")
    !isempty(plot_channels) || error("plot_channels must not be empty.")
    if !isnothing(plot_start) && !isnothing(plot_stop) && plot_start > plot_stop
        error("plot_start must be <= plot_stop.")
    end

    labels = string.(events.response_decoded)
    no_mask = labels .== "No"
    yes_mask = labels .== "Yes"
    if !any(no_mask) || !any(yes_mask)
        @warn "Need both response_decoded levels No and Yes for clustered ERP plot; plot skipped." n_no=sum(no_mask) n_yes=sum(yes_mask)
        return nothing
    end

    keep = trues(length(times))
    if !isnothing(plot_start)
        keep .&= times .>= float(plot_start)
    end
    if !isnothing(plot_stop)
        keep .&= times .<= float(plot_stop)
    end
    any(keep) || error("No time points in selected plot window.")
    times_plot = times[keep]

    # Compact panel layout for quick visual channel-by-channel inspection.
    n_panels = length(plot_channels)
    n_cols = min(3, n_panels)
    n_rows = cld(n_panels, n_cols)
    fig = Figure(size = (460 * n_cols, 300 * n_rows))

    for (i, ch) in enumerate(plot_channels)
        ch_idx = findfirst(==(ch), all_channels)
        isnothing(ch_idx) && error("Channel '$ch' not found in all_channels.")

        data_2d = Matrix(data_epoched[ch_idx, :, :]) # time x trial
        pvals = vec(pvals_ct[ch_idx, :])
        clusters = significant_time_clusters(times_plot, pvals[keep]; alpha = alpha)

        erp_no = vec(mean(data_2d[:, no_mask], dims = 2))
        erp_yes = vec(mean(data_2d[:, yes_mask], dims = 2))
        se_no = vec(std(data_2d[:, no_mask], dims = 2)) ./ sqrt(sum(no_mask))
        se_yes = vec(std(data_2d[:, yes_mask], dims = 2)) ./ sqrt(sum(yes_mask))

        r = cld(i, n_cols)
        c = ((i - 1) % n_cols) + 1
        ax = Axis(
            fig[r, c],
            xlabel = "Time (s)",
            ylabel = "Amplitude",
            title = "Channel $ch",
        )

        x = times_plot
        y_no = erp_no[keep]
        y_yes = erp_yes[keep]
        s_no = se_no[keep]
        s_yes = se_yes[keep]

        # ERP lines + uncertainty ribbons.
        band!(ax, x, y_no .- s_no, y_no .+ s_no, color = (:steelblue, 0.20))
        band!(ax, x, y_yes .- s_yes, y_yes .+ s_yes, color = (:darkorange, 0.20))
        lines!(ax, x, y_no, color = :steelblue, linewidth = 1.8, label = "No")
        lines!(ax, x, y_yes, color = :darkorange, linewidth = 1.8, label = "Yes")
        vlines!(ax, [0.0], color = :black, linewidth = 1.2, linestyle = :dash)

        # Gray spans mark significant permutation clusters for this channel.
        for (t0, t1) in clusters
            vspan!(ax, t0, t1, color = (:gray, 0.22))
        end

        if i == 1
            axislegend(ax; position = :rt)
        end
    end

    mkpath(outdir)
    safe_coef = replace(coef_name, r"[^A-Za-z0-9._-]+" => "_")
    filename = "clusterperm_erpclusters_multichannel_n-$(length(plot_channels))_coef-$(safe_coef).pdf"
    outfile = joinpath(outdir, filename)
    save(outfile, fig)
    @info "Saved clustered ERP multichannel plot" outfile n_channels=length(plot_channels) alpha
    return outfile
end

function make_pvalue_result_df(times::AbstractVector, pvals_ct::AbstractMatrix, channels::Vector{String})
    n_channels, n_time = size(pvals_ct)
    n_channels == length(channels) || error("channels length mismatch with pvals.")
    n_time == length(times) || error("times length mismatch with pvals.")

    if n_channels == 1
        return DataFrame(time = collect(times), pvalue = vec(pvals_ct[1, :]))
    end

    channel_col = Vector{String}(undef, n_channels * n_time)
    time_col = Vector{Float64}(undef, n_channels * n_time)
    pval_col = Vector{Float64}(undef, n_channels * n_time)
    idx = 1
    for ci in 1:n_channels
        ch = channels[ci]
        for ti in 1:n_time
            channel_col[idx] = ch
            time_col[idx] = float(times[ti])
            pval_col[idx] = float(pvals_ct[ci, ti])
            idx += 1
        end
    end
    return DataFrame(channel = channel_col, time = time_col, pvalue = pval_col)
end

function save_permutation_plot(
    times::AbstractVector,
    pvals::AbstractVector,
    data_2d::AbstractMatrix,
    events::DataFrame,
    outdir::String,
    channel::String,
    coef_name::String;
    alpha::Float64 = 0.05,
    plot_start::Union{Nothing, Real} = nothing,
    plot_stop::Union{Nothing, Real} = nothing,
)
    length(times) == length(pvals) || error("times and pvals must have the same length.")
    if !hasproperty(events, :response_decoded)
        @warn "response_decoded column missing; clustered ERP plot skipped."
        return nothing, Tuple{Float64, Float64}[]
    end
    if size(data_2d, 1) != length(times)
        error("data_2d time dimension must match times length.")
    end
    if size(data_2d, 2) != nrow(events)
        error("data_2d trial dimension must match number of event rows.")
    end
    if !isnothing(plot_start) && !isnothing(plot_stop) && plot_start > plot_stop
        error("plot_start must be <= plot_stop.")
    end

    labels = string.(events.response_decoded)
    no_mask = labels .== "No"
    yes_mask = labels .== "Yes"
    if !any(no_mask) || !any(yes_mask)
        @warn "Need both response_decoded levels No and Yes for clustered ERP plot; plot skipped." n_no=sum(no_mask) n_yes=sum(yes_mask)
        return nothing, Tuple{Float64, Float64}[]
    end

    keep = trues(length(times))
    if !isnothing(plot_start)
        keep .&= times .>= float(plot_start)
    end
    if !isnothing(plot_stop)
        keep .&= times .<= float(plot_stop)
    end
    any(keep) || error("No time points in selected plot window.")

    times_plot = times[keep]
    pvals_plot = pvals[keep]
    clusters = significant_time_clusters(times_plot, pvals_plot; alpha = alpha)

    erp_no  = vec(mean(data_2d[:, no_mask],  dims = 2))
    erp_yes = vec(mean(data_2d[:, yes_mask], dims = 2))
    se_no  = vec(std(data_2d[:, no_mask],  dims = 2)) ./ sqrt(sum(no_mask))
    se_yes = vec(std(data_2d[:, yes_mask], dims = 2)) ./ sqrt(sum(yes_mask))

    # Build ERP results DataFrame in the format UnfoldMakie expects.
    results = DataFrame(
        time = vcat(collect(times_plot), collect(times_plot)),
        estimate = vcat(erp_no[keep], erp_yes[keep]),
        coefname = vcat(fill("No", length(times_plot)), fill("Yes", length(times_plot))),
    )
    results.stderror = vcat(se_no[keep], se_yes[keep])

    fig = Figure(size = (980, 520))
    plot_erp!(
        fig,
        results,
        mapping = (; color = :coefname => "Response"),
        axis = (; xlabel = "Time (s)", ylabel = "Amplitude",
                  title = "ERP with significant clusters ($channel, $(coef_name))"),
        stderror = true,
    )
    ax = fig.content[1]
    vlines!(ax, [0.0], color = :black, linewidth = 1.5, linestyle = :dash)

    for (t0, t1) in clusters
        vspan!(ax, t0, t1, color = (:gray, 0.25))
    end
    #axislegend(ax; position = :rt)

    mkpath(outdir)
    safe_coef = replace(coef_name, r"[^A-Za-z0-9._-]+" => "_")
    filename = "clusterperm_erpclusters_channel-$(channel)_coef-$(safe_coef).pdf"
    outfile = joinpath(outdir, filename)
    try
        save(outfile, fig)
        @info "Saved clustered ERP plot" outfile n_clusters=length(clusters) alpha
    catch err
        fallback = joinpath(pwd(), filename)
        @warn "Failed to save clustered ERP plot to RESDIR-derived path, using current directory instead." outdir error=sprint(showerror, err) fallback
        save(fallback, fig)
        outfile = fallback
        @info "Saved clustered ERP plot" outfile n_clusters=length(clusters) alpha
    end

    return outfile, clusters
end

"""
Draw one clustered ERP panel (ERP + cluster shading) onto a provided axis.

This is used to compose a three-panel comparison figure (elec/mecha/combined)
for the same coefficient.
"""
function draw_clustered_erp_panel!(
    ax,
    times::AbstractVector,
    pvals::AbstractVector,
    data_2d::AbstractMatrix,
    events::DataFrame,
    panel_title::String;
    alpha::Float64 = 0.05,
    plot_start::Union{Nothing, Real} = nothing,
    plot_stop::Union{Nothing, Real} = nothing,
)
    length(times) == length(pvals) || error("times and pvals must have the same length.")
    size(data_2d, 1) == length(times) || error("data_2d time dimension must match times length.")
    size(data_2d, 2) == nrow(events) || error("data_2d trial dimension must match number of event rows.")
    if !isnothing(plot_start) && !isnothing(plot_stop) && plot_start > plot_stop
        error("plot_start must be <= plot_stop.")
    end

    labels = string.(events.response_decoded)
    no_mask = labels .== "No"
    yes_mask = labels .== "Yes"
    if !any(no_mask) || !any(yes_mask)
        # Panel-level skip: keep the figure layout and indicate why content is missing.
        text!(ax, 0.5, 0.5, text = "Missing No/Yes trials", align = (:center, :center), space = :relative)
        ax.title = panel_title
        return
    end

    keep = trues(length(times))
    if !isnothing(plot_start)
        keep .&= times .>= float(plot_start)
    end
    if !isnothing(plot_stop)
        keep .&= times .<= float(plot_stop)
    end
    any(keep) || error("No time points in selected plot window.")

    times_plot = times[keep]
    pvals_plot = pvals[keep]
    clusters = significant_time_clusters(times_plot, pvals_plot; alpha = alpha)

    erp_no  = vec(mean(data_2d[:, no_mask],  dims = 2))
    erp_yes = vec(mean(data_2d[:, yes_mask], dims = 2))
    se_no  = vec(std(data_2d[:, no_mask],  dims = 2)) ./ sqrt(sum(no_mask))
    se_yes = vec(std(data_2d[:, yes_mask], dims = 2)) ./ sqrt(sum(yes_mask))

    x = times_plot
    y_no = erp_no[keep]
    y_yes = erp_yes[keep]
    s_no = se_no[keep]
    s_yes = se_yes[keep]

    band!(ax, x, y_no .- s_no, y_no .+ s_no, color = (:steelblue, 0.20))
    band!(ax, x, y_yes .- s_yes, y_yes .+ s_yes, color = (:darkorange, 0.20))
    lines!(ax, x, y_no, color = :steelblue, linewidth = 2.0, label = "No")
    lines!(ax, x, y_yes, color = :darkorange, linewidth = 2.0, label = "Yes")
    vlines!(ax, [0.0], color = :black, linewidth = 1.2, linestyle = :dash)

    for (t0, t1) in clusters
        vspan!(ax, t0, t1, color = (:gray, 0.22))
    end

    ax.title = panel_title
end

"""
Save one comparison figure per coefficient with three panels:
Electrical, Mechanical, and Combined.
"""
function save_cross_modality_comparison_plots(
    comparison_store::Dict{String, Dict{String, NamedTuple}},
    outroot::String;
    alpha::Float64 = 0.05,
    plot_start::Union{Nothing, Real} = nothing,
    plot_stop::Union{Nothing, Real} = nothing,
)
    labels = ["elec", "mecha", "combined"]
    if !all(haskey(comparison_store, lbl) for lbl in labels)
        @warn "Not all run types are available for cross-modality plotting; skipped." available=collect(keys(comparison_store))
        return nothing
    end

    # Plot only coefficients that exist in all three runs.
    common_coefs = collect(keys(comparison_store["elec"]))
    for lbl in labels[2:end]
        common_coefs = intersect(common_coefs, collect(keys(comparison_store[lbl])))
    end
    isempty(common_coefs) && (@warn "No common coefficients across elec/mecha/combined for cross-modality plots."; return nothing)
    sort!(common_coefs)

    cmp_dir = joinpath(outroot, "comparison", Dates.format(now(UTC), dateformat"yyyymmdd-HHMMSS"))
    mkpath(cmp_dir)

    for coef_name in common_coefs
        fig = Figure(size = (1500, 440))
        safe_coef = replace(coef_name, r"[^A-Za-z0-9._-]+" => "_")
        plot_channel = comparison_store["elec"][coef_name].channel

        for (j, lbl) in enumerate(labels)
            entry = comparison_store[lbl][coef_name]
            ax = Axis(
                fig[1, j],
                xlabel = "Time (s)",
                ylabel = "Amplitude",
                title = "$(uppercase(lbl))",
            )
            draw_clustered_erp_panel!(
                ax,
                entry.times,
                entry.pvals,
                entry.data_2d,
                entry.events,
                uppercase(lbl);
                alpha = alpha,
                plot_start = plot_start,
                plot_stop = plot_stop,
            )
            if j == 1
                axislegend(ax; position = :rt)
            end
        end

        Label(
            fig[0, 1:3],
            "Coefficient: $coef_name  |  Channel: $plot_channel",
            fontsize = 18,
            tellwidth = false,
        )

        outfile = joinpath(cmp_dir, "comparison_erpclusters_channel-$(plot_channel)_coef-$(safe_coef).pdf")
        save(outfile, fig)
        @info "Saved cross-modality comparison plot" outfile
    end

    return cmp_dir
end

function save_pvalue_csv(
    result::DataFrame,
    outdir::String,
    channel::String,
    coef_name::String,
)
    mkpath(outdir)
    safe_coef = replace(coef_name, r"[^A-Za-z0-9._-]+" => "_")
    filename = "unfold_clusterperm_channel-$(channel)_coef-$(safe_coef).csv"
    outfile = joinpath(outdir, filename)
    try
        CSV.write(outfile, result)
        @info "Saved p-values" outfile
    catch err
        fallback = joinpath(pwd(), filename)
        @warn "Failed to save p-value CSV to RESDIR-derived path, using current directory instead." outdir error=sprint(showerror, err) fallback
        CSV.write(fallback, result)
        outfile = fallback
        @info "Saved p-values" outfile
    end
    return outfile
end

function save_permutation_timing_csv(
    outdir::String,
    channel::String,
    coef_name::String,
    seconds::Real,
    n_permutations::Integer,
    clusterforming_threshold::Real,
    seed::Integer,
)
    mkpath(outdir)
    safe_coef = replace(coef_name, r"[^A-Za-z0-9._-]+" => "_")
    filename = "unfold_clusterperm_timing_channel-$(channel)_coef-$(safe_coef).csv"
    outfile = joinpath(outdir, filename)
    timing = DataFrame(
        permutation_seconds = [float(seconds)],
        n_permutations = [n_permutations],
        clusterforming_threshold = [float(clusterforming_threshold)],
        seed = [seed],
    )
    try
        CSV.write(outfile, timing)
        @info "Saved permutation timing" outfile permutation_seconds=round(float(seconds); digits=3)
    catch err
        fallback = joinpath(pwd(), filename)
        @warn "Failed to save permutation timing CSV to RESDIR-derived path, using current directory instead." outdir error=sprint(showerror, err) fallback
        CSV.write(fallback, timing)
        outfile = fallback
        @info "Saved permutation timing" outfile permutation_seconds=round(float(seconds); digits=3)
    end
    return outfile
end

function sanitize_token(s::AbstractString)
    cleaned = replace(strip(s), r"[^A-Za-z0-9._-]+" => "_")
    cleaned = replace(cleaned, r"_+" => "_")
    cleaned = strip(cleaned, '_')
    return isempty(cleaned) ? "na" : cleaned
end

function make_run_paths(
    outroot::String,
    modalities::AbstractVector,
    formula_string::AbstractString;
    baseline_tag::AbstractString = "auto",
    epoch_tag::AbstractString = "epochs",
)
    ts = Dates.format(now(UTC), dateformat"yyyymmdd-HHMMSS")
    modality_tag = sanitize_token(join(string.(modalities), "-"))
    formula_id = bytes2hex(sha1(formula_string))[1:10]
    baseline_id = sanitize_token(baseline_tag)
    epoch_id = sanitize_token(epoch_tag)
    run_id = "$(ts)__mod-$(modality_tag)__ep-$(epoch_id)__bl-$(baseline_id)__f-$(formula_id)"
    runs_root = joinpath(outroot, "runs")
    run_dir = joinpath(runs_root, run_id)
    return run_id, run_dir, runs_root, modality_tag, formula_id
end

function save_run_manifest_and_index(
    run_dir::String,
    runs_root::String,
    run_id::AbstractString,
    modality_tag::AbstractString,
    formula_id::AbstractString,
    formula_string::AbstractString,
    channel::AbstractString,
    coef_name::AbstractString,
    subjects::AbstractVector{<:AbstractString},
    modalities::AbstractVector{<:AbstractString},
    n_trials::Integer,
    n_timepoints::Integer,
    n_permutations::Integer,
    clusterforming_threshold::Real,
    seed::Integer,
    pvalue_csv::AbstractString,
    timing_csv::AbstractString,
    plot_file::Union{Nothing, AbstractString},
    epoch_stem::AbstractString,
    metadata_stem::AbstractString,
)
    ts_iso = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
    row = DataFrame(
        timestamp_utc = [ts_iso],
        run_id = [run_id],
        modality = [modality_tag],
        formula_id = [formula_id],
        formula_string = [formula_string],
        channel = [channel],
        coef_name = [coef_name],
        subjects = [join(subjects, ",")],
        modalities = [join(modalities, ",")],
        n_trials = [n_trials],
        n_timepoints = [n_timepoints],
        n_permutations = [n_permutations],
        clusterforming_threshold = [float(clusterforming_threshold)],
        seed = [seed],
        pvalue_csv = [pvalue_csv],
        timing_csv = [timing_csv],
        plot_file = [isnothing(plot_file) ? "" : plot_file],
        epoch_stem = [epoch_stem],
        metadata_stem = [metadata_stem],
    )

    manifest_file = joinpath(run_dir, "manifest.csv")
    CSV.write(manifest_file, row)
    @info "Saved run manifest" manifest_file

    index_file = joinpath(runs_root, "index.csv")
    CSV.write(index_file, row; append = isfile(index_file), writeheader = !isfile(index_file))
    @info "Updated run index" index_file run_id
end


end
