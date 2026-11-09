# Normalise one MAT entry into a Vector{String} of labels.
#
# MAT.jl decodes a MATLAB char array of size N x L into an N-element
# Vector{String}, but collapses the 1 x L case into a single String (see
# decode_char_array in MAT.jl). Files written by scipy's savemat from a numpy
# "<U" array take exactly that char-array path, so a parcel/channel with a
# single neighbour arrives here as a bare String rather than a 1-element vector.
# Char arrays are also space-padded to the longest label in the group, so the
# padding has to be trimmed or the names never match the data's channel names.
# Real FieldTrip files store cell arrays of char and need neither fix; they pass
# through unchanged.
function _as_label_vector(x)
    labels = if x isa AbstractString
        [String(x)]                       # 1 x L char array collapsed by MAT.jl
    elseif x isa AbstractArray
        isempty(x) ? String[] : String.(vec(x))
    else
        error("Unsupported label entry of type $(typeof(x)) in neighbour file")
    end
    # Trailing spaces come from char-array padding; labels never end in spaces.
    return String.(rstrip.(labels))
end

# Lowercase name -> index, skipping any name whose lowercase form is not unique.
# Used only as a fallback when an exact label match fails.
function _unambiguous_lowercase_index(names::AbstractVector{<:AbstractString})
    counts = Dict{String, Int}()
    for name in names
        key = lowercase(name)
        counts[key] = get(counts, key, 0) + 1
    end
    return Dict(lowercase(name) => i for (i, name) in enumerate(names) if counts[lowercase(name)] == 1)
end

"""
    read_fieldtrip_neighbors(filepath::AbstractString; channel_order=nothing)

Read a FieldTrip/MNE neighbor definition file (`.mat` format) and convert it to an adjacency matrix.

# Arguments
- `filepath::AbstractString`: Path to the neighbor `.mat` file

# Keyword Arguments
- `channel_order::Union{Nothing,AbstractVector{<:AbstractString}} = nothing`: Optional vector 
  of channel names specifying the desired order of channels in the adjacency matrix. If provided,
  the adjacency matrix will be reordered to match your data's channel order. **Pass the channel
  order of the data the adjacency will be applied to** — row `i` of the returned matrix describes
  the neighbourhood of `channel_order[i]`. Labels are matched exactly first, then case-insensitively
  (e.g. "FPz" against a file's "Fpz"); channels not found at all will have no neighbors.

# Returns
- `adjacency::Matrix{Bool}`: An N×N boolean adjacency matrix
- `channel_names::Vector{String}`: Vector of channel names in the order they appear in the matrix

# Example files
- `biosemi32_neighb.mat`,`easycap32ch-avg_neighb.mat`, `easycap64ch-avg_neighb.mat`
- `eeg1010_neighb.mat`, `elec1005_neighb.mat`

"""
function read_fieldtrip_neighbors(filepath::AbstractString;
                                   channel_order::Union{Nothing,AbstractVector{<:AbstractString}} = nothing)
    if !isfile(filepath)
        error("File not found: $filepath")
    end
        
    data = Main.MAT.matread(filepath)
        
    neighbors_struct = data["neighbours"]
    labels_data = neighbors_struct["label"]
    neighblabels_data = neighbors_struct["neighblabel"]
 
    # channel names and neighbour lists (normalised: see _as_label_vector)
    file_channels = _as_label_vector(labels_data)
    file_neighblabels = [_as_label_vector(nl) for nl in neighblabels_data]

    length(file_neighblabels) == length(file_channels) ||
        error("Neighbour file has $(length(file_channels)) labels but $(length(file_neighblabels)) neighbour lists: $filepath")

    # Determine output channel order
    if channel_order === nothing
        out_channels = file_channels
    else
        out_channels = String.(channel_order)
    end
    
    n = length(out_channels)

    # Build index mapping: file channel name -> file index
    file_ch_idx = Dict(ch => i for (i, ch) in enumerate(file_channels))

    # Build index mapping: output channel name -> output index
    out_ch_idx = Dict(ch => i for (i, ch) in enumerate(out_channels))

    # Case-insensitive fallbacks. EEG label spelling drifts between files: the
    # neighbour .mat writes "Fpz" where the epoch files write "FPz", and an
    # unmatched channel would silently end up with no neighbours at all. Only
    # unambiguous lowercase keys are indexed, so a genuine duplicate can never
    # resolve to the wrong channel.
    file_ch_idx_ci = _unambiguous_lowercase_index(file_channels)
    out_ch_idx_ci = _unambiguous_lowercase_index(out_channels)
    lookup(exact, ci, name) = haskey(exact, name) ? exact[name] : get(ci, lowercase(name), nothing)

    # Build adjacency matrix in output order
    adj = falses(n, n)
    fallback_matches = Tuple{String, String}[]  # (requested name, file spelling)

    for (i, ch) in enumerate(out_channels)
        # Find this channel in the file
        file_i = lookup(file_ch_idx, file_ch_idx_ci, ch)
        if isnothing(file_i)
            # Channel not in neighbor file - leave with no neighbors
            continue
        end
        if file_channels[file_i] != ch
            push!(fallback_matches, (ch, file_channels[file_i]))
        end

        # Get neighbors from file
        for neigh in file_neighblabels[file_i]
            j = lookup(out_ch_idx, out_ch_idx_ci, neigh)
            if !isnothing(j)
                adj[i, j] = true
                adj[j, i] = true
            end
        end
    end

    isempty(fallback_matches) ||
        @info "Matched channel labels case-insensitively (requested => file spelling)" pairs=fallback_matches

    return adj, out_channels
end
