using UnfoldMixedModels

# """
#     find_connected_clusters_2d(data::AbstractMatrix, adjacency::AbstractMatrix{Bool}, threshold::Real)

# Find spatiotemporal clusters in a 2D data matrix (channels × time) using spatial connectivity.

# A cluster is a set of spatiotemporally connected samples where:
# 1. Each sample exceeds the threshold
# 2. Samples are connected in time (consecutive time points)
# 3. Samples are connected in space (via the adjacency matrix)

# # Arguments
# - `data::AbstractMatrix`: A channels × time matrix of test statistics
# - `adjacency::AbstractMatrix{Bool}`: Spatial adjacency matrix (channels × channels)
# - `threshold::Real`: Cluster-forming threshold

# # Returns
# - `clusters::Vector{Vector{CartesianIndex}}`: Vector of clusters, where each cluster 
#   is a vector of CartesianIndex positions in the data matrix
# - `cluster_stats::Vector{Float64}`: Vector of cluster statistics (sum of values in each cluster)

# # Examples
# ```julia
# data = rand(10, 50)  # 10 channels, 50 time points
# adj, _ = read_fieldtrip_neighbors("biosemi64_neighb.mat")
# clusters, stats = find_connected_clusters_2d(data, adj, 1.96)
# ```
# """
function find_connected_clusters_2d(
    data::AbstractMatrix,
    adjacency::AbstractMatrix{Bool},
    threshold::Real,
)
    n_channels, n_times = size(data)
    
    # Validate adjacency matrix
    if size(adjacency) != (n_channels, n_channels)
        error("Adjacency matrix size ($(size(adjacency))) must match number of channels ($n_channels)")
    end
    
    # Find all samples exceeding threshold 
    above_threshold = data .>= threshold
    
    # Track which samples have been assigned to clusters
    visited = falses(size(data))
    
    clusters = Vector{Vector{CartesianIndex{2}}}()
    cluster_stats = Float64[]
    
    # Iterate through all positions
    for idx in CartesianIndices(data)
        # Skip if below threshold or already visited
        if !above_threshold[idx] || visited[idx]
            continue
        end
        
        # Start a new cluster
        cluster = CartesianIndex{2}[]
        queue = [idx]
        cluster_sum = 0.0
        
        while !isempty(queue)
            current = popfirst!(queue)
            
            # Skip if already visited
            if visited[current]
                continue
            end
            
            visited[current] = true
            push!(cluster, current)
            cluster_sum += data[current]
            
            # Find neighbors (spatiotemporal connectivity)
            ch, t = Tuple(current)
            
            # Temporal neighbors (same channel, adjacent time points)
            for new_t in (t-1, t+1)
                if 1 <= new_t <= n_times
                    neighbor = CartesianIndex(ch, new_t)
                    if above_threshold[neighbor] && !visited[neighbor]
                        push!(queue, neighbor)
                    end
                end
            end
            
            # Spatial neighbors (adjacent channels, same time point)
            for new_ch in 1:n_channels
                if adjacency[ch, new_ch]  # Channels are spatial neighbors
                    neighbor = CartesianIndex(new_ch, t)
                    if above_threshold[neighbor] && !visited[neighbor]
                        push!(queue, neighbor)
                    end
                end
            end
        end
        
        # Add cluster if it contains any samples
        if !isempty(cluster)
            push!(clusters, cluster)
            push!(cluster_stats, cluster_sum)
        end
    end
    
    return clusters, cluster_stats
end


# """
#     spatiotemporal_cluster_pvalues(
#         rng::AbstractRNG,
#         observed::AbstractMatrix,
#         permuted::AbstractArray{<:Real,3},
#         adjacency::AbstractMatrix{Bool},
#         threshold::Real
#     )

# Compute p-values for spatiotemporal cluster-based permutation test.

# This function implements a cluster-based permutation test that accounts for 
# both spatial (across channels) and temporal (across time) structure in the data.

# # Arguments
# - `rng::AbstractRNG`: Random number generator for reproducibility
# - `observed::AbstractMatrix`: Observed test statistics (channels × time)
# - `permuted::AbstractArray{<:Real,3}`: Permuted test statistics (channels × time × n_permutations)
# - `adjacency::AbstractMatrix{Bool}`: Spatial adjacency matrix (channels × channels)
# - `threshold::Real`: Cluster-forming threshold

# # Returns
# - `pvalues::Matrix{Float64}`: P-values for each sample (channels × time). Every
#   sample of a cluster carries that cluster's p-value; samples in no cluster are 1.0.
# - `cluster_ids::Matrix{Int}`: 0 = no cluster, +k = k-th positive, -k = k-th negative
# - `null_max_stats::Vector{Float64}`: max cluster mass per permutation (0.0 when a
#   permutation produced no suprathreshold cluster). Keeping this lets significance be
#   re-evaluated at another alpha without repeating the permutations.

# # Algorithm
# 1. Find clusters in observed data using spatiotemporal connectivity
# 2. For each permutation, find clusters and compute max cluster statistic
# 3. Build null distribution from max cluster statistics
# 4. Compute p-value for each observed cluster based on null distribution
# 5. Assign p-values to all samples within each cluster

# # Examples
# ```julia
# observed = modelfit_statistics  # channels × time
# permuted = permutation_statistics  # channels × time × n_perms
# adj, _ = read_fieldtrip_neighbors("biosemi64_neighb.mat")
# pvals = spatiotemporal_cluster_pvalues(rng, observed, permuted, adj, 1.96)
# ```
# """
function spatiotemporal_cluster_pvalues(
    rng::AbstractRNG,
    observed::AbstractMatrix,
    permuted::AbstractArray{<:Real,3},
    adjacency::AbstractMatrix{Bool},
    threshold::Real,
)
    n_channels, n_times = size(observed)
    n_permutations = size(permuted, 3)
    
    # Find positive and negative clusters 
    pos_clusters, pos_cluster_stats = find_connected_clusters_2d(observed, adjacency, threshold)
    neg_clusters, neg_cluster_stats = find_connected_clusters_2d(-observed, adjacency, threshold)
    obs_clusters = [pos_clusters; neg_clusters]
    obs_cluster_stats = [pos_cluster_stats; neg_cluster_stats]

    # Build null distribution of maximum cluster statistics
    null_max_stats = zeros(n_permutations)

    for perm_idx in 1:n_permutations
        perm_data = permuted[:, :, perm_idx]
        _, pos_perm_stats = find_connected_clusters_2d(perm_data, adjacency, threshold)
        _, neg_perm_stats = find_connected_clusters_2d(-perm_data, adjacency, threshold)
        perm_cluster_stats = [pos_perm_stats; neg_perm_stats]

        if !isempty(perm_cluster_stats)
            null_max_stats[perm_idx] = maximum(perm_cluster_stats)
        else
            null_max_stats[perm_idx] = 0.0
        end
    end
    
    # Compute p-values for each observed cluster
    pvalues = ones(n_channels, n_times)    # Initialize with 1.0 (non-significant)
    cluster_ids = zeros(Int, n_channels, n_times)  # 0 = no cluster; +k = k-th pos cluster; -k = k-th neg cluster

    n_pos = length(pos_clusters)

    for (i, (cluster, cluster_stat)) in enumerate(zip(obs_clusters, obs_cluster_stats))
        # Count how many permutations had a max cluster stat >= this cluster's stat.
        # The +1 in numerator and denominator counts the observed data as one draw from
        # the null, so p can never be 0: the floor is 1/(n_permutations + 1). At 2000
        # permutations that is ~5e-4, which must be reported as "p < .001", not "p = 0".
        n_extreme = sum(null_max_stats .>= cluster_stat)
        cluster_pvalue = (n_extreme + 1) / (n_permutations + 1)

        cluster_id = i <= n_pos ? i : -(i - n_pos)

        for idx in cluster
            pvalues[idx] = cluster_pvalue
            cluster_ids[idx] = cluster_id
        end
    end

    # null_max_stats is returned so callers can persist the null distribution alongside
    # the p-values; without it the permutation result cannot be re-thresholded later.
    return pvalues, cluster_ids, null_max_stats
end

