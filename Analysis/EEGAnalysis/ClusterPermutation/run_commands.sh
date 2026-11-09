

# Required core settings
export SUBJECTS=013,014,015,016,017 # example, empty for all
export MODALITY=both	                # mecha | elec | both


# Data stems
export EPOCH_STEM=epochs_preprestim_corrected
export METADATA_STEM=allmetadata_preprestim

# Permutation + RNG
export N_PERMUTATIONS=2000
export RNG_SEED=1

# Time window for cropping
export TIME_START=-0.55
export TIME_STOP=0.3

export BASELINE_START=-0.55
export BASELINE_STOP=-0.45

# Per-modality baseline override (only needed for MODALITY=both). Takes precedence over
# BASELINE_START/BASELINE_STOP above. 
export BASELINE_START_ELEC=-0.55
export BASELINE_STOP_ELEC=-0.45
export BASELINE_START_MECHA=-0.55
export BASELINE_STOP_MECHA=-0.45

# Plotting / selection (optional)
export PLOT_CHANNEL=C6              # single channel
export PLOT_CHANNELS="C6,CP4"             # e.g. "C4,Cz,Pz" (overrides PLOT_CHANNEL)
export PLOT_START=""                # e.g. -0.1
export PLOT_STOP=""                 # e.g. 0.25

# Coefficient selector (optional)
# COEF=all | intensity_level_centered | response_decoded | force | modality |
# for interactions: intensity_response OR modality_response OR modality_intensity
export COEF=all

# Julia path
export JULIA_DEPOT_PATH=/data/p_02989/shared_workspace/grossmannr_wd/.julia_depot

/data/u_grossmannr_software/julia/.julia/juliaup/julia-1.11.6+0.x64.linux.gnu/bin/julia \
  --project=Analysis/EEGAnalysis/ClusterPermutation \
  Analysis/EEGAnalysis/ClusterPermutation/unfold_cluster_permutation_clusterMass.jl

