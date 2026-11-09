# Temporal (single-channel) cluster-mass permutation test.
# Run from the working directory ($WDDIR), with DATADIR/RESDIR already exported.

export SUBJECTS="013,014,015,016,017,018,019,020,021,022"
export MODALITY="both"
export CHANNELS=C6           # exactly one channel; CHANNELS=all is rejected
export EPOCH_STEM=epochs_preprestim_corrected
export METADATA_STEM=allmetadata_preprestim
export N_PERMUTATIONS=2000
export RNG_SEED=1
export TIME_START=-0.55
export TIME_STOP=0.3
export COEF="all" # all, response_decoded, intensity_level_centered, intensity_response, force, modality

# Surface Laplacian (CSD): set to true to run on the CSD cache written by
# compute_csd_epochs.py instead of the average-referenced epochs. Generate it first with:
#   python Analysis/EEGAnalysis/ClusterPermutation/compute_csd_epochs.py
# Note the CSD is a spatial high-pass: it strongly attenuates slow potentials such as the
# CNV in the cue-stimulus foreperiod. Amplitudes become uV/m^2; the cluster z-statistic is
# unaffected by the scale change.
export LAPLACIAN=false
# Baseline:
export BASELINE_START_ELEC=-0.55
export BASELINE_STOP_ELEC=-0.45
export BASELINE_START_MECHA=-0.55
export BASELINE_STOP_MECHA=-0.45
export JULIA_DEPOT_PATH=/data/p_02989/shared_workspace/grossmannr_wd/.julia_depot

/data/u_grossmannr_software/julia/.julia/juliaup/julia-1.11.6+0.x64.linux.gnu/bin/julia \
  --project=Analysis/EEGAnalysis/ClusterPermutation \
  Analysis/EEGAnalysis/ClusterPermutation/unfold_temporal_permutation_clusterMass.jl
  
