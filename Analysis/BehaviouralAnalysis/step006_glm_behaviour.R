library(ggplot2)
library(dplyr)
library(broom)
library(readr)
library(mgcv)
library(lme4)
library(data.table)

# Get the DATADIR environment variable and read in data
datadir <- Sys.getenv("DATADIR")
resdir <- Sys.getenv("RESDIR")
fig_path <- file.path(resdir, 'Figures', 'Main', 'behavioural_results')
behav_dir <- file.path(datadir, "derivatives", "eeg-preproc")
stats_res <- file.path(resdir, "Statistics", "behav")

# subject IDs and file names
subjects <- sprintf("%03d", 13:52)
modalities <- c('elec', 'mecha')

# Pre-allocate list to store each subject's data
df_list_meta <- list()

# Load and process each subject
for (mod in modalities) {
  for (i in seq_along(subjects)) {
    sub_id <- subjects[i]
    
    file_path_meta <- file.path(behav_dir,
                                paste0("sub-", sub_id), paste0("ses-", mod),
                                paste0("sub-", sub_id, "_ses-", mod, "_task-NT_allmetadata.tsv")
    )
    
    # Skip iteration if file does not exist
    if (!file.exists(file_path_meta)) {
      message(paste("No data for subject", sub_id))
      next
    }
    
    # Read large file efficiently
    df_metadata <- fread(file_path_meta, sep = "\t")
    
    # Convert intensity_level to factor and numeric
    df_metadata$intensity_level <- factor(df_metadata$intensity_level)
    df_metadata$modality <- factor(df_metadata$modality)
    df_metadata$intensity_num <- as.numeric(df_metadata$intensity_level) -1
    df_metadata$response_bin <- ifelse(df_metadata$response_decoded == "Yes", 1, ifelse(df_metadata$response_decoded == "No", 0, NaN))
    
    # exclude bad data
    df_metadata <- df_metadata[analysis_excluded == FALSE]
    
    desired_cols <- c("sub", "modality", "block", "intensity_level", "intensity_norm", "intensity_num", "response_decoded", "response_bin", "force_trigger_exists", "response_time", "force", "force_quantile", "force_zeromean", "analysis_excluded")
    
    df_metadata <- df_metadata[, desired_cols, with = FALSE]
    df_list_meta[[length(df_list_meta) + 1]] <- df_metadata
  }
}
# Combine all into one big data.table
df_all_meta <- rbindlist(df_list_meta)
df_all_meta <- as.data.table(df_all_meta)
df_all_meta <- df_all_meta[!is.na(response_time) &
                             intensity_num < 7 & 
                             intensity_num > 1 &
                             response_time > 0]

# 5. check if we can predict response time using intensity level and response
df_all_meta$intensity_num_z <- scale(df_all_meta$intensity_num)
df_all_meta$intensity_centered <- scale(df_all_meta$intensity_num, center=TRUE, scale =FALSE)
df_all_meta$block_z <- scale(df_all_meta$block)

# filter for mechanical only and force values with triggers
df_mecha <- df_all_meta[
  modality == "Mechanical" &
    !is.na(response_decoded) & 
    !is.na(force) &
    !is.na(force_quantile) #&
    #force_trigger_exists == 'Yes'
]

df_elec <- df_all_meta[
  modality == "Electrical" &
    !is.na(response_decoded) 
]

# convert force levels to numbers for analysis
df_mecha$force_quantile <- factor(df_mecha$force_quantile, levels = c("Low", "Medium", "High"), ordered = TRUE)
df_mecha$force_quantile_num <- as.numeric(df_mecha$force_quantile)

# normalize force values
df_mecha$force_normalized <- (df_mecha$force - min(df_mecha$force, na.rm = TRUE)) / (max(df_mecha$force, na.rm = TRUE) - min(df_mecha$force, na.rm = TRUE))


sapply(df_mecha[, .(modality, intensity_level, force, force_quantile_num)], function(x) length(levels(x)))
table(df_all_meta$modality)
table(df_all_meta$intensity_level)
table(df_all_meta$block)
table(df_mecha$force_quantile_num)
any(df_all_meta$response_time <= 0)


library(lmerTest)

# build four models:
# 1. for investigating the force effect only for mechanical data)
model1 <- glmer(response_bin ~ intensity_centered + force_normalized + block_z + (1|sub), family = binomial(link = "logit"), data = df_mecha)
# 2. check the quantiles bins not continuous force
model2 <- glmer(response_bin ~ intensity_centered + force_quantile_num + block_z + (1+block_z|sub), family = binomial(link = "logit"), data = df_mecha)
# this one seems to have the lowest AIC, but when I add  + (1|sub) the model failed to converge
model2_1 <- glmer(response_bin ~ intensity_centered + force_quantile_num + block_z + (1|sub), family = binomial(link = "logit"), data = df_mecha)
model2_2 <- glmer(response_bin ~ intensity_centered + force_quantile_num + (1|sub) + (0+block_z|sub), family = binomial(link = "logit"), data = df_mecha)
# 2.3. electrical model
model2_3 <- glmer(response_bin ~ intensity_centered + (1|sub) + (0+block|sub), family = binomial(link = "logit"), data = df_elec)

# 3. this time using normed intensity values (not the numbers 2-6) 
model3 <- glmer(response_bin ~ intensity_norm + force_normalized + block_z + (1|sub), family = binomial(link = "logit"), data = df_mecha)
model3_3 <- glmer(response_bin ~ intensity_norm + force_normalized + block_z + (1+block_z|sub), family = binomial(link = "logit"), data = df_mecha)
# 4. basic check of intensity level effect on response
model4 <- glmer(response_bin ~ modality + intensity_centered + block_z + (1|sub), family = binomial(link = "logit"), data = df_all_meta)
model4_inter <- glmer(response_bin ~ modality * intensity_centered + (1|sub) + (0+block_z|sub), family = binomial(link = "logit"), data = df_all_meta)


model5 <- glmer(response_time ~ modality + intensity_centered * response_bin + block_z + (1+block_z|sub), family = Gamma(link = "log"), data = df_all_meta)
model6 <- glmer(response_time ~ modality + intensity_centered * response_bin + block_z + (1|sub), family = Gamma(link = "log"), data = df_all_meta)
# this converges
model_z <- glmer(response_time ~ modality + intensity_num_z * response_bin + block_z_z + (1+block_z|sub), family = Gamma(link = "log"), data = df_all_meta)

# compute AIC
AIC(model1, model2, model2_1, model2_2, model3, model3_3)
AIC(model4, model4_inter)
AIC(model5, model6, model_z)

# MECHANICAL model: write and save the results
coef_summary <- as.data.frame(summary(model2_2)$coefficients)
coef_summary$formula <- paste(deparse(formula(model2_2)), collapse=" ")
coef_summary$convergence <- model2_2@optinfo$conv$lme4$messages
write.csv(coef_summary, file = file.path(stats_res, "model_mecha_summary.csv"), row.names = TRUE)
coef_summary

# write and save the results
coef_summary <- as.data.frame(summary(model4_inter)$coefficients)
coef_summary$formula <- paste(deparse(formula(model4_inter)), collapse=" ")
coef_summary$convergence <- model4_inter@optinfo$conv$lme4$messages
write.csv(coef_summary, file = file.path(stats_res, "model_combined_summary.csv"), row.names = TRUE)
coef_summary

# ELECTRICAL model: write and save the results
coef_summary <- as.data.frame(summary(model2_3)$coefficients)
coef_summary$formula <- paste(deparse(formula(model2_3)), collapse=" ")
coef_summary$convergence <- model2_3@optinfo$conv$lme4$messages
write.csv(coef_summary, file = file.path(stats_res, "model_elec_summary.csv"), row.names = TRUE)
coef_summary


# write and save the results
coef_summary <- as.data.frame(summary(model5)$coefficients)
coef_summary$formula <- paste(deparse(formula(model5)), collapse=" ")
coef_summary$convergence <- model5@optinfo$conv$lme4$messages
write.csv(coef_summary, file = file.path(stats_res, "model5_summary.csv"), row.names = TRUE)
coef_summary

coef_summary <- as.data.frame(summary(model_z)$coefficients)
coef_summary$formula <- paste(deparse(formula(model_z)), collapse=" ")
coef_summary$convergence <- model_z@optinfo$conv$lme4$messages
write.csv(coef_summary, file = file.path(stats_res, "model_z_summary.csv"), row.names = TRUE)
coef_summary

