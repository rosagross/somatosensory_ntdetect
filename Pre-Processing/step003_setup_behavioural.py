'''
Reading in data from several subjects, combining and saving them in one DataFrame.  
'''

# %% imports
import numpy as np
import scipy.io as sio
import os
import pandas as pd
import re
import glob
from helper_functions_behav import *

# define directories
RAWDIR = os.getenv('RAWDIR')
DATADIR = os.getenv('DATADIR')

# %% Load data (.mat files) of the threshold assessment and the main experiment 
# '../../data/NTdetect_Pilot' 

# since I don't know how many blocks/runs we have for every subject I extract it
re_TA = re.compile(r'thr1F_(\d+)_data_block(\d+)_run(\d+).mat')
re_NT = re.compile(r'NTDetect_(\d+)_data_block(\d+)_run(\d+).mat')

# subjects to load
subject_IDs = np.arange(1, 53)
# '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 
# missing: 30M, 32M&E, 36E,37M&E, 46E

# setup DataFrame
df_TA_allsubs = pd.DataFrame()
df_NT_allsubs = pd.DataFrame()
df_psi_allsubs = pd.DataFrame()

for sub in subject_IDs:
    if sub < 10: 
            sub = '0' + str(sub) 
    sub = str(sub)
    print('subject', sub)
    
    rawdata_dir = os.path.join(RAWDIR, 'sub-0' + sub)  #'../../../p_02989/BehaviouralData/Pilot'

    # Define the directory names
    dir_name_mechanical = os.path.join(rawdata_dir, f'ID{sub}_Mechanical/raw_behavioural')
    dir_name_electrical = os.path.join(rawdata_dir, f'ID{sub}_Electrical/raw_behavioural')

    # List all files matching the pattern in both directories
    files_TA_mechanical = glob.glob(os.path.join(dir_name_mechanical, f'thr1F_{sub}_data_block*_run*.mat'))
    files_TA_electrical = glob.glob(os.path.join(dir_name_electrical, f'thr1F_{sub}_data_block*_run*.mat'))
    files_TA = files_TA_mechanical + files_TA_electrical
    
    # List all files matching the pattern in both directories for NT files
    files_NT_mechanical = glob.glob(os.path.join(dir_name_mechanical, f'NTDetect_{sub}_data_block*_run*.mat'))
    files_NT_electrical = glob.glob(os.path.join(dir_name_electrical, f'NTDetect_{sub}_data_block*_run*.mat'))
    files_NT = files_NT_mechanical + files_NT_electrical

    for file in files_TA:
        session_type = 'TA'
        
        block, run, data_TA_exp_data, data_TA_setup, data_TA_p_data = get_data_singlesub(file, re_TA, 'TA')

        # DataFrame THRESHOLD ASSESSMENT contains following details
        # subjectID, block, run, trialID, intensity,
        # response, response decoded, response time, age
        if data_TA_exp_data['resp_btn_map'].item()%2 != 0:
            response_button_mapping = [1, 2]
        else:
            response_button_mapping = [2, 1] 

        response_decoded = np.where(data_TA_exp_data['resp_btn'].item() == response_button_mapping[1], 'No',
                                    np.where(data_TA_exp_data['resp_btn'].item() == response_button_mapping[0], 'Yes', 'None'))

        trialnr = len(data_TA_exp_data['seq'].item())
        modality = data_TA_p_data['stim_type'].item()
        dict_data = {'sub' : np.tile([str(sub)], trialnr),
                    'block' : np.tile([block], trialnr),
                    'run' : np.tile([run], trialnr),
                    'trialID' : np.arange(trialnr),
                    'trial_type' : data_TA_exp_data['seq'].item(),
                    'response' : data_TA_exp_data['resp_btn'].item(), 
                    'response_time' : data_TA_exp_data['resp_t'].item(),
                    'actual_intensity' : data_TA_exp_data['intensity'].item(),
                    'intensity_level' : data_TA_exp_data['intensity'].item(),
                    'response_decoded' : response_decoded,
                    'modality' : modality}
        data_df = pd.DataFrame(dict_data)
        print('Modality', modality)

        # append the dataframe to the dataframe for all subjects
        df_TA_allsubs = pd.concat([df_TA_allsubs, data_df])

        # dataframe for psychometric curve results
        psi_data = {'sub' : str(sub),
                    'block' : block,
                    'run' : run,
                    'threshold' : data_TA_exp_data['PF_params_PM'].item()[0],
                    'slope' : data_TA_exp_data['PF_params_PM'].item()[1],
                    'guess_rate' : data_TA_exp_data['PF_params_PM'].item()[2],
                    'lapse_rate' : data_TA_exp_data['PF_params_PM'].item()[3],
                    'intens1_ratio' : data_TA_setup['intens1_p'].item(),
                    'intens1' : data_TA_exp_data['intens1'].item(),
                    'intens2_ratio' : data_TA_setup['intens2_p'].item(),
                    'intens2' : data_TA_exp_data['intens2'].item(),
                    'intens3_ratio' : data_TA_setup['intens3_p'].item(),
                    'intens3' : data_TA_exp_data['intens3'].item(),
                    'intens4_ratio' : data_TA_setup['intens4_p'].item(),
                    'intens4' : data_TA_exp_data['intens4'].item(),
                    'intens5_ratio' : data_TA_setup['intens5_p'].item(),
                    'intens5' : data_TA_exp_data['intens5'].item(),
                    'intens6_ratio' : data_TA_setup['intens6_p'].item(),
                    'intens6' : data_TA_exp_data['intens6'].item(),
                    'intens7_ratio' : data_TA_setup['intens7_p'].item(),
                    'intens7' : data_TA_exp_data['intens7'].item()}

        psi_df = pd.DataFrame([psi_data])

        df_psi_allsubs =  pd.concat([df_psi_allsubs, psi_df]) 
        
        if modality=='Electrical':
            ses_label = 'elec' 
        elif modality == 'Mechanical':
            ses_label = 'mecha' 

        # create dir and save data 
        behav_dir = os.path.join(DATADIR, 'sub-0' + sub, f'ses-{ses_label}', 'eeg')
        if not os.path.exists(behav_dir):
            os.makedirs(behav_dir)

        data_df.to_csv(os.path.join(behav_dir, f'sub-0{sub}_ses-{ses_label}_task-{session_type}_run-{block}_beh.tsv'), index=False, sep='\t')
        psi_df.to_csv(os.path.join(behav_dir, f'sub-0{sub}_ses-{ses_label}_task-{session_type}_run-{block}_psi.tsv'), index=False, sep='\t')


    for file in files_NT:
        if sub ==38:
            break

        session_type = 'NT'
        block, run, data_NT_exp_data, _, data_NT_p_data = get_data_singlesub(file, re_NT, 'NT')

        # if the button mapping is 1, then YES answer is the 1
        if data_NT_exp_data['resp_btn_map'].item() == 1:
            response_button_mapping = [1, 2]
        else:
            response_button_mapping = [2, 1] 

        response_decoded = np.where(data_NT_exp_data['resp_btn'].item() == response_button_mapping[1], 'No',
                                    np.where(data_NT_exp_data['resp_btn'].item() == response_button_mapping[0], 'Yes', 'None'))

        trialnr = len(data_NT_exp_data['seq'].item())
        modality = str(data_NT_p_data.item()[4])

        # set up data dict --> data frame
        dict_data = {'sub' : np.tile([str(sub)], trialnr),
                    'block' : np.tile([block], trialnr),
                    'run' : np.tile([run], trialnr),
                    'trialID' : np.arange(trialnr),
                    'intensity_level' : data_NT_exp_data['seq'].item(),
                    'near' : np.tile([data_NT_exp_data['near'].item()], trialnr),
                    'response' : data_NT_exp_data['resp_btn'].item(), 
                    'response_time' : data_NT_exp_data['resp_t'].item(),
                    'actual_intensity' : data_NT_exp_data['intensities'].item(),
                    'response_decoded' : response_decoded,
                    'modality' : np.tile(str(data_NT_p_data.item()[4]), trialnr),
                    'fixation onset' : data_NT_exp_data['onset_fix'].item()[:,0],
                    'cue_onset' : data_NT_exp_data['onset_cue'].item()[:,0],
                    'stimulus_onset_pre' : data_NT_exp_data['ao_trigger_pre'].item(),
                    'stimulus_onset_post': data_NT_exp_data['ao_trigger_post'].item(),
                    'Response onset' : data_NT_exp_data['onset_resp'].item()[:,0]}


        dataNT_df = pd.DataFrame(dict_data)
        print('Modality', modality)


        # map the type to the intensity (which percentage it should be, according to the 
        # threshold assessment...)
        unique_intensities = np.unique(np.sort(dataNT_df['actual_intensity']))

        # append the dataframe to the dataframe for all subjects
        df_NT_allsubs = pd.concat([df_NT_allsubs, dataNT_df])

        if modality=='Electrical':
            ses_label = 'elec' 
        elif modality == 'Mechanical':
            ses_label = 'mecha' 

        behav_dir = os.path.join(DATADIR, 'sub-0' + sub, f'ses-{ses_label}', 'eeg')
        if not os.path.exists(behav_dir):
            os.makedirs(behav_dir)

        # save data 
        dataNT_df.to_csv(os.path.join(behav_dir, f'sub-0{sub}_ses-{ses_label}_task-{session_type}_run-{block}_beh.tsv'), index=False, sep='\t')

# %%
