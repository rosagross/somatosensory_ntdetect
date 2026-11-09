'''
In this file the timing of the experiment (main block) is checked. 
Aspects that are checked:
- trigger occurences
- trigger times of the electrical stimulation
- trigger times of the mechanical stimulation 
'''

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import pandas as pd
import scipy.io as sio
import seaborn as sns
import glob

# %% load data 
RAWDIR = os.getenv('RAWDIR')
DATADIR = os.getenv('DATADIR')

session_type = 'NT' 
subID = '19'
modality = 'Mechanical'  # 'Electrical' or 'Mechanical'
subjects = np.arange(48, 49)

# %%
times_df = pd.DataFrame()

for subID in subjects:
    subID = str(subID)

    # read in the eeg raw data
    EEG_data_dir = os.path.join(RAWDIR, 'sub-0'+subID, 'ID'+subID+'_'+modality, 'converted_EEG', session_type) 
    file_name = '*.set'
    all_sessions = glob.glob(os.path.join(EEG_data_dir, file_name))
    if len(all_sessions) == 0:
        print(f"No EEG .set files found for subject {subID} modality {modality}, skipping...")
        continue
    ses = all_sessions[1]
    raw = mne.io.read_raw_eeglab(ses, preload=True)
    raw.set_channel_types({
        'ECG': 'ecg',
        'VEOG': 'eog',
        'CNAP': 'misc'  # the median nerve measurement
    })
    events_from_annot, event_dict = mne.events_from_annotations(raw)

    # Plot events on a time scale to get an overview of the experiment 
    fig = mne.viz.plot_events(
        events_from_annot, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp
    )


    # Define epoch parameters
    tmin = -0.051  # start time before event
    tmax = 0.2   # end time after event
    baseline = (None, 0)  # baseline correction (None means start of the epoch)
    event_id = {'B - Stimulation': 15, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6} # event_dict

    # load the epochs
    epochs = mne.Epochs(raw, events_from_annot, event_id, tmin, tmax, baseline=baseline)

     
    # just load the annotations to analyze the sequence timing 

    # check timing of the trials
    button_mapping = int(subID)%2
    startstop_trigger = '171'
    if modality == 'Electrical':
        stim_trigger = 'A - Stimulation'
    elif modality == 'Mechanical':
        stim_trigger = 'B - Stimulation'

    event_id = event_dict[stim_trigger]

    if (button_mapping == 1) and (session_type=='NT'):
        event_meaning = {startstop_trigger:'startstop',stim_trigger:'Stimulation', '1': 'Yes', '2': 'No', 
                            '4': 'CueOnset', '5': 'StimulusOffset', '6': 'RespWindowOnset'}

    elif (button_mapping != 1) and (session_type=='NT'):
        event_meaning = {startstop_trigger:'startstop',stim_trigger:'Stimulation', '2': 'Yes', '1': 'No', 
                        '4': 'CueOnset', '5': 'StimulusOffset', '6': 'RespWindowOnset'}

    events_df = raw.annotations.to_data_frame()
    sfreq = raw.info['sfreq']
    events_df['timestamp'] = events_from_annot[:, 0]/sfreq
    events_df['sample'] = events_from_annot[:, 0]
    events_df['trigger_name'] = events_df['description'].map(event_meaning)
    # trial sequences
    trial_df = events_df[['timestamp', 'trigger_name']]

     
    # TODO: the stimulus offset is saved in data.ao_trigger_post(i,1), we don't have a trigger for this since the experiment code crashed otherwise

    # Check trial timing

    trial_dicts = {'cueonset' : np.array(events_df[events_df['trigger_name']=='CueOnset']['sample']),
                'stimtrigger' : np.array(events_df[events_df['trigger_name']=='Stimulation']['sample']),
                'respwin_onset' : np.array(events_df[events_df['trigger_name']=='RespWindowOnset']['sample'])}

    skip_subject = False
    for key in trial_dicts:
        if not len(trial_dicts[key]) == 198:
            print(f"Subject {subID} modality {modality} is incomplete...")
            skip_subject = True
            
    if skip_subject:
        # skip to next subject
        continue

    trial_df = pd.DataFrame(trial_dicts)/sfreq
    trial_diff = np.diff(trial_df)
    columns = ['cue-stimonset', 'stim-responseonset']
    trial_diff_df = pd.DataFrame(trial_diff,
                    columns=columns)

    times_df = pd.concat([times_df, trial_diff_df], ignore_index=True)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Plot each row (trial) as a separate line
    for ax, col in zip(axes, columns):
        ax.boxplot(trial_diff_df[col], vert=False)
        ax.scatter(trial_diff_df[col], np.ones(len(trial_diff_df[col])))
        ax.set_ylabel(col)
        ax.set_yticks([])

    plt.xlabel('Time (sec)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    sns.despine(fig)

    # check timing between response and next trial
    # split the data into the part that contains responses
    start_exp = events_df[events_df['trigger_name'] == 'RespWindowOnset'].index[0]
    end_exp = events_df[events_df['trigger_name'] == 'RespWindowOnset'].index[-1]
    responses_df = events_df.iloc[start_exp: end_exp]

    response_onsets = []
    button_onsets = []
    cue_starts = []

    cue_samples = np.array(responses_df[responses_df['description']=="4"]['sample'])
    respwin_samples = np.array(responses_df[responses_df['description']=="6"]['sample'])
    button_samples = np.array(responses_df[responses_df['description'].isin(["1","2"])]['sample'])

    for cue, respwin in zip(cue_samples, respwin_samples):
        # Find the next response window after this cue
        # Look for a button press after respwin onset but before next respwin onset
        next_respwin = respwin_samples[respwin_samples > respwin]
        next_respwin = next_respwin[0] if len(next_respwin) > 0 else np.inf
        
        # Find button presses within this response window
        response_in_window = button_samples[(button_samples >= respwin) & (button_samples < next_respwin)]
        
        if len(response_in_window) > 0:
            response_onsets.append(respwin)
            button_onsets.append(response_in_window[0])  # take first button press
            cue_starts.append(cue)

    response_dicts = {
        'response_onset': np.array(response_onsets),
        'button_onset':   np.array(button_onsets),
        'cue_start':      np.array(cue_starts)
}

    response_df = pd.DataFrame(response_dicts)/sfreq
    response_diff = np.diff(response_df)
    columns = ['response-button', 'button_cuestart']
    response_diff_df = pd.DataFrame(response_diff,
                    columns=columns)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Plot each row (trial) as a separate line
    for ax, col in zip(axes, columns):
        ax.boxplot(response_diff_df[col], vert=False)
        ax.scatter(response_diff_df[col], np.ones(len(response_diff_df[col])))
        ax.set_ylabel(col)
        ax.set_yticks([])

    plt.xlabel('Time (sec)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    sns.despine(fig)

# for every trigger name 

# load the timing data (EEG data without real brain activity, just triggers)

# extract one example sequence 

# %%
# plot timing across subjects
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, times_df.columns):
    ax.boxplot(times_df[col], vert=False)
    ax.scatter(times_df[col], np.ones(len(times_df[col])))
    ax.set_ylabel(col)
    ax.set_yticks([])
plt.xlabel('Time (sec)')
plt.legend(loc='upper right')
plt.tight_layout()
sns.despine(fig)

# compute mean and std  
mean_times = times_df.mean()
std_times = times_df.std()
print('Mean times between events:')
print(mean_times)
print('Standard deviation of times between events:')
print(std_times)


# %%
