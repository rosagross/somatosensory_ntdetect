# %%
import os 
import mne 
import matplotlib.pyplot as plt
import numpy as np
from mne.channels import find_ch_adjacency
from scipy.io import savemat, loadmat

DATADIR = os.getenv('DATADIR')
RESDIR = os.getenv('RESDIR')

# fif epoch filename
subIDs = [13] # np.arange(13,52)
modalities = ['mecha'] #, 'elec']
for subID in subIDs:
    for modality in modalities:
        sub_dir = os.path.join(DATADIR, 'derivatives', 'eeg-preproc', f'sub-0{subID}', f'ses-{modality}')
        all_epochs_filename = f"sub-0{subID}_ses-{modality}_task-NT_epochs"
        epoch_fif = os.path.join(sub_dir, f"{all_epochs_filename}.fif")
                
        # load epoch 
        try:
            epochs = mne.read_epochs(epoch_fif, preload=True)
        except:
            print('no epoch file for sub', subID)
            continue

        # load adjacency
        adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type="eeg")

        # get the data from the adjacency matrix into arrays
        adj_array = adjacency.nonzero()
        label_idx = adj_array[0]
        neighbour_idx = adj_array[1]

        neighbours = []
        labels = []

        ch_names = np.array(ch_names)
        ch_names = np.where(ch_names == 'FPz', 'Fpz', ch_names)

        # put the neighbour for each channel in separate lists
        for i, ch in enumerate(ch_names):

            print(i, ch)
            current_label_idx = label_idx==i
            current_neighb_idx = neighbour_idx[current_label_idx]
            neighb_labels = ch_names[current_neighb_idx]

            # delete the target channel from the list
            mask = neighb_labels != ch
            result = neighb_labels[mask]

            # save in dict
            neighbours.append(result)


        # save them into .mat file
        dtype = [('label', 'O'), ('neighblabel', 'O')]
        fields = np.empty((1, len(ch_names)), dtype=dtype)
        fields['label'] = ch_names
        fields['neighblabel'] = neighbours
        savemat(os.path.join(sub_dir, f'sub-0{subID}_ses-{modality}_task-NT_channel_adjacency.mat'), {'neighbours': fields})

        print('Saved file!')