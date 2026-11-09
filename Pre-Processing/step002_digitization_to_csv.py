'''
Brainstorm digitization results in .m files. Though, we want to use 
'''
# %%
from scipy.io import loadmat
import pandas as pd
import numpy as np
import h5py
import os
import json
from helper_functions_eeg import assign_modality
            
# %%
RAWDIR = os.getenv('RAWDIR')
DATADIR = os.getenv('DATADIR')

subID = '31'
date = '20250410' # make sure that this matches the day with session_path
modality = 'Mechnical'
session_type = 'NT'

ses_label, _ = assign_modality(modality)

output_dir = os.path.join(DATADIR, 'sub-0'+subID) 
data_dir = os.path.join(RAWDIR, 'Digitization', 'NTdetectMechanicElectric', 'data', 'Digitize', 'Subject0'+subID+'_'+date+'_01')
# create directory for the specific session 
session_path = os.path.join(output_dir, f'ses-{ses_label}', 'eeg')
if not os.path.exists(session_path):
    os.makedirs(session_path)

# Load .m file
mat_file = data_dir + f'/channel_Subject0{subID}_{date}_01.mat'

# Open the .mat file
with h5py.File(mat_file, 'r') as f:
        # Access the 'Channel' group
    channels = f['Channel']

    # Extract electrode names
    elec_names = [
        f[ref][()].tobytes().decode('utf-8').replace('\x00', '') 
        for ref in channels['Name'][:, 0]
    ]
    # Extract electrode locations
    elec_locs = [
        f[ref][()] 
        for ref in channels['Loc'][:, 0]
    ]

    # Convert to a DataFrame
    df_channlocs = pd.DataFrame({
        'name': elec_names,
        'x': [loc[0,0] for loc in elec_locs],
        'y': [loc[0,1] for loc in elec_locs],
        'z': [loc[0,2] for loc in elec_locs]
    })

# save into .tsv file
df_channlocs.to_csv(os.path.join(session_path, f'sub-0{subID}_ses-{ses_label}_task-{session_type}_electrodes.tsv'), sep="\t", index=False) # raw (converting .mat to tsv, no processing/analysis applied)

# head shape points
with h5py.File(mat_file, 'r') as f:
    channels = f['HeadPoints']

    # Extract electrode names
    point_names = [
        f[ref][:].tobytes().decode('utf-8').replace('\x00', '')
        for ref in channels['Label'][:, 0]
    ]
    # Extract electrode locations
    point_locs = [
        loc 
        for loc in channels['Loc']
    ]

    # Convert to a DataFrame
    df_headpoints = pd.DataFrame({
        'name': point_names,
        'x': [loc[0] for loc in point_locs],
        'y': [loc[1] for loc in point_locs],
        'z': [loc[2] for loc in point_locs]
    })

# save into .csv file
df_headpoints.to_csv(os.path.join(session_path, f'sub-0{subID}_ses-{ses_label}_task-{session_type}_headshape.tsv'), sep="\t", index=False) # raw (converting .mat to csv, no processing/analysis applied)

# write coordinate system file
coordsystem = {
    "EEGCoordinateSystem": "Head",
    "EEGCoordinateUnits": "m",
    "EEGCoordinateSystemDescription": (
        "Coordinates exported from Brainstorm."
        "Positions are based on fiducials (LPA, RPA, nasion)"
    )
}

coord_filepath = os.path.join(session_path, f'sub-0{subID}_ses-{ses_label}_task-{session_type}_coordsystem.json')
with open(coord_filepath, "w") as f:
    json.dump(coordsystem, f, indent=4)

# %%
