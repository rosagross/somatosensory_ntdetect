"""
File: step003_1_eegprepro_automatic.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-08-07
Description: 

1. Load Data
2. Montage Mapping
3. Apply bandpass filter
4. Downsample and detrend 
5. Set reference to average
6. Extract the block and run number 
6. Detect bad channels 
    6.1 by powerspectrum density
    6.2 by manual channel rejection (noted in preproc_notes.json)
8. Cut data in epochs around the stimulus
9. Extract ECG, respiration and force data
10. Bad channel detection with Ransac (manually check outcome!)
11. ICA cleaning: Filter eye movements, muscle activity, line noise
12. Save cleaned epochs
13. Bad epoch detection (via threshold)

"""

# %%
import os
import json
import glob
import argparse
from datetime import datetime, timedelta 
from pathlib import Path
import pickle
import mne
print(mne.__version__)
print(mne.__file__)
import numpy as np
from scipy import interpolate
import pandas as pd
from meegkit.detrend import detrend
from autoreject import Ransac  # Jas et al., 2016
from mne_icalabel import label_components
from mne.preprocessing import ICA
from helper_functions_eeg import *

# we want to loop through the session folders
RAWDIR = os.getenv("RAWDIR")
DATADIR = os.getenv("DATADIR")
RESDIR = os.getenv("RESDIR")
WDDIR = os.getenv("WDDIR")
figuredir = os.path.join(RESDIR, "Figures", "Main", "prepro_results")

# %%
# Read in preprocessing parameters
with open(os.path.join(WDDIR, 'Pre-Processing', 'preproc_parameter.json'), 'r') as json_file:
    params = json.load(json_file)

# %%

# parse subjects from command line
#subIDs, modalities = parse_subjects()
subIDs = np.arange(13, 53)
modalities = ['Electrical', 'Mechanical'] #, 'Electrical']
file_suffix = '_preprestim'
cnap_only = False

# Due to technical issues we only have force trigger available for some subjects 
subswithtrigger = params["scale"]["subs_with_trigger"].keys()
session_type = params["session_type"]

# 31, 34, 35, 43 sampling issues
# 38 no mechanical session

# %%

for modality in modalities:
    ses_label, stim_trigger = assign_modality(modality)

    for subID in subIDs:
        subID = str(subID)
        print("sub", subID, modality)

        EEG_data_dir = os.path.join(
            RAWDIR,
            "sub-0" + subID,
            "ID" + subID + "_" + modality,
            "converted_EEG",
            session_type,
        )
        output_dir = os.path.join(DATADIR, "sub-0" + subID)
        montage = params["montage"]["montage_type"]
        digi_file = f"sub-0{subID}_ses-{ses_label}_task-{session_type}_electrodes.tsv"
        hp_file = f"sub-0{subID}_ses-{ses_label}_task-{session_type}_headshape.tsv"

        # create directory for the specific session
        session_path = os.path.join(output_dir, f"ses-{ses_label}", "eeg")
        if not os.path.exists(session_path):
            os.makedirs(session_path)

        # init dict to store information
        preprocessing_notes = {}  # saves all prepro steps that were executed

        all_sessions = glob.glob(os.path.join(EEG_data_dir, "*.set"))
        sorted_sessions = sorted(all_sessions, key=natural_session_key)
        # also extract all .csv files with the extracted start times
        all_starttimes = glob.glob(os.path.join(EEG_data_dir, "*.csv"))

        # loop through all .set files
        for i, ses in enumerate(sorted_sessions):
            print(ses)
            print("Current working directory:", os.getcwd())

            raw = mne.io.read_raw_eeglab(ses, preload=True)
            raw.set_channel_types(
                {
                    "ECG": "ecg",
                    "VEOG": "eog",
                    "CNAP": "misc",  # the median nerve measurement
                }
            )
         
            preprocessing_notes["Subject"] = subID
            preprocessing_notes["SessionFile"] = ses

            # 2. Montage mapping
            #########################

            if montage == "individual":
                # Load digitization data
                digi_path = os.path.join(output_dir, f"ses-{ses_label}", "eeg")
                head_file = os.path.join(digi_path, hp_file)
                headpoint_df = pd.read_csv(head_file, keep_default_na=False, sep="\t")
                channel_file = os.path.join(digi_path, digi_file)
                channel_df = pd.read_csv(channel_file, keep_default_na=False, sep="\t")

                ch_pos = {}  # Empty dictionary for channel positions
                coord_frame = "head"  # Fixed as required
                nasion = np.mean(
                    [
                        np.array(headpoint_df[["x", "y", "z"]].iloc[0]),
                        np.array(headpoint_df[["x", "y", "z"]].iloc[3]),
                    ],
                    axis=0,
                )
                lpa = np.mean(
                    [
                        np.array(headpoint_df[["x", "y", "z"]].iloc[1]),
                        np.array(headpoint_df[["x", "y", "z"]].iloc[4]),
                    ],
                    axis=0,
                )
                rpa = np.mean(
                    [
                        np.array(headpoint_df[["x", "y", "z"]].iloc[2]),
                        np.array(headpoint_df[["x", "y", "z"]].iloc[5]),
                    ],
                    axis=0,
                )
                hsp = np.array(
                    headpoint_df[["x", "y", "z"]][6:]
                )  # Can be customized if needed
                hpi = None  # Can be customized if needed
                ch_names_new = []
                ch_names_old = raw.info.ch_names
                n_channels = len(ch_names_old)

                for _, row in channel_df.iterrows():
                    ch_names_new.append(row["name"])
                    coordinates = np.array([row["x"], row["y"], row["z"]])
                    ch_pos[row["name"]] = coordinates

                # Create and set digitization montage
                dig = mne.channels.make_dig_montage(
                    ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, hpi=None
                )  # , coord_frame=coord_frame)

                # set montage
                # set Respiration channel type
                if (
                    (int(subID) > 10)
                    or ((int(subID) == 8) & (modality == "Electrical"))
                    or ((int(subID) == 10) & (modality == "Mechanical"))
                    or ((int(subID) == 7) & (modality == "Electrical"))
                ):
                    raw.set_channel_types({"RESP": "resp"})
                raw.set_montage(dig)

            elif montage == "standard":
                # Check if 'FPz' exists in the data set and rename it to the standard 'Fpz'
                # .. for some reason sometimes they use the lower case letter..
                if "FPz" in raw.info["ch_names"]:
                    # Rename 'FPz' to 'Fpz'
                    print("rename")
                    raw.rename_channels({"FPz": "Fpz"})

                # set Respiration channel type
                raw.set_channel_types({"RESP": "resp"})

                # Apply the standard 10-05 montage to the EEG channels
                montage_1005 = mne.channels.make_standard_montage("standard_1005")
                raw.set_montage(montage_1005)
            
            preprocessing_notes["Montage"] = montage

            # 3. Filter with bandpass
            #########################
            raw_filtered = raw.copy()
            l_freq_orig = params["filter"]["low_freq"]
            h_freq_orig = params["filter"]["high_freq"]
            _ = raw_filtered.filter(l_freq=l_freq_orig, h_freq=h_freq_orig)
            preprocessing_notes["FilterLow"] = l_freq_orig
            preprocessing_notes["FilterHigh"] = h_freq_orig

            # 4. Downsample and detrend
            ###########################
            sfreq_downsampled = params["downsampling"]["sfreq_downsampled"]
            sfreq = raw_filtered.info["sfreq"]
            raw_downsampled = raw_filtered.copy().resample(sfreq_downsampled)
            preprocessing_notes["downsampled_from"] = sfreq
            preprocessing_notes["downsampled_to"] = sfreq_downsampled
            X = (
                raw_downsampled.get_data().T
            )  # transpose so the data is organized time-by-channels
            X, _, _ = detrend(X, order=1)
            preprocessing_notes["detrend"] = "yes"
            raw_downsampled._data = X.T  # overwrite raw data
            raw_filtered = raw_downsampled.copy()

            # 5. Set reference to average
            #############################
            ref_type = params["reference"]["type"]
            raw_filtered.set_eeg_reference(
                ref_type, projection=True
            )  # compute the reference
            raw_filtered.apply_proj()
            preprocessing_notes["Reference"] = ref_type

        
            # HANDLE EXCEPCTIONS
            # pay attention to the following exception:
            exception7 = '2025-02-11T153830_session4.set'

            # expecption: 10 test stimuli were executed after the block (but still recorded) so this part
            # has to be cropped/cut out
            if (
                (subID == "16")
                & (modality == "Electrical")
                & (ses == os.path.join(EEG_data_dir, exception7))
            ):
                raw_filtered.crop(tmax=600)

            # get ALL events (start, stop, stimulus, cue, button, ...) to get the start and endpoint of measurement
            events_from_annot, event_dict = mne.events_from_annotations(raw_filtered)


            # 6. Extract the block and run number
            #####################################
            block, run, startstop_trigger = get_block_run(events_from_annot, event_dict, ses, EEG_data_dir, subID, modality)

            # handle exceptions
            block, run = check_exceptions(modality, block, run, subID, ses, EEG_data_dir)

            print("block:", block)
            print("run:", run)
            preprocessing_notes["block"] = block
            preprocessing_notes["run"] = run

            if not cnap_only:
                # 7. Detect bad channels by powerspectrum density and manually
                ######################################################
                
                bads_psd = reject_chan_psd(raw_filtered)
                
                # if there are manually selected channels, add them to the bads list
                try: 
                    bads_manual = params["bad_channels"][str(subID)][modality][str(block)]
                except:
                    bads_manual = []
                
                bads = bads_psd + bads_manual
                # interpolate bad channels
                interpolate_bads(raw_filtered, bads)
                
                # note which channels were interpolated using psd 
                preprocessing_notes["PsdInterpolated"] = f"{bads_psd}"
                preprocessing_notes["PsdNrInterpolatedChan"] = f"{len(bads_psd)}"
                # note which channels were interpolated manually
                preprocessing_notes["ManuallyInterpolated"] = f"{bads_manual}"
                preprocessing_notes["ManuallyNrInterpolatedChan"] = f"{len(bads_manual)}"


            # 8. CUT DATA IN EPOCHS around the stimulus
            ################################################

            # Define epoch parameters
            tmin = params["epochs"]["tmin"]  # start time before event
            tmax = params["epochs"]["tmax"]  # end time after event
            # baseline = (None, 0)  # baseline correction (None means start of the epoch)
            event_id = event_dict[
                stim_trigger
            ]  # chose the event that you want to extract

            # load annotations
            annotations = raw_filtered.annotations.to_data_frame()

            # load event annotations with meaning into data frame
            button_mapping = int(subID) % 2

            # the meaning of the triggers depends on the button mapping and the session type
            if (button_mapping == 1) and (session_type == "NT"):
                event_meaning = {
                    startstop_trigger: "startstop",
                    stim_trigger: "Stimulation",
                    "1": "yes",
                    "2": "No",
                    "4": "CueOnset",
                    "5": "StimulusOffset",
                    "6": "RespWindowOnset",
                    "20": "Catch",
                    "21": "IntensLevel1",
                    "22": "IntensLevel2",
                    "23": "IntensLevel3",
                    "24": "IntensLevel4",
                    "25": "IntensLevel5",
                    "26": "IntensLevel6",
                    "27": "IntensLevel7",
                }
            elif (button_mapping == 1) and (session_type == "TA"):
                event_meaning = {
                    startstop_trigger: "startstop",
                    stim_trigger: "Stimulation",
                    "1": "yes",
                    "2": "No",
                    "3": "TrialStart",
                    "4": "CueOnset",
                    "5": "StimulusOffset",
                    "6": "RespWindowOnset",
                }

            elif (button_mapping != 1) and (session_type == "NT"):
                event_meaning = {
                    startstop_trigger: "startstop",
                    stim_trigger: "Stimulation",
                    "2": "yes",
                    "1": "No",
                    "4": "CueOnset",
                    "5": "StimulusOffset",
                    "6": "RespWindowOnset",
                    "20": "Catch",
                    "21": "IntensLevel1",
                    "22": "IntensLevel2",
                    "23": "IntensLevel3",
                    "24": "IntensLevel4",
                    "25": "IntensLevel5",
                    "26": "IntensLevel6",
                    "27": "IntensLevel7",
                }
            elif (button_mapping != 1) and (session_type == "TA"):
                event_meaning = {
                    startstop_trigger: "startstop",
                    stim_trigger: "Stimulation",
                    "1": "No",
                    "2": "yes",
                    "3": "TrialStart",
                    "4": "CueOnset",
                    "5": "StimulusOffset",
                    "6": "RespWindowOnset",
                }
            else:
                raise ValueError("No valid button mapping!")

            sfreq = raw_filtered.info["sfreq"]
            annotations["total_secs"] = events_from_annot[:, 0] / sfreq # the first entry contains sampling time
            annotations["trigger_meaning"] = annotations["description"].map(
                event_meaning
            )

            # check there is a trial missing (this only happened in Mechanical stimulation sessions)
            if modality == "Mechanical":
                triggertimes = annotations[
                    annotations["description"] == "B - Stimulation"
                ]["total_secs"]
                cue_stim = 0.436  # time between cue onset and stim onset
                added_annots = pd.DataFrame()

                for i, row in annotations.iterrows():
                    trigger_meaning = row["trigger_meaning"]

                    if trigger_meaning == "CueOnset":
                        # check if the next one is a stimulus
                        stim_idx = i + 1
                        if (
                            annotations.iloc[stim_idx]["trigger_meaning"]
                            != "Stimulation"
                        ):
                            # compute the seconds timepoint and onset of the missed stimulus
                            stim_timestamp = row["total_secs"] + cue_stim
                            stim_onsettime = row["onset"] + timedelta(seconds=cue_stim)

                            # add event to annotations here
                            new_annotation = {
                                "onset": stim_onsettime,  # if present
                                "duration": 0.0,
                                "description": "B - Stimulation",
                                "total_secs": stim_timestamp,
                                "trigger_meaning": "Stimulation",
                            }

                            added_annots = pd.concat(
                                (added_annots, pd.DataFrame([new_annotation]))
                            )

                preprocessing_notes["MissingTrialsAdded"] = len(added_annots)

                # append lost triggers and sort by timestamp
                annotations = pd.concat([annotations, added_annots], ignore_index=True)
                annotations = annotations.sort_values(by="total_secs").reset_index(
                    drop=True
                )

            else:
                preprocessing_notes["MissingTrialsAdded"] = 0

            annotations["block"] = block
            annotations["run"] = run
            annotations["subID"] = subID

            # delete the first 200seconds of anotations from sub 14 elec block 1 (dont just crop them!!)
            # otherwise part of the signal is lost when setting new annotations! 
            exception6 = '2025-02-04T144324_session1.set' # sub14 elec block1
            if (
                (subID == "14")
                & (modality == "Electrical")
                & (ses == os.path.join(EEG_data_dir, exception6))
            ):
                annotations = annotations.iloc[281:]

            # in sub 52 elec block 8 the first stimulus cue is not recorded
            # if we want a bigger window around the stimulus we need to remove the first stimulus
            exception9 = '2025-06-27T132126_session18.set' # sub 52 elec block8
            if (
                (subID == "52")
                & (modality == "Electrical")
                & (ses == os.path.join(EEG_data_dir, exception9))
                & (tmin <= -0.2)
            ):
                annotations = annotations.iloc[4:]

            # set the annotations
            raw_filtered.set_annotations(
                mne.Annotations(
                    onset=annotations["total_secs"].values,
                    duration=annotations["duration"].values,
                    description=annotations["description"].values,
                    orig_time=annotations["orig_time"].iloc[0]
                    if "orig_time" in annotations.columns
                    else None,
                )
            )
            # reload the events
            events_from_annot_updated, event_dict_updated = mne.events_from_annotations(
                raw_filtered
            )

            # load epochs
            epochs = mne.Epochs(
                raw_filtered,
                events_from_annot_updated,
                event_id,
                tmin,
                tmax,
                preload=True,
                baseline=None,
            )
            
            # for some subjects the starting trigger was not assigned properly
            # here we deal with the exceptions so that the signal is cut correctly
            if (subID in ['38', '39']) and (modality == "Electrical") and (block == 2):
                handle_startstop_exceptions(annotations, '25')
            elif (subID in ['38', '39']) and (modality == "Electrical") and (block == 5):
                handle_startstop_exceptions(annotations, '23')
            elif (subID in ['38', '39']) and (modality == "Electrical") and (block == 8):
                handle_startstop_exceptions(annotations, '21')
            elif (subID in ['38', '39']) and (modality == "Electrical") and (block == 6):
                handle_startstop_exceptions(annotations, '1')
            # for subjects > 45 (electrical, except ID47) the startstop trigger was not assigned properly
            # for block 5, the startstop trigger is the same as intensity level 3..
            elif ((subID in ["46", "44", "35"]) or (subID >= "48")) and (modality == "Electrical") and (block == 5):
                handle_startstop_exceptions(annotations, '23')
                
            
            # crop the data to remove time before start and after stop of the experiment
            startstop_times = annotations[annotations["trigger_meaning"] == "startstop"]

            # handle exceptions for subject 14 with cropping the signal
            if (
                (subID == "14")
                & (modality == "Electrical")
                & (ses == os.path.join(EEG_data_dir, exception6))
            ):
                raw_filtered.crop(
                    startstop_times["total_secs"].iloc[0] - 260,
                    startstop_times["total_secs"].iloc[1] - 260,
                )
            else:
                # exceptions when only start or stop trigger was recorded
                if (
                    len(startstop_times) == 1
                ):  # only end trigger was recorded 
                    if startstop_times["total_secs"].iloc[0] > 300: 
                        raw_filtered.crop(tmax=startstop_times["total_secs"].iloc[0])
                    # in case the battery crashed we only have the starting trigger
                    elif startstop_times["total_secs"].iloc[0] < 20: 
                        raw_filtered.crop(tmin=startstop_times["total_secs"].iloc[0])
                
                # crop signal to start and end trigger
                else:
                    raw_filtered.crop(
                        startstop_times["total_secs"].iloc[0],
                        startstop_times["total_secs"].iloc[1],
                    )

            # read in the information about start and stop time to match it with the force
            # sensor data later. Note that this is only done for mechanical stimulation sessions!
            # get the session number to find the right file with the starting time
            ses_number = ses.split("NT/")[1].split(".")[0].split("session")
            ses_number = ses_number[0] + ses_number[1]
            # get the corresponding start time csv
            for start_file in all_starttimes:
                csv_number = start_file.split("session_")[1].split(".")[0]
                if csv_number == ses_number:
                    matching_file = start_file
                    break
            try:
                xml_info = pd.read_csv(matching_file)
            except:
                raise ValueError("No starting times available!")
            
            start_time = xml_info["StartDateTime"][0]
            start_time = datetime.fromisoformat(start_time)

            # add the event times to the dataframe with the scale results
            # it stores the difference of the event times to the starting time
            trigger_distance = np.array(
                [
                    timedelta(seconds=timestamp)
                    for timestamp in annotations["total_secs"]
                ]
            )
            realtime_distance = [
                start_time + trigger_delta for trigger_delta in trigger_distance
            ]
            annotations["timestamp"] = realtime_distance

            if not cnap_only:
                # 9. Extract ECG, respiration and force data (if it exists)
                ####################################################
                raw_physio = raw_downsampled.copy().resample(100)
                
                if "RESP" in raw.info["ch_names"]:
                    physio_data_df, times_samples_ecg = create_physio_df(["ECG", "RESP", "CNAP"], raw_physio)
                else:
                    physio_data_df, times_samples_ecg = create_physio_df(["ECG", "CNAP"], raw_physio)
                    physio_data_df.insert(1, "RESP", [np.nan] * len(physio_data_df))

                times_ecg = start_time + pd.to_timedelta(times_samples_ecg, unit="s")
                physio_data_df["timestamp"] = pd.to_datetime(times_ecg).tz_localize(None)
                physio_data_df.rename(columns={0: "ecg"}, inplace=True)
                physio_data_df.rename(columns={1: "resp"}, inplace=True)
                physio_data_df.rename(columns={2: "cnap"}, inplace=True)

            # Extract CNAP separately: low-pass 200 Hz, keep 500 Hz sampling
            cnap_target_sfreq = params["downsampling"]["sfreq_downsampled"]
            raw_cnap = raw.copy()
            raw_cnap.filter(l_freq=None, h_freq=200, picks=["CNAP"])
            if raw_cnap.info["sfreq"] != cnap_target_sfreq:
                raw_cnap.resample(cnap_target_sfreq)

            cnap_df, times_samples_cnap = create_physio_df(["CNAP"], raw_cnap)
            times_cnap = start_time + pd.to_timedelta(times_samples_cnap, unit="s")
            cnap_df["timestamp"] = pd.to_datetime(times_cnap).tz_localize(None)
            cnap_df.rename(columns={0: "cnap"}, inplace=True)
            cnap_df["total_secs"] = (
                cnap_df["timestamp"] - cnap_df["timestamp"].iloc[0]
            ).dt.total_seconds()

            if not cnap_only:

                # load scale data
                force_dir = os.path.join(
                    RAWDIR, f"sub-0{subID}", f"ID{subID}_{modality}", "raw_fingerforce"
                )
                file_name = f"scale_data_sub{subID}_block{block}run{run}.csv"
                scale_file_path = os.path.join(force_dir, file_name)

                # Check if the file exists
                print("Scale file:", scale_file_path)
                if os.path.exists(scale_file_path):
                    
                    scale_data = read_scale_data(scale_file_path)

                    # compute the time in seconds from the start of the EEG recording
                    # physio is downsampled, so this is not that accurate
                    annotations["timestamp"] = [
                        row.replace(tzinfo=None) for row in annotations["timestamp"]
                    ]
                    reference_time = annotations["timestamp"].iloc[0] - pd.Timedelta(seconds=annotations["total_secs"].iloc[0])
                    scale_data["total_secs"] = (
                        scale_data["timestamp"] - reference_time
                    ).dt.total_seconds()

                    # get the triggers and in the next step check if there are actually triggers
                    try:
                        trigger_scale = scale_data[scale_data["trigger_code"] == 1]
                    except:
                        trigger_scale = []
                        print("no trigger code")

                    if (subID in subswithtrigger) and (modality == "Mechanical") and (len(trigger_scale)>0):
                        
                        # add info
                        preprocessing_notes["force_trigger_exists"] = "Yes"
                        annotations["force_trigger_exists"] = "Yes"

                        # sometimes the scale sent fast triggers in the beginning right after the start
                        # those ones need to be removed, as they are not stimulation triggers (but test triggers!)
                        scale_data, trigger_scale = exclude_invalid_triggers(scale_data, trigger_scale)
                        
                        
                        # get EEG triggers
                        stim_annot = annotations[
                            annotations["trigger_meaning"] == "Stimulation"
                        ]

                        tolerance_delay = pd.Timedelta(seconds=params["trigger_matching"]["tolerance_delay_sec"])
                        
                        trigger_diff_df = merge_trigger_df(stim_annot, trigger_scale, tolerance_delay)

                        # comute the difference
                        trigger_diffs = (
                            trigger_diff_df["total_secs_x"]
                            - trigger_diff_df["total_secs_y"]
                        )
                        print("Mean trigger difference", np.mean(trigger_diffs))

                        # EXCEPTIONS: in sub34 block6 the triggers were missing, use the value of previous block for the time delay
                        if (block == "6") & (subID == "34"):
                            trigger_diffs = np.array([-0.15])
                            preprocessing_notes["force_trigger_exists"] = "No"
                            annotations["force_trigger_exists"] = "No"
                        
                        # sub 45 block 1&2: 0.3246 


                        # to check if something went wrong when checking the trigger delay
                        if np.abs(np.mean(trigger_diffs)) > 1.5:
                            raise ValueError("Check trigger matching!!")

                        # correct for time delay
                        scale_data["timestamp_corrected"] = scale_data["timestamp"] + pd.Timedelta(
                            seconds=np.mean(trigger_diffs)
                        )
                        scale_data["total_secs_corrected"] = scale_data["total_secs"] + np.mean(
                            trigger_diffs
                        )
                        trigger_scale["timestamp_corrected"] = trigger_scale[
                            "timestamp"
                        ] + pd.Timedelta(seconds=np.mean(trigger_diffs))
                        trigger_scale["total_secs_corrected"] = trigger_scale[
                            "total_secs"
                        ] + np.mean(trigger_diffs)

                        # save trigger information separately for later checks
                        trigger_scale.to_csv(
                            os.path.join(
                                session_path,
                                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_forcetrigger{file_suffix}.csv",
                            ),
                            index=False,
                        )

                        preprocessing_notes["force_mean_triggerdelay"] = np.mean(
                            trigger_diffs
                        )
                        preprocessing_notes["force_std_triggerdelay"] = np.std(
                            trigger_diffs
                        )

                    else:
                        # for recordings without triggers in the force signal we use the triggers recorded by the EEG
                        # we set the delay to 0 for now (maybe we need to correct it, when looking at the respiration delay)
                        computed_delay = 0
                        scale_data["timestamp_corrected"] = scale_data["timestamp"] + pd.Timedelta(
                            seconds=computed_delay
                        )
                        scale_data["total_secs_corrected"] = scale_data["total_secs"] + computed_delay
                        annotations["force_trigger_exists"] = "No"

                        preprocessing_notes["force_trigger_exists"] = "No"
                        preprocessing_notes["force_mean_triggerdelay"] = np.nan
                        preprocessing_notes["force_std_triggerdelay"] = np.nan

                    # retreive starting time of the resp signal
                    physio_data_df["total_secs"] = (
                        physio_data_df["timestamp"] - physio_data_df["timestamp"].iloc[0]
                    ).dt.total_seconds()

                    # add some timing info about scale measurement
                    time_diff_scale = np.diff(
                        scale_data["total_secs_corrected"]
                    )  # for keeping record of measumeremnt quality
                    preprocessing_notes["force_signal_exists"] = "Yes"
                    preprocessing_notes["force_mean_sampling"] = np.mean(time_diff_scale)
                    preprocessing_notes["force_std_sampling"] = np.std(time_diff_scale)

                    # drop duplicates
                    scale_data = scale_data.drop_duplicates(
                        subset=["total_secs_corrected"], keep="first"
                    )

                    # interpolate
                    fs = 1 / 0.01  # target sampling freq
                    max_gap = 1
                    t_irregular = scale_data["total_secs_corrected"].values
                    y_irregular = scale_data["force_value"].values
                    mask = np.isfinite(t_irregular) & np.isfinite(y_irregular)
                    t = t_irregular[mask]
                    y = y_irregular[mask]
                    t_uniform = np.linspace(
                        0,
                        physio_data_df["total_secs"].max(),
                        len(physio_data_df["total_secs"]),
                    )
                    interp_func = interpolate.interp1d(
                        t, y, kind="cubic", bounds_error=False, fill_value=np.nan
                    )
                    y_uniform = interp_func(t_uniform)

                    # correct places with large gaps
                    interval_idx = np.searchsorted(t_irregular, t_uniform, side="right") - 1
                    valid_interval = (interval_idx >= 0) & (
                        interval_idx < len(t_irregular) - 1
                    )
                    gaps = np.diff(t_irregular)
                    ok_mask = np.zeros_like(t_uniform, dtype=bool)
                    ok_mask[valid_interval] = gaps[interval_idx[valid_interval]] <= max_gap
                    y_uniform[~ok_mask] = np.nan

                    # add force values to the physio dataframe
                    physio_data_df["force"] = y_uniform
                    # compute three quantiles of the force signal for later analysis
                    t_uniform = np.linspace(
                        0, physio_data_df["total_secs"].max(), len(physio_data_df["total_secs"])
                    )
                    y_uniform_zeromean = y_uniform - np.nanmean(y_uniform)
                    physio_data_df['force_zeromean'] = y_uniform_zeromean

                    # if we have enough data points, separate in three quantiles
                    nan_ratio = np.sum(np.isnan(y_uniform_zeromean)) / len(y_uniform_zeromean)
                    if nan_ratio < 0.6:
                        physio_data_df['force_quantile'] = pd.qcut(physio_data_df['force_zeromean'], q=3, labels=['Low', 'Medium', 'High'])
                    else:
                        physio_data_df["force_quantile"] = [np.nan] * len(physio_data_df)

                else:
                    print("No force data available!")
                    # if we don't have the force data, add a column with nan's
                    physio_data_df["total_secs"] = (
                        physio_data_df["timestamp"] - physio_data_df["timestamp"].iloc[0]
                    ).dt.total_seconds()
                    physio_data_df["force"] = [np.nan] * len(physio_data_df)
                    physio_data_df["force_zeromean"] = [np.nan] * len(physio_data_df)
                    physio_data_df["force_quantile"] = [np.nan] * len(physio_data_df)

                    # add preprocessing info
                    preprocessing_notes["force_trigger_exists"] = "No"
                    preprocessing_notes["force_mean_triggerdelay"] = np.nan
                    preprocessing_notes["force_std_triggerdelay"] = np.nan
                    preprocessing_notes["force_signal_exists"] = "No"
                    preprocessing_notes["force_mean_sampling"] = np.nan
                    preprocessing_notes["force_std_sampling"] = np.nan
                    annotations["force_trigger_exists"] = "No"

                physio_data_df.to_csv(
                    os.path.join(
                        session_path,
                        f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_physio.tsv.gz",
                    ),
                    index=False,
                    header=False,
                    sep="\t",
                    compression="gzip",
                )

                # create the meta file
                physio_metadata = {
                    "SamplingFrequency": raw_physio.info["sfreq"],
                    "StartTime": start_time.isoformat(),
                    "Columns": [
                        "cardiac",
                        "respiratory",
                        "cnap",
                        "timestamp",
                        "total_secs",
                        "force",
                        "force_zeromean",
                        "force_quantile"
                    ],
                    "Manufacturer": "NeurOne",
                    "cardiac": {
                        "Description": "continuous measurements by ECG electrodes",
                        "Units": "microVolts",
                        "TermURL": "https://www.ncbi.nlm.nih.gov/mesh/68004562",
                    },
                    "respiratory": {
                        "Description": "continuous measurements by respiration belt",
                        "Units": "ARU (Arbitrary Respiratory Unit); 1 mV/ARU",
                    },
                    "cnap": {
                        "Description": "peripheral median nerve activity measured at the axilla",
                        "Units": "microVolts",
                        "TermURL": "https://www.ncbi.nlm.nih.gov/mesh/68004562",
                    },
                    "timestamp": {
                        "Description": "timestamp computed using the starting time in the yaml info file of the recording",
                        "Units": "time",
                    },
                    "total_secs": {
                        "Description": "Time that passed from the moment the recording was started",
                        "Units": "seconds",
                    },
                    "force": {
                        "Description": "weight of the finger on the stimulator",
                        "Units": "gramms",
                    },
                    "force_zeromean": {
                        "Description": "weight of the finger on the stimulator (zero meaned)",
                        "Units": "gramms",
                    },
                    "force_quantile": {
                        "Description": "force separated in three quantiles: Low, Medium, High based on zero meaned signal",
                        "Units": "gramms",
                    },
                }
                with open(
                    os.path.join(
                        session_path,
                        f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_physio.json",
                    ),
                    "w"
                    , encoding="utf-8"
                ) as f:
                    json.dump(physio_metadata, f, indent=4)

            
                # add the force value to the annotations
                annotations = pd.merge_asof(
                    annotations,
                    physio_data_df[["total_secs", "force", "force_zeromean", "force_quantile"]],
                    left_on="total_secs",
                    right_on="total_secs",
                    direction="nearest",  # or 'backward' / 'forward'
                    tolerance=0.25,  # None ⇒ take the absolute nearest
                )
                out_clean = annotations.dropna(subset=["force"])
                print("force values nr:", len(out_clean))

            # subID 17 mecha block 7 doesn't have the complete first epoch, so we need to remove
            # the annotation as well 
            exception8 = '2025-02-19T121905_session1.set'
            if ses == os.path.join(EEG_data_dir, exception8):
                annotations = annotations[4:].copy()

            # save CNAP as a separate file (1000 Hz, low-pass 200 Hz)
            cnap_df.to_csv(
                os.path.join(
                    session_path,
                    f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_cnap.tsv.gz",
                ),
                index=False,
                header=False,
                sep="\t",
                compression="gzip",
            )

            cnap_metadata = {
                "SamplingFrequency": cnap_target_sfreq,
                "StartTime": start_time.isoformat(),
                "Columns": [
                    "cnap",
                    "timestamp",
                    "total_secs",
                ],
                "Manufacturer": "NeurOne",
                "cnap": {
                    "Description": "peripheral median nerve activity measured at the axilla",
                    "Units": "microVolts",
                    "TermURL": "https://www.ncbi.nlm.nih.gov/mesh/68004562",
                },
                "timestamp": {
                    "Description": "timestamp computed using the starting time in the yaml info file of the recording",
                    "Units": "time",
                },
                "total_secs": {
                    "Description": "Time that passed from the moment the recording was started",
                    "Units": "seconds",
                },
            }
            
            if cnap_only:
                with open(
                    os.path.join(
                        session_path,
                        f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_cnap.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(cnap_metadata, f, indent=4)

            # save the annotations (trigger) information
            annotations.to_csv(
                os.path.join(
                    session_path,
                    f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_events{file_suffix}.csv",
                ),
                index=False,
            )  # processed (no data but info about trigger meanings in the eeg data/epochs)

            if cnap_only:
                print('CNAP extraction done!')
                continue

            # 10. DETECT BAD CHANNELS using RANSAC
            ###########################################
            # though, for some subjects RANSAC removes too many channels, those ones are cleaned manually (above)
            if subID not in ['13', '15', '20', '39', '25', '26', '27', '28', '29', '34', '47', '52']:
                # pick only EEG channels for Ransac bad channel detection
                picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)
                ransac = Ransac(verbose="progressbar", picks=picks, n_jobs=3)

                # which channels have been marked bad by RANSAC
                epochs = ransac.fit_transform(epochs)

                print("\nChannels".join(ransac.bad_chs_))
                print("Channels interpolated:", ransac.bad_chs_)
                preprocessing_notes["RansacInterpolated"] = f"{ransac.bad_chs_}"
                preprocessing_notes["RansacNrInterpolatedChan"] = f"{len(ransac.bad_chs_)}"
            else:
                preprocessing_notes["RansacInterpolated"] = []
                preprocessing_notes["RansacNrInterpolatedChan"] = "0"

            # 11. ICA cleaning
            #####################################################
            l_freq = params["ICA"]["l_freq"]  
            h_freq = params["ICA"]["h_freq"]
            preprocessing_notes["ICA_l_freq"] = l_freq
            preprocessing_notes["ICA_h_freq"] = h_freq

            # filtering epochs to focus on relevant frequencies
            epochs_ica = epochs.copy()
            epochs_ica.filter(l_freq, h_freq, method="iir")
            # only chose the eeg channels
            picks = mne.pick_types(epochs_ica.info, eeg=True, eog=False, ecg=False)

            # Define parameters for ICA
            # Set number of components, or a ratio. The ratio indicated how much variance the components explain in the data
            # in the case of this block the automatic ICA did not work properly due to unusual noise in the signal
            if (
                ((subID == "07") & (modality == "Mechanical") & (block == "4"))
                or ((subID == "14") & (modality == "Mechanical") & (block == "8"))
                or ((subID == "33") & (modality == "Mechanical") & (block == "3"))
            ):
                n_components = 15
            else:
                n_components = params["ICA"]["n_components"] 

            random_state = params["ICA"]["random_state"] # Seed for random state to ensure reproducibility
            method = params["ICA"]["method"]  # ICA method (can also be 'infomax', 'picard', etc.)

            # Create and fit ICA
            ica = ICA(
                n_components=n_components, random_state=random_state, method=method,
                fit_params=dict(extended=True)) # recommended settings for using ICAlabel afterwards
            ica = ica.fit(epochs_ica, picks=picks)

            # Plot ICA components
            fig_ica = ica.plot_components(show=False)

            if isinstance(fig_ica, list):
                for f, fig in enumerate(fig_ica):
                    fig.savefig(
                        os.path.join(
                            figuredir, f"sub-{subID}_modality-{ses_label}_ICA-{f}.png"
                        )
                    )
            else:
                fig_ica.savefig(
                    os.path.join(
                        figuredir, f"sub-{subID}_modality-{ses_label}_ICA-1.png"
                    )
                )

            # get the fraction of variance explained by the first component (should explain the most
            # of the variance, usually around 20-40%)
            explained_var_ratio = ica.get_explained_variance_ratio(
                epochs_ica, components=[0], ch_type="eeg"
            )
            # This time, print as percentage.
            ratio_percent = round(100 * explained_var_ratio["eeg"])
            print(f"{ratio_percent}% " f"Variance in EEG explained by first component")
            preprocessing_notes["ICA1comp_variance"] = ratio_percent

            # Finding bad compontents using the eye and the ecg electrode
            # you can use EOG/ECG channels to help identify eye and heart artifacts
            # If you have EOG channels, use the following:
            eog_idx, eog_scores = ica.find_bads_eog(epochs_ica, ch_name="VEOG")
            ecg_idx, ecg_scores = ica.find_bads_ecg(
                epochs_ica, ch_name="ECG", method="correlation"
            )
            if (subID == "07") & (modality == "Mechanical") & (block == "4"):
                ica.exclude = [
                    0,
                    1,
                    13,
                ]  # in the case of this block the automatic ICA did not work properly due to unusual noise in the signal
                print("components removed manually:", ica.exclude)
            else:
                ica.exclude = eog_idx + ecg_idx
                print("components found based on EOG:", eog_idx)
                print("components found based on ECG:", ecg_idx)

            # find more bad components using the iclabel method
            ic_labels = label_components(epochs_ica, ica, method="iclabel")
            labels = ic_labels["labels"]
            exclude_idx = [
                idx
                for idx, label in enumerate(labels)
                if label not in ["brain", "other"]
            ]
            print(f"Excluding these ICA components: {exclude_idx}")
            ica.exclude = ica.exclude + exclude_idx

            # Remove the bad components and save them
            # do so on the original data!
            ica.apply(epochs_ica)

            # store info about ICA analysis 
            preprocessing_notes["ICA_random_seed"] = random_state
            preprocessing_notes["ICA_method"] = method
            preprocessing_notes["ICA_method_findbads"] = ["correlation", "iclabel"]
            preprocessing_notes["ICA_excluded"] = ica.exclude

            # reference to average
            epochs_ica.set_eeg_reference("average", ch_type="eeg")
            preprocessing_notes["ReferenceafterICA"] = "average"

            ########### Save the clean epochs
            # save ica solution
            with Path(
                session_path,
                f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_ica{file_suffix}.pkl",
            ).open(
                "wb"
            ) as f:  # processed (ICA solutions)
                pickle.dump(ica, f)

            # 12. Save cleaned epochs
            ##########################
            epochs_ica.reset_drop_log_selection()
            epochs_test = epochs_ica.copy()
            epochs_test.reset_drop_log_selection()
            epochs_ica.save(
                os.path.join(
                    session_path,
                    f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_epochs{file_suffix}.fif",
                ),
                overwrite=True,
            )

            # 13. Bad epoch detection
            ##########################
            print("Drop log length (198!)", len(epochs_ica.drop_log))
            eeg_thresh = params["bad_epoch_detection"]["EEG_threshold"]
            eog_thresh = params["bad_epoch_detection"]["EOG_threshold"]
            reject = dict(eeg=eeg_thresh, eog=eog_thresh)  # V (EEG channels)  # V (EOG channels)
            epochs_test.reject_tmin = params["bad_epoch_detection"]["tmin"]
            epochs_ica.reject_tmax = params["bad_epoch_detection"]["tmax"]

            # Drop epochs_ica based on these parameters
            epochs_ica.drop_bad(reject=reject)

            print("Drop log length (198!)", len(epochs_ica.drop_log))
            dropped_indices = np.where([len(r) > 0 for r in epochs.drop_log])[0]
            print("Dropped epochs:", dropped_indices)
            preprocessing_notes["Nr_dropped_epochs"] = len(dropped_indices)

            # save epochs to drop
            bad_epochs_df = pd.DataFrame(dropped_indices)
            bad_epochs_df.to_csv(
                os.path.join(
                    session_path,
                    f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_badEpochs{file_suffix}.csv",
                ),
                index=False,
            )  # processed (no data but information on which epochs to drop)

            # save notes
            preprocessing_notes["date_preproc"] = datetime.today().strftime("%Y-%m-%d")
            pd.DataFrame.from_dict(data=preprocessing_notes, orient="index").to_csv(
                os.path.join(
                    session_path,
                    f"sub-0{subID}_ses-{ses_label}_task-{session_type}_run-{block}_lfreqori-{l_freq_orig}_lfreq-{l_freq}_hfreq-{h_freq}_preprosteps{file_suffix}.csv",
                )
            )  # processed (no data but documentation of each step that has been performed during preprocessing)

            print("Finished sub:", subID, modality, block)

# %%
