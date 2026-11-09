"""
File: unit_tests_preproc.py
Author: Rosa Grossmann
Contact: grossmannr@cbs.mpg.de
Date: 2025-10-06
Description: Unit tests for pre-processing scripts

What to test:
- 

"""
import pytest
import os
import numpy as np
import pandas as pd
from helper_functions_eeg import read_scale_data, exclude_invalid_triggers, merge_trigger_df

# test cases:
# 1. same number of triggers in force recording and epochs: subject 47
# 2. 
def test_force_values():
    """In case we only have part of the force data (e.g., started too late, etc.),
    check if the force values are correctly assigned to the dataframe."""

def test_exclude_invalid_triggers():
    """Check if invalid triggers are correctly excluded from the scale data."""
    # test cases are (scale_data, trigger_scale, expected)
    case1_path = '/data/p_02989/shared_workspace/rawdir/sub-047/ID47_Mechanical/raw_fingerforce/scale_data_sub47_block8run1.csv'
    scale_data1 = read_scale_data(case1_path)
    trigger_scale1 = scale_data1[scale_data1["trigger_code"] == 1]
    scale_data1_expected = scale_data1
    trigger_scale1_expected = trigger_scale1

    cases = [
        (scale_data1, trigger_scale1, scale_data1_expected, trigger_scale1_expected),  # expected to exclude first 10 triggers
    ]
    for scale_data, trigger_scale, expected_scale, expected_trigger in cases:
        result_scale, result_trigger = exclude_invalid_triggers(scale_data, trigger_scale)
        assert result_scale.equals(expected_scale), f"Expected {expected_scale}, but got {result_scale}"
        assert result_trigger.equals(expected_trigger), f"Expected {expected_trigger}, but got {result_trigger}"


def test_merge_trigger_df():
    """Check if the trigger info from scale and EEG are aligned correctly."""

    cases = [
        ('sub49_block7_mecha', 1),
        ('sub33_block1_mecha', 1)
    ]
    print(os.getcwd())

    for case in cases:
        stim_annot = pd.read_csv(f'Pre-Processing/TestData/test_{case[0]}_merge_trigger_df_stimannot.csv')
        stim_annot['timestamp'] = pd.to_datetime(stim_annot['timestamp'])
        trigger_scale =pd.read_csv(f'Pre-Processing/TestData/test_{case[0]}_merge_trigger_df_triggerscale.csv')
        trigger_scale['timestamp'] = pd.to_datetime(trigger_scale['timestamp'])
        expected = pd.read_csv(f'Pre-Processing/TestData/test_{case[0]}_merge_trigger_df_expected.csv')
        expected['timestamp'] = pd.to_datetime(expected['timestamp'])
        tolerance_delay = pd.Timedelta(seconds=case[1])
        result = merge_trigger_df(stim_annot, trigger_scale, tolerance_delay)

        # reset indexing for comparison
        result = result.reset_index(drop=True)
        expected = expected.reset_index(drop=True)

        assert result.equals(expected), f"Expected {expected}, but got {result}"

"""
def test_merge_trigger_df():
    case1_path_annot = '/data/p_02989/shared_workspace/datadir/sub-047/ses-mecha/eeg/sub-047_ses-mecha_task-NT_run-8_events.csv'
    annot1 = pd.read_csv(case1_path_annot)  
    annot1['timestamp'] = pd.to_datetime(annot1['timestamp'])
    case1_path_scale = '/data/p_02989/shared_workspace/rawdir/sub-047/ID47_Mechanical/raw_fingerforce/scale_data_sub47_block8run1.csv'
    scale_data1 = read_scale_data(case1_path_scale)
    reference_time = annot1["timestamp"].iloc[0] - pd.Timedelta(seconds=annot1["total_secs"].iloc[0])
    scale_data1["total_secs"] = (
        scale_data1["timestamp"] - reference_time
    ).dt.total_seconds()
    stim_annot1 = annot1[
                        annot1["trigger_meaning"] == "Stimulation"
                    ]
    
    trigger_scale1 = scale_data1[scale_data1["trigger_code"] == 1]
    expected_len1 = 198
    time_diff_min1 = -0.194
    time_diff_max1 = -0.138

    cases = [
        (stim_annot1, trigger_scale1, pd.Timedelta(seconds=1), expected_len1, time_diff_min1, time_diff_max1)
        # Add test cases here
    ]

    for case in cases:
        stim_annot, trigger_scale, tolerance_delay, expected_len, time_diff_min, time_diff_max = case
        result = merge_trigger_df(stim_annot, trigger_scale, tolerance_delay)

        # Test properties instead of exact values
        assert len(result) == expected_len, f"Expected 198 rows, got {len(result)}"
        assert list(result.columns) == ['total_secs_x', 'timestamp', 'total_secs_y'], "Column names mismatch"
        assert result['total_secs_x'].notna().all(), "total_secs_x has null values"
        assert result['total_secs_y'].notna().all(), "total_secs_y has null values"
        
        # Check time differences are within tolerance
        time_diff = (result['total_secs_x'] - result['total_secs_y'])
        assert time_diff.max() < time_diff_max, f"Max time difference {time_diff.max()} exceeds threshold"
        assert time_diff.min() > time_diff_min, f"Min time difference {time_diff.min()} exceeds threshold"

"""


def test_reject_chan_psd():
    "check if the raw object is actually changed"




def test_placeholder():
    assert True