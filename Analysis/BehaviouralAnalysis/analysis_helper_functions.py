import os
import re
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
import scipy
from scipy.signal import resample, correlate, correlation_lags, coherence, welch, csd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

'''
def is_valid(row):
    """Criterion for including the trials.

    """
    
    yes = row.get('Yes', 0)
    no = row.get('No', 0)
    intensity = row['intensity_level']
    
    # at least 15 trials of yes/no should be present
    if intensity == 0:
        break
''' 

def read_physio_data(physio_data_path, physio_meta_path):
    """ Read physio data (ecg, resp, force, etc.) from tsv file, as well as the headers. """
    physio_df = pd.read_csv(physio_data_path, sep="\t", header=None)

    # Load existing JSON
    with open(physio_meta_path, "r", encoding="utf-8") as f:
        physio_metadata = json.load(f)
    
    physio_df = name_physio_columns(physio_df, physio_metadata)

    return physio_df, physio_metadata

def name_physio_columns(physio_df, physio_metadata):
    """Rename columns of physio dataframe. Check if the dataframe has the correct columns/values."""
    if (physio_df.shape[1]==10) & (len(physio_metadata['Columns'])==8):
        print('add columns')
        physio_metadata['Columns'].append('resp_projection')
        physio_metadata['Columns'].append('force_without_resp')

    assert physio_df.shape[1] == len(physio_metadata['Columns']), f"Physio dataframe {physio_df.shape[1]} must have as many columns as meta file {len(physio_metadata['Columns'])}"
    assert physio_df[0].dtype == float, "First column must be float values"
    assert physio_df[1].dtype == float, "Second column must be float values"
    assert physio_df[2].dtype == float, "Third column must be float values"
    assert np.min(physio_df[4]) >= 0, "Fifth column must be all positive values"

    physio_df[3] = pd.to_datetime(physio_df[3])
    physio_df.rename(columns={0: "ecg"}, inplace=True)
    physio_df.rename(columns={1: "resp"}, inplace=True)
    physio_df.rename(columns={2: "cnap"}, inplace=True)
    physio_df.rename(columns={3: "timestamp"}, inplace=True)
    physio_df.rename(columns={4: "total_secs"}, inplace=True)
    physio_df.rename(columns={5: "force"}, inplace=True)
    physio_df.rename(columns={6: "force_zeromean"}, inplace=True)
    physio_df.rename(columns={7: "force_quantile"}, inplace=True)
    
    # data about respiration projection was already added 
    if physio_df.shape[1] == 10:
        physio_df.rename(columns={8: "resp_projection"}, inplace=True)
        physio_df.rename(columns={9: "force_without_resp"}, inplace=True)

    return physio_df

def parse_subjects():
    """
    Parse command line arguments to get subject IDs.
    If no subject IDs are provided, process all subjects from 13 to 52.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject_id",
        type=str,
        nargs="+",
        help="Subject IDs to process (e.g., '01 02 15' for subjects 1, 2, and 15)",
        required=False,
    )
    args = parser.parse_args()
    if args.subject_id:
        subIDs = args.subject_id
    else:
        # process all subjects from subject 13 (no pilot subjects)
        subIDs = map(str, range(13, 53))

    return subIDs

def check_force_availability(subjects):
    """ prints if force data / triggers were recorded """
    for sub in subjects:
        for i in range(1, 9):
            block = i
            scale_file_path = f"/data/p_02989/shared_workspace/rawdir/sub-0{sub}/ID{sub}_Mechanical/raw_fingerforce/scale_data_sub{sub}_block{block}run1.csv"
            if os.path.exists(scale_file_path):
                scale_data = pd.read_csv(scale_file_path)
                try:
                    trigger_scale_len = len(scale_data[scale_data["trigger_code"] == 1])
                    if trigger_scale_len != 198:
                        print(f"{trigger_scale_len} triggers for sub {sub} block {block}")
                except:
                    print("no trigger codes", block, sub)
            else:
                print("no force data sub", sub, 'block', block)

def plot_forcesignal(
    subID,
    stim_annot,
    physio_df,
    start_sec,
    stop_sec,
    block,
    trigger_scale=None,
    fig_dir_sub=None,
):
    """
    Parameters:
    stim_annot: dataframe with annotations about trigger times
    physio_df: dataframe with columns 'ecg, 'resp', 'cnap', 'timestamp', 'total_secs', 'force'
    start_sec: starting point of experiment in absolute seconds
    stop_sec: stopping point of experiment in absolute seconds
    
    """

    # choose a timewindow to plot
    mask = (physio_df["total_secs"] >= start_sec) & (
        physio_df["total_secs"] <= stop_sec
    )
    physio_df_window = physio_df[mask]

    # cut annotations according to the data window
    start_annot = physio_df_window["timestamp"].iloc[0]
    stop_annot = physio_df_window["timestamp"].iloc[-1]
    mask_events = (stim_annot["timestamp"] >= start_annot) & (
        stim_annot["timestamp"] <= stop_annot
    )
    annotations_window = stim_annot[mask_events]

    # Plot scale results, ecg and respiration
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12,8), sharex=True)

    ax = axs[0]
    ax.plot(
        physio_df_window["timestamp"],
        physio_df_window["force"],
        label="Force (g)",
        color="black",
    )

    # Plot vertical lines for events
    for _, row in annotations_window.iterrows():
        event = row["trigger_meaning"]
        timestamp = row["timestamp"]
        color = (
            "green"
            if event == "Stimulation"
            else "blue"
            if (event == "Yes") or (event == "No")
            else "gray"
        )
        #ax.axvline(x=timestamp, color=color, linestyle="--", label=event)

    # uncoment to plot trigger scale lines
    '''
    if trigger_scale is not None:
        mask_events = (trigger_scale["timestamp_corrected"] >= start_annot) & (
            trigger_scale["timestamp_corrected"] <= stop_annot
        )
        trigger_scale_window = trigger_scale[mask_events]
        for _, row in trigger_scale_window.iterrows():
            timestamp = row["timestamp"]
            ax.axvline(x=timestamp, color="red", linestyle="--", label='trigger scale')
            timestamp_corrected = row["timestamp_corrected"]
            ax.axvline(x=timestamp_corrected, color="pink", linestyle="--", label='trigger scale corrected')
    '''

    # Duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_ylabel("Force (grams)")
    ax.set_title("Force over time & Stimulus triggers")

    # Plot ECG signal
    ax_ecg = axs[2]
    ax_ecg.set_ylabel("V")
    ax_ecg.set_title("ECG")
    ax_ecg.plot(physio_df_window["timestamp"], physio_df_window["ecg"])

    # Plot respiration signal
    ax_resp = axs[1]
    ax_resp.set_ylabel("V")
    ax_resp.set_title("Respiration signal")
    ax_resp.plot(physio_df_window["timestamp"], physio_df_window["resp"])
    ax_resp.set_xlabel("Time")

    # plot triggers
    """for _, row in annotations_window.iterrows():
        timestamp = row["timestamp"]
        ax_ecg.axvline(x=timestamp, color="green", linestyle="--", alpha=0.7)
        ax_resp.axvline(x=timestamp, color="green", linestyle="--", alpha=0.7)"""

    plt.tight_layout()
    sns.despine(trim=True)
    #plt.show()

    if fig_dir_sub:
        fig.savefig(
            os.path.join(fig_dir_sub, f"signal_window_sub{subID}_block{block}.pdf")
        )

def compute_coherence_and_delay(force_signal, resp_signal, fs, coh_window):

    """ Compute coherence between respiration and force signals. """
    # normally compute coherence on big segment
    f_coherence, y_coherence = coherence(
        force_signal, resp_signal, fs, nperseg=coh_window
    )

    # compute the cross-spectral density of two signals
    f_csd, Pxy = csd(force_signal, resp_signal, fs, nperseg=coh_window)

    # compute the angle, which is the phase from the cross-spectral density
    phase = np.angle(Pxy)

    # mask to only look at frequencies up to 2 Hz for plotting
    mask = (f_coherence <= 2) & (f_coherence > 0)
    f_coherence = f_coherence[mask]
    y_coherence = y_coherence[mask]
    phase = phase[mask]

    # we select a timewindow of 0.2 to 0.35 Hz
    # which is the normal respiration frequency range
    freq_range_resp = (f_coherence >= 0.2) & (f_coherence <= 0.35)
    f_coherence_resp = f_coherence[freq_range_resp]
    y_coherence_resp = y_coherence[freq_range_resp]
    phase_resp = phase[freq_range_resp]

    # get the phase of the frequency with the maximum coherence
    idx_max = np.argmax(y_coherence_resp)
    phase_max = phase_resp[idx_max]
    freq_max = f_coherence_resp[idx_max]

    # compute delay
    delay = -phase_max / (2 * np.pi * freq_max) if freq_max != 0 else 0

    # max coherence value
    max_coh = y_coherence_resp[idx_max]

    return f_coherence, y_coherence, delay, freq_max, max_coh


def plot_power_spectrum(ax, frequency, signal, title, ylabel="Power", xlabel="Frequency (Hz)"):
    """
    Plots the power spectrum for a single signal on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        frequency (array-like): The frequency data.
        signal (array-like): The signal data.
        title (str): The plot title.
        ylabel (str): Label for the Y axis ("Power" by default).
        xlabel (str): Label for the X axis ("Frequency (Hz)" by default).
    """
    ax.plot(frequency, np.abs(signal))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# Define the logistic function for the psychometric curve
def logistic(x, L, k, x0):
    """
    L : curves maximum value
    k : steepness
    x0 : midpoint
    """
    return L / (1 + np.exp(-k * (x - x0)))#

def sigmoid(x, k, x0):
    """
    Sigmoid function for fitting psychometric curves.
    Parameters:
    x : intensity values
    k : steepness
    x0 : midpoint
    maximum is always 100 (response freq)
    """
    return 100 / (1 + np.exp(-k * (x - x0)))

def linear(x, a, b):
    '''
    a: steepness
    b: shift on y-axis
    '''
    return a * x + b    

def fit_curves(intensities_block, proportions_block, initial_guess_sigm):
    ''' Fit linear and sigmoid curve and save results in new dataframe'''

    fitting_df = pd.DataFrame() 

    # linear fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(intensities_block, proportions_block)

    # sigmoidal fit 
    params, covariance = curve_fit(sigmoid, intensities_block, proportions_block, p0=initial_guess_sigm, maxfev=100000)
    # Generate smooth curve for the fitted function
    x_fit = np.linspace(np.min(intensities_block), np.max(intensities_block), 7)
    y_fit = sigmoid(x_fit, *params)
    
    # get the R2
    ss_res = np.sum((proportions_block - y_fit) ** 2)
    ss_tot = np.sum((proportions_block - np.mean(proportions_block)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return  [r_squared, params[0], params[1], r_value**2, slope, intercept, x_fit, y_fit]

def response_proportions(df):
    '''
    Calculate the Yes/No proportions for a given intensity

    Parameter:
    df - dataFrame with intensities and responses
    Return:
    proportions - numpy array with yes/no response proportion
    intensities - numpy array with intensities
    '''
    signal_detection = []
    intensities = np.unique(df['intensity'])
    df_noCatch = df[df['intensity'] > 0]

    # Compute true negatives and false positives
    false_positive = df[(df['intensity']==0) & (df['response_decoded']=='Yes')]
    true_negative = df[(df['intensity']==0) & (df['response_decoded']=='No')]

    # Calculate the response proportion
    for i in intensities:
        responses_df = df[df['intensity']==i]
        responses_intensity = responses_df
        amount_responses = len(responses_intensity['response_decoded'])
        false_negative = responses_intensity[responses_intensity['response_decoded']=='No']
        true_positive = responses_intensity[responses_intensity['response_decoded']=='Yes']
        
        # Count occurrences of 'yes' and 'no'
        count_yes = np.sum(responses_intensity['response_decoded'] == 'Yes')
        count_no = np.sum(responses_intensity['response_decoded'] == 'No')

        print(i)
        # Calculate the proportion, handling division by zero
        if count_no == 0:
            proportion = 1  # Handle the case when there are no 'no' values
        elif count_yes == 0:
            proportion = 0  # Handle the case when there are no 'yes' values
        else:
            proportion = count_yes / (count_no+count_yes)
        data = [i, proportion, len(true_positive),
                                 len(false_negative), len(true_negative),
                                 len(false_positive), count_yes, count_no,
                                 amount_responses]
        print('data', data)
        signal_detection.append([i, proportion, len(true_positive),
                                 len(false_negative), len(true_negative),
                                 len(false_positive), count_yes, count_no,
                                 amount_responses])

    signal_detection_df = pd.DataFrame(signal_detection, columns=['intensity', 'proportion', 'TP',
                                                                  'FN', 'TN', 'FP', 'count_yes',
                                                                  'count_no', 'total_responses'])

    return signal_detection_df


def calc_response_prop(data):
    """
    parameter: 
    data - data frame with intensities per trial and decoded responses
    
    returns:
    curve_intensities - array of intensities for plotting the psychometric curve
    proportions - response proportions
    """
    proportions = []
    curve_intensities = []
    intensities = np.unique(data['intensity_level'])

    for i in intensities:
        responses_intensity = data[data['intensity_level']==i]['response_decoded']
        # Count occurrences of 'yes' and 'no'
        count_yes = np.sum(responses_intensity == 'Yes')
        count_no = np.sum(responses_intensity == 'No')

        # the intensity should no be empty
        if len(responses_intensity) > 0:
            # Calculate the proportion, handling division by zero
            if count_no == 0:
                proportion = 1  # Handle the case when there are no 'no' values
            elif count_yes == 0:
                proportion = 0  # Handle the case when there are no 'yes' values
            else:
                proportion = count_yes / (count_no+count_yes)

            proportions.append(proportion)
            curve_intensities.append(i)
        
    return np.array(curve_intensities), np.array(proportions)

def get_data_singlesub(file, pattern, filetype):
    '''
    Takes in file, subject and pattern. Extracts the block and the run
    Returns block, run and the datasets
    '''

    filename = os.path.basename(file)
    match = pattern.search(filename)
    if match:
        print('MATCH')
        block = match.group(2)
        run = match.group(3)

        # load the file data 
        if filetype == 'TA':
            data_p_data = sio.loadmat(file, squeeze_me=True)['p_data']
            data_setup = sio.loadmat(file, squeeze_me=True)['thr1F']
            data_exp_data = sio.loadmat(file, squeeze_me=True)['thr1F_data']
        else: 
            data_p_data = sio.loadmat(file, squeeze_me=True)['p_data']
            data_setup = sio.loadmat(file, squeeze_me=True)['s']
            data_exp_data = sio.loadmat(file, squeeze_me=True)['exp_data']
    else:
        print('NO MATCH:', filename)
        block, run, data_exp_data, data_setup, data_p_data = [None, None, None, None, None]

    return block, run, data_exp_data, data_setup, data_p_data

def get_detection_statistics(df):
    '''
    Calculate the Yes/No proportions for a given intensity

    Parameter:
    df - dataFrame with intensities and responses
    Return:
    proportions - numpy array with yes/no response proportion
    intensities - numpy array with intensities
    '''
    signal_detection = []
    subjects = np.unique(df['sub'])

    # Calculate the response proportion
    for sub in subjects:
        df_sub = df[df['sub']==sub]
        blocks = np.unique(df['block'])
        for block in blocks: 
            df_block = df_sub[df_sub['block']==block]
            runs = np.unique(df_block['run'])
            for run in runs:
                df_run = df_block[df_block['run']==run]
                stims = np.unique(df_block['stim_type'])
                for stim_type in stims:
                    df_stim = df_run[df_run['stim_type']==stim_type]
                    print('sub', sub, stim_type)

                    # Compute true negatives and false positives
                    false_positive = df_stim[(df_stim['intensity']==0) & (df_stim['response_decoded']=='Yes')]
                    true_negative = df_stim[(df_stim['intensity']==0) & (df_stim['response_decoded']=='No')]
                    
                    intensities = np.unique(df_stim['intensity'])
                    for i in intensities:
                        print(i)
                        responses_df = df_stim[df_stim['intensity']==i]
                        i_type = np.unique(responses_df['intensity_level'])[0]
                        amount_responses = len(responses_df['response_decoded'])
                        false_negative = responses_df[responses_df['response_decoded']=='No']
                        true_positive = responses_df[responses_df['response_decoded']=='Yes']
                        
                        # Count occurrences of 'yes' and 'no'
                        count_yes = np.sum(responses_df['response_decoded'] == 'Yes')
                        count_no = np.sum(responses_df['response_decoded'] == 'No')

                        # Calculate the proportion, handling division by zero
                        if count_no == 0:
                            proportion = 1  # Handle the case when there are no 'no' values
                        elif count_yes == 0:
                            proportion = 0  # Handle the case when there are no 'yes' values
                        else:
                            proportion = count_yes / (count_no+count_yes)

                        signal_detection.append([sub, stim_type, block, run, i, i_type, proportion,
                                                len(true_positive), len(false_negative), len(true_negative),
                                                len(false_positive), count_yes, count_no, amount_responses])

    signal_detection_df = pd.DataFrame(signal_detection, columns=['sub', 'stim_type', 'block', 'run', 'intensity', 'intensity_level',
                                                                  'proportion', 'TP', 'FN', 'TN', 'FP', 'count_yes',
                                                                  'count_no', 'total_responses'])

    return signal_detection_df

def annotate_fig(dataname):
    """ Write on the figure with which data it was generated, the date and the script name."""
    plt.annotate(dataname, xy=(-2, 0.3), xycoords='axes fraction', fontsize=8, ha='center')
    plt.annotate(datetime.datetime.now(), xy=(-2, 0.2), xycoords='axes fraction', fontsize=8, ha='center')
    plt.annotate(f"generated in {os.path.basename(__file__)}", xy=(-2, 0.1), xycoords='axes fraction', fontsize=8, ha='center')

