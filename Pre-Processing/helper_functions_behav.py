import numpy as np
import pandas as pd
import os
import scipy.io as sio
import scipy
from scipy.optimize import curve_fit

def assign_modality(modality):
    if modality=='Electrical':
        ses_label = 'elec'
    else:
        ses_label = 'mecha'
    
    return ses_label

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


# Define the logistic function for the psychometric curve
def logistic(x, L, k, x0):
    # L : curves maximum value
    # k : steepness
    # x0 : midpoint 
    return L / (1 + np.exp(-k * (x - x0)))


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
    intensities = np.unique(data['intensity'])

    for i in intensities:
        responses_intensity = data[data['intensity']==i]['response_decoded']
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

def bin_responses(row, response_format):
    if row[response_format] <= 5:
        return '0-5%'
    elif 5 < row[response_format] <= 20:
        return '5-20%'
    elif 20 < row[response_format] <= 35:
        return '20-35%'
    elif 35 < row[response_format] <= 50:
        return '35-50%'
    elif 50 < row[response_format] <= 65:
        return '50-65%'    
    elif 65 < row[response_format] <= 80:
        return '65-80%'
    elif 80 < row[response_format] <= 95:
        return '80-95%'
    else:
        return '95-100%'