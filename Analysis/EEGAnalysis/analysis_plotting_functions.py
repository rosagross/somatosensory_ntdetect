import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from plotting_style import figure_style
colors, _ = figure_style()

RESDIR = os.getenv('RESDIR')
figuredir = os.path.join(RESDIR, 'Figures', 'Main', 'eeg_results')
if not os.path.exists(figuredir):
    os.makedirs(figuredir)


def plot_single_ERP(time_steps, perceived_mean, unperceived_mean, subID, channel, modality):
    """ plot single subject ERP """
    plt.figure(figsize=(20, 5))
    plt.plot(time_steps, perceived_mean, color=colors['perceived'], label='all trials')
    #plt.plot(time_steps, unperceived_mean, color=colors['unperceived'], label='Unperceived')
    plt.title(f'Subject {subID} - Channel {channel} - modality {modality}')
    plt.xticks(np.arange(-0.4, 0.2, 0.05))
    plt.legend()
    sns.despine(trim=True)
    plt.savefig(os.path.join(figuredir, 'single', f'Subject_{subID}_Channel_{channel}_{modality}_ERP.png'))


def plot_grand_ERP(all_unperceived, all_perceived, time_steps, tmin, tmax, modality, channel):
    """ plot grand average ERP across subjects """

    modality_label = 'Electrical' if modality == 'elec' else 'Mechanical'

    timewindow = (time_steps >= tmin) & (time_steps <= tmax)
    time_steps = time_steps[timewindow]
    all_unperceived = all_unperceived[timewindow, :]
    all_perceived = all_perceived[timewindow, :]    

    plt.figure()
    mean_vals_unperceived = np.mean(all_unperceived, axis=1)  
    std_vals_unperceived = np.std(all_unperceived, axis=1)    
    mean_vals_perceived = np.mean(all_perceived, axis=1)  
    std_vals_perceived = np.std(all_perceived, axis=1)    
    sem_vals_unperceived = sem(all_unperceived, axis=1)    
    sem_vals_perceived = sem(all_perceived, axis=1)    
    n_perceived = all_perceived.shape[1]
    ci95_perceived = 1.96 * std_vals_perceived / np.sqrt(n_perceived)
    n_unperceived = all_unperceived.shape[1]
    ci95_unperceived = 1.96 * std_vals_unperceived / np.sqrt(n_unperceived)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=time_steps, y=mean_vals_perceived, label='Perceived', color=colors['perceived'])
    sns.lineplot(x=time_steps, y=mean_vals_unperceived, label='Unperceived', color=colors['unperceived'])
    #plt.fill_between(time_steps, mean_vals_perceived - std_vals_perceived, mean_vals_perceived + std_vals_perceived, alpha=0.3, label='Std Dev', color=colors['perceived'])
    #plt.fill_between(time_steps, mean_vals_unperceived - std_vals_unperceived, mean_vals_unperceived + std_vals_unperceived, alpha=0.3, label='Std Dev', color=colors['unperceived'])
    plt.fill_between(time_steps, mean_vals_unperceived - ci95_unperceived, mean_vals_unperceived + ci95_unperceived, alpha=0.3, label='', color=colors['unperceived'])
    plt.fill_between(time_steps, mean_vals_perceived - ci95_perceived, mean_vals_perceived + ci95_perceived, alpha=0.3, label='', color=colors['perceived'])
    # add vertical line at 0s
    if time_steps[0] < -0.4:
        plt.axvline(x=-0.4, color='red', linestyle='--')
    if time_steps[0] < 0:
        plt.axvline(x=0, color='black', linestyle='--')
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('µV')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'{channel} - {modality_label} \ntrials: Perceived n={n_perceived}, Unperceived n={n_unperceived}', fontsize=14)
    sns.despine(trim=True)
    plt.savefig(os.path.join(figuredir, f'Grand_Average_ERP_channel-{channel}-{modality}.pdf'))
    plt.savefig(os.path.join(figuredir, f'Grand_Average_ERP_channel-{channel}-{modality}.png'), dpi=300)
