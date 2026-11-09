import seaborn as sns
import matplotlib as plt
import numpy as np
import tkinter as tk

def figure_style():
    """
    Set style for plotting figures.
    Modified from Guido Meijer (@guidomeijer in 2024)
    """
    sns.set_theme(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 12,
                "figure.titlesize": 18,
                "axes.titlesize": 16,
                "axes.labelsize": 20,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 18,
                'legend.title_fontsize': 18,
                'legend.frameon': False,
                 })
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


    subject_pal = sns.color_palette(
        np.concatenate((sns.color_palette('tab20'),
                        [plt.colors.to_rgb('maroon'), np.array([0, 0, 0])])))
    
    colors = {'perceived' : sns.color_palette("colorblind")[1],
              'unperceived' : sns.color_palette("colorblind")[0],
              'subject_palette': subject_pal,
              'grey': [0.7, 0.7, 0.7],
              'wt': [0.7, 0.7, 0.7],
              'mechanical': sns.color_palette('colorblind')[3],
              'electrical': sns.color_palette('colorblind')[4]}
    screen_width = 1920 #tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi