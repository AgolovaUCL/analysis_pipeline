
import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filters
import warnings
from astropy.stats import circmean
from astropy.convolution import convolve, convolve_fft
from skimage.morphology import disk
import random
import os

def make_roseplots(derivatives_base, rawsession_folder, trials_to_include, deg: int, path_to_df = None):
    """
    This code creates the roseplots as visualized for the spatiotemporal analysis. 
    For the location of the arms (N, NE, etc.) is assumes we're using the new camera (the one with 25 frame rate and north on top)

    Input: 
    derivatives_base; path to derivatives folder
    rawsession_folder: path to rawdata folder
    trials_to_include: trials to include in the analysis
    deg: degree that the data was binned into
    path_to_df: path to df containing MRl data (optional)
    """

    # df path
    if path_to_df is not None:
        df_path = pd.read_csv(path_to_df)
    else:
        df_path_base = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data', 'directional_tuning')
        df_options = glob.glob(os.path.join(df_path_base, "directional_tuning*.csv"))
        if len(df_options) == 1:
            df_path = df_options[0]
        else:
            print(df_options)
            user_input = input('Please provide the number of the file in the list you would like to look at (starting at 1): ')
            user_input = np.int32(user_input)
            df_path = df_options[user_input - 1]
    print(f"Making roseplot from data from {df_path}")
    df_all = pd.read_csv(df_path)
    df = df_all[df_all['significant'] == 'sig']

    # Dataframe with raised arms
    csv_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.csv'))

    if len(csv_path) > 0:
        behaviour_df = pd.read_csv(csv_path[0], header=None)
    else:
        excel_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.xlsx'))
        if len(excel_path) > 0:
            behaviour_df = pd.read_excel(excel_path[0], header=None)
        else:
            csv_path = glob.glob(os.path.join(rawsession_folder, 'behaviour*.csv'))
            if len(csv_path) > 0:
                behaviour_df = pd.read_csv(csv_path[0], header=None)
            else:
                excel_path = glob.glob(os.path.join(rawsession_folder, 'behaviour*.xlsx'))
                if len(excel_path) > 0:
                    behaviour_df  = pd.read_excel(excel_path[0], header=None)
                else:
                    raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')
        
    # Direction of arms and their angles
    arms_dir = ["N", "NW", "SW", "S", "SE", "NE"]
    arms_angles_start = [30, 90, 150, 210, 270, 330]

    # Output path: 
    output_path_plot = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'roseplots', f'roseplot_{deg}_degrees.png')
 
    # Plot: 3 columns for epochs, final column for correct/incorrect
    fig, axs = plt.subplots(len(trials_to_include), 4, figsize = [3*4, 4*len(trials_to_include)], subplot_kw = {'projection': 'polar'})
    num_bins = 24


    for tr in trials_to_include:
        for e in np.arange(1,4):
            num_spikes_arr = []
            mean_dir_arr = []
            sum_count_bin = []

            # filter for this trial and epoch
            filtered_df = df[(df['trial'] == tr) & (df['epoch'] == e)]

            # if any cells were significant this trial epoch
            if len(filtered_df) > 0:
                mean_dir_arr = np.array(filtered_df['mean_direction'])
                num_spikes_arr = np.array(filtered_df['num_spikes'])

                # Binning the data
                counts, bin_edges = np.histogram(mean_dir_arr, bins = num_bins, range = (-180, 180))

                # Finding the bin of each element in the mean dir arr
                bin_idx = np.digitize(mean_dir_arr, bin_edges) - 1 

                # Count the number of spikes for each element in bin
                for i in range(len(bin_edges) - 1):
                    indices = np.where(bin_idx == i)
                    num_spikes_i = num_spikes_arr[indices]
                    sum_count_bin.append(np.sum(num_spikes_i))

                # finding bin centers for plotting
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_centers = np.deg2rad(bin_centers)
                width = np.diff(bin_centers)[0]

                # bar length = sum of number of spikes
                axs[tr-1, e-1].bar(
                    bin_centers,
                    sum_count_bin,
                    width=width,
                    bottom=0.0,
                    alpha=0.8,
                    zorder = 2
                )

                axs[tr-1, e-1].text(
                    np.pi/3,                # angle in radians
                    1.25* np.nanmax(sum_count_bin),         # radius (just outside the bar)
                    f"n = {len(filtered_df)}",   # label text
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation_mode='anchor',
                    color = 'r',
                )

                # Overlay the arm choices
                if e > 1:
                    arm = behaviour_df.iloc[tr-1, 1]
                    index = np.where(np.array(arms_dir) == arm)[0][0]
                    angle_start = arms_angles_start[index]
                    theta = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_start + 60), 100) 
                    r = np.ones_like(theta) * np.nanmax(sum_count_bin)
                    axs[tr-1, e-1].fill_between(theta, 0, r, color='lightgreen', alpha=0.5, zorder=0)

                if e > 2:
                    arm = behaviour_df.iloc[tr-1, 2]
                    index = np.where(np.array(arms_dir) == arm)[0][0]
                    angle_start = arms_angles_start[index]
                    theta = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_start + 60), 100) 
                    r = np.ones_like(theta) * np.nanmax(sum_count_bin)
                    axs[tr-1, e-1].fill_between(theta, 0, r, color='pink', alpha=0.5, zorder=0)

            
            axs[tr-1, e-1].set_title(f" Tr {tr} epoch {e}")

        if behaviour_df.iloc[tr-1, 3] == "Y":
            text = "Correct"
            c = 'g'
        else:
            text = "Incorrect"
            c = 'r'

        axs[tr-1, 3].remove() 
        axs[tr-1, 3] = fig.add_subplot(len(trials_to_include), 5, (tr-1)*5 + 5) 
        axs[tr-1, 3].axis('off') 
        axs[tr-1, 3].text(0.0, 0.5, text, fontsize=11, va='center', ha='left', wrap=True, c= c)


    plt.tight_layout()
    plt.savefig(output_path_plot)
    plt.show()




