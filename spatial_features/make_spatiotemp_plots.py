import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from get_sig_cells import get_sig_cells
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import warnings
from astropy.stats import circmean

def make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 30, sample_rate = 30000):
    """
    Function to create plots for the spatiotemporal experiment
    Functions to make rate maps and circular plots were adapted from Cristina's code

    Input:
    derivatives base - path to derivatives folder for this session
    rawsession folder - path to raw session data
    trials_to_include - array with trial number we will use
    frame_rate - frame rate used by camera
    sample_rate - sample rate used by neuropixel recording

    Ouput:
    Plots
    Table with significant cells
    """
    # Parameters
    n_trials = len(trials_to_include)
    n_epochs = 3
    n_rows = n_trials
    n_cols = n_epochs * 2 + 1
    min_spikes = 5 # minimum number of spikes for a cell to be deemed significant
    directional_data_all_units = pd.DataFrame(
    columns=[
        'cell', 'trial', 'epoch', 'MRL', 'mean_direction',
        'percentiles95', 'percentiles99', 'significant', 'num_spikes'
    ]
    )

    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')
    good_units_ids = [el for el in unit_ids if labels[el] == 'good']

    # Get directory for the positional data
    pos_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    if not os.path.exists(pos_data_dir):
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")
    
    # Get the data with the epoch times
    csv_path = os.path.join(rawsession_folder, 'task_metadata', 'timestamps.csv')
    xlsx_path = os.path.join(rawsession_folder, 'task_metadata', 'timestamps.xlsx')
    if os.path.exists(csv_path):
        epoch_times = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        epoch_times = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"Epoch times file does not exist: {csv_path} or {xlsx_path}")
    
    # loading dataframe with unit information
    path_to_df = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "all_units_overview","unit_metrics.csv")
    df_unit_metrics = pd.read_csv(path_to_df) 

    bin_edges = np.linspace(0,360,73)


    # Tables for statistics
    # to add later
    trial_dur_so_far = 0

    for unit_id in tqdm(unit_ids):
        # Loading data
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data

        # Unit information
        row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]

        unit_firing_rate = row['firing_rate'].values[0]
        unit_label = row['label'].values[0]

        unit_label = "TEST"
        # Creating figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize = [3*n_cols, 3*n_rows])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}. Label: {unit_label}. Firing rate = {unit_firing_rate:.1f} Hz", fontsize = 18)
        counter = 0 # counts which axs we're on
        
        # Looping over trials
        for tr in trials_to_include:
            # Trial times
            trial_row = epoch_times[(epoch_times.trialnumber == tr)]
            start_time = trial_row.iloc[0, 1]
        
            # Positional data
            trial_csv_name = f'XY_HD_t{tr}.csv'
            trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
            xy_hd_trial = pd.read_csv(trial_csv_path)            
            xy = xy_hd_trial.iloc[:, :2].to_numpy().T
            x = xy_hd_trial.iloc[:, 0].to_numpy()
            y = xy_hd_trial.iloc[:, 1].to_numpy()
            hd = xy_hd_trial.iloc[:, 2].to_numpy()
            sync = np.arange(len(x))
            
            # Length of trial
            spike_train =  [el for el in spike_train if el > np.round(trial_dur_so_far+ start_time)] # filtering for current trial
            spike_train = [el - trial_dur_so_far*frame_rate for el in spike_train] # shifting time spikes so they 0 is the start time of the trial
            spike_train = [el for el in spike_train if el < len(x)]
            
            # Make plots
            firing_rate_map, x_bins, y_bins, location_occupancy = spatialheatmap(xy, spike_train, sync, frame_rate, bin_size=2)
            plotspatialheatmap(firing_rate_map, x_bins, y_bins, ax = axs[counter])
            counter += 1

            for e in range(1, n_epochs + 1):
                # Obtain epoch start and end times
                epoch_start = trial_row.iloc[0, 2*e-1]
                epoch_end = trial_row.iloc[0, 2*e]

                spike_train_this_epoch = [np.int32(el) for el in spike_train if el > frame_rate*epoch_start and el < frame_rate *epoch_end]
                spike_train_this_epoch = np.asarray(spike_train_this_epoch, dtype=int)


                # spike plot
                x_until_now = x[:np.int32(epoch_end*frame_rate)]
                y_until_now = y[:np.int32(epoch_end*frame_rate)]
                axs[counter].scatter(x_until_now, y_until_now, color = 'black')
                axs[counter].scatter(x[spike_train_this_epoch], y[spike_train_this_epoch], color = 'r')
                axs[counter].set_aspect('equal', adjustable='box')
                counter += 1

                # HD calculations
                hd_this_epoch = hd[np.int32(epoch_start*frame_rate):np.int32(epoch_end*frame_rate)]
                occupancy_counts, _ = np.histogram(hd_this_epoch, bins=bin_edges)
                occupancy_time = occupancy_counts / frame_rate
                hd_this_epoch_rad = np.deg2rad(hd_this_epoch)
                counts, _ = np.histogram(hd_this_epoch, bins=bin_edges)
                bin_idx = np.digitize(hd_this_epoch, bin_edges) - 1  # zero-based index for Python

                direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, np.nan, dtype=float), where=occupancy_time!=0)
                bin_idx = np.clip(bin_idx, 0, len(direction_firing_rate) - 1)

                # Weigths for the spikes
                W = direction_firing_rate[bin_idx]

                # 6. Circular statistics
                MRL = resultant_vector_length(hd_this_epoch_rad, w=W)
                mu = circmean(hd_this_epoch_rad, weights=W)
                mean_angle_deg = np.rad2deg(mu)

                # Plot
                sync = np.arange(len(hd_this_epoch))
                fig.delaxes(axs[counter])
                axs[counter] = fig.add_subplot(n_rows, n_cols, counter+1, polar=True)
                headdir_bin_centers, firing_rate, headdir_occupancy = calculate_firing_rate_by_headdir(hd_this_epoch, spike_train_this_epoch, sync)
                headdir_bin_centers_rad = np.deg2rad(headdir_bin_centers)
                width = 2 * np.pi / len(headdir_bin_centers_rad)  # width of each bar (in radians)

                headdir_bin_centers_rad = np.deg2rad(headdir_bin_centers)   # Your bin centers, in radians

                width = np.diff(headdir_bin_centers_rad)[0]
                # If using 24 bins (e.g., 15 degree bins), this will be 2*pi/24
                firing_rate[np.isnan(firing_rate)] = 0

                bars = axs[counter].bar(
                    headdir_bin_centers_rad,
                    firing_rate,
                    width=width,
                    bottom=0.0,
                    alpha=0.8
                )
                axs[counter].set_theta_zero_location('N')
                axs[counter].set_theta_direction(-1)
                counter += 1

                # Get cell significance
                """
                percentiles_95_value, percentiles_99_value = get_sig_cells(spike_train_this_epoch, hd, epoch_end - epoch_start)
                # Adding significance data
                percentiles_95_value, percentiles_99_value = get_sig_cells(spike_train_this_epoch, hd, epoch_end - epoch_start)

                # Add significance data for every element (even if not significant)
                new_element = {
                    'cell': unit_id,
                    'trial': tr,
                    'epoch': e,
                    'MRL': MRL,
                    'mean_direction': mu,
                    'percentiles95': percentiles_95_value,
                    'percentiles99': percentiles_99_value,
                    'significant': 'ns', 
                    'num_spikes': len(spike_train_this_epoch)
                }

                if MRL > percentiles_95_value:
                    new_element['significant'] = 'sig'
    
                directional_data_all_units.loc[len(directional_data_all_units)] = new_element
                """
                                
        

        output_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'task_overview')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"unit_{unit_id}_spatiotemp.png")
        plt.savefig(output_path)
        plt.show()
        breakpoint()
        plt.close(fig)
    
    directional_data_sig_units = directional_data_all_units[directional_data_all_units['significant'] == 'sig']
    output_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'data_files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'directional_data_all_units.csv')
    directional_data_all_units.to_csv(output_path, index = False)

    output_path = os.path.join(output_dir, 'directional_data_sig_units.csv')
    directional_data_sig_units.to_csv(output_path, index = False)


def spatialheatmap(xycoords, spikes, sync, frame_rate = 30, bin_size=2, sigma=1):
    # Define the spatial bins
    x_bins = np.arange(np.nanmin(xycoords[0]), np.nanmax(xycoords[0]) + bin_size, bin_size)
    y_bins = np.arange(np.nanmin(xycoords[1]), np.nanmax(xycoords[1]) + bin_size, bin_size)

    # Calculate the occupancy map
    occupancy, _, _ = np.histogram2d(xycoords[0], xycoords[1], bins=[x_bins, y_bins])

    # Calculate the spike map
    spike_x = np.interp(spikes, sync, xycoords[0])
    spike_y = np.interp(spikes, sync, xycoords[1])
    spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_bins, y_bins])

    # Normalize the spike map by the occupancy map to get the firing rate map
    # Avoid division by zero by setting occupancy bins with zero time to NaN
    firing_rate_map = np.divide(spike_map, occupancy, where=occupancy != 0)*frame_rate


    firing_rate_map = gaussian_filter(firing_rate_map, sigma=sigma, mode='constant', cval=np.nan)
    firing_rate_map[occupancy == 0] = np.nan
    return firing_rate_map, x_bins, y_bins, occupancy

def plotspatialheatmap(firing_rate_map, x_bins, y_bins, neuronname=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    cb = ax.imshow(
        firing_rate_map.T,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        interpolation='none',
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
    )
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    if neuronname:
        ax.set_title(neuronname)
    ax.set_aspect('equal', adjustable='box')
    #plt.colorbar(cb, ax=ax)
    plt.tight_layout()
    if ax is None:
        return fig, ax
    else:
        return ax


def calculate_firing_rate_by_headdir(headdir, spikes, sync, num_bins=24):
    # Bin the head direction data
    headdir_bins = np.linspace(0, 360, num_bins + 1)
    headdir_bin_centers = (headdir_bins[:-1] + headdir_bins[1:]) / 2

    # Calculate the occupancy in each bin
    occupancy, _ = np.histogram(headdir, bins=headdir_bins)

    # Calculate the number of spikes in each bin
    spike_headdir = np.interp(spikes, sync, headdir)
    spike_counts, _ = np.histogram(spike_headdir, bins=headdir_bins)

    # Calculate the firing rate in each bin (spikes per bin / occupancy per bin) adjust for 35Hz sampling
    firing_rate = np.divide(spike_counts, occupancy, where=occupancy != 0)*30
    firing_rate[occupancy == 0] = np.nan  # Avoid division by zero

    return headdir_bin_centers, firing_rate, occupancy


def plot_circular_firing_rate(headdir_bin_centers, firing_rate, neuronname=None, ax=None, savepath=None):
    """
    Plots firing rate as a function of head direction in a polar plot.
    If ax is provided, plot on it. Otherwise, create a new figure/axis.
    Optionally save to savepath.
    """
    # Convert head direction to radians
    headdir_bin_centers_rad = np.deg2rad(headdir_bin_centers)
    width = 2 * np.pi / len(headdir_bin_centers_rad)  # width of each bar (in radians)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        created_fig = True

    bars = ax.bar(headdir_bin_centers_rad, firing_rate, width=width, bottom=0.0, alpha=0.8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if neuronname:
        ax.set_title(f'Firing Rate by Head Direction\n{neuronname}')
    # You can add more styling or colorbars here if needed

    plt.tight_layout()

    if savepath is not None:
        if created_fig:
            plt.savefig(savepath, dpi=300)
        else:
            ax.figure.savefig(savepath, dpi=300)

    # Only show if we created the figure in here (avoids showing it when used as a subplot)
    if created_fig:
        plt.show()
        return fig, ax
    else:
        return ax


def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    Copied from Pycircstat documentation
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # Copied from picircstat documentation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))

d = input('give the session number ')
d = int(d)
if d == 1:
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
    trials_to_include = np.arange(1,9)
elif d == 2:
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-02_date-03072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-02_date-03072025"
    trial_numbers = np.arange(4,10)
elif d== 5:
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-05_date-18072025"
    trials_to_include = np.arange(1,11)
make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 25, sample_rate = 30000)
