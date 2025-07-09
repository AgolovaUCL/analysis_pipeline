import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import spikeinterface.extractors as se
import os
from tqdm import tqdm
import pandas as pd

def make_spatial_plots_spatiotemporal_task(base_path, trials_to_include, frame_rate = 30, sampling_rate = 30000):
    kilosort_output_path = os.path.join(base_path, 'ephys', 'concat_run', 'sorting', 'shank_0', 'sorter_output')
    xy_path = os.path.join(base_path, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    plot_path = os.path.join(base_path, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'rate_maps')
    
    # Load sorting data
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')
    good_units_ids = [el for el in unit_ids if labels[el] == 'good']


    # Load xy data
    for trial in trials_to_include:
        # load trial
        path_to_csv = os.path.join(xy_path, 'XY_HD_t1.csv')
        xy_df = pd.read_csv(path_to_csv)

    num_frames = len(xy_df)
    sync = np.arange(num_frames) / frame_rate

    for unit_id in tqdm(unit_ids):
        # Loading data
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.float64(spike_train_unscaled/sampling_rate)
        xycoords = xy_df.iloc[:, :2].to_numpy().T

        # savepath
        filename = f'unit_{unit_id:03d}_ratemap.png'
        output_folder = os.path.join(plot_path, labels[unit_id])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, filename)

        # Plotting
        firing_rate_map, x_bins, y_bins, location_occupancy = spatialheatmap(xycoords, spike_train, sync, frame_rate, bin_size=2)
        plotspatialheatmap(firing_rate_map, x_bins, y_bins, unit_id, output_path)

    



def spatialheatmap(xycoords, spikes, sync, frame_rate = 30, bin_size=2, sigma=1):
    # Define the spatial bins
    x_bins = np.arange(np.min(xycoords[0]), np.max(xycoords[0]) + bin_size, bin_size)
    y_bins = np.arange(np.min(xycoords[1]), np.max(xycoords[1]) + bin_size, bin_size)

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

def plotspatialheatmap(firing_rate_map, x_bins, y_bins, neuronname, savepath):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    cb = ax.imshow(firing_rate_map.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none',
              extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')
    ax.set_title(neuronname)

    plt.colorbar(cb)
    plt.tight_layout()
    plt.savefig(savepath)


base_path = r'Z:\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-001_id-2B\ses-05_test\all_trials'
trials_to_include = [1]
make_spatial_plots_spatiotemporal_task(base_path, trials_to_include, frame_rate = 30, sampling_rate = 30000)
