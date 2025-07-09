import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import probeinterface
import os
from pathlib import Path
from tqdm import tqdm


def make_circular_firing_template(base_path, trials_to_include, frame_rate = 30, sampling_rate = 30000):
    kilosort_output_path = os.path.join(base_path, 'ephys', 'concat_run', 'sorting', 'shank_0', 'sorter_output')
    xy_path = os.path.join(base_path, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    plot_path = os.path.join(base_path, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'circular_firing_rate')
    
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
        headdir = xy_df.iloc[:,2:].values
        headdir = headdir.flatten()

        # savepath
        filename = f'unit_{unit_id:03d}_circular_firing_rate.png'
        output_folder = os.path.join(plot_path, labels[unit_id])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, filename)

        # Plotting
        headdir_bin_centers, firing_rate, headdir_occupancy = calculate_firing_rate_by_headdir(headdir, spike_train, sync)
        plot_circular_firing_rate(headdir_bin_centers, firing_rate, unit_id, output_path)


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

def plot_circular_firing_rate(headdir_bin_centers, firing_rate, neuronname, savepath):
    # Convert head direction to radians
    headdir_bin_centers_rad = np.deg2rad(headdir_bin_centers)

    # Create a polar plot
    plt.close()
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)
    width = 2 * np.pi / len(headdir_bin_centers_rad)  # width of each bar (in radians)
    ax.bar(headdir_bin_centers_rad, firing_rate, width=width, bottom=0.0, alpha=0.8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'Firing Rate by Head Direction for unit {neuronname}')
    plt.savefig(savepath , dpi=300)
    plt.show()
    breakpoint()


base_path = r'Z:\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-001_id-2B\ses-05_test\all_trials'
trials_to_include = [1]
make_circular_firing_template(base_path, trials_to_include, frame_rate = 30, sampling_rate = 30000)
