import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from spatial_functions import get_ratemaps
import json

def plot_ratemaps_and_hd(derivatives_base, frame_rate = 25, sample_rate = 30000):
    """ 
    Makes a plot for each unit with its ratemap (left) and directional firing rate (right)

    Inputs: derivatives base
    
    """
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, 'ephys', "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')

    # Limits
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    
    xmin = limits['xmin']
    xmax = limits['xmax']
    ymin = limits['ymin']
    ymax = limits['ymax']
    
    # ---- Load maze outline coordinates ----
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    if os.path.exists(outline_path):
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print("⚠️ Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None
        
    # Get directory for the positional data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    pos_data = pd.read_csv(pos_data_path)

    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()
    
    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'ratemaps_and_hd')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
        
    # Loop over units
    print("Plotting ratemaps and hd")
    for unit_id in tqdm(unit_ids):
        # Load spike data
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
        spike_train = [np.int32(el) for el in spike_train if el < len(x)]  # Ensure spike train is within bounds of x and y
        # Make plot
        fig, axs = plt.subplots(1, 2, figsize = [8, 4])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)


        # Plot ratemap
        rmap, x_edges, y_edges = get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)
            
        im = axs[0].imshow(rmap.T, 
                cmap='viridis', 
                interpolation = None,
                origin='lower', 
                aspect='auto', 
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

        axs[0].set_title(f"n = {len(spike_train)}")
        axs[0].set_xlim(xmin, xmax)
        axs[0].set_ylim(ymax, ymin)
        axs[0].set_aspect('equal')
        if outline_x is not None and outline_y is not None:
                axs[0].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        fig.colorbar(im, ax=axs[0], label='Firing rate')

        # Plot HD
        # Obtaining hd for this triald calculating how much the animal sampled in each bin
        num_bins = 24
        hd_filtered = hd[~np.isnan(hd)]
        hd_filtered= np.deg2rad(hd_filtered)
        occupancy_counts, _ = np.histogram(hd_filtered, bins=num_bins, range = [-np.pi, np.pi])
        occupancy_time = occupancy_counts / frame_rate 

        # Getting the spike times and making a histogram of them
        spikes_hd = hd[spike_train]
        spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
        spikes_hd_rad = np.deg2rad(spikes_hd)
        counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

        # Calculating directional firing rate
        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        fig.delaxes(axs[1])
        axs[1] = fig.add_subplot(1,2,2, polar=True)

        # Plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_centers)[0]
        axs[1].bar(
            bin_centers,
            direction_firing_rate,
            width=width,
            bottom=0.0,
            alpha=0.8
        )

        output_path = os.path.join(output_folder, f"unit_{unit_id}_rm_hd.png")
        plt.savefig(output_path)
        plt.close(fig)
    print(f"Saved plots to {output_folder}")
        
        

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    plot_ratemaps_and_hd(derivatives_base)



