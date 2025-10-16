import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from spatial_functions import get_ratemaps
import json

def test_rmap(derivatives_base, pixels_per_cm = 9, frame_rate = 25, sample_rate = 30000):
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
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'ratemaps_test')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
        
    # Loop over units
    print("Plotting ratemaps and hd")
    unit_id = 2
    # Load spike data
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
    spike_train = [np.int32(el) for el in spike_train if el < len(x)]  # Ensure spike train is within bounds of x and y
    # Make plot
    fig, axs = plt.subplots(3, 3, figsize = [20, 20])

    fig.suptitle(f"Unit {unit_id}", fontsize = 18)

    bin_sizes = [2*pixels_per_cm, 3*pixels_per_cm, 4*pixels_per_cm]
    kernel_sizes = [3,5, 7]
    for i in range(3):
        for j in range(3):
            ax = axs[i,j]
            # Plot ratemap

            rmap, x_edges, y_edges = get_ratemaps(spike_train, x, y,  kernel_sizes[j], binsize=bin_sizes[i], stddev=5)
                
            im = ax.imshow(rmap.T, 
                    cmap='viridis', 
                    interpolation = None,
                    origin='lower', 
                    aspect='auto', 
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

            ax.set_title(f"n = {len(spike_train)}")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
            ax.set_aspect('equal')
            if outline_x is not None and outline_y is not None:
                ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
            fig.colorbar(im, ax=ax, label='Firing rate')
            ax.set_title(f'Binsize = {bin_sizes[i]}, kernel = {kernel_sizes[j]}')

    plt.savefig(os.path.join(output_folder, "kernel_size_test.png"))
    plt.show()
    print(f"Saved plots to {output_folder}")
        
        

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    test_rmap(derivatives_base)



