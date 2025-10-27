import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from spatial_features.spatial_functions import get_ratemaps
from spatial_features.get_sig_cells import get_sig_cells
import json
from typing import Literal
import warnings
from astropy.stats import circmean

def plot_ratemaps_and_hd(derivatives_base, unit_type: Literal['pyramidal', 'good', 'all'],  frame_rate = 25, sample_rate = 30000):
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

    if unit_type == 'good':
        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting","sorter_output", "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features","all_units_overview", "pyramidal_units_2D.csv")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
    
    
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
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'ratemaps_and_hd')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
        
    # Loop over units
    print("Plotting ratemaps and hd")

    # Obtaining hd for this trial how much the animal sampled in each bin
    num_bins = 24
    hd_filtered = hd[~np.isnan(hd)]
    hd_filtered= np.deg2rad(hd_filtered)
    occupancy_counts, _ = np.histogram(hd_filtered, bins=num_bins, range = [-np.pi, np.pi])
    occupancy_time = occupancy_counts / frame_rate 


    for unit_id in tqdm(unit_ids):
        # Load spike data
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train_pre = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
        spike_train = [np.int32(el) for el in spike_train_pre if el < len(x)]  # Ensure spike train is within bounds of x and y
        # Make plot
        fig, axs = plt.subplots(1, 3, figsize = [15, 5])
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

    
        # 2. Plot occupancy
        x_no_nan =  x[~pd.isna(x)]
        y_no_nan = y[~pd.isna(y)]
        heatmap_data, xbins, ybins  = np.histogram2d(x_no_nan, y_no_nan, bins=20, range=[[xmin, xmax], [ymin, ymax]])
        heatmap_data = heatmap_data/frame_rate

        im = axs[1].imshow(
                heatmap_data.T,
                cmap='viridis',
                interpolation=None,
                origin='upper',
                aspect='auto',
                extent=[xmin, xmax, ymax, ymin]
            )
        fig.colorbar(im, ax=axs[1], label='Seconds')
        if outline_x is not None and outline_y is not None:
            axs[1].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        axs[1].set_title('Occupancy full session')
        axs[1].set_aspect('equal')

        # Plot HD
        # Getting the spike times and making a histogram of them
        spikes_hd = hd[spike_train]
        spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
        spikes_hd_rad = np.deg2rad(spikes_hd)
        counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

        
        # Calculating directional firing rate
        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        fig.delaxes(axs[2])
        axs[2] = fig.add_subplot(1,3,3, polar=True)

        # MRL adn significance
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)
        perc_95, perc_99, MRL_values, shift_value = get_sig_cells(spike_train, np.deg2rad(hd),0, len(hd) -1, occupancy_time )
        angle = circmean(bin_centers, weights=direction_firing_rate)
        
        if MRL> perc_99:
            text = f'MRL: {MRL:.2f}**'
            print(perc_99)
        elif MRL > perc_95:
            text = f'MRL: {MRL:.2f}*'
            print(perc_95)
        else:
            text = f'MRL: {MRL:.2f}, ns'
        # Plotting
        width = np.diff(bin_centers)[0]
        axs[2].bar(
            bin_centers,
            counts,
            width=width,
            bottom=0.0,
            alpha=0.8
        )
        max_rate = np.nanmax(direction_firing_rate)
        axs[2].plot(
            [angle, angle],               # theta values
            [0, max_rate],                # r values (from center to max)
            color='red',                  # choose your color
            linewidth=2,                  # line thickness
            label='Mean direction'
        )

        # Optional: add text and legend
        axs[2].text(0.05, 1.05, text, transform=axs[2].transAxes)
        axs[2].legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))


        #axs[2].text(180, np.nanmax(direction_firing_rate) * 1.5, text)
        print(f'angle = {np.rad2deg(angle)}')
        output_path = os.path.join(output_folder, f"unit_{unit_id}_rm_hd.png")
        
        plt.show()
        #plt.savefig(output_path)
        #plt.close(fig)
        try:
            plt.hist(MRL_values, bins = 30, color = 'lightblue')
            plt.axvline(x = perc_95, color = 'b', label = '95th perc')
            plt.axvline(x = perc_99, color = 'g', label  = '99th perc')
            plt.axvline(x = MRL, color = 'r', label = 'MRL')
            plt.legend()
            plt.show()
        except:
            print("skipping")
        
    
    print(f"Saved plots to {output_folder}")
        
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

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-003_id-2F\ses-01_date-17092025\all_trials"
    plot_ratemaps_and_hd(derivatives_base)



