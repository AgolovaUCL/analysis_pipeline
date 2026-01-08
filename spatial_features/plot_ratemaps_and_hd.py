import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
import warnings

from spatial_features.spatial_functions import get_ratemaps
from spatial_features.get_sig_cells import get_sig_cells
from astropy.stats import circmean
from spatial_features.utils.spatial_features_utils import load_unit_ids, get_outline, get_limits, get_posdata, get_occupancy_time, get_ratemaps, get_spike_train_frames, get_directional_firingrate
from spatial_features.utils.spatial_features_plots import plot_rmap, plot_occupancy, plot_directional_firingrate



def plot_ratemaps_and_hd(derivatives_base, unit_type: Literal['pyramidal', 'good', 'all'],  frame_rate = 25, sample_rate = 30000, saveplots=True, show_plots= False):
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
    
    # Output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'ratemaps_and_hd')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
    # Limits and outline
    xmin, xmax, ymin, ymax = get_limits(derivatives_base)
    outline_x, outline_y = get_outline(derivatives_base)
        
    # Get directory for the positional data
    x, y, hd,_ = get_posdata(derivatives_base, method = "ears")

    # Obtaining hd for this trial how much the animal sampled in each bin
    num_bins = 24
    occupancy_time = get_occupancy_time(hd, frame_rate, num_bins = num_bins)

     # Loop over units
    print("Plotting ratemaps and hd")
    for unit_id in tqdm(unit_ids):
        
        # Load spike data
        spike_train = get_spike_train_frames(sorting, unit_id, x, sample_rate, frame_rate)
        
        # Make plot
        fig, axs = plt.subplots(1, 3, figsize = [15, 5])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)

        # ===== Plot ratemap ====
        rmap, x_edges, y_edges= get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)


        plot_rmap(rmap, xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x, outline_y, ax = axs[0], fig = fig, title = f"n = {len(spike_train)}")

        # ==== Plot occupancy ====
        plot_occupancy(x, y, xmin, xmax, ymin, ymax, outline_x, outline_y, frame_rate, axs[1], fig)

        # === Plot HD ===
        direction_firing_rate, bin_centers = get_directional_firingrate(hd, spike_train, num_bins, occupancy_time)
        fig.delaxes(axs[2])
        axs[2] = fig.add_subplot(1,3,3, polar=True)
        plot_directional_firingrate(bin_centers, direction_firing_rate, ax = axs[2])

        
        if False:
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

        output_path = os.path.join(output_folder, f"unit_{unit_id}_rm_hd.png")
        if saveplots:
            plt.savefig(output_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        if False:
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
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    plot_ratemaps_and_hd(derivatives_base,unit_type = "good", saveplots=False, show_plots=True)



