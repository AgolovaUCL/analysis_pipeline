
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np
import os
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from spatial_functions import get_ratemaps
from get_sig_cells import get_sig_cells
import json
import warnings
from astropy.stats import circmean

def mask_posdata(pos_data, mask):
    """
    Masks positional data

    Args:
        pos_data: array with x, y, hd, x_bin and y_bin
        mask: 2D boolean
    returns hd_masked
    """
    xsize, ysize = mask.shape

    x_bin = pos_data["x_bin"].to_numpy()
    y_bin = pos_data["y_bin"].to_numpy()
    hd = pos_data["hd"].to_numpy()

    valid = (
        (x_bin >= 0) & (x_bin < xsize) &
        (y_bin >= 0) & (y_bin < ysize)
    )

    valid_mask = np.zeros_like(valid, dtype=bool)
    valid_indices = np.where(valid)
    valid_mask[valid_indices] = mask[x_bin[valid], y_bin[valid]]

    # Return masked hd values (ignore NaNs)
    return hd[valid_mask & ~np.isnan(hd)]
                    
    
def plot_rmap_interactive(derivatives_base, unit_id,  frame_rate = 25, sample_rate = 30000):
    """ 
    Makes a plot for each unit with its ratemap (left), occupancy (middle) and directional firing rate (right).
    User adjustable

    Inputs: derivatives base
    
    """
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, 'ephys', "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
   
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
    
        
    # Loop over units
    print("Plotting ratemaps and hd")

    # Obtaining hd for this trial how much the animal sampled in each bin
    num_bins = 24
    hd_filtered = hd[~np.isnan(hd)]
    hd_filtered= np.deg2rad(hd_filtered)
    occupancy_counts, _ = np.histogram(hd_filtered, bins=num_bins, range = [-np.pi, np.pi])
    occupancy_time = occupancy_counts / frame_rate 


    # Load spike data
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_pre = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
    spike_train = [np.int32(el) for el in spike_train_pre if el < len(x)]  # Ensure spike train is within bounds of x and y
    
    input = 'c'
    
    mask = None 
    
    # Get original ratemap
    rmap, x_edges, y_edges = get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)
    
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    
    mask_x = np.isnan(x)
    mask_y = np.isnan(y)
    
    y_bin[mask_y] = -1
    x_bin[mask_x] = -1
    
    pos_data['x_bin'] = x_bin
    pos_data['y_bin'] = y_bin
     
    while input != 'q':
        # Make plot
        fig, axs = plt.subplots(1, 4, figsize = [20, 5])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)

        # ====== Plot ratemap ======
        if mask is not None:
            hd_masked = mask_posdata(pos_data, mask)

            occupancy_counts_masked, _ = np.histogram(hd_masked, bins=num_bins, range = [-np.pi, np.pi])
            occupancy_time_masked = occupancy_counts_masked / frame_rate 
    
            xsize, ysize = mask.shape
            spike_train_filt = []
            for s in spike_train:
                indx = x_bin[s]
                indy = y_bin[s]
                if indx < xsize and indy < ysize and mask[indx, indy]:
                    spike_train_filt.append(s)
        else:
            spike_train_filt = spike_train
            
        is_filt = np.isin(spike_train, spike_train_filt)
        
        im = axs[0].imshow(rmap.T, 
                cmap='viridis', 
                interpolation = None,
                origin='lower', 
                aspect='auto', 
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])


        axs[0].set_title(f"{len(spike_train_filt)}/{len(spike_train)}")
        axs[0].set_xlim(xmin, xmax)
        axs[0].set_ylim(ymax, ymin)
        axs[0].set_aspect('equal')
        if outline_x is not None and outline_y is not None:
                axs[0].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        #fig.colorbar(im, ax=axs[0], label='Firing rate')
        fig.colorbar(im, ax=axs[0], label='Firing rate')

        # 2. spikemap
        x_spikes = x[spike_train]
        y_spikes = y[spike_train]
        hd_spikes = hd[spike_train]


        valid = ~np.isnan(x_spikes) & ~np.isnan(y_spikes) & ~np.isnan(hd_spikes)
        x_spikes = x_spikes[valid]
        y_spikes = y_spikes[valid]
        hd_spikes = hd_spikes[valid]
        is_filt = is_filt[valid]  

        u = np.cos(np.deg2rad(hd_spikes))
        v = np.sin(np.deg2rad(hd_spikes))

        # Assign colors efficiently
        colors = np.where(is_filt, 'blue', 'red')

        # Plot
        axs[1].quiver(x_spikes, y_spikes, u, v, color=colors, scale = 30)
        axs[1].set_xlim(xmin, xmax)
        axs[1].set_ylim(ymax, ymin)
        axs[1].set_aspect('equal')
        if outline_x is not None and outline_y is not None:
                axs[1].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        axs[1].set_title('Blue: within interval. Red: outside interval')
        
        # Plot HD
        # Getting the spike times and making a histogram of them
        
        spikes_hd = hd[spike_train_filt]
        spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
        spikes_hd_rad = np.deg2rad(spikes_hd)
        counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

        
        # Calculating directional firing rate
        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        if mask is not None:
            direction_firing_rate_masked = np.divide(counts, occupancy_time_masked,out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0 )
        else:
            direction_firing_rate_masked = direction_firing_rate
        fig.delaxes(axs[2])
        axs[2] = fig.add_subplot(1,4,3, polar=True)

        # MRL adn significance
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
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
        # Plotting
        width = np.diff(bin_centers)[0]
        axs[2].bar(
            bin_centers,
            direction_firing_rate,
            width=width,
            bottom=0.0,
            alpha=0.8
        )
        if False:
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

        # Plotting
        fig.delaxes(axs[3])
        axs[3] = fig.add_subplot(1,4,4, polar=True)
        width = np.diff(bin_centers)[0]
        axs[3].bar(
            bin_centers,
            direction_firing_rate_masked,
            width=width,
            bottom=0.0,
            alpha=0.8
        )
        
        plt.tight_layout()
        plt.show(block=False)   # show figure but don’t block execution yet

        # Now let the user draw on the ratemap (axs[0])
        mask = select_region(rmap, axs[0], x_edges, y_edges)
        inside_coords = np.argwhere(mask)
        print("Number of points inside polygon:", inside_coords.shape[0])


def select_region(data, ax, x_edges, y_edges):
    """
    Allows the user to draw a polygon on the given Axes and returns a boolean mask
    for the region inside the polygon (same shape as `data`).
    """

    mask_container = {'done': False, 'mask': None}

    def onselect(verts):

        ny, nx = data.shape
        x_lin = np.linspace(x_edges[0], x_edges[-1], ny)  # ny instead of nx
        y_lin = np.linspace(y_edges[0], y_edges[-1], nx)  # nx instead of ny
        X, Y = np.meshgrid(x_lin, y_lin)

        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(verts)
        mask = path.contains_points(points).reshape((nx, ny)).T  # transpose back
        mask_container['mask'] = mask
        mask_container['done'] = True

        ax.contour(mask, colors='r', linewidths=0.8)
        plt.draw()

        selector.disconnect_events()
        plt.close()

    selector = PolygonSelector(ax, onselect, props=dict(color='r', linewidth=2, alpha=0.6))

    print("Draw your polygon on the ratemap (double-click to close it)...")
    plt.show(block=True)

    # Wait until user finishes
    while not mask_container['done']:
        plt.pause(0.1)

    return mask_container['mask']


        
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
    unit_id = 61
    plot_rmap_interactive(derivatives_base, unit_id)


