


from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np
import os
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from spatial_functions import get_ratemaps, get_ratemaps_restrictedx
from get_sig_cells import get_sig_cells
import json
import warnings
from astropy.stats import circmean
from matplotlib.gridspec import GridSpec

def plot_rmap_interactive(derivatives_base, unit_id, task, frame_rate = 25, sample_rate = 30000):
    """ 
    Makes a plot for each unit with its ratemap (left), occupancy (middle) and directional firing rate (right).
    User adjustable

    Inputs: derivatives base
    
    """
    # Load data files
    rawsession_folder = derivatives_base.replace('derivatives', 'rawdata')
    rawsession_folder = os.path.dirname(rawsession_folder)
    
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
    bin_length = 60
    hd_restrict = None
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
    
    # Get the data with trials length
    path_to_df = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    if not os.path.exists(path_to_df):
        raise FileExistsError('trials_length.csv doesnt exist')
    trial_length_df = pd.read_csv(path_to_df)
    
    goal1_endtimes = None
    inputs = None
    x_int = None
    x_restrict = None
    y_restrict = None
    if task == 'hct':
        print("HCT: adding goal times to spikecount over trials")
        trialday_path = os.path.join(rawsession_folder, 'behaviour', 'alltrials_trialday.csv')
        trialday_df  = pd.read_csv(trialday_path)
        if len(trialday_df) != len(trial_length_df):
            raise ValueError("length alltrials_trialday.csv is not the same as length trials to include. Remove unneeded trials")
        else:
            goal1_endtimes = np.array(trialday_df['Goal 1 end'])
            
    while input != 'q':
        # Make plot
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])

        ax1 = fig.add_subplot(gs[0, 0])           # ratemap
        ax2 = fig.add_subplot(gs[0, 1])           # quiver plot
        ax3 = fig.add_subplot(gs[0, 2], polar=True)  # polar plot
        ax4 = fig.add_subplot(gs[1, :])           # bottom row spans all columns

        fig.subplots_adjust(
            wspace=0.35,  # horizontal spacing
            hspace=0.45   # vertical spacing
        )
                
        fig.suptitle(f"Unit {unit_id}", fontsize = 18, y = 0.95)

        # ====== Plot ratemap ======
        
        # Masking values
        if inputs is not None:
            spike_train_filt = []
            for s in spike_train:
                s_in_sec = s/frame_rate # Adjusting s to be in seconds
                for start, end in x_int:
                    if s_in_sec > start and s_in_sec < end:
                        spike_train_filt.append(s)
        else:
            spike_train_filt = spike_train
        
        # Get original ratemap
        if inputs is not None:
            rmap, x_edges, y_edges = get_ratemaps_restrictedx(spike_train_filt, x, y, x_restrict, y_restrict, 3, binsize=36, stddev=25)
      
        im = ax1.imshow(rmap.T, 
                cmap='viridis', 
                interpolation = None,
                origin='lower', 
                aspect='auto', 
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])


        ax1.set_title(f"n = {len(spike_train)}")
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymax, ymin)
        ax1.set_aspect('equal')
        if outline_x is not None and outline_y is not None:
                ax1.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        #fig.colorbar(im, ax=axs[0], label='Firing rate')
        #fig.colorbar(im, ax=ax1, label='Firing rate', shrink = 0.5)

        # 2. spikemap
        is_filt = np.isin(spike_train, spike_train_filt)
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
        ax2.scatter(x_spikes, y_spikes, color=colors)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymax, ymin)
        ax2.set_aspect('equal')
        if outline_x is not None and outline_y is not None:
                ax2.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        ax2.set_title('Blue: within interval. Red: outside interval')
        # Plot HD
        # Getting the spike times and making a histogram of them
         
        spikes_hd = hd[spike_train_filt]
        spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
        spikes_hd_rad = np.deg2rad(spikes_hd)
    
        counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

        
        
        # Calculating directional firing rate
        if hd_restrict is None:
            direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        else:
            hd_filtered_r = hd_restrict[~np.isnan(hd_restrict)]
            hd_filtered_r= np.deg2rad(hd_filtered_r)
            occupancy_counts_r, _ = np.histogram(hd_filtered_r, bins=num_bins, range = [-np.pi, np.pi])
            occupancy_time_r = occupancy_counts_r / frame_rate 
            direction_firing_rate = np.divide(counts, occupancy_time_r, out=np.full_like(counts, 0, dtype=float), where=occupancy_time_r!=0)
        # MRL adn significance
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plotting
        width = np.diff(bin_centers)[0]
        ax3.bar(
            bin_centers,
            direction_firing_rate,
            width=width,
            bottom=0.0,
            alpha=0.8
        )

        ax3.set_title(f'n (in region): {len(spike_train_filt)}')
        
        # Spike count over time
        
        total_trial_length = 0
        for tr in range(1, len(trial_length_df + 1)):
            trial_row = trial_length_df[(trial_length_df.trialnumber == tr)]
            trial_length = trial_row.iloc[0, 2]
            total_trial_length += trial_length

        n_bins = total_trial_length/bin_length


        # Simulated adjacent trials
        trial_lengths = np.array(trial_length_df['trial length (s)'])
        trial_ends = np.cumsum(trial_lengths)
        trial_starts = np.concatenate(([0], trial_ends[:-1]))

        # Plot
        spike_train_s = np.array(spike_train_unscaled)/sample_rate
        ax4.hist(spike_train_s, bins = np.int32(n_bins))
        # Vertical lines at trial boundaries
        for start in trial_starts[1:]:
            ax4.axvline(x=start, color='black', linestyle='--', linewidth=1)
        ax4.axvline(x=trial_ends[-1], color='black', linestyle='--', linewidth=1)

        # Get current y-axis limits
        ymin4, ymax4 = ax4.get_ylim()

        # Label position: slightly below the top of the y-axis
        label_y = ymax4

        # Place trial labels
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            mid = (start + end) / 2
            ax4.text(mid, label_y, f'Trial {i+1}',
                    ha='center', va='top', fontsize=9, color='black')
            if goal1_endtimes is not None:
                ax4.axvspan(
                            start,
                            start + goal1_endtimes[i],
                            facecolor='lightblue',  # or 'lightblue'
                            alpha=0.5,
                            zorder = 0
                        )
        if x_int is not None:
            for start, end in x_int:
                ax4.axvspan(
                            start,
                            end,
                            facecolor = 'red',  # or 'lightblue'
                            alpha=0.5,
                            zorder = 0
                        )
        # Optional: adjust y-limit if you want more headroom
        ax4.set_ylim(ymin4, ymax4 * 1.05)
        ax4.set_xlim(0, np.max(trial_ends))
        # Axis labels and title
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Number of spikes per minute")

        plt.tight_layout()
        print("Select intervals (up to 10 in total)")
        inputs = plt.ginput(20)
        x_points =  [val[0] for val in inputs] # only getting x values (These correspond to time)
        x_points.sort()
        
        if len(inputs) %2 != 0: # uneven number of points selected
            print("Uneven number of points selected. Last point will be ignored for intervals")
            x_points = x_points[:-1]
            
        x_int = []
        for i in range(np.int32(len(x_points)/2)):
            x_int.append((x_points[2*i], x_points[2*i + 1]))

        mask_time = []
        for i in range(len(pos_data)):
            in_int = False
            for start, end in x_int:
                if i/frame_rate > start and i/frame_rate < end:
                    mask_time.append(True)
                    in_int = True
                    break
            if not in_int:
                mask_time.append(False)
        x_restrict = x[mask_time]
        y_restrict = y[mask_time]
        hd_restrict = hd[mask_time]
        
        plt.show(block=True)
        plt.pause(0.1)
        plt.close(fig)


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
    plot_rmap_interactive(derivatives_base, unit_id, task = 'hct')



