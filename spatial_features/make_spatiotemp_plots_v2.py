import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import warnings
from astropy.stats import circmean
from astropy.convolution import convolve, convolve_fft
import astropy.convolution as cnv
from skimage.morphology import disk
import random


def calculate_firing_rate_by_headdir(headdir, spikes, sync, num_bins=24):
    # Bin the head direction data
    headdir_bins = np.linspace(-180, 180, num_bins + 1)
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

def get_sig_cells_m2(spike_train_this_epoch, hd,epoch_start_frame, epoch_end_frame,  occupancy_time, n_bins = 72, frame_rate = 25):
    num_shifts = 1000
    shift_min = 2*frame_rate
    shift_max = np.int32(epoch_end_frame - epoch_start_frame) - 20*frame_rate
    if shift_min > shift_max:
        shift_max_temp = shift_min
        shift_min = shift_max
        shift_max = shift_max_temp
        if shift_min < 0:
            shift_min = 0
    MRL_values = []


    current_data = spike_train_this_epoch


    if np.nanmin(hd) < -np.pi or np.nanmax(hd) > np.pi:
        hd_rad = np.deg2rad(hd)
    else:
        hd_rad = hd

    for shift_idx in range(num_shifts):
        # Get random shift value
        random_shift = random.randint(shift_min, shift_max)

        # Add or subtract the random_shift value from each element in the variable
        shifted_data = current_data + random_shift

        range_min = np.int32(epoch_start_frame)
        range_max = np.int32(epoch_end_frame)
        range_size = range_max - range_min + 1
        
        # Ensure shifted_data stays within the range [range_min, range_max]
        shifted_data = np.mod(shifted_data - range_min, range_size) + range_min

        # Calculate angles_degrees and MRL
        angles_radians= hd_rad[shifted_data]; 
        mask = ~np.isnan(angles_radians)
        angles_radians= angles_radians[mask]

        counts, bin_edges = np.histogram(angles_radians, bins=n_bins,range = [-np.pi, np.pi] )

        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, np.nan, dtype=float), where=occupancy_time!=0)
        direction_firing_rate[np.isnan(direction_firing_rate)] = 0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)

        MRL_values.append( MRL)

    perc_95_val = np.percentile(MRL_values, 95)
    perc_99_val = np.percentile(MRL_values, 99)
    return perc_95_val, perc_99_val, MRL_values

def resultant_vector_length_m2(angles, w=None):
    """
    angles: array of angles in radians
    w: optional weights
    """
    if w is None:
        w = np.ones_like(angles)
    x = np.sum(w * np.cos(angles))
    y = np.sum(w * np.sin(angles))
    R = np.sqrt(x**2 + y**2)
    return R / np.sum(w)


def get_ratemaps(spikes, x, y, n: int, binsize = 15, stddev = 5):
    """
    Calculate the rate map for given spikes and positions.

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    pos_binned, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_binned, _, _ = np.histogram2d(spikes_x, spikes_y, bins=[x_bins, y_bins])
    
    g = cnv.Box2DKernel(n)
    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    return rmap, x_edges, y_edges, pos_binned, spikes_binned, smoothed_pos, smoothed_spikes


def make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 25, sample_rate = 30000):
    n_trials = len(trials_to_include)
    n_epochs = 3
    n_rows = n_trials
    n_cols = n_epochs * 2 + 1
    
    directional_data_all_units = pd.DataFrame(
    columns=[
        'cell', 'trial', 'epoch', 'MRL', 'mean_direction',
        'percentiles95', 'percentiles99', 'significant', 'num_spikes'
    ]
    )

    # Load data files
    kilosort_output_path = os.path.join(derivatives_base,  "concat_run","sorting", "sorter_output" )
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
    
    csv_path = glob.glob(os.path.join(rawsession_folder, 'behaviour*.csv'))
    if len(csv_path) > 0:
        epoch_times_allcols = pd.read_csv(csv_path[0], header=None)
    else:
        excel_path = glob.glob(os.path.join(rawsession_folder, 'behaviour*.xlsx'))
        if len(excel_path) > 0:
            epoch_times_allcols = pd.read_excel(excel_path[0], header=None)
        else:
            raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')


    # loading dataframe with unit information
    path_to_df = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "all_units_overview","unit_metrics.csv")
    df_unit_metrics = pd.read_csv(path_to_df) 


    epoch_times= epoch_times_allcols.iloc[:, [10, 12, 14, 16, 18]]
    epoch_times.columns = ['epoch 1 end', 'epoch 2 start', 'epoch 2 end', 'epoch 3 start', 'epoch 3 end']
    epoch_times.insert(0, "epoch 1 start", np.zeros(len(epoch_times)))
    epoch_times.insert(0,'trialnumber',  trials_to_include)

    trials_length_path = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    trials_length = pd.read_csv(trials_length_path)
        

    directional_data_all_units = pd.DataFrame(
    columns=[
        'cell', 'trial', 'epoch', 'MRL', 'mean_direction',
        'percentiles95', 'percentiles99', 'significant', 'num_spikes'
    ]
    )
    output_folder_plot = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'task_overview')
    output_folder_data = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data')

    if not os.path.exists(output_folder_plot):
        os.makedirs(output_folder_plot)
        
    if not os.path.exists(output_folder_data):
        os.makedirs(output_folder_data)
    
    # Looping over all units
    for unit_id in tqdm(unit_ids):

        # Loading data
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data

        # Unit information
        row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]

        unit_firing_rate = row['firing_rate'].values[0]
        unit_label = row['label'].values[0]
        n_rows = len(trials_to_include)
        # Creating figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize = [3*n_cols, 3*n_rows])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}. Label: {unit_label}. Firing rate = {unit_firing_rate:.1f} Hz", fontsize = 18)
        counter = 0 # counts which axs we're on

        trial_dur_so_far = 0

        # Looping over trials
        for tr in trials_to_include:
            spike_train_this_trial = np.copy(spike_train)
            # Trial times
            trial_row = epoch_times[(epoch_times.trialnumber == tr)]
            start_time = trial_row.iloc[0, 1]

            trial_length_row = trials_length[(trials_length.trialnumber == tr)]
            trial_length = trial_length_row.iloc[0, 2]
            # Positional data
            trial_csv_name = f'XY_HD_t{tr}.csv'
            trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
            xy_hd_trial = pd.read_csv(trial_csv_path)            
            x = xy_hd_trial.iloc[:, 0].to_numpy()
            y = xy_hd_trial.iloc[:, 1].to_numpy()
            hd = xy_hd_trial.iloc[:, 2].to_numpy()
            
            # Length of trial
            spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)*frame_rate] # filtering for current trial
            spike_train_this_trial = [el - trial_dur_so_far*frame_rate for el in spike_train_this_trial]
            spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)]

            # Make plots
            rmap, x_edges, y_edges, pos_binned, spikes_binned, pos_smoothed, spikes_smoothed = get_ratemaps(spike_train_this_trial, x, y, 3, binsize=25, stddev=5)
            axs[counter].imshow(rmap.T, 
                    cmap='viridis', 
                    interpolation = None,
                    origin='lower', 
                    aspect='auto', 
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

            axs[counter].set_title(f"n = {len(spike_train_this_trial)}")
            axs[counter].set_xlim(550, 2050)
            axs[counter].set_ylim(1750, 250)
            axs[counter].set_aspect('equal')
            counter += 1

            trial_dur_so_far += trial_length

            for e in range(1, n_epochs + 1):
                # Obtain epoch start and end times
                epoch_start = trial_row.iloc[0, 2*e-1]
                epoch_end = trial_row.iloc[0, 2*e]

                spike_train_this_epoch = [np.int32(el) for el in spike_train_this_trial if el > frame_rate*epoch_start and el < frame_rate *epoch_end]
                spike_train_this_epoch = np.asarray(spike_train_this_epoch, dtype=int)

                
                # spike plot
                x_until_now = x[:np.int32(epoch_end*frame_rate)]
                y_until_now = y[:np.int32(epoch_end*frame_rate)]
                axs[counter].scatter(x_until_now, y_until_now, color = 'black', s= 0.7)
                if len(spike_train_this_epoch) > 0:
                    axs[counter].scatter(x[spike_train_this_epoch], y[spike_train_this_epoch], color = 'r', s= 0.7)
                axs[counter].set_xlim(550, 2050)
                axs[counter].set_ylim(1750,250)

                axs[counter].set_aspect('equal', adjustable='box')
                counter += 1

                # HD calculations
                if len(spike_train_this_epoch) > 0:
                    num_bins = 24

                    hd_this_epoch = hd[np.int32(epoch_start*frame_rate):np.int32(epoch_end*frame_rate)]
                    hd_this_epoch = hd_this_epoch[~np.isnan(hd_this_epoch)]
                    hd_this_epoch_rad = np.deg2rad(hd_this_epoch)
                    occupancy_counts, _ = np.histogram(hd_this_epoch_rad, bins=num_bins, range = [-np.pi, np.pi])
                    occupancy_time = occupancy_counts / frame_rate 

                    spikes_hd = hd[spike_train_this_epoch]
                    spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
                    spikes_hd_rad = np.deg2rad(spikes_hd)
                    counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

                    direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, np.nan, dtype=float), where=occupancy_time!=0)
                    direction_firing_rate = np.array([0 if np.isnan(el) else el for el in direction_firing_rate])

                    # Getting significance

                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)
                    percentiles_95_value, percentiles_99_value, MRL_values = get_sig_cells_m2(spike_train_this_epoch, hd, epoch_start*frame_rate, epoch_end*frame_rate, occupancy_time, n_bins = num_bins)
                    mu = circmean(bin_centers, weights = direction_firing_rate)

                    # Add significance data for every element (even if not significant)
                    new_element = {
                        'cell': unit_id,
                        'trial': tr,
                        'epoch': e,
                        'MRL': MRL,
                        'mean_direction': np.rad2deg(mu),
                        'percentiles95': percentiles_95_value,
                        'percentiles99': percentiles_99_value,
                        'significant': 'ns', 
                        'num_spikes': len(spike_train_this_epoch)
                    }

                    if MRL > percentiles_95_value:
                        new_element['significant'] = 'sig'


                    
                    directional_data_all_units.loc[len(directional_data_all_units)] = new_element
                # Plot
                fig.delaxes(axs[counter])
                axs[counter] = fig.add_subplot(n_rows, n_cols, counter+1, polar=True)
                width = np.diff(bin_centers)[0]

                if len(spike_train_this_epoch) > 0:
                    bars = axs[counter].bar(
                        bin_centers,
                        direction_firing_rate,
                        width=width,
                        bottom=0.0,
                        alpha=0.8
                    )

                    if MRL > percentiles_95_value:
                        text = f"MRL = {MRL:.2f}*"
                    else:
                        text = f"MRL = {MRL:.2f}"
                    axs[counter].text(
                        np.pi/3,                # angle in radians
                        np.nanmax(direction_firing_rate)*1.25,         # radius (just outside the bar)
                        text,   # label text
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation_mode='anchor',
                        color = 'r',
                    )
        

                    
                axs[counter].set_title(f"n = {len(spike_train_this_epoch)}")

                counter += 1 

                


        output_path = os.path.join(output_folder_plot, f"unit_{unit_id}_spatiotemp.png")
        plt.savefig(output_path)
        plt.close(fig)

    output_path = os.path.join(output_folder_data, f"directional_tuning_{np.int32(360/num_bins)}_degrees.csv")
    directional_data_all_units.to_csv(output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)

    make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 25, sample_rate = 30000)