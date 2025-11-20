import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from astropy.stats import circmean
import astropy.convolution as cnv

from spatial_features.get_sig_cells import get_sig_cells, resultant_vector_length # CHECK IF WORKS
from spatial_features.utils.spatial_features_utils import load_unit_ids, get_spikes_epoch, get_spikes_tr, get_trial_length_info, get_unit_info, get_posdata, get_occupancy_time, get_ratemaps, get_spike_train_frames, get_directional_firingrate
from spatial_features.utils.spatial_features_plots import plot_rmap, plot_occupancy, plot_directional_firingrate, plot_spikes_spatiotemp


def load_directories(derivatives_base):
    """ Loads df directories"""
    rawsession_folder = derivatives_base.replace('derivatives', 'rawdata')
    rawsession_folder = os.path.dirname(rawsession_folder)
    # Get directory for the positional data
    pos_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    if not os.path.exists(pos_data_dir):
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")
    
    csv_path = os.path.join(rawsession_folder, 'task_metadata', 'epoch_times.csv')
    epoch_times = pd.read_csv(csv_path)

    # loading dataframe with unit information
    path_to_df = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "all_units_overview","unit_metrics.csv")
    df_unit_metrics = pd.read_csv(path_to_df) 


    trials_length_path = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    trials_length = pd.read_csv(trials_length_path)
        
    # Output folder
    output_folder_plot = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'task_overview')
    output_folder_data = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data')
    print(f"Figures will be saved to {output_folder_plot}")
    if not os.path.exists(output_folder_plot):
        os.makedirs(output_folder_plot)
        
    if not os.path.exists(output_folder_data):
        os.makedirs(output_folder_data)
    return pos_data_dir, output_folder_data, output_folder_plot, epoch_times, df_unit_metrics, trials_length


def get_xy_pos(pos_data_dir, tr):
    """ Gets xy pos of the animal for this trial"""
    trial_csv_name = f'XY_HD_t{tr}.csv'
    trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
    xy_hd_trial = pd.read_csv(trial_csv_path)            
    x = xy_hd_trial.iloc[:, 0].to_numpy()
    y = xy_hd_trial.iloc[:, 1].to_numpy()
    hd = xy_hd_trial.iloc[:, 2].to_numpy()
    if np.nanmax(hd) > 2*np.pi + 0.1:
        hd_rad = np.deg2rad(hd)
    else:
        hd_rad = hd
    return x, y, hd_rad

def make_new_element(unit_id, tr, e, MRL, mu, percentiles_95_value, percentiles_99_value, spike_train_this_epoch):
    """ Makes new element for df"""
    new_element = {
        'cell': unit_id,
        'trial': tr,
        'epoch': e,
        'MRL': MRL,
        'mean_direction': np.rad2deg(mu),
        'mean_direction_rad':mu,
        'percentiles95': percentiles_95_value,
        'percentiles99': percentiles_99_value,
        'significant': 'ns', 
        'num_spikes': len(spike_train_this_epoch)
    } 
    if MRL > percentiles_95_value:
        new_element['significant'] = 'sig'   
    return new_element

def make_spatiotemp_plots(derivatives_base, trials_to_include, unit_type = "all", frame_rate = 25, sample_rate = 30000, num_bins = 24):
    """
    Makes the plots for the spatiotemporal experiments. Saves figures into analysis/cell_characteristics/spatial_features/spatial_plots/...
    with the following layout:
    One row for each trial
    Left column: Ratemap for trial
    Then for each epoch: left column - spike map, right column - head direction plot, with MRl denoted

    Inputs:
    derivatives_base: path to derivatives folder
    trials_to_include: trials for this recording day
    unit_type: unit types to make plots for (pyramidal, good, or all)
    frame_rate: frame_rate of camera (default = 25)
    sample_rate: sample rate of recording device (default = 30000)
    num_bins: number of bins used to bin the spatial data (default = 24, giving 15 degree bins)
    
    Returns:
    Path to df with MRL data for all units (which can be used in roseplot)
    """
    # Loading directories
    pos_data_dir, output_folder_data, output_folder_plot, epoch_times, df_unit_metrics, trials_length = load_directories(derivatives_base)
    
    # Load kilosort files
    kilosort_output_path = os.path.join(derivatives_base,  "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
    
    # For plotting
    n_epochs = 3
    n_rows = len(trials_to_include)
    n_cols = n_epochs * 2 + 1
    xmin, xmax, ymin, ymax = 500, 2000, 250, 1750
    # In this df the directional data of all units will be saved
    directional_data_all_units = pd.DataFrame(
        columns=[
            'cell', 'trial', 'epoch', 'MRL', 'mean_direction', 'mean_direction_rad',
            'percentiles95', 'percentiles99', 'significant', 'num_spikes'
        ]
    )
    

    # Looping over all units
    for unit_id in tqdm(unit_ids):

        # Loading data
        x_pos, _, _, _ = get_posdata(derivatives_base)
        spike_train = get_spike_train_frames(sorting, unit_id, x_pos, sample_rate, frame_rate)

        # Unit information
        unit_firing_rate, unit_label = get_unit_info(df_unit_metrics, unit_id)

        # Creating figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize = [3*n_cols, 3*n_rows])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}. Label: {unit_label}. Firing rate = {unit_firing_rate:.1f} Hz", fontsize = 18)
        counter = 0 # counts which axs we're on

        # Duration of trial (starts at 0)
        trial_dur_so_far = 0 # NOTE: There may be errors if trial 1 (or g0) is excluded from analysis

        # Looping over trials
        for tr in trials_to_include:

            # Trial times
            start_time, trial_length, trial_row = get_trial_length_info(epoch_times, trials_length,  tr)
            
            # Positional data
            x, y, hd_rad = get_xy_pos(pos_data_dir, tr)

            # Length of trial
            spike_train_this_trial = get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate)

            # Make plots
            # Obtaining ratemap data
            rmap, x_edges, y_edges = get_ratemaps(spike_train_this_trial, x, y, 3, binsize=25, stddev=5)
            plot_rmap(rmap,xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x = None, outline_y = None, ax = axs[counter], fig)
            counter += 1
            trial_dur_so_far += trial_length

            # Looping over epochs
            for e in range(1, n_epochs + 1):
                # Obtain epoch start and end times
                epoch_start = trial_row.iloc[0, 2*e-1]
                epoch_end = trial_row.iloc[0, 2*e]
                spike_train_this_epoch = get_spikes_epoch(spike_train_this_trial, epoch_start, epoch_end, frame_rate)
                
                
                # Spike plot
                plot_spikes_spatiotemp(spike_train_this_epoch, x, y, epoch_end, frame_rate, xmin, xmax, ymin, ymax, ax = axs[counter], title = None)
                counter += 1

                # HD calculations
                if len(spike_train_this_epoch) > 0:
                    # Obtaining hd for this epoch and calculating how much the animal sampled in each bin
                    hd_this_epoch = hd_rad[np.int32(epoch_start*frame_rate):np.int32(epoch_end*frame_rate)]
                    occupancy_time = get_occupancy_time(hd_this_epoch, frame_rate, num_bins = num_bins)
                    direction_firing_rate, bin_centers = get_directional_firingrate(hd_rad, spike_train_this_epoch, num_bins, occupancy_time)
                    
                    # Getting significance
                    MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)
                    percentiles_95_value, percentiles_99_value, _ = get_sig_cells(spike_train_this_epoch, hd_rad, epoch_start*frame_rate, epoch_end*frame_rate, occupancy_time, n_bins = num_bins)
                    mu = circmean(bin_centers, weights = direction_firing_rate)

                    # Add significance data for every element (even if not significant)
                    new_element = make_new_element(unit_id, tr, e, MRL, mu, percentiles_95_value, percentiles_99_value, spike_train_this_epoch)

                    directional_data_all_units.loc[len(directional_data_all_units)] = new_element
                
                # Plot
                fig.delaxes(axs[counter])
                axs[counter] = fig.add_subplot(n_rows, n_cols, counter+1, polar=True)

                plot_directional_firingrate(bin_centers, direction_firing_rate, ax = axs[counter], title = f"n = {len(spike_train_this_epoch)}", MRL = MRL, percentiles_95_value = percentiles_95_value)
                counter += 1 

        # Saving data
        output_path = os.path.join(output_folder_plot, f"unit_{unit_id}_spatiotemp.png")
        plt.savefig(output_path)
        plt.close(fig)

    # Saving directional data
    output_path = os.path.join(output_folder_data, f"directional_tuning_{np.int32(360/num_bins)}_degrees.csv")
    directional_data_all_units.to_csv(output_path)
    print(f"Data saved to {output_path}")
    return output_path, np.int32(360/num_bins)

if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)

    make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 25, sample_rate = 30000)