
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
from skimage.morphology import disk



def plot_firing_each_epoch(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 25, sample_rate = 30000):
    n_trials = len(trials_to_include)
    n_epochs = 3
    
    n_cols = 3
    n_rows = np.ceil(n_trials/n_cols).astype(int)
    
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

    # Get the data with the epoch times
    csv_path = os.path.join(rawsession_folder, 'task_metadata', 'epoch_times.csv')
    epoch_times = pd.read_csv(csv_path)

    # loading dataframe with unit information
    path_to_df = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "all_units_overview","unit_metrics.csv")
    df_unit_metrics = pd.read_csv(path_to_df) 


    trials_length_path = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    trials_length = pd.read_csv(trials_length_path)
    # Tables for statistics
    # to add later
    trial_dur_so_far = 0

    for unit_id in tqdm(unit_ids):
        unit_id = int(unit_id)
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        #spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
        spike_train = np.round(spike_train_unscaled/sample_rate) # Now its in seconds

        # Unit information
        unit_firing_rate = 5
        row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]

        unit_firing_rate = row['firing_rate'].values[0]
        unit_label = row['label'].values[0]

        # Creating figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize = [n_cols*7, n_rows*5])
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
            # Length of trial
            spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)] # filtering for current trial
            spike_train_this_trial = [el - trial_dur_so_far for el in spike_train_this_trial]
            spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)/frame_rate]

            # Make plots
            # === Histogram of spike times for this trial ===
            axs[counter].set_title(f"Trial {tr} | n = {len(spike_train_this_trial)} spikes")
                # Plot histogram of spike times
            axs[counter].hist(
                spike_train_this_trial,
                bins=50,
                range=(0, len(x)/frame_rate),
                color='black',
                alpha=0.7,
                zorder = 2
            )

            # Plot dotted lines for epoch start and end
            for e in range(1, n_epochs + 1):
                epoch_start = trial_row.iloc[0, 2*e - 1]
                epoch_end = trial_row.iloc[0, 2*e]

                axs[counter].axvspan(
                    epoch_start,
                    epoch_end,
                    facecolor='lightblue',  # or 'lightblue'
                    alpha=0.5,
                    zorder = 0
                )
                epoch_mid = (epoch_start + epoch_end)/ 2

                # Add text label "Epoch {e}" at the midpoint
                axs[counter].text(
                    epoch_mid,
                    axs[counter].get_ylim()[1] *0.95,  # slightly above the top of the histogram
                    f"Epoch {e}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
            

            axs[counter].set_xlim(0, len(x)/frame_rate)
            axs[counter].set_xlabel("Time (s)")
            axs[counter].set_ylabel("Spike count")
            counter += 1
            trial_dur_so_far += trial_length


        output_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'epochs_firing_rate')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"unit_{unit_id}_epoch_firing.png")
        plt.savefig(output_path)
        plt.close(fig)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)
    plot_firing_each_epoch(derivatives_base, rawsession_folder, trials_to_include)