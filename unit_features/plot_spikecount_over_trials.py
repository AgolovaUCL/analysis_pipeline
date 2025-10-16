import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import json
import re
import warnings
from astropy.stats import circmean
from astropy.convolution import convolve, convolve_fft
from skimage.draw import disk
from pathlib import Path

def plot_spikecount_over_trials(derivatives_base, rawsession_folder,trials_to_include, frame_rate = 25, sample_rate = 30000):
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')
    good_units_ids = [el for el in unit_ids if labels[el] == 'good']

    # Get the data with the epoch times
    path_to_df = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    trial_length_df = pd.read_csv(path_to_df)

    bin_length = 60
    output_folder = os.path.join(derivatives_base,"analysis", "cell_characteristics", "unit_features", "spikecount_over_trials")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loading data
    print("Plotting spikecount over trials")
    for unit_id in tqdm(unit_ids): 
        
        output_path = os.path.join(output_folder, f"unit_{unit_id}_sc_over_trials.png")
        
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled/sample_rate) # spike times in seconds

        total_trial_length = 0
        for tr in trials_to_include:
            trial_row = trial_length_df[(trial_length_df.trialnumber == tr)]
            trial_length = trial_row.iloc[0, 2]
            total_trial_length += trial_length

        n_bins = total_trial_length/bin_length



        # Simulated adjacent trials
        trial_lengths = np.array(trial_length_df['trial length (s)'])
        trial_ends = np.cumsum(trial_lengths)
        trial_starts = np.concatenate(([0], trial_ends[:-1]))

        # Plot
        fig =  plt.figure(figsize=(12, 5))
        plt.hist(spike_train, bins = np.int32(n_bins))

        # Vertical lines at trial boundaries
        for start in trial_starts[1:]:
            plt.axvline(x=start, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=trial_ends[-1], color='black', linestyle='--', linewidth=1)

        # Get current y-axis limits
        ymin, ymax = plt.ylim()

        # Label position: slightly below the top of the y-axis
        label_y = ymax

        # Place trial labels
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            mid = (start + end) / 2
            plt.text(mid, label_y, f'Trial {i+1}',
                    ha='center', va='top', fontsize=9, color='black')

        # Optional: adjust y-limit if you want more headroom
        plt.ylim(ymin, ymax * 1.05)
        plt.xlim(0, np.max(trial_ends))
        # Axis labels and title
        plt.xlabel("Time (seconds)")
        plt.ylabel("Number of spikes per minute")
        plt.title(f"Unit {unit_id}: Spike count across trials")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    print(f"Plots saved in {output_folder}")

if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)
    plot_spikecount_over_trials(derivatives_base, rawsession_folder, trials_to_include)