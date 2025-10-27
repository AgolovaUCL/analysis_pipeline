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
from typing import Literal

def plot_spikecount_over_trials(derivatives_base, unit_type: Literal['pyramidal', 'good', 'all'], trials_to_include, task, frame_rate = 25, sample_rate = 30000):
    """For each unit, creates a plot of its spikecount throughout time.
    Also indicates the trials

    Args:
        derivatives_base (_type_): _description_
        trials_to_include (_type_): _description_
        frame_rate (int, optional): _description_. Defaults to 25.
        sample_rate (int, optional): _description_. Defaults to 30000.

    Raises:
        FileExistsError: raises error if trials_length.csv doesn't exist
    """
    # Load data files
    rawsession_folder = derivatives_base.replace('derivatives', 'rawdata')
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    kilosort_output_path = os.path.join(derivatives_base,"ephys",  "concat_run","sorting", "sorter_output" )
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
    
    # Get the data with trials length
    path_to_df = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    if not os.path.exists(path_to_df):
        raise FileExistsError('trials_length.csv doesnt exist')
    trial_length_df = pd.read_csv(path_to_df)

    bin_length = 60
    output_folder = os.path.join(derivatives_base,"analysis", "cell_characteristics", "unit_features", "spikecount_over_trials")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    goal1_endtimes = None
    if task == 'hct':
        print("HCT: adding goal times to spikecount over trials")
        trialday_path = os.path.join(rawsession_folder, 'behaviour', 'alltrials_trialday.csv')
        trialday_df  = pd.read_csv(trialday_path)
        if len(trialday_df) != len(trials_to_include):
            raise ValueError("length alltrials_trialday.csv is not the same as length trials to include. Remove unneeded trials")
        else:
            goal1_endtimes = np.array(trialday_df['Goal 1 end'])
            
    # Loading data
    print("Plotting spikecount over trials")
    for unit_id in tqdm(unit_ids): 
        
        output_path = os.path.join(output_folder, f"unit_{unit_id}_sc_over_trials.png")
        
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled/sample_rate) # spike times in seconds
        if len(spike_train) == 0:
            continue
        
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
        fig =  plt.figure(figsize=(15, 5))
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
    trials_to_include = np.arange(1,10)
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    plot_spikecount_over_trials(derivatives_base, trials_to_include)