
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


# Loading data

d = 1
if d == 1:
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
    trials_to_include = np.arange(1,9)
elif d == 2:
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-02_date-03072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-02_date-03072025"
    trials_to_include = np.arange(4,10)
else:
    derivatives_base = r"Z:\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-001_id-2B\ses-05_test\all_trials"
    rawsession_folder = r"Z:\Eylon\Data\Honeycomb_Maze_Task\rawdata\sub-001_id-2B\ses-05_test"
    trials_to_include = np.array([1,2])
frame_rate = 30
sample_rate = 30000


n_trials = len(trials_to_include)
n_epochs = 3
n_rows = n_trials
n_cols = n_epochs * 2 + 1
min_spikes = 5 # minimum number of spikes for a cell to be deemed significant
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

# Get the data with the epoch times
csv_path = os.path.join(rawsession_folder, 'task_metadata', 'timestamps.csv')
xlsx_path = os.path.join(rawsession_folder, 'task_metadata', 'timestamps.xlsx')
if os.path.exists(csv_path):
    epoch_times = pd.read_csv(csv_path)
elif os.path.exists(xlsx_path):
    epoch_times = pd.read_excel(xlsx_path)
else:
    raise FileNotFoundError(f"Epoch times file does not exist: {csv_path} or {xlsx_path}")

# loading dataframe with unit information
path_to_df = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "all_units_overview","unit_metrics.csv")
df_unit_metrics = pd.read_csv(path_to_df) 

bin_edges = np.linspace(0,360,73)


trials_length_path = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
trials_length = pd.read_csv(trials_length_path)
# Tables for statistics
# to add later
trial_dur_so_far = 0

unit_id = input('give the unit id: ')
unit_id = int(unit_id)
spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data


# Unit information
unit_firing_rate = 5
row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]

unit_firing_rate = row['firing_rate'].values[0]
unit_label = row['label'].values[0]

# Creating figure
fig, axs = plt.subplots(3, 3, figsize = [30, 30])
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
    xy = xy_hd_trial.iloc[:, :2].to_numpy().T
    x = xy_hd_trial.iloc[:, 0].to_numpy()
    y = xy_hd_trial.iloc[:, 1].to_numpy()
    hd = xy_hd_trial.iloc[:, 2].to_numpy()
    sync = np.arange(len(x))
    # Length of trial
    spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)*frame_rate] # filtering for current trial
    spike_train_this_trial = [el - trial_dur_so_far*frame_rate for el in spike_train_this_trial]
    spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)]

    # Make plots
    # === Histogram of spike times for this trial ===
    axs[counter].set_title(f"Trial {tr} | n = {len(spike_train_this_trial)} spikes")
        # Plot histogram of spike times
    axs[counter].hist(
        spike_train_this_trial,
        bins=50,
        range=(0, len(x)),
        color='black',
        alpha=0.7,
        zorder = 2
    )

    # Plot dotted lines for epoch start and end
    for e in range(1, n_epochs + 1):
        epoch_start = trial_row.iloc[0, 2*e - 1]
        epoch_end = trial_row.iloc[0, 2*e]

        axs[counter].axvspan(
            epoch_start*frame_rate,
            epoch_end*frame_rate,
            facecolor='lightblue',  # or 'lightblue'
            alpha=0.5,
            zorder = 0
        )
        epoch_mid = (epoch_start + epoch_end) *frame_rate/ 2

        # Add text label "Epoch {e}" at the midpoint
        axs[counter].text(
            epoch_mid,
            axs[counter].get_ylim()[1] *0.95,  # slightly above the top of the histogram
            f"Epoch {e}",
            ha='center',
            va='bottom',
            fontsize=8,
        )
    

    axs[counter].set_xlim(0, len(x))
    axs[counter].set_xlabel("Frame")
    axs[counter].set_ylabel("Spike count")
    counter += 1
    trial_dur_so_far += trial_length


output_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'task_overview')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"unit_{unit_id}_spatiotemp.png")
#plt.savefig(output_path)
plt.show()


