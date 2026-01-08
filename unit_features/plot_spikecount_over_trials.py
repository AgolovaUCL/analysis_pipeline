import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal

def load_unit_ids(derivatives_base, unit_type, unit_ids):
    """ Returns unit_ids, the unit_ids that we will create rmaps for"""
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
        print("Using pyramidal units")
    elif unit_type == "all":
        print("Using all units")
        unit_ids = unit_ids
    else:
        raise ValueError("unit_type not good, pyramidal, or all. Provide correct input")
    return unit_ids

def get_trials_length_df(rawsession_folder):
    """ Returns trial_length_df"""
    # Get the data with trials length
    path_to_df = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    if not os.path.exists(path_to_df):
        raise FileExistsError('trials_length.csv doesnt exist')
    trial_length_df = pd.read_csv(path_to_df)
    return trial_length_df

def get_goal_1_end_times(rawsession_folder, trials_to_include):
    """ Returns end times of goal 1 based on alltrials trial_day"""
    print("HCT: adding goal times to spikecount over trials")
    trialday_path = os.path.join(rawsession_folder, 'behaviour', 'alltrials_trialday.csv')
    trialday_df  = pd.read_csv(trialday_path)
    if len(trialday_df) != len(trials_to_include):
        raise ValueError("length alltrials_trialday.csv is not the same as length trials to include. Remove unneeded trials")
    else:
        goal1_endtimes = np.array(trialday_df['Goal 1 end'])
    return goal1_endtimes

def get_spike_train_s(sorting, unit_id, sample_rate):
    """ Returns spiketrian in seconds for unit_id"""
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train = np.round(spike_train_unscaled/sample_rate) # spike times in seconds
    return spike_train

def get_total_trial_length(trials_to_include, trial_length_df):
    """ Returns sum of all trials length in seconds"""
    total_trial_length = 0
    for tr in trials_to_include:
        trial_row = trial_length_df[(trial_length_df.trialnumber == tr)]
        trial_length = trial_row.iloc[0, 2]
        total_trial_length += trial_length   
    return total_trial_length

def make_plot(spike_train, trial_starts, trial_ends, output_path, n_bins, unit_id, goal1_endtimes = None):
    """ Makes plot for unit_id and saves it"""
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
        if goal1_endtimes is not None:
            plt.axvspan(
                        start,
                        start + goal1_endtimes[i],
                        facecolor='lightblue',  # or 'lightblue'
                        alpha=0.5,
                        zorder = 0
                    )
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

def plot_spikecount_over_trials(derivatives_base, unit_type: Literal['pyramidal', 'good', 'all'], trials_to_include, task, frame_rate = 25, sample_rate = 30000):
    r"""For each unit, creates a plot of its spikecount throughout time.
    Also indicates the trials

    Args:
        derivatives_base (_type_): _description_
        trials_to_include (_type_): _description_
        frame_rate (int, optional): _description_. Defaults to 25.
        sample_rate (int, optional): _description_. Defaults to 30000.


    Raises:
    ValueError
        - If `unit_type` is not one of {'good', 'pyramidal', 'all'} in `load_unit_ids`.
        - If the number of rows in `alltrials_trialday.csv` does not match the number 
        of trials in `trials_to_include` in `get_goal_1_end_times`.

    FileNotFoundError
        - If `trials_length.csv` is missing in `get_trials_length_df`.
    
    Outputs:
        derivatives_base\analysis\cell_characteristics\unit_features\spikecount_over_trials\unit_{}_sc_over_trials.png: shows the spike counts over trials
    """
    # Load data files
    rawsession_folder = derivatives_base.replace('derivatives', 'rawdata')
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    kilosort_output_path = os.path.join(derivatives_base,"ephys",  "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    
    output_folder = os.path.join(derivatives_base,"analysis", "cell_characteristics", "unit_features", "spikecount_over_trials")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    trial_length_df = get_trials_length_df(rawsession_folder)

   
    # Definitions
    goal1_endtimes = None
    bin_length = 60
    if task == 'hct':
        goal1_endtimes = get_goal_1_end_times(rawsession_folder, trials_to_include)
            
    # Loading data
    print("Plotting spikecount over trials")
    for unit_id in tqdm(unit_ids): 
        # output path for unit
        output_path = os.path.join(output_folder, f"unit_{unit_id}_sc_over_trials.png")
        
        # loading spiketrain in seconds
        spike_train = get_spike_train_s(sorting, unit_id, sample_rate)

        # skip units with zero spikes
        if len(spike_train) == 0:
            continue
        
        # Getting number of bins
        total_trial_length = get_total_trial_length(trials_to_include, trial_length_df) # total length in s
        n_bins = total_trial_length/bin_length


        # Simulated adjacent trials
        trial_lengths = np.array(trial_length_df['trial length (s)'])
        trial_ends = np.cumsum(trial_lengths)
        trial_starts = np.concatenate(([0], trial_ends[:-1]))

        # Plot
        make_plot(spike_train, trial_starts, trial_ends, output_path, n_bins, unit_id, goal1_endtimes)
        
    print(f"Plots saved in {output_folder}")

if __name__ == "__main__":
    trials_to_include = np.arange(1,10)
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    plot_spikecount_over_trials(derivatives_base, trials_to_include)