
import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_directories(derivatives_base, rawsession_folder):
    """ Loads paths and dfs that we need"""
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
    
    output_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'epochs_firing_rate')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return pos_data_dir, epoch_times, df_unit_metrics, trials_length, output_dir

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

def get_unit_info(df_unit_metrics, unit_id):
    """ Loads unit firing rate and label for unit = unit_id"""
    row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]
    unit_firing_rate = row['firing_rate'].values[0]
    unit_label = row['label'].values[0]
    return unit_firing_rate, unit_label


def load_trial_xpos(pos_data_dir, tr):
    """ Returns x pos for trial tr"""
    trial_csv_name = f'XY_HD_t{tr}.csv'
    trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
    xy_hd_trial = pd.read_csv(trial_csv_path)
                
    x = xy_hd_trial.iloc[:, 0].to_numpy()  
    return x 

def get_trial_length_info(epoch_times, trials_length,  tr):
    """ Returns start time of trial and trial length"""
    trial_row = epoch_times[(epoch_times.trialnumber == tr)]
    start_time = trial_row.iloc[0, 1]

    trial_length_row = trials_length[(trials_length.trialnumber == tr)]
    trial_length = trial_length_row.iloc[0, 2]
    return start_time, trial_length, trial_row
            
def get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate):
    """ Restricts spiketrain to current trial"""
    spike_train_this_trial = np.copy(spike_train)
    spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)] # filtering for current trial
    spike_train_this_trial = [el - trial_dur_so_far for el in spike_train_this_trial]
    spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)/frame_rate]
    return spike_train_this_trial

def plot_trial_firing(spike_train_this_trial,trial_row, n_epochs, tr, x, frame_rate, ax):
    """ Makes subplot of firing for one trial per each epoch"""
    ax.set_title(f"Trial {tr} | n = {len(spike_train_this_trial)} spikes")
     # Plot histogram of spike times
    ax.hist(
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

        ax.axvspan(
            epoch_start,
            epoch_end,
            facecolor='lightblue',  # or 'lightblue'
            alpha=0.5,
            zorder = 0
        )
        epoch_mid = (epoch_start + epoch_end)/ 2

        # Add text label "Epoch {e}" at the midpoint
        ax.text(
            epoch_mid,
            ax.get_ylim()[1] *0.95,  # slightly above the top of the histogram
            f"Epoch {e}",
            ha='center',
            va='bottom',
            fontsize=8,
        )
    

    ax.set_xlim(0, len(x)/frame_rate)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike count")       
    
def plot_firing_each_epoch(derivatives_base, trials_to_include, unit_type = "all", frame_rate = 25, sample_rate = 30000):
    """For each unit, creates an n by 3 plot showing the firing rate for each trial 
    with each epoch imdicated. 

    Args:
        derivatives_base: path to derivatives folder_
        trials_to_include: array with trial numbers.
        unit_type: type of units to visualize for (pyramidal, good, or all)
        frame_rate (int, optional): Frame rate of video. Defaults to 25.
        sample_rate (int, optional): Sample rate of recording. Defaults to 30000.

    Raises:
        FileNotFoundError: _description_
        
    Outputs:
    """
    # Loading rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base,"ephys",  "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)

    # Loading data
    pos_data_dir, epoch_times, df_unit_metrics, trials_length, output_dir = load_directories(derivatives_base, rawsession_folder)
    
    # Trial information    
    n_trials = len(trials_to_include)
    n_epochs = 3
    
    n_cols = 3
    n_rows = np.ceil(n_trials/n_cols).astype(int)
    trial_dur_so_far = 0

    print(f"Plotting firing for each epoch. Saving to {output_dir}")
    for unit_id in tqdm(unit_ids):
        # Getting spiketrian for unit
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled/sample_rate) # Now its in seconds

        # Unit information
        unit_firing_rate, unit_label = get_unit_info(df_unit_metrics, unit_id)

        # Creating figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize = [n_cols*7, n_rows*5])
        fig.suptitle(f"Unit {unit_id}. Label: {unit_label}. Firing rate = {unit_firing_rate:.1f} Hz", fontsize = 18)
        axs = axs.flatten()
        counter = 0 # counts which axs we're on
        trial_dur_so_far = 0
        
        # Looping over trials
        for tr in trials_to_include:
            # Trial times
            start_time, trial_length, trial_row = get_trial_length_info(epoch_times, trials_length,  tr)
            
            # Get spiketrain for this trial
            x = load_trial_xpos(pos_data_dir, tr)
            spike_train_this_trial = get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate)
            
            # Make plots
            plot_trial_firing(spike_train_this_trial,trial_row, n_epochs, tr, x, frame_rate, ax = axs[counter])
            
            counter += 1
            trial_dur_so_far += trial_length


        output_path = os.path.join(output_dir, f"unit_{unit_id}_epoch_firing.png")
        plt.savefig(output_path)
        plt.close(fig)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)
    plot_firing_each_epoch(derivatives_base, rawsession_folder, trials_to_include)