import numpy as np
import pandas as pd
import json
import os
import glob

def create_intervals_df(rawsession_folder):
    """
    Finds the start and stop time for each trial, so that we can restrict spike times to for each goal
    NOTE Is based on the labview data, which isn't synced with the spike data yet.
    
    
    Args:
        rawsession_folder: path to rawsession folder

    Raises:
        FileNotFoundError: No file for today_alltrials found
    
    Exports:
        Interval times for goal into rawsession_folder/task_metadata/goal_{1,2}_intervals.csv
    """
    matches = glob.glob(os.path.join(rawsession_folder, "behaviour", "today_alltrials*.csv"))
    if len(matches) == 0:
        raise FileNotFoundError("No file found matching today_alltrials*.csv in the specified folder.")
    path = matches[0]
    
    df = pd.read_csv(path)
    
    # Goal 1
    goal_1_start_times = np.zeros(len(df))
    goal_1_end_times = df['goal 1 end'].to_numpy() # Change to whatever this is
    trial_numbers = np.arange(1, len(df)+1)
    
    # Goal 2
    goal_2_start_times = df['goal 2 start'].to_numpy()
    goal_2_end_times = df['trial length'].to_numpy() # Change to whatever this is

    # Add trial length to df
    path = os.path.join(rawsession_folder, "task_metadata", "trials_length.csv")

    df_length = pd.read_csv(path)
    lengths = df_length['trial length (s)'].to_numpy()
    
    # Add length of trial-1 to trial
    goal_1_start_times = [el + lengths[i- 1] if i > 0 else el for i, el in enumerate(goal_1_start_times)]
    goal_1_end_times = [el + lengths[i-1] if i > 0 else el for i, el in enumerate(goal_1_end_times)]

    goal_2_start_times = [el + lengths[i-1] if i > 0 else el for i, el in enumerate(goal_2_start_times)]
    goal_2_end_times = [el + lengths[i-1] if i > 0 else el for i, el in enumerate(goal_2_end_times)]
    
    # Goal 1 df
    goal_1_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'start_time': goal_1_start_times,
        'end_time': goal_1_end_times
    })
    
    goal_2_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'start_time': goal_2_start_times,
        'end_time': goal_2_end_times
    })
    
    export_path_g1 = os.path.join(rawsession_folder, "task_metadata", "goal_1_intervals.csv")
    export_path_g2 = os.path.join(rawsession_folder, "task_metadata", "goal_2_intervals.csv")

    goal_1_df.to_csv(export_path_g1, index=False)
    goal_2_df.to_csv(export_path_g2, index=False)


    