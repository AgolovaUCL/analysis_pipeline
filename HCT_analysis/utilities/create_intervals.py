import numpy as np
import pandas as pd
import json
import os
import glob

def create_intervals_df(rawsession_folder):
    """
    Finds the start and stop time for each trial, so that we can restrict spike times to for each goal
    NOTE Is based on the labview data
    
    
    Args:
        rawsession_folder: path to rawsession folder

    Raises:
        FileNotFoundError: No file for today_alltrials found
    
    Exports:
        Interval times for goal into rawsession_folder/task_metadata/goal_{1,2}_intervals.csv
    """
    matches = glob.glob(os.path.join(rawsession_folder, "behaviour", "alltrials_trialday.csv"))
    if len(matches) == 0:
        raise FileNotFoundError("No file found matching today_alltrials*.csv in the specified folder.")
    path = matches[0]
    
    df = pd.read_csv(path)
    
    # Goal 1
    goal_1_start_times = np.zeros(len(df))
    #goal_1_start_times = [14.2, 15.2, 13.3, 23, 25.3, 25.5, 21.3, 28.3, 31.3]

    goal_1_end_times = df['Goal 1 end'].to_numpy() # Change to whatever this is
    trial_numbers = np.arange(1, len(df)+1)
    
    # Goal 2
    goal_2_start_times = df['Goal 2 start'].to_numpy()
    goal_2_end_times = df['Trial duration'].to_numpy() # Change to whatever this is

    # Add trial length to df
    path = os.path.join(rawsession_folder, "task_metadata", "trials_length.csv")

    df_length = pd.read_csv(path)
    lengths = df_length['trial length (s)'].to_numpy()
    
    if len(df_length) != len(goal_1_end_times):
        print("Error, lengths of dataframes not aligned. Remove unused trials from alltrials_trialday")
        return -1
    
    # Getting the cumulative length
    cumul_length = [0]
    length_so_far = 0
    
    for i in range(len(lengths) - 1):
        length_so_far += lengths[i]
        cumul_length.append(length_so_far)
        
    # Add cumulative length
    goal_1_start_times = [goal_1_start_times[i] + cumul_length[i] for i in range(len(lengths))]
    goal_1_end_times = [goal_1_end_times[i] + cumul_length[i] for i in range(len(lengths))]
    
    goal_2_start_times = [goal_2_start_times[i] + cumul_length[i] for i in range(len(lengths))]
    goal_2_end_times = [goal_2_end_times[i] + cumul_length[i] for i in range(len(lengths))]
    
    # Goal 1 df
    goal_1_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'start_time': goal_1_start_times,
        'end_time': goal_1_end_times
    })
    
    goal_2_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'start_time': goal_2_start_times,
        'end_time': goal_2_end_times,
        "trial_end": cumul_length[1:]
    })

    export_path_g1 = os.path.join(rawsession_folder, "task_metadata", "goal_1_intervals.csv")
    export_path_g2 = os.path.join(rawsession_folder, "task_metadata", "goal_2_intervals.csv")

    goal_1_df.to_csv(export_path_g1, index=False)
    goal_2_df.to_csv(export_path_g2, index=False)
    print(f"Dataframes saved to {export_path_g1}")
    
def create_intervals_df_m2(rawsession_folder):
    """
    Finds the start and stop time for each trial, so that we can restrict spike times to for each goal
    NOTE Is based on the labview data
    
    Method 2: using starts and ends csv
    
    Args:
        rawsession_folder: path to rawsession folder

    Raises:
        FileNotFoundError: No file for today_alltrials found
    
    Exports:
        Interval times for goal into rawsession_folder/task_metadata/goal_{1,2}_intervals.csv
    """
    matches = glob.glob(os.path.join(rawsession_folder, "task_metadata", "starts_and_ends.csv"))
    if len(matches) == 0:
        raise FileNotFoundError("No file found matching starts_and_ends.csv in the specified folder.")
    path = matches[0]
    
    df = pd.read_csv(path)

    # Goal 1
    goal_1_start_times = np.array(df.iloc[:,0])
    goal_1_end_times = np.array(df.iloc[:,1]) # Change to whatever this is
    trial_numbers = np.arange(1, len(df)+1)
    
    # Goal 2
    goal_2_start_times = np.array(df.iloc[:,2])
    goal_2_end_times = np.array(df.iloc[:,3]) # Change to whatever this is

    # Add trial length to df
    path = os.path.join(rawsession_folder, "task_metadata", "trials_length.csv")

    df_length = pd.read_csv(path)
    lengths = df_length['trial length (s)'].to_numpy()
    
    if len(df_length) != len(goal_1_end_times):
        print("Error, lengths of dataframes not aligned. Remove unused trials from alltrials_trialday")
        return -1
    
    # Getting the cumulative length
    cumul_length = [0]
    length_so_far = 0
    
    for i in range(len(lengths)):
        length_so_far += lengths[i]
        cumul_length.append(length_so_far)
        
    # Add cumulative length
    goal_1_start_times = [goal_1_start_times[i] + cumul_length[i] for i in range(len(lengths))]
    goal_1_end_times = [goal_1_end_times[i] + cumul_length[i] for i in range(len(lengths))]
    
    goal_2_start_times = [goal_2_start_times[i] + cumul_length[i] for i in range(len(lengths))]
    goal_2_end_times = [goal_2_end_times[i] + cumul_length[i] for i in range(len(lengths))]

    # Goal 1 df
    goal_1_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'start_time': goal_1_start_times,
        'end_time': goal_1_end_times
    })
    
    goal_2_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'start_time': goal_2_start_times,
        'end_time': goal_2_end_times,
        "trial_end": cumul_length[1:]
    })

    export_path_g1 = os.path.join(rawsession_folder, "task_metadata", "goal_1_intervals.csv")
    export_path_g2 = os.path.join(rawsession_folder, "task_metadata", "goal_2_intervals.csv")

    goal_1_df.to_csv(export_path_g1, index=False)
    goal_2_df.to_csv(export_path_g2, index=False)
    print(f"Dataframes saved to {export_path_g1}")

    


if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    create_intervals_df_m2(rawsession_folder)