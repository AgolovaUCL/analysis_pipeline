import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.get_length_all_trials import get_length_all_trials
def combine_pos_csvs(derivatives_base, trials_to_include, frame_rate= 25):
    """
    Combines all data from XY_HD_t{tr}.csv (for tr in trials_to_include) into one csv called HD_XY_alltrials.csv
    and saves it in the same folder as the XY_HD_t{tr}.csvss

    Adds a bit of padding to match the trial length
    Input:
    dervitives_base: path to derivatives folder
    trials_to_include: our trial numbers
    
    """
    rawsession_folder = derivatives_base.replace("derivatives", "rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)

    folder_path = os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD")
    if not os.path.exists(folder_path):
        raise Exception("Path to XY and HD data does not exist")
    
    trials_length_csv = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    if not os.path.exists(trials_length_csv):
        get_length_all_trials(rawsession_folder, trials_to_include)
    trials_length = pd.read_csv(trials_length_csv)
    print("Using trials length file")
    print(trials_length)
    
    data = {"x": [], "y": [], "hd": []}
    df = pd.DataFrame(data)
    df_center = pd.DataFrame(data) # This will store the coordinates with the center positions

    total_length_s = 0
    for tr in trials_to_include:
        trials_length_tr = trials_length[trials_length["trialnumber"] == tr]
        length_s = trials_length_tr.iloc[0,2]
        length_fr= np.int32(length_s * frame_rate)
        total_length_s += length_s
        input_path = os.path.join(folder_path, f"XY_HD_t{tr}.csv")

        if not os.path.exists(input_path):
            raise Exception(f"Path to XY data for trial {tr} not found")
        
        df_tr = pd.read_csv(input_path)

        if len(df_tr) > length_fr:
            raise ValueError(f"Positional data for trial {tr} is longer than recording data. Fix error.")
        
        padding_len = length_fr - len(df_tr) # This is how much longer the recording session was than the actual trial. 
        # We add extra lines to the positional data (with nan values) in order to match the length
        nan_rows = np.repeat(np.nan, padding_len)
        padding = pd.DataFrame({"x": nan_rows, "y": nan_rows, 'hd': nan_rows})
        
        
        df = pd.concat([df, df_tr, padding])
        
        #### CENTER POSITION
        input_path = os.path.join(folder_path, f"XY_HD_center_t{tr}.csv")
        if not os.path.exists(input_path):
            raise Exception(f"Path to XY data for center for trial {tr} not found")
        
        df_tr_center = pd.read_csv(input_path)
        df_center = pd.concat([df_center, df_tr_center, padding])

    output_path = os.path.join(folder_path, "XY_HD_alltrials.csv")
    df.to_csv(output_path, index = False)

    output_path_center = os.path.join(folder_path, "XY_HD_alltrials_center.csv")
    df_center.to_csv(output_path_center, index = False)
    print(f"Dataframe saved to {output_path_center}")
    
    len_df_s = len(df)/frame_rate
    diff_s = total_length_s - len_df_s
    print(f"Difference between concat df length and total trial length: {diff_s:.2f} s")

if __name__ == "__main__":
    trials_to_include = np.arange(1,8)
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    combine_pos_csvs(derivatives_base, trials_to_include)