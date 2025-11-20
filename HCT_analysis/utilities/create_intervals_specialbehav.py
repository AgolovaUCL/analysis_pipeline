import numpy as np
import pandas as pd
import os

def create_intervals_specialbehav(rawsession_folder):
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
    print("Creating fdx")
    # Add trial length to df
    path = os.path.join(rawsession_folder, "task_metadata", "trials_length.csv")

    df_length = pd.read_csv(path)
    lengths = df_length['trial length (s)'].to_numpy()

    # Df with restrictions
    try: 
        path = os.path.join(rawsession_folder, "task_metadata", "restricted_df.xlsx")
        df_restricted = pd.read_excel(path)
    except:
        path = os.path.join(rawsession_folder, "task_metadata", "restricted_df.csv")
        df_restricted = pd.read_csv(path)
        

    # Getting the cumulative length
    cumul_length = [0]
    length_so_far = 0
    
    for i in range(len(lengths) - 1):
        length_so_far += lengths[i]
        cumul_length.append(length_so_far)
   
    df_restricted = df_restricted.add(cumul_length, axis = 0)

    output_path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")
    df_restricted.to_csv(output_path, index = False)
    print(f"Saved data to {output_path}")
    
if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    #rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-003_id-2F\ses-01_date-17092025"
    create_intervals_specialbehav(rawsession_folder)