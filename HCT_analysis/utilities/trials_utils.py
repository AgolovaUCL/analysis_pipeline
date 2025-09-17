
import os
import glob 
import pandas as pd
import numpy as np

def append_alltrials(rawsession_folder):
    """ 
    Takes the alltrials csv and creates a new csv only with the rows of the trial date

    Args:
        rawsession_folder (str): path to the rawdata folder
    
    Exports:
        alltrials_trialday.csv into rawsession/behaviour
    """
    folder = os.path.basename(rawsession_folder)        # Obtains session name, example: "ses-02_date-05092025"
    date = folder.split("date-")[-1]  # Obtains number after 'date', for example "05092025"
    
    alltrials_paths = glob.glob(os.path.join(rawsession_folder,"behaviour" ,"alltrials*.csv"))
    alltrials_path = alltrials_paths[0]
    
    df = pd.read_csv(alltrials_path)
    
    # If date starts with 0, remove it
    if date.startswith("0"):
        date = date[1:]

    df = df[df['Date'] == int(date)]
    
    output_path = os.path.join(rawsession_folder,"behaviour", "alltrials_trialday.csv")
    df.to_csv(output_path, index=False)
    print(f"Created {output_path}")

def get_goal_numbers(rawsession_folder):
    """
    Obtains goal numbers from alltrials_trialday.csv
    
    Args:
        rawsession_folder (str): path to the rawdata folder
    
    Returns:
        [goal1, goal2]
    """
    
    df_path = os.path.join(rawsession_folder, "task_metadata", "alltrials_trialday.csv")
    df = pd.read_csv(df_path)
    goal1 = df['Goal 1'].values[0]
    goal2 = df['Goal 2'].values[0]
    return [np.int32(goal1), np.int32(goal2)]

if __name__ == "__main__":
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-02_date-05092025"
    append_alltrials(rawsession_folder)
    goal_numbers = get_goal_numbers(rawsession_folder)
    print(f"Goal numbers: {goal_numbers}")