
import os
import glob 
import pandas as pd
import numpy as np
import json

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

def get_goal_numbers(derivatives_base):
    """
    Obtains goal numbers from alltrials_trialday.csv
    
    Args:
        rawsession_folder (str): path to the rawdata folder
    
    Returns:
        [goal1, goal2]
    """
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    df_path = os.path.join(rawsession_folder, "behaviour", "alltrials_trialday.csv")
    df = pd.read_csv(df_path)
    goal1 = df['Goal 1'].values[0]
    goal2 = df['Goal 2'].values[0]
    
    ## Gets goal coordinates
    params_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path) as f:
        params= json.load(f)
    hcoord_tr = params["hcoord_tr"]
    vcoord_tr= params["vcoord_tr"]
    
    goal1_coords = [hcoord_tr[np.int32(goal1 -1)], vcoord_tr[np.int32(goal1 - 1)]]
    goal2_coords = [hcoord_tr[np.int32(goal2 -1)], vcoord_tr[np.int32(goal2 - 1)]]
    
    coords_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "goal_coords.json")
    coords = {
        "goal1_coords": goal1_coords,
        "goal2_coords": goal2_coords
    }
    os.makedirs(os.path.dirname(coords_path), exist_ok=True)
    with open(coords_path, 'w') as f:
        json.dump(coords, f, indent=4)
        
    return [np.int32(goal1), np.int32(goal2)]

def get_coords(derivatives_base):
    params_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path) as f:
        params= json.load(f)
    hcoord_tr = params["hcoord_tr"]
    vcoord_tr= params["vcoord_tr"]
    return hcoord_tr, vcoord_tr
def get_goal_coordinates(derivatives_base, rawsession_folder):
    """
    Returns:
        Goal coordinates. If json file with them doesn't exist, it makes it
    """
    coords_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "goal_coords.json")
    if not os.path.exists(coords_path):
        get_goal_numbers(derivatives_base, rawsession_folder)
    
    with open(coords_path) as f:
        data= json.load(f)

    goal1_coords = data["goal1_coords"]
    goal2_coords = data["goal2_coords"]
    return [goal1_coords, goal2_coords]

def get_limits_from_json(derivatives_base):
    """Gets the xy limits from the json file created in the get_limits.py function"""
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    return limits["xmin"], limits["xmax"], limits["ymin"], limits["ymax"]

if __name__ == "__main__":
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-02_date-05092025"
    append_alltrials(rawsession_folder)
    goal_numbers = get_goal_numbers(rawsession_folder)
    print(f"Goal numbers: {goal_numbers}")