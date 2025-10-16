import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from matplotlib.path import Path

def add_platforms_to_csv(derivatives_base):
    """
    Adds platforms to XY_HD_alltrials_center.csv and saves it as XY_HD_w_platforms.csv

    Input:
    dervitives_base: path to derivatives folder
    
    """
    # Loading positional data
    folder_path = os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD")
    pos_path = os.path.join(folder_path, "XY_HD_alltrials_center.csv")
    
    if not os.path.exists(pos_path):
        print("Error, XY_HD_alltrials_center.csv not found")
        
    pos_data = pd.read_csv(pos_path)
    
    # Loading hexagon parameters
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    pos_data_w_plat = add_platforms_to_all(pos_data, params)
    output_path = os.path.join(folder_path, "XY_HD_w_platforms.csv")
    pos_data_w_plat.to_csv(output_path, index = False)
    print(f"Saved data to {output_path}")
    
    
def add_platforms_to_all(pos_data, params):
    platforms = []
    hcoord = params["hcoord_tr"]
    vcoord = params["vcoord_tr"]
    hex_side_length = params["hex_side_length"]
        
    for i in tqdm(range(len(pos_data))):
        x = pos_data['x'].iloc[i]
        y = pos_data['y'].iloc[i]
        if np.isnan(x):
            plat = np.nan
        else:
            plat = get_platform_number(x, y, hcoord, vcoord, hex_side_length)
        platforms.append(plat)
        
    pos_data['platform'] = platforms
    return pos_data

def is_point_in_platform(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            hcoord + hex_side_length * np.cos(angle),
            vcoord + hex_side_length * np.sin(angle)
        ])
    hexagon_path = Path(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

# Updated code to find the hexagon the rat is in
def get_platform_number(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):

    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if is_point_in_platform(rat_locx, rat_locy, x, y, hex_side_length):
            return i + 1
    return np.nan

if __name__ == "__main__":
    derivatives_base = r"C:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    add_platforms_to_csv(derivatives_base)
    
    
    