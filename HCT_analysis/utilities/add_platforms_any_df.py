import os
import numpy as np
import pandas as pd
from matplotlib.path import Path
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
def add_platforms_to_csv(pos_data, derivatives_base):
    """
    Adds platforms to df and returns it

    Input:
    pos_data - df with pos data
    
    """
    
    # Loading hexagon parameters
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    pos_data_w_plat = add_platforms_to_all(pos_data, params)

    return pos_data_w_plat
    
    
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
