import numpy as np
import pandas as pd
import json
import os
import glob


def plot_intervals(rawsession_folder, spiketrain = None, plot_name = None):
    """
    

    Args:
        rawsession_folder (_type_): _description_
        spiketrain (_type_, optional): _description_. Defaults to None.
        plot_name (_type_, optional): _description_. Defaults to None.
    """
    
    export_path_g1 = os.path.join(rawsession_folder, "task_metadata", "goal_1_intervals.csv")
    export_path_g2 = os.path.join(rawsession_folder, "task_metadata", "goal_2_intervals.csv")

    goal_1_df = pd.read_csv(export_path_g1)
    goal_2_df = pd.read_csv(export_path_g2)