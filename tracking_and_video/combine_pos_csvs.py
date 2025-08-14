import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def combine_pos_csvs(derivatives_base, trials_to_include):
    """
    Combines all data from XY_HD_t{tr}.csv (for tr in trials_to_include) into one csv called HD_XY_alltrials.csv
    and saves it in the same folder as the XY_HD_t{tr}.csvss

    Input:
    dervitives_base: path to derivatives folder
    trials_to_include: our trial numbers
    
    """

    folder_path = os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD")
    if not os.path.exists(folder_path):
        raise Exception("Path to XY and HD data does not exist")
    
    data = {"x": [], "y": [], "hd": []}
    df = pd.DataFrame(data)
    for tr in trials_to_include:
        input_path = os.path.join(folder_path, f"XY_HD_t{tr}.csv")

        if not os.path.exists(input_path):
            raise Exception(f"Path to XY data for trial {tr} not found")
        
        df_tr = pd.read_csv(input_path)

        df = pd.concat([df, df_tr])

    output_path = os.path.join(folder_path, "XY_HD_alltrials.csv")
    df.to_csv(output_path, index = False)
    print(f"Dataframe saved to {output_path}")
