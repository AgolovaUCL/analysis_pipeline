import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def combine_all_trials(derivatives_base, trials_to_include):
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

    print(df)
    output_path = os.path.join(folder_path, "XY_HD_alltrials.csv")
    df.to_csv(output_path, index = False)
    print(f"Dataframe saved to {output_path}")


derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
trials_to_include = np.arange(1,9)
combine_all_trials(derivatives_base, trials_to_include)
