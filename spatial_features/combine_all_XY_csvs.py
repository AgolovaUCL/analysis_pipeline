import os
import pandas as pd
import numpy as np


def combine_all_XY_csvs(derivatives_base, trials_to_include):
    
    XY_and_HD_path = os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD")
    
    for tr in trials_to_include:
        trial_str = f"t{tr}"
        csv_path = os.path.join(XY_and_HD_path, f"XY_HD_{trial_str}.csv")
        
        if tr == trials_to_include[0]:
            all_XY_df = pd.read_csv(csv_path)
        else:
            trial_XY_df = pd.read_csv(csv_path)
            all_XY_df = pd.concat([all_XY_df, trial_XY_df], ignore_index=True)
    
    all_XY_df.to_csv(os.path.join(XY_and_HD_path, "XY_HD_alltrials.csv"), index=False)
    print(f"Combined XY and HD data for trials {trials_to_include} into {os.path.join(XY_and_HD_path, 'XY_HD_alltrials.csv')}")
    
if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    trials_to_include = np.arange(1, 11)  # Adjust the range as needed
    combine_all_XY_csvs(derivatives_base, trials_to_include)