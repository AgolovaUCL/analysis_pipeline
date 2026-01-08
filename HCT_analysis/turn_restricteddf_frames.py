import os
import pandas as pd
import numpy as np

def turn_restricteddf_frames(derivatives_base, frame_rate = 25):
    """ Converts restricted df columns from seconds to frames"""
    
    # rawsession folder
    rawsession_folder = derivatives_base.replace("derivatives", "rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # get file
    path = os.path.join(rawsession_folder, 'task_metadata', 'restricted_final.csv')
    df = pd.read_csv(path)
    
    # make into frames
    df = np.round(df*frame_rate).astype(int) 
    
    # export
    output_folder = os.path.join(rawsession_folder, 'task_metadata', 'restricted_df_frames.csv')
    df.to_csv(output_folder, index = False)
    
    print(f"Saved df to {output_folder}")
    
if __name__ == "__main__":
    derivatives_base =  r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    turn_restricteddf_frames(derivatives_base)
