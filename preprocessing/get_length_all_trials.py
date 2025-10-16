# %%
import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import json
import re

import warnings
from pathlib import Path



def get_length_all_trials(rawsession_folder, trials_to_include):
    ephys_path = os.path.join(rawsession_folder, 'ephys')
    output_folder = os.path.join(rawsession_folder, "task_metadata")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, "trials_length.csv")

    # Step 1: Find all run folders (e.g., ses-01_g0, ses-01_g1, etc.)
    pattern = os.path.join(ephys_path, "ses*")
    run_folders = [folder for folder in glob.glob(pattern) if os.path.isdir(folder)]

    print(f"Found {len(run_folders)} run folder(s) in {ephys_path}:\n")


    for idx, folder in enumerate(run_folders):
        print(f"  {idx+1}. {folder}")

    g_numbers = []
    trials_length = []

    # Step 2: Process each run folder
    for run_folder in run_folders:

        basename = os.path.basename(run_folder)
        match = re.search(r'ses[-_]\d+_g(\d+)', basename)
        if match:
            group_number = int(match.group(1))
            g_numbers.append(group_number)
        else:
            print(f"  Warning: Could not extract group number from {basename}")

            
        # Step 2a: Get subfolder inside run_folder (e.g., ses-01_g0_imec0)
        subfolders = [f for f in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, f))]
        if not subfolders:
            print(f"  Warning: No subfolders found in {run_folder}. Skipping.")
            continue

        subfolder_name = subfolders[0]
        subfolder_path = os.path.join(run_folder, subfolder_name)

        # Step 2b: Look for meta file
        meta_pattern = os.path.join(subfolder_path, "*meta*")
        meta_matches = glob.glob(meta_pattern)

        if not meta_matches:
            print(f"  Warning: No meta files found in {subfolder_path}. Skipping.")
            continue
        if len(meta_matches) > 1:
            print(f"  Warning: Multiple meta files found. Using the first one.")

        meta_path = meta_matches[0]

        # Step 2c: Parse meta file to get fileTimeSecs
        try:
            with open(meta_path, 'r') as f:
                content = f.read()

            match = re.search(r'fileTimeSecs\s*=\s*([\d.]+)', content)
            if match:
                file_time_secs = float(match.group(1))
                print(f"  Found fileTimeSecs = {file_time_secs}")
            else:
                print(f"  Warning: 'fileTimeSecs' not found in {meta_path}")
            trials_length.append(file_time_secs)
        except Exception as e:
            print(f"  Error reading meta file {meta_path}: {e}")

    data = {"trialnumber": trials_to_include, "g": g_numbers, "trial length (s)": trials_length}
    trial_length_df = pd.DataFrame(data)
    trial_length_df.to_csv(output_path, index = False)
    print(f"Saved to {output_path}")

