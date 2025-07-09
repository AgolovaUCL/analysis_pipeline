import numpy as np
import os
from pathlib import Path
import glob
import re


def get_trial_length(session_folder, trials_to_include):
    ephys_path = os.path.join(session_folder, 'ephys')
    

    total_trial_length = 0.0

    for i in trials_to_include:
        # obtain run folder
        pattern = os.path.join(ephys_path, f"run-{i:03d}*")
        matches = glob.glob(pattern)

        dir = matches[0] 

        if len(matches) > 1:
            print(f"Warning: Multiple directories found for run-{i:03d}. Using the first one: {dir}")
        if not os.path.isdir(dir):
            print(f"Warning: Directory {dir} does not exist. Skipping this run.")
            continue
        # get meta file
        pattern = os.path.join(dir, "*meta*")
        matches = glob.glob(pattern)
        meta_path = matches[0] if matches else None

        if not meta_path:
            print(f"Warning: No meta file found in {dir}. Skipping this run.")
            continue
        if len(matches) > 1:
            print(f"Warning: Multiple meta files found in {dir}. Using the first one: {meta_path}")

        # obtain trial length from meta file
        try:
            meta_file = Path(meta_path)

            if not meta_file.exists():
                print(f"Warning: file not found: {meta_file}")
                continue

            with open(meta_file, 'r') as f:
                content = f.read()

            match = re.search(r'fileTimeSecs\s*=\s*([\d.]+)', content)
            if match:
                file_time_secs = float(match.group(1))
                total_trial_length += file_time_secs
            else:
                print(f"Warning: 'fileTimeSecs' not found in {meta_file}")

        except Exception as e:
            print(f"Error reading {meta_path}: {e}")
        # add to total trial length
    return total_trial_length