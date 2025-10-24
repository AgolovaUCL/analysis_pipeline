import numpy as np
import os
from pathlib import Path
import glob
import re


def get_trial_length(session_folder, trials_to_include):
    ephys_path = os.path.join(session_folder, 'ephys')
    

    total_trial_length = 0.0

    trial_numbers = [el - 1 for el in trials_to_include]  # Convert to zero-based index

        # obtain run folder
    pattern = os.path.join(ephys_path, f"ses-{2:02d}*")
    global_matches = glob.glob(pattern)


    for i in range(len(global_matches)):
        dir = global_matches[i]

        if not os.path.isdir(dir):
            print(f"Warning: Directory {dir} does not exist. Skipping this run.")
        # get meta file
        subfolders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
        subfolder_name = subfolders[0]
        subfolder_path = os.path.join(dir, subfolder_name)

        pattern = os.path.join(subfolder_path, "*meta*")
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
    print(total_trial_length)
  
    return total_trial_length

if __name__ == "__main__":
    session_folder = r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task\rawdata\sub-003_id-2F\ses-01_date-17092025"
    trials_to_include = np.arange(1,7)
    get_trial_length(session_folder, trials_to_include)