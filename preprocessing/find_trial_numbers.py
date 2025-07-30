import os
import numpy as np
from pathlib import Path
import re

def find_trial_numbers(session_folder, trials_to_exclude):
    """
    Function finds the numbers of the trials we'll use.
    It finds the total number of trials by countin the amount of folders in the rawsession_folder directory
    If there are folder that are not run-..., they may cause issues

    Input:
    session_folder: Path to the session folder containing ephys data
    trials_to_exclude: List of trial numbers to exclude from the analysis

    Output:
    num_trials: Total number of trials found in the session folder
    trials_to_include: Array of trial numbers to include in the analysis
    """

    ephys_path = os.path.join(session_folder, 'ephys')

    # List all entries in the ephys_path directory
    folders = os.listdir(ephys_path)

    # Pattern to match: ends with '_g' followed by 1 or 2 digits (e.g., folder_g5, session_g13)
    pattern = re.compile(r'_g(\d{1,2})$')

    numbers = []
    for folder in folders:
        match = pattern.search(folder)
        if match:
            numbers.append(int(match.group(1)))

    if numbers:
        largest = max(numbers)
        print(f"Largest number: {largest}")
    else:
        print("No matching folders found.")
    last_trial = largest + 1
    trials_to_include = np.arange(1, last_trial + 1 )
    if trials_to_exclude is not None:
        trials_to_include = np.setdiff1d(trials_to_include, trials_to_exclude)

    return len(trials_to_include), trials_to_include
