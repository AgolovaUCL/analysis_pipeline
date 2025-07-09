import os
import numpy as np
from pathlib import Path


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
    num_trials =  sum(1 for p in Path(ephys_path).iterdir() if p.is_dir())

    trials_to_include = np.arange(1, num_trials + 1)
    if trials_to_exclude is not None:
        trials_to_include = np.setdiff1d(trials_to_include, trials_to_exclude)

    return num_trials, trials_to_include
