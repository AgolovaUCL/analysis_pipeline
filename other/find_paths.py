import numpy as np
import os
import glob

def find_paths(base_path, subject_number, session_number, trial_session_name):
    """
    Finds the paths to the raw session folder and derivatives folder
    Args:
        base_path: path to the folder with the data (for example, Z:\Eylon\Data\Spatiotemporal_Task)
        subject_number: subject number (for example, 002)
        session_number: session number (for example, 01)
        trial_session_name: trial session name (for example, all_trials)

    Raises:
        FileNotFoundError: If the required folders are not found

    Returns:
        rawsession_folder: path to the raw session folder
        derivatives_base: path to the derivatives folder
        rawsubject_folder: path to the raw subject folder
        session_name: name of the session
    """
    pattern = os.path.join(base_path, rf"rawdata\sub-{subject_number}*")
    matching_folders = glob.glob(pattern)
    if matching_folders:
        subject_folder = matching_folders[0]  
        print(f"Subject folder:         {subject_folder}")
    else:
        raise FileNotFoundError(f"No subject folder found for pattern {pattern}")
    rawsubject_folder = subject_folder  # Full path to the subject folder

    # === Session
    session_pattern = os.path.join(subject_folder, rf"ses-{session_number}*")
    matching_sessions = glob.glob(session_pattern)

    if matching_sessions:
        session_name = os.path.basename(matching_sessions[0])
        rawsession_folder = matching_sessions[0]  # Full path
        print(f"Session folder:         {rawsession_folder}")
    else:
        raise FileNotFoundError(f"No session folder found for pattern {session_pattern}")
    
    # === Derivates folder
    subject_name = os.path.basename(subject_folder)

    derivatives_base = os.path.join(base_path, 'derivatives', subject_name, session_name, trial_session_name)
    if not os.path.exists(derivatives_base):
        os.makedirs(derivatives_base)

    print("Derivatives folder:      ", derivatives_base)

    return  derivatives_base, rawsession_folder, rawsubject_folder, session_name