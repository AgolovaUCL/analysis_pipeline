'''
This script is the main pipeline used for the data analysis
Prior to running, install the environment processing-pipeline (as can be found in environment.yml)

'''
import numpy as np
import os
import glob
import time
from pathlib import Path

from preprocessing.spikewrap import run_spikewrap
from preprocessing.find_trial_numbers import find_trial_numbers
from unit_features.postprocessing_spikeinterface import run_spikeinterface
from spatial_features.make_spatiotemp_plots import make_spatiotemp_plots
from preprocessing.get_length_all_trials import get_length_all_trials
from unit_features.classify_cells import classify_cells
from unit_features.plot_spikecount_over_trials import plot_spikecount_over_trials
from spatial_features.get_spatial_features import get_spatial_features
from spatial_features.roseplot import make_roseplots
from tracking_and_video.combine_pos_csvs import combine_pos_csvs
from spatial_features.plot_ratemaps_and_hd import plot_ratemaps_and_hd

task = 'spatiotemporal' # Tasks may be: 'HCT', 'spatiotemporal', or 'basic processing'
# basic processing preprocesses the data and makes spatial plots
# For trials_to_exclude, use 1 based indexing!! (so g0 --> t1)
test = 5
if test == 0:
    base_path = r'Z:\Eylon\Data\Honeycomb_Maze_Task' # this folder contains the derivates and rawdata folder
    subject_number = '001'
    session_number = '05'
    trial_session_name = 'all_trials' # Derivatives folder will be called this 
    trials_to_exclude = []
elif test == 1:
    base_path = r'Z:\Eylon\Data\Spatiotemporal_Task' # this folder contains the derivates and rawdata folder
    subject_number = '002'
    session_number = '01'
    trial_session_name = 'all_trials' # Derivatives folder will be called this 
    trials_to_exclude = [6]
elif test ==3:
    base_path = r'Z:\Eylon\Data\Spatiotemporal_Task' # this folder contains the derivates and rawdata folder
    subject_number = '002'
    session_number = '03'
    trial_session_name = 'all_trials' # Derivatives folder will be called this 
    trials_to_exclude = []
elif test == 5:
    base_path = r'D:\Spatiotemporal_task' # this folder contains the derivates and rawdata folder
    subject_number = '002'
    session_number = '05'
    trial_session_name = 'all_trials' # Derivatives folder will be called this 
    trials_to_exclude = []  
else:
    base_path = r'Z:\Eylon\Data\Spatiotemporal_Task' # this folder contains the derivates and rawdata folder
    subject_number = '002'
    session_number = '02'
    trial_session_name = 'all_trials' # Derivatives folder will be called this 
    trials_to_exclude = [1,2]

base_path = input('Please give the base path (up until Spatiotemporal task): ')
base_path = Path(base_path)
subject_number = input('Please provide the subject number (with zeroes. For example: 002): ')
session_number = input('Please provide the number of the session (with zeroes, for example 04): ')
trial_session_name = input('Please provide the name of the trial session: ')
a=int(input("How many trials to exclude?: "))
trials_to_exclude = []
for i in range(a):
    x=float(input("Input trial to exclude: "))
    trials_to_exclude.append(x)

# === Finding the subject folder and session name ===
# === Subject
print("=== Folder names ===")
pattern = os.path.join(base_path, rf"rawdata\sub-{subject_number}*")
matching_folders = glob.glob(pattern)
print(matching_folders)
if matching_folders:
    subject_folder = matching_folders[0]  
    print(f"Subject folder:         {subject_folder}")
else:
    print("No matching folder found.")
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



# Getting the numbers of the trials to include
final_trial_number, trials_to_include    = find_trial_numbers(rawsession_folder, trials_to_exclude)
n_trials = len(trials_to_include)
print("\n=== Other session information ==="
      f"\nTrials that we will use: {trials_to_include}")

# === Running Spikewrap preprocessing ===
run_spikewrap(derivatives_base, rawsubject_folder, session_name) ## CHANGE TO PASS OUTPUT TO THIS

# === Post processing ===
trial_length = run_spikeinterface(derivatives_base)

# Obtain length for all of the trials, making a csv out of its
get_length_all_trials(rawsession_folder, trials_to_include)

# === Classifying neurons ===
classify_cells(derivatives_base) # Note, it does not return a df yet! Still to fix

# === Extract spatial data == 
# Here we get the functions from spatial_processsing_pipeline
# STILL TO ADD

combine_pos_csvs(derivatives_base, trials_to_include)

# === Plot features for neurons and obtain spatial characeristics ===
# Firing rate over time
plot_spikecount_over_trials(derivatives_base, rawsession_folder, trials_to_include)
# TEST THIS FUNCTION AGAIN


# Rate map + head direction
plot_ratemaps_and_hd(derivatives_base)

# Obtain spatial score
#s WRITE CODE STILL 
get_spatial_features(derivatives_base) # DOES NOT DO ANYTHING YET

if task == 'spatiotemporal':
    degrees_df_path, deg = make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include)
    make_roseplots(derivatives_base, rawsession_folder, trials_to_include, deg, path_to_df = degrees_df_path)
    