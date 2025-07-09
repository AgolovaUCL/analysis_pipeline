'''
This script is the main pipeline used for the data analysis

'''
import numpy as np
import os
import glob
import time

from preprocessing.spikewrap import run_spikewrap
from preprocessing.get_trial_length import get_trial_length
from unit_features.obtain_waveform import obtain_waveform
from unit_features.plot_wv_and_autocorr import plot_wv_and_autocorr
from preprocessing.find_trial_numbers import find_trial_numbers
from unit_features.postprocessing_spikeinterface import postprocessing_spikeinterface
#from spatial_features.make_spatiotemp_plots import make_spatiotemp_plots

task = 'spatiotemporal' # Tasks may be: 'HCT', 'spatiotemporal', or 'basic processing'
# basic processing preprocesses the data and makes spatial plots

"""
base_path = r'Z:\Eylon\Data\Honeycomb_Maze_Task' # this folder contains the derivates and rawdata folder
subject_number = '001'
session_number = '05'
trial_session_name = 'all_trials' # Derivatives folder will be called this 
trials_to_exclude = []
"""
base_path = r'Z:\Eylon\Data\Spatiotemporal_Task' # this folder contains the derivates and rawdata folder
subject_number = '002'
session_number = '01'
trial_session_name = 'all_trials' # Derivatives folder will be called this 
trials_to_exclude = [6]


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

# Obtaining trial length
#trial_length = get_trial_length(rawsession_folder, trials_to_include)
#formatted_time = time.strftime('%H:%M:%S', time.gmtime(trial_length))
#print(f"Total trial length:      {formatted_time}")

# === Running Spikewrap preprocessing ===
#run_spikewrap(rawsubject_folder, session_name) ## CHANGE TO PASS OUTPUT TO THIS

# Post processing
postprocessing_spikeinterface(derivatives_base)

# === Extract unit_features, make pots and classify === 
# If user_relable = true, change this part so that the user can change it

#plot_wv_and_autocorr(derivatives_base, trial_length)
# obtain_waveform()  ## do we want this as a seperate plot?
# get_waveform_characteristics()
# classify_neurons()

# === Making spatial plots ===
# plot_trajectory()
# make_rate_maps()
# make_hd_plots()

#if task == 'spatiotemporal':
    #make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include, n_trials)