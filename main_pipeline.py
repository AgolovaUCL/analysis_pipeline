'''
This script is the main pipeline used for the data analysis
Prior to running, install the environment processing-pipeline (as can be found in environment.yml)

'''
import numpy as np
import os
import subprocess
import json
import datetime

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
import tracking_pipeline.run_inference as run_inference
from tracking_pipeline.overlay_video_HD import overlay_video_HD
from unit_features.plot_firing_each_epoch import plot_firing_each_epoch
from other.append_config import append_config
from other.find_paths import find_paths
from spatial_features.HD_across_epoch import sig_across_epochs
# basic processing preprocesses the data and makes spatial plots
# For trials_to_exclude, use 1 based indexing!! (so g0 --> t1)

# Currently using processing_pipeline, not processing_pipelin2
user = input("Please input user (Sophia: s, Eylon: e): ")

if user == 's' or user == 'S' or user == 'sophia' or user == 'Sophia':
    sleap_env_path = r"C:\Users\Sophia\AppData\Local\anaconda3\envs\sleap\python.exe"
    movement_env_path = r"C:\Users\Sophia\AppData\Local\anaconda3\envs\movement-env2\python.exe"
elif user == 'e' or user == 'E' or user == 'eylon' or user == 'Eylon':
    sleap_env_path = r"C:\Users\Eylon\.conda\envs\sleap\python.exe"
    movement_env_path = r"C:\Users\Eylon\.conda\envs\movement-env\python.exe"
else:
    raise ValueError("Please input a valid user (Sophia or Eylon) or add user to code")


task_input = input('Please give the task (h: HCT, s: spatiotemporal): ')
if task_input == 'h' or task_input == 'H' or task_input == 'hct' or task_input == 'HCT':
    task = 'HCT'
elif task_input == 's' or task_input == 'S' or task_input == 'spatiotemporal' or task_input == 'Spatiotemporal':
    task = 'spatiotemporal'

base_path = input(r'Please give the base path (for example, D:\Spatiotemporal_task): ')
base_path = Path(base_path)
subject_number = input('Please provide the subject number (with zeroes. For example: 002): ')
session_number = input('Please provide the number of the session (with zeroes, for example 04): ')
trial_session_name = input('Please provide the name of the trial session (for example, all_trials): ')
a=int(input("How many trials to exclude?: "))
trials_to_exclude = []
for i in range(a):
    x=float(input("Input trial to exclude: "))
    trials_to_exclude.append(x)


centroid_model_folder = r"Z:\Eylon\SLEAP_NEWCAMERA_21072025\models\250731_163959.centroid.n=1377"
centered_model_folder = r"Z:\Eylon\SLEAP_NEWCAMERA_21072025\models\250801_114358.centered_instance.n=1377"

if not os.path.exists(centroid_model_folder) or not os.path.exists(centered_model_folder):
    raise FileNotFoundError("Centered or centroid model folder does not exist. Append within code")


# === Finding the subject folder and session name ===
derivatives_base, rawsession_folder, rawsubject_folder, session_name = find_paths(base_path, subject_number, session_number, trial_session_name)

# === Adding data to config file ===
config_data = {
    'inputs': {
        'name': user,
        'date': str(datetime.date.today()),
        'time': str(datetime.datetime.now().time()),
        'task': task,
        'base_path': str(base_path),
        'subject_number': subject_number,
        'session_number': session_number,
        'trial_session_name': trial_session_name,
        'trials_to_exclude': list(trials_to_exclude),
    }
}
append_config(derivatives_base, config_data)

# Getting the numbers of the trials to include
final_trial_number, trials_to_include    = find_trial_numbers(rawsession_folder, trials_to_exclude)
n_trials = len(trials_to_include)
print("\n=== Other session information ==="
      f"\nTrials that we will use: {trials_to_include}")

append_config(derivatives_base, {'session': {'trials_included': [int(x) for x in trials_to_include]}})

# === Running Spikewrap preprocessing ===
run_spikewrap(derivatives_base, rawsubject_folder, session_name) ## CHANGE TO PASS OUTPUT TO THIS

breakpoint()
# === Post processing ===
trial_length = run_spikeinterface(derivatives_base)

# Obtain length for all of the trials, making a csv out of its
get_length_all_trials(rawsession_folder, trials_to_include)

# === Classifying neurons ===
classify_cells(derivatives_base, analyse_only_good=True)

# === Extract spatial data == 
# Running inference using SLEAP environment
script_path = run_inference.__file__
data_dict = {
    "derivatives_base": derivatives_base,
    "rawsession_folder": rawsession_folder,
    "centroid_model_folder": centroid_model_folder,
    "centered_model_folder": centered_model_folder
}
result = subprocess.run(
    [sleap_env_path, script_path],
    input=json.dumps(data_dict),
    capture_output=True,
    text=True,
    check=True
)
print(result.stdout)  

# Running movement using movement environment
this_file = Path(__file__).resolve()
script_path_movement = this_file.parent / "tracking_pipeline" / "run_movement.py"
script_path_movement = str(script_path_movement)


data_dict = {
    "derivatives_base": derivatives_base,
    "trials_to_include": trials_to_include.tolist()
}

result = subprocess.run(
    [movement_env_path, script_path_movement],
    input=json.dumps(data_dict),
    capture_output=True,
    text=True
)
# Overlay the videos with the head direction
overlay_video_HD(derivatives_base, rawsession_folder, trials_to_include)

# Combining all positional csv to get csv for all trials
combine_pos_csvs(derivatives_base, trials_to_include)

# === Plot features for neurons and obtain spatial characeristics ===
# Firing rate over time
plot_spikecount_over_trials(derivatives_base, rawsession_folder, trials_to_include)

# Rate map + head direction
plot_ratemaps_and_hd(derivatives_base)

# Obtain spatial score
get_spatial_features(derivatives_base) # DOES NOT DO ANYTHING YET

if task == 'spatiotemporal':
    plot_firing_each_epoch(derivatives_base, rawsession_folder, trials_to_include)
    degrees_df_path, deg = make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include)
    make_roseplots(derivatives_base, rawsession_folder, trials_to_include, deg, path_to_df = degrees_df_path)
    sig_across_epochs(derivatives_base, trials_to_include)
    