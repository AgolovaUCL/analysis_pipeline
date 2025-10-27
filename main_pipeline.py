'''
This script is the main pipeline used for the data analysis
Prior to running, install the environment processing-pipeline (as can be found in environment_processing.yml)

'''

import datetime
from preprocessing.spikewrap import run_spikewrap
from preprocessing.zero_pad_trials import zero_pad_trials
from unit_features.postprocessing_spikeinterface import run_spikeinterface
from preprocessing.get_length_all_trials import get_length_all_trials
from other.append_config import append_config
from other.find_paths import find_paths
# basic processing preprocesses the data and makes spatial plots

# Currently using processing_pipeline, not processing_pipelin2
r"""user = input("Please input user (Sophia: s, Eylon: e): ")

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
    x=float(input("Input trsial to exclude: "))
    trials_to_exclude.append(x)

"""
user = "Sophia"
task = "HCT"
base_path = r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Sophia\analysis_sessions\ses-01_preprocessing\neural_data"
subject_number = "001"
session_number = "01"
trial_session_name = "all_trials"
trial_numbers = [1,2]

# === Finding the subject folder and session name ===
derivatives_base, rawsession_folder, rawsubject_folder, session_name = find_paths(base_path, subject_number, session_number, trial_session_name)

breakpoint()
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
        'trial_numbers': trial_numbers
    }
}
# Adding data to config file
append_config(derivatives_base, config_data)


# === Zero padding trials ===
zero_pad_trials(rawsession_folder)


# === Running Spikewrap preprocessing ===
run_spikewrap(derivatives_base, rawsubject_folder, session_name) ## CHANGE TO PASS OUTPUT TO THIS

# === Post processing ===
trial_length = run_spikeinterface(derivatives_base)

# Obtain length for all of the trials, making a csv out of its
get_length_all_trials(rawsession_folder, trial_numbers)
