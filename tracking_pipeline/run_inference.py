import subprocess
import pathlib
import os
import numpy as np
"""
this is a quick script that allows you to run your Sleap inference on video files in a directory by calling subprocess. 
It will create a new folder in the OUTPUT_FOLDER with the same structure as the ROOT_FOLDER and save the inference results there.
it will save a .slp file but also an .h5 file for further analysis for example with movement (https://movement.neuroinformatics.dev/index.html)


-------
Parameters:
video_folder: the folder where your video files are stored
dest_folder: the folder where you want to save the inference results
centroid_model_folder: folder where centroid model is
centered_model_folder: folder where centered model is


call_inference()
params:
fpath: the path to the video file you want to run inference on
dest_folder: the folder where you want to save the inference results
command_inf: the command to run the inference, you need to change the model paths to your own models
and potentially adjust inference parameters.
when you want to run this script you need to have sleap installed in the environment where you run this script. 
then just open a command line terminal activate the environment and type: python run_inference_on_all.py
--------
"""


def call_inference_on_all(derivatives_base, rawsession_folder, centroid_model_folder, centered_model_folder,  ext=".avi"):
    video_folder = os.path.join(rawsession_folder, 'tracking')
    dest_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'inference_results')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    source_folder = pathlib.Path(video_folder)
    fpaths = list(source_folder.rglob(f"*{ext}"))
    for fpath in fpaths:
        relative_path_parent = fpath.relative_to(source_folder).parent
        dest_path = pathlib.Path(dest_folder) / relative_path_parent
        call_inference(fpath, dest_path, centered_model_folder, centroid_model_folder)
    return fpaths

def call_inference(fpath, dest_folder, centered_model_folder, centroid_model_folder):
    fpath = pathlib.Path(fpath)
    dest_path = dest_folder / f"{fpath.stem}_inference.slp"
    if dest_path.exists():
        print(f"Skipping {fpath} as {dest_path} already exists.")
        return

    if fpath.exists():
        print(f"processing: {fpath}")
        command_inf = f"sleap-track -m {centered_model_folder} -m {centroid_model_folder} -o {dest_path}  --tracking.tracker none {fpath}  "
        print(f"{command_inf}")
        subprocess.call(command_inf, shell=True)
        final_dest_path = dest_folder / f"{fpath.stem}.h5"
        command_conv = f"sleap-convert --format analysis -o {final_dest_path} {dest_path} "
        subprocess.call(command_conv, shell=True)

    else:
        raise FileNotFoundError(f"File {fpath} does not exist. Please check your input.")


trials_to_include = np.arange(1,11)
derivatives_base = r"Z:\Eylon\Data\Spatiotemporal_Task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
rawsession_folder = r"Z:\Eylon\Data\Spatiotemporal_Task\rawdata\sub-003_id_2V\ses-01_date-30072025"
centroid_model_folder = r"Z:\Eylon\SLEAP_NEWCAMERA_21072025\models\250731_163959.centroid.n=1377"
centered_model_folder = r"Z:\Eylon\SLEAP_NEWCAMERA_21072025\models\250801_114358.centered_instance.n=1377"


call_inference_on_all(derivatives_base, rawsession_folder, centroid_model_folder, centered_model_folder,  ext=".avi")

