import numpy as np
import run_inference
#from overlay_video_HD import overlay_video_HD
import subprocess
import json
from pathlib import Path


"""
This file is a pipeline to run the spatial analysis (from videos + sleap model --> positional data in .csv format and videos overlayed with HD)

Inputs:
derivatives_base: derivatives folder path (e.g. r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trial")
rawsession_folder: path to folder with raw data
trials_to_include: array with trial number
sleap_env_path: path to SLEAP environment
movement_env_path: path to movement environment
centroid_model_folder: path to centroid model
centered_model_folder: path to centered model


Note: create this environment in conda
conda create -n movement-env -c conda-forge movement napari pyqt
pip install sleap
pip install -U scikit-learn
pip install python-rapidjson
pip install cattrs
pip install tensorflow
"""
# You can find environments by activating it in anaconda prompt and doing echo %CONDA_PREFIX% (Windows)
# For Eylons comp
sleap_env_path = r"C:\Users\eylon\.conda\envs\sleap\python.exe"
movement_env_path = r"C:\Users\eylon\.conda\envs\movement-env\python.exe"

# For Sophia's comp
sleap_env_path = r"C:\Users\Sophia\AppData\Local\anaconda3\envs\sleap\python.exe"
movement_env_path = r"C:\Users\Sophia\AppData\Local\anaconda3\envs\movement-env2\python.exe"

trials_to_include = np.arange(1,11)

derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
centroid_model_folder = r"Z:\Eylon\SLEAP_NEWCAMERA_21072025\models\250731_163959.centroid.n=1377"
centered_model_folder = r"Z:\Eylon\SLEAP_NEWCAMERA_21072025\models\250801_114358.centered_instance.n=1377"


# Running inference
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

# Running movement
this_file = Path(__file__).resolve()
script_path_movement = this_file.parent / "run_movement.py"
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

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

#run_movement(derivatives_base, trials_to_include, frame_rate = 25)
#overlay_video_HD(derivatives_base, rawsession_folder, trials_to_include)