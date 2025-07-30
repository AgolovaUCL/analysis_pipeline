import subprocess
import pathlib
import os
#your source directory where the video files are stored
ROOT_FOLDER = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-02_date-03072025\tracking\video"
#your output directory where the inference results will be saved
OUTPUT_FOLDER = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-02_date-03072025\all_trials\analysis\spatial_behav_data\XY_and_HD\inference_results"


"""
this is a quick script that allows you to run your Sleap inference on video files in a directory by calling subprocess. 
It will create a new folder in the OUTPUT_FOLDER with the same structure as the ROOT_FOLDER and save the inference results there.
it will save a .slp file but also an .h5 file for further analysis for example with movement (https://movement.neuroinformatics.dev/index.html)


-------
Parameters:
ROOT_FOLDER: the folder where your video files are stored
OUTPUT_FOLDER: the folder where you want to save the inference results


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


def call_inference_on_all(dest_folder=OUTPUT_FOLDER, ext=".avi"):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    source_folder = pathlib.Path(ROOT_FOLDER)
    fpaths = list(source_folder.rglob(f"*{ext}"))
    for fpath in fpaths:
        relative_path_parent = fpath.relative_to(source_folder).parent
        dest_path = pathlib.Path(dest_folder) / relative_path_parent
        call_inference(fpath, dest_path)
    return fpaths

def call_inference(fpath, dest_folder):
    fpath = pathlib.Path(fpath)
    dest_path = dest_folder / f"{fpath.stem}_inference.slp"
    if dest_path.exists():
        print(f"Skipping {fpath} as {dest_path} already exists.")
        return

    if fpath.exists():
        print(f"processing: {fpath}")
        command_inf = f"sleap-track -m \"Z:\\Eylon\\Spatiotemporal_task\\New Sleap Model 20250709\\models\\250714_121249.centroid.n=552\"  -m \"Z:\\Eylon\\Spatiotemporal_task\\New Sleap Model 20250709\\models\\250714_135712.centered_instance.n=552\"  -o {dest_path}  --tracking.tracker none {fpath}  "
        print(f"{command_inf}")
        subprocess.call(command_inf, shell=True)
        final_dest_path = dest_folder / f"{fpath.stem}.h5"
        command_conv = f"sleap-convert --format analysis -o {final_dest_path} {dest_path} "
        subprocess.call(command_conv, shell=True)

    else:
        raise FileNotFoundError(f"File {fpath} does not exist. Please check your input.")

if __name__ == "__main__":
    call_inference_on_all()