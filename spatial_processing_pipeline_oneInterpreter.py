import numpy as np
from tracking_pipeline.run_movement import run_movement
from tracking_pipeline.overlay_video_HD import overlay_video_HD
from tracking_pipeline.combine_pos_csvs import combine_pos_csvs
from maze_and_platforms.overlay_maze_image_fromVideo import overlay_maze_image
from tracking_pipeline.add_platforms_to_csv import add_platforms_to_csv

r"""
Run this script in your movement environment (movement-env2 for Sophia and )
Inputs:
derivatives_base: derivatives folder path (e.g. r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trial")
rawsession_folder: path to folder with raw data
trials_to_include: array with trial number
centroid_model_folder: path to centroid model
centered_model_folder: path to centered model


Note: create this environment in conda
conda create -n movement-env -c conda-forge movement napari pyqt"""

trials_to_include = np.arange(1,10)
print(trials_to_include)
derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"

# Add platform coordinates to the center positional csv
good_overlay = overlay_maze_image(derivatives_base, rawsession_folder)

# run_movement gives us the xy coordinates and hd 
#run_movement(derivatives_base, trials_to_include)

# overlays the hd on the video
#overlay_video_HD(derivatives_base, rawsession_folder, trials_to_include)

# combines all the positional csvs. Output: XY_HD_alltrials.csv, XY_HD_alltrials_center.csv (gives the center coordinates)
combine_pos_csvs(derivatives_base, trials_to_include)

if good_overlay == 'y':
    # NOTE: THIS CURRENTLY ONLY ACCOUNTS FOR CW PLATFORMS!!! Wait but that doesnt matter right?
    add_platforms_to_csv(derivatives_base)
else:
    print("Overlay parameters were not accepted. Platforms will not be assigned to csv file.")
    print("Adjust parameters in overlay_maze_image_fromVideo function to continue analysis.")
    print("You can use maze_and_platforms\find_hexagon.ipynb for easier finding of the parameters")