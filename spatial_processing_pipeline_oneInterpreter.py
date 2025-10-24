import numpy as np
from tracking_pipeline.run_movement import run_movement
from tracking_pipeline.overlay_video_HD import overlay_video_HD
from tracking_pipeline.combine_pos_csvs import combine_pos_csvs
from maze_and_platforms.overlay_maze_image_fromVideo import overlay_maze_image
from tracking_pipeline.add_platforms_to_csv import add_platforms_to_csv
from maze_and_platforms.overlay_hexagonal_grid import plot_maze_outline
from spatial_features.make_spatiotemp_plots import make_spatiotemp_plots
from HCT_analysis.get_limits import get_limits
from spatial_features.plot_ratemaps_and_hd import plot_ratemaps_and_hd
#from unit_features.plot_firing_each_epoch import plot_firing_each_epoch
#from unit_features.plot_spikecount_over_trials import plot_spikecount_over_trials
#from spatial_features.get_spatial_features import get_spatial_features
from spatial_features.roseplot import make_roseplots
from spatial_features.HD_across_epoch import sig_across_epochs
from spatial_features.combine_autowv_ratemaps import combine_autowv_ratemaps
import os
from unit_features.plot_spikecount_over_trials import plot_spikecount_over_trials
r"""
Run this script in your movement environment (movement-env2 for Sophia and )
Inputs:
derivatives_base: derivatives folder path (e.g. r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trial")
trials_to_include: array with trial number
centroid_model_folder: path to centroid model
centered_model_folder: path to centered model


Note: create this environment in conda
conda create -n movement-env -c conda-forge movement napari pyqt"""

trials_to_include = np.arange(1,11)
print(trials_to_include)
derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-05_date-19092025\all_trials"
rawsession_folder = derivatives_base.replace("derivatives", "rawdata")
rawsession_folder = os.path.dirname(rawsession_folder)

task = 'hct' # hct or spatiotemp

# Add platform coordinates to the center positional csv
good_overlay, img = overlay_maze_image(derivatives_base, rawsession_folder, method = "video")

if task == 'hct':
    get_limits(derivatives_base)
    
plot_maze_outline(derivatives_base, img)

# run_movement gives us the xy coordinates and hd 
#run_movement(derivatives_base, trials_to_include)

# overlays the hd on the video
#overlay_video_HD(derivatives_base, rawsession_folder, [1])


# combines all the positional csvs. Output: XY_HD_alltrials.csv, XY_HD_alltrials_center.csv (gives the center coordinates)
combine_pos_csvs(derivatives_base, trials_to_include)

# Rate map + hd for each unit
plot_ratemaps_and_hd(derivatives_base, unit_type = 'all')

# Spikecount over time for each unit
#plot_spikecount_over_trials(derivatives_base, 'all', trials_to_include)

breakpoint()
# Combines autocorrelogram + wv, ratemap + hd, and spikecount over tiem map
combine_autowv_ratemaps(derivatives_base, unit_type = 'all')

if good_overlay == 'y' and task == 'hct':
    # adds the platform to each position in alltrials_center.csv
    add_platforms_to_csv(derivatives_base)

elif task == 'spatiotemp':
    plot_firing_each_epoch(derivatives_base, rawsession_folder, trials_to_include)
    degrees_df_path, deg = make_spatiotemp_plots(derivatives_base, rawsession_folder, trials_to_include)
    make_roseplots(derivatives_base, rawsession_folder, trials_to_include, deg, path_to_df = degrees_df_path)
    sig_across_epochs(derivatives_base, trials_to_include) 
else:
    print("Overlay parameters were not accepted. Platforms will not be assigned to csv file.")
    print("Adjust parameters in overlay_maze_image_fromVideo function to continue analysis.")
    print("You can use maze_and_platforms\find_hexagon.ipynb for easier finding of the parameters")