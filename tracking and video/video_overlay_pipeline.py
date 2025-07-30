import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from combine_selected_positions import combine_selected_positions
from calculate_HD import calculate_headdirection_and_center
from overlay_video_HD import overlay_video_HD   
from overlay_video_keypoints import overlay_video_keypoints

def video_overlay_pipeline(derivatives_base, rawsession_folder, trials_to_include):
    """
    Combines data from Bonsai, calculates HD and then overlays it on video
    """
    positional_data_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    uncombined_data_folder = os.path.join(positional_data_folder, 'keypoints_uncombined')
    combined_data_folder = os.path.join(positional_data_folder, 'keypoints_combined')
    videos_folder = os.path.join(rawsession_folder, "tracking", "video")

    if not os.path.exists(combined_data_folder):
        os.makedirs(combined_data_folder)

    # For each trial
    for tr in tqdm(trials_to_include):
        outputpath_keypoints = os.path.join(combined_data_folder, f'keypoints_t{tr}.csv')
        outputpath_XY = os.path.join(positional_data_folder, f'XY_HD_t{tr}.csv')

        # combining keypoints into one csv
        input_folder = os.path.join(uncombined_data_folder, f't{tr}')
        combine_selected_positions(input_folder, outputpath_keypoints)   

        # calculating HD
        calculate_headdirection_and_center(outputpath_keypoints, outputpath_XY)

        # Overlay video with HD
        #overlay_video_HD(derivatives_base, rawsession_folder, [tr])
        overlay_video_keypoints(derivatives_base, rawsession_folder, [tr])
    
    print("Process completed")

derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
trials_to_include = [4]
video_overlay_pipeline(derivatives_base, rawsession_folder, trials_to_include)