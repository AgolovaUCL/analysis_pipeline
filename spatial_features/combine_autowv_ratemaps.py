
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from typing import Literal
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
import pandas as pd

def combine_autowv_ratemaps(derivatives_base, unit_type: Literal['pyramidal', 'good', 'all']):
    """
    Combines:
      - left: autocorrelogram + waveform
      - right-top: spike count over trials
      - right-bottom: ratemap + head direction
    
    All images are scaled to have the same width on the right column and total equal height.
    Saves to:
      analysis/cell_characteristics/spatial_features/autowv_ratemap_combined/unit_{unit}.png
    """
    
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, 'ephys', "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    if unit_type == 'good':
        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting","sorter_output", "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features","all_units_overview", "pyramidal_units_2D.csv")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
    
    output_folder =os.path.join(derivatives_base, "analysis", "cell_characteristics", "spatial_features",  "autowv_ratemap_combined")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
            
    print("Combining autocorrelogram + waveform plots with ratemap + hd plots")
    print(f"Path to output folder: {output_folder}")
    for unit_id in tqdm(unit_ids):
        try:
            left_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "auto_and_wv", f"unit_{unit_id:03d}.png")
            b_right_path =  os.path.join(derivatives_base, "analysis", "cell_characteristics", "spatial_features",  "ratemaps_and_hd", f"unit_{unit_id}_rm_hd.png")
            t_right_path = os.path.join(derivatives_base,  "analysis", "cell_characteristics", "unit_features", "spikecount_over_trials", f"unit_{unit_id}_sc_over_trials.png")
            # Open images
            left = Image.open(left_path)
            b_right = Image.open(b_right_path)
            t_right = Image.open(t_right_path)
        except:
            print(f"Could not find images for unit {unit_id}, skipping.")
            continue



        total_width = left.width + np.max([b_right.width, t_right.width])
        total_height = np.max([left.height, b_right.height + t_right.height])

        # Create a blank white canvas
        combined = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

        # Paste the images
        combined.paste(left, (0, 0))  # left side
        combined.paste(t_right, (left.width,b_right.height))  # right side
        combined.paste(b_right, (left.width,0)) # right side

        # Save the combined result
        combined.save(os.path.join(output_folder, f"unit_{unit_id:03d}.png"))

