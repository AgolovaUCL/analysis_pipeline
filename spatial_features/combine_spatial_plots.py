
from PIL import Image
import numpy as np
# Paths to your 3 plots
import os
from tqdm import tqdm

for i in tqdm(range(1, 439)):
    
    
    try:
        left_path = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials\analysis\cell_characteristics\unit_features\auto_and_wv\all\unit_{i:03d}.png".format(i=i)
        top_right_path = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials\analysis\cell_characteristics\spatial_features\vector_fields\vector_fields_unit_{i}.png".format(i=i)
        bottom_right_path = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials\analysis\cell_characteristics\spatial_features\ratemaps_and_hd\unit_{i}_rm_hd.png".format(i=i)
        # Open images
        left = Image.open(left_path)
        top_right = Image.open(top_right_path)
        bottom_right = Image.open(bottom_right_path)
    except:
        print(f"Could not find images for unit {i}, skipping.")
        continue

    # Open images
    left = Image.open(left_path)
    top_right = Image.open(top_right_path)
    bottom_right = Image.open(bottom_right_path)

    # Scale top-right image to match bottom-right width
    if top_right.width != bottom_right.width:
        scale_factor = bottom_right.width / top_right.width
        new_height = int(top_right.height * scale_factor)
        top_right = top_right.resize((bottom_right.width, new_height))

    if left.height != (bottom_right.height + top_right.height):
        target_height = bottom_right.height + top_right.height
        scale_factor = target_height / left.height
        new_width = int(left.width * scale_factor)
        new_height = target_height
        left = left.resize((new_width, new_height))



        
    total_width = max(left.width + top_right.width, left.width + bottom_right.width)
    height = max(left.height, top_right.height + bottom_right.height)

    # Create a blank white canvas
    combined = Image.new("RGB", (total_width, height), color=(255, 255, 255))

    # Paste the images
    combined.paste(left, (0, 0))  # left side
    combined.paste(top_right, (left.width, 0))  # top right
    combined.paste(bottom_right, (left.width, top_right.height))  # bottom right

    # Save the combined result
    output_folder = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials\analysis\cell_characteristics\spatial_features\combined_plots"
    combined.save(os.path.join(output_folder, f"combined_unit_{i}.png"))

