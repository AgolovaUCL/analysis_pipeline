import os
import numpy as np
import pandas as pd

# Define base path
base_path = r'Z:\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-001_id-2B\ses-05_test\all_trials'  # <--- CHANGE THIS to your base path

kilosort_output_path = os.path.join(base_path, 'ephys', 'concat_run', 'sorting', 'shank_0', 'sorter_output')
xy_path = os.path.join(base_path, 'analysis', 'spatial_behav_data', 'XY_and_HD')
plot_path = os.path.join(base_path, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'rate_maps')

# Make sure xy_path exists
os.makedirs(xy_path, exist_ok=True)

# Generate random xy positions
num_samples = 1000  # Change this as needed
x = np.random.uniform(0, 100, num_samples)  # Example: random x between 0 and 100
y = np.random.uniform(0, 100, num_samples)  # Example: random y between 0 and 100
hd = np.random.uniform(0, 360, num_samples)


# Create DataFrame
xy_df = pd.DataFrame({'x': x, 'y': y, 'hd': hd})

# Save to CSV
trial_num = 2
csv_filename = f'XY_HD_t{trial_num}.csv'
csv_path = os.path.join(xy_path, csv_filename)
xy_df.to_csv(csv_path, index=False)

print(f"Saved random xy positions to {csv_path}")
