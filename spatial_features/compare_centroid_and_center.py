import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set trial number here
trial_num = 4 # Change this to any trial number you want
trial_name = '4_only_center_4' # Change this to any trial number you want

# Build paths using the trial number
base_dir = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials\analysis\spatial_behav_data\XY_and_HD"
centroid_path = os.path.join(base_dir, f"XY_HD_t{trial_num}_centroid.csv")
center_x_path = os.path.join(base_dir, "keypoints_uncombined", f"t{trial_name}", "center_x.csv")
center_y_path = os.path.join(base_dir, "keypoints_uncombined", f"t{trial_name}", "center_y.csv")


# Load data
centroid_df = pd.read_csv(centroid_path)
center_x_df = pd.read_csv(center_x_path)
center_y_df = pd.read_csv(center_y_path)

# Combine x and y into a single DataFrame
combined_df = pd.DataFrame({
    'x' : center_x_df['center_x'],
    'y' : center_y_df['center_y']
})

# Round x and y to whole numbers (keeping NaN)
combined_df['x'] = combined_df['x'].round().astype('Int64')
combined_df['y'] = combined_df['y'].round().astype('Int64')
centroid_df['x'] = centroid_df['x'].round().astype('Int64')
centroid_df['y'] = centroid_df['y'].round().astype('Int64')

# Compute differences
diff_df = pd.DataFrame({
    'x_diff': combined_df['x'] - centroid_df['x'],
    'y_diff': combined_df['y'] - centroid_df['y']
})

# Get first 5000 values
x_diff_first_5000 = diff_df['x_diff'][:5000]
y_diff_first_5000 = diff_df['y_diff'][:5000]

# Plotting all in one figure
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 1. Differences over time
axs[0, 0].plot(diff_df['x_diff'], label='X Difference', color='red')
axs[0, 0].plot(diff_df['y_diff'], label='Y Difference', color='blue')
axs[0, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[0, 0].set_title('Differences between Center and Centroid Coordinates')
axs[0, 0].set_xlabel('Time (frames)')
axs[0, 0].set_ylabel('Difference (pixels)')
axs[0, 0].legend()

# 2. First 5000 differences
axs[0, 1].plot(x_diff_first_5000, label='X Difference (first 5000)', color='red')
axs[0, 1].plot(y_diff_first_5000, label='Y Difference (first 5000)', color='blue')
axs[0, 1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[0, 1].set_title('First 5000 frames')
axs[0, 1].set_xlabel('Time (frames)')
axs[0, 1].set_ylabel('Difference (pixels)')
axs[0, 1].legend()

# 3. X coordinates over time
axs[1, 0].plot(centroid_df['x'], label='Centroid X', color='blue')
axs[1, 0].plot(combined_df['x'], label='Center X', color='red')
axs[1, 0].set_title('X Coordinates over Time')
axs[1, 0].set_xlabel('Time (frames)')
axs[1, 0].set_ylabel('X Coordinate (pixels)')
axs[1, 0].legend()

# 4. Y coordinates over time
axs[1, 1].plot(centroid_df['y'], label='Centroid Y', color='blue')
axs[1, 1].plot(combined_df['y'], label='Center Y', color='red')
axs[1, 1].set_title('Y Coordinates over Time')
axs[1, 1].set_xlabel('Time (frames)')
axs[1, 1].set_ylabel('Y Coordinate (pixels)')
axs[1, 1].legend()

plt.suptitle(f"Spatial Data Analysis: Trial {trial_num}", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

for i, j in enumerate(x_diff_first_5000):
    if j > 5 or j < -5:
        print(i)
    break

