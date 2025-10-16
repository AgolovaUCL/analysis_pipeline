import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import json
def restrict_posdata(pos_data, rawsession_folder,  frame_rate = 25):
    """
    Restricts the pos_data to the intervals of the goal.
    
    Args:
        pos_data (DataFrame): position data
        rawsession_folder (str): path to the raw session folder
        goal (int): goal number (1 or 2)

    Returns:
        DataFrame: restricted position data
    """

    # Limits
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    
    xmin = limits['xmin']
    xmax = limits['xmax']
    ymin = limits['ymin']
    ymax = limits['ymax']
    
    # ---- Load maze outline coordinates ----
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    if os.path.exists(outline_path):
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print("⚠️ Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None
        
    for goal in [1,2]:
        pos_data_org = pos_data.copy()
        path = os.path.join(rawsession_folder, "task_metadata", f"goal_{goal}_intervals.csv")
        
        intervals_df = pd.read_csv(path)

        # Convert to list of tuples
        intervals = list(zip(intervals_df['start_time'], intervals_df['end_time']))

        # Convert to frame number
        intervals = [(int(start * frame_rate), int(end * frame_rate)) for start, end in intervals]

        pos_data_org['frame'] = np.arange(1,len(pos_data_org) + 1)
        mask = np.zeros(len(pos_data), dtype=bool)
        for start, end in intervals:
            mask |= (pos_data_org['frame'] >= start) & (pos_data_org['frame'] <= end)
        pos_data_org = pos_data_org[mask]
        
        print(f"Len dataframe for goal {goal}: {len(pos_data_org)}")
        output_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{goal}_trials.csv')
        pos_data_org.to_csv(output_path, index=False)
        print(f"Saved restricted position data for goal {goal} to {output_path}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax = ax.flatten()
    
    for i, goal in enumerate([1,2, 3]):
        if goal < 3:
            goal_path = os.path.join(
                derivatives_base,
                'analysis',
                'spatial_behav_data',
                'XY_and_HD',
                f'XY_HD_goal{goal}_trials.csv'
            )
            goal_df = pd.read_csv(goal_path)
            title = f'Goal {goal}'
        else:
            goal_df = pos_data
            title = 'All trials'
            
        x = goal_df['x'].values
        x = x[~np.isnan(x)]
        y = goal_df['y'].values
        y = y[~np.isnan(y)]


        heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins = 20)
        im = ax[i].imshow(
                heatmap_data.T,
                cmap='viridis',
                interpolation=None,
                origin='upper',
                aspect='auto',
                extent=[xmin, xmax, ymax, ymin]
            )
        fig.colorbar(im, ax=ax[i], label='Seconds')
        ax[i].set_title(title)
        ax[i].set_aspect('equal')
        if outline_x is not None and outline_y is not None:
            ax[i].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    plt.tight_layout()
    plt.show()
                

    
if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base= r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
    pos_data = pd.read_csv(pos_data_path)

    restrict_posdata(pos_data, rawsession_folder)