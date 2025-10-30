import os
import numpy as np
import pandas as pd
from matplotlib.path import Path
import matplotlib.pyplot as plt
import json
from add_platforms_any_df import add_platforms_to_csv
def restrict_posdata_specialbehav(pos_data, derivatives_base, rawsession_folder,  frame_rate = 25):
    """
    Restricts the pos_data to the intervals of the goal.
    
    Args:
        pos_data (DataFrame): position data
        rawsession_folder (str): path to the raw session folder
        goal (int): goal number (0, 1 or 2): NOTE, 0 here stands for rat going to g2 durin g1

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
        print(" Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None
        
    for goal in [0, 1,2]:
        pos_data_org = pos_data.copy()
        
        
        path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")

        intervals_df = pd.read_csv(path)

        start_col = 2*goal
        end_col = 2*goal + 1
        
        # get only start and end col
        intervals_df_restr = intervals_df.iloc[:, start_col:end_col + 1]
        
        # Convert to list of tuples
        intervals = list(zip(intervals_df_restr.iloc[:,0], intervals_df_restr.iloc[:,1]))
        
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
        print(len(pos_data_org))
        print(f"Saved restricted position data for goal {goal} to {output_path}")

    # Saving all into one df
    df_restricted_all =  pd.DataFrame(columns=['x', 'y', 'hd'])
    for goal in [0,1,2]:
        df_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{goal}_trials.csv')
        df_goal = pd.read_csv(df_path)
        df_restricted_all = pd.concat([df_restricted_all, df_goal])
        df_restricted_all = df_restricted_all.sort_values(by='frame').reset_index(drop=True)

    print("Adding platforms to interval df")
    df_restricted_all = add_platforms_to_csv(df_restricted_all, derivatives_base)
    output_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_allintervals.csv')
    df_restricted_all.to_csv(output_path, index = False)
    
    fig, ax = plt.subplots(1, 5, figsize=(30, 4))
    ax = ax.flatten()
    
    for i, goal in enumerate([0,1,2, 3, 4]):
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
        elif goal == 3:
            goal_df = pos_data
            title = 'All trials'
        else:
            goal_df = df_restricted_all
            title = 'All trials, only during intervals'
            
        x = goal_df['x'].values
        x = x[~np.isnan(x)]
        y = goal_df['y'].values
        y = y[~np.isnan(y)]


        heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=30, range=[[xmin, xmax], [ymin, ymax]])
        
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
    output_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data',"occupancy_heatmaps")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, 'session_overview.png')
    plt.savefig(output_path)
    plt.show()
                



    
if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base= r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_w_platforms.csv')
    pos_data = pd.read_csv(pos_data_path)

    restrict_posdata_specialbehav(pos_data, derivatives_base, rawsession_folder)
    
        