import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import json

                
def occupancy_each_trial(derivatives_base, trials_to_include,  frame_rate = 25):
    """
    PLots occupancy for each trial

    Args:
        derivatives_base (_type_): _description_
        frame_rate (int, optional): _description_. Defaults to 25.
    """
    # pos data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
    pos_data = pd.read_csv(pos_data_path)


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
    
    print("Making occupancy plots per trial")
    for tr in trials_to_include:  
        fig, ax = plt.subplots(1,5, figsize = (25, 4))
        for goal in [0, 1,2, 3, 4]:
            pos_data_org = pos_data.copy()
            path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")

            intervals_df = pd.read_csv(path)
            if goal < 4: 
                if goal < 3:
                    start_col = 2*goal
                    end_col = 2*goal + 1
                    title = f'Goal {goal}' if goal > 0 else 'Going to G2 during G1'
                    
                elif goal == 3:
                    start_col = 0
                    end_col = 5
                    title = 'Full trial (start to G2)'
                
                # get only start and end col
                intervals_df_restr = intervals_df.iloc[:, [start_col, end_col]]
                
                # Convert to list of tuples
                intervals = (np.int32(intervals_df_restr.iloc[tr-1,0]*frame_rate), np.int32(intervals_df_restr.iloc[tr-1,1]*frame_rate))
                
                # Convert to frame number

                pos_data_org['frame'] = np.arange(1,len(pos_data_org) + 1)

                mask = (pos_data_org['frame'] > intervals[0]) & (pos_data_org['frame'] <intervals[1])
                goal_df = pos_data_org[mask]
            elif goal == 4:
                path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_t{tr}.csv')
                goal_df = pd.read_csv(path)
                title = 'Full trial (full video)'
            x = goal_df['x'].values
            x = x[~np.isnan(x)]
            y = goal_df['y'].values
            y = y[~np.isnan(y)]


            heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=30, range=[[xmin, xmax], [ymin, ymax]])
            
            im = ax[goal].imshow(
                        heatmap_data.T,
                        cmap='viridis',
                        interpolation='none',
                        origin='upper',
                        extent=[xmin, xmax, ymax, ymin]
                    )
            fig.colorbar(im, ax=ax[goal], label='Seconds')
            ax[goal].set_title(title)
            ax[goal].set_aspect('equal')
            if outline_x is not None and outline_y is not None:
                ax[goal].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        plt.tight_layout()
        plt.title(f'Trial {tr}')
        output_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data',"occupancy_heatmaps")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, f'trial_{tr}_occupancy.png')
        plt.savefig(output_path)
        plt.close(fig)
    print(f"Plots saved to {output_folder}")
            
                
                
            
import matplotlib.animation as animation
import cv2

def make_occupancy_video(derivatives_base, rawsession_folder, trial_num, frame_rate=25, step_size=50):
    """
    Creates a video showing how occupancy builds up over time during a given trial.

    Args:
        derivatives_base (str): Path to derivatives base folder
        rawsession_folder (str): Path to raw session folder
        trial_num (int): Trial index (1-based, as in your CSV)
        frame_rate (int): Frame rate of recording (Hz)
        step_size (int): Number of frames to add per animation step
    """

    # --- Load position data ---
    pos_data_path = os.path.join(
        derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv'
    )
    pos_data = pd.read_csv(pos_data_path)

    # --- Load maze limits ---
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
    xmin, xmax, ymin, ymax = limits['xmin'], limits['xmax'], limits['ymin'], limits['ymax']

    # --- Load maze outline (optional) ---
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    if os.path.exists(outline_path):
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        outline_x, outline_y = None, None
        print(" Maze outline JSON not found; skipping red outline overlay.")

    # --- Load interval for full trial (goal 3) ---
    intervals_path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")
    intervals_df = pd.read_csv(intervals_path)
    start_frame = int(intervals_df.iloc[trial_num - 1, 0] * frame_rate)
    end_frame = int(intervals_df.iloc[trial_num - 1, 5] * frame_rate)

    pos_data['frame'] = np.arange(1, len(pos_data) + 1)
    mask = (pos_data['frame'] >= start_frame) & (pos_data['frame'] <= end_frame)
    df_trial = pos_data.loc[mask]

    x = df_trial['x'].dropna().values
    y = df_trial['y'].dropna().values
    n_frames = len(x)

    # --- Prepare video writer ---
    output_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'occupancy_videos')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"occupancy_trial{trial_num}.mp4")

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.close(fig)

    # --- Define update function for animation ---
    def update(frame_idx):
        ax.clear()
        curr_x = x[:frame_idx]
        curr_y = y[:frame_idx]
        heatmap_data, xbins, ybins = np.histogram2d(curr_x, curr_y, bins=30, range=[[xmin, xmax], [ymin, ymax]])
        heatmap_data = np.ma.masked_where(heatmap_data == 0, heatmap_data)

        im = ax.imshow(
            heatmap_data.T,
            cmap='viridis',
            interpolation='none',
            origin='lower',
            extent=[xmin, xmax, ymin, ymax]
        )

        if outline_x is not None and outline_y is not None:
            ax.plot(outline_x, outline_y, 'r-', lw=2)

        ax.set_title(f"Trial {trial_num} — Frame {frame_idx}/{n_frames}", fontsize=12)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        return [im]

    # --- Create animation ---
    n_steps = n_frames // step_size
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(step_size, n_frames, step_size),
        blit=True, repeat=False
    )

    ani.save(output_path, writer='ffmpeg', fps=10, dpi=150)
    print(f"✅ Occupancy video saved to:\n{output_path}")            
            
    


    
if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base= r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
    pos_data = pd.read_csv(pos_data_path)


    occupancy_each_trial(derivatives_base, trials_to_include = np.arange(1,10))
   