
import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap_xy(derivatives_base, trials_to_include, frame_rate = 25):
    """
    Makes a heatmap of the xy position of the animal for each trial. 
    This is used to check whether all xy positions are correct

    saved in spatial_behav_data/position_heatmaps
    Args:
        derivatives_base (_type_): _description_
    """
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    # Input and output folders
    input_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    output_folder = os.path.join(derivatives_base,'analysis', 'spatial_behav_data', 'position_heatmaps')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pos_data_all_csv = os.path.join(input_folder, 'XY_HD_alltrials_center.csv')
    pos_data_all = pd.read_csv(pos_data_all_csv)
    
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
        
        
    for tr in trials_to_include:
        fig, axs = plt.subplots(1, 3, figsize = [18, 6])
        axs = axs.flatten()
        csv_name = f'XY_HD_center_t{tr}.csv'
        pos_data = pd.read_csv(os.path.join(input_folder, csv_name))
        
        x = pos_data['x'].values
        x = x[~pd.isna(x)]
        y = pos_data['y'].values
        y = y[~pd.isna(y)]
        
        for g in [0, 1, 2]:
            ax = axs[g]
            if g == 0: # Meaning full trial
                csv_name = f'XY_HD_center_t{tr}.csv'
                pos_data = pd.read_csv(os.path.join(input_folder, csv_name))
                
                x = pos_data['x'].values
                x = x[~pd.isna(x)]
                y = pos_data['y'].values
                y = y[~pd.isna(y)]
                title = 'Full trial'
                heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=20, range=[[xmin, xmax], [ymin, ymax]])
                heatmap_data = heatmap_data/frame_rate
            else:
                path = os.path.join(rawsession_folder, "task_metadata", f"goal_{g}_intervals.csv")
        
                intervals_df = pd.read_csv(path)
                intervals = list(zip(intervals_df['start_time'], intervals_df['end_time']))

                # Convert to frame number
                intervals = [(int(start * frame_rate), int(end * frame_rate)) for start, end in intervals]

                pos_data_all['frame'] = np.arange(1,len(pos_data_all) + 1)
                mask = np.zeros(len(pos_data_all), dtype=bool)
                start, end = intervals[tr-1]
                mask[start:end] = True
                pos_data_all_tr = pos_data_all[mask]
                x = pos_data_all_tr['x'].values
                x = x[~pd.isna(x)]
                y = pos_data_all_tr['y'].values
                y = y[~pd.isna(y)]
                title = f'Goal {g}'
                heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=20, range=[[xmin, xmax], [ymin, ymax]])
                heatmap_data = heatmap_data/frame_rate
            
            im = ax.imshow(
                heatmap_data.T,
                cmap='viridis',
                interpolation=None,
                origin='upper',
                aspect='auto',
                extent=[xmin, xmax, ymax, ymin]
            )
            fig.colorbar(im, ax=ax, label='Seconds')
            if outline_x is not None and outline_y is not None:
                ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
            ax.set_title(title)
            ax.set_aspect('equal')
        plt.savefig(os.path.join(output_folder, f'position_heatmap_trial{tr}.png'), dpi=300)
        plt.close(fig)
    
    # Lastly plot all trials together, with subplot 1 and 2 also split for goal
    fig, axs = plt.subplots(1, 3, figsize = [18, 6])
    axs = axs.flatten()
    
    for g in [0, 1, 2]:
        
        ax = axs[g]
        if g == 0: # Meaning full trial
            x = pos_data_all['x'].values
            x = x[~pd.isna(x)]
            y = pos_data_all['y'].values
            y = y[~pd.isna(y)]
            title = 'All trials'
        else:
            path = os.path.join(rawsession_folder, "task_metadata", f"goal_{g}_intervals.csv")
        
            intervals_df = pd.read_csv(path)
            intervals = list(zip(intervals_df['start_time'], intervals_df['end_time']))

            # Convert to frame number
            intervals = [(int(start * frame_rate), int(end * frame_rate)) for start, end in intervals]
            mask = np.zeros(len(pos_data_all), dtype=bool)
            for start, end in intervals:
                mask[start:end] = True
            pos_data_masked = pos_data_all[mask]
            x = pos_data_masked['x'].values
            x = x[~pd.isna(x)]
            y = pos_data_masked['y'].values
            y = y[~pd.isna(y)]
            title = f'Goal {g}'
        heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=20, range=[[xmin, xmax], [ymin, ymax]])
        heatmap_data = heatmap_data/frame_rate
        
        im = ax.imshow(
            heatmap_data.T,
            cmap='viridis',
            interpolation=None,
            origin='upper',
            aspect='auto',
            extent=[xmin, xmax, ymax, ymin]
        )
        fig.colorbar(im, ax=ax, label='Seconds')
        if outline_x is not None and outline_y is not None:
            ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
        ax.set_title(title)
        ax.set_aspect('equal')
    plt.savefig(os.path.join(output_folder, 'position_heatmap_alltrials.png'), dpi=300)
    plt.close(fig)
    print(f"Saved heatmaps to {output_folder}")
    
    
def plot_heatmap_pos_data(pos_data, frame_rate = 25):
    """
    Quick code to do the same based on the pos data. I use it to test pos data g1 and g2
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
        

  
    # Lastly plot all trials together, with subplot 1 and 2 also split for goal
    fig, ax = plt.subplots(1, 1, figsize = [18, 6])
    
    x = pos_data['x'].values
    x = x[~pd.isna(x)]
    y = pos_data['y'].values
    y = y[~pd.isna(y)]

    heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=20, range=[[xmin, xmax], [ymin, ymax]])
    heatmap_data = heatmap_data/frame_rate
    
    im = ax.imshow(
        heatmap_data.T,
        cmap='viridis',
        interpolation=None,
        origin='upper',
        aspect='auto',
        extent=[xmin, xmax, ymax, ymin]
    )
    fig.colorbar(im, ax=ax, label='Seconds')
    if outline_x is not None and outline_y is not None:
        ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    ax.set_title('Pos data')
    ax.set_aspect('equal')
    plt.show()
              
        
if __name__ == "__main__":
    derivatives_base= r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    trials_to_include = np.arange(1,10)
    #plot_heatmap_xy(derivatives_base, trials_to_include)
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_goal2_trials.csv')
    pos_data_g2 = pd.read_csv(pos_data_path)
    print(len(pos_data_g2))

    plot_heatmap_pos_data(pos_data_g2)