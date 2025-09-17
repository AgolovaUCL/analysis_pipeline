import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib

def plot_startplatforms(derivatives_base, rawsession_folder, goal_platforms):
    """
    Shows a maze plot with all the start platforms

    Exports:
        start_platforms.png into {derivatives_base}/analysis/maze_behaviour
    """
    
    path = os.path.join(rawsession_folder, "behaviour", "concatenated_trials.csv")
    df = pd.read_csv(path)
    
    trial_numbers = df['trial_number'].unique()
    
    start_platforms = []
    for t in trial_numbers:
        df_trial = df[df['trial_number'] == t]
        start_platform = df_trial['start platform'].iloc[0]
        start_platforms.append(start_platform)
        
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.set_aspect('equal')

    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()
    # Add some coloured hexagons and adjust the orientation to match the rotated grid
    for i, (x, y) in enumerate(zip(hcoord_rotated, vcoord_rotated)):
        if i + 1 in start_platforms:
            colour = "red"
        elif i + 1 == goal_platforms[0]:
            colour = "green"
        elif i + 1 == goal_platforms[1]:
            colour = "blue"
        else:
            colour = "grey"
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                            orientation=np.radians(60),  # Rotate hexagons to align with grid
                            facecolor=colour, alpha=0.2, edgecolor='k')
        ax.text(x, y, i + 1,  ha='center', va='center', size=15)  # Start numbering from 1
        ax.add_patch(hex)
    start_patch = mpatches.Patch(color="red", alpha=0.2, label="Starts")
    goal_patch = mpatches.Patch(color="green", alpha=0.2, label="Goal 1")
    goal2_patch = mpatches.Patch(color="blue", alpha=0.2, label="Goal 2")

    ax.legend(handles=[start_patch, goal_patch, goal2_patch], loc="upper right")

        # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
    ax.set_title('Start platforms (red) and goal platforms (green)')
    output_folder = os.path.join(derivatives_base, "analysis", "maze_behaviour")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, "start_platforms.png"), dpi=300)
    plt.show()

    print(f"File saved to output folder: {output_folder}")
    

def plot_occupancy(derivatives_base, rawsession_folder, goal_platforms):
    """
    Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) and all trials (right)
    low: red, high: green

    Exports:
        occupancy.png into {derivatives_base}/analysis/maze_behaviour
    """
    
    path = os.path.join(rawsession_folder, "behaviour", "concatenated_trials.csv")
    df = pd.read_csv(path)

    # 3 x 1 subplot
    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    ax = ax.flatten()

    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()
    
    cmap = matplotlib.colormaps['RdYlGn']
    
    # Go through goals and all trials
    for g in [0,1,2]: # g = 0: goal 1, g = 1: goal 2, g = 2: all trials
        if g == 0:
            df_g = df[df['goal'] == goal_platforms[0]]
            title = "Goal 1"
        elif g == 1:
            df_g = df[df['goal'] == goal_platforms[1]]
            title = "Goal 2"
        else:
            df_g = df
            title = "Full trials"

        occupancy = []
        
        for plat in np.arange(1, 62):
            df_p = df_g[df_g['start platform'] == plat]
            occupancy.append(len(df_p))

        occupancy_norm = occupancy/np.max(occupancy)
        
        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord_rotated, vcoord_rotated)):
            colour = cmap(occupancy_norm[i])  # Map normalized occupancy to colormap
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                orientation=np.radians(60),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor='k')
            ax[g].text(x, y, occupancy[i],  ha='center', va='center', size=15)  # Start numbering from 1
            ax[g].add_patch(hex)
            # Also add scatter points in hexagon centres
        ax[g].scatter(hcoord, vcoord, alpha=0, c = 'grey')
        ax[g].set_title(title)
        ax[g].set_aspect('equal')
    output_folder = os.path.join(derivatives_base, "analysis", "maze_behaviour")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, "occupancy.png"), dpi=300)
    plt.show()

    print(f"File saved to output folder: {output_folder}")

def plot_propcorrect(derivatives_base, rawsession_folder, goal_platforms):
    """
    Shows a maze plot with the proportion of correct choices for goal 1 (left) goal 2 (middle) and all trials (right)

    Exports:
        proportion_correct.png into {derivatives_base}/analysis/maze_behaviour
    """
    
    path = os.path.join(rawsession_folder, "behaviour", "concatenated_trials.csv")
    df = pd.read_csv(path)

    # 3 x 1 subplot
    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    ax = ax.flatten()

    # coordinates
    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()
    
    # Colour map
    cmap = matplotlib.colormaps['RdYlGn']
    
    # Go through goals and all trials
    for g in [0,1,2]: # g = 0: goal 1, g = 1: goal 2, g = 2: all trials
        if g == 0:
            df_g = df[df['goal'] == goal_platforms[0]]
            title = "Goal 1"
        elif g == 1:
            df_g = df[df['goal'] == goal_platforms[1]]
            title = "Goal 2"
        else:
            df_g = df
            title = "Full trials"

        prop_correct_arr = []
        
        for plat in np.arange(1, 62):
            df_p = df_g[df_g['start platform'] == plat]
            num_rows = len(df_p)
            num_correct = len(df_p[df_p['correct choice'] == 1])
            if num_rows > 0:
                prop_correct = num_correct / num_rows
                prop_correct_arr.append(prop_correct)
            else:
                prop_correct_arr.append(np.nan)

        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord_rotated, vcoord_rotated)):
            if np.isnan(prop_correct_arr[i]):
                colour = 'grey'
                text = ""
            else:
                colour = cmap(prop_correct_arr[i])  # Map normalized occupancy to colormap
                text = np.round(prop_correct_arr[i], 1)
                
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                orientation=np.radians(60),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor='k')
            ax[g].text(x, y, text,  ha='center', va='center', size=15)  # Start numbering from 1
            ax[g].add_patch(hex)
            # Also add scatter points in hexagon centres
        ax[g].scatter(hcoord, vcoord, alpha=0, c = 'grey')
        ax[g].set_title(title)
        ax[g].set_aspect('equal')
    output_folder = os.path.join(derivatives_base, "analysis", "maze_behaviour")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, "proportion_correct.png"), dpi=300)
    plt.show()

    print(f"File saved to output folder: {output_folder}") 

        
def get_coords():   
    # Generate coordinates for a large hexagon with radius 4
    radius = 4
    coord = hex_grid(radius)

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartesian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord]

    theta = np.radians(270)

    # Rotate the coordinates
    hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord, vcoord)]
    vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord, vcoord)]
            
    return hcoord, vcoord, hcoord_rotated, vcoord_rotated
    
def hex_grid(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-02_testHCT\test"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-02_date-05092025"
    plot_propcorrect(derivatives_base, rawsession_folder, [48, 29])