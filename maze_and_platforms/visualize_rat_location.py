import numpy as np
import pandas as pd
import os
import json
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from find_platforms import get_platform_number

def visualize_rat_location(ratloc_x, ratloc_y, derivatives_base, rawsession_folder):
    """
    Adds platforms to XY_HD_alltrials_center.csv and saves it as XY_HD_w_platforms.csv

    Input:
    dervitives_base: path to derivatives folder
    
    """
    # Loading hexagon parameters
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    # Getting image
    pattern = "T*.avi"
    files = glob.glob(os.path.join(rawsession_folder, 'tracking', pattern))

    video_path = files[0]
    cap = cv2.VideoCapture(video_path)

    ret, img = cap.read()   # img will hold the first frame as a NumPy array
    cap.release()
    if not ret:
        print("Failed to read first frame")

    cap.release()
    print(params['theta'])
    
    plat = get_platform_number(ratloc_x, ratloc_y,params['hcoord_tr'], params['vcoord_tr'], params['hex_side_length'] )
    print(f"Platform is {plat}")
    makefig_with_xy(ratloc_x, ratloc_y, params['rotation'], params['hex_side_length'], params['hcoord_tr'], params['vcoord_tr'], img)
    
    
def makefig_with_xy(ratloc_x, ratloc_y, angle, radius, hcoord_translated, vcoord_translated, img):# Create the figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Display the image
    ax.imshow(img, cmap='gray')  
    
    
    # Overlay the hexagons
    for i, (x, y) in enumerate(zip(hcoord_translated, vcoord_translated)):
        hex = RegularPolygon((x, y), numVertices=6, radius= radius,
                             orientation=np.radians(angle),  # Rotate hexagons to align with grid
                             facecolor='none', alpha=1, edgecolor='y')
        ax.text(x, y, i + 1, ha='center', va='center', size=10)  # Start numbering from 1
        ax.add_patch(hex)
    
    # Add scatter points for hexagon centers (optional)
    ax.scatter(hcoord_translated, vcoord_translated, alpha=0, c='grey')
    
    # Plot the rat's location
    ax.scatter(ratloc_x, ratloc_y, c='blue', s=100, label='Rat Location')
    
    # Set limits to match the image dimensions
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Flip y-axis for image alignment

    plt.show()


# Function to calculate Cartesian coordinates with scaling
def calculate_cartesian_coords(coord, hex_side_length):
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord


def hex_grid(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

if __name__ == "__main__":
    x = 1308
    y = 1276
    
    derivatives_base = r"C:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    rawsession_folder = r"C:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"

    visualize_rat_location(x, y, derivatives_base, rawsession_folder)