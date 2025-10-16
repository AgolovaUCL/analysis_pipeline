import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import cv2
import json
import glob

def overlay_maze_image(derivatives_base, rawsession_folder):
    """
    In the code files, there's an image saved where all maze platforms are up. This function overlays the hex grid
    on top of that image to show the maze layout
    
    Outputs:
        Into derivatives/analysis/maze_overlay:
            maze_overlay.png: image with hex grid overlay
            maze_overlay_params.json: parameters used to create the overlay
    
    """
    # Getting image
    """
    pattern = "T*.avi"
    files = glob.glob(os.path.join(rawsession_folder, 'tracking', pattern))

    video_path = files[0]
    cap = cv2.VideoCapture(video_path)

    ret, img = cap.read()   # img will hold the first frame as a NumPy array
    cap.release()
    if not ret:
        print("Failed to read first frame")

    cap.release()
    """
    img_path = r"C:\Users\Sophia\Documents\analysis_pipeline\code\config_files\camera_image.png"
    img = cv2.imread(img_path)
    
    # Output folder
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Parameters that control the overlay
    radius = 4
    hex_side_length = 88# Length of the side of the hexagon
    theta = np.radians(305)  # Rotation angle in radians
    desired_x, desired_y = 1320, 950
    coord = hex_grid(radius)
    rotation = 25
    
    # Calculate initial Cartesian coordinates
    hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)


    # Rotate the coordinates
    hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [-v for v in vcoord_rotated]


    # Calculate the translation needed to align the first rotated coordinate
    dx = desired_x - hcoord_rotated[30]
    dy = desired_y - vcoord_rotated[30]

    # Apply the translation
    hcoord_translated = [x + dx for x in hcoord_rotated]
    vcoord_translated = [y + dy for y in vcoord_rotated]

    good_overlay = makefig(rotation,hex_side_length, hcoord_translated, vcoord_translated, img, output_path)
    
    # Save parameters to json file
    params = {
        "radius": radius,
        "hex_side_length": hex_side_length,
        "theta": theta,
        "x_center": desired_x,
        "y_center": desired_y,
        "rotation": rotation,
        "hcoord_tr": hcoord_translated,
        "vcoord_tr": vcoord_translated
    }

    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
        print(f"Parameters saved to: {params_path}")
    print(f"Maze overlay saved to {output_path}")
    
    if good_overlay == 'y':
        print("Overlay approved, continuing all operations")
    else:
        print("Overlay not approved. Adjust parameters in overlay_maze_image_fromVideo function")
        print('Spatial processing pipeline will assign platforms to positional csvs.\n ')
    return good_overlay

def makefig(angle, radius, hcoord_translated, vcoord_translated, img, output_path):# Create the figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Display the image
    ax.imshow(img, cmap='gray')  
    
    
    # Overlay the hexagons
    for i, (x, y) in enumerate(zip(hcoord_translated, vcoord_translated)):
        hex = RegularPolygon((x, y), numVertices=6, radius=radius,
                             orientation=np.radians(angle),  # Rotate hexagons to align with grid
                             facecolor='none', alpha=1, edgecolor='y')
        ax.text(x, y, i + 1, ha='center', va='center', size=10)  # Start numbering from 1
        ax.add_patch(hex)
    
    # Add scatter points for hexagon centers (optional)
    ax.scatter(hcoord_translated, vcoord_translated, alpha=0, c='grey')
    
    # Set limits to match the image dimensions
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Flip y-axis for image alignment
    plt.savefig(output_path)
    plt.show()
    good_overlay = input('Enter whether overlay is good (y) or not (n): ')
    while good_overlay not in ['y', 'n']:
        print("Please input y or n")
        good_overlay = input('Enter input: ')
    plt.close()
    return good_overlay

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
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    overlay_maze_image(derivatives_base, rawsession_folder)