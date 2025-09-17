import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import cv2
import json

def overlay_maze_image(derivatives_base):
    """
    In the code files, there's an image saved where all maze platforms are up. This function overlays the hex grid
    on top of that image to show the maze layout
    
    Outputs:
        Into derivatives/analysis/maze_overlay:
            maze_overlay.png: image with hex grid overlay
            maze_overlay_params.json: parameters used to create the overlay
    
    """
    # Getting image
    current_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(current_dir)
    image_path = os.path.join(base_dir, "config_files", "camera_image.png")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Output folder
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Parameters that control the overlay
    radius = 4
    hex_side_length = 89# Length of the side of the hexagon
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

    makefig(rotation,hex_side_length, hcoord_translated, vcoord_translated, img, output_path)
    
    # Save parameters to json file
    params = {
        "radius": radius,
        "hex_side_length": hex_side_length,
        "theta_degrees": np.degrees(theta),
        "desired_x": desired_x,
        "desired_y": desired_y,
        "rotation": rotation
    }
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Maze overlay saved to {output_path}")

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
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-02_testHCT\test"
    overlay_maze_image(derivatives_base)