import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2
def plot_maze_outline(derivatives_base):
    """
    Plots and saves the outer boundary of the honeycomb maze 
    (edges of the outermost platforms), and saves outline coordinates to JSON.
    
    Input:
        derivatives_base (str): Path to derivatives base directory 
                                (e.g. .../derivatives/sub-XXX/.../all_trials)
    """
    # ---- Load parameters ----
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find {params_path}. Run overlay_maze_image() first.")
    print(params_path)
    with open(params_path, 'r') as f:
        params = json.load(f)

    hcoord = np.array(params["hcoord_tr"])
    vcoord = np.array(params["vcoord_tr"])
    side = params["hex_side_length"]
    print(side)
    rotation_deg = params["rotation"]

    # ---- Load background image (optional) ----
    img_path = r"C:\Users\Sophia\Documents\analysis_pipeline\code\config_files\camera_image.png"
    img = cv2.imread(img_path)
    # ---- Compute vertices for all platforms ----
    rotation_rad = np.radians(rotation_deg)
    base_hex = np.array([
        [np.cos(np.radians(0)),   np.sin(np.radians(0))],
        [np.cos(np.radians(60)),  np.sin(np.radians(60))],
        [np.cos(np.radians(120)), np.sin(np.radians(120))],
        [np.cos(np.radians(180)), np.sin(np.radians(180))],
        [np.cos(np.radians(240)), np.sin(np.radians(240))],
        [np.cos(np.radians(300)), np.sin(np.radians(300))]
    ]) * side

    all_vertices = []
    R = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad),  np.cos(rotation_rad)]
    ])

    for x_c, y_c in zip(hcoord, vcoord):
        rotated_hex = base_hex @ R.T
        hex_vertices = rotated_hex + np.array([x_c, y_c])
        all_vertices.append(hex_vertices)

    all_vertices = np.vstack(all_vertices)

    # ---- Compute convex hull ----
    hull = ConvexHull(all_vertices)
    hull_points = np.append(hull.vertices, hull.vertices[0])  # close loop
    outline_x = all_vertices[hull_points, 0].tolist()
    outline_y = all_vertices[hull_points, 1].tolist()

    # ---- Save outline coordinates ----
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    os.makedirs(os.path.dirname(outline_path), exist_ok=True)
    with open(outline_path, "w") as f:
        json.dump({"outline_x": outline_x, "outline_y": outline_y}, f, indent=4)
    print(f"Outline coordinates saved to: {outline_path}")

    # ---- Plot outline ----
    fig, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
        ax.imshow(img, cmap="gray")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    ax.plot(outline_x, outline_y, "r-", lw=2, label="Maze outline")
    ax.scatter(hcoord, vcoord, color="yellow", s=10, label="Platform centers")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Maze Outer Hexagon Outline (Edges)")
    
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Maze outline figure saved to: {output_path}")


if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    plot_maze_outline(derivatives_base)
