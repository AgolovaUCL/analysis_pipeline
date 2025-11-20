import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2

def get_image(rawsession_folder, method="video"): 
    """ Gets image as either first video frame from the tracking folder 
    or the image from the code\config_files\image.png """
    
    # Getting image
    if method == "video":
        # Here it loads the first frame of the first video in the tracking folder
        pattern = "T*.avi"
        files = glob.glob(os.path.join(rawsession_folder, 'tracking', pattern)) # finds matches
        video_path = files[0] # takes first video
        
        # reads video
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()   
        if not ret:
            print("Failed to read first frame")
        cap.release()
    elif method == "image":
        cwd = os.getcwd()
        #cwd = os.path.dirname(cwd) #Uncomment if running script from this code
        config_folder = os.path.join(cwd, "config_files")
        img_path = os.path.join(config_folder, "camera_image.png")
        if not os.path.exists(img_path):
            raise FileExistsError("Img path not found")
        img = cv2.imread(img_path)
    else:
        raise ValueError("Method not image or video. PLease provide valid input")
    return img

def get_params(derivatives_base):
    """ Loads parameters from maze_overlay_params.json"""
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find {params_path}. Run overlay_maze_image() first.")

    with open(params_path, 'r') as f:
        params = json.load(f)

    hcoord = np.array(params["hcoord_tr"])
    vcoord = np.array(params["vcoord_tr"])
    side = params["hex_side_length"]
    rotation_deg = params["rotation"]
    return hcoord, vcoord, side, rotation_deg

def get_vertices(hcoord, vcoord, rotation_deg, side):
    """ Computes the 6 vertices location for all of the platforms """

    # This creates an array that the location of the 6 vertices for each platform, scaled to each edge is length side
    base_hex = np.array([
        [np.cos(np.radians(0)),   np.sin(np.radians(0))],
        [np.cos(np.radians(60)),  np.sin(np.radians(60))],
        [np.cos(np.radians(120)), np.sin(np.radians(120))],
        [np.cos(np.radians(180)), np.sin(np.radians(180))],
        [np.cos(np.radians(240)), np.sin(np.radians(240))],
        [np.cos(np.radians(300)), np.sin(np.radians(300))]
    ]) * side

    all_vertices = []
    rotation_rad = np.radians(rotation_deg)
    
    # This is the rotation matrix, to rotate the base_hex by rotation_rad
    R = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad),  np.cos(rotation_rad)]
    ])
    
    # To each center point of the platform, we add the location of the vertices
    for x_c, y_c in zip(hcoord, vcoord):
        rotated_hex = base_hex @ R.T
        hex_vertices = rotated_hex + np.array([x_c, y_c])
        all_vertices.append(hex_vertices)

    all_vertices = np.vstack(all_vertices)
    return all_vertices

def get_outline(all_vertices):
    # ---- Compute convex hull ----
    hull = ConvexHull(all_vertices)
    hull_points = np.append(hull.vertices, hull.vertices[0])  # close loop
    outline_x = all_vertices[hull_points, 0].tolist()
    outline_y = all_vertices[hull_points, 1].tolist()
    return outline_x, outline_y

def save_outline(derivatives_base, outline_x, outline_y):
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    os.makedirs(os.path.dirname(outline_path), exist_ok=True)
    with open(outline_path, "w") as f:
        json.dump({"outline_x": outline_x, "outline_y": outline_y}, f, indent=4)
    print(f"Outline coordinates saved to: {outline_path}")

def plot_outline(derivatives_base,img,  outline_x, outline_y, hcoord, vcoord):
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
    
def plot_maze_outline(derivatives_base, img = None, method = "video"):
    """
    Plots and saves the outer boundary of the honeycomb maze 
    (edges of the outermost platforms), and saves outline coordinates to JSON.
    
    Input:
        derivatives_base (str): Path to derivatives base directory 
                                (e.g. .../derivatives/sub-XXX/.../all_trials)
    
    Outputs:
        derivatives_base\analysis\maze_overlay\maze_outline_coords.json: outline coordinates for maze
        derivatives_base\analysis\maze_overlay\maze_outline.png: image with maze outline visualised
    """
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Loading data
    hcoord, vcoord, side, rotation_deg = get_params(derivatives_base)

    if img is None:
        img = get_image(rawsession_folder, method = method)
        
    # Location of all vertices
    all_vertices = get_vertices(hcoord, vcoord, rotation_deg, side)

    # get outline
    outline_x, outline_y = get_outline(all_vertices)
    
    # saving
    save_outline(derivatives_base, outline_x, outline_y)

    # plotting 
    plot_outline(derivatives_base, img, outline_x, outline_y, hcoord, vcoord)

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    plot_maze_outline(derivatives_base)
