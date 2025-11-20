
from utilities.load_and_save_data import load_pickle, save_pickle
import matplotlib.pyplot as plt
import os
import glob
import cv2
import numpy as np
import cv2

def visualize_sinks(derivatives_base):
    """
    Overlays consinks on image

    Args:
        derivatives_base (_type_): _description_
    """
     # Path to rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    sink_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'consink_data_newmethod')
    data = load_pickle('sink_bins', sink_folder)

    x = data['x']
    y = data['y']

    x_new = np.repeat(x, len(y))
    y_new = np.tile(y, len(x))
    pattern = "T*.avi"
    files = glob.glob(os.path.join(rawsession_folder, 'tracking', pattern))

    video_path = files[0]
    cap = cv2.VideoCapture(video_path)

    ret, img = cap.read()   # img will hold the first frame as a NumPy array
    cap.release()
    if not ret:
        print("Failed to read first frame")

    cap.release()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
        ax.imshow(img, cmap="gray")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

   
    ax.scatter(x_new, y_new, color="red", s=15)

    ax.set_aspect("equal")
    ax.set_title("Sink positions")
    
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "sink_pos.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Maze outline figure saved to: {output_path}")

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    visualize_sinks(derivatives_base)