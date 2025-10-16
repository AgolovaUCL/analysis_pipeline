import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
import json 

def get_limits(derivatives_base, rawsession_folder):
    """
    This function gets the limits for field on which the consinks will be placed.

    Args:
        derivatives_base: Path to derivatives folder
        rawsession_folder: Path to rawsession folder
        
    Creates:
        derivatives_base\analysis\maze_overlay\limits.png - limits overlayed on maze image
        derivatives_base\analysis\maze_overlay\limits.json - json with limits
        
    """
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

    # Output folder
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    good_limits = False
 
    # Default parameters
    xmin = 491
    xmax = 2176
    ymin = 194
    ymax = 1702   
    while not good_limits:
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Display the image
        ax.imshow(img, cmap='gray')  
        
        ax.vlines(x = xmin, ymin = ymin, ymax = ymax, color = 'r')
        ax.vlines(x = xmax, ymin = ymin, ymax = ymax, color = 'r')
        ax.hlines(y=ymin, xmin=xmin, xmax=xmax,  color='r')
        ax.hlines(y=ymax, xmin=xmin, xmax=xmax,  color='r')
        
        print("If you want to define new limits, click on the top left and then the bottom right of it")
        print("If you're happy with the limits, press escape")
        clicked = plt.waitforbuttonpress(timeout=-1)  # -1 → wait indefinitely
        
        if clicked: 
            plt.savefig(output_path)
            plt.close(fig)
            good_limits = True
            print("ESC pressed — keeping current limits.")
            break

        xy_coordinates = plt.ginput(2, timeout=0)  # timeout=0 → wait indefinitely
        (xmin, ymin), (xmax, ymax) = xy_coordinates
        print(f"New limits: xmin={xmin:.1f}, xmax={xmax:.1f}, ymin={ymin:.1f}, ymax={ymax:.1f}")
        plt.close()
            
    if xmin > xmax:
        xmin_temp = xmin
        xmin = xmax
        xmax = xmin_temp
    if ymin > ymax:
        ymin_temp = ymin
        ymin = ymax
        ymax = ymin_temp
    print(f"Final limits: xmin={xmin:.1f}, xmax={xmax:.1f}, ymin={ymin:.1f}, ymax={ymax:.1f}")
        
    limits= {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax
    }

    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    os.makedirs(os.path.dirname(limits_path), exist_ok=True)
    with open(limits_path, 'w') as f:
        json.dump(limits, f, indent=4)
        
    print(f"Limits saved to {limits_path}")

        

        
        
        
        

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"    
    get_limits(derivatives_base, rawsession_folder)