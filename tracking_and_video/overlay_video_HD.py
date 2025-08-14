import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
import os
import glob

def overlay_video_HD(derivatives_base, rawsession_folder, trials_to_include):
    pos_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    if not os.path.exists(pos_data_dir):
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")

    video_data_dir = os.path.join(rawsession_folder, "tracking")
    if not os.path.exists(video_data_dir):
        raise FileNotFoundError(f"Video data directory does not exist: {video_data_dir}")
    
    output_dir = os.path.join(derivatives_base, "analysis", "processed_video")
    os.makedirs(output_dir, exist_ok=True)
    for tr in trials_to_include:

        trial_csv_name = f'XY_HD_t{tr}.csv'
        trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
        df = pd.read_csv(trial_csv_path)

        output_path = os.path.join(output_dir, f"t{tr}_with_HD.avi")
        print(f"Output path for video overlay: {output_path}")

        pattern = f"*T{tr}*"
        files = glob.glob(os.path.join(video_data_dir, pattern))

        video_path = files[0]
        do_overlay(video_path, df, output_path)


def do_overlay(video_path, df, output_path):
    head_pos_x_col = df.iloc[:, 0].to_numpy()
    head_pos_y_col = df.iloc[:, 1].to_numpy()
    hd_col = df.iloc[:, 2].to_numpy()
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(head_pos_x_col) and pd.notna(head_pos_x_col[frame_idx]) and pd.notna(head_pos_y_col[frame_idx]) and pd.notna(hd_col[frame_idx]):
            x = int(head_pos_x_col[frame_idx])
            y = int(head_pos_y_col[frame_idx])
            hd = hd_col[frame_idx]

            # Draw a red square of 5x5 pixels at the head_pos
            size = 5
            cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), (0, 0, 255), -1)

            # Calculate the end point of the arrow
            length = 30  # Length of the arrow
            end_x = int(x + length * np.cos(np.radians(hd)))
            end_y = int(y - length * np.sin(np.radians(hd)))

            # Draw the arrow
            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


"""
derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
trials_to_include = np.arange(1,9)
"""
derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\all_trials"
rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-05_date-18072025"
trials_to_include = np.arange(1,11)

overlay_video_HD(derivatives_base, rawsession_folder, trials_to_include)