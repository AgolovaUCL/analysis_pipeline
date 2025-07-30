import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
import os
import glob

def overlay_video_centroid(derivatives_base, rawsession_folder, trials_to_include, path_to_centroid):
    combined_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'keypoints_combined')
    if not os.path.exists(combined_data_dir):
        raise FileNotFoundError(f"Uncombined data directory does not exist: {combined_data_dir}")

    video_data_dir = os.path.join(rawsession_folder, "tracking", "video")
    if not os.path.exists(video_data_dir):
        raise FileNotFoundError(f"Video data directory does not exist: {video_data_dir}")
    
    output_dir = os.path.join(derivatives_base, "analysis", "processed_video")
    os.makedirs(output_dir, exist_ok=True)
    for tr in trials_to_include:

        df = pd.read_csv(path_to_centroid)

        output_path = os.path.join(output_dir, f"t{tr}_centroid.avi")
        print(f"Output path for video overlay: {output_path}")

        pattern = f"*T{tr}*"
        files = glob.glob(os.path.join(video_data_dir, pattern))

        video_path = files[0]
        print(f"Video path found: {video_path}")
        do_overlay(video_path, df, output_path)


def do_overlay(video_path, df, output_path):
    head_pos_x_col = df.iloc[:, 0].to_numpy()
    head_pos_y_col = df.iloc[:, 1].to_numpy()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(head_pos_x_col) and pd.notna(head_pos_x_col[frame_idx]) and pd.notna(head_pos_y_col[frame_idx]):
            x = int(head_pos_x_col[frame_idx])
            y = int(head_pos_y_col[frame_idx])


            # Draw a red square of 5x5 pixels at the head_pos
            size = 5
            cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), (0, 0, 255), -1)


        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

overlay_video_centroid(derivatives_base=r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials",
                       rawsession_folder=r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025",
                       trials_to_include=[5],
                       path_to_centroid = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials\analysis\spatial_behav_data\XY_and_HD\XY_HD_t5_centroid.csv")