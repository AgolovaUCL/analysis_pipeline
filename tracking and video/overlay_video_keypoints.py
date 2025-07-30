import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
import os
import glob

def overlay_video_keypoints(derivatives_base, rawsession_folder, trials_to_include):
    combined_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'keypoints_combined')
    if not os.path.exists(combined_data_dir):
        raise FileNotFoundError(f"Uncombined data directory does not exist: {combined_data_dir}")

    video_data_dir = os.path.join(rawsession_folder, "tracking", "video")
    if not os.path.exists(video_data_dir):
        raise FileNotFoundError(f"Video data directory does not exist: {video_data_dir}")
    
    output_dir = os.path.join(derivatives_base, "analysis", "processed_video")
    os.makedirs(output_dir, exist_ok=True)
    for tr in trials_to_include:

        trial_csv_name = f'keypoints_t{tr}.csv'
        trial_csv_path = os.path.join(combined_data_dir, trial_csv_name)
        df = pd.read_csv(trial_csv_path)

        output_path = os.path.join(output_dir, f"t{tr}_keypoints.avi")
        print(f"Output path for video overlay: {output_path}")

        pattern = f"*T{tr}*"
        files = glob.glob(os.path.join(video_data_dir, pattern))

        video_path = files[0]
        print(f"Video path found: {video_path}")
        do_overlay(video_path, df, output_path)


def do_overlay(video_path, df, output_path):
    KEYPOINTS = [
        ("left_ear_x", "left_ear_y"),
        ("right_ear_x", "right_ear_y"),
        ("center_x", "center_y"),
        ("headcap_x", "headcap_y"),
        ("snout_x", "snout_y"),
    ]
    KEYPOINT_COLORS = [
        (0, 0, 255),   # left_ear: Red
        (0, 255, 0),   # right_ear: Green
        (255, 0, 0),   # center: Blue
        (0, 255, 255), # headcap: Yellow
        (255, 0, 255), # snout: Magenta
    ]

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw all keypoints
        for i, (x_col, y_col) in enumerate(KEYPOINTS):
            if frame_idx < len(df) and pd.notna(df.at[frame_idx, x_col]) and pd.notna(df.at[frame_idx, y_col]):
                x = int(df.at[frame_idx, x_col])
                y = int(df.at[frame_idx, y_col])
                cv2.circle(frame, (x, y), 5, KEYPOINT_COLORS[i], -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
