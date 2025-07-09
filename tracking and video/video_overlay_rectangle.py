import numpy as np
import pandas as pd
import cv2

path = r"Z:\Eylon\Trials_dec_24\Data\Data_with_hd\Z1T1_hd.csv"

video_path = r"Z:\Eylon\Trials_dec_24\Videos\Video_Z1_T1_20241220_140042.avi"

def overlay_video_with_head_pos_and_hd(video_path,  output_path):
    # Read the CSV file

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        x = 800
        y = 1700
        range = 90

        # Draw a red square of 5x5 pixels at the head_pos
        cv2.rectangle(frame, (x-range, y-range), (x+range, y+range), (1,1,1), -1)


        x = 1600
        y = 1150
        range = 90

        # Draw a red square of 5x5 pixels at the head_pos
        cv2.rectangle(frame, (x-range, y-range), (x+range, y+range), (1,1,1), -1)

        out.write(frame)
        frame_idx += 1


    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
overlay_video_with_head_pos_and_hd(r"Z:\Eylon\Data\Spatiotemporal_Task\rawdata\sub-002_id-1U\ses-01_date-02072025\tracking\T2_1U_02072025.avi", r"Z:\Eylon\Data\Spatiotemporal_Task\rawdata\sub-002_id-1U\ses-01_date-02072025\tracking\T2_1U_02072025_with_overlay.avi")
