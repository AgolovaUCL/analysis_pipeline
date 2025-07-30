import numpy as np
import pandas as pd
import cv2
import os
import glob

def overlay_video_HD_threemethods(derivatives_base, rawsession_folder, trials_to_include):
    """
    For each trial in `trials_to_include`, loads XY+HD data from
    `<derivatives_base>/analysis/spatial_behav_data/XY_and_HD/XY_HD_t{trial}_full.csv`,
    finds the matching raw video under `rawsession_folder/tracking/video/*T{trial}*.avi`,
    and writes a single AVI with all three HD methods overlaid in different colours.
    """
    pos_data_dir   = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    video_data_dir = os.path.join(rawsession_folder, "tracking", "video")
    output_dir     = os.path.join(derivatives_base, "analysis", "processed_video")
    os.makedirs(output_dir, exist_ok=True)

    for tr in trials_to_include:
        csv_path = os.path.join(pos_data_dir, f'XY_HD_t{tr}_full.csv')
        df = pd.read_csv(csv_path)

        # find matching video
        vid_files = glob.glob(os.path.join(video_data_dir, f"*T{tr}*.avi"))
        if not vid_files:
            print(f"⚠️ No video for trial {tr}")
            continue
        video_path = vid_files[0]

        output_path = os.path.join(output_dir, f"t{tr}_allMethods_with_HD.avi")
        print(f"→ Writing {output_path}")
        do_overlay_all_methods(video_path, df, output_path)


def do_overlay_all_methods(video_path, df, output_path):
    """
    Overlays head-position + three HD arrows per frame.
    DataFrame columns:
      0: head_x,  1: head_y,
      2: hd_method1, 3: hd_method2, 4: hd_method3, ...
    """
    # head x/y
    x_col = df.iloc[:,0].to_numpy()
    y_col = df.iloc[:,1].to_numpy()
    # the three HD series (in radians)
    hd_cols = [df.iloc[:,2].to_numpy(),
               df.iloc[:,3].to_numpy(),
               df.iloc[:,4].to_numpy()]
    # choose three distinct colours (BGR)
    colours = [(0,255,0),   # method1 = green
               (255,0,0),   # method2 = blue
               (0,165,255)]   # method3 = orange

    cap    = cv2.VideoCapture(video_path)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

    frame_idx = 0
    max_frames = 2*30*60  # or whatever limit you like
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        # draw head square
        if frame_idx < len(x_col) and pd.notna(x_col[frame_idx]) and pd.notna(y_col[frame_idx]):
            x, y = int(x_col[frame_idx]), int(y_col[frame_idx])
            cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0,0,255), -1)

            # draw each HD arrow
            length = 30
            for hd_arr, col in zip(hd_cols, colours):
                hd = hd_arr[frame_idx]
                if pd.notna(hd):2
                    hd_rot = -hd
                    end_x = int(x + length * np.cos(np.radians(hd_rot)))
                    end_y = int(y - length * np.sin(np.radians(hd_rot)))
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), col, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    t = input('Provide the trial number  ')
    t = int(t)
    if t == 1:
        derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
        rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
        trials_to_include = range(1,9)
    elif t == 2:
        derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-02_date-03072025\all_trials"
        rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-02_date-03072025"
        trials_to_include = range(4,10)

    overlay_video_HD_threemethods(derivatives_base, rawsession_folder, trials_to_include)