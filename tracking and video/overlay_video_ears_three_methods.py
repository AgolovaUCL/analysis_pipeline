import os
import glob
import pandas as pd
import cv2

def overlay_video_ears_different_methods(derivatives_base, rawsession_folder, trials_to_include):
    """
    For each trial in `trials_to_include`, finds all CSVs in
    `<derivatives_base>/analysis/spatial_behav_data/XY_and_HD/t{trial}_tests/`,
    loads the positional data (x, y, left ear, right ear),
    locates the corresponding raw video in `rawsession_folder/tracking/video/*T{trial}*.avi`,
    and writes out one overlaid AVI per CSV into
    `<derivatives_base>/analysis/processed_video/{csv_basename}.avi`.
    
    Parameters
    ----------
    derivatives_base : str
        Path to the derivatives folder containing the `analysis/spatial_behav_data` tree.
    rawsession_folder : str
        Path to the raw session root, under which `tracking/video/` lives.
    trials_to_include : iterable of int
        Trial numbers to process (e.g. range(1,9)).
    """
    pos_data_dir   = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    video_data_dir = os.path.join(rawsession_folder, "tracking", "video")
    output_dir     = os.path.join(derivatives_base, "analysis", "processed_video")
    os.makedirs(output_dir, exist_ok=True)

    for tr in trials_to_include:
        # 1) find all CSVs for this trial
        csv_folder = os.path.join(pos_data_dir, f"t{tr}_tests")
        csv_paths  = glob.glob(os.path.join(csv_folder, "*.csv"))
        if not csv_paths:
            print(f"⚠️  No CSVs found in {csv_folder}, skipping trial {tr}")
            continue

        # 2) find the one matching video for this trial
        vid_pattern = f"*T{tr}*"
        vid_files   = glob.glob(os.path.join(video_data_dir, vid_pattern))
        if not vid_files:
            print(f"⚠️  No video found for trial {tr} with pattern {vid_pattern}")
            continue
        video_path = vid_files[0]
        print(f"-- Trial {tr}: overlaying onto {os.path.basename(video_path)}")

        # 3) loop over each CSV
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)

            csv_name    = os.path.splitext(os.path.basename(csv_path))[0]
            output_path = os.path.join(output_dir, f"{csv_name}.avi")
            print(f"    → {csv_name} → {output_path}")

            do_overlay_ears(video_path, df, output_path)


def do_overlay_ears(video_path, df, output_path):
    # now columns are: [0]=ignore, [1]=x, [2]=y, [3]=x_le, [4]=y_le, [5]=x_re, [6]=y_re
    x_col      = df.iloc[:, 1].to_numpy()
    y_col      = df.iloc[:, 2].to_numpy()
    x_left_ear = df.iloc[:, 3].to_numpy()
    y_left_ear = df.iloc[:, 4].to_numpy()
    x_right_ear= df.iloc[:, 5].to_numpy()
    y_right_ear= df.iloc[:, 6].to_numpy()

    cap   = cv2.VideoCapture(video_path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc= cv2.VideoWriter_fourcc(*'XVID')
    out   = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #max_length = len(x_col)
        max_length = 900
        if frame_idx >= max_length:
            break
    
        if frame_idx < max_length and pd.notna(x_col[frame_idx]) and pd.notna(y_col[frame_idx]):
            x  = int(x_col[frame_idx])
            y  = int(y_col[frame_idx])

            # head = red, ears = orange
            cv2.rectangle(frame, (x-5, y-5),   (x+5, y+5),   (0,0,255),     -1)
                # draw right ear only if both coords are finite
            xr_val, yr_val = x_right_ear[frame_idx], y_right_ear[frame_idx]
            if pd.notna(xr_val) and pd.notna(yr_val):
                xr, yr = int(xr_val), int(yr_val)
                cv2.rectangle(frame, (xr-5, yr-5), (xr+5, yr+5), (0, 165, 255), -1)

            # draw left ear only if both coords are finite
            xl_val, yl_val = x_left_ear[frame_idx], y_left_ear[frame_idx]
            if pd.notna(xl_val) and pd.notna(yl_val):
                xl, yl = int(xl_val), int(yl_val)
                cv2.rectangle(frame, (xl-5, yl-5), (xl+5, yl+5), (0, 165, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-01_date-02072025"
    trials_to_include = [1]
    overlay_video_ears_different_methods(derivatives_base, rawsession_folder, trials_to_include)
