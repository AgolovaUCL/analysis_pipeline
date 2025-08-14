import pandas as pd
import numpy as np

def calculate_headdirection_and_center(filepath, outputpath):
    df = pd.read_csv(filepath)

    def calc_angle(x1, y1, x2, y2):
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            return np.nan
        return np.arctan2(y2 - y1, x2 - x1)  # returns radians

    # Prepare columns for results
    hd_list = []
    hd_ear_list = []
    hd_snoutcenter_list = []
    hd_headcapsnout_list = []
    center_x_list = []
    center_y_list = []

    for _, row in df.iterrows():
        # Head direction (ear baseline)
        hd_ear = calc_angle(
            row['left_ear_x'], row['left_ear_y'],
            row['right_ear_x'], row['right_ear_y']
        )
        # Perpendicular angle (add pi/2)
        hd_ear_perp = (hd_ear + np.pi/2) if not np.isnan(hd_ear) else np.nan

        # Head direction (snout to center)
        hd_headcapcenter = calc_angle(
            row['headcap_x'], row['headcap_y'],
            row['center_x'], row['center_y']
        )

        # Head direction (headcap to snout)
        hd_headcapsnout = calc_angle(
            row['headcap_x'], row['headcap_y'],
            row['snout_x'], row['snout_y']
        )

        # Combine available angles for the mean (circular mean)
        hd_angles = []
        if not np.isnan(hd_ear_perp): hd_angles.append(hd_ear_perp)
        if not np.isnan(hd_headcapcenter): hd_angles.append(hd_headcapcenter)
        if not np.isnan(hd_headcapsnout): hd_angles.append(hd_headcapsnout)

        # Circular mean (avoid wraparound bias)
        if hd_angles:
            hd_mean = np.arctan2(
                np.nanmean(np.sin(hd_angles)),
                np.nanmean(np.cos(hd_angles))
            )
        else:
            hd_mean = np.nan

        # Center as midpoint between ears
        if not (np.isnan(row['left_ear_x']) or np.isnan(row['right_ear_x'])):
            center_x = (row['left_ear_x'] + row['right_ear_x']) / 2.
        else:
            center_x = np.nan

        if not (np.isnan(row['left_ear_y']) or np.isnan(row['right_ear_y'])):
            center_y = (row['left_ear_y'] + row['right_ear_y']) / 2.
        else:
            center_y = np.nan

        hd_list.append(hd_mean)
        hd_ear_list.append(hd_ear_perp)
        hd_snoutcenter_list.append(hd_headcapcenter)
        hd_headcapsnout_list.append(hd_headcapsnout)
        center_x_list.append(center_x)
        center_y_list.append(center_y)

    hd_list = np.rad2deg(hd_list)  # Convert radians to degrees
    # Add results to DataFrame
    df['x'] = center_x_list
    df['y'] = center_y_list
    df['hd'] = hd_list

    # create new df with x, y hd 
    hd_df = pd.DataFrame({
        'x': center_x_list,
        'y': center_y_list,
        'hd': hd_list
    })
    print(f"HD calculated and saved to {outputpath}")
    hd_df.to_csv(outputpath, index=False)

