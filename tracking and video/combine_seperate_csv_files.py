import os
import pandas as pd

def combine_selected_positions(input_folder, output_path):
    clusters = ['left_ear', 'right_ear', 'center', 'headcap', 'snout']
    axes = ['x', 'y']
    columns = [f"{cluster}_{axis}" for cluster in clusters for axis in axes]
    
    series_list = []
    for col in columns:
        fpath = os.path.join(input_folder, f"{col}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")
        # Read as a single-column CSV (with or without header)
        # Try to detect header: if first row is not a number, assume header
        with open(fpath) as f:
            first_line = f.readline().strip()
        if not first_line.replace('.', '', 1).replace('-', '', 1).isdigit():
            s = pd.read_csv(fpath).iloc[:, 0]
        else:
            s = pd.read_csv(fpath, header=None).iloc[:, 0]
        series_list.append(s.rename(col))
    
    # Concatenate horizontally
    combined = pd.concat(series_list, axis=1)
    
    # Ensure output folder exists
    combined.to_csv(output_path, index=False)
    print(f"Combined CSV saved to {output_path}")

