import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from spatial_features.utils.spatial_features_plots import plot_roseplots, add_arm_overlay_roseplot


def get_MRL_data(derivatives_base, path_to_df = None):
    """ Gets the path for the MRL data used, either the path is provided or user provides it"""
    # df path
    if path_to_df is not None:
        df_all = pd.read_csv(path_to_df)
        print(f"Making roseplot from data from {os.path.basename(path_to_df)}")
    else:
        df_path_base = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data')
        df_options = glob.glob(os.path.join(df_path_base, "directional_tuning*.csv"))
        if len(df_options) == 1:
            df_path = df_options[0]
        else:
            print([os.path.basename(f) for f in df_options])
            user_input = input('Please provide the number of the file in the list you would like to look at (starting at 1): ')
            user_input = np.int32(user_input)
            df_path = df_options[user_input - 1]
            print(f"Making roseplot from data from {os.path.basename(df_options[user_input - 1])}")
        df_all = pd.read_csv(df_path)
            
    return df_all

def get_directories(derivatives_base, deg):
    """ Returns directories"""
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    # Dataframe with raised arms
    csv_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.csv'))

    if len(csv_path) > 0:
        behaviour_df = pd.read_csv(csv_path[0], header=None)
    else:
        excel_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.xlsx'))
        if len(excel_path) > 0:
            behaviour_df = pd.read_excel(excel_path[0], header=None)
        else:
            raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')
    
    # Output path: 
    output_folder_plot = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'roseplots')
    if not os.path.exists(output_folder_plot):
        os.makedirs(output_folder_plot)
    output_path_plot = os.path.join(output_folder_plot, f'roseplot_{deg}_degrees.png')
    return behaviour_df, output_path_plot

def get_sum_bin(mean_dir_arr, num_spikes_arr, bin_edges):
    """ Gets the number of spieks in each bin"""
    sum_count_bin = []

    bin_idx = np.digitize(mean_dir_arr, bin_edges) - 1 

    # Count the number of spikes for each element in bin
    for i in range(len(bin_edges)-1):
        indices = np.where(bin_idx == i)
        num_spikes_i = num_spikes_arr[indices]
        sum_count_bin.append(np.sum(num_spikes_i))
    return sum_count_bin
                     
def make_roseplots(derivatives_base, trials_to_include, deg: int, path_to_df = None):
    """
    This code creates the roseplots as visualized for the spatiotemporal analysis. 
    For the location of the arms (N, NE, etc.) is assumes we're using the new camera (the one with 25 frame rate and north on top)

    Input: 
    derivatives_base; path to derivatives folder
    rawsession_folder: path to rawdata folder
    trials_to_include: trials to include in the analysis
    deg: degree that the data was binned into
    path_to_df: path to df containing MRl data (optional)
    """
    
    # Getting df with MRL values for each unit and epoch
    df_all = get_MRL_data(derivatives_base, path_to_df)
    df = df_all[df_all['significant'] == 'sig']

    # Dataframe with raised arms
    behaviour_df, output_path_plot = get_directories(derivatives_base, deg)

    # Direction of arms and their angles
    arms_dir = ["N", "NW", "SW", "S", "SE", "NE"]
    arms_angles_start = [30, 90, 150, 210, 270, 330]
    # Plot: 3 columns for epochs, final column for correct/incorrect
    fig, axs = plt.subplots(len(trials_to_include), 4, figsize = [3*4, 4*len(trials_to_include)], subplot_kw = {'projection': 'polar'})
    num_bins = 24

    for tr in trials_to_include:
        for e in np.arange(1,4):
            
            num_spikes_arr = []
            mean_dir_arr = []


            # filter for this trial and epoch
            filtered_df = df[(df['trial'] == tr) & (df['epoch'] == e)]

            # if any cells were significant this trial epoch
            if len(filtered_df) > 0:
                mean_dir_arr = np.array(filtered_df['mean_direction'])
                num_spikes_arr = np.array(filtered_df['num_spikes'])

                # Binning the data
                counts, bin_edges = np.histogram(mean_dir_arr, bins = num_bins, range = (-180, 180))

                # Finding the bin of each element in the mean dir arr
                sum_count_bin = get_sum_bin(mean_dir_arr, num_spikes_arr, bin_edges)
                
                plot_roseplots(filtered_df,behaviour_df, arms_dir, arms_angles_start, sum_count_bin, bin_edges,e,  tr, axs[tr-1, e-1])

            
            axs[tr-1, e-1].set_title(f" Tr {tr} epoch {e}")

        add_arm_overlay_roseplot(behaviour_df, tr, trials_to_include,  axs[tr-1, 3], fig)

    plt.tight_layout()
    plt.savefig(output_path_plot)
    plt.show()
    print(f"Saved figure to {output_path_plot}")

if __name__ == "__main__":
    trials_to_include = np.arange(1,6)
    print(trials_to_include)
    derivatives_base = r"S:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\rerun_1212"
    make_roseplots(derivatives_base, trials_to_include, 15)
