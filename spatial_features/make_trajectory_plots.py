import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def make_trajectory_plots(derivatives_base, trials_to_include):

    pos_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    if not os.path.exists(pos_data_dir):
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")

    output_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'animal_trajectory')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "trajectory_overview.png")

    n_trials = len(trials_to_include)
    n_rows = np.min([n_trials, 4])
    n_cols = int(np.ceil(n_trials / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize =[n_cols * 4, n_rows * 4])
    axs = axs.flatten()
    plt.suptitle("Animal trajectory throughout trials")

    for i, tr in enumerate(trials_to_include):
        # get trial data for trial i
        trial_csv_name = f'XY_HD_t{tr}.csv'
        trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
        xy_hd_trial = pd.read_csv(trial_csv_path)            
        x = xy_hd_trial.iloc[:, 0].to_numpy()
        y = xy_hd_trial.iloc[:, 1].to_numpy()

        # add to plot
        axs[i].scatter(x, y, c = 'black')
        axs[i].set_title(f"Trial {i}")
    plt.savefig(output_path)
    print(f"Trajectory overview saved in {output_path}")
    plt.close(fig)

derivatives_base = r"Z:\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-001_id-2B\ses-05_test\all_trials"
trials_to_include = np.array([1,2])
make_trajectory_plots(derivatives_base, trials_to_include)
