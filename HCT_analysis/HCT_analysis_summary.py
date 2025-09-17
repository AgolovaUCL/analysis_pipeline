import numpy as np
import os
import sys
from utilities.concat_trials import concat_trials
from utilities.create_intervals import create_intervals_df
from utilities.trials_utils import append_alltrials, get_goal_numbers
from plotting.make_maze_plots import plot_occupancy, plot_propcorrect, plot_startplatforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from display_maze.overlay_maze_image import overlay_maze_image
# Running HCT analysis
def run_HCT(rawsession_folder, derivatives_base):
    """
    Combination of all functions needed for HCT analysis
    
    Maze plots can be found in {derivatives_base}/analysis/maze_behaviour
    """
    # overlay maze image
    overlay_maze_image(derivatives_base)
    # Concatenates all trial files in the form {ratID}_{date}_g{trial}.csv into one df
    concat_trials(rawsession_folder)
    
    # Takes the alltrials csv and creates a new csv only with the rows of the trial date
    append_alltrials(rawsession_folder)
    
    # Find goal numbers
    goal_platforms = get_goal_numbers (rawsession_folder)
    
    # CFinds the start and stop time for each trial, so that we can restrict spike times to for each goal
    # NOTE Is based on the labview data, which isn't synced with the spike data yet.
    create_intervals_df(rawsession_folder)
    
    ############ Plotting ###########
    # Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_occupancy(derivatives_base, rawsession_folder, goal_platforms)
    
    # Shows a maze plot with all the start platforms
    plot_startplatforms(derivatives_base, rawsession_folder, goal_platforms)

    # Shows a maze plot with the proportion of correct choices for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_propcorrect(derivatives_base, rawsession_folder, goal_platforms)
    ################################
    
    