import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, get_directions_to_position, get_relative_directions_to_position
from calculate_occupancy import get_relative_direction_occupancy_by_position, concatenate_dlc_data, get_axes_limits, calculate_frame_durations, get_direction_bins, bin_directions
from utilities.load_and_save_data import load_pickle, save_pickle
import pycircstat as pycs

def rel_dir_distribution_all_sinks(spike_train, sink_bins, candidate_sinks, direction_bins, pos_data):
   
    """ 
    Create array to store the relative direcion histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 3D array, with
    dimensions (n_y_bins, n_x_bins, n_direction_bins). 

    PRETTY SURE THIS DOESN'T NEED THE FIRST SET OF X AND Y LOOPS!!!!!!!!!!!!!!! FIX IT!!!!!!!!!! <- note by Jake, unsure if true

    """
    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()

    # Finding spike times for this unit
    x = x[spike_train]
    y = y[spike_train]
    hd = hd[spike_train]


    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((len(candidate_sinks['y']), len(candidate_sinks['x']), len(direction_bins) - 1))  
   

    x_bin = np.digitize(x, sink_bins['x']) - 1
    # find x_bin == n_x_bins, and set it to n_x_bins - 1
    x_bin[x_bin == (len(sink_bins['x'])-1)] = len(sink_bins['x'])-2

    y_bin = np.digitize(y, sink_bins['y']) - 1
    # find y_bin == n_y_bins, and set it to n_y_bins - 1
    y_bin[y_bin == (len(sink_bins['y'])-1)] = len(sink_bins['y'])-2

    # loop through the x and y bins
    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            x_positions = x[indices]
            y_positions = y[indices]
            positions = {'x': x_positions, 'y': y_positions}

            # get the head directions for these indices
            hd_temp = hd[indices]

            # loop through candidate consink positions
            for i2, x_sink in enumerate(candidate_sinks['x']):
                for j2, y_sink in enumerate(candidate_sinks['y']):

                    # get directions to sink                    
                    directions = get_directions_to_position([x_sink, y_sink], positions)

                    # get the relative direction
                    relative_direction = get_relative_directions_to_position(directions, hd_temp)

                    # bin the relative directions 
                    rel_dir_binned_counts, _ = bin_directions(relative_direction, direction_bins)
                    rel_dir_dist[j2, i2, :] = rel_dir_dist[j2, i2, :] + rel_dir_binned_counts
    
    return rel_dir_dist

def rel_dir_ctrl_distribution_all_sinks(spike_train, reldir_occ_by_pos, sink_bins, candidate_sinks, pos_data):
    """
    For a given unit, produces relative direction occupancy distributions 
    for each candidate consink position based on the number of spikes fired 
    at each positional bin. 
    """

    # get head directions as np array
    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()

    # Finding spike times for this unit
    x = x[spike_train]
    y = y[spike_train]
    hd = hd[spike_train]

    x_bin = np.digitize(x, sink_bins['x']) - 1
    # find x_bin == n_x_bins, and set it to n_x_bins - 1
    x_bin[x_bin == (len(sink_bins['x'])-1)] = len(sink_bins['x'])-2

    y_bin = np.digitize(y, sink_bins['y']) - 1
    # find y_bin == n_y_bins, and set it to n_y_bins - 1
    y_bin[y_bin == (len(sink_bins['y'])-1)] = len(sink_bins['y'])-2

    direction_bins = get_direction_bins(n_bins=12)
    rel_dir_ctrl_dist = np.zeros((len(candidate_sinks['y']), len(candidate_sinks['x']), len(direction_bins) -1))

    # loop through the x and y bins
    n_spikes_total = 0
    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            n_spikes = len(indices)
            if n_spikes == 0:
                continue
            n_spikes_total = n_spikes_total + n_spikes

            rel_dir_ctrl_dist = rel_dir_ctrl_dist + reldir_occ_by_pos[j,i,:,:,:] * n_spikes

    return rel_dir_ctrl_dist, n_spikes_total

def normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total):
    """
    Normalise the relative direction distribution by the control distribution. 
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_div_ctrl = rel_dir_dist/rel_dir_ctrl_dist 

    # now we want the counts in each histogram to sum to the total number of spikes
    if len(rel_dir_dist_div_ctrl.shape) > 1:
        sum_rel_dir_dist_div_ctrl = rel_dir_dist_div_ctrl.sum(axis=2)
        sum_rel_dir_dist_div_ctrl_ex = sum_rel_dir_dist_div_ctrl[:,:,np.newaxis]

    else:
        sum_rel_dir_dist_div_ctrl_ex = rel_dir_dist_div_ctrl.sum()
        
    normalised_rel_dir_dist = (rel_dir_dist_div_ctrl/sum_rel_dir_dist_div_ctrl_ex) * n_spikes_total

    return normalised_rel_dir_dist


def mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution. 
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1])/2

    n_y_bins = normalised_rel_dir_dist.shape[0]
    n_x_bins = normalised_rel_dir_dist.shape[1]

    mrl = np.zeros((n_y_bins, n_x_bins))
    mean_angle = np.zeros((n_y_bins, n_x_bins))

    for i in range(n_y_bins):
        for j in range(n_x_bins):

            mrl[i,j] = pycs.resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist[i,j,:])
            mean_angle[i,j] = pycs.mean(dir_bin_centres, w=normalised_rel_dir_dist[i,j,:])

    return mrl, mean_angle

def find_consink(spike_train, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks, pos_data):
    """
    Find the consink position that maximises the mean resultant length of the normalised relative direction distribution. 
    """
    #  get control occupancy distribution
    rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution_all_sinks(spike_train, reldir_occ_by_pos, sink_bins, candidate_sinks, pos_data)

    # rel dir distribution for each possible consink position
    rel_dir_dist = rel_dir_distribution_all_sinks(spike_train, sink_bins, candidate_sinks, direction_bins, pos_data)

    # normalise rel_dir_dist by rel_dir_ctrl_dist
    normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total)

    # calculate the mean resultant length of the normalised relative direction distribution
    mrl, mean_angle = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)

    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angle[max_mrl_indices[0][0], max_mrl_indices[1][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle

def main(derivatives_base, frame_rate = 25, sample_rate = 30000):
    """
    Code to find consinks, based on Jake's code


    """
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    pos_data = pd.read_csv(pos_data_path)

    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()

    # get x and y limits
    limits = get_axes_limits()

    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'spatial_features', 'consink_data')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # For now, we're ignoring choice types
    reldir_occ_by_pos = {}
    sink_bins = {}
    candidate_sinks = {}

    # Direction bins
    direction_bins = get_direction_bins(n_bins=12)

    # Loading or creating data 
    file_name = f'reldir_occ_by_pos.npy'
    if os.path.exists(os.path.join(output_folder, file_name)) == False:
        reldir_occ_by_pos, sink_bins, candidate_sinks = get_relative_direction_occupancy_by_position(pos_data, limits)
        np.save(os.path.join(output_folder , file_name), reldir_occ_by_pos)
        # save sink bins and candidate sinks as pickle files
        save_pickle(sink_bins, f'sink_bins', output_folder)
        save_pickle(candidate_sinks, f'candidate_sinks', output_folder)     

    else:
        reldir_occ_by_pos = np.load(os.path.join(output_folder, file_name))
        sink_bins = load_pickle(f'sink_bins', output_folder)
        candidate_sinks = load_pickle(f'candidate_sinks', output_folder)

    ################# CALCULATE CONSINKS ###########################################     
    consinks = {}
    consinks_df = {}

        
    for unit_id in unit_ids:
        # Find spiketrain
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
        spike_train = [np.int32(el) for el in unit_id]
        # get consink  
        max_mrl, max_mrl_indices, mean_angle = find_consink(spike_train, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks, pos_data)
        consink_position = np.round([candidate_sinks['x'][max_mrl_indices[1][0]], candidate_sinks['y'][max_mrl_indices[0][0]]], 3)
        consinks[unit_id] = {'mrl': max_mrl, 'position': consink_position, 'mean_angle': mean_angle}

        # create a data frame with the consink positions
        # FIX THIS< NOT CORRECT
        consinks_df[unit_id] = pd.DataFrame(consinks).T
        
        

    # save as csv            
    consinks_df.to_csv(os.path.join(output_folder, f'consinks_df.csv'))

    # save consinks_df 
    save_pickle(consinks_df, 'consinks_df', output_folder)

        
if __name__ == "__main__":
    
    derivatives_base = None
    # main()
    main(derivatives_base)


