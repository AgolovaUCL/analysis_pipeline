import numpy as np
import pandas as pd
from matplotlib.path import Path
import os
import pycircstat as pycs
import spikeinterface.extractors as se
from utilities.platform_utils import radius, hex_side_length, theta, desired_x, desired_y, hcoord_translated, vcoord_translated, get_platform_center, add_platforms_csv, calculate_occupancy_plats, get_hd_distr_allplats, get_firing_rate_platforms, get_norm_hd_distr
from utilities.restrict_spiketrain import restrict_spiketrain
from calculate_occupancy import get_direction_bins
def calculate_popsink(derivatives_base, goal_platforms, goal_coordinates, only_good_units = False, frame_rate = 25, sample_rate = 30000):
    """
    calculates the population sink for the whole trial, for units split into goal 1 and units split into goal 2
    """
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    if only_good_units:
        # STILL ADD CODE HERE
        pass

    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    add_platforms_csv(pos_data_path)
    pos_data = pd.read_csv(pos_data_path)

    # Output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'spatial_features', 'population_sink')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Needed for analysis
    platform_occupancy = calculate_occupancy_plats(pos_data)
    hd_distr_allplats, bin_centers = get_hd_distr_allplats(pos_data)


    # Loop over all units
    for g in [0,1,2]:
       # Here g == 0 corresponds to the whole session, not split into goals
        scaled_vecs_allplats = []
        for unit_id in unit_ids:
            # Load spike times
            spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
            spike_train_secs = np.round(spike_train_unscaled/sample_rate) # trial data is now in frames in order to match it with xy data
            if g > 0:
                spike_train_restricted = restrict_spiketrain(spike_train_secs)
            else:
                spike_train_restricted = spike_train_secs
            spike_train = spike_train_restricted*frame_rate
            spike_train = [np.int32(el) for el in spike_train]

            # Get firing rate
            plat_firing_rate = get_firing_rate_platforms(spike_train, pos_data, platform_occupancy)
            # Normalised hd distr
            norm_hd_distr = get_norm_hd_distr(spike_train, pos_data, hd_distr_allplats)

            # normalised MRL and mean angle
            norm_MRL = [pycs.resultant_vector_length(bin_centers, w=norm_hd_distr[i]) for i in range(61)]
            norm_mean_angle = [pycs.mean(bin_centers, w=norm_hd_distr[i]) for i in range(61)]

            # This is by how much we'll scale the unit vectors
            scale_factor = [norm_MRL[p] *plat_firing_rate[p] for p in range(61)]

            # Make unit vectors from mean angle
            scaled_vecs = [[np.cos(norm_mean_angle[p])*scale_factor[p], np.sin(norm_mean_angle[p]*scale_factor[p])] for p in range(61)]

            scaled_vecs_allplats.append(scaled_vecs)
        scaled_vecs_allplats = np.array(scaled_vecs_allplats) 
        mean_norm_vecs = np.mean(scaled_vecs_allplats, axis=0) 
        mean_norm_vecs_length = np.linalg.norm(mean_norm_vecs, axis=1)
        mean_norm_vecs_angle = np.arctan2(mean_norm_vecs[:,1], mean_norm_vecs[:,0])   
        
        ##### CALCULATING CONSINK ##### 
        
        # 'spike data'
        pos = [get_platform_center(p+1) for p in range(61)] 
        hd = mean_norm_vecs_angle 
        nspikes = np.round(np.array(mean_norm_vecs_length)*100).astype(int) 
        
        direction_bins = get_direction_bins(num_bins = 12) # easy, use consink function # SHOULD BE THE SAME AS BIN CENTERS
        pos_data = ... # Can replicate
        spike_train = ... # can replicate
        candidate_sinks = ... # Load from pickle
        sink_bins = ...# bit trickies
        reldir_occ_by_pos = # Have to rewrite function for that
                        max_mrl, max_mrl_indices, mean_angle = find_consink(
                    spike_train, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks, pos_data
                )
                        
    # add plot
            

    