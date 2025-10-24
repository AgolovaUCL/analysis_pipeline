import matplotlib
import numpy as np
import pandas as pd
from matplotlib.path import Path
import os
import spikeinterface.extractors as se
from utilities.platforms_utils import get_platform_center, calculate_occupancy_plats, get_hd_distr_allplats, get_firing_rate_platforms, get_norm_hd_distr
from utilities.restrict_spiketrain import restrict_spiketrain
from calculate_occupancy import get_direction_bins
from population_sink.get_relDirDist import calculate_relDirDist
from population_sink.calculate_MRLval import mrlData
from utilities.mrl_func import resultant_vector_length
from astropy.stats import circmean
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.trials_utils import get_limits_from_json, get_goal_numbers, get_coords
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from tqdm import tqdm
matplotlib.use("TkAgg")

def calculate_popsink(derivatives_base,  unit_type, title = 'Population Sink', frame_rate = 25, sample_rate = 30000, code_to_run = []):
    """
    calculates the population sink for the whole trial, for units split into goal 1 and units split into goal 2
    
    NOTE: Again, reldirdist can be calulcated two ways: per goal or for full trial. I have to try both methods!!!!!!
    """
    if unit_type not in ['pyramidal', 'good', 'all']:
        raise ValueError('unit type not correctly defined')
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    goal_numbers = get_goal_numbers(derivatives_base)
    
    hcoord, vcoord = get_coords(derivatives_base)
    
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    if unit_type == 'good':

        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting","sorter_output", "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
        # Loading pyramidal units
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features","all_units_overview", "pyramidal_units_2D.csv")
        print("Getting pyramidal units 2D")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units

    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_w_platforms.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])
        
    # Output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'population_sink')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Needed for analysis
    platform_occupancy = calculate_occupancy_plats(pos_data)
    hd_distr_allplats, bin_centers = get_hd_distr_allplats(pos_data)

    # To load data
    data_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'consink_data')
        
    # Loop over all units
    if 1 in code_to_run:
        for g in [0,1,2]:
        # Here g == 0 corresponds to the whole session, not split into goals
            scaled_vecs_allplats = []
            for unit_id in tqdm(unit_ids):
                # Load spike times
                spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
                spike_train_secs = np.round(spike_train_unscaled/sample_rate) # trial data is now in frames in order to match it with xy data
                
                # If we're only looking at one goal, restrict the spiketrain to match only period of goal 1 or 2
                if g > 0:
                    spike_train_restricted = restrict_spiketrain(spike_train_secs, rawsession_folder=rawsession_folder, goal=g)
                else:
                    spike_train_restricted = spike_train_secs
                spike_train = spike_train_restricted*frame_rate
                spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]

                # Get firing rate
                plat_firing_rate = get_firing_rate_platforms(spike_train, pos_data, platform_occupancy)
                # Normalised hd distr
                norm_hd_distr = get_norm_hd_distr(spike_train, pos_data, hd_distr_allplats)

                # normalised MRL and mean angle
                norm_MRL = [resultant_vector_length(bin_centers, w=norm_hd_distr[i]) for i in range(61)]
                norm_mean_angle = []
                for i in range(61):
                    if np.all(np.isnan(norm_hd_distr[i])) or np.all(norm_hd_distr[i] == 0):
                        norm_mean_angle.append(np.nan)
                    else:
                        norm_mean_angle.append(circmean(bin_centers, weights=norm_hd_distr[i]))

                # This is by how much we'll scale the unit vectors
                scale_factor = [norm_MRL[p] *plat_firing_rate[p] for p in range(61)]

                # Make unit vectors from mean angle
                scaled_vecs = [[np.cos(norm_mean_angle[p])*scale_factor[p], np.sin(norm_mean_angle[p]*scale_factor[p])] for p in range(61)]

                scaled_vecs_allplats.append(scaled_vecs)
            
            scaled_vecs_allplats = np.array(scaled_vecs_allplats) 
            # Taking the mean for each platform
            mean_norm_vecs = np.nanmean(scaled_vecs_allplats, axis=0) 
            
            # Getting the length of each vec
            mean_norm_vecs_length = np.linalg.norm(mean_norm_vecs, axis=1)
            mean_norm_vecs_angle = np.arctan2(mean_norm_vecs[:,1], mean_norm_vecs[:,0])  # and angle 
            
            # Nans for length and angle
            indices_to_del = np.where(np.isnan(mean_norm_vecs_length))
            indices_to_del = indices_to_del[0]
            
            ##### CALCULATING CONSINK ##### 
            
            # We replace this with sinkbins
            sink_bins = load_pickle('sink_bins', data_folder)
            direction_bins = load_pickle('direction_bins', data_folder) # Should be the same as angle edges
            
            # 'spike data'
            pos = np.array([get_platform_center(hcoord, vcoord, p+1) for p in range(61)]) # positions are the center for each platform
            plats = np.arange(1,62)
            hd = mean_norm_vecs_angle # hd is the mean angle for each platform
            nspikes = np.round(np.array(mean_norm_vecs_length)*100).astype(int) 
            
            # Delelete nans
            plats = np.delete(plats, indices_to_del)
            pos = np.delete(pos, indices_to_del, axis=0)
            hd = np.delete(hd, indices_to_del)
            nspikes = np.delete(nspikes, indices_to_del)
            if len(plats) != len(hd) or len(plats) != len(pos) or len(hd) != len(pos) or len(nspikes) != len(pos):
                raise ValueError("Lengths of plats, hd, and pos do not match after removing NaNs.")
            
            spikePos = np.repeat(pos, nspikes, axis = 0)
            spikeHD = np.repeat(hd, nspikes)
            spikePlats = np.repeat(plats, nspikes)
            relDirDist= calculate_relDirDist(pos_data,  sink_bins, direction_bins)# Have to rewrite function for that
            
            mrl_dataset= mrlData(spikePos, spikeHD, spikePlats, relDirDist, direction_bins, sink_bins)
            
            # Saving
            if g == 0:
                name = "popsink_wholetrial"
            else:
                name = f"popsink_g{g}"
            
            save_pickle(mrl_dataset, name,data_folder )
    if 2 in code_to_run:
        # Plotting
        wholetrial_data = load_pickle('popsink_wholetrial', data_folder)
        g1_data = load_pickle('popsink_g1', data_folder)
        g2_data = load_pickle('popsink_g2', data_folder)
        
        mrls = [wholetrial_data['mrl'], g1_data['mrl'], g2_data['mrl']]
        coords = [wholetrial_data['coor'], g1_data['coor'], g2_data['coor']]
        angles = [wholetrial_data['dir_deg'], g1_data['dir_deg'], g2_data['dir_deg']]
        
        hcoord, vcoord = get_coords(derivatives_base)
        limits = get_limits_from_json(derivatives_base)
        plot_popsink(mrls, coords, angles, goal_numbers, hcoord, vcoord,  limits, output_folder, plot_name=title)
        
        


def plot_popsink(mrls, coords, angles, goal_numbers, hcoord, vcoord,  limits, output_folder, plot_name='Population Sinks'):
    """
    Plots popsink and the goal

    Args:
        data_folder (_type_): _description_
        popsink_coor (_type_): _description_
        goals (_type_): _description_
    """

    x_min, x_max, y_min, y_max = limits
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.flatten()
    
    for j in range(3): # j = 0, full trial. j = 1, goal 1. j = 2, goal 2
        ax = axs[j]
        mrl = mrls[j]
        popsink_coor = coords[j]
        popsink_angle = angles[j]
        
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            if j > 0 and i + 1 == goal_numbers[j -1]:
                colour = 'green'
            elif j == 0 and i + 1 in goal_numbers:
                colour = 'green'
            else:
                colour = 'grey'
            hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                                orientation=np.radians(28),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor='k')
            ax.text(x, y, i + 1, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
        # plot the goal positions
        circle = plt.Circle((popsink_coor[0], popsink_coor[1]), 60, color='r', fill=True)
        ax.add_patch(circle)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_max, y_min])
        ax.set_aspect('equal')
        
        # Add small text with MRL and angle on bottom
        ax.text(700, 300, f'MRL: {mrl:.3f}, Angle: {popsink_angle:.1f}°', 
                ha='center', va='center')
        if j == 0:
            title = 'all trials'
        else:
            title = f'goal {j}'
        ax.set_title(title)
    plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
    print(f"Saved population sink plot to {output_folder}")
    plt.show()
                

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    calculate_popsink(derivatives_base, unit_type = 'pyramidal', title = 'Pyramidal Consink New Timestamps', code_to_run = [1,2], frame_rate = 25, sample_rate = 30000)
    