import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from calculate_pos_and_dir import get_directions_to_position, get_relative_directions_to_position
from calculate_occupancy import get_relative_direction_occupancy_by_position, get_axes_limits, get_direction_bins, \
    bin_directions, get_relative_direction_occupancy_by_position_platformbins
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from utilities.trials_utils import get_goal_coordinates, get_goal_numbers, get_coords_127sinks, get_unit_ids, get_pos_data, verify_allnans, get_spike_train, get_sink_positions_platforms, translate_positions
from matplotlib.patches import RegularPolygon
import matplotlib

matplotlib.use("QtAgg")
from joblib import Parallel, delayed
from utilities.mrl_func import resultant_vector_length
from astropy.stats import circmean
from tqdm import tqdm
from typing import Literal
from plotting.plot_sinks import plot_all_consinks, plot_all_consinks_127sinks

num_candidate_sinks = 127
""" In this code, the bins are the platforms"""
import warnings


def rel_dir_distribution_all_sinks(spike_train, direction_bins, reldir_allframes):
    """
    Create array to store the relative direction histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 2D array, with
    dimensions (n_sinks, n_direction_bins).

    """

    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((num_candidate_sinks, len(direction_bins) - 1))

    # loop through candidate consink positions
    for s in range(num_candidate_sinks):
        reldirections_sink = reldir_allframes[:, s]

        # get the relative direction
        relative_direction = reldirections_sink[spike_train]

        relative_direction = relative_direction[~np.isnan(relative_direction)]
        # bin the relative directions
        rel_dir_binned_counts, _ = bin_directions(relative_direction, direction_bins)
        rel_dir_dist[s, :] = rel_dir_dist[s, :] + rel_dir_binned_counts

    if np.all(rel_dir_dist == 0):
        print("rel dir dist all 0")

    return rel_dir_dist


def get_dir_allframes(pos_data, sink_positions):
    """ Gets directions from each frame to each sink"""

    sinkdir_allframes = np.zeros(
        (len(pos_data), num_candidate_sinks)
    )

    reldir_allframes = np.zeros(
        (len(pos_data), num_candidate_sinks)
    )


    x_org = pos_data.iloc[:, 0].to_numpy()
    y_org = pos_data.iloc[:, 1].to_numpy()
    hd_org = pos_data.iloc[:, 2].to_numpy()
    positions = {'x': x_org, 'y': y_org}
    for s in range(num_candidate_sinks):
        platform_loc = sink_positions[s]
        directions = get_directions_to_position([platform_loc[0], platform_loc[1]], positions)
        sinkdir_allframes[:, s] = directions
        relative_direction = get_relative_directions_to_position(directions, hd_org)
        reldir_allframes[:, s] = relative_direction
    return sinkdir_allframes, reldir_allframes


def rel_dir_ctrl_distribution_all_sinks(spike_train, reldir_occ_by_pos, platforms_spk):
    """
    For a given unit, produces relative direction occupancy distributions
    for each candidate consink position based on the number of spikes fired
    at each positional bin.

    NOTE: The input for the spikes will already be restricted to each goal
    """


    direction_bins = get_direction_bins(n_bins=12)
    rel_dir_ctrl_dist = np.zeros((num_candidate_sinks, len(direction_bins) - 1))

    # loop through the x and y bins
    n_spikes_total = 0

    for p in range(61):
        # get the indices where platforms_spj == p + t1
        indices = np.where(platforms_spk == p + 1)[0]

        # Number of spikes this cell fired in the bin
        n_spikes = len(indices)
        if n_spikes == 0:
            continue
        # number of spikes cell fired in total so far
        n_spikes_total = n_spikes_total + n_spikes

        # Add (n_y_sinks, n_x_sinks, n_dir_bins)*n_spikes (scale by how many spikes are fired there)
        rel_dir_ctrl_dist = rel_dir_ctrl_dist + reldir_occ_by_pos[:, p,  :] * n_spikes

    if np.all(rel_dir_ctrl_dist == 0):
        # print("All zeroes in rel dir ctrl dist. Breakpoint")
        pass
    return rel_dir_ctrl_dist, n_spikes_total


def normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total):
    """
    Normalise the relative direction distribution by the control distribution.
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_div_ctrl = np.divide(
        rel_dir_dist,
        rel_dir_ctrl_dist,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where=rel_dir_ctrl_dist != 0
    )

    # now we want the counts in each histogram to sum to the total number of spikes
    if len(rel_dir_dist_div_ctrl.shape) > 1: ## NOTE: check whether this is correct here.
        sum_rel_dir_dist_div_ctrl = rel_dir_dist_div_ctrl.sum(axis=1)
        sum_rel_dir_dist_div_ctrl_ex = sum_rel_dir_dist_div_ctrl[:, np.newaxis]

    else:
        sum_rel_dir_dist_div_ctrl_ex = rel_dir_dist_div_ctrl.sum()

    normalised_rel_dir_dist = np.divide(
        rel_dir_dist_div_ctrl,
        sum_rel_dir_dist_div_ctrl_ex,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where=sum_rel_dir_dist_div_ctrl_ex != 0
    )
    normalised_rel_dir_dist = normalised_rel_dir_dist * n_spikes_total

    return normalised_rel_dir_dist


def mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution.
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1]) / 2


    mrl = np.zeros(num_candidate_sinks)
    mean_angle = np.zeros(num_candidate_sinks)

    for s in range(num_candidate_sinks):
            mrl[s] = resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist[s, :])

            mean_angle[s] = circmean(dir_bin_centres, weights=normalised_rel_dir_dist[s, :])

            # warnings.filterwarnings("error")
    return mrl, mean_angle


def rel_dir_distribution_m2(spike_train, platforms_spk,  direction_bins, reldir_allframes):
    """
    METHOD 2 (normalize each platform)
    Create array to store the relative direction histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 3D array, with
    dimensions (n_sinks, n_platforms, n_direction_bins).

    """

    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((num_candidate_sinks, 61, len(direction_bins) - 1))


    # n_spikes_per_platform
    n_spikes_per_platform = np.array([
        np.sum(platforms_spk == (p + 1)) for p in range(61)
    ])

    platform_masks = [(platforms_spk == (p + 1)) for p in range(61)]

    # loop through candidate consink positions
    for s in range(num_candidate_sinks):
        reldirections_sink = reldir_allframes[:, s]

        # get the relative direction
        relative_direction = reldirections_sink[spike_train]

        for p in range(61):
            mask_p = platform_masks[p]
            relative_direction_p = relative_direction[mask_p]
            relative_direction_p = relative_direction_p[~np.isnan(relative_direction_p)]

            if len(relative_direction_p) == 0:
                continue
            # bin the relative directions
            rel_dir_binned_counts, _ = bin_directions(relative_direction_p, direction_bins)
            rel_dir_dist[s, p, :] += rel_dir_binned_counts

    if np.all(rel_dir_dist == 0):
        print("rel dir dist all 0")

    return rel_dir_dist, n_spikes_per_platform

def normalize_rel_dir_dist_m2(rel_dir_dist, reldir_occ_by_pos,  n_spikes_per_platform):
    """
    Method 2 (normalize each platform)
    Normalise the relative direction distribution by the control distribution.
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_div_ctrl = np.divide(
        rel_dir_dist,
        reldir_occ_by_pos,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where= reldir_occ_by_pos != 0
    )

    # Step 2: normalize within each (sink, platform)
    # sum over direction bins
    sum_per_sink_platform = rel_dir_dist_div_ctrl.sum(axis=2)  # (n_sinks, n_platforms), gives the sum of the bins for each sink for each platform

    # expand dims for broadcasting
    sum_per_sink_platform_ex = sum_per_sink_platform[:, :, np.newaxis]

    rel_dir_dist_norm = np.divide(
        rel_dir_dist_div_ctrl,
        sum_per_sink_platform_ex,
        out=np.zeros_like(rel_dir_dist_div_ctrl),
        where=sum_per_sink_platform_ex != 0
    )

    # multiply by number of spikes on each platform
    rel_dir_dist_norm *= n_spikes_per_platform[np.newaxis, :, np.newaxis]

    # Step 3: sum across platforms
    rel_dir_dist_final = rel_dir_dist_norm.sum(axis=1)  # (n_sinks, n_dir_bins)

    return rel_dir_dist_final



def find_consink_method2(spike_train, reldir_occ_by_pos,  direction_bins,pos_data,
                 reldir_allframes, platforms_spk = None, verify_nans = True):
    """ Here the firing for each platform is divided by the rel_dir_occ for that platform
    platfomrs_spk is not none for when we use this function to calculate the population sink"""

    spike_train = np.array(spike_train)

    if verify_nans:
        all_nans = verify_allnans(spike_train, pos_data)

        if all_nans:
            print("All nans. Returning nan")
            return np.nan, np.nan, np.nan

    if platforms_spk is None:
        platforms = pos_data['platform'].to_numpy()
        platforms_spk = platforms[spike_train]
        mask = np.isnan(platforms_spk)
        platforms_spk = platforms_spk[~mask]

    rel_dir_dist,  n_spikes_per_platform = rel_dir_distribution_m2(spike_train, platforms_spk,  direction_bins, reldir_allframes)

    normalised_rel_dir_dist = normalize_rel_dir_dist_m2(rel_dir_dist, reldir_occ_by_pos,  n_spikes_per_platform)

    mrl, mean_angles = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)
    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angles[max_mrl_indices[0][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle

def find_consink(spike_train, reldir_occ_by_pos,  direction_bins,pos_data,
                 reldir_allframes, platforms_spk = None, verify_nans = True):
    """
    Find the consink position that maximises the mean resultant length of the normalised relative direction distribution.
    """
    if verify_nans:
        spike_train = np.array(spike_train)
        all_nans = verify_allnans(spike_train, pos_data)

        if all_nans:
            print("All nans. Returning nan")
            return np.nan, np.nan, np.nan

    if platforms_spk is None:
        # get head directions as np array
        platforms = pos_data['platform'].to_numpy()
        platforms_spk = platforms[spike_train]

        mask = np.isnan(platforms_spk)
        platforms_spk = platforms_spk[~mask]


    #  get control occupancy distribution
    rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution_all_sinks(spike_train, reldir_occ_by_pos,
                                                                            platforms_spk)

    # rel dir distribution for each possible consink position
    rel_dir_dist = rel_dir_distribution_all_sinks(spike_train, direction_bins, reldir_allframes)

    # normalise rel_dir_dist by rel_dir_ctrl_dist
    normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total)
    if np.isnan(normalised_rel_dir_dist).any():
        breakpoint()
    # calculate the mean resultant length of the normalised relative direction distribution
    mrl, mean_angles = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)
    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angles[max_mrl_indices[0][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle


def shift_spiketrain_pergoal(spike_train, goal, intervals_frames, n_frames: int, frame_rate=25):
    """Shift the spike train by a random amount. Restrict it to goal intervals

    Args:
        spike_train (array): firing times of unit (in frames)
        goal(int): goal number
        n_frames (int): length of the recording (in frames)
    """
    if goal < 3:
        start_col = goal * 2
        end_col = goal*2 + 1
    else:
        start_col = 0
        end_col = intervals_frames.shape[1] - 1
    spike_train = np.array(spike_train)
    min_shift = 2 * frame_rate
    lengths = [intervals_frames.iloc[tr, end_col] - intervals_frames.iloc[tr, start_col] for tr in
               range(len(intervals_frames))]
    max_shift = np.max(lengths) - min_shift + 1


    # pick a shift randomly between those two numbers
    shift = np.random.randint(min_shift, max_shift)

    shifted_data = np.empty(0, dtype=int)

    for tr in range(len(intervals_frames)):
        start_frame = intervals_frames.iloc[tr, start_col]
        end_frame = intervals_frames.iloc[tr, end_col]
        spike_train_tr = spike_train[(spike_train >= start_frame) & (spike_train <= end_frame)]

        if len(spike_train_tr) == 0:
            continue
        shifted_data_tr = spike_train_tr + shift

        range_min = start_frame
        range_max = end_frame
        range_size = range_max - range_min + 1

        shifted_data_tr = np.mod(shifted_data_tr - range_min, range_size) + range_min

        if np.min(shifted_data_tr) < range_min or np.max(shifted_data_tr) > range_max:
            breakpoint()

        shifted_data = np.append(shifted_data, shifted_data_tr)

    shifted_data = np.array(shifted_data)

    return shifted_data


def calculate_translated_mrl(spiketrain, dlc_data, reldir_occ_by_pos,  direction_bins,
                             reldir_allframes, rawsession_folder, intervals_frames, method, goal):
    n_frames = len(dlc_data)
    translated_spiketrain = shift_spiketrain_pergoal(spiketrain, goal, intervals_frames, n_frames)

    if method == 1:
        mrl, _, _ = find_consink(translated_spiketrain, reldir_occ_by_pos,  direction_bins,
                                 dlc_data, reldir_allframes)
    elif method ==2:
        mrl, _, _ = find_consink_method2(translated_spiketrain, reldir_occ_by_pos, direction_bins,
                                 dlc_data, reldir_allframes)
    return mrl


def recalculate_consink_to_all_candidates_from_translation(spiketrain, dlc_data, reldir_occ_by_pos,
                                                           direction_bins,reldir_allframes,
                                                           rawsession_folder, intervals_frames, method, goal):
    n_shuffles = 1000
    mrl = np.zeros(n_shuffles)

    # mrl = Parallel(n_jobs=-1, verbose=0)(delayed(calculate_translated_mrl)(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles))

    mrl = Parallel(n_jobs=-1, verbose=0)(
        delayed(calculate_translated_mrl)(spiketrain, dlc_data, reldir_occ_by_pos,  direction_bins,
                                           reldir_allframes, rawsession_folder, intervals_frames, method, goal)
        for s in range(n_shuffles))



    # mrl = [calculate_translated_mrl(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles)]
    # remove nan values from mrl
    mrl = np.array(mrl)
    mrl = mrl[~np.isnan(mrl)]
    if len(mrl) == 0:
        return (np.nan, np.nan, np.nan)
    mrl = np.round(mrl, 3)
    mrl_95 = np.percentile(mrl, 95)
    mrl_999 = np.percentile(mrl, 99.9)

    if len(mrl) < 1000:
        print(len(mrl))
    ci = (mrl_95, mrl_999, n_shuffles - len(mrl))  # last one is the length after nans are removed

    return ci


def calculate_averagesink(consinks_df, hcoord, vcoord, include_g0):
    average_positions = {}
    for g in [0,1,2]:
        if g == 0 and not include_g0:
            continue # skip g == 0

        position = []
        for cluster in consinks_df.index:
            sig = consinks_df.loc[cluster, 'sig_g' + str(g)]
            if sig == "sig":

                consink_plat = consinks_df.loc[cluster, 'platform_g' + str(g)]
                position.append([hcoord[np.int32(consink_plat)- 1], vcoord[np.int32(consink_plat) - 1]])
        # make the axes equal
        avg_pos = np.mean(position, axis = 0)
        average_positions[g] = avg_pos
    return average_positions


def main(derivatives_base, rel_dir_occ: Literal['all trials', 'intervals'],
         unit_type: Literal['pyramidal', 'good', 'all'], method: Literal[1,2], code_to_run=[], include_g0 = True, frame_rate=25, sample_rate=30000):
    """
    Code to find consinks, based on Jake's code


    """
    # Path to rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)

    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting", "sorter_output")
    sorting = se.read_kilosort(
        folder_path=kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    unit_ids = get_unit_ids(derivatives_base, unit_ids, unit_type)

    # Loading xy data
    pos_data, pos_data_g0, pos_data_g1, pos_data_g2, pos_data_reldir = get_pos_data(derivatives_base, rel_dir_occ)

    # restricted df frames
    path = os.path.join(rawsession_folder, 'task_metadata', 'restricted_df_frames.csv')
    intervals_frames = pd.read_csv(path)

    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features',
                                 'consink_data_platformbins')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Direction bins (from -pi to pi)
    direction_bins = get_direction_bins(n_bins=12)

    # gets translated positions
    platforms_trans = translate_positions()
    # Loading or creating data
    sink_positions = get_sink_positions_platforms(derivatives_base)
    goal_numbers= get_goal_numbers(derivatives_base)
    sinkdir_allframes, reldir_allframes = get_dir_allframes(pos_data, sink_positions)
    file_name = 'reldir_occ_by_pos.npy'

    if -1 in code_to_run:
        print("Calculating relative direction occupancy by position")
        reldir_occ_by_pos= get_relative_direction_occupancy_by_position_platformbins(pos_data_reldir, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
        reldir_occ_by_pos_g1 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g1, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
        reldir_occ_by_pos_g2 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g2, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
        np.save(os.path.join(output_folder, file_name), reldir_occ_by_pos)
        np.save(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'), reldir_occ_by_pos_g1)
        np.save(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'), reldir_occ_by_pos_g2)

    else:
        print("Loading reldir occ, not calculating")
        reldir_occ_by_pos = np.load(os.path.join(output_folder, file_name))
        reldir_occ_by_pos_g1 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'))
        reldir_occ_by_pos_g2 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'))

    ################# CALCULATE CONSINKS ###########################################
    consinks = {}

    if 0 in code_to_run:
        print("Calculating consinks")
        for unit_id in tqdm(unit_ids):
            consinks[unit_id] = {'unit_id': unit_id}

            for g in [0, 1, 2]:
                if g == 0:
                    # store with goal suffix
                    consinks[unit_id][f'mrl_g{g}'] = np.nan
                    consinks[unit_id][f'position_g{g}'] = np.nan
                    consinks[unit_id][f'mean_angle_g{g}'] = np.nan
                    consinks[unit_id][f'numspikes_g{g}'] = np.nan
                    consinks[unit_id][f'platform_g{g}'] = np.nan
                    continue
                if g == 1:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                elif g == 2:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                else:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos

                # Find spiketrain
                spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
                spike_train_secs = spike_train_unscaled / sample_rate  # This is in seconds now
                # Restrict spiketrain to goal
                spike_train_secs_g = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder, goal=g)
                # Now let spiketrain be in frame_rate
                spike_train = np.round(spike_train_secs_g * frame_rate)
                spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]

                # Skip empty spikes
                if len(spike_train) == 0:
                    continue

                # get consink
                if method == 1:
                    max_mrl, max_mrl_indices, mean_angle = find_consink(
                        spike_train, reldir_occ_by_pos_cur, direction_bins, pos_data,
                        reldir_allframes
                    )
                elif method == 2:
                    max_mrl, max_mrl_indices, mean_angle = find_consink_method2(
                        spike_train, reldir_occ_by_pos_cur, direction_bins, pos_data,
                        reldir_allframes
                    )
                consink_plat = max_mrl_indices[0][0] + 1
                original_plat = np.where(platforms_trans == consink_plat)[0]

                if len(original_plat) == 0:
                    original_plat = np.nan
                else:
                    original_plat = original_plat[0] + 1

                # store with goal suffix
                consinks[unit_id][f'mrl_g{g}'] = max_mrl
                consinks[unit_id][f'platform_g{g}'] = consink_plat
                consinks[unit_id][f'original_platform_g{g}'] = original_plat
                consinks[unit_id][f'mean_angle_g{g}'] = mean_angle
                consinks[unit_id][f'numspikes_g{g}'] = len(spike_train)

        # Create dataframe
        consinks_df = pd.DataFrame(consinks).T
        print(consinks_df)

        # save as csv
        consinks_df.to_csv(os.path.join(output_folder, f'consinks_df_m{method}.csv'), index=False)
        print(f"Data saved to {os.path.join(output_folder, f'consinks_df_m{method}.csv')}")
        # save consinks_df
        save_pickle(consinks_df, f'consinks_df_m{method}', output_folder)

    # ######################### TEST STATISTICAL SIGNIFICANCE OF CONSINKS #########################
    # shift the head directions relative to their positions, and recalculate the tuning to the
    # previously identified consink position.

    if 1 in code_to_run:
        print("Assessing significance")
        # load the consinks_df
        consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)

        # make columns for the confidence intervals; place them directly beside the mrl column
        idx_g0 = consinks_df.columns.get_loc('mrl_g0')

        # if the columns don't exist, insert them
        if 'ci_95_g1' not in consinks_df.columns:
            consinks_df.insert(idx_g0 + 1, 'ci_95_g0', np.nan)
            consinks_df.insert(idx_g0 + 2, 'ci_999_g0', np.nan)
            consinks_df.insert(idx_g0 + 3, 'sig_g0', np.nan)
            idx_g1 = consinks_df.columns.get_loc('mrl_g1')
            consinks_df.insert(idx_g1 + 1, 'ci_95_g1', np.nan)
            consinks_df.insert(idx_g1 + 2, 'ci_999_g1', np.nan)
            consinks_df.insert(idx_g1 + 3, 'sig_g1', np.nan)
            idx_g2 = consinks_df.columns.get_loc('mrl_g2')
            consinks_df.insert(idx_g2 + 1, 'ci_95_g2', np.nan)
            consinks_df.insert(idx_g2 + 2, 'ci_999_g2', np.nan)
            consinks_df.insert(idx_g2 + 3, 'sig_g2', np.nan)

        for unit_id in tqdm(unit_ids):
            for g in [0, 1, 2]:
                if g == 0:
                    consinks_df.loc[unit_id, f'ci_95_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'ci_999_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'sig_g{g}'] = np.nan
                    continue
                if g == 1:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                elif g == 2:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                else:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos

                # print(f'Were at {unit_id} with { g}' )

                if consinks_df.loc[unit_id, f'numspikes_g{g}'] < 30:
                    consinks_df.loc[unit_id, f'ci_95_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'ci_999_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'sig_g{g}'] = np.nan
                    continue

                spike_train = get_spike_train(sorting, unit_id, pos_data, rawsession_folder, g=g, frame_rate=frame_rate,
                                              sample_rate=sample_rate)

                ci = recalculate_consink_to_all_candidates_from_translation(spike_train, pos_data,
                                                                            reldir_occ_by_pos_cur,
                                                                            direction_bins,
                                                                            reldir_allframes, rawsession_folder,
                                                                            intervals_frames, method = method, goal=g)

                consinks_df.loc[unit_id, f'ci_95_g{g}'] = ci[0]
                consinks_df.loc[unit_id, f'ci_999_g{g}'] = ci[1]
                mrl_val = consinks_df.loc[unit_id, f'mrl_g{g}']
                if np.isfinite(ci[0]) and np.isfinite(mrl_val) and mrl_val > ci[0]:
                    sig = 'sig'
                else:
                    sig = 'ns'
                consinks_df.loc[unit_id, f'sig_g{g}'] = sig

        print(f"Saved consink data to the following folder: {output_folder}")
        try:
            consinks_df.to_csv(os.path.join(output_folder, f'consinks_df_m{method}.csv'))
        except:
            breakpoint()
        save_pickle(consinks_df, f'consinks_df_m{method}', output_folder)

    if 2 in code_to_run:
        print("Calculating mean sink position")
        hcoord, vcoord = get_coords_127sinks(derivatives_base)
        consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)
        average_sink = calculate_averagesink(consinks_df, hcoord, vcoord, include_g0)
        save_pickle(average_sink, f'average_sink_m{method}', output_folder)

    ######################## PLOT ALL CONSINKS #################################
    hcoord, vcoord = get_coords_127sinks(derivatives_base)
    x_diff = np.mean(np.diff(hcoord))

    y_diff = np.mean(np.diff(vcoord))
    jitter = (x_diff / 3, y_diff / 3)

    # Check if consinks_df is a dictionary otherwise convert
    consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)
    average_sink = load_pickle(f'average_sink_m{method}', output_folder)
    plot_all_consinks_127sinks(consinks_df, goal_numbers, hcoord, vcoord, platforms_trans,  jitter=jitter, plot_dir=output_folder,average_sink = average_sink, include_g0 = include_g0,
                      plot_name=f'ConSinks Good Units method {method}')


if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    main(derivatives_base, 'all trials', 'pyramidal', method = 1, code_to_run=[-1,0,1,2], include_g0 = False)


