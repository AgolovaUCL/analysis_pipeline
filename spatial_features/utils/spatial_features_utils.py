import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Literal
import warnings
from astropy.stats import circmean
import astropy.convolution as cnv

from spatial_features.spatial_functions import get_ratemaps
from spatial_features.get_sig_cells import get_sig_cells

def load_unit_ids(derivatives_base, unit_type, unit_ids):
    """ Returns unit_ids, the unit_ids that we will create rmaps for"""
    if unit_type == 'good':
        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting","sorter_output", "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features","all_units_overview", "pyramidal_units_2D.csv")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
        print("Using pyramidal units")
    elif unit_type == "all":
        print("Using all units")
        unit_ids = unit_ids
    else:
        raise ValueError("unit_type not good, pyramidal, or all. Provide correct input")
    return unit_ids
 
def get_limits(derivatives_base):
    """ Reads in limits from limits.json"""
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    
    xmin = limits['x_min']
    xmax = limits['x_max']
    ymin = limits['y_min']
    ymax = limits['y_max']
    return xmin, xmax, ymin, ymax

def get_outline(derivatives_base):
    """Obtains outline of maze from maze_outline_coords.json"""
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    if os.path.exists(outline_path):
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print("Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None     
    return outline_x, outline_y

def get_trial_length_info(epoch_times, trials_length,  tr):
    """ Returns start time of trial and trial length"""
    trial_row = epoch_times[(epoch_times.trialnumber == tr)]
    start_time = trial_row.iloc[0, 1]

    trial_length_row = trials_length[(trials_length.trialnumber == tr)]
    trial_length = trial_length_row.iloc[0, 2]
    return start_time, trial_length, trial_row
            
def get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate = 25):
    """ Restricts spiketrain to current trial"""
    spike_train_this_trial = np.copy(spike_train)
    spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)*frame_rate] # filtering for current trial
    spike_train_this_trial = [el - np.round(trial_dur_so_far*frame_rate) for el in spike_train_this_trial]
    spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)]
    return spike_train_this_trial

def get_spikes_epoch(spike_train_this_trial, epoch_start, epoch_end, frame_rate):
    """ Restricts spiketrain to this epoch"""
    spike_train_this_epoch = [np.int32(el) for el in spike_train_this_trial if el > frame_rate*epoch_start and el < frame_rate *epoch_end]
    spike_train_this_epoch = np.asarray(spike_train_this_epoch, dtype=int)
    return spike_train_this_epoch
                     
def get_posdata(derivatives_base, method = "ears"):
    """ Loads pos data path"""
    if method == "ears":
        pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    elif method == "center":
        pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
    else:
        raise ValueError("Method must be ears or center")
    pos_data = pd.read_csv(pos_data_path)

    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()
    return x, y, hd, pos_data

def get_occupancy_time(hd, frame_rate, num_bins = 24):
    """ Obtains occupancy time for each bin for hd"""
    num_bins = 24
    hd_filtered = hd[~np.isnan(hd)]
    if np.nanmax(hd_filtered) > 2*np.pi:
        hd_filtered= np.deg2rad(hd_filtered)
    occupancy_counts, _ = np.histogram(hd_filtered, bins=num_bins, range = [-np.pi, np.pi])
    occupancy_time = occupancy_counts / frame_rate 
    return occupancy_time

def get_spike_train_frames(sorting, unit_id, x, sample_rate, frame_rate):
    """ Returns spike train in frames. Excludes values above len(x)"""
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_pre = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
    spike_train = [np.int32(el) for el in spike_train_pre if el < len(x)]  # Ensure spike train is within bounds of x and y
    return spike_train

def get_unit_info(df_unit_metrics, unit_id):
    """ Loads unit firing rate and label for unit = unit_id"""
    row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]
    unit_firing_rate = row['firing_rate'].values[0]
    unit_label = row['label'].values[0]
    return unit_firing_rate, unit_label

def load_trial_xpos(pos_data_dir, tr):
    """ Returns x pos for trial tr"""
    trial_csv_name = f'XY_HD_t{tr}.csv'
    trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
    xy_hd_trial = pd.read_csv(trial_csv_path)
                
    x = xy_hd_trial.iloc[:, 0].to_numpy()  
    return x 

def get_directional_firingrate(hd, spike_train, num_bins, occupancy_time):
    """ Gets the diretional firing rate and the bin centers"""
    
    # Get counts per bin
    spikes_hd = hd[spike_train]
    spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
    if np.nanmax(hd) > 2*np.pi:
        spikes_hd_rad = np.deg2rad(spikes_hd)
    else:
        spikes_hd_rad = spikes_hd
    counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )
    # Calculating directional firing rate
    direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)

    # Getting bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return direction_firing_rate, bin_centers
        
        
def get_ratemaps(spikes, x, y, n: int, binsize = 15, stddev = 5, frame_rate = 25):
    """
    Calculate the rate map for given spikes and positions.

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_no_nan = x[~np.isnan(x)]
    y_no_nan = y[~np.isnan(y)]
    
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    pos_binned, x_edges, y_edges = np.histogram2d(x_no_nan, y_no_nan, bins=[x_bins, y_bins])
    pos_binned = pos_binned/frame_rate
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_x_no_nan = spikes_x[~np.isnan(spikes_x)]
    spikes_y_no_nan = spikes_y[~np.isnan(spikes_y)]
    spikes_binned, _, _ = np.histogram2d(spikes_x_no_nan, spikes_y_no_nan, bins=[x_bins, y_bins])
    

    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    # Removing values with very low occupancy (these sometimes have very large firing rate)
    occupancy_threshold = 0.4
    rmap[smoothed_pos < occupancy_threshold] = np.nan



    return rmap, x_edges, y_edges


def get_ratemaps_restrictedx(spikes, x, y, x_restr, y_restr,  n: int, binsize = 15, stddev = 5, frame_rate = 25):
    """
    Calculate the rate map for given spikes and positions. x_restr and y_restr are used to calculate the occupancy map (since we're restricting over a time interval here)
    used for plot_rmap_interactive

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_no_nan = x_restr[~np.isnan(x_restr)]
    y_no_nan = y_restr[~np.isnan(y_restr)]
    
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    # Pos: only for restricted data
    pos_binned, x_edges, y_edges = np.histogram2d(x_no_nan, y_no_nan, bins=[x_bins, y_bins])
    pos_binned = pos_binned/frame_rate
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_x_no_nan = spikes_x[~np.isnan(spikes_x)]
    spikes_y_no_nan = spikes_y[~np.isnan(spikes_y)]
    spikes_binned, _, _ = np.histogram2d(spikes_x_no_nan, spikes_y_no_nan, bins=[x_bins, y_bins])
    

    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    # Removing values with very low occupancy (these sometimes have very large firing rate)
    occupancy_threshold =0.4
    rmap[smoothed_pos < occupancy_threshold] = np.nan


    return rmap, x_edges, y_edges

