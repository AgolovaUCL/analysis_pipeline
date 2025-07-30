import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
import probeinterface
import os
import time
from scipy.io import savemat


from spikeinterface.qualitymetrics import (
    compute_snrs,
    compute_firing_rates,
    compute_isi_violations,
    calculate_pc_metrics,
    compute_quality_metrics,
)
from spikeinterface.comparison import compare_sorter_to_ground_truth
from probeinterface import Probe, ProbeGroup
from pathlib import Path
from tqdm import tqdm


def run_spikeinterface_features(derivatives_base, run_analyzer_from_memory = False, gains = 10, sampling_rate = 30000):
    print("=== Running spikeinterface features ===")
    recording_path = os.path.join(derivatives_base, "concat_run", "preprocessed", "traces_cached_seg0.raw")
    probe_path =  os.path.join(derivatives_base, "concat_run", "preprocessed", "probe.json")
    kilosort_output_path = os.path.join(derivatives_base, "concat_run","sorting", "sorter_output" )
    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    spikeinterface_recording_path = os.path.join(derivatives_base, "concat_run","sorting", "spikeinterface_recording.json" )


    kilosort_output_path = Path(kilosort_output_path)
    if not os.path.exists(unit_features_path):
        os.makedirs(unit_features_path)

    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")
    if not os.path.exists(analyzer_path):
        os.makedirs(analyzer_path)

    # Obtain gain_to_uV and offset value
    with open(spikeinterface_recording_path, "r") as f:
        data = json.load(f)



    try:
        # Format option 1
        gain_to_uV = data["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording_list"][0]["properties"]["gain_to_uV"][0]
        offset_to_uV = data["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording_list"][0]["properties"]["offset_to_uV"][0]
    except (KeyError, IndexError, TypeError):
        # format option 2
        parent = data['kwargs']['parent_recording']
        properties = parent['properties']
        gain_to_uV = properties['gain_to_uV'][0]
        offset_to_uV = properties['offset_to_uV'][0]

    print("gain_to_uV:", gain_to_uV)
    print("offset_to_uV:", offset_to_uV)

    # Loading recording data
    print("Loading recording")
    recording = se.read_binary(
        file_paths = recording_path,
        # Info below is found in the json file in the same folder
        sampling_frequency=sampling_rate,
        dtype = np.int16,  
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
        num_channels = 384,
        )
    total_samples = recording.get_num_frames()
    sampling_rate = recording.get_sampling_frequency()
    total_duration_sec = total_samples / sampling_rate

    formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_duration_sec))
    print(f"Total trial length: {formatted_time}")
    
    time_output_folder = unit_features_path = os.path.join(derivatives_base, "analysis", "metadata")
    if not os.path.exists(time_output_folder):
        os.makedirs(time_output_folder)
    time_output_path = os.path.join(time_output_folder, "trialduration.txt")

    with open(time_output_path, "w") as f:
        f.write('%f' % total_duration_sec)

    # Loading kilosort data
    print("Loading kilosort data")
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )

    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')
    good_units_ids = [el for el in unit_ids if labels[el] == 'good']
    colour_scheme = ['blue' if labels[el] == 'good' else 'red' for el in unit_ids]
    colour_scheme_good_units = ['blue' for el in unit_ids if labels[el] == 'good']
    # Loading probe data. Assumes one probe
    probe_group = probeinterface.read_probeinterface(probe_path)
    probe = probe_group.probes[0]   
    recording = recording.set_probe(probe)

    unit_id = 307
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train = np.float64(spike_train_unscaled/sampling_rate)
    

    bins, cou = interspike_histogram(spike_train, spike_train, 50)
    binwidth = bins[1] - bins[0]

    max_peak_10 = np.max(cou[51:61]) # maximum value in first 10 ms
    mean_val_40_50 = np.mean(cou[90:]) # mean value in 40-50 ms

    burst_index_temp = max_peak_10 - mean_val_40_50

    if burst_index_temp > 0:
        burst_index = burst_index_temp/max_peak_10
    elif burst_index_temp < 0:
        burst_index = burst_index_temp/mean_val_40_50
    else:
        burst_index = 0

    return burst_index


def interspike_histogram(spkTr1, spkTr2, maxInt):
    """
    Calculate the interspike histogram between two spike trains.
    Python version of Marius' code

    Inputs:
    spkTr1: Spike train 1 
    spkTr2: Spike train 2
    maxInt: Maximum interval for histogram in ms

    Outputs:
    bin_centers: Centers of the histogram bins
    counts: Counts of spikes in each bin
    """
    # Convert to ms
    spkTr1 = spkTr1 * 1000
    spkTr2 = spkTr2 * 1000
    n_divisions = 50 # Default

    # Finding intervals
    nSpk = len(spkTr1)

    int_matrix = np.zeros((nSpk, nSpk - 1))

    for ii in range(1, nSpk):
        # Shifting spkTr2 by ii positions
        shifted = np.roll(spkTr2, ii)
        # Fill column ii-1 with element-wise difference
        int_matrix[:, ii - 1] = spkTr1 - shifted

    binwidth = maxInt / n_divisions
    bins = np.arange(-maxInt, maxInt + binwidth, binwidth)
    if nSpk == 0 or len(spkTr2) == 0:
        # This is necessary for graceful failure when nSpk=0
        counts = np.full_like(bins, np.nan)
    else:
        # This is normal
        counts, _ = np.histogram(int_matrix.flatten(), bins=bins)

    bin_centers = bins[:-1] + binwidth / 2

    return bin_centers, counts


run_from_memory = True
#derivatives_base = r"Z:\Eylon\Data\Spatiotemporal_Task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
derivatives_base = r"Z:\Eylon\Data\Spatiotemporal_Task\derivatives\sub-002_id-1U\ses-02_date-03072025\all_trials"
run_spikeinterface_features(derivatives_base, run_analyzer_from_memory = run_from_memory, gains = 10, sampling_rate = 30000)
