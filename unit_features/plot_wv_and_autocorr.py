import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import glob
import string

def plot_wv_and_autocorr(derivates_folder, total_trial_length):
    # params
    sample_rate = 30000
    frame_rate = 30

    # path to save the plots
    batched_data_path = os.path.join(derivates_folder, 'ephys_batched')
    figures_data_folder = os.path.join(batched_data_path, 'cells')
    
    # give error if path does not exist
    if not os.path.exists(batched_data_path):
        raise FileNotFoundError(f"The directory {batched_data_path} does not exist.")
    if not os.path.exists(figures_data_folder):
        os.makedirs(figures_data_folder, exist_ok=True)
    
    for category in ['good', 'mua']:
            # if not exist, create the folder
            if not os.path.exists(os.path.join(figures_data_folder, category)):
                os.makedirs(os.path.join(figures_data_folder, category), exist_ok=True)

    # load the files from the merged bin directory
    data = pd.read_csv(os.path.join(batched_data_path , 'cluster_KSLabel.tsv'), sep='\t')
    cluster_num = data.iloc[:, 0].values
    labels = data.iloc[:, 1].values


    # Load numpy files
    templates = np.load(os.path.join(batched_data_path, 'templates.npy'))
    spike_times = np.load(os.path.join(batched_data_path, 'spike_times.npy'))
    spike_clusters = np.load(os.path.join(batched_data_path, 'spike_clusters.npy'))
    merged_path = os.path.join(batched_data_path, 'merged.bin')

    try:
        merged_data = open(merged_path, 'rb')
    except IOError:
        raise RuntimeError(f"Could not open merged.bin at {merged_path}")

        

    for i in range(len(cluster_num)):
        try:
            cell_nr = cluster_num[i]
            label = labels[cell_nr]  

            if label == 'mua':
                output_path = os.path.join(figures_data_folder, 'mua')
            elif label == 'good':
                output_path = os.path.join(figures_data_folder, 'good')
        
            print(f'Currently at cell {cell_nr}')

            spike_times_cell = spike_times[spike_clusters == cell_nr]
            cell_spike_coords = np.round(spike_times_cell.astype(float) * frame_rate / sample_rate).astype(int)
            firing_rate = len(cell_spike_coords) / total_trial_length
            tc = spike_times_cell/firing_rate

            if len(tc) > 1e4:
                tc = tc[:int(1e4)]

            fig = plt.figure(figsize=(9, 18))
            fig.suptitle(f'Cell {cell_nr}, label: {label}, firing rate: {firing_rate:.1f} Hz, number of spikes: {len(cell_spike_coords)}')

            # Autocorrelogram 10 ms window
            plt.subplot(3, 1, 1)
            bins, cou = interspike_histogram(tc, tc, 10)
            binwidth = bins[1] - bins[0]  # assuming uniform spacing
            plt.bar(bins, cou, width=binwidth, align='center')
            plt.title(f'{np.sum(cou[51:60])}/{len(tc)}')
            plt.xlim([-11, 11])

            # Autocorrelogram 500 ms window
            plt.subplot(3, 1, 2)
            bins, cou = interspike_histogram(tc, tc, 500)
            binwidth = bins[1] - bins[0]
            plt.bar(bins, cou, width=binwidth, align='center')
            plt.xlim([-550, 550])

            # Waveform

            num_channels = 384
            num_samples = 50
            num_spikes = min(100, len(spike_times_cell))

            hoopla = np.zeros((num_channels, num_samples), dtype=np.float64)

            for uoi in range(num_spikes):
                spike_index = int(spike_times_cell[uoi]) - 10  # start 10 samples before
                if spike_index < 0:
                    continue  # skip if spike too early in the recording
                offset = spike_index * num_channels * 2  # int16 = 2 bytes
                merged_data.seek(offset)
                chunk = np.frombuffer(merged_data.read(num_channels * num_samples * 2), dtype=np.int16)
                if chunk.size != num_channels * num_samples:
                    continue  # incomplete read near file end
                chunk = chunk.reshape((num_channels, num_samples), order='F')
                hoopla += chunk

            # Offset channels vertically for plotting
            gap = 50
            apo_cr = np.tile(np.arange(0, num_channels * gap, gap).reshape(-1, 1), (1, num_samples))
            hoopla_appended = (-0.05 * hoopla + apo_cr).T  # transpose to [time x channel]

            # Find the best (most dynamic) channel
            col_ranges = np.ptp(hoopla_appended, axis=0)
            best_column = np.argmax(col_ranges)
            largest_wave = hoopla_appended[:, best_column]

            # Time axis in ms
            time = np.arange(num_samples) / sample_rate * 1000

            # Plot
            plt.subplot(3,1,3)
            plt.plot(time, largest_wave, 'k')
            plt.xlabel('Time (ms)')
            plt.title('Waveform')
            plt.ylim([0, num_channels * gap])
            plt.tight_layout()


            filename = f'cell_{cell_nr:03d}.png'
            save_path = os.path.join(output_path, filename)
            plt.savefig(save_path)
            plt.close(fig)

        except Exception as e:
            print(f'Error with cell {cell_nr}: {e}')
            continue

    merged_data.close()



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

