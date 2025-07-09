import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import glob
import string

np.set_printoptions(precision=3, suppress=True)
alphabet = list(string.ascii_uppercase)


def obtain_waveform(basefolder, value='good', unitmatch=False, phyidname=False):
    datapath = os.path.join(basefolder, 'ephys_batched')
    savefolder = os.path.join(basefolder, 'analysis', 'cell_characteristics', 'unit_features')

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    extractSpT(datapath, savefolder, value=value, phyidname=phyidname)
    extractWF(datapath, savefolder, value=value)

def extractSpT(datapath, savefolder, value='good', ksversion=2.5, mod='no', phyidname=False):
    os.makedirs(savefolder, exist_ok=True)

    # Use a shared spiketimes folder for both 'good' and 'mua'
    spiketimes_path = os.path.join(savefolder, 'spiketimes')
    os.makedirs(spiketimes_path, exist_ok=True)

    # Load relevant files
    channelmap = np.load(os.path.join(datapath, 'channel_map.npy'))
    channelposition = np.load(os.path.join(datapath, 'channel_positions.npy'))
    try:
        clusterinfo = np.loadtxt(os.path.join(datapath, 'cluster_group.tsv'), dtype='str')
        clusterheader = clusterinfo[0]
    except:
        clusterinfo = pd.read_table(os.path.join(datapath, 'cluster_group.tsv'))
        print('missing values in cluster_info, opening with pandas')
        clusterheader = clusterinfo.columns.to_numpy()
        clusterinfo = clusterinfo.to_numpy()

    groupcolumn = np.where(clusterheader == 'KSLabel')[0][0]
    clusterinfo = clusterinfo[clusterinfo[:, groupcolumn] == value]

    if mod == 'no':
        allsptime = np.load(os.path.join(datapath, 'spike_times.npy'))
    elif mod == 'yes':
        allsptime = np.load(os.path.join(datapath, 'spike_times_mod.npy'))
    allclusters = np.load(os.path.join(datapath, 'spike_clusters.npy'))

    header = ['ClusterID', 'FileName', 'Shank', 'XCoord', 'YCoord', 'Channel', 'SpikeNum', 'SameDepthUnits', 'Label']
    all_rows = []

    for clu in clusterinfo:
        chheader = np.where(clusterheader == 'ch')[0][0]
        channel = int(clu[chheader])

        try:
            cluid = np.where(clusterheader == 'cluster_id')[0][0]
        except:
            cluid = np.where(clusterheader == 'id')[0][0]

        channelindex = np.where(channelmap == channel)[1][0]
        clusterID = int(clu[cluid])

        depth = int(channelposition[channelindex][1])
        shank = int(np.round(channelposition[channelindex][0] / 250))

        sp = np.array(np.ravel(allsptime[allclusters == clusterID]) / 30, np.float_)

        onebelow = channelposition[channelindex - 1][1]
        if depth == onebelow:
            samechannel = (np.where((np.array(clusterinfo[:, chheader], int) == channel) |
                                     (np.array(clusterinfo[:, chheader], int) == channel - 1)))[0].shape[0]
        else:
            samechannel = (np.where((np.array(clusterinfo[:, chheader], int) == channel) |
                                     (np.array(clusterinfo[:, chheader], int) == channel + 1)))[0].shape[0]

        for i in range(samechannel):
            filename = f"{depth}_{shank}_{alphabet[i]}.npy"
            filepath = os.path.join(spiketimes_path, filename)
            if os.path.exists(filepath):
                continue
            else:
                if phyidname:
                    np.save(os.path.join(spiketimes_path, f"{clusterID}.npy"), sp)
                    title = str(clusterID)
                else:
                    np.save(filepath, sp)
                    title = f"{depth}_{shank}_{alphabet[i]}"
                break

        cluinfo = [
            clusterID, title, shank, channelposition[channelindex][0], channelposition[channelindex][1],
            channel, clu[-2], samechannel, value
        ]
        all_rows.append(cluinfo)

    full_header = np.vstack((header, np.array(all_rows, dtype=object)))
    output_csv = os.path.join(savefolder, 'neuroninfo_all.csv')
    if os.path.exists(output_csv):
        existing = np.loadtxt(output_csv, delimiter=',', dtype=str)
        full_header = np.vstack((existing[1:], full_header[1:]))
        full_header = np.vstack((header, full_header))

    np.savetxt(output_csv, full_header, delimiter=',', fmt='%s')
    print(f'extract SpikeT done for {value}')



def extractWF(datapath, savefolder, value='good', probeChNum=385):
    sppath = os.path.join(savefolder, 'spiketimes')
    infopath = os.path.join(savefolder, 'neuroninfo_all.csv')

    # Correct path definitions (remove leading slashes!)
    if value == 'good':
        savepath_plots = os.path.join(savefolder, 'waveform_plots', 'good')
        savepath_data = os.path.join(savefolder, 'waveform_data', 'good')
    elif value == 'mua':
        savepath_plots = os.path.join(savefolder, 'waveform_plots', 'mua')
        savepath_data = os.path.join(savefolder, 'waveform_data', 'mua')

    # Clean and recreate output folders
    shutil.rmtree(savepath_data, ignore_errors=True)
    shutil.rmtree(savepath_plots, ignore_errors=True)
    os.makedirs(savepath_data, exist_ok=True)
    os.makedirs(savepath_plots, exist_ok=True)

    # Load raw data and neuron info
    rawdatapath = glob.glob(os.path.join(datapath, '*.bin'))[0]
    info = np.loadtxt(infopath, delimiter=',', dtype=str)

    waveformNumExport = 1000

    for neu in info[1:]:
        savename = neu[1]
        channel = int(neu[5])
        cumulativeWaveform = np.zeros((probeChNum, 60)).T

        spaikai = np.array(np.load(os.path.join(sppath, savename + '.npy')) * 30, int)
        numSpikes = min(len(spaikai), waveformNumExport)

        spikeValue = np.random.choice(spaikai.flatten(), size=numSpikes, replace=False) - 20

        for jj in spikeValue:
            data = np.memmap(rawdatapath, np.int16, mode='r+', shape=(60, probeChNum),
                             offset=int(jj) * probeChNum * 2)
            cumulativeWaveform += data
            del data

        meanWaveform = cumulativeWaveform / numSpikes
        np.save(os.path.join(savepath_data, savename + '.npy'), meanWaveform)

        # Plotting
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), facecolor='white')
        wfheatmap = np.zeros(60)

        for ii in np.arange(-15, 15):
            row = channel + ii
            if 0 <= row < probeChNum:
                ax[0].plot(meanWaveform.T[row])
                wfheatmap = np.vstack((wfheatmap, meanWaveform.T[row]))

        wfheatmap = wfheatmap[1:]
        ax[1].imshow(wfheatmap, aspect='auto', interpolation='None')
        ax[1].grid(False)
        ax[0].grid(False)
        plt.savefig(os.path.join(savepath_plots, savename + '_plot.png'), dpi=300)

    print('extract WF done')
