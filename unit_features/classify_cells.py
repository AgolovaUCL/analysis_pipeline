import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
import probeinterface
from pathlib import Path


def classify_cells(derivatives_base,  gains = 10, sampling_rate = 30000):
    print("=== Classifying cells ===")
    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    csv_path = os.path.join(unit_features_path,"all_units_overview", "unit_waveform_metrics.csv")
    df = pd.read_csv(csv_path)
    df['peak_to_valley'] = 1000*df['peak_to_valley']
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    ax = ax.flatten()
    ax[0].hist(df['peak_to_valley'], color = 'blue', edgecolor = 'black', bins =20)
    ax[0].set_title('Distribution of peak to valley time (ms) (all units)')
    min_pv = min(df['peak_to_valley'])
    max_pv = max(df['peak_to_valley'])
    ax[0].set_xlim(min_pv, max_pv)

    
    ax[1].hist(df['peak_trough_ratio'], color = 'blue', edgecolor = 'black', bins = 20)
    ax[1].set_title('Distribution of peak to trough ratio (all units)')
    min_pt = min(df['peak_trough_ratio'])
    max_pt = max(df['peak_trough_ratio'])
    ax[1].set_xlim(min_pt, max_pt)

    df = df[df['label'] == 'good']
    ax[2].hist(df['peak_to_valley'], color = 'lightblue', edgecolor = 'black', bins =20)
    ax[2].set_title('Distribution of peak to valley time (ms) (good units)')
    ax[2].set_xlim(min_pv, max_pv)

    ax[3].hist(df['peak_trough_ratio'], color = 'lightblue', edgecolor = 'black', bins = 20)
    ax[3].set_title('Distribution of peak to trough ratio (good units)')
    ax[3].set_xlim(min_pt, max_pt)

    plt.show()

#derivatives_base = r"Z:\Eylon\Data\Spatiotemporal_Task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
derivatives_base =  r"Z:\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-001_id-2B\ses-05_test\all_trials"
classify_cells(derivatives_base,  gains = 10, sampling_rate = 30000)