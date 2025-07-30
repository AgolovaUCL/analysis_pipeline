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
from scipy.cluster.vq import kmeans2

def classify_cells(derivatives_base):
    """
    Classifies cells into pyramidal and interneurons based on their peak-to-valley time.
    Inputs:
    derivatives_base: path to the base directory containing the analysis results

    Returns:
    
    Notes:
    Function finds unit_waveform_metrics.csv and performs k-means clustering on the peak-to-valley time
    
    """

    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    csv_path = os.path.join(unit_features_path,"all_units_overview", "unit_waveform_metrics.csv")
    df_original = pd.read_csv(csv_path)
    df = df_original[df_original['label'] == 'good']

    csv_path = os.path.join(unit_features_path,"all_units_overview", "unit_metrics.csv")
    df_metrics_original = pd.read_csv(csv_path)


    # Adjusting it to ms
    df['peak_to_valley'] = 1000*df['peak_to_valley']
    
    # Classification
    x = df['peak_to_valley'].values.reshape(-1, 1) 

    # Run kmeans2
    centroids, labels = kmeans2(x, 2, minit='points')

    if centroids[0] < centroids[1]:
        label_order = ['interneurons', 'pyramidal']
        label_int = 0
    else:
        label_order = ['pyramidal', 'interneurons']
        label_int = 1

    renamed_labels = np.array([label_order[label] for label in labels])
    # Plotting
    df['kmeans_group'] = renamed_labels
    plt.hist(df.loc[df['kmeans_group']=='interneurons', 'peak_to_valley'], bins=30, alpha=0.6, label='interneuron')
    plt.hist(df.loc[df['kmeans_group']=='pyramidal', 'peak_to_valley'], bins=30, alpha=0.6, label='pyramidal')
    plt.axvline(centroids[label_int], color='blue', linestyle='--')
    plt.axvline(centroids[1- label_int], color='orange', linestyle='--')
    plt.legend()
    plt.xlabel('Peak to Valley Time (ms)')
    plt.ylabel('Cell count')
    plt.title('Cell clustering by peak-to-valley time')
    plt.show()


    plt.show()

#derivatives_base = r"Z:\Eylon\Data\Spatiotemporal_Task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trials"
derivatives_base =   r"Z:\Eylon\Data\Spatiotemporal_Task\derivatives\sub-002_id-1U\ses-02_date-03072025\all_trials"
classify_cells(derivatives_base)