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

def classify_cells(derivatives_base, analyse_only_good = True):
    """
    Classifies cells into pyramidal and interneurons based on their peak-to-valley time.
    
    Inputs:
    derivatives_base: path to the base directory containing the analysis results
    analyse_only_good: if True, leads to analysis of only good neurons
    
    Returns:
    None

    Notes:
    Function finds unit_waveform_metrics.csv and performs k-means clustering on the peak-to-valley time. 
    User selects a cutoff value in the plot and classification occurs based on that
    """

    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    csv_path = os.path.join(unit_features_path,"all_units_overview", "unit_waveform_metrics.csv")
    df_original = pd.read_csv(csv_path)
    
    if analyse_only_good:
        df = df_original[df_original['label'] == 'good']
    else:
        df = df_original
    
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
    xy_coords = plt.ginput(1)
    cutoff = xy_coords[0][0]
    print(f"Cutoff value set to {cutoff:.2f}ms")
    plt.close()

    counter = 0
    df_metrics_original['type'] = pd.Series([np.nan] * len(df_metrics_original), dtype=object)
    for unit_id in df_metrics_original['unit_ids']:
        row = df[df['unit_ids'] == unit_id]
        
        if row.empty:
            df_metrics_original.loc[df_metrics_original['unit_ids'] == unit_id, 'type'] = 'unclassified'
        elif row['peak_to_valley'].iloc[0] > cutoff:
            df_metrics_original.loc[df_metrics_original['unit_ids'] == unit_id, 'type'] = 'pyramidal'
            counter += 1
        else:
            df_metrics_original.loc[df_metrics_original['unit_ids'] == unit_id, 'type'] = 'interneuron'

    print(f"Found {counter} pyramidal neurons")
    df_metrics_original.to_csv(csv_path)
    print(f"Saved dataframe to {csv_path}")

    plot_output_path = os.path.join(unit_features_path, "all_units_overview", "firing_rate_histogram.png")
    plt.hist(df_metrics_original.loc[df_metrics_original['type']=='interneuron', 'firing_rate'], bins=30, alpha=0.6, label='interneuron')
    plt.hist(df_metrics_original.loc[df_metrics_original['type']=='pyramidal', 'firing_rate'], bins=30, alpha=0.6, label='pyramidal')
    plt.title('Firing rate for pyramidal cells and interneurons')
    plt.legend()
    plt.savefig(plot_output_path)
    plt.show()
if __name__ == "__main__":
    #derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\all_trials"
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-04_date-17072025\all_trials"
    classify_cells(derivatives_base, analyse_only_good=True)