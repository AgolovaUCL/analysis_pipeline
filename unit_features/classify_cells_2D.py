import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.cluster.vq import kmeans2

def classify_cells_2D(derivatives_base, analyse_only_good=True):
    """
    Classifies cells into pyramidal and interneurons based on both 
    peak-to-valley time and firing rate using 2D k-means clustering.

    Inputs:
    ----------
    derivatives_base : str
        Path to the base directory containing analysis results.
    analyse_only_good : bool, default=True
        If True, analyzes only 'good' neurons.

    Output files:
    --------------
    - unit_metrics_classified_2D.csv : includes new 'type_2D' column
    - pyramidal_units_2D.csv : list of pyramidal units
    - cluster_scatter_2D.png : scatter plot of clustering results
    """

    # Paths
    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    wf_path = os.path.join(unit_features_path, "all_units_overview", "unit_waveform_metrics.csv")
    metrics_path = os.path.join(unit_features_path, "all_units_overview", "unit_metrics.csv")

    # Load data
    df_wf = pd.read_csv(wf_path)
    df_metrics = pd.read_csv(metrics_path)

    # Filter to "good" units if needed
    if analyse_only_good:
        
        labels_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting", "sorter_output", "cluster_group.tsv")
        labels_df = pd.read_csv(labels_path, sep="\t")
        good_units = labels_df[labels_df['group'] == 'good']['cluster_id'].values
        print(f"Number of good units: {len(good_units)}")
        df = df_wf[df_wf['unit_ids'].isin(good_units)].copy()
        """
        good_units_path = r"S:\Honeycomb_maze_task\rawdata\sub-003_id-2F\ses-01_date-17092025\task_metadata\good_unit_ids.csv"
        good_units_df = pd.read_csv(good_units_path)
        good_units = good_units_df['unit_ids']
        good_units = np.array(good_units)
        """

    else:
        df = df_wf.copy()

    # Add firing rate info from unit_metrics
    df = df.merge(df_metrics[['unit_ids', 'firing_rate']], on='unit_ids', how='left')

    # Convert peak-to-valley to ms
    df['peak_to_valley'] = 1000 * df['peak_to_valley']

    # Prepare data for clustering
    X = df[['peak_to_valley', 'firing_rate']].dropna().values

    # Run k-means
    centroids, labels = kmeans2(X, 2, minit='points')
    # Label assignment
    # Pyramidal cells → lower firing rate on average
    if centroids[0, 1] < centroids[1, 1]:
        label_order = ['pyramidal', 'interneuron']
    else:
        label_order = ['interneuron', 'pyramidal']

    df['cluster_label'] = labels
    df['type_2D'] = [label_order[label] for label in labels]

    # Plot results
    plt.figure(figsize=(8, 6))
    for label, color in zip(['pyramidal', 'interneuron'], ['orange', 'blue']):
        subset = df[df['type_2D'] == label]
        plt.scatter(subset['peak_to_valley'], subset['firing_rate'], alpha=0.6, label=label, color=color)

    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.xlabel('Peak-to-Valley Time (ms)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('2D Clustering of Cells by Waveform and Firing Rate')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(unit_features_path, "all_units_overview", "cluster_scatter_2D.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"Saved clustering plot to {plot_path}")

    # Assign labels to full metrics dataframe
    df_metrics['type_2D'] = np.nan
    for unit_id in df_metrics['unit_ids']:
        match = df[df['unit_ids'] == unit_id]
        if not match.empty:
            df_metrics.loc[df_metrics['unit_ids'] == unit_id, 'type_2D'] = match['type_2D'].iloc[0]
        else:
            df_metrics.loc[df_metrics['unit_ids'] == unit_id, 'type_2D'] = 'unclassified'

    # Save results
    output_metrics_path = os.path.join(unit_features_path, "all_units_overview", "unit_metrics_classified_2D.csv")
    df_metrics.to_csv(output_metrics_path, index=False)
    print(f"Saved classified metrics to {output_metrics_path}")

    # Save pyramidal units separately
    pyramidal_units = df_metrics[df_metrics['type_2D'] == 'pyramidal']['unit_ids']
    pyramidal_units_path = os.path.join(unit_features_path, "all_units_overview", "pyramidal_units_2D.csv")
    pd.DataFrame(pyramidal_units, columns=['unit_ids']).to_csv(pyramidal_units_path, index=False)
    print(f"Saved pyramidal unit IDs to {pyramidal_units_path}")


if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-03_date-17092025\all_trials"
    classify_cells_2D(derivatives_base, analyse_only_good=True)
