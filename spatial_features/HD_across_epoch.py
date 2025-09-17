import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
def sig_across_epochs(derivatives_base, trials_to_include, num_epochs = 3):
    """


    Args:
        derivatives_base (_type_): _description_
    """
    
    path = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data')
    df_options = glob.glob(os.path.join(path, "directional_tuning*.csv"))
    
    if len(df_options) == 1:
        df_path = df_options[0]
    else:
        print([os.path.basename(f) for f in df_options])
        user_input = input('Please provide the number of the file in the list you would like to look at (starting at 1): ')
        user_input = np.int32(user_input)
        df_path = df_options[user_input - 1]
        print(f"Making roseplot from data from {os.path.basename(df_options[user_input - 1])}")

    df  = pd.read_csv(df_path)
    # Initialise empty to store results of data with trial and epoch column
    df_results = pd.DataFrame(columns = ['trial', 'epoch', 'proportion'])
    
    for tr in trials_to_include:
        for e in np.arange(2, num_epochs + 1):
            # Filter so epoch in df is e-1 and so its only significant == sig
            df_filtered = df[(df['epoch'] == e - 1) & (df['significant'] == 'sig') & (df['trial'] == tr)]
            
            cells = df_filtered['cell'].unique()

            df_e = df[(df['epoch'] == e) & (df['significant'] == 'sig') & (df['trial'] == tr)]
            cells_e = df_e['cell'].unique()
            
            # Count what proportion of cells in cells is in cells_e
            proportion = len(np.intersect1d(cells, cells_e)) / len(cells) if len(cells) > 0 else 0
            
            df_results = pd.concat([df_results, pd.DataFrame({'trial': [tr], 'epoch': [e], 'proportion': [proportion]})], ignore_index=True)

    # Make heatmap plots
    # Make heatmap plots
    pivoted = df_results.pivot(index='trial', columns='epoch', values='proportion')

    plt.figure(figsize=(5, 6))
    plt.title('Significant Cells Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Trial')

    im = plt.imshow(pivoted, aspect='auto', cmap='viridis', origin='upper', vmin=0, vmax=1)
    plt.colorbar(im, label='Proportion of Significant Cells')

    # Fix ticks
    plt.xticks(np.arange(len(pivoted.columns)), pivoted.columns)
    plt.yticks(np.arange(len(pivoted.index)), pivoted.index)

    plt.savefig(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'sig_across_epochs.png'))
    plt.show()
    # Save results to CSV
    df_results.to_csv(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data', 'sig_across_epochs.csv'), index=False)
    
    
    # ====== Significance across trials======= 
    df_results = pd.DataFrame(columns = ['trial I','trial II', 'proportion'])
    
    for i in np.arange(0, len(trials_to_include)):
        for j in np.arange(0, len(trials_to_include)):
            tr = trials_to_include[i]
            tr_2 = trials_to_include[j]
            df_filtered = df[(df['significant'] == 'sig') & (df['trial'] == tr)]
            
            cells = df_filtered['cell'].unique()

            df_e = df[(df['significant'] == 'sig') & (df['trial'] == tr_2)]
            cells_e = df_e['cell'].unique()
            
            # Count what proportion of cells in cells is in cells_e
            intersection = len(np.intersect1d(cells, cells_e))
            union = len(np.union1d(cells, cells_e))

            proportion = intersection / union if union > 0 else 0

            df_results = pd.concat([df_results, pd.DataFrame({'trial I': [tr], 'trial II': [tr_2], 'proportion': [proportion]})], ignore_index=True)

    pivoted = df_results.pivot(index='trial I', columns='trial II', values='proportion')

    plt.figure(figsize=(7, 6))
    plt.title('Significant Cells Across Trials')
    plt.xlabel('Trial II')
    plt.ylabel('Trial I')

    im = plt.imshow(pivoted, aspect='auto', cmap='viridis', origin='upper', vmin=0, vmax=1)
    plt.colorbar(im, label='Proportion of Significant Cells Shared')

    # Fix ticks
    plt.xticks(np.arange(len(pivoted.columns)), pivoted.columns)
    plt.yticks(np.arange(len(pivoted.index)), pivoted.index)

    plt.savefig(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'sig_across_trials.png'))
    plt.axis('scaled')
    plt.show()
    
    # Save results to CSV
    df_results.to_csv(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data', 'sig_across_trials.csv'), index=False)
    
    
if __name__ == "__main__":
    """
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1, 11)
    """
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\all_trials"
    trials_to_include = np.arange(1,11)
    sig_across_epochs(derivatives_base, trials_to_include)
   