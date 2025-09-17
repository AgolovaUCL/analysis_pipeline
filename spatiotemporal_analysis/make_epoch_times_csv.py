
import numpy as np
import os
import glob
import pandas as pd


def make_epoch_times_csv(rawsession_folder, trials_to_include):
    
    epoch_times_path = os.path.join(rawsession_folder, 'task_metadata', 'epoch_times.csv')

    if os.path.exists(epoch_times_path):
        pass
    else:
        csv_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.csv'))
        if len(csv_path) > 0:
            epoch_times_allcols = pd.read_csv(csv_path[0], header=None)
        else:
            excel_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.xlsx'))
            if len(excel_path) > 0:
                epoch_times_allcols = pd.read_excel(excel_path[0], header=None)
            else:
                raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')

    
    epoch_times= epoch_times_allcols.iloc[:, [10, 12, 14, 16, 18]]
    epoch_times.columns = ['epoch 1 end', 'epoch 2 start', 'epoch 2 end', 'epoch 3 start', 'epoch 3 end']
    epoch_times.insert(0, "epoch 1 start", np.zeros(len(epoch_times)))
    epoch_times.insert(0,'trialnumber',  trials_to_include)

    epoch_times.to_csv(epoch_times_path, index=False)
    print(f"Data saved to {epoch_times_path}")
    
if __name__ == "__main__":
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-05_date-18072025"
    trials_to_include = np.arange(1,11)
    make_epoch_times_csv(rawsession_folder, trials_to_include)
    