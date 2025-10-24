import numpy as np
import os
import pandas as pd
file_path = r"S:\Honeycomb_maze_task\rawdata\sub-003_id-2F\ses-01_date-17092025\task_metadata\good_units_2F_Ses1.xlsx"
df = pd.read_excel(file_path, header = None)
types = df.iloc[:,0]
types = np.array(types)
ind = np.where(pd.notna(types))[0]
ind = [el +1 for el in ind]

df = pd.DataFrame(ind, columns=['unit_ids'])

df.to_csv( r"S:\Honeycomb_maze_task\rawdata\sub-003_id-2F\ses-01_date-17092025\task_metadata\good_unit_ids.csv", index=False)
