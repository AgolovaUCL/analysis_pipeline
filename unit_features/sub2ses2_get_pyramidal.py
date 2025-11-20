import numpy as np
import pandas as pd
# Getting pyramidal units for sub2ses2

excel_path = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials\analysis\cell_characteristics\unit_features\all_units_overview\pyramidal units without questionsmarks.xlsx"
df = pd.read_excel(excel_path)
data = df.iloc[:,0].to_numpy()

int_elemnts = []
str_elements = []
for el in data:
    if isinstance(el, int):
        int_elemnts.append(el)
        if el > 373:
            breakpoint()
    else:
        str_elements.append(el)

for el in str_elements:
    vals = el.split("-")
    for v in range(int(vals[0]), int(vals[1]) + 1):
        int_elemnts.append(v)
        if v > 373:
            breakpoint()

good_units_path = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials\ephys\concat_run\sorting\sorter_output\cluster_group.tsv"
good_units = pd.read_csv(good_units_path, sep="\t")
good_units_rows = good_units.loc[int_elemnts, :]
good_units_final = good_units_rows[good_units_rows['group'] == 'good']
good_clusters = good_units_final['cluster_id']
print(good_clusters)
output_path = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials\analysis\cell_characteristics\unit_features\pyramidal_units_2D.csv"
good_clusters.to_csv(output_path, index=False)