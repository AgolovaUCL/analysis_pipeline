import numpy as np


bin_1_occ = np.zeros(12)
bin_1_occ[4] = 1
bin_1_occ[9] = 9

bin_2_occ = np.zeros(12)
bin_2_occ[4] = 9
bin_2_occ[6] = 1

bin_1_spikes = np.zeros(12)
bin_1_spikes[4] = 10

bin_2_spikes = np.zeros(12)
bin_2_spikes[6] = 10

rel_dir_ctrl_dist = bin_1_occ*np.sum(bin_1_spikes) + bin_2_occ*np.sum(bin_2_spikes)
print()
print(rel_dir_ctrl_dist)