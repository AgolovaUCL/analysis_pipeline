import matplotlib.pyplot as plt
import numpy as np

occ_b1 = np.zeros(12)
occ_b1[4] = 5
occ_b1[9] = 10

occ_b2 = np.zeros(12)
occ_b2[4] = 10
occ_b2[6] = 5

spk_b1 = np.zeros(12)
spk_b1[4] = 10

spk_b2 = np.zeros(12)
spk_b2[6] = 10

dir_fr1 = np.divide(spk_b1, occ_b1, out=np.zeros_like(spk_b1), where=occ_b1!= 0)
dir_fr2 = np.divide(spk_b2, occ_b2, out=np.zeros_like(spk_b2), where=occ_b2!= 0)

f1 = occ_b1 * np.sum(spk_b1)
f2 = occ_b2 * np.sum(spk_b2)

rel_dir_ctr_dist = f1 + f2
rel_dir_dist = spk_b1 + spk_b2

normalised_dist = np.divide(rel_dir_dist, rel_dir_ctr_dist, out=np.zeros_like(rel_dir_dist), where=rel_dir_ctr_dist!=0)

direction_bins = np.linspace(-np.pi, np.pi, 13)
bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2
print(direction_bins)

arr = [spk_b1,  occ_b1, dir_fr1, spk_b2, occ_b2, dir_fr2, rel_dir_dist, rel_dir_ctr_dist, normalised_dist]
titles = ['Spikes bin 1', 'Occupancy bin 1', 'Directional firing rate bin 1', 'Spikes bin 2', 'Occupancy bin 2', 'Directional firing rate bin 2', 'Rel Dir dist', 'Rel Dir Ctrl Dist', 'Final normalised distribution']
c = ['g', 'b', 'orange']
c = np.tile(c, 3)
fig, ax = plt.subplots(3,3, figsize = [13, 13], subplot_kw={'projection': 'polar'})
ax = ax.flatten()

for i in range(len(arr)):
    width = np.diff(bin_centers)[0]
    ax[i].bar(
        bin_centers,
        arr[i],
        width=width,
        facecolor = c[i],
        bottom=0.0,
        alpha=0.8
    )
    ax[i].set_title(titles[i])
plt.tight_layout()
plt.show()

