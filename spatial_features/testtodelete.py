import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 3)
axs = axs.flatten()
counter = 2

fig.delaxes(axs[counter])
axs[counter] = fig.add_subplot(2, 3, counter+1, polar=True)
theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
bars = axs[counter].bar(theta, np.random.rand(12), width=0.4)
axs[counter].set_theta_zero_location('N')
axs[counter].set_theta_direction(-1)
plt.show()