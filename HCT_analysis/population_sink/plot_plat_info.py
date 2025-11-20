import numpy as np
import pandas as pd
from matplotlib.path import Path
import os
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from tqdm import tqdm

def plot_plat_info(allfiring_rates, allnorm_MRL, allscales, hcoord, vcoord):
    """
    Used in popsink new method to plot different information 
    """
    fig, axs = plt.subplots(1, 3, figsize=(8,8))
    axs = axs.flatten()
    cmap = plt.cm.RdYlGn
    
    firingrate_pp = []
    firingrate_ppmean = []
    for p in range(61):
        firingrate = [el[p] for el in allfiring_rates]
        firingrate_pp.append(firingrate)
        firingrate_ppmean.append(np.nanmean(firingrate))
    for j in range(3): 
        ax = axs[j]
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            colour = 'grey'
            
            hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                                orientation=np.radians(28),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor='k')
            ax.text(x, y, i + 1, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
        # plot the goal positions
        ax.set_aspect('equal')
        
        # Add small text with MRL and angle on bottom
    plt.show()          