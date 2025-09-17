import numpy as np
import pandas as pd
from matplotlib.path import Path
import os

"""
Different utilities relating to finding platforms locations etc.
"""

def hex_grid(radius):
    # Finds coordinates of platforms on hexagonal grid
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

def calculate_cartesian_coords(coord, hex_side_length):
    # Calculates cartesian coordinates from hexagonal coordinates
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord

# PARAMETERS FOR OUR CAMERA
radius = 4
hex_side_length = 87# Length of the side of the hexagon
theta = np.radians(305)  # Rotation angle in radians
desired_x, desired_y = 1320, 950

# Generate coordinates for a large hexagon with radius 4
coord = hex_grid(radius)

# Calculate initial Cartesian coordinates
hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)
# Rotate the coordinates
hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord2, vcoord2)]
vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord2, vcoord2)]
vcoord_rotated = [-v for v in vcoord_rotated]
# Calculate the translation needed to align the first rotated coordinate
dx = desired_x - hcoord_rotated[30]
dy = desired_y - vcoord_rotated[30]
# Apply the translation
hcoord_translated = [x + dx for x in hcoord_rotated]
vcoord_translated = [y + dy for y in vcoord_rotated]




# Updated code to find the hexagon the rat is in
def get_platform_number(rat_locx, rat_locy, hcoord, vcoord):
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if is_point_in_platform(rat_locx, rat_locy, x, y):
            return i + 1
    return -1 

def get_platform_center(platform):
    return [hcoord_translated[platform-1], vcoord_translated[platform-1]]

def get_nearest_platform(rat_locx, rat_locy, hcoord, vcoord):
    platform = get_platform_number(rat_locx, rat_locy, hcoord, vcoord)
    if platform != -1:
        return platform
    else:
        min_dist = 10**5
        closest_platform = 0
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            dist = np.sqrt((rat_locx - x)**2 + (rat_locy - y)**2)
            if dist < min_dist:
                closest_platform = i + 1
                min_dist = dist
        return closest_platform

def is_point_in_platform(rat_locx, rat_locy, hcoord, vcoord, hex_side_length=hex_side_length):
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            hcoord + hex_side_length * np.cos(angle),
            vcoord + hex_side_length * np.sin(angle)
        ])
    hexagon_path = Path(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

def add_platforms_csv(csv_path):
    df = pd.read_csv(csv_path)

    # FIRST CHECK IF THIS IS ALREADY THER, NOT SURE HOW
    platforms = []
    for i in range(len(df)):
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]
        if np.isnan(x) or np.isnan(y):
            platforms.append(np.nan)
        else:
            platform = get_nearest_platform(x, y, hcoord_translated, vcoord_translated)
            platforms.append(platform)

    df['platform'] = platforms
    df.to_csv(csv_path)
    
    
    
def get_goals_coords(goals):
    return [get_platform_center(goals[0]), get_platform_center(goals[1])]

def calculate_occupancy_plats(pos_data):
    
    platforms_occupancy = []
    for i in range(61):
        platforms_i = pos_data[pos_data['platforms'] == i + 1]
        occupancy_i = len(platforms_i)
        platforms_occupancy.append(occupancy_i)
    return platforms_occupancy

def get_firing_rate_platforms(spike_train, pos_data, platform_occupancy):
    platforms = pos_data['platforms']
    platforms_spk = platforms[spike_train]
    firing_rate = []

    for p in np.arange(1,62):
        # Filter platforms_spk for platform = p
        platform_p = platforms_spk[platforms_spk == p]

        # Compute firing rate
        rate = len(platform_p)/platform_occupancy[p-1]

        firing_rate.append(rate)

    return firing_rate

def get_hd_distribution(hd, num_bins):
    hd_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    hd_hist, bin_edges = np.histogram(hd, bins=hd_bins)
    bin_centers = (hd_bins[:-1] + hd_bins[1:]) / 2

    return hd_hist, bin_centers

def get_hd_distr_allplats(pos_data, num_bins = 12):
    hd_distr_allplats = []
    for p in np.arange(1, 62):
        hd = pos_data['hd']
        platforms = pos_data['platforms']
        if np.nanmax(hd) > 2*np.pi + 0.1:
            hd = np.deg2rad(hd)
        indices = np.where(platforms == p)
        hd_p = hd[indices]
        hd_distr, bin_centers = get_hd_distribution(hd_p, num_bins=num_bins)
        hd_distr_allplats.append(hd_distr)
    return hd_distr_allplats, bin_centers

    
def get_norm_hd_distr(spike_train, pos_data, hd_distr_allplats, num_bins = 12):
    hd = pos_data['hd']
    platforms = pos_data['platforms']
    if np.nanmax(hd) > 2*np.pi + 0.1:
        hd = np.deg2rad(hd)
    
    hd = hd[spike_train]
    platforms = platforms[spike_train]

    norm_hd = []
    for p in np.arange(1,62):
        # Get indices where platforms = p
        indices = np.where(platforms == p)
        hd_p = hd[indices]
        hd_p_distr, _ = get_hd_distribution(hd_p, num_bins = num_bins)
        hd_p_norm = hd_p_distr/hd_distr_allplats[p-1]
        norm_hd.append(hd_p_norm)
    return norm_hd

