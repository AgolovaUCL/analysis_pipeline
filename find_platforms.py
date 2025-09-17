import numpy as np
from matplotlib.path import Path

radius = 4
hex_side_length = 89# Length of the side of the hexagon
theta = np.radians(305)  # Rotation angle in radians
desired_x, desired_y = 1320, 950

def calculate_cartesian_coords(coord, hex_side_length):
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord

def hex_grid(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

def is_point_in_platform(rat_locx, rat_locy, hcoord, vcoord, hex_radius=hex_side_length):
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            hcoord + hex_radius * np.cos(angle),
            vcoord + hex_radius * np.sin(angle)
        ])
    hexagon_path = Path(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

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
        return closest_platform
    