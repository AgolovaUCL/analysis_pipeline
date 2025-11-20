import numpy as np
from matplotlib.path import Path


def calculate_cartesian_coords(coord, hex_side_length):
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord

def translate_coords(hcoord2, vcoord2, theta, x_center, y_center):
    hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [-v for v in vcoord_rotated]


    # Calculate the translation needed to align the first rotated coordinate
    dx = x_center - hcoord_rotated[0]
    dy = y_center - vcoord_rotated[0]

    # Apply the translation
    hcoord_translated = [x + dx for x in hcoord_rotated]
    vcoord_translated = [y + dy for y in vcoord_rotated]
    return hcoord_translated, vcoord_translated
    
def get_coordinates(params):
    radius = params['radius']
    hex_side_length = params['hex_side_length']
    theta = params['theta_angle']
    x_center = params['x_center']
    y_center = params['y_center']
    coord = hex_grid(radius)
    
    hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)
    
    hcoord_translated, vcoord_translated = translate_coords(hcoord2, vcoord2, theta, x_center, y_center)
    return hcoord_translated, vcoord_translated
    
def hex_grid(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

def is_point_in_platform(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            hcoord + hex_side_length * np.cos(angle),
            vcoord + hex_side_length * np.sin(angle)
        ])
    hexagon_path = Path(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

# Updated code to find the hexagon the rat is in
def get_platform_number(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):

    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if is_point_in_platform(rat_locx, rat_locy, x, y, hex_side_length):
            return i + 1
    return np.nan

def get_platform_center(platform, params):
    hcoord_translated, vcoord_translated = get_coordinates(params)
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
    