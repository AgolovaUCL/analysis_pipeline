import numpy as np

y_diff = [0]
x_diff = [1400]

directions = np.arctan2(y_diff, x_diff)
print(np.rad2deg(directions))