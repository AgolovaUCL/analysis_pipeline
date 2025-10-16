
import numpy as np

q = 47**2 + 84**2
print(np.sqrt(q)) # Length of edge in pixels

len_cm = 11 # length of edge in cm

len_pixels = np.sqrt(q)
print(len_pixels/len_cm)