import numpy as np

arr = np.array([1,2,1,2])
indices = np.where(arr == 1)[0]
print(indices)