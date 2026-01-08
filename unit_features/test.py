import numpy as np
from random import *
arr = np.arange(1,101)
print(arr)

mrl_95 = np.percentile(arr, 95)
print(mrl_95)

arr = np.append(arr, np.nan)
print(arr)

mrl_95_2 = np.percentile(arr, 95)
print(mrl_95_2)

arr2 = [randint(1, 100) for _ in range(1000)]
sorted_arr2 = np.sort(arr2)

print(np.percentile(sorted_arr2, 95))
print(sorted_arr2[949])