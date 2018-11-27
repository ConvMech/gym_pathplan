import numpy as np

a = np.array([1,2,3,4,5,6,5,6])
b = np.array([0,0,0,1,1,1,0,0])

one_min = min(a[b==1])
zero_min = min(a[b==0])

print(one_min,zero_min)

print(1 not in a)