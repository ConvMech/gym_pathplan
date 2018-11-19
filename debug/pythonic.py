import numpy as np

hist = np.ones(3)
res = np.zeros(3)

a = np.zeros((10,10))

a[2][4] = 1

x = np.array([1,2,13])
y = np.array([3,4,5])

np.clip(x,0,9,out=x)
np.clip(y,0,9,out=y)

print(a[x,y])

print(hist)

hist = hist * (a[x,y] == 0)

print(hist)

distance = np.array([100,200,300])
res[hist == 1] = distance[hist == 1]

print(res)