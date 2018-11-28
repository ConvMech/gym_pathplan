import numpy as np

a = np.array([1,2,3,4,5,6,5,6,7])
b = np.array([0,0,0,0,1,0,0,0,0])

splited_a = np.array_split(a, 3)
splited_b = np.array_split(b, 3)

def getMindis(a,b,tp=0):
	return min(a[b==tp])

one_min = min(a[b==1])
zero_min = min(a[b==0])

print(one_min,zero_min)

print(1 not in a)

print(splited_a)
print(splited_b)


print(np.arange(len(b))[b==1][0])