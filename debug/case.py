
import numpy as np

dim = 3
intensities = [1,1,1,1,1,1,3,1,1]
distances = [1,2,5,7,8,5,6,8,7]
ind = np.arange(0,len(intensities), dim)  
#max_ind = [i+np.max(distances[i : i+dim]) for i in ind]
#print(max_ind)
transfer_dist = [np.mean(distances[i : i+dim]) for i in ind]
print(transfer_dist)
transfer_intens = [np.max(intensities[i : i+dim]) for i in ind]
print(transfer_intens)

print('=====')
channel1_ind = [i for i in range(len(transfer_intens)) if transfer_intens[i] == 1]
channel2_ind = [i for i in range(len(transfer_intens)) if transfer_intens[i] == 3]
print(channel1_ind)
print(channel2_ind)
print('=====')
channel1_dist = [transfer_dist[i] if i in channel1_ind else 0 for i in range(len(transfer_dist))]
channel2_dist = [transfer_dist[i] if i in channel2_ind else 0 for i in range(len(transfer_dist))]
channel1_intens = [1 if i in channel1_ind else 0 for i in range(len(transfer_intens))]
channel2_intens = [1 if i in channel2_ind else 0 for i in range(len(transfer_intens))]
print(channel1_dist)
print(channel2_dist)
print('=====')
print(channel1_intens)
print(channel2_intens)

a=np.array([1,2])
b=np.array([3,4])
print(np.vstack([a,b]))


print(np.vstack([np.array(channel1_dist), np.array(channel2_dist)]))