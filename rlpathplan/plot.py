import matplotlib
matplotlib.use('TkAgg')
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_ave(x,h):
	# compute ave with window h
	ave = []
	t = 0
	while t <= x.size - 1:
		ave.append(np.mean(x[t:t+h]))
		t += h
	return np.array(ave)

env = 'PathHallway-v0'

algo = 'ppo'
# algo = 'rl2'
# algo = 'sticky'

policy = 'mlp'
# policy = 'rnn'

hdict = []
# seeddict = [100,101]
seeddict = [100, 101, 102]
colordict = ['r','b','g','m', 'y']
# lrdict = [3e-4]
lrdict = [0.0]
hdict = []
labels = []
numunits = 3
numlayers = 4
cliprange = 0.2
h = 10
ent = 0.0
idx = 0
objindex = 5
for lr,color in zip(lrdict,colordict):
	
	for seed, color in zip(seeddict, colordict):
		r = np.load('CKPT/{}_{}/seed_{}lr_{}policy_{}/epoch_episode_rewards.npy'.format(algo, env, seed, lr, policy))
		t = np.load('CKPT/{}_{}/seed_{}lr_{}policy_{}/total_timesteps.npy'.format(algo, env, seed, lr, policy))
		print ("t and r shape:", t.shape, r.shape)
		plt.plot(compute_ave(t,h),compute_ave(r,h),color)
		print(seed,np.max(r), r[-5:])

plt.grid()
plt.show()

