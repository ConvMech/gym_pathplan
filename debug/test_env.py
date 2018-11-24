import gym
import time
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('PathAngle-v0')

# test
#for i in range (100):
env.reset()
for _ in range(10):
	env.reset()
	done = False
	while not done:
		env.render()
		a = np.random.choice(env.action_space,1)[0]
		obs, r, done, _ = env.step(a)
		# print ("reward:", r)
		# print ("obs:", obs)
