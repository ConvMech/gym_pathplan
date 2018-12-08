import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import time

env = gym.make('PathCNN-v0')
#env = gym.make('PathAngle-v0')
#env = gym.make('PathObstacle-v1')

# test
#for i in range (100):
env.reset()
for _ in range(100):
	env.reset()
	done = False
	while not done:
		a = env.action_space.sample()
		#print("before")
		#print("action",a)
		obs, r, done, _ = env.step(a)
		#print(obs)
		print(obs.shape)
		env.render()
		#print(obs)
		#print("out",done)
		# print ("reward:", r)
		# print ("obs:", obs)
