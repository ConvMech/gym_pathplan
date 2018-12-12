import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import time

#env = gym.make('PathAngleSpeed-v0')
#env = gym.make('PathRandom-v0')
env = gym.make('PathCNN-v0')
#env = gym.make('PathObstacle-v0')
#env = gym.make('PathPartial-v0')
#env = gym.make('PathTarget-v0')


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
		#print(obs.shape)
		#print(obs)
		env.render()
		#time.sleep(1)
		#exit()
		#print(obs)
		#print("out",done)
		# print ("reward:", r)
		# print ("obs:", obs)
