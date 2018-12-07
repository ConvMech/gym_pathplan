import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import time

<<<<<<< HEAD
env = gym.make('PathCNN-v0')
#env = gym.make('PathObstacle-v0')
=======
#env = gym.make('PathAngle-v0')
#env = gym.make('PathObstacle-v1')
env = gym.make('PathPartial-v0')
>>>>>>> 8c0c511c2e1f464506cf41e01ccf58a5c131e2a2

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
		env.render()
<<<<<<< HEAD
		#time.sleep(10)
		#exit()
=======
		#time.sleep(0.5)
>>>>>>> 8c0c511c2e1f464506cf41e01ccf58a5c131e2a2
		#print(obs)
		#print("out",done)
		# print ("reward:", r)
		# print ("obs:", obs)
