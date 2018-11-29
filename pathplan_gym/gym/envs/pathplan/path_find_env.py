import pygame
import numpy as np 
import gym
from gym import error, spaces, utils

from gym.envs.pathplan.path_find import PathFinding
from gym.envs.pathplan.path_find import PathFindingAngle
from gym.envs.pathplan.rendering import MapViewer

class PathFindingEnv(gym.Env):
	metadata = {'render.modes': ['human', 'array']}
	#TODO: adjust screen size to the right size
	def __init__(self, rows, cols, screen_size=(1500, 300)):
		n_actions = 2
		self.task = PathFinding(rows, cols)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_state().shape
		self.observation_space = spaces.Box(low=0, high=3, shape=shape, dtype=np.int8)
		# self.action_space=spaces.Discrete(4)
		self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')

	def reset(self, test=0):
		return self.task.reset(test)

	def step(self, action):
		return self.task.step(action)

	def render(self, mode='human'):
		# map_s = self.task.get_map()
		map_s = self.task.lidar_map
		#print(set(map_s.flatten()))
		if mode is 'human':
			self.viewer.draw(map_s)
		elif mode is 'array':
			return map_s

	def close(self):
		self.viewer.stop()


class PathFindingEnvA(gym.Env):
	def __init__(self, rows, cols, screen_size=(1500, 300)):
		n_actions = 3
		self.task = PathFindingAngle(rows, cols)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.int8)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(3)
	
	def reset(self, test=0):
		return self.task.reset(test)

	def step(self, action):
		return self.task.step(action)

	def render(self, mode='human'):
		# map_s = self.task.get_map()
		map_s = self.task.lidar_map
		#print(set(map_s.flatten()))
		if mode is 'human':
			self.viewer.draw(map_s)
		elif mode is 'array':
			return map_s

	def close(self):
		self.viewer.stop()


class PathFindingHallwayEnv(PathFindingEnv):
	def __init__(self):
		PathFindingEnv.__init__(self, 30, 150)


class PathFindingAngleEnv(PathFindingEnvA):
	def __init__(self):
<<<<<<< HEAD
		PathFindingEnvA.__init__(self, 30, 40, screen_size=(400,300))
=======
		PathFindingEnvA.__init__(self, 30, 40,screen_size=(400, 300))
>>>>>>> 7812165ccd7fdd5de09d01d63abfc51c5d5a1d2c


