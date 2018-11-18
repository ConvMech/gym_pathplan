import pygame
import numpy as np 
import gym
from gym import error, spaces, utils

from gym.envs.pathplan.path_find import PathFinding
from gym.envs.pathplan.rendering import MapViewer

class PathFindingEnv(gym.Env):
	metadata = {'render.modes': ['human', 'array']}
	#TODO: adjust screen size to the right size
	def __init__(self, rows, cols, screen_size=(1500, 300)):
		self.task = PathFinding(rows, cols)
		self.task.reset()

		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works

		shape = self.task.get_state().shape
		self.observation_space = spaces.Box(low=0, high=3, shape=shape, dtype=np.int8)
		self.action_space=spaces.Discrete(4)

	def reset(self):
		return self.task.reset()

	def step(self, action):
		return self.task.step(action)

	def render(self, mode='human'):
		map_s = self.task.get_state()

		if mode is 'human':
			self.viewer.draw(map_s)
		elif mode is 'array':
			return map_s

	def close(self):
		self.viewer.stop()


class PathFindingHallwayEnv(PathFindingEnv):
	def __init__(self):
		PathFindingEnv.__init__(self, 30, 150)


