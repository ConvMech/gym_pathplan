import pygame
import numpy as np 
import gym
from gym import error, spaces, utils

from gym.envs.pathplan.path_find import PathFinding
from gym.envs.pathplan.path_find import PathFindingAngle
from gym.envs.pathplan.path_find import PathFindingCNN
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

class PathFindingHallwayEnv(PathFindingEnv):
	def __init__(self):
		PathFindingEnv.__init__(self, 30, 150)


class PathFindingAngleEnv(PathFindingEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 3
		self.task = PathFindingAngle(rows, cols)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(11)

	def render(self, mode='human'):
		map_s = self.task.lidar_map
		if mode is 'human':
			self.viewer.draw(map_s,self.task.player,self.task.simple_state)
		elif mode is 'array':
			return map_s

	def reset(self, test=0):
		return self.task.reset(test,simple=True)

class PathFindingObstacleEnv(PathFindingEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 3
		self.task = PathFindingAngle(rows, cols,difficulty=5)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(11)

	def render(self, mode='human'):
		map_s = self.task.lidar_map
		if mode is 'human':
			self.viewer.draw(map_s,self.task.player,self.task.simple_state)
		elif mode is 'array':
			return map_s

	def reset(self, test=0):
		return self.task.reset(test,simple=True)

class PathFindingObstacleDynamicEnv(PathFindingObstacleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 3
		self.task = PathFindingAngle(rows, cols,difficulty=5,obdynamic=True,goalSize=5)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(11)

class PathFindingObstaclePartialEnv(PathFindingObstacleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 3
		self.task = PathFindingAngle(rows, cols,difficulty=5,obdynamic=False,goalSize=5,lidarAngle=180,tarDynamic=False)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(11)

class PathFindingTargetDynamicEnv(PathFindingObstacleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 3
		self.task = PathFindingAngle(rows, cols,difficulty=0,obdynamic=False,goalSize=5,lidarAngle=360,tarDynamic=True)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(11)


class PathFindingCnnEnv(PathFindingAngleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 3
		self.task = PathFindingCNN(rows, cols,difficulty=0,obdynamic=False,goalSize=2,lidarAngle=360,tarDynamic=False,object_speed=0.4)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		self.action_space = spaces.Discrete(11)

	def render(self, mode='human'):
		map_s = self.task.lidar_map
		if mode is 'human':
			self.viewer.draw(map_s,self.task.player)
		elif mode is 'array':
			return map_s

	def reset(self, test=0):
		return self.task.reset(test,simple=False)



