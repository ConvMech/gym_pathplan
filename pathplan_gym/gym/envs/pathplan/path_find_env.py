import pygame
import numpy as np 
import gym
import copy
from gym import error, spaces, utils

from gym.envs.pathplan.path_find import PathFinding
from gym.envs.pathplan.path_find import PathFindingAngle
from gym.envs.pathplan.path_find import PathFindingCNN
from gym.envs.pathplan.path_find import PathFindingAngleSpeed
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
		n_actions = 11
		self.task = PathFindingAngle(rows, cols)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(n_actions)

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
		n_actions = 11
		self.task = PathFindingAngle(rows, cols,lidarAngle=360,tarSize=5,numObstacle=5)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(n_actions)

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
		n_actions = 11
		self.task = PathFindingAngle(rows, cols,lidarAngle=360,tarSize=5,numObstacle=5,tarDynamic=False,obDynamic=True)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(n_actions)

class PathFindingObstaclePartialEnv(PathFindingObstacleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 11
		self.task = PathFindingAngle(rows, cols,lidarAngle=180,tarSize=5,numObstacle=5,tarDynamic=False,obDynamic=False)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(n_actions)

class PathFindingTargetDynamicEnv(PathFindingObstacleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 11
		self.task = PathFindingAngle(rows, cols,lidarAngle=360,tarSize=5,numObstacle=0,tarDynamic=True,obDynamic=False)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_simple_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		# 0: forward, 1: left, 2: right
		self.action_space = spaces.Discrete(n_actions)


class PathFindingCnnEnv(PathFindingAngleEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 11
		self.task = PathFindingCNN(rows, cols,lidarAngle=60,tarSize=5,numObstacle=5,
			tarDynamic=False,obDynamic=False,playerSpeed=0.6,nAction=n_actions,rangeLim=180)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		self.action_space = spaces.Discrete(n_actions)

	def render(self, mode='human',storeHistory=False):
		map_s = self.task.lidar_map
		if mode is 'human':
			self.viewer.draw(map_s,self.task.player)
		elif mode is 'array':
			return map_s
		if storeHistory:
			return [copy.deepcopy(map_s),copy.deepcopy(self.task.player)]

	def reset(self, test=0):
		return self.task.reset(test,simple=False)


class PathFindingCnnRandomEnv(PathFindingCnnEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_actions = 11
		self.task = PathFindingCNN(rows, cols,lidarAngle=60,
        tarSize=2,numObstacle=5,tarDynamic=True,obDynamic=True,
        playerSpeed=0.3, tarSpeed=0.2, obSpeed=0.1,
        randTarStatic=True,randObStatic=True,nAction=n_actions,rangeLim=45)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		self.action_space = spaces.Discrete(n_actions)



class PathFindingAngleSpeedEnv(PathFindingCnnEnv):
	def __init__(self, rows=30, cols=40, screen_size=(400,300)):
		n_angles = 7 # left45,left30,left15,forward,right15,right30,right45
		n_speeds = 7 # 0,0.2v,0.5v,0.8v,v,1.2v,1.5v
		self.task = PathFindingAngleSpeed(rows,cols,n_angles,n_speeds,lidarAngle=60,tarSize=2,numObstacle=0,tarDynamic=False,obDynamic=False,playerSpeed=0.3)
		self.task.reset()
		self.viewer = MapViewer(screen_size[0], screen_size[1], rows, cols) #test if *(screen_size) works
		shape = self.task.get_state().shape
		diag = np.sqrt(screen_size[0] ** 2 + screen_size[1] ** 2)
		self.observation_space = spaces.Box(low=0, high=diag, shape=shape, dtype=np.float32)
		self.action_space = spaces.Discrete(n_angles*n_speeds)
