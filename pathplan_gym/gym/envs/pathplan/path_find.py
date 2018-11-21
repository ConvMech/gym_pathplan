import random
import numpy as np 
import scipy.misc

from gym.envs.pathplan import obstacle_gen
from gym.envs.pathplan import discrete_lidar 
from gym.envs.pathplan import robot
from gym.envs.pathplan import dynamic_object as do


class PathFinding(object):
	"""value in map: 0: nothing 1: wall 2: player 3: goal"""
	def __init__(self, rows=200, cols=1000):
		self.rows = rows
		self.cols = cols
		self.shape = (rows, cols)
		self.map_s = None
		self.player = None
		self.goal = None
		self.obstacle = []
		self.terminal = True
		self.lidar_map = None
		self.difficulty = 10
		self.obs = discrete_lidar.obeservation(angle=60, lidarRange=30, beems=1080)

	def reset(self):
		self.terminal = False
		self.map_s,self.obstacle = obstacle_gen.generate_map(self.shape, self.rows//5, self.difficulty) # TODO: 10 is the number of obstacles.
		self.ob_num = len(self.obstacle)
		# self.player = self.map_s.start
		self.player = robot.RobotPlayer(self.map_s.start[0], self.map_s.start[1], 0)
		#self.goal = self.map_s.goal
		self.goal = do.target(self.map_s.goal[0],self.map_s.goal[1],np.pi * 45/180.0,v=0.5)
		_, _, _, self.lidar_map = self.obs.observe(mymap=self.get_map(), location=self.player.position(), theta=self.player.theta)
		self.lidar_map[self.player.position()] = 2
		return self.get_state()

	def get_map(self):
		"""return a (n, n) grid"""
		state = np.array(self.map_s.dom, copy=True)
		state[self.player.position()] = 2
		state[self.goal.position()] = 3
		return state

	def get_state_map(self):
		"""return a (n, n) grid"""
		state = self.get_map()
		state = state.flatten()
		return state

	def get_state(self):
		"""return a (n, n) grid"""
		state = self.get_map()
		distances, intensities, _, self.lidar_map = self.obs.observe(mymap=state, location=self.player.position(), theta=self.player.theta)
		self.lidar_map[self.player.position()] = 2
		observations = np.array([distances, intensities])
		return observations.flatten()

	def step(self, a):
		if self.terminal:
			return self.step_return(1)

		self.map_s = self.goal.update(self.map_s)

		for i in range(self.ob_num):
			self.map_s = self.obstacle[i].update(self.map_s)

		self._set_action(a)
		self.player.try_forward()

		next_i, next_j = self.player.nposition()

		if self.map_s.is_legal(next_i, next_j):
			self.player.forward()
		else:
			self.terminal = True
			return self.step_return(-1)

		if self.player == self.goal.position():
			self.terminal = True
			return self.step_return(1)

		return self.step_return(-0.001)

	def set_range(self,upper,lower,value):
		a_range = (upper - lower) / 2
		center = (upper + lower) / 2
		return center + value * a_range

	def _set_action(self, action):
		# TODO: set range
		v = self.set_range(self.player.v_upper,self.player.v_lower,action[0])
		w = self.set_range(self.player.w_upper,self.player.w_lower,action[1])

		self.player.set_action(v,w)

	def step_return(self, reward):
		return self.get_state(), reward, self.terminal, {}

