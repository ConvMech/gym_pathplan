import random
from queue import PriorityQueue
from gym.envs.pathplan import obstacle_gen
import numpy as np 
import scipy.misc

class PathFinding(object):
	"""value in map: 0: nothing 1: wall 2: player 3: goal"""
	def __init__(self, rows=200, cols=1000):
		self.rows = rows
		self.cols = cols
		self.shape = (rows, cols)
		self.map_s = None
		self.player = None
		self.goal = None
		self.terminal = True
		self.difficulty = 10

	def reset(self):
		self.terminal = False
		self.map_s = obstacle_gen.generate_map(self.shape, self.rows//5, self.difficulty) # TODO: 10 is the number of obstacles.
		self.player = self.map_s.start
		self.goal = self.map_s.goal
		return self.get_state()

	def get_map(self):
		"""return a (n, n) grid"""
		state = np.array(self.map_s.dom, copy=True)
		state[self.player[0], self.player[1]] = 2
		state[self.goal[0], self.goal[1]] = 3
		return state

	def get_state(self):
		"""return a (n, n) grid"""
		state = self.get_map()
		state = state.flatten()
		return state

	def step(self, a):
		if self.terminal:
			return self.step_return(1)
		assert 0 <= a and a < 4

		di, dj = obstacle_gen.MOVEMENT[a]
		pi, pj = self.player

		next_i, next_j = pi + di, pj + dj

		if self.map_s.is_legal(next_i, next_j):
			self.player = (next_i, next_j)

		if self.player == self.goal:
			self.terminal = True
			return self.step_return(1)

		return self.step_return(-0.001)

	def step_return(self, reward):
		return self.get_state(), reward, self.terminal, {}

