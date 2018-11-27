import random
import numpy as np 
import scipy.misc

from gym.envs.pathplan import obstacle_gen
from gym.envs.pathplan import discrete_lidar 
from gym.envs.pathplan import robot
from gym.envs.pathplan import dynamic_object as do


class PathFinding(object):
	"""value in map: 0: nothing 1: wall/obstacle 2: player 3: goal"""
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
		self.obs = discrete_lidar.obeservation(angle=30, lidarRange=30, beems=1080)
		self.steps = 0
		self.target_speed = 0.2
		self.ob_speed = 0.1
		self.speed_low = 0.1
		self.speed_high = 0.5
		self.player_speed = 0.5
		self.difficulty = 5
		self.target_dynamic = True
		self.obstacle_dynamic = True


	def change_obdir(self,ob):
		ob.theta = np.random.uniform(-np.pi,np.pi)
		return ob

	def random_dir(self,frequency = 10):
		if self.steps % frequency == 0:
			for i in range(self.ob_num):
				self.obstacle[i] = self.change_obdir(self.obstacle[i])

	def random_speed(self,low,high,randomStatic = True):
		ob_is_dynamic = np.random.randint(2, size = self.ob_num)
		for i in range(self.ob_num):
			self.obstacle[i].vel = np.random.uniform(low,high)
			if randomStatic:
				if not ob_is_dynamic[i]:
					self.obstacle[i].vel = 0
				

	def reset(self, test=0):
		self.terminal = False

		if test == 0:
			self.map_s,self.obstacle = obstacle_gen.generate_map(self.shape, self.rows//5, self.difficulty,self.ob_speed) 
			self.goal_theta = np.random.uniform(-np.pi,np.pi)
		else:
			# reset enviornment to stored origin 
			self.map_s = self.goal.set_position(self.map_s,self.map_s.goal[0],self.map_s.goal[1],self.goal_theta,v=self.target_speed)
			for i in range(self.ob_num):
				x,y,theta,v = self.map_s.ob_origin[i]
				self.map_s = self.obstacle[i].return_origin(self.map_s,x,y,theta,v)

		self.ob_num = len(self.obstacle)
		self.player = robot.RobotPlayer(self.map_s.start[0], self.map_s.start[1], 0, v=self.player_speed)

		if self.target_dynamic:
			self.goal = do.target(self.map_s.goal[0],self.map_s.goal[1],self.goal_theta,v=self.target_speed)
		else:
			self.goal = do.target(self.map_s.goal[0],self.map_s.goal[1],0,v=0)
		
		self.distances, self.intensities, _, self.lidar_map = self.obs.observe(mymap=self.get_map(), location=self.player.position(), theta=self.player.theta)
		self.lidar_map[self.player.position()] = 2
		
		if self.obstacle_dynamic:
			# reset obstacles with random speed
			if test == 0:
				self.random_speed(self.speed_low,self.speed_high,randomStatic=True)
		else:
			self.random_speed(0,0,randomStatic=False)
		
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
		self.distances, self.intensities, _, self.lidar_map = self.obs.observe(mymap=state, location=self.player.position(), theta=self.player.theta)
		#print('distances:', 'min:',min(distances), 'loc:',np.argmin(distances), distances)
		#print('intensities:','loc:',np.argmin(distances),'type:',intensities[np.argmin(distances)],intensities)
		self.lidar_map[self.player.position()] = 2
		observations = np.array([self.distances, self.intensities])
		return observations.flatten()

	def step(self, a):
		self.steps += 1
		if self.terminal:
			return self.step_return(1)

		self.random_dir(10) # random change obstacle direction

		if self.target_dynamic:
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
		#return self.get_state_map(), reward, self.terminal, {}



class PathFindingAngle(object):
	def __init__(self, rows=200, cols=1000):
		"""value in map: 0: nothing 1: wall/obstacle 2: player 3: goal"""
		self.rows = rows
		self.cols = cols
		self.shape = (rows, cols)
		self.map_s = None
		self.player = None
		self.goal = None
		self.obstacle = []
		self.terminal = True
		self.lidar_map = None
		self.obs = discrete_lidar.obeservation(angle=360, lidarRange=300, beems=1080)
		self.steps = 0
		self.target_speed = 0.2
		self.ob_speed = 0.1
		self.speed_low = 0.1
		self.speed_high = 0.5
		self.player_speed = 0.5
		self.difficulty = 2
		self.target_dynamic = False
		self.obstacle_dynamic = False


	def change_obdir(self,ob):
		ob.theta = np.random.uniform(-np.pi,np.pi)
		return ob

	def random_dir(self,frequency = 10):
		if self.steps % frequency == 0:
			for i in range(self.ob_num):
				self.obstacle[i] = self.change_obdir(self.obstacle[i])

	def random_speed(self,low,high,randomStatic = True):
		ob_is_dynamic = np.random.randint(2, size = self.ob_num)
		for i in range(self.ob_num):
			self.obstacle[i].vel = np.random.uniform(low,high)
			if randomStatic:
				if not ob_is_dynamic[i]:
					self.obstacle[i].vel = 0
				

	def reset(self, test=0):
		self.terminal = False

		if test == 0:
			self.map_s,self.obstacle = obstacle_gen.generate_map(self.shape, self.rows//5, self.difficulty,self.ob_speed) 
			self.goal_theta = np.random.uniform(-np.pi,np.pi)
		else:
			# reset enviornment to stored origin 
			self.map_s = self.goal.set_position(self.map_s,self.map_s.goal[0],self.map_s.goal[1],self.goal_theta,v=self.target_speed)
			for i in range(self.ob_num):
				x,y,theta,v = self.map_s.ob_origin[i]
				self.map_s = self.obstacle[i].return_origin(self.map_s,x,y,theta,v)

		self.ob_num = len(self.obstacle)
		self.player = robot.RobotPlayer(self.map_s.start[0], self.map_s.start[1], 0, v=self.player_speed)
		
		if self.target_dynamic:
			self.goal = do.target(self.map_s.goal[0],self.map_s.goal[1],self.goal_theta,v=self.target_speed)
		else:
			self.goal = do.target(self.map_s.goal[0],self.map_s.goal[1],0,v=0)
		
		self.distances, self.intensities, _, self.lidar_map = self.obs.observe(mymap=self.get_map(), location=self.player.position(), theta=self.player.theta)
		self.lidar_map[self.player.position()] = 2
		
		if self.obstacle_dynamic:
			# reset obstacles with random speed
			if test == 0:
				self.random_speed(self.speed_low,self.speed_high,randomStatic=True)
		else:
			self.random_speed(0,0,randomStatic=False)

		this_i, this_j = self.player.position()
		self.this_state = self.StateType(this_i, this_j, self.distances, self.intensities)
		self.this_dist = self.distances
		self.this_intens = self.intensities
		
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
		self.distances, self.intensities, _, self.lidar_map = self.obs.observe(mymap=state, location=self.player.position(), theta=self.player.theta)
		#print('distances:', 'min:',min(distances), 'loc:',np.argmin(distances), distances)
		#print('intensities:','loc:',np.argmin(distances),'type:',intensities[np.argmin(distances)],intensities)
		self.lidar_map[self.player.position()] = 2
		observations = np.array([self.distances, self.intensities])
		return observations.flatten()



	def StateType(self, x, y, dist, intens, d_min = 5):
		'''
		Winning State: terminal state, if player hits target
		Failure State: terminal state, if player hits obstacle
		Safe State: d_obs > d_min
		Non-safe State: d_obs < d_min
		'''
		if self.map_s.dom[x, y] == 3:
			return 'WS'
		if not self.map_s.is_legal(x, y):
			return 'FS'
		if dist[np.argmin(dist)] > d_min and intens[np.argmin(dist)] == 1:
			return 'SS'
		else:
			return 'NS'


	def step(self, a):
		'''
		Safe State => Winning State, reward = 2
		Safe State => Safe State, reward = 1
		Safe State => Non-safe State, reward = -1
		Non-safe State => Failure State, reward = -2
		Non-safe State => Safe State, reward = 1
		Non-safe State => Non-safe State(distance to obstacle increase), reward = 0
		Non-safe State => Non-safe State(distance to obstacle decrease), reward = -1
		'''
		self.steps += 1

		# In this step
		# 1) Update target, obstacle, map_s, player action
		self.random_dir()
		
		if self.target_dynamic:
			self.map_s = self.goal.update(self.map_s)
		
		for i in range(self.ob_num):
			self.map_s = self.obstacle[i].update(self.map_s)

		self.player.set_angle(a)
		
		# 2) Try forward player movement, lidar map observation
		self.player.try_forward()
		self.player.try_forward_lidar(self.get_map(), self.obs)

		
		next_i, next_j = self.player.nposition()
		next_dist, next_intens = self.player.n_distances, self.player.n_intensities
		next_state = self.StateType(next_i, next_j, next_dist, next_intens)


		'''
		if self.this_state == 'SS' and next_state == 'WS':
			self.terminal = True
			reward = 2
		if self.this_state == 'NS' and next_state == 'FS':
			self.terminal = True
			reward = -2
		if self.this_state == 'NS' and next_state == 'SS':
			self.player.forward()
			reward = 1
		if self.this_state == 'SS' and next_state == 'SS':
			self.player.forward()
			reward = 1
		if self.this_state == 'SS' and next_state == 'NS':
			self.player.forward()
			reward = -1
		if self.this_state == 'NS' and next_state == 'NS':
			if min(next_dist) > min(self.this_dist):
				reward = 0
			else:
				reward = -1
		'''

		

		#print(self.this_state,"->",next_state, self.terminal)

		self.this_state = next_state
		self.this_dist = next_dist
		self.this_intens = next_intens

		return self.step_return(reward)


	def step_return(self, reward):
		#print(self.get_state(), reward, self.terminal, {})
		return self.get_state(), reward, self.terminal, {}
		#return self.get_state_map(), reward, self.terminal, {}