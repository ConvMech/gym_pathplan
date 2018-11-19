import numpy as np

class RobotPlayer(object):
	def __init__(self, x, y, theta):
		self.xpos = x
		self.ypos = y
		self.theta = theta
		self.vel = 0
		self.w = 0
		self.n_xpos = 0
		self.n_ypos = 0
		self.n_theta = 0
		self.v_upper = 3
		self.v_lower = -1
		self.w_upper = np.pi * 30.0 / 180.0
		self.w_lower = -np.pi * 30.0 / 180.0

	def forward(self):
		self.xpos = self.n_xpos
		self.ypos = self.n_ypos
		self.theta = self.n_theta
		# print ("car position:", self.xpos, self.ypos, self.theta)

	def try_forward(self):
		self.n_xpos = self.xpos + int(self.vel * np.cos(self.theta))
		self.n_ypos = self.ypos + int(self.vel * np.sin(self.theta))
		self.n_theta = self.theta + self.w 
		# print ("car position:", self.xpos, self.ypos, self.theta)


	def set_action(self, vel, w):
		self.vel = vel
		self.w = w

	def position(self):
		return self.xpos, self.ypos

	def nposition(self):
		return self.n_xpos, self.n_ypos