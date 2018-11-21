import numpy as np

class RobotPlayer(object):
	def __init__(self, x, y, theta):
		self.xpos = x
		self.ypos = y
		self.theta = theta
		self.vel = 0
		self.w = 0

	def forward(self):
		self.xpos += int(self.vel * np.cos(self.theta))
		self.ypos += int(self.vel * np.sin(self.theta))
		self.theta += self.w 
		print ("car position:", self.xpos, self.ypos, self.theta)

	def set_action(self, vel, w):
		self.vel = vel
		self.w = w

	def position(self):
		return self.xpos, self.ypos

class angleRobot(RobotPlayer):
	def __init__(self, x, y, theta):
		RobotPlayer.__init__(self, x, y, theta)

	