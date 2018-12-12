import numpy as np
import matplotlib.pyplot as plt
from gym.envs.pathplan.rendering import MapViewer

class DynamicObject(object):
	def __init__(self,x,y,theta,v=1):
		self.xpos = float(x)
		self.ypos = float(y)
		self.theta = float(theta)
		self.vel = float(v)
		self.n_xpos = 0.0
		self.n_ypos = 0.0
		self.n_theta = float(theta)
		
	def try_forward(self):
		self.n_xpos = self.xpos + self.vel * np.cos(self.n_theta)
		self.n_ypos = self.ypos + self.vel * np.sin(self.n_theta)
		#print("adding",self.xpos,self.n_xpos,self.vel * np.cos(self.n_theta))

	def forward(self):
		self.xpos = self.n_xpos
		self.ypos = self.n_ypos
		self.theta = self.n_theta

	def try_bounce(self,bounceType):
		#print("before",bounceType,self.theta)
		if bounceType == 'x':
			self.n_theta = - self.theta
		elif bounceType == 'y':
			if self.theta > 0:
				self.n_theta = np.pi - self.theta
			elif self.theta < 0:
				self.n_theta = - np.pi -self.theta 
		elif bounceType == 'r':
			if self.theta > 0:
				self.n_theta = self.theta - np.pi
			elif self.theta < 0:
				self.n_theta = self.theta + np.pi
		#print("after",bounceType,self.n_theta)

	def bounce(self,map_s):
		assert self.theta >-np.pi and self.theta <= np.pi
		if self.theta == 0:
			self.n_theta = np.pi
			self.try_forward()
			return True
		elif self.theta == np.pi:
			self.n_theta = 0
			self.try_forward()
			return True
		else:
			for poss in ['x','y','r']:
				self.try_bounce(poss)
				self.try_forward()
				if (map_s.is_legal(*self.nposition())):
					return True
			return False

	def update(self,map_s):
		map_s.dom[self.position()] = 0
		self.try_forward()
		if map_s.is_legal(*self.nposition()):
			#print("legal!",self.n_xpos,self.n_ypos)
			self.forward()
		else:
			#print("not legal!",self.nposition())
			res = self.bounce(map_s)
			if res:
				self.forward()
				
		map_s.dom[self.position()] = 3	
		return map_s

	def position(self):
		return int(self.xpos), int(self.ypos)

	def nposition(self):
		return int(self.n_xpos), int(self.n_ypos)

	def set_position(self,map_s,x,y,theta,v=1):
		map_s.dom[self.position()] = 0
		self.xpos = float(x)
		self.ypos = float(y)
		self.theta = float(theta)
		self.vel = float(v)
		self.n_xpos = 0.0
		self.n_ypos = 0.0
		self.n_theta = float(theta)
		map_s.dom[self.position()] = 3
		return map_s

class ObstacleObject(DynamicObject):
	def __init__(self,x,y,theta,area,v=1):
		self.xpos = float(x)
		self.ypos = float(y)
		self.theta = float(theta)
		self.vel = float(v)
		self.n_xpos = 0.0
		self.n_ypos = 0.0
		self.n_theta = float(theta)
		self.area = area

	def real_cord(self):
		return [(self.xpos + dx, self.ypos + dy) for dx, dy in self.area]

	def try_cord(self):
		return [(self.n_xpos + dx, self.n_ypos + dy) for dx, dy in self.area]

	def change_number(self,map_s,num = 4):
		c = self.real_cord()
		for i in c:
			map_s.dom[int(i[0]),int(i[1])] = num
		return map_s

	def obstacle_legal(self,map_s):
		c = self.try_cord()
		for i in c:
			if not map_s.is_legal(int(i[0]),int(i[1])):
				return False
		return True

	def bounce(self,map_s):
		assert self.theta >-np.pi and self.theta <= np.pi
		if self.theta == 0:
			self.n_theta = np.pi
			self.try_forward()
			return self.obstacle_legal(map_s)
		elif self.theta == np.pi:
			self.n_theta = 0
			self.try_forward()
			return self.obstacle_legal(map_s)
		else:
			for poss in ['x','y','r']:
				self.try_bounce(poss)
				self.try_forward()
				if self.obstacle_legal(map_s):
					return True
			return False

	def update(self,map_s):
		map_s = self.change_number(map_s,0)
		self.try_forward()
		if self.obstacle_legal(map_s):
			self.forward()
		else:
			res = self.bounce(map_s)
			if res:
				self.forward()
				
		map_s = self.change_number(map_s,1)
		return map_s

	def return_origin(self,map_s,x,y,theta,v):
		map_s = self.change_number(map_s,0)
		self.xpos = float(x)
		self.ypos = float(y)
		self.theta = float(theta)
		self.vel = float(v)
		map_s = self.change_number(map_s,1)
		return map_s

class TargetObject(ObstacleObject):
	#def __init__(self,x,y,theta,v):
	#	DynamicObject.__init__(self,x,y,theta,v)
	def __init__(self,x,y,theta,area,v):
		ObstacleObject.__init__(self,x,y,theta,area,v)

	def update(self,map_s):
		map_s = self.change_number(map_s,0)
		self.try_forward()
		if self.obstacle_legal(map_s):
			self.forward()
		else:
			res = self.bounce(map_s)
			if res:
				self.forward()
				
		map_s = self.change_number(map_s,3)
		return map_s

	def return_origin(self,map_s,x,y,theta,v):
		map_s = self.change_number(map_s,0)
		self.xpos = float(x)
		self.ypos = float(y)
		self.theta = float(theta)
		self.vel = float(v)
		map_s = self.change_number(map_s,3)
		return map_s
