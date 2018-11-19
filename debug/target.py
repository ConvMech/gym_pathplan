import numpy as np
import matplotlib.pyplot as plt
from gym.envs.pathplan import obstacle_gen
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

class target(DynamicObject):
	def __init__(self,x,y,theta,v):
		DynamicObject.__init__(self,x,y,theta,v)

def main():
	viewer = MapViewer(1500, 300, 30,150)
	map_s = obstacle_gen.generate_map((30,150),6,10)
	goal = target(map_s.goal[0],map_s.goal[1],np.pi * 45/180.0,v=2)
	for _ in range(10000):
		#print(goal.xpos,goal.ypos)
		map_s = goal.update(map_s)
		viewer.draw(map_s.dom)
	viewer.stop()

if __name__ == "__main__":
	main()