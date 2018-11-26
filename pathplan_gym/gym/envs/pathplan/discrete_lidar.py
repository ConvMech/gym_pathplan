import numpy as np
import matplotlib.pyplot as plt

class obeservation(object):
	def __init__ (self,angle=360,lidarRange=300,accuracy=1,beems=1080):
		# angle: the angular range of lidar
		# lidarRange: the maximum distance of lidar's capacity
		# accuracy: increment step size of each laser beem
		# beems: how many beems exit in the range of the angle
		self.angle = np.pi / 180.0 * angle
		self.range = lidarRange
		self.accuracy = accuracy
		self.beems = beems

	def render(self,mymap):
		plt.imshow(mymap)
		plt.show()

	def observe(self,mymap,location,theta):
		angle_start = theta - self.angle/2
		angle_end = theta + self.angle/2
		angles = np.linspace(angle_start,angle_end,num=self.beems)
		beemsLayer = np.zeros_like(mymap)
		distance_obs = np.zeros(self.beems)
		intensity_obs = np.zeros(self.beems)
		objects = [1, 3]
		cosangle = np.cos(angles)
		sinangle = np.sin(angles)
		history = np.ones(self.beems)
		distance = 0

		while distance < self.range:
			distance += self.accuracy
			x = np.int32(location[0] + distance * cosangle)
			y = np.int32(location[1] + distance * sinangle)
			# print (x, y)
			x = np.clip(x, 0, mymap.shape[0]-1)
			y = np.clip(y, 0, mymap.shape[1]-1)
			# print ("clipped:", x, y)
			# break

			intensity_obs[history==1] = mymap[x[history==1],y[history==1]]
			history = history * np.logical_or((mymap[x,y]==0), (mymap[x,y]==2))
			distance_obs[history==1] = distance
			
			beemsLayer = self.drawPoints(beemsLayer,x,y,history=history, value=1)
		
		mymap[beemsLayer==1] = 4
		lidar_map = mymap.copy()

		return distance_obs,intensity_obs,beemsLayer, lidar_map

	def findTarget(self,mymap,x,y,value=3):
		if mymap[int(x)][int(y)] == value:
			return True
		else:
			return False

	def findObject(self,mymap,x,y,value=1):
		if mymap[int(x)][int(y)] == value:
			return True
		else:
			return False

	def find_obstacle(self,mymap,x,y,value=1):
		return mymap[x,y] == value

	def drawPoint(self,mymap,x,y,value=2):
		mymap[int(x)][int(y)] = value
		return mymap

	def drawPoints(self,mymap,x,y,history=None, value=4):
		#print (history)
		mymap[x[history==1], y[history==1]] = value
		return mymap


def main():
	shape=(100,100)
	mymap = np.zeros(shape)
	mymap[70:90,40:60] = 1
	mymap[10:30,40:60] = 1
	ob = obeservation(angle=360,beems=500)
	res,beemsLayer = ob.observe(mymap=mymap,location=(50,50),theta=0)
	print(res)
	print(intense)
	mymap[beemsLayer==1] = 2
	plt.imshow(lidarmap)
	plt.show()

if __name__ == "__main__":
	main()