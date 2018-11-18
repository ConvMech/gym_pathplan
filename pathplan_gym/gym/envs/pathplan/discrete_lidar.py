import numpy as np
import matplotlib.pyplot as plt

class obeservation():
	def __init__ (self,angle=60,lidarRange=50,accuracy=1,beems=100):
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
		res = []
		for angle in angles:
			objectDistance = 0
			distance = 0
			while distance < self.range:
				distance += self.accuracy
				x = location[0] + distance * np.cos(angle)
				y = location[1] + distance * np.sin(angle)
				if (x >= mymap.shape[0] or x<0 
					or y>=mymap.shape[1] or y<0):
					continue

				if self.findObject(mymap,x,y,value=1):
					objectDistance = distance
					#mymap = self.drawPoint(mymap,x,y,value=2)
					break
				else:
					objectDistance = max(objectDistance,distance)

				beemsLayer = self.drawPoint(beemsLayer,x,y,value=1)

			res.append(objectDistance)

		return res,beemsLayer

	def findObject(self,mymap,x,y,value=1):
		if mymap[int(x)][int(y)] == value:
			return True
		else:
			return False

	def drawPoint(self,mymap,x,y,value=2):
		mymap[int(x)][int(y)] = value
		return mymap


def main():
	shape=(100,100)
	mymap = np.zeros(shape)
	mymap[20:50,40:60] = 1
	ob = obeservation()
	res,beemsLayer = ob.observe(mymap=mymap,location=(0,shape[1]/2),theta=-0.2)
	print(res)
	mymap[beemsLayer==1] = 2
	plt.imshow(mymap)
	plt.show()

if __name__ == "__main__":
	main()