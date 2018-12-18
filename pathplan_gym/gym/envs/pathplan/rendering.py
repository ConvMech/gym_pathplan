import pygame
import numpy as np
import platform 
# conda install -c cogsci pygame

BLACK = (0 , 0 , 0)
YELLOW = (255 , 255 , 0)  
GREEN = (0 , 255 , 0)  
WHITE = (255 , 255 , 255)
ORANGE = (255,69,0)
DGREEN = (34,139,34)

class MapViewer(object):
	def __init__(self, screen_width, screen_height, map_rows, map_cols,playerSize=20):
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.map_rows = map_rows
		self.map_cols = map_cols

		self.started = False

		self.system = platform.system()
		self.player_size = playerSize
		self.player_shape = (100,100) #col,row

	def spwan_player(self,drawDir=True):
		shape = self.player_shape
		player = pygame.Surface(shape, pygame.SRCALPHA, 32)
		player = player.convert_alpha()	
		center_c = self.player_shape[0]//2
		center_r = self.player_shape[1]//2
		# 0 degree pointing right
		point_1 = [center_c+self.player_size//2,center_r]
		point_2 = [center_c-self.player_size//2,center_r-self.player_size//2]
		point_3 = [center_c-self.player_size//2,center_r+self.player_size//2]
		if drawDir:
			pygame.draw.line(player, GREEN,[shape[0],shape[1]//2],[center_c,center_r],2)
		pygame.draw.polygon(player, GREEN, [point_1,point_2,point_3])   
		#player.set_colorkey(WHITE)   
		#player.fill(WHITE)
		return player

	def add_dash(self,layer,angle,length,color):
		shape = self.player_shape
		center_c = self.player_shape[0]//2
		center_r = self.player_shape[1]//2
		theta = angle
		destination = [center_c+np.cos(theta)*length,center_r+np.sin(theta)*length]
		pygame.draw.line(layer,color,[center_c,center_r],destination,3)
		return layer

	def start(self):
		pygame.init()
		pygame.font.init()
		pygame.display.set_caption("WalkFollower")

		self.font = pygame.font.SysFont('Arial', size=16)
		print("init screen size",self.screen_width ,self.screen_height)
		self.screen = pygame.display.set_mode((self.screen_width + 5, self.screen_height + 5), 0, 32)
		self.surface = pygame.Surface(self.screen.get_size())
		self.surface = self.surface.convert()
		self.surface.fill((255, 255, 255))

		self.tile_w = (self.screen_height) / self.map_rows
		self.tile_h = (self.screen_width) / self.map_cols

		self.started = True

	def stop(self):
		try:
			pygame.dispaly.quit()
			pygame.quit()
		except:
			pass

	def draw(self, map_s,player=None,obs=None):
		"""map is a numpy array with int value"""
		if not self.started:
			self.start()

		self.surface.fill((0, 0, 0))

		if player:
			map_s = self.delete_origin_player(map_s,player)

		for (i, j), value in np.ndenumerate(map_s):
			x, y = j, i # TODO: actually I do not know if neccessary

			quad = self.screen_quad_position(x, y)
			# print ("value is:", value)
			color = self.get_color(int(value))

			pygame.draw.rect(self.surface, color, quad)

		self.screen.blit(self.surface, (0, 0))

		splayer = self.spwan_player()
		if obs:
			# for debugging use, can draw arbitry line to from the player
			splayer = self.add_dash(splayer,obs[1],obs[0]*50,ORANGE)
			splayer = self.add_dash(splayer,obs[3],obs[2]*50,BLACK)

		if player:
			rplayer,rect = self.rotate_player(splayer,player.theta,player.position())
			self.screen.blit(rplayer,rect)

		pygame.display.flip()
		pygame.event.get()

	def delete_origin_player(self,map_s,player):
		x,y = player.position()
		map_s[x][y] = 4
		return map_s 

	def rotate_player(self,player,angle,position):
		image = player.copy()
		#image.set_colorkey(WHITE)
		old_rect = image.get_rect()  
		old_rect.center = self.screen_position(position)
		old_center = old_rect.center
		angle = angle/np.pi * 180 - 90
		new_image = pygame.transform.rotate(image,angle)
		rect = new_image.get_rect()   
		rect.center = old_center
		return new_image,rect

	def screen_position(self,position):
		# first column, then rows
		return (position[1]*self.tile_h ,position[0]*self.tile_w)

	def screen_quad_position(self, x, y):
		return x * self.tile_w, y * self.tile_h,  self.tile_w + 1, self.tile_h + 1

	def get_color(self, value):
		if self.system == "Darwin":
			COLORS = [pygame.Color(255, 255, 255, 255), 
					  pygame.Color(0,0,0,255),
					  pygame.Color(0,255,0,255),
					  pygame.Color(255,0,0,255),
					  pygame.Color(255,255,0,255)]
		else:
			COLORS = [0xFFFFFF, 0x000000, 0x00FF00, 0xFF0000, 0xFFFF00, 0x333333]
		if value in range(-1, 5):
			return COLORS[value]
		return 0xFFFF00

	def trajectoryDrawer(self,map_s,playerHistory,saveName,lidarMap=False,skip=2):
		if not self.started:
			self.start()

		i = 0
		self.surface.fill((0, 0, 0))
		self.screen.blit(self.surface, (0, 0))

		for (i, j), value in np.ndenumerate(map_s):
			x, y = j, i # TODO: actually I do not know if neccessary
			if not lidarMap:
				if value == 4:
					value = 0
			quad = self.screen_quad_position(x, y)
			# print ("value is:", value)
			color = self.get_color(int(value))

			pygame.draw.rect(self.surface, color, quad)

		self.screen.blit(self.surface, (0, 0))

		oldPos = None

		line = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA, 32)
		for player in playerHistory:

			i += 1
			splayer = self.spwan_player(drawDir=False)
			rplayer,rect = self.rotate_player(splayer,player.theta,player.position())
			if oldPos:
				#print("line")
				pygame.draw.line(line,DGREEN,
					self.screen_position(oldPos),self.screen_position(player.position()),3)

			oldPos = player.position()

			if i % skip == 0:
				self.screen.blit(rplayer,rect)

		self.screen.blit(line,(0, 0))

		pygame.display.flip()
		pygame.event.get()
		pygame.image.save(self.screen,saveName)

		return


