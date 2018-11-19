import pygame
import numpy as np 
class MapViewer(object):
 	def __init__(self, screen_width, screen_height, map_rows, map_cols):
 		self.screen_width = screen_width
 		self.screen_height = screen_height
 		self.map_rows = map_rows
 		self.map_cols = map_cols

 		self.started = False

 	def start(self):
 		pygame.init()
 		pygame.font.init()
 		pygame.display.set_caption("WalkFollower")

 		self.font = pygame.font.SysFont('Arial', size=16)
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

 	def draw(self, map_s):
 		"""map is a numpy array with int value"""
 		if not self.started:
 			self.start()

 		self.surface.fill((0, 0, 0))

 		for (i, j), value in np.ndenumerate(map_s):
 			x, y = j, i # TODO: actually I do not know if neccessary

 			quad = self.screen_quad_position(x, y)
 			# print ("value is:", value)
 			color = self.get_color(int(value))

 			pygame.draw.rect(self.surface, color, quad)

 		self.screen.blit(self.surface, (0, 0))
 		pygame.display.flip()

 	def screen_quad_position(self, x, y):
 		return x * self.tile_w, y * self.tile_h,  self.tile_w + 1, self.tile_h + 1

 	def get_color(self, value):
 		COLORS = [0xFFFFFF, 0x000000, 0x00FF00, 0xFF0000, 0xFFFF00, 0x333333]
 		if value in range(-1, 5):
 			return COLORS[value]
 		return 0xFFFF00

