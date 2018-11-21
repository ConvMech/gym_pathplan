import numpy as np
import pygame

def createSurface(surface,map_s):

	def screen_quad_position(x, y):
		return x , y , 1, 1

	def get_color(value):
		COLORS = [0xFFFFFF, 0x000000, 0x00FF00, 0xFF0000, 0x0000FF, 0x333333]
		if value in range(-1, 5):
			return COLORS[value]
		return 0xFFFF00

	for (i, j), value in np.ndenumerate(map_s):
		x, y = j, i 
		quad = screen_quad_position(x, y)
		color = get_color(int(value))
		pygame.draw.rect(surface, color, quad)

	return surface

pygame.init()
display = pygame.display.set_mode((350, 350))

x = np.arange(0, 300)
y = np.arange(0, 300)
X, Y = np.meshgrid(x, y)
Z = X + Y
Z = 255*Z/Z.max()

myZ = np.zeros((300,300))
myZ[150:200,150:200] = 100

surf = pygame.Surface((300,300), pygame.SRCALPHA)   # per-pixel alpha
surf.fill((255,255,255,128)) 
surf = createSurface(surf,Z)

'''
surf2 = pygame.Surface((300,300), pygame.SRCALPHA)   # per-pixel alpha
surf2.fill((255,255,255,128)) 
surf2 = createSurface(surf2,myZ)
'''

running = True

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	display.blit(surf, (0, 0))
	display.blit(surf2, (0, 0))

	pygame.display.update()

pygame.quit()
