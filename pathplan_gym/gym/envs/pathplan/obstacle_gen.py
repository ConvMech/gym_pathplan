import numpy as np
import queue
import random
from gym.envs.pathplan.dynamic_object import obstacle
from gym.envs.pathplan.dynamic_object import target

MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1)]
class ObstacleGen(object):
    def __init__(self, dom_size, size_max):
        self.dom_size = dom_size # dom_size should be a list or tuple with two elements
        self.start = None
        self.goal = None
        self.dom = np.zeros(dom_size)
        self.mask = None
        self.size_max = size_max
        self.type = ['circle', 'rect']
        self.ob_origin = []

    def insert_shape(self, type, params,fill=1):
        img = self.dom.copy()
        if type is 'circle':
            r = params[0]
            x = params[1]
            y = params[2]
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if (x - i) ** 2 + (y - j) ** 2 < r ** 2:
                        img[i, j] = fill
        if type is 'rect':
            cx = params[0]
            cy = params[1]
            h = params[2]
            w = params[3]
            for i in range(cx, min(cx + w, img.shape[0])):
                for j in range(cy, min(cy + h, img.shape[1])):
                    img[i, j] = fill
        return img
 
    def add_rand_obs(self, type):
        if type is 'circle':
            rand_rad = int(np.random.random(1) * self.size_max)
            randx = int(np.random.random(1) * self.dom_size[0])
            randy = int(np.random.random(1) * self.dom_size[1])
            im_try = self.insert_shape(type, (rand_rad, randx, randy))
        if type is 'rect':
            rand_h = int(np.random.random(1) * self.size_max)
            rand_w = int(np.random.random(1) * self.size_max)
            rand_x = int(np.random.random(1) * self.dom.shape[0])
            rand_y = int(np.random.random(1) * self.dom.shape[1])
            im_try = self.insert_shape(type, (rand_x, rand_y, rand_h, rand_w))
        self.dom = im_try
        return 1

    def expand_target_size(self,size,type='circle'):
        tx,ty = self.goal
        im_try = self.insert_shape(type, (size, tx,ty),fill=3)
        self.dom = im_try

    def add_N_rand_obs(self, N):
        res = 0
        for i in range(N):
            rand_type = self.type[np.random.randint(2)]
            res += self.add_rand_obs(rand_type)
        return res

    def get_image(self):
        return self.dom

    def add_border(self):
        for i in range(self.dom.shape[0]):
            self.dom[i,0] = 1
            self.dom[i,self.dom.shape[1]-1] = 1
        for j in range(self.dom.shape[1]):
            self.dom[0,j] = 1
            self.dom[self.dom.shape[0]-1, j] = 1   

    def delete_border(self):
        for i in range(self.dom.shape[0]):
            self.dom[i,0] = 0
            self.dom[i,self.dom.shape[1]-1] = 0
        for j in range(self.dom.shape[1]):
            self.dom[0,j] = 0
            self.dom[self.dom.shape[0]-1, j] = 0

    def spawn_start_goal(self, start_pos=0.7, goal_pos=0.8):
        # spawn a start point at the first 70% percent
        x, y = np.where(self.dom[:, :int(start_pos * self.dom_size[1])] == 0)
        free_pos = list(zip(x, y))
        start = random.sample(free_pos, 1)[0]
        # spawn a start point at the last 20% percent
        x, y = np.where(self.dom[:, :int(goal_pos * self.dom_size[1])] == 0)
        free_pos = list(zip(x, y))
        goal = random.sample(free_pos, 1)[0]
        return start, goal

    def path_exists(self, start, goal):
        """test if there exists a path from start to goal using depth first search"""
        stack = [(start, [start])]
        visited = set()
        while stack:
            (vertex, path) = stack.pop()
            visited.add(vertex)

            legal_cells = set(self.legal_directions(*vertex)) - visited

            for next in legal_cells:
                if next == goal:
                    return True
                stack.append((next, path + [next]))
        return False

    def legal_directions(self, pos_x, pos_y):
        possible_moves = [(pos_x + dx, pos_y + dy) for dx, dy in MOVEMENT]
        return [(next_x, next_y) for next_x, next_y in possible_moves if self.is_legal(next_x, next_y)]

    def is_legal(self, x, y):
        if x < 0 or x >= self.dom_size[0]:
            return False
        if y < 0 or y >= self.dom_size[1]:
            return False
        return self.dom[x,y] == 0

    def search_object(self,is_random=True,v=0.2,objectId=1):

        def is_object(pos_x, pos_y):
            if pos_x < 0 or pos_x >= self.dom_size[0] or pos_y < 0 or pos_y >= self.dom_size[1]:
                return False
            return self.dom[pos_x,pos_y] == objectId

        def object_neighbour(pos_x, pos_y):
            res = []
            possible_moves = [(pos_x + dx, pos_y + dy) for dx, dy in MOVEMENT]
            for next_x, next_y in possible_moves:
                if is_object(next_x, next_y) and ((next_x, next_y) not in self.explored):
                    res.append((next_x,next_y))
            return res

        def bfs(start):
            area = []
            saved_start = start

            myq = queue.Queue()
            myq.put(start)
            self.explored.add(start)

            while not myq.empty():
                for _ in range(myq.qsize()):
                    node = myq.get()
                    area.append((node[0]-saved_start[0],node[1]-saved_start[1]))

                    legal_neighbours = object_neighbour(*node)
                    for neighbour in legal_neighbours:
                        self.explored.add(neighbour)
                        myq.put(neighbour)

            return area,saved_start

        self.explored = set()
        objects = []
        i = 0
        for x in range(self.dom_size[0]):
            for y in range(self.dom_size[1]):
                if is_object(x,y) and ((x,y) not in self.explored):
                    i += 1
                    area,start = bfs((x,y))
                    if is_random:
                        angle = np.random.uniform(-np.pi,np.pi)
                    else:
                        angle = 45./180*np.pi

                    if objectId == 1: # create obstacle list
                        ob = obstacle(start[0],start[1],angle,area,v=v)
                    elif objectId == 3: #create targte list
                        ob = target(start[0],start[1],angle,area,v=v)

                    self.ob_origin.append((start[0],start[1],angle,v))
                    objects.append(ob)

        return objects



def generate_map(shape, obs_size, num_obstacles,speed,target_size=0):
    # shape[tuple]: (rows, cols) should be cols >> rows rows > 5, cols > 20
    assert shape[0] > 5
    assert shape[1] > 20
    assert shape[1] > shape[0]
    assert obs_size < shape[0] - 2

    rows = shape[0]
    cols = shape[1]

    # generate a map
    map_s = ObstacleGen(shape, obs_size)
    #print ("get the map object")

    num_obs = map_s.add_N_rand_obs(num_obstacles)
    #print ("added obstaclea")

    #sample start and goal for the map
    while map_s.start is None or map_s.goal is None:
        start, goal = map_s.spawn_start_goal(1,1)
        print ("checking if a valid env")
        if map_s.path_exists(start, goal):
            map_s.start = start 
            map_s.goal = goal

    if target_size:
        map_s.expand_target_size(target_size)
        map_s.add_border()
    targets = map_s.search_object(is_random=True,v=speed,objectId=3)

    map_s.delete_border()
    #print ("delete border")

    objs = map_s.search_object(is_random=True,v=speed)

    map_s.add_border()
    #print ("added border")

    return map_s,objs,targets[0] #assume using only one target