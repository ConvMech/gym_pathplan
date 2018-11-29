import numpy as np
from gym.envs.pathplan.discrete_lidar import obeservation


class RobotPlayer(obeservation):
    def __init__(self, x, y, theta, v=0.1):
        self.xpos = x
        self.ypos = y
        self.theta = theta
        self.vel = v
        self.w = 0
        self.n_xpos = 0
        self.n_ypos = 0
        self.n_theta = 0
        self.v_upper = 3
        self.v_lower = -1
        self.w_upper = np.pi * 30.0 / 180.0
        self.w_lower = -np.pi * 30.0 / 180.0
        obeservation.__init__ (self,angle=360,lidarRange=50,accuracy=1,beems=1080)
        self.n_distances = np.zeros(self.beems)
        self.n_intensities = np.zeros(self.beems)

    def forward(self):
        self.xpos = self.n_xpos
        self.ypos = self.n_ypos
        self.theta = self.n_theta
        # print ("car position:", self.xpos, self.ypos, self.theta)

    def try_forward(self):
        self.n_xpos = self.xpos + self.vel * np.cos(self.theta)
        self.n_ypos = self.ypos + self.vel * np.sin(self.theta)
        #self.n_theta = self.theta + self.w 
        self.n_theta = self.theta
        # print ("car position:", self.xpos, self.ypos, self.theta)

    def try_forward_lidar(self, state, obs):
        self.n_distances, self.n_intensities, _, _ = obs.observe(mymap=state, location=self.nposition(), theta=self.n_theta)


    def set_action(self, vel, w):
        self.vel = vel
        self.w = w

    
    def set_angle(self, action):
        angle = self.theta - np.pi/2 + (action - 1) * (18./180 * np.pi)
        if angle <= -np.pi:
            self.theta = angle + 2*np.pi
        elif angle > np.pi:
            self.theta = angle - 2*np.pi
        else:
            self.theta = angle

    '''
    if action == 1:
        if self.theta > np.pi/2 and self.theta <= np.pi:
            self.theta = self.theta + angle - 2*np.pi
        else:
            self.theta += angle
    elif action == 2:
        if self.theta > -np.pi and self.theta <= -np.pi/2:
            self.theta = self.theta - angle + 2*np.pi
        else:
            self.theta -= angle
    else:
        self.theta = self.theta
    '''
    
    '''
    def set_angle(self, action):
        action_space = np.linspace(-np.pi, np.pi, 11)
        self.theta = action_space[action]
    '''

    def position(self):
        return int(self.xpos), int(self.ypos)

    def nposition(self):
        return int(self.n_xpos), int(self.n_ypos)