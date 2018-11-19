import numpy as np
from collections import deque
from baselines.ddpg.memory import array_min2d


class FIFOMemory(object):
    def __init__(self, limit):
        self.limit = limit

        self.observations0 = deque(maxlen=limit)
        self.actions = deque(maxlen=limit)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(0, self.limit, size=batch_size)

        obs0_batch = [self.observations0[i] for i in batch_idxs]
        action_batch = [self.actions[i] for i in batch_idxs]

        result = {
            'obs0': array_min2d(obs0_batch),
            'actions': array_min2d(action_batch),
        }
        return result

    def append(self, obs0, action, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)

    @property
    def nb_entries(self):
        return len(self.observations0)

    def reset(self):
    	"""
    	in case we want to clear memory
    	"""
    	limit = self.limit
    	self.observations0 = deque(maxlen=limit)
    	self.actions = deque(maxlen=limit)