import gym
import numpy as np
from gym import spaces

class RL2Env(gym.Wrapper):

	def __init__(self, env, num_episodes=2):
		"""each episode of this environment contains num_episodes episodes of the base environment"""
		gym.Wrapper.__init__(self, env)
		self.num_episodes = num_episodes
		self.prev_action = np.zeros(env.action_space.high.size)
		self.prev_reward = np.array([0.0])
		self.prev_done = np.array([1.0])
		self.episode_counter = 0
		obs_size = self.env.observation_space.high.size + self.env.action_space.high.size + 2
		self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,))

		# render purpose
		self.episodes_reward_record = [[] for _ in range(num_episodes)]

	def reset(self, test=0):
		"""return concatenation of [obs, prev_action, prev_done]"""
		# update episode counter
		self.episode_counter = 0
		self.episodic_rewards = 0.0
		# update action and done
		self.prev_action = np.zeros(self.env.action_space.high.size)
		self.prev_reward = np.array([0.0])
		self.prev_done = np.array([1.0])
		# reset base env
		obs = self.env.reset(test)
		return np.concatenate([obs, self.prev_action, self.prev_reward, self.prev_done])

	def step(self, action):
		"""return concatenation of [obs, prev_action, prev_done]"""		
		# step the base env
		obs, r, done, _ = self.env.step(action)
		# update action
		self.prev_action = action
		self.prev_reward = np.array([r])
		self.prev_done = np.array([1.0]) if done else np.array([0.0])

		# update episodic rewards
		self.episodic_rewards += r

		# check done
		rl2done = False
		if done:
			# update the counter
			self.episode_counter += 1
			if self.episode_counter == self.num_episodes:
				rl2done = True # rl2 env done
				print('episod {} rewards {}'.format(self.episode_counter, self.episodic_rewards))
			else:
				# rl2 env is not done, reset old env
				obs = self.env.reset()
				print('base env reset')
				print('episod {} rewards {}'.format(self.episode_counter, self.episodic_rewards))
			self.episodes_reward_record[self.episode_counter-1].append(self.episodic_rewards)
			self.episodic_rewards =  0.0                
			np.save("episodes reward", self.episodes_reward_record)

		# concate and return
		return np.concatenate([obs, self.prev_action.flatten(), self.prev_reward, self.prev_done]), r, rl2done, {}
	#def observation_space(self):
	#	totalsize = self.env.observation_space.high.size + self.env.action_space.high.size + 2
	#	return spaces.Box(-np.inf, np.inf, shape=(totalsize,))

class StickyActionEnv(gym.Wrapper):

	def __init__(self, env, skip=5):
		"""execute an input action for multiple steps"""
		gym.Wrapper.__init__(self, env)
		self._skip = skip

	def reset(self, test=0):
		return self.env.reset(test)

	def step(self, action):
		total_reward = 0.0
		done = None
		for i in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			total_reward += reward
			if done:
				break
		output_ob = obs # output the last obs in the sequence. is it a good choice?

		return output_ob, total_reward, done, info

"""
e = gym.make('Pendulum-v0')
e = StickyActionEnv(e, 10)
e.reset()
done = False
while not done:
	obs,_,done,_ = e.step(e.action_space.sample())
	print(obs)
"""


