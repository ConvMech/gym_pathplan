import gym

from stable_baselines import A2C,PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# Create and wrap the environment
env = gym.make('PathAngle-v0')
env = DummyVecEnv([lambda: env])

# Load the trained agent
model = PPO2.load("PathAngle")
#model = A2C.load("ppo2_lunar")
# Enjoy trained agent

obs = env.reset()
count = 0
for _ in range(100):
	env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs)
		obs, r, done, _ = env.step(action)
		env.render()
		#time.sleep(0.1)
		if r == 100:
			count += 1

print('success rate:',count/100.0)
