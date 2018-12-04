import gym

from stable_baselines import A2C,PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('PathAngle-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")

model = PPO2.load("PathAngle")
#model = A2C.load("ppo2_lunar")
# Enjoy trained agent
obs = env.reset()

successCount = 0.0
total = 100
for i in range(total):
	while True:
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    env.render()
	    if dones:
	    	if rewards[0] == 100.0:
	    		successCount += 1
	    	#print(i,rewards[0])
	    	break
print("total success rate {}%".format(100.0*successCount/total))

