import gym
import tensorflow as tf
from stable_baselines import A2C,PPO2
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from cnn_policy import CustomPolicy,CustomPolicy2

register_policy('CustomPolicy', CustomPolicy)


# Create and wrap the environment
env = gym.make('PathRandom-v0')
env = DummyVecEnv([lambda: env])

#model = PPO2(CustomPolicy2, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")

model = PPO2.load('Pathplan_partial', policy=CustomPolicy2,env=env,verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")

model.learn(total_timesteps=1000000,tb_log_name="partial3")
# Save the agent
model.save("Pathplan_partial3")
#model.save("ppo2_lunar")
'''
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO2.load("PathAngle_NSreward")
#model = A2C.load("ppo2_lunar")
# Enjoy trained agent
obs = env.reset()
for i in range(10000):
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	env.render()
'''