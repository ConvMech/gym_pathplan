import gym
import tensorflow as tf
from stable_baselines import A2C,PPO2
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from cnn_policy import CustomPolicy,CustomPolicy2

register_policy('CustomPolicy', CustomPolicy)
register_policy('CustomPolicy2', CustomPolicy2)


# Create and wrap the environment
env = gym.make('PathRandom-v0')
env = DummyVecEnv([lambda: env])

#model = PPO2(CustomPolicy2, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")


<<<<<<< HEAD
model.learn(total_timesteps=1000000,tb_log_name="partial3")
# Save the agent
model.save("Pathplan_partial3")
=======
model = PPO2(policy=CustomPolicy,env=env,verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")

model.learn(total_timesteps=500000,tb_log_name="CNN_env")
# Save the agent
model.save("Pathplan_CNN-1")
>>>>>>> 293dadf93c687995f32914f1d435a3c893243a52
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
