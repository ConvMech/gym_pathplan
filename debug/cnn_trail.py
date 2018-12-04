import gym

from stable_baselines import A2C,PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# Create and wrap the environment
env = gym.make('PathCNN-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")
# Train the agent
model.learn(total_timesteps=100000,tb_log_name="first_run")

model.save("PathAngle_cnn1")
# Save the agent