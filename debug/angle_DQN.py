import gym

from stable_baselines import A2C,PPO2,DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# Create and wrap the environment
env = gym.make('PathObstacle-v0')
env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")
# Train the agent
model.learn(total_timesteps=100000,tb_log_name="first_run")

# Save the agent
model.save("PathAngle_DQN_ob")