import gym

from stable_baselines import A2C,PPO2,DQN
from stable_baselines.a2c.utils import linear
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from dqn_policy import DQNCustomPolicy

register_policy('DQNCustomPolicy', DQNCustomPolicy)


# Create and wrap the environment
env = gym.make('PathCNN-v0')
env = DummyVecEnv([lambda: env])

model = DQN(policy=DQNCustomPolicy, env=env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")
# Train the agent
model.learn(total_timesteps=100000,tb_log_name="2_run")

# Save the agent
model.save("PathAngle_DQN_cnn-2")