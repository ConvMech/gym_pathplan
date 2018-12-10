import gym
import tensorflow as tf
from stable_baselines import A2C,PPO2
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from cnn_policy import CustomPolicy

# Create and wrap the environment
env = gym.make('PathCNN-v0')
env = DummyVecEnv([lambda: env])
#model = PPO2(CustomPolicy, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")
model = PPO2.load('Pathplan_dynamic_ob', policy=CustomPolicy)

obs = env.reset()

successCount = 0.0
total = 100
for i in range(total):
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        #time.sleep(0.1)
        if dones:
            if rewards[0] == 100.0:
                successCount += 1
            print(i,rewards[0])
            break

print("total success rate {}%".format(100.0*successCount/total))