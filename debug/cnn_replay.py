import gym
import numpy as np
import tensorflow as tf
from stable_baselines import A2C,PPO2
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from cnn_policy import CustomPolicy,CustomPolicy2,CustomPolicy3
from gym.envs.pathplan.rendering import MapViewer
import time

# Create and wrap the environment
#env = gym.make('PathRandom-v0')
env = gym.make('PathCNN-v0')
env = DummyVecEnv([lambda: env])
#model = PPO2(CustomPolicy, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")

model = PPO2.load('Pathplan_partial', policy=CustomPolicy2)

obs = env.reset()

successCount = 0.0
total = 10
for i in range(total):
    stepHistory = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        history = env.render(storeHistory=True)

        success = False
        #time.sleep(0.1)  
        if dones:
            if rewards[0] == 100.0:
                successCount += 1
                success = True
            print(i,rewards[0])
            break

        stepHistory.append(history)
        
    if len(stepHistory) > 0:
        flagString = 'success' if success else 'fail'
        name = "/Users/tommy/Workspace/rlProject/report/{}_{}.jpg".format(i,flagString)
        stepHistory = np.array(stepHistory)
        view = MapViewer(400,300,30,40,playerSize=15)
        view.trajectoryDrawer(stepHistory[-1][0],stepHistory[:,1],name,skip=4)

print("total success rate {}%".format(100.0*successCount/total))