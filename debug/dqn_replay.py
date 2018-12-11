import gym

from stable_baselines import A2C,PPO2,DQN
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from dqn_policy import DQNCustomPolicy


# Create and wrap the environment
env = gym.make('PathCNN-v0')
env = DummyVecEnv([lambda: env])
#model = PPO2(CustomPolicy, env, verbose=1,tensorboard_log="./ppo2_proj_tensorboard/")
model = DQN.load('PathAngle_DQN_cnn-1', policy=DQNCustomPolicy)

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