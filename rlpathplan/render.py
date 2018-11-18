#!/usr/bin/env python
import os
import gym
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from baselines import bench, logger

class CALLBACK(object):

    def __init__(self, arg):
        self.arg = arg
        self.directory = 'CKPT/{}_{}/seed_{}lr_{}policy_{}'.format(arg['algo'], arg['env'], arg['seed'], arg['lr'], arg['policy'])
        print ("directory:", self.directory)
        assert os.path.exists(self.directory)
        # self.pickledir = self.directory + '/model'
        self.pickledir = self.directory 
        self.model_dir = self.directory + '/model'
        assert os.path.exists(self.model_dir)
        self.model_dir += '/model'

    def __call__(self, lcl, glb):
        return False 


def test(env_id, num_episode, seed, lr, policy_index=1, wrapper_index=0, K=11):

    from ppo_utils import ppo2
    from utils import wrapper, rl2wrapper

    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    
    # choose a policy
    if policy_index == 0:
        from utils.policies import MlpDiscretePolicy as a_policy
        policy_name = "mlp"
    elif policy_index == 1:
        from utils.policies import MuJoCoLstmDiscretePolicy as a_policy
        policy_name = "rnn"
    else:
        from baselines.ppo2.policies import LstmPolicy as a_policy
        policy_name = "lstm"
    policy = a_policy


    # config environment for training 
    def make_env():
        env = gym.make(env_id)
        env = wrapper.discretizing_wrapper(env, K)
        # env = rl2wrapper.StickyActionEnv(env, 10) # execute an action 10 steps
        # env = rl2wrapper.RL2Env(env, 2)
        env = bench.Monitor(env, logger.get_dir())
        return env

    def make_sticky():
        env = gym.make(env_id)
        env = wrapper.discretizing_wrapper(env, K)
        env = rl2wrapper.StickyActionEnv(env, 10) # execute an action 10 steps
        env = bench.Monitor(env, logger.get_dir())
        return env

    def make_rl2():
        env = gym.make(env_id)
        env = wrapper.discretizing_wrapper(env, K)
        env = rl2wrapper.StickyActionEnv(env, 10) # execute an action 10 steps
        env = rl2wrapper.RL2Env(env, 2)
        env = bench.Monitor(env, logger.get_dir())
        return env

    if wrapper_index == 0:
        algo = "ppo"
        env = DummyVecEnv([make_env])
    elif wrapper_index == 1:
        algo = "sticky"
        env = DummyVecEnv([make_sticky])
    elif wrapper_index == 2:
        algo = "rl2"
        env = DummyVecEnv([make_rl2])
    env = VecNormalize(env)

    # build call back
    arg = {}
    arg['algo'] = algo
    arg['seed'] = seed
    arg['env'] = env_id
    arg['lr'] = lr
    arg['policy'] = policy_name
    arg['bins'] = K
    callback = CALLBACK(arg)
    
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    set_global_seeds(seed)
    
    ppo2.render_object(policy=policy, env=env, nsteps=512, 
        episodes=num_episode, callback=callback)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='GraspCubeFree-v0')
    # parser.add_argument('--env', help='ID', default='GraspCubeInit-v0')
    # parser.add_argument('--env', help='ID', default='GraspCubeExp-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=100)
    parser.add_argument('--entcoef', type=float, default=0.0)
    parser.add_argument('--num-episode', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--policy', type=int, default=0,
        help="0 for mlp policy; 1 for rnn policy")
    parser.add_argument('--wrapper', type=int, default=0,
        help="0 for ppo; 1 for sticky ppo; 2 for rl2")
    args = parser.parse_args()
    logger.configure()
    test(args.env, num_episode=args.num_episode, seed=args.seed, 
        policy_index=args.policy, wrapper_index=args.wrapper, lr=args.lr)


if __name__ == '__main__':
    main()
