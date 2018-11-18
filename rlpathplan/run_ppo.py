#!/usr/bin/env python
import os
import gym
import pickle
import argparse
import numpy as np
import tensorflow as tf
from baselines import bench, logger

try:
    import lqr
except:
    pass

def linearlr(frac):
    return (3e-5 - frac * (3e-5 - 3e-6))

class CALLBACK(object):

    def __init__(self, arg):
        self.arg = arg
        self.directory = 'CKPT/{}_{}/seed_{}lr_{}policy_{}'.format(arg['algo'], arg['env'], arg['seed'], arg['lr'], arg['policy'])
        self.timesteps_origin = 0
        if arg['continue-train'] == 1:
            logger.info('loading model')
            assert os.path.exists(self.directory)
            self.epoch_episode_rewards = list(np.load(self.directory + '/epoch_episode_rewards.npy'))
            self.epoch_episode_steps = list(np.load(self.directory + '/epoch_episode_steps.npy'))
            self.epoch_xpos = list(np.load(self.directory + '/epoch_xpos.npy'))
            self.total_timesteps = list(np.load(self.directory + '/total_timesteps.npy'))
            self.loss_dict = {}  # do not reload loss
            assert os.path.exists(self.directory + '/model')
            self.timesteps_origin = self.total_timesteps[-1]
        elif arg['continue-train'] == 0:
            logger.info('training from scratch')
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            self.epoch_episode_rewards = []
            self.epoch_episode_steps = []
            self.epoch_xpos = []
            self.total_timesteps = []
            self.loss_dict = {}
        else:
            raise NotImplementedError

    def __call__(self, lcl, glb):
        self.epoch_episode_rewards.append(lcl['eprewmean'])
        self.epoch_episode_steps.append(lcl['eplenmean'])
        #self.epoch_xpos.append(np.copy(lcl['epxposlist']))
        self.total_timesteps.append(lcl['total_timesteps_sofar'] + self.timesteps_origin)
        np.save(self.directory + '/epoch_episode_rewards', self.epoch_episode_rewards)   
        np.save(self.directory + '/epoch_episode_steps', self.epoch_episode_steps)  
        np.save(self.directory + '/total_timesteps', self.total_timesteps)
        # record obs_rms
        e = lcl['env']
        with open(self.directory + '/ob_rms.pkl', 'wb') as output:
            pickle.dump(e.ob_rms, output, pickle.HIGHEST_PROTOCOL)
        with open(self.directory + '/ret_rms.pkl', 'wb') as output:
            pickle.dump(e.ret_rms, output, pickle.HIGHEST_PROTOCOL)
        # Save tensorflow models.
        saver = lcl['saver']
        sess = lcl['model'].sess
        saver.save(sess, self.directory + '/model/model')
        # save states for recurrent model
        states = lcl['runner'].states
        np.save(self.directory + '/states', states)
        return False 


def train(env_id, num_timesteps, seed, lr, entcoef, continue_train, 
    nsteps, wrapper_index=0, policy_index=0, K=None):
    assert K is not None
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
        env = DummyVecEnv([make_env for _ in range(4)])
    elif wrapper_index == 1:
        algo = "sticky"
        env = DummyVecEnv([make_sticky for _ in range(4)])
    elif wrapper_index == 2:
        algo = "rl2"
        env = DummyVecEnv([make_rl2 for _ in range(4)])
    env = VecNormalize(env)


    # build call back
    arg = {}
    arg['algo'] = algo
    arg['seed'] = seed
    arg['env'] = env_id
    arg['lr'] = lr
    arg['policy'] = policy_name
    arg['continue-train'] = continue_train
    arg['K'] = K
    callback = CALLBACK(arg)

    # config tensorflow training
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    set_global_seeds(seed)

    # added linear learning rate
    if lr == 0:
        lr = linearlr

    ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=entcoef,total_timesteps=num_timesteps, callback=callback,
        lr=lr, cliprange=0.2)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='GraspCubeFree-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--entcoef', type=float, default=0.0)
    parser.add_argument('--continue-train', type=int, default=0) # 1 for continued training
    parser.add_argument('--nsteps', type=int, default=512) # nenvs * nsteps is a batch
    parser.add_argument('--bins', type=int, default=11)
    parser.add_argument('--policy', type=int, default=0,
        help="0 for mlp policy; 1 for rnn policy")
    parser.add_argument('--wrapper', type=int, default=0,
        help="0 for ppo; 1 for sticky ppo; 2 for rl2")
    args = parser.parse_args()
    logger.configure()
    train(args.env, nsteps=args.nsteps, entcoef=args.entcoef, K=args.bins, 
        lr=args.lr, policy_index=args.policy, wrapper_index=args.wrapper,
        num_timesteps=args.num_timesteps, seed=args.seed, continue_train=args.continue_train)


if __name__ == '__main__':
    main()

