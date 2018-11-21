#!/usr/bin/env python
import argparse
from baselines import bench, logger
import os
import numpy as np

"""
with annealed clipping rate
"""

class linear_schedule(object):
    
    def __init__(self, maxrate, minrate):
        assert maxrate >= minrate
        self.maxrate = maxrate
        self.minrate = minrate

    def __call__(self, frac):
        return self.maxrate * frac + self.minrate * (1 - frac)


class CALLBACK(object):

    def __init__(self, arg):
        self.arg = arg
        self.directory = 'ppo2_implicit_annealed_{}/seed_{}_lr_{}_lrq_{}cliprangeq_{}'.format(arg['env'], arg['seed'], arg['lr'], arg['lrq'],
                          arg['cliprangeq'])
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.loss_dict = {}

    def __call__(self, lcl, glb):
        self.epoch_episode_rewards.append(lcl['eprewmean'])
        self.epoch_episode_steps.append(lcl['eplenmean'])
        np.save(self.directory + '/epoch_episode_rewards', self.epoch_episode_rewards)   
        np.save(self.directory + '/epoch_episode_steps', self.epoch_episode_steps)  
        lossvals = lcl['lossvals']
        for lossname, lossval in zip(lcl['model'].loss_names, lossvals):
            if lossname in self.loss_dict.keys():
                self.loss_dict[lossname].append(lossval)
            else:
                self.loss_dict[lossname] = [lossval]
            np.save(self.directory + '/{}'.format(lossname), self.loss_dict[lossname])
        if not os.path.exists(self.directory + '/model'):
            os.makedirs(self.directory + '/model')
        # Save tensorflow models.
        saver = lcl['saver']
        sess = lcl['model'].sess
        saver.save(sess, self.directory + '/model/model')
        return False 


def train(env_id, num_timesteps, seed, lr, lr_q, cliprangeq):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2_implicit
    from baselines.ppo2.policies import ImplicitMLPPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = ImplicitMLPPolicy

    # build call back
    arg = {}
    arg['seed'] = seed
    arg['env'] = env_id
    arg['lr'] = lr
    arg['lrq'] = lr_q
    arg['cliprangeq'] = cliprangeq
    callback = CALLBACK(arg)

    cliprangeq = linear_schedule(maxrate=cliprangeq, minrate=0.001)
    
    ppo2_implicit.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=lr,
        lr_q=lr_q,
        cliprangeq=cliprangeq,
        total_timesteps=num_timesteps, callback=callback)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=3e-5)  # lr for policy and value function update
    parser.add_argument('--lr-q', type=float, default=3e-4)  # lr for q (advantage) function update
    parser.add_argument('--cliprangeq', type=float, default=0.2)  # clip adv function
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, lr=args.lr, lr_q=args.lr_q,
          cliprangeq=args.cliprangeq)


if __name__ == '__main__':
    main()

