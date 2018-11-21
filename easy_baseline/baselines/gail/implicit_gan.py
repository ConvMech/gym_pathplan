"""
Behavior Cloning using GAN (Implicit Policy)
"""
import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.statistics import stats

from baselines.ddpg.ddpg_implicit import DDPG_paramnoise
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum

from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier


def allmean(x, nworkers):
    assert isinstance(x, np.ndarray)
    out = np.empty_like(x)
    MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
    out /= nworkers
    return out


def setup_and_learn(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, actor, critic, classifier,
    normalize_returns, normalize_observations, critic_l2_reg, classifier_l2_reg, actor_lr, critic_lr, classifier_lr,
    action_noise, popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, fifomemory,
    tau=0.01, eval_env=None, callback=None, entropy_coeff=1., reward_giver=None, expert_dataset=None, g_step=4, d_step=1, 
    d_stepsize=3e-4, max_timesteps=0, max_iters=0, timesteps_per_batch=1024, adversary_hidden_size=100, adversary_entcoeff=1e-3, task='train', expert_path=None): # TODO: max_episodes
    """
    set up learning agent and execute training
    """
    logger.info('Initialize policy')
    logger.info('noisynet implementation of DDPG')

    assert task == 'train'

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG_paramnoise(actor, critic, classifier, memory, fifomemory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, critic_l2_reg=critic_l2_reg, classifier_l2_reg=classifier_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, classifier_lr=classifier_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, entropy_coeff=entropy_coeff)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    logger.info('Initialize Discriminator')
    reward_giver = TransitionClassifier(env, adversary_hidden_size, entcoeff=adversary_entcoeff)
    d_adam = MpiAdam(reward_giver.get_trainable_variables())

    logger.info('Load Expert Data')
    dataset = Mujoco_Dset(expert_path=expert_path, traj_limitation=-1)  # TODO: customize


    logger.info('Start training')
    with U.single_threaded_session() as sess:
        # init agent
        agent.initialize(sess)
        # tf saver
        saver = tf.train.Saver()
        # finalize graph
        sess.graph.finalize()

        learn(env, agent, reward_giver, dataset,
              g_step, d_step, d_stepsize=d_stepsize, timesteps_per_batch=timesteps_per_batch,
              nb_train_steps=nb_train_steps, max_timesteps=max_timesteps, max_iters=max_iters,  # TODO: max_episodes
              callback=callback, d_adam=d_adam, sess=sess, saver=saver
              )