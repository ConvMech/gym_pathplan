"""
Evaluation for trained policy
"""
import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg_implicit import DDPG_paramnoise
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def evaluate(env, nb_episodes=10, agent=None, withreward=False):

    logger.info('evaluating trained policy')

    if True:

        episode_reward = 0.
        episode_step = 0

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_obs = []
        epoch_episode_actions = []
        if withreward == True:
            reward_record = []

        trajs_obs = {}
        trajs_actions = {}
        if withreward == True:
            trajs_reward = {}

        if agent is not None:
            for ep in range(nb_episodes):

                agent.reset()
                obs = env.reset()
                done = False

                while not done:

                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                    assert action.shape == env.action_space.shape
                    max_action = env.action_space.high
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_episode_actions.append(action)
                    epoch_episode_obs.append(obs)
                    if withreward == True:
                        reward_record.append(r)

                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        trajs_obs[ep] = np.array(epoch_episode_obs)
                        trajs_actions[ep] = np.array(epoch_episode_actions)
                        if withreward == True:
                            trajs_reward[ep] = np.array(reward_record)

                        epoch_episode_obs = []
                        epoch_episode_actions = []
                        episode_reward = 0
                        episode_step = 0
                        if withreward == True:
                            reward_record = []
        else:
            raise ValueError("agent cannot be None")

    print(np.array(epoch_episode_rewards))
 
    if withreward == True:
        return np.array(epoch_episode_rewards), np.array(epoch_episode_steps), trajs_obs, trajs_actions, trajs_reward
    else: 
        return np.array(epoch_episode_rewards), np.array(epoch_episode_steps), trajs_obs, trajs_actions


def evaluate_render(env, nb_episodes=10, agent=None):

    logger.info('evaluating trained policy')

    if True:

        episode_reward = 0.
        episode_step = 0

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_obs = []
        epoch_episode_actions = []

        trajs_obs = {}
        trajs_actions = {}
        if agent is not None:
            for ep in range(nb_episodes):

                agent.reset()
                obs = env.reset()
                done = False
                #if ep >= 490:
                if True:
                    env.render()

                while not done:

                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                    assert action.shape == env.action_space.shape
                    max_action = env.action_space.high
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    #if ep >= 490:
                    if True:
                        env.render()     

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_episode_actions.append(action)
                    epoch_episode_obs.append(obs)

                    obs = new_obs

                    if done:
                        logger.info('reward:{}'.format(episode_reward))
                        #if ep % 10 == 0:
                        #    logger.info('evaluating ep {}'.format(ep))
                        #if ep >= 490:
                        #    logger.info('reward:{}'.format(episode_reward))
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        trajs_obs[ep] = np.array(epoch_episode_obs)
                        trajs_actions[ep] = np.array(epoch_episode_actions)

                        epoch_episode_obs = []
                        epoch_episode_actions = []
                        episode_reward = 0
                        episode_step = 0

        else:
            raise ValueError("agent cannot be None")


def evaluate_ant_running(env, nb_episodes=10, agent=None, withreward=False):
    """
    evaluate ant running environment, qpos is passed as dict info at each time step
    """

    logger.info('evaluating trained policy')

    if True:

        episode_reward = 0.
        episode_step = 0

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_obs = []
        epoch_episode_actions = []
        fixed_pos_action = []
        if withreward == True:
            reward_record = []

        trajs_obs = {}
        trajs_actions = {}
        if withreward == True:
            trajs_reward = {}

        if agent is not None:
            for ep in range(nb_episodes):

                agent.reset()
                obs = env.reset()
                #print(obs)
                done = False

                while not done:

                    # colloect actions on the state obs
                    obs_test = np.zeros_like(obs)
                    obs_test[2] = 0.75
                    obs_test[3] = 1.0
                    action, q = agent.pi(obs_test, apply_noise=False, compute_Q=True)
                    fixed_pos_action.append(action)

                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                    assert action.shape == env.action_space.shape
                    max_action = env.action_space.high
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_episode_actions.append(action)
                    epoch_episode_obs.append(info['pos'])
                    if withreward == True:
                        reward_record.append(r)

                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        trajs_obs[ep] = np.array(epoch_episode_obs)
                        trajs_actions[ep] = np.array(epoch_episode_actions)
                        if withreward == True:
                            trajs_reward[ep] = np.array(reward_record)

                        epoch_episode_obs = []
                        epoch_episode_actions = []
                        episode_reward = 0
                        episode_step = 0
                        if withreward == True:
                            reward_record = []
        else:
            raise ValueError("agent cannot be None")
 
    if withreward == True:
        return np.array(epoch_episode_rewards), np.array(epoch_episode_steps), trajs_obs, trajs_actions, trajs_reward, fixed_pos_action
    else: 
        return np.array(epoch_episode_rewards), np.array(epoch_episode_steps), trajs_obs, trajs_actions, fixed_pos_action


def collect_trajectories(env, numtrajs=10, agent=None):
    """
    collect trajectories for imitation learning
    """

    logger.info('collecting trajs of trained policy')

    if True:

        obs_record = []
        acs_record = []
        rews_record = []
        ep_rets_record = []

        if agent is not None:
            for ep in range(numtrajs):

                agent.reset()
                obs = env.reset()
                #print(obs)
                done = False

                obs_e = []
                acs_e = []
                rews_e = []
                ret_e = 0.0

                while not done:

                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                    assert action.shape == env.action_space.shape
                    max_action = env.action_space.high
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                    obs_e.append(obs)
                    acs_e.append(action)
                    rews_e.append(r)
                    ret_e += r

                    obs = new_obs

                    if done:
                        obs_record.append(np.array(obs_e))
                        acs_record.append(np.array(acs_e))
                        rews_record.append(np.array(rews_e))
                        ep_rets_record.append(ret_e)
        else:
            raise ValueError("agent cannot be None")
    return obs_record, acs_record, rews_record, ep_rets_record