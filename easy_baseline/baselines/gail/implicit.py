"""
Implicit Policy Optimization
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


def train_one_batch(env, agent, reward_giver, timesteps_per_batch, nb_train_steps, g_step=3):
    """
    generate one batch of trajectories and update impplicit policy parameters
    """
    # reset agent and clear memory buffer
    agent.reset()
    agent.memory.reset()
    agent.fifomemory.reset()

    max_action = env.action_space.high
    obs_record = []
    action_record = []

    obs = env.reset()
    done = False

    epoch_actor_losses_record = []
    epoch_critic_losses_record = []
    epoch_classifier_losses_record = []
    epoch_approx_entropy_record = []

    logger.info("Collect trajectories on env")
    logger.info("num of policy gradients {}".format(g_step))
    for _ in range(g_step):

        t = 0

        while t < timesteps_per_batch:
            # Predict next action.
            action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
            assert action.shape == env.action_space.shape

            # Execute next action.
            assert max_action.shape == action.shape
            r = reward_giver.get_reward(obs, max_action * action)
            new_obs, _, done, info = env.step(max_action * action)
            t += 1
        
            obs_record.append(obs)
            action_record.append(max_action * action)
            agent.store_transition(obs, action, r, new_obs, done)
            obs = new_obs

            if done:
                # Episode done.
                agent.reset()
                obs = env.reset()

        logger.info("Training Implicit Policy")
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_classifier_losses = []
        epoch_approx_entropy = []
        for t_train in range(nb_train_steps):

            cl, al, cll, ae = agent.train()
            epoch_actor_losses.append(al)
            epoch_critic_losses.append(cl)
            epoch_classifier_losses.append(cll)
            epoch_approx_entropy.append(ae)
            agent.update_target_net()

        logger.info('actor loss {}'.format(mpi_mean(epoch_actor_losses))) 
        logger.info('critic loss {}'.format(mpi_mean(epoch_critic_losses)))
        logger.info('classifier loss {}'.format(mpi_mean(epoch_classifier_losses)))
        logger.info('approx entropy {}'.format(mpi_mean(epoch_approx_entropy)))

        epoch_actor_losses_record += [mpi_mean(epoch_actor_losses)]
        epoch_critic_losses_record += [mpi_mean(epoch_critic_losses)]
        epoch_classifier_losses_record += [mpi_mean(epoch_classifier_losses)]
        epoch_approx_entropy_record += [mpi_mean(epoch_approx_entropy)]

    losses_record = {}
    losses_record['actor_loss'] = epoch_actor_losses_record
    losses_record['critic_loss'] = epoch_critic_losses_record
    losses_record['classifier_loss'] = epoch_classifier_losses_record
    losses_record['entropy'] = epoch_approx_entropy_record
    return obs_record, action_record, losses_record


def learn(env, agent, reward_giver, expert_dataset,
          g_step, d_step, d_stepsize=3e-4, timesteps_per_batch=1024,
          nb_train_steps=50, max_timesteps=0, max_iters=0,  # TODO: max_episodes
          callback=None, d_adam=None, sess=None, saver=None
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    # Prepare for rollouts
    # ----------------------------------------
    timesteps_so_far = 0
    iters_so_far = 0

    assert sum([max_iters > 0, max_timesteps > 0]) == 1

    # TODO: implicit policy does not admit pretraining?

    # set up record
    policy_losses_record = {}
    discriminator_losses_record = {}

    while True:
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        #elif max_episodes and episodes_so_far >= max_episodes:
        #    break
        elif max_iters and iters_so_far >= max_iters:
            break

        logger.log("********** Iteration %i ************" % iters_so_far)
        logger.log("********** Steps %i ************" % timesteps_so_far)

        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        ob_policy, ac_policy, losses_record = train_one_batch(env, agent, reward_giver, timesteps_per_batch, nb_train_steps, g_step)
        assert len(ob_policy) == len(ac_policy) == timesteps_per_batch * g_step

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_policy))
        batch_size = len(ob_policy) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob_policy, ac_policy),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g, nworkers), d_stepsize)
            d_losses.append(newlosses)
        
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
        timesteps_so_far += timesteps_per_batch * g_step
        iters_so_far += 1

        # record
        for k,v in losses_record.items():
            if k in policy_losses_record.keys():
                policy_losses_record[k] += v
            else:
                policy_losses_record[k] = v
        for idx,k in enumerate(reward_giver.loss_name):
            if k in discriminator_losses_record.keys():
                discriminator_losses_record[k] += [np.mean(d_losses, axis=0)[idx]]
            else:
                discriminator_losses_record[k] = [np.mean(d_losses, axis=0)[idx]]

        # logging
        logger.record_tabular("Epoch Actor Losses", np.mean(losses_record['actor_loss']))
        logger.record_tabular("Epoch Critic Losses", np.mean(losses_record['critic_loss']))
        logger.record_tabular("Epoch Classifier Losses", np.mean(losses_record['classifier_loss']))
        logger.record_tabular("Epoch Entropy", np.mean(losses_record['entropy']))
        if rank == 0:
            logger.dump_tabular()

        # Call callback
        if callback is not None:
            callback(locals(), globals())


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
