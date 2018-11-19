"""
Training for Implicit Policy Optimization
"""
import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg_interpolate import DDPG_paramnoise
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

#from baselines.ddpg.evaluate import evaluate
import copy


def training_interpolate(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, maxactor, maxentactor, critic, classifier,
    normalize_returns, normalize_observations, critic_l2_reg, classifier_l2_reg, maxactor_lr, maxentactor_lr, critic_lr, classifier_lr,
    action_noise, popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, fifomemory,
    tau=0.01, eval_env=None, callback=None, entropy_coeff=1., beta=0.0, pretrained='none'):
    rank = MPI.COMM_WORLD.Get_rank()

    logger.info('noisynet implementation of DDPG')

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG_paramnoise(maxactor, maxentactor, critic, classifier, memory, fifomemory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, critic_l2_reg=critic_l2_reg, classifier_l2_reg=classifier_l2_reg,
        maxactor_lr=maxactor_lr, maxentactor_lr=maxentactor_lr, critic_lr=critic_lr, classifier_lr=classifier_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, entropy_coeff=entropy_coeff, beta=beta)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Copy an env for evaluation
    env_eval = copy.deepcopy(env.env)

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None
    
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        # load pretrained agent if possible
        if pretrained == 'none':
            logger.info('Training from scratch...')
        else:
            logger.info('Loading pretrained model from {}'.format(pretrained))
            #assert os.path.exists(pretrained)
            saver.restore(sess, pretrained)

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0
        total_time = 0

        epoch = 0
        start_time = time.time()

        total_time_record = []
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_maxactor_losses_record = []
        epoch_maxentactor_losses_record = []
        epoch_critic_losses_record = []
        epoch_classifier_losses_record = []
        epoch_approx_entropy_record = []
        epoch_end_xpos = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1
                    total_time += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        total_time_record.append(total_time)
                        epoch_end_xpos.append(obs[0])
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_maxactor_losses = []
                epoch_maxentactor_losses = []
                epoch_critic_losses = []
                epoch_classifier_losses = []
                epoch_approx_entropy = []
                for t_train in range(nb_train_steps):

                    cl, maxal, maxental, cll, ae = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_maxactor_losses.append(maxal)
                    epoch_maxentactor_losses.append(maxental)
                    epoch_classifier_losses.append(cll)
                    epoch_approx_entropy.append(ae)
                    agent.update_target_net()

                    #epoch_actor_losses_record.append(mpi_mean(epoch_actor_losses))
                    #epoch_critic_losses_record.append(mpi_mean(epoch_critic_losses))
                    #epoch_classifier_losses_record.append(mpi_mean(epoch_classifier_losses))
                    #epoch_approx_entropy_record.append(mpi_mean(epoch_approx_entropy))

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    # eval for one episode
                    eval_episode_reward = 0.0
                    eval_done = False
                    eval_obs = eval_env.reset()
                    while not eval_done:
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)
                        eval_episode_reward += eval_r
                        eval_qs.append(eval_q)
                    eval_episode_rewards.append(eval_episode_reward)
                    eval_episode_rewards_history.append(eval_episode_reward)

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/nb-epoch'] = epoch
            combined_stats['rollout/nb-cycle'] = cycle
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)
    
            # Train statistics.
            combined_stats['train/loss_maxactor'] = mpi_mean(epoch_maxactor_losses)
            combined_stats['train/loss_maxentactor'] = mpi_mean(epoch_maxentactor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/loss_classifier'] = mpi_mean(epoch_classifier_losses)
            combined_stats['train/approx_entropy'] = mpi_mean(epoch_approx_entropy)

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = mpi_mean(eval_qs)
                combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t
            
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

            # Call the callback
            if callback is not None:
                if callback(locals(),globals()):  # callback returns a boolean value
                    break

        # Evaluate the policy on env to record trajs
        if callback is not None:
            callback.final_call(locals(), globals())
