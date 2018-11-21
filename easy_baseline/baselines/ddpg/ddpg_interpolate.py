"""
DDPG for Implicit Policy Optimization
"""
from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ddpg.util import reduce_std, mpi_mean

import tensorflow as tf
tfd = tf.contrib.distributions


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


class DDPG_paramnoise(object):
    """
    Implicit Policy Optimization for DDPG
    noise injected in the middle of blackbox (param noise)
    """
    def __init__(self, maxactor, maxentactor, critic, classifier, memory, fifomemory, observation_shape, action_shape, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., classifier_l2_reg=0., maxactor_lr=1e-4, maxentactor_lr=1e-4, critic_lr=1e-3, classifier_lr=1e-3, clip_norm=None,
        reward_scale=1., entropy_coeff=1., beta=0.0):
        # Inputs.
        self.obs0_act = tf.placeholder(tf.float32, shape=(1,) + observation_shape, name='obs0_act')
        self.obs0_train = tf.placeholder(tf.float32, shape=(batch_size,) + observation_shape, name='obs0_train')
        self.obs1 = tf.placeholder(tf.float32, shape=(batch_size,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions_act = tf.placeholder(tf.float32, shape=(1,) + action_shape, name='actions_act')
        self.actions_train = tf.placeholder(tf.float32, shape=(64,) + action_shape, name='actions_train')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.fifomemory = fifomemory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.action_shape = action_shape
        self.critic = critic
        self.maxactor = maxactor
        self.maxentactor = maxentactor
        self.classifier = classifier
        self.maxactor_lr = maxactor_lr
        self.maxentactor_lr = maxentactor_lr
        self.critic_lr = critic_lr
        self.classifier_lr = classifier_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.classifier_l2_reg = classifier_l2_reg
        self.entropy_coeff = entropy_coeff

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0_act = tf.clip_by_value(normalize(self.obs0_act, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs0_train = tf.clip_by_value(normalize(self.obs0_train, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        
        self.normalized_obs0_act = normalized_obs0_act  # record normalized_obs0
        self.normalized_obs0_train = normalized_obs0_train
        self.normalized_obs1 = normalized_obs1

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_maxactor = copy(maxactor)
        target_maxentactor = copy(maxentactor)
        target_maxactor.name = 'target_maxactor'
        self.target_maxactor = target_maxactor
        target_maxentactor.name = 'target_maxentactor'
        self.target_maxentactor = target_maxentactor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.maxactor_tf_act = maxactor(normalized_obs0_act)
        self.maxentactor_tf_act = maxentactor(normalized_obs0_act)
        self.maxactor_tf_train = maxactor(normalized_obs0_train, reuse=True)
        self.maxentactor_tf_train = maxentactor(normalized_obs0_train, reuse=True)
        nb_actions = maxactor.nb_actions

        # Create interpolated action for act
        batch_act = self.maxactor_tf_act.get_shape().as_list()[0]
        mask_act = tf.random_uniform(tf.stack([batch_act]), minval=0, maxval=1, dtype=tf.float32) < beta
        self.actor_tf_act = tf.where(mask_act, self.maxactor_tf_act, self.maxentactor_tf_act)

        # Create interpolated action for train
        batch_train = self.maxactor_tf_train.get_shape().as_list()[0]
        mask_train = tf.random_uniform(tf.stack([batch_train]), minval=0, maxval=1, dtype=tf.float32) < beta
        self.actor_tf_train = tf.where(mask_train, self.maxactor_tf_train, self.maxentactor_tf_train)

        # Create graphs for critic for train
        self.normalized_critic_tf = critic(normalized_obs0_train, self.actions_train)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_maxactor_tf = critic(normalized_obs0_train, self.maxactor_tf_train, reuse=True)
        self.normalized_critic_with_maxentactor_tf = critic(normalized_obs0_train, self.maxentactor_tf_train, reuse=True)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0_act, self.actor_tf_act, reuse=True)  # act
        self.critic_with_maxactor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_maxactor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.critic_with_maxentactor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_maxentactor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        # Create interpolated target action for train
        batch_train = normalized_obs0_train.get_shape().as_list()[0]
        mask_train = tf.random_uniform(tf.stack([batch_train]), minval=0, maxval=1, dtype=tf.float32) < beta
        self.target_actions = tf.where(mask_train, self.target_maxactor(normalized_obs1), self.target_maxentactor(normalized_obs1))        
        Q_obs1 = denormalize(target_critic(normalized_obs1, self.target_actions), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Create graphs for critic for act
        #self.normalized_critic_tf_act = critic(normalized_obs0_act, self.actions_act)
        #self.critic_tf_act = denormalize(tf.clip_by_value(self.normalized_critic_tf_act, self.return_range[0], self.return_range[1]), self.ret_rms)

        # Classifier Network
        self.random_actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='random_actions')
        #self.logit = classifier(normalized_obs0_train, self.actor_tf_train)  # actions produced by policy for backprop
        self.logit = classifier(normalized_obs0_train, self.maxentactor_tf_train)
        self.random_logit = classifier(normalized_obs0_train, self.random_actions, reuse=True)

        # Set up parts.
        self.setup_approx_entropy()
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        self.setup_classifier_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.maxactor.vars, self.target_maxactor.vars, self.tau)
        actor_init_updates_, actor_soft_updates_ = get_target_updates(self.maxentactor.vars, self.target_maxentactor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, actor_init_updates_, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, actor_soft_updates_, critic_soft_updates]

    def setup_approx_entropy(self):
        logger.info('setting up approx entropy')
        self.approx_entropy = -tf.reduce_mean(self.logit)

    def setup_actor_optimizer(self):
        # maxactor
        logger.info('setting up maxactor optimizer')
        self.maxactor_loss = -tf.reduce_mean(self.critic_with_maxactor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.maxactor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        # Add entropy into actor loss
        self.maxactor_grads = U.flatgrad(self.maxactor_loss, self.maxactor.trainable_vars, clip_norm=self.clip_norm)
        self.maxactor_optimizer = MpiAdam(var_list=self.maxactor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

        # maxentactor
        logger.info('setting up maxentactor optimizer')
        self.maxentactor_loss = -tf.reduce_mean(self.critic_with_maxentactor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.maxentactor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        logger.info('using entropy coeff {}'.format(self.entropy_coeff))
        self.maxentactor_loss += -self.entropy_coeff * self.approx_entropy
        # Add entropy into actor loss
        self.maxentactor_grads = U.flatgrad(self.maxentactor_loss, self.maxentactor.trainable_vars, clip_norm=self.clip_norm)
        self.maxentactor_optimizer = MpiAdam(var_list=self.maxentactor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)        

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_classifier_optimizer(self):
        logger.info('setting up classifier optimizer')
        #self.classifier_loss = - (tf.reduce_mean(tf.log(1e-8 + tf.sigmoid(self.logit)))
        #                          + tf.reduce_mean(tf.log(1e-8 + 1 - tf.sigmoid(self.random_logit))))
        label_zeros = tf.zeros_like(self.logit)
        label_ones = tf.ones_like(self.random_logit)
        self.classifier_loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=label_zeros))
                                + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.random_logit, labels=label_ones)))
        if self.classifier_l2_reg > 0.:
            classifier_reg_vars = [var for var in self.classifier.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in classifier_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.classifier_l2_reg))
            classifier_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.classifier_l2_reg),
                weights_list=classifier_reg_vars
            )
            self.classifier_loss += classifier_reg
        classifier_shapes = [var.get_shape().as_list() for var in self.classifier.trainable_vars]
        classifier_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in classifier_shapes])
        logger.info('  classifier shapes: {}'.format(classifier_shapes))
        logger.info('  classifier params: {}'.format(classifier_nb_params))
        self.classifier_grads = U.flatgrad(self.classifier_loss, self.classifier.trainable_vars, clip_norm=self.clip_norm)
        self.classifier_optimizer = MpiAdam(var_list=self.classifier.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)        

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean
        
        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []
        
        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']
        
        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']
        
        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        #ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        #names += ['reference_actor_Q_mean']
        #ops += [reduce_std(self.critic_with_actor_tf)]
        #names += ['reference_actor_Q_std']
        
        ops += [tf.reduce_mean(self.actor_tf_train)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf_train)]
        names += ['reference_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, apply_noise=True, compute_Q=True):
        if apply_noise:
            actor_tf = self.actor_tf_act  # TODO: handle apply_noise=False mode
        else:
            actor_tf = self.actor_tf_act  # should take the mean?? probably not
        feed_dict = {self.obs0_act: [obs]}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)
        self.fifomemory.append(obs0, action)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q, new_mean, new_std)
            # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get a batch from recent policy then update classifier
        batch_recent = self.fifomemory.sample(batch_size=self.batch_size)
        random_actions = np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=[self.batch_size,
                                           np.prod(np.array(self.action_shape))]).astype('float32')
        ops = [self.classifier_grads, self.classifier_loss, self.approx_entropy]
        classifier_grads, classifier_loss, approx_entropy = self.sess.run(ops, feed_dict={
            self.obs0_train: batch_recent['obs0'],
            self.random_actions: random_actions
            })
        self.classifier_optimizer.update(classifier_grads, stepsize=self.classifier_lr)

        # Get all gradients and perform a synced update.
        ops = [self.maxactor_grads, self.maxactor_loss, self.maxentactor_grads, self.maxentactor_loss, self.critic_grads, self.critic_loss]
        maxactor_grads, maxactor_loss, maxentactor_grads, maxentactor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0_train: batch['obs0'],
            self.actions_train: batch['actions'],
            self.critic_target: target_Q,
        })
        self.maxactor_optimizer.update(maxactor_grads, stepsize=self.maxactor_lr)
        self.maxentactor_optimizer.update(maxentactor_grads, stepsize=self.maxentactor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, maxactor_loss, maxentactor_loss, classifier_loss, approx_entropy

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.maxactor_optimizer.sync()
        self.maxentactor_optimizer.sync()
        self.critic_optimizer.sync()
        self.classifier_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
            #print(self.stats_sample['obs0'].shape, self.stats_sample['actions'].shape)
            #print(self.obs0_train, self.actions_train)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0_train: self.stats_sample['obs0'],
            self.actions_train: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))
        return stats

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()