"""
Models for normalizing flows implicit policy
"""
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from baselines.ddpg.models import Model


class NormalizingFlowStateModel(object):
    """
    Normalizingflow state conditional model
    accept state from outside computation graph
    Args:
        state_step: state tf variable for forward sampling (batch dimension cannot be None)
        state_train: state tf variable for backward training (batch dimension cannot be None)
        action: action tf variable for backward training
        name: name of the model
        reuse: if to reuse model
        num_units: num of hidden unit in s,t net
        num_layers: num of alternating units
    """
    def __init__(self, state, nb_action, name, reuse, num_units=3, num_layers=4):

        input_dim = nb_action # mujoco: not atari
        self.input_dim = input_dim
        m1, m2 = make_mask(input_dim)
        self.mask = [m1, m2] * num_layers
        print(self.mask)
        self.name = name
        self.state = state
        self.action = action
        self.reuse = reuse
        self.num_units = num_units

        with tf.variable_scope(name) as scope:
            self.build_forward(state, reuse=reuse)
            self.build_inverse(state, reuse=reuse)
            self.build_entropy()

    def build_forward(self, state, reuse):
        # build noise samples
        batch_size = [state.get_shape().as_list()[0], self.input_dim]
        noise_dist = tfd.Normal(loc=0., scale=1.)
        noise_samples = noise_dist.sample(batch_size)
        # build noise
        x_ph = noise_samples
        s_ph = state
        self.x_ph = x_ph
        self.s_ph = s_ph
        x = x_ph
        layers = []
        for i in range(len(self.mask)):
            layer = NormalizingFlowLayer(self.input_dim, self.mask[i], name='Nlayer_{}'.format(i), num_units=self.num_units)
            x = layer.forward(x, reuse=reuse)
            if i == 0: # fuse state information into the first layer
                self.statelayer = Layer(self.input_dim, name='statelayer', num_units=64)
                s_processed = self.statelayer(s_ph, reuse=reuse)
                x += s_processed
            layers.append(layer)
        self.y_sample = x
        self.layers = layers

    def build_entropy(self):
        """
        build entropy to allow for gradient computation of gradient
        a bit different from conventional methods since we use new seeds (instead of 
        sampled actions, since that way we need to track the original seeds and generate
        old actions)
        """
        # sample only one action per state seems to be of high variance
        # sample multiple actions per state
        # build noise samples
        batch_size = [self.state.get_shape().as_list()[0] * ENTROPY_NUMSAMPLES, self.input_dim]
        noise_dist = tfd.Normal(loc=0., scale=1.)
        noise_samples = noise_dist.sample(batch_size)        

        # build noise
        x_ph = noise_samples
        s_ph = tf.tile(self.state, [ENTROPY_NUMSAMPLES, 1])  # replicate states
        x = x_ph
        for i,layer in enumerate(self.layers):
            x = layer.forward(x, reuse=True)  # we are sure to reuse
            if i == 0: # fuse state information into the first layer
                s_processed = self.statelayer(s_ph, reuse=True)
                x += s_processed
        y_sample = x        

        y_ph = y_sample  # action from forward sampling
        y = y_ph
        log_det_sum = 0
        for i, layer in enumerate(self.layers[::-1]):
            if i == len(self.layers) - 1: # subtract state component
                s_processed = self.statelayer(s_ph, reuse=True)  # we are sure to reuse
                y -= s_processed
            y, logdet = layer.inverse(y)
            log_det_sum += logdet
        x_sample = y
        log_prior = -tf.reduce_sum(tf.square(x_sample), axis=1) / 2.0
        log_det = log_det_sum
        log_prob = log_prior - log_det
        self.entropy = -tf.reduce_mean(log_prob)
        

class NormalizingFlowsActor(Model):
    def __init__(self, nb_actions, name='actor', num_units=3, num_layers=4):
        super(NormalizingFlowsActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.num_units = num_units
        self.num_layers = num_layers
        self.name = name

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            self.model = NormalizingFlowStateModel(obs, self.nb_action, 'normalizingflow', reuse, num_units=self.num_units, num_layers=self.num_layers)
        return self.model.y_sample