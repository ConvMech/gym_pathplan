"""
Models for Implicit Policy Optimization
"""
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from baselines.ddpg.models import Model


class NoiseDense(tf.layers.Dense):
    """
    Noise Dense Layer
    """
    def __init__(self, *args, rho_W=-4, rho_b=-4, factorized=False, **kwargs):
        super(NoiseDense, self).__init__(*args, **kwargs)
        self.rho_W = rho_W
        self.rho_b = rho_b
        self.factorized = factorized

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to NoiseDense'
                             'should be defined. Found None')
        self.input_spec = tf.layers.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel_mu = self.add_variable('kernel_mu',
                                         shape=[input_shape[-1].value, self.units],
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint,
                                         dtype=self.dtype,
                                         trainable=True)
        self.kernel_rho = self.add_variable('kernel_rho',
                                         shape=[input_shape[-1].value, self.units],
                                         initializer=tf.constant_initializer(self.rho_W),
                                         regularizer=None,
                                         constraint=None,
                                         dtype=self.dtype,
                                         trainable=True)        
        self.kernel = tf.random_normal(tf.shape(self.kernel_mu),
                                       mean=self.kernel_mu, stddev=tf.nn.softplus(self.kernel_rho))
        if self.use_bias:
            self.bias_mu = self.add_variable('bias_mu',
                                          shape=[self.units,],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)
            self.bias_rho = self.add_variable('bias_rho',
                                          shape=[self.units,],
                                          initializer=tf.constant_initializer(self.rho_b),
                                          regularizer=None,
                                          constraint=None,
                                          dtype=self.dtype,
                                          trainable=True)
            self.bias = tf.random_normal(tf.shape(self.bias_mu),
                                           mean=self.bias_mu, stddev=tf.nn.softplus(self.bias_rho))
        else:
            self.bias = None
        self.built = True


def noisedense(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None,
    rho_W=-4,
    rho_b=-4,
    factorized=False):
    """Functional interface for the  noise densely-connected layer.
    """
    layer = NoiseDense(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse,
                rho_W=rho_W,
                rho_b=rho_b,
                factorized=factorized)
    return layer.apply(inputs)


class NoiseActor(Model):
    def __init__(self, nb_actions, rho_W=-4, rho_b=-4, factorized=False, name='actor', layer_norm=True):
        super(NoiseActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.rho_W = rho_W
        self.rho_b = rho_b
        self.factorized = factorized

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = noisedense(x, self.nb_actions, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized,
                           kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class NoiseDropoutActor(Model):
    def __init__(self, nb_actions, rho_W=-4, rho_b=-4, factorized=False, name='actor', layer_norm=True, p=0.5):
        super(NoiseDropoutActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.rho_W = rho_W
        self.rho_b = rho_b
        self.factorized = factorized
        self.prob = p

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, keep_prob=self.prob)

            x = noisedense(x, self.nb_actions, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized,
                           kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Classifier(Model):
    """
	classifier takes a state/action pair and computes its logit for being 
	sampled from the given policy
	"""
    def __init__(self, name='classifier', layer_norm=True):
        super(Classifier, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class NoiseSoftMaxActor(Model):
    # output a probability vector (softmax at last layer)
    def __init__(self, nb_actions, rho_W=-4.0, rho_b=-4.0, factorized=False, name='actor', layer_norm=True):
        super(NoiseSoftMaxActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.rho_W = rho_W
        self.rho_b = rho_b
        self.factorized = factorized

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = noisedense(x, self.nb_actions, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized,
            	kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #x = tf.nn.tanh(x)
            x = tf.nn.softmax(x)
        return x


class NoiseDropoutSoftMaxActor(Model):
    # output a probability vector (softmax at last layer)
    def __init__(self, nb_actions, rho_W=-4.0, rho_b=-4.0, factorized=False, name='actor', layer_norm=True, p=0.5):
        super(NoiseDropoutSoftMaxActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.rho_W = rho_W
        self.rho_b = rho_b
        self.factorized = factorized
        self.prob = p

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, keep_prob=self.prob)

            x = noisedense(x, self.nb_actions, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized,
            	kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #x = tf.nn.tanh(x)
            x = tf.nn.softmax(x)
        return x


# === stochastically combining multiple actors into one actor === #
class NoiseDropoutMultipleActor(Model):
    def __init__(self, nb_actions, rho_W=-4, rho_b=-4, factorized=False, name='actor', layer_norm=True, p=0.5, K=10):
        super(NoiseDropoutMultipleActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.rho_W = rho_W
        self.rho_b = rho_b
        self.factorized = factorized
        self.prob = p
        self.name = name
        self.K = K

    def __call__(self, obs, reuse=False):
        xdict = []
        K = self.K
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            for k in range(K):
                x = obs
                x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
            
                x = noisedense(x, 64, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                x = tf.nn.dropout(x, keep_prob=self.prob)

                x = noisedense(x, self.nb_actions, rho_W=self.rho_W, rho_b=self.rho_b, factorized=self.factorized)
                               #kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)
                xdict.append(x)
        xall = tf.concat(xdict, axis=1) # size of [batchsize, action dim * K]
        # generate masks
        logits = [0.0] * K
        num_samples = obs.shape.as_list()[0]
        categorical_mask = tf.multinomial([logits], num_samples)
        #print('categoricalmask', categorical_mask)
        onehot_mask = tf.squeeze(tf.one_hot(categorical_mask, K), 0)
        #print('onehotmask', onehot_mask)
        onehot_mask_tiled = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(onehot_mask,axis=2),[1,1,self.nb_actions]),[-1,self.nb_actions * K, 1]),axis=2)
        # select
        action_tiled = tf.multiply(onehot_mask_tiled, xall)  # size of [batchsize, action dim * K]
        action = tf.reshape(action_tiled, [-1, K, self.nb_actions])  # size of [batchsize, K, action dim]
        x = tf.reduce_sum(action, axis=1)
        return x  # size of [batchsize, action dim]


debug = False
if debug:
    x = tf.placeholder(tf.float32, [None, 4])
    actor = NoiseActor(2,rho_W=0,rho_b=0)
    act = actor(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        xx = np.ones([10,4], dtype=np.float32)
        for _ in range(10):
            print(sess.run(act, feed_dict={x: xx}))








