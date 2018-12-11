import gym
import tensorflow as tf
from stable_baselines import A2C,PPO2
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv

def nature_cnn1d_2(scaled_images, **kwargs):
	layer1 = tf.layers.conv1d(scaled_images, filters=20, kernel_size=4, strides=4, 
			padding='valid', data_format='channels_last', activation=tf.nn.relu)

	max_pool_1 = tf.layers.max_pooling1d(inputs=layer1, pool_size=4, strides=4, padding='same')

	layer2 = tf.layers.conv1d(max_pool_1, filters=64, kernel_size=4, strides=4, 
		padding='valid', data_format='channels_last', activation=tf.nn.relu)

	max_pool_2 = tf.layers.max_pooling1d(inputs=layer2, pool_size=2, strides=2, padding='same')

	flatten = tf.contrib.layers.flatten(max_pool_2)

	mlp = tf.contrib.layers.fully_connected(flatten,64)
	return mlp

def nature_cnn1d(scaled_images, **kwargs):

	layer1 = tf.layers.conv1d(scaled_images, filters=4, kernel_size=20, strides=5, 
		padding='valid', data_format='channels_last', activation=tf.nn.relu)
	max_pool_1 = tf.layers.max_pooling1d(inputs=layer1, pool_size=20, strides=5, padding='same')

	flatten = tf.contrib.layers.flatten(max_pool_1)
	mlp = tf.contrib.layers.fully_connected(flatten,16)
	return mlp

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CustomPolicy(ActorCriticPolicy):
	"""
	Policy object that implements actor critic, using a feed forward neural network.

	:param sess: (TensorFlow session) The current TensorFlow session
	:param ob_space: (Gym Space) The observation space of the environment
	:param ac_space: (Gym Space) The action space of the environment
	:param n_env: (int) The number of environments to run
	:param n_steps: (int) The number of steps to run for each environment
	:param n_batch: (int) The number of batch to run (n_envs * n_steps)
	:param reuse: (bool) If the policy is reusable or not
	:param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
	:param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
	:param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
	:param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
	"""

	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
		super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
										   reuse=reuse, scale=True)

		with tf.variable_scope("model", reuse=reuse):
			activ = tf.nn.relu

			extracted_features = nature_cnn1d(self.processed_x, **kwargs)
			extracted_features = tf.layers.flatten(extracted_features)

			pi_h = extracted_features
			for i, layer_size in enumerate([128, 128, 128]):
				pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
			pi_latent = pi_h

			vf_h = extracted_features
			for i, layer_size in enumerate([32, 32]):
				vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
			value_fn = tf.layers.dense(vf_h, 1, name='vf')
			vf_latent = vf_h

			self.proba_distribution, self.policy, self.q_value = \
				self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

		self.value_fn = value_fn
		self.initial_state = None
		self._setup_init()

	def step(self, obs, state=None, mask=None, deterministic=False):
		if deterministic:
			action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
												   {self.obs_ph: obs})
		else:
			action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
												   {self.obs_ph: obs})
		return action, value, self.initial_state, neglogp

	def proba_step(self, obs, state=None, mask=None):
		return self.sess.run(self.policy_proba, {self.obs_ph: obs})

	def value(self, obs, state=None, mask=None):
		return self.sess.run(self._value, {self.obs_ph: obs})

class CustomPolicy2(CustomPolicy):
	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
		super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
										   reuse=reuse, scale=True)

		with tf.variable_scope("model", reuse=reuse):
			activ = tf.nn.relu

			extracted_features = nature_cnn1d_2(self.processed_x, **kwargs)
			extracted_features = tf.layers.flatten(extracted_features)

			pi_h = extracted_features
			for i, layer_size in enumerate([128, 128, 128]):
				pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
			pi_latent = pi_h

			vf_h = extracted_features
			for i, layer_size in enumerate([32, 32]):
				vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
			value_fn = tf.layers.dense(vf_h, 1, name='vf')
			vf_latent = vf_h

			self.proba_distribution, self.policy, self.q_value = \
				self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

		self.value_fn = value_fn
		self.initial_state = None
		self._setup_init()

