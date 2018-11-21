import tensorflow as tf
tfd = tf.contrib.distributions

observations_ph = tf.placeholder(tf.float32, [None, 10])
if True:
    if True:
        batch_size = observations_ph.get_shape().as_list()[0]
        #random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)







