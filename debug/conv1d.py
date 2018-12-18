import tensorflow as tf
import numpy as np

seq_len = 1080
n_channels = 2

raw = np.random.rand(1080,2)
raw = [raw]

inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels],name = 'inputs')

layer1 = tf.layers.conv1d(inputs_, filters=20, kernel_size=4, strides=2, 
        padding='valid', data_format='channels_last', activation=tf.nn.relu)

max_pool_1 = tf.layers.max_pooling1d(inputs=layer1, pool_size=4, strides=2, padding='same')

layer2 = tf.layers.conv1d(max_pool_1, filters=64, kernel_size=4, strides=2, 
    padding='valid', data_format='channels_last', activation=tf.nn.relu)

max_pool_2 = tf.layers.max_pooling1d(inputs=layer2, pool_size=2, strides=2, padding='same')


flatten = tf.contrib.layers.flatten(max_pool_2)

flatten = tf.contrib.layers.fully_connected(flatten,256)

mlp = tf.contrib.layers.fully_connected(flatten,64)

with tf.Session() as sess:
	sess.run(tf.initializers.global_variables())
	res = sess.run(mlp,feed_dict = {inputs_: raw})
	print(res.shape)
 
