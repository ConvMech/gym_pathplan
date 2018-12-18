import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

n_input = 360
time_step = 5

# 5-D tensor
x = tf.placeholder(tf.float32, [None, time_step, n_input,3])
y = tf.placeholder(tf.float32, [None, n_classes])

def convlstm(x):
    convlstm_layer= tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=1,
                input_shape=[28, 28, channel],
                output_channels=32,
                kernel_shape=[3, 3],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv_lstm_cell")
    
    initial_state = convlstm_layer.zero_state(batch_size, dtype=tf.float32)
    outputs,_=tf.nn.dynamic_rnn(convlstm_layer,x,initial_state=initial_state,time_major=False,dtype="float32")
    return outputs

lstm_out = convlstm(x)

max_pool_2 = tf.layers.max_pooling1d(inputs=layer2, pool_size=2, strides=2, padding='same')

flatten = tf.contrib.layers.flatten(max_pool_2)

mlp = tf.contrib.layers.fully_connected(flatten,64)

with tf.Session() as sess:
	sess.run(tf.initializers.global_variables())
	res = sess.run(max_pool_2,feed_dict = {inputs_: raw})
	print(res.shape)
 
