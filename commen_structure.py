import numpy as np
import tensorflow as tf

def _weight_variable( shape, name):
    with tf.variable_scope(name):
        # edge=math.sqrt(6.0/(shape[0]+shape[1]))
        # initializer = tf.random_uniform_initializer(minval=-edge,maxval=edge)
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

def _bias_variable( shape, name):
    with tf.variable_scope(name):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

# def conv3d(input, output_channels, ft_size,phase,pooling, name="conv3d"):
#     with tf.variable_scope(name):
#         filter =_weight_variable(shape=[ft_size]*3+[input.shape[-1]]+[output_channels],name='w')
#         conv = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding='VALID')
#         biases = _bias_variable(output_channels,name='b')
#         activation = tf.nn.relu(conv + biases)
#         if pooling:
#             activation=pooling3d(activation)
#         #先激活，后bn效果更好
#         bn = tf.contrib.layers.batch_norm(activation, center=True, scale=True,
#                                           decay=0.999, is_training=phase,
#                                           updates_collections=None)
#     return bn

def conv3d(input, output_channels, phase,pooling,filter_size=3,stride=1, scope="conv3d"):
    with tf.variable_scope(scope):
        filter =_weight_variable(shape=[filter_size]*3+[input.shape[-1]]+[output_channels],name='w')
        conv = tf.nn.conv3d(input, filter, strides=[1]+[stride]*3+[1], padding='SAME')
        #标准模式，先bn，再激活
        h = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                          decay=0.999, is_training=phase,
                                          updates_collections=None)
        h = tf.nn.relu(h)
        if pooling:
            h=pooling3d(h)
    return h,filter


#要计算位置，不能要，否则就不准确了
def pooling3d(input_tensor):
    """
	Handy wrapper function for convolutional networks.

	Performs 2D max pool with a default stride of 2.
	"""
    pool = tf.nn.max_pool3d(input_tensor, ksize=[1, 2, 2,2, 1], strides=[1, 2, 2,2, 1], padding='SAME')
    return pool


def dense(x, output_size, scope):
    return tf.contrib.layers.fully_connected(x, output_size,
                                             activation_fn=None,
                                             scope=scope)

def dense_relu_batch(x, output_size, phase, scope):
    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(x, output_size,
                                               activation_fn=None,
                                               scope='dense')
        h = tf.contrib.layers.batch_norm(h,
                                          center=True, scale=True,
                                          decay=0.999,
                                          is_training=phase,
                                          scope='bn')
        h = tf.nn.relu(h, 'relu')
        return h

def dense_relu_batch_dropout(x, output_size, phase,keep_prob, scope):
    with tf.variable_scope(scope):
        h=dense_relu_batch(x, output_size, phase, scope)
        h=tf.nn.dropout(h, keep_prob)
        return h

def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat


def mse(output, target):
    with tf.variable_scope('MSE'):
        return tf.reduce_mean(tf.reduce_sum(tf.square(output - target), axis=1))


