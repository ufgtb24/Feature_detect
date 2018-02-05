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
    return h


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

class Optimizer(object):
    def __init__(self, loss, initial_learning_rate, name,op_type='Adam', max_global_norm=1.0,
                   num_steps_per_decay=10000,decay_rate=0.1,momentum=None):
        """ Create a simple optimizer.

        This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.

        Args:
            loss: A 0-D float32 Tensor.
            initial_learning_rate: A float.
            num_steps_per_decay: An integer.
            decay_rate: A float. The factor applied to the learning rate
                every `num_steps_per_decay` steps.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm because `max_global_norm`.
        """
        with tf.variable_scope('Optimizer'+name):
            trainables = tf.trainable_variables()
            grads = tf.gradients(loss, trainables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
            grad_var_pairs = zip(grads, trainables)

            global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

            if op_type=='Momentum' or op_type=='RMSProp':
                assert  momentum != None,'No momentum specified'

            if op_type == 'GradientDescent' or op_type== 'Momentum':
                learning_rate = tf.train.exponential_decay(
                    initial_learning_rate, global_step, num_steps_per_decay,
                    decay_rate, staircase=True)
            else:
                learning_rate=initial_learning_rate

            optimizer={
                'GradientDescent':tf.train.GradientDescentOptimizer(learning_rate),
                'Momentum':tf.train.MomentumOptimizer(learning_rate,momentum=momentum),
                'AdaGrad': tf.train.AdagradOptimizer(learning_rate),
                'AdaDelta': tf.train.AdadeltaOptimizer(learning_rate),
                'RMSProp': tf.train.RMSPropOptimizer(learning_rate,momentum=momentum),
                'Adam': tf.train.AdamOptimizer(learning_rate),
            }[op_type]

            self._optimize_op = optimizer.apply_gradients(grad_var_pairs,
                                                          global_step=global_step)

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op
