import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.python.training import moving_averages


def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            # avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
            avg, var = tf.nn.moments(x, list(range(len(shape)-1)))

            #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            # update_moving_avg=tf.assign_add(moving_avg, moving_avg*decay+avg*(1-decay))

            update_moving_avg = moving_averages.assign_moving_average(
                moving_avg, avg, decay, zero_debias=False)

            #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            # update_moving_var=tf.assign_add(moving_var, moving_var*decay+var*(1-decay))

            update_moving_var = moving_averages.assign_moving_average(
                moving_var, var, decay, zero_debias=False)

            # #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            # update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
            # #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            # update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
            #
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output

@add_arg_scope
def bn_layer_top(x , is_training,scope='batch_norm', epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    #assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )