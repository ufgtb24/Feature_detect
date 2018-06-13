# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception Resnet V2 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from my_batch_norm import bn_layer_top
from myFunc import space_to_depth


def passthrough_layer(lowRes, highRes, kernel, depth, size, name):
    # 先降维
    highRes = slim.conv3d(highRes, depth, kernel, name)
    # space_to_depth https: // www.w3cschool.cn / tensorflow_python / tensorflow_python - rkfq2kf9.html
    # 不损失数据量的“下采样”，将size x size x 1 大小的数据块转换成 1 x 1 x (size*size) 的深度块
    highRes = space_to_depth(highRes, size)
    y = tf.concat([lowRes, highRes], axis=4)
    return y


def block16(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 16x16 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv3d(net, 16, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv3d(net, 16, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv3d(tower_conv1_0, 16, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv3d(net, 16, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv3d(tower_conv2_0, 24, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv3d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv3d(mixed, net.get_shape()[4], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
        
        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv3d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv3d(net, 64, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv3d(tower_conv1_0, 80, [1, 1, 3],
                                        scope='Conv2d_0b_1')
            tower_conv1_2 = slim.conv3d(tower_conv1_1, 96, [1, 3, 1],
                                        scope='Conv2d_0c_2')
            tower_conv1_3 = slim.conv3d(tower_conv1_2, 128, [3, 1, 1],
                                        scope='Conv2d_0c_3')
        mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_3])
        up = slim.conv3d(mixed, net.get_shape()[4], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
        
        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def block4(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv3d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv3d(net, 96, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv3d(tower_conv1_0, 112, [1, 1, 3],
                                        scope='Conv2d_0b_1')
            tower_conv1_2 = slim.conv3d(tower_conv1_1, 128, [1, 3, 1],
                                        scope='Conv2d_0c_2')
            tower_conv1_3 = slim.conv3d(tower_conv1_2, 144, [3, 1, 1],
                                        scope='Conv2d_0c_3')
        mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_3])
        up = slim.conv3d(mixed, net.get_shape()[4], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
        
        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def inception_resnet_v2_base(inputs,
                             output_stride=16,
                             align_feature_maps=True,
                             scope=None,
                             activation_fn=tf.nn.relu):
    """Inception model from  http://arxiv.org/abs/1602.07261.
  
    Constructs an Inception Resnet v2 network from inputs to the given final
    endpoint. This method can construct the network up to the final inception
    block Conv2d_7b_1x1.
  
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
      output_stride: A scalar that specifies the requested ratio of input to
        output spatial resolution. Only supports 8 and 16.
      align_feature_maps: When true, changes all the VALID paddings in the network
        to SAME padding so that the feature maps are aligned.
      scope: Optional variable_scope.
      activation_fn: Activation function for block scopes.
  
    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.
  
    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
        or if the output_stride is not 8 or 16, or if the output_stride is 8 and
        we request an end point after 'PreAuxLogits'.
    """
    if output_stride != 8 and output_stride != 16:
        raise ValueError('output_stride must be 8 or 16.')
    
    padding = 'SAME' if align_feature_maps else 'VALID'
    
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
        with slim.arg_scope([slim.conv3d, slim.max_pool3d, slim.avg_pool3d],
                            stride=1, padding='SAME'):
            # input is 128 1
            # 64 16
            net = slim.conv3d(inputs, 16, 3, stride=2, padding=padding,
                              scope='Conv2d_1a_3x3')
            
            # 64 32
            net = slim.conv3d(net, 32, 3, padding=padding,
                              scope='Conv2d_2a_3x3')
            # 32 32
            passthrough_32 = slim.max_pool3d(net, 3, stride=2, padding=padding,
                                  scope='MaxPool_3a_3x3')
            # 32 40
            net = slim.conv3d(passthrough_32, 40, 1, padding=padding,
                              scope='Conv2d_3b_1x1')
            
            # 32 96
            net = slim.conv3d(net, 96, 3, padding=padding,
                              scope='Conv2d_4a_3x3')
            # 16 96
            net = slim.max_pool3d(net, 3, stride=2, padding=padding,
                                  scope='MaxPool_5a_3x3')
            
            # 16 160
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv3d(net, 48, 1, scope='Conv2d_1x1')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv3d(net, 24, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv3d(tower_conv1_0, 32, 5,
                                                scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    tower_conv2_0 = slim.conv3d(net, 32, 1, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = slim.conv3d(tower_conv2_0, 48, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv2_2 = slim.conv3d(tower_conv2_1, 48, 3,
                                                scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    tower_pool = slim.avg_pool3d(net, 3, stride=1, padding='SAME',
                                                 scope='AvgPool_0a_3x3')
                    tower_pool_1 = slim.conv3d(tower_pool, 32, 1,
                                               scope='Conv2d_0b_1x1')
                net = tf.concat(
                    [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 4)
            
            # 16 160
            net = slim.repeat(net, 5, block16, scale=0.17,
                              activation_fn=activation_fn)

            # 16 160+16*8=288
            net= passthrough_layer(net,passthrough_32,3,16,2,'passThrough_32_16')
            
            # 16 160
            passthrough_16=slim.conv3d(net,160,1)

            
            # 8 544
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv3d(passthrough_16, 192, 3, stride=2,
                                             padding=padding,
                                             scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv3d(net, 128, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv3d(tower_conv1_0, 128, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv1_2 = slim.conv3d(tower_conv1_1, 192, 3,
                                                stride=2,
                                                padding=padding,
                                                scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    tower_pool = slim.max_pool3d(net, 3, stride=2,
                                                 padding=padding,
                                                 scope='MaxPool_1a_3x3')
                # 8 544
                net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 4)
                
            
            
            with slim.arg_scope([slim.conv3d], rate=1):
                # 8 544
                net = slim.repeat(net, 5, block8, scale=0.10,
                                  activation_fn=activation_fn)

            # 8 544+32*8=800
            net = passthrough_layer(net, passthrough_16, 3, 32, 2, 'passThrough_16_8')

            # 8 544
            passthrough_8 = slim.conv3d(net, 544, 1)

            # 4 1040
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv3d(passthrough_8, 128, 1, scope='Conv2d_0a_1x1')
                    tower_conv_1 = slim.conv3d(tower_conv, 192, 3, stride=2,
                                               padding=padding,
                                               scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    tower_conv1 = slim.conv3d(net, 128, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv3d(tower_conv1, 144, 3, stride=2,
                                                padding=padding,
                                                scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    tower_conv2 = slim.conv3d(net, 128, 1, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = slim.conv3d(tower_conv2, 144, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv2_2 = slim.conv3d(tower_conv2_1, 160, 3, stride=2,
                                                padding=padding,
                                                scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_3'):
                    tower_pool = slim.max_pool3d(net, 3, stride=2,
                                                 padding=padding,
                                                 scope='MaxPool_1a_3x3')
                net = tf.concat(
                    [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 4)
            
            # 4 1040
            net = slim.repeat(net, 4, block4, scale=0.20, activation_fn=activation_fn)
            net = block4(net, activation_fn=None)

            # 4 1040+64*8=1552
            net = passthrough_layer(net, passthrough_8, 3, 48, 2, 'passThrough_8_4')

            
            # 4 768
            net = slim.conv3d(net, 768, 1, scope='Conv2d_7b_1x1')
            return net


def inception_resnet_v2(inputs,
                        output_dim,
                        is_training=None,
                        reuse=None,
                        scope='InceptionResnetV2',
                        activation_fn=tf.nn.relu):
    """Creates the Inception Resnet V2 model.
  
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
        Dimension batch_size may be undefined. If create_aux_logits is false,
        also height and width may be undefined.
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before  dropout)
        are returned instead.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
      create_aux_logits: Whether to include the auxilliary logits.
      activation_fn: Activation function for conv3d.
  
    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0 or
        None).
      end_points: the set of end_points from the inception model.
    """
    
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs],
                           reuse=reuse) as scope:
        with slim.arg_scope([bn_layer_top, slim.dropout],
                            is_training=is_training):
            # 4 768
            net = inception_resnet_v2_base(inputs, scope=scope,
                                           activation_fn=activation_fn)
            
            with tf.variable_scope('Logits'):
                kernel_size = [4, 4, 4]
                net = slim.avg_pool3d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_8x8')
                
                net = slim.flatten(net)
                net = slim.dropout(net, keep_prob=0.5,scope='Dropout')
                pred = slim.fully_connected(net, output_dim, activation_fn=None,
                                              scope='Logits')
        
        return pred




def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  activation_fn=tf.nn.relu):
    """Returns the scope with the default parameters for inception_resnet_v2.
  
    Args:
      weight_decay: the weight decay for weights variables.
      batch_norm_decay: decay for the moving average of batch_norm momentums.
      batch_norm_epsilon: small float added to variance to avoid dividing by zero.
      activation_fn: Activation function for conv3d.
  
    Returns:
      a arg_scope with the parameters needed for inception_resnet_v2.
    """
    # Set weight_decay for weights in conv3d and fully_connected layers.
    with slim.arg_scope([slim.conv3d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_regularizer=slim.l2_regularizer(weight_decay)):
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv3d],
                            activation_fn=activation_fn,
                            normalizer_fn=bn_layer_top
                            ) as scope:
            return scope
