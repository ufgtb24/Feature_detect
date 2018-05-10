# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from my_batch_norm import bn_layer_top
import tensorflow as tf

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception model from http://arxiv.org/abs/1512.00567.

  Constructs an Inception v3 network from inputs to the given final endpoint.
  This method can construct the network up to the final inception block
  Mixed_7c.

  Note that the names of the layers in the paper do not correspond to the names
  of the endpoints registered by this function although they build the same
  network.

  Here is a mapping from the old_names to the new names:
  Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with variable_scope.variable_scope(scope, 'Base', [inputs]):
    with arg_scope(
        [layers.conv3d, layers_lib.max_pool3d, layers_lib.avg_pool3d],
        stride=1,
        padding='SAME'):
      # 128 x 128 x 1
      end_point = 'Conv2d_1'
      net = layers.conv3d(inputs, depth(16), 3, stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # above is 64 x 64 x 16
      end_point = 'Conv2d_2'
      net = layers.conv3d(net, depth(32), 3, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 64 x 64 x 32
      end_point = 'MaxPool_2'
      net = layers_lib.max_pool3d(net, 2, stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 32 x 32 x 32.
      pass
      # increase dimension
      end_point = 'Conv2d_3'
      net = layers.conv3d(net, depth(40), 1, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 32 x 32 x 40.
      end_point = 'Conv2d_4'
      net = layers.conv3d(net, depth(96), 3, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 32 x 32 x 96.
      end_point = 'MaxPool_4'
      net = layers_lib.max_pool3d(net, 2, stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # above is 16 x 16 x 96.

      # Inception blocks
    with arg_scope(
        [layers.conv3d, layers_lib.max_pool3d, layers_lib.avg_pool3d],
        stride=1,
        padding='SAME'):
     # mixed: below is 16 x 16 x 128.
      end_point = 'Mixed_5a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(32), 1, scope='Conv2d_0a_1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(24), 1, scope='Conv2d_0a_1')
          branch_1 = layers.conv3d(
              branch_1, depth(32), 5, scope='Conv2d_0b_5')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv3d(
              net, depth(32), 1, scope='Conv2d_0a_1')
          branch_2 = layers.conv3d(
              branch_2, depth(48), 3, scope='Conv2d_0b_3')
          branch_2 = layers.conv3d(
              branch_2, depth(48), 3, scope='Conv2d_0c_3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_3')
          branch_3 = layers.conv3d(
              branch_3, depth(16), 1, scope='Conv2d_0b_1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_1: 16 x 16 x 144.
      end_point = 'Mixed_5b'
      with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
              branch_0 = layers.conv3d(
                  net, depth(32), 1, scope='Conv2d_0a_1')
          with variable_scope.variable_scope('Branch_1'):
              branch_1 = layers.conv3d(
                  net, depth(24), 1, scope='Conv2d_0a_1')
              branch_1 = layers.conv3d(
                  branch_1, depth(32), 5, scope='Conv2d_0b_5')
          with variable_scope.variable_scope('Branch_2'):
              branch_2 = layers.conv3d(
                  net, depth(32), 1, scope='Conv2d_0a_1')
              branch_2 = layers.conv3d(
                  branch_2, depth(48), 3, scope='Conv2d_0b_3')
              branch_2 = layers.conv3d(
                  branch_2, depth(48), 3, scope='Conv2d_0c_3')
          with variable_scope.variable_scope('Branch_3'):
              branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_2')
              branch_3 = layers.conv3d(
                  branch_3, depth(32), 1, scope='Conv2d_0b_1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed: 8 x 8 x 384.
      end_point = 'Mixed_6a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), 1, scope='Conv2d_0a_1')
          branch_0 = layers.conv3d(
              branch_0,
              depth(192), 3,
              stride=2,
              scope='Conv2d_1a_3')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(32), 1, scope='Conv2d_0a_1')
          branch_1 = layers.conv3d(
              branch_1, depth(48), 3, scope='Conv2d_0b_3')
          branch_1 = layers.conv3d(
              branch_1,
              depth(48), 3,
              stride=2,
              scope='Conv2d_1a_1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool3d(
              net, 2, stride=2,scope='MaxPool_1a_2')
        net = array_ops.concat([branch_0, branch_1, branch_2], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed4: 8 x 8 x 384.
      end_point = 'Mixed_6b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(64), 1, scope='Conv2d_0a_1')

          branch_1 = layers.conv3d(
              branch_1, depth(64), [1,1,3], scope='Conv2d_0b_1x1x3')
          branch_1 = layers.conv3d(
              branch_1, depth(64), [1,3,1], scope='Conv2d_0c_1x3x1')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [3,1,1], scope='Conv2d_0d_3x1x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv3d(
              net, depth(64), 1, scope='Conv2d_0a_1')

          branch_2 = layers.conv3d(
              branch_2, depth(64), [3,1,1], scope='Conv2d_0a_3x1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(64), [1,3,1], scope='Conv2d_0b_1x3x1')
          branch_2 = layers.conv3d(
              branch_2, depth(64), [1,1,3], scope='Conv2d_0c_1x1x3')

          branch_2 = layers.conv3d(
              branch_2, depth(64), [3,1,1], scope='Conv2d_0d_3x1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(64), [1,3,1], scope='Conv2d_0e_1x3x1')
          branch_2 = layers.conv3d(
              branch_2, depth(96), [1,1,3], scope='Conv2d_0f_1x1x3')

        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_2')
          branch_3 = layers.conv3d(
              branch_3, depth(96), 1, scope='Conv2d_0b_1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed4: 8 x 8 x 384.
      end_point = 'Mixed_6c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(80), 1, scope='Conv2d_0a_1')

          branch_1 = layers.conv3d(
              branch_1, depth(80), [1,1,3], scope='Conv2d_0b_1x1x3')
          branch_1 = layers.conv3d(
              branch_1, depth(80), [1,3,1], scope='Conv2d_0c_1x3x1')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [3,1,1], scope='Conv2d_0c_3x1x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv3d(
              net, depth(80), 1, scope='Conv2d_0a_1')

          branch_2 = layers.conv3d(
              branch_2, depth(80), [3,1,1], scope='Conv2d_0b_3x1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(80), [1,3,1], scope='Conv2d_0b_1x3x1')
          branch_2 = layers.conv3d(
              branch_2, depth(80), [1,1,3], scope='Conv2d_0c_1x1x3')

          branch_2 = layers.conv3d(
              branch_2, depth(80), [3,1,1], scope='Conv2d_0d_3x1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(80), [1,3,1], scope='Conv2d_0e_1x3x1')
          branch_2 = layers.conv3d(
              branch_2, depth(96), [1,1,3], scope='Conv2d_0f_1x7')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_2')
          branch_3 = layers.conv3d(
              branch_3, depth(96), 1, scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed4: 8 x 8 x 384.
      end_point = 'Mixed_6d'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')

          branch_1 = layers.conv3d(
              branch_1, depth(96), [1,1,3], scope='Conv2d_0b_1x1x3')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [1,3,1], scope='Conv2d_0c_1x3x1')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [3,1,1], scope='Conv2d_0c_3x1x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')

          branch_2 = layers.conv3d(
              branch_2, depth(96), [3,1,1], scope='Conv2d_0b_3x1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(96), [1,3,1], scope='Conv2d_0b_1x3x1')
          branch_2 = layers.conv3d(
              branch_2, depth(96), [1,1,3], scope='Conv2d_0c_1x1x3')

          branch_2 = layers.conv3d(
              branch_2, depth(96), [3,1,1], scope='Conv2d_0d_3x1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(96), [1,3,1], scope='Conv2d_0e_1x3x1')
          branch_2 = layers.conv3d(
              branch_2, depth(96), [1,1,3], scope='Conv2d_0f_1x1x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_2')
          branch_3 = layers.conv3d(
              branch_3, depth(96), 1, scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points


      # mixed_8: 4 x 4 x 640.
      end_point = 'Mixed_7a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')
          branch_0 = layers.conv3d(
              branch_0,
              depth(160), 3,
              stride=2,
              scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(96), 1, scope='Conv2d_0a_1')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [1,1,3], scope='Conv2d_0a_1x1x3')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [1,3,1], scope='Conv2d_0b_1x3x1')
          branch_1 = layers.conv3d(
              branch_1, depth(96), [3,1,1], scope='Conv2d_0c_3x1x1')
          branch_1 = layers.conv3d(
              branch_1,
              depth(96), 3,
              stride=2,
              scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool3d(
              net, 2, stride=2,  scope='MaxPool_1a_2')
        net = array_ops.concat([branch_0, branch_1, branch_2], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # mixed_9: 4 x 4 x 1408.
      end_point = 'Mixed_7b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(160), 1, scope='Conv2d_0a_1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(192), 1, scope='Conv2d_0a_1')
          branch_1 = array_ops.concat(
              [
                  layers.conv3d(
                      branch_1, depth(192), [1,1,3], scope='Conv2d_0b_1x1x3'),
                  layers.conv3d(
                      branch_1, depth(192), [1,3,1], scope='Conv2d_0c_1x3x1'),
                  layers.conv3d(
                      branch_1, depth(192), [3,1,1], scope='Conv2d_0d_3x1x1'),
              ],
              4)
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv3d(
              net, depth(224), 1, scope='Conv2d_0a_1x1')
          branch_2 = layers.conv3d(
              branch_2, depth(192), 3, scope='Conv2d_0b_3x3')
          branch_2 = array_ops.concat(
              [
                  layers.conv3d(
                      branch_2, depth(192), [1,1,3], scope='Conv2d_0c_1x1x3'),
                  layers.conv3d(
                      branch_2, depth(192), [1,3,1], scope='Conv2d_0d_1x3x1'),
                  layers.conv3d(
                      branch_2, depth(192), [3,1,1], scope='Conv2d_0e_3x1x1'),
              ],
              4)
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_2')
          branch_3 = layers.conv3d(
              branch_3, depth(96), 1, scope='Conv2d_0b_1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_10: 4 x 4 x 1408.
      end_point = 'Mixed_7c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv3d(
              net, depth(160), 1, scope='Conv2d_0a_1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv3d(
              net, depth(192), 1, scope='Conv2d_0a_1')
          branch_1 = array_ops.concat(
              [
                  layers.conv3d(
                      branch_1, depth(192), [1,1,3], scope='Conv2d_0b_1x1x3'),
                  layers.conv3d(
                      branch_1, depth(192), [1,3,1], scope='Conv2d_0c_1x3x1'),
                  layers.conv3d(
                      branch_1, depth(192), [3,1,1], scope='Conv2d_0d_3x1x1'),
              ],
              4)
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv3d(
              net, depth(224), 1, scope='Conv2d_0a_1')
          branch_2 = layers.conv3d(
              branch_2, depth(192), 3, scope='Conv2d_0b_3')
          branch_2 = array_ops.concat(
              [
                  layers.conv3d(
                      branch_2, depth(192), [1,1,3], scope='Conv2d_0c_1x1x3'),
                  layers.conv3d(
                      branch_2, depth(192), [1,3,1], scope='Conv2d_0d_1x3x1'),
                  layers.conv3d(
                      branch_2, depth(192), [3,1,1], scope='Conv2d_0e_3x1x1'),
              ],
              4)
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool3d(net, 2, scope='AvgPool_0a_3')
          branch_3 = layers.conv3d(
              branch_3, depth(96), 1, scope='Conv2d_0b_1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 4)
      end_points[end_point] = net
      return net, end_points


def inception_v3(inputs,
                 output_dim,
                 is_training=None,
                 min_depth=16,
                 depth_multiplier=1.0,
                 reuse=None,
                 scope='InceptionV3'):
  """Inception model from http://arxiv.org/abs/1512.00567.

  "Rethinking the Inception Architecture for Computer Vision"

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna.

  With the default arguments this method constructs the exact model defined in
  the paper. However, one can experiment with variations of the inception_v3
  network by changing arguments dropout_keep_prob, min_depth and
  depth_multiplier.

  The default image size used to train this network is 299x299.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(
      scope, reuse=reuse):
    with arg_scope(
        [bn_layer_top, layers_lib.dropout], is_training=is_training):
      net, end_points = inception_v3_base(
          inputs,
          scope='Base',
          min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      with variable_scope.variable_scope('Logits'):
        # kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
        kernel_size = [4,4,4]
        net = layers_lib.avg_pool3d(
            net,
            kernel_size,
            padding='VALID',
            scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 1024
        net = layers_lib.dropout(
            net, keep_prob=0.5, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 1024
        logits = layers.conv3d(
            net,
            output_dim, 1,
            activation_fn=None,
            normalizer_fn=None,
            scope='task_spec_conv')
      # 6
      logits = array_ops.squeeze(logits, [1, 2, 3], name='SpatialSqueeze')
  return logits

inception_v3.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.tf.contrib.slim.ops._two_element_tuple
  cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [
        min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
    ]
  return kernel_size_out


def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1):
  """Defines the default InceptionV3 arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """

  # Set weight_decay for weights in Conv and FC layers.
  with arg_scope(
      [layers.conv3d, layers_lib.fully_connected],
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope(
        [layers.conv3d],
        weights_initializer=init_ops.truncated_normal_initializer(
            stddev=stddev),
        activation_fn=nn_ops.relu,
        normalizer_fn=bn_layer_top,
        ) as sc:
      return sc
