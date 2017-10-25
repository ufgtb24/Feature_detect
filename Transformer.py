import tensorflow as tf
from math import pi, sin, cos
import numpy as np


# 从目标变回输入的矩阵，需要角度和平移相反
# 输入共7维



def get_transformer_matrix(theta,u,t):
    '''
    :param theta: [b,1]
    :param u: [b,3]
    :param t: [b,3]
    :return:
    '''

    line_1 = tf.stack([tf.cos(theta) + u[:, [0]] ** 2 * (1 - tf.cos(theta)),
                       u[:, [0]] * u[:, [1]] * (1 - tf.cos(theta)) - u[:, [2]] * tf.sin(theta),
                       u[:, [0]] * u[:, [2]] * (1 - tf.cos(theta)) + u[:, [1]] * tf.sin(theta),
                       t[:, [0]]])
    line_2 = tf.stack([u[:, [0]] * u[:, [1]] * (1 - tf.cos(theta)) + u[:, [2]] * tf.sin(theta),
                       tf.cos(theta) + u[:, [1]] ** 2 * (1 - tf.cos(theta)),
                       u[:, [1]] * u[:, [2]] * (1 - tf.cos(theta)) - u[:, [0]] * tf.sin(theta),
                       t[:, [1]]])
    line_3 = tf.stack([u[:, [0]] * u[:, [2]] * (1 - tf.cos(theta)) - u[:, [1]] * tf.sin(theta),
                       u[:, [1]] * u[:, [2]] * (1 - tf.cos(theta)) + u[:, [0]] * tf.sin(theta),
                       tf.cos(theta) + u[:, [2]] ** 2 * (1 - tf.cos(theta)),
                       t[:, [2]]])
    matrix = tf.stack([line_1, line_2, line_3])

    matrix = tf.squeeze(matrix, axis=[3])
    #[b,3,4]
    return tf.transpose(matrix, [2, 0, 1])


def get_transformer_matrix_np(theta,u,t):
    '''
    :param theta: [b,1]
    :param u: [b,3]
    :param t: [b,3]
    :return:
    '''
    line_1 = np.stack([np.cos(theta) + u[:, [0]] ** 2 * (1 - np.cos(theta)),
                       u[:, [0]] * u[:, [1]] * (1 - np.cos(theta)) - u[:, [2]] * np.sin(theta),
                       u[:, [0]] * u[:, [2]] * (1 - np.cos(theta)) + u[:, [1]] * np.sin(theta),
                       t[:, [0]]])
    line_2 = np.stack([u[:, [0]] * u[:, [1]] * (1 - np.cos(theta)) + u[:, [2]] * np.sin(theta),
                       np.cos(theta) + u[:, [1]] ** 2 * (1 - np.cos(theta)),
                       u[:, [1]] * u[:, [2]] * (1 - np.cos(theta)) - u[:, [0]] * np.sin(theta),
                       t[:, [1]]])
    line_3 = np.stack([u[:, [0]] * u[:, [2]] * (1 - np.cos(theta)) - u[:, [1]] * np.sin(theta),
                       u[:, [1]] * u[:, [2]] * (1 - np.cos(theta)) + u[:, [0]] * np.sin(theta),
                       np.cos(theta) + u[:, [2]] ** 2 * (1 - np.cos(theta)),
                       t[:, [2]]])
    matrix = np.stack([line_1, line_2, line_3])

    return np.transpose(np.squeeze(matrix), [2, 0, 1])


# def Rotate(pointToRotate, point1, point2, theta):
#
#
#     u= []
#     squaredSum = 0
#     for i,f in zip(point1, point2):
#         u.append(f-i)
#         squaredSum += (f-i) **2
#
#     u = [i/squaredSum for i in u]
#
#     r = R(theta, u)
#     rotated = []
#
#     for i in range(3):
#         rotated.append(round(sum([r[j][i] * pointToRotate[j] for j in range(3)])))
#
#     return rotated
#
#
# point = [1,0,0]
# p1 = [0,0,0]
# p2 = [0,1,0]

def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].

    The layer is composed of 3 elements:

    - localisation_net: takes the original image as input and outputs
      the parameters of the affine transformation that should be applied
      to the input image.

    - affine_grid_generator: generates a grid of (x,y) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.

    - bilinear_sampler: takes as input the original image and the grid
      and produces the output image using bilinear interpolation.

    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, H, W, C).

    - theta: affine transform tensor of shape (B, 6). Permits cropping,
      translation and isotropic scaling. Initialize to identity matrix.
      It is the output of the localization network.

    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).

    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)

    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    height = tf.shape(input_fmap)[1]  # y
    width = tf.shape(input_fmap)[2]
    depth = tf.shape(input_fmap)[3]

    # reshape theta to (B, 3, 4)
    theta = tf.reshape(theta, [B, 3, 4])

    # generate grids of same size [B,3,height,width,depth]
    batch_grids = affine_grid_generator(height, width, depth, theta)

    # extract x and y coordinates
    x_s = tf.squeeze(batch_grids[:, 0:1, :, :, :], axis=1)  # [batch_size, height, width, depth]
    y_s = tf.squeeze(batch_grids[:, 1:2, :, :, :], axis=1)  # [batch_size, height, width, depth]
    z_s = tf.squeeze(batch_grids[:, 2:3, :, :, :], axis=1)  # [batch_size, height, width, depth]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s, z_s)
    return out_fmap


def get_pixel_value(img, x, y, z):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, height,width,depth, C)
    - x: flattened tensor of shape (B,height,width,depth )
    - y: flattened tensor of shape (B,height,width,depth )
        x,y是整数索引，代表在img中的位置
    Returns
    -------
    - output: tensor of shape (B, height,width,depth, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = shape[3]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, (1, height, width, depth))

    indices = tf.stack([b, x, y, z], 4)  # [batch_size,W,H,D,4]

    # indices的最后一rank向量(4)用来索引img中的数值[C]，前面的ranks用来定义返回matrix的层次结构[B,H,W,D]
    # 把索引到的数值[C]放入结构[B,H,W],构成[B,H,W,C]
    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, depth, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
      变换后图像的宽和高

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized gird (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    # grab batch size
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, height)
    y = tf.linspace(-1.0, 1.0, width)
    z = tf.linspace(-1.0, 1.0, depth)

    # meshgrid产生的矩阵维度和输入向量的大小是逆序的，越靠前的产生的元素越里层
    x_t, y_t, z_t = tf.meshgrid(x, y, z)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])  # [height*width*depth]
    y_t_flat = tf.reshape(y_t, [-1])
    z_t_flat = tf.reshape(z_t, [-1])

    # reshape to (x_t, y_t , z_t, 1)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_flat, ones])  # [4, height*width*depth]

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)  # [1,4, height*width*depth]
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))  # [num_batch ,4, height*width*depth]

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')  # (num_batch, 3, 4)
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    # 对网格点坐标进行变换
    # [B,3,4] x [B,4,height*width*depth]  ==> [B,3,height*width*depth]
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 3, height*width*depth), 表示在 S 中的坐标

    # reshape to (num_batch, 3, height, width,depth)
    batch_grids = tf.reshape(batch_grids, [num_batch, 3, height, width, depth])
    # batch_grids是变换前图像的对应点(x_s,y_s)在变换后图像的栅格坐标系( -1<x_t<1, -1<y_t<1 )中的坐标
    # 是论文中的sampling_grid，和上面代码中sampling_grid( 对应论文中的regular_grid )意义相反
    return batch_grids


def bilinear_sampler(img, x, y, z):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
      是原图像中对应点在sampling_grids中的坐标  shape=[batch_size,H,W]

    Returns
    -------
    - interpolated images according to grids. Same size as grid.

    """
    # prepare useful params
    B = tf.shape(img)[0]
    height = tf.shape(img)[1]
    width = tf.shape(img)[2]
    depth = tf.shape(img)[3]

    max_y = tf.cast(width - 1, 'int32')
    max_x = tf.cast(height - 1, 'int32')
    max_z = tf.cast(depth - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    z = tf.cast(z, 'float32')

    # rescale x  [0, W] 因为要在 input_map 中采样，因此需要转换的input_map的量纲下
    # rescale y  [0, H] 转换结果不一定是整数
    # W,H是 img 的大小
    x = 0.5 * ((x + 1.0) * tf.cast(height, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(width, 'float32'))
    z = 0.5 * ((z + 1.0) * tf.cast(depth, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    # 有一部分逆向转换得到的原图像的坐标会超出范围，取不到值。
    # 用边缘值取代超出范围的值
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    # get pixel value at corner coords 获得8副图像 [B,height,width,depth]
    Ia = get_pixel_value(img, x0, y0, z0)
    Ib = get_pixel_value(img, x0, y1, z0)
    Ic = get_pixel_value(img, x1, y0, z0)
    Id = get_pixel_value(img, x1, y1, z0)
    Ie = get_pixel_value(img, x0, y0, z1)
    If = get_pixel_value(img, x0, y1, z1)
    Ig = get_pixel_value(img, x1, y0, z1)
    Ih = get_pixel_value(img, x1, y1, z1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    z0 = tf.cast(z0, 'float32')
    z1 = tf.cast(z1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y) * (z1 - z)  # [B,height,width,depth]
    wb = (x1 - x) * (y - y0) * (z1 - z)
    wc = (x - x0) * (y1 - y) * (z1 - z)
    wd = (x - x0) * (y - y0) * (z1 - z)
    we = (x1 - x) * (y1 - y) * (z - z0)
    wf = (x1 - x) * (y - y0) * (z - z0)
    wg = (x - x0) * (y1 - y) * (z - z0)
    wh = (x - x0) * (y - y0) * (z - z0)

    # add dimension for addition
    # wa = tf.expand_dims(wa, axis=4)  # [B,H,W,1],可以和[B,H,W,C]进行element-wise的broadcast
    # wb = tf.expand_dims(wb, axis=4)
    # wc = tf.expand_dims(wc, axis=4)
    # wd = tf.expand_dims(wd, axis=4)
    # we = tf.expand_dims(we, axis=4)  # [B,H,W,1],可以和[B,H,W,C]进行element-wise的broadcast
    # wf = tf.expand_dims(wf, axis=4)
    # wg = tf.expand_dims(wg, axis=4)
    # wh = tf.expand_dims(wh, axis=4)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id,
                    we * Ie, wf * If, wg * Ig, wh * Ih
                    ])  # [B,height,width,depth]
    # 仿射变换后的输出图像
    return out
