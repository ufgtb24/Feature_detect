import tensorflow as tf
# def space_to_depth(x, block_size):
#     x = np.asarray(x)
#     batch, height, width, depth = x.shape
#     reduced_height = height // block_size
#     reduced_width = width // block_size
#     y = x.reshape(batch, reduced_height, block_size,
#                          reduced_width, block_size, depth)
#     z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
#     return z
#
# def space_to_depth_3d(x, block_size):
#     x = np.asarray(x)
#     batch, height, width, length, depth = x.shape
#     reduced_height = height // block_size
#     reduced_width = width // block_size
#     reduced_length = length // block_size
#     y = x.reshape(batch, reduced_height, block_size,
#                          reduced_width, block_size, reduced_length,block_size,depth)
#
#     z = np.transpose(y,[0,1,3,5,2,4,6,7]).reshape(batch, reduced_height, reduced_width,reduced_length, -1)
#     return z

def space_to_depth(x, block_size):
    batch, depth, height, width, length = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    reduced_length = length // block_size
    
    y = tf.reshape(x,[-1, depth,reduced_height, block_size,
                         reduced_width, block_size, reduced_length,block_size])
    depth=depth*block_size*block_size*block_size
    z = tf.reshape(tf.transpose(y,[0,1,2,4,6,3,5,7]),[-1, depth,reduced_height, reduced_width,reduced_length])
    return z


if __name__ == '__main__':
    pass
    