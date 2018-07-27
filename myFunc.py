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
    batch, height, width, length, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    reduced_length = length // block_size
    
    y = tf.reshape(x,[-1, reduced_height, block_size,
                         reduced_width, block_size, reduced_length,block_size,depth])
    depth=depth*block_size*block_size*block_size
    z = tf.reshape(tf.transpose(y,[0,1,3,5,2,4,6,7]),[-1, reduced_height, reduced_width,reduced_length,depth])
    return z


if __name__ == '__main__':
    pass
    