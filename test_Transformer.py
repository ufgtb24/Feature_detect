import mayavi.mlab
import numpy as np
from math import pi ,sin, cos
import tensorflow as tf
import Transformer as tr

data = (100, 100, 100)
data = np.zeros(data)
data[40:60, 40:60, 40:60] = 1
data=data[np.newaxis,:]
input=tf.constant(data,dtype=tf.float32)

# theta=np.array([[ -pi/6,  -0.209429,  0.472427,  0.856126, 2.2164,  9.19458,  -4.53156]])
# theta=np.array([[ -pi/6,  -0.209429,  0.472427,  0.856126, 1.2164,  1.19458,  -1.53156]])
# translate<1
theta=np.array([[-pi/4]],dtype=np.float32)
u=np.array([[1,  0, 0]],dtype=np.float32)
t=np.array([[0.1, 0,  0]],dtype=np.float32)
trans_matrix=tr.get_transformer_matrix(theta,u,t)

output=tr.spatial_transformer_network(input,trans_matrix)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_array=sess.run(output)[0]

    xx, yy, zz = np.where(output_array !=0)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=(0, 1, 0),
                         scale_factor=1)

    mayavi.mlab.show()