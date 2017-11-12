import tensorflow as tf
# init = tf.constant(3,dtype=tf.float32)
# with tf.variable_scope('xxx'):
#     a=tf.get_variable('a',initializer=init)
# with tf.control_dependencies([
#    a.assign(tf.ones([],dtype=tf.float32)*7)
# ]):
#     a=tf.identity(a)
#
# with tf.variable_scope('xxx',reuse=True):
#     b=tf.get_variable('a',initializer=init)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     ai,bi=sess.run([a,b])
#     print(ai,' ',bi)


w = tf.get_variable("w", shape=(), dtype=tf.int32,
                    initializer=tf.constant_initializer(0))

def body(i,x):

    with tf.control_dependencies([tf.assign(w, 0)]):
        update = tf.assign(w, w + 1)

    with tf.control_dependencies([update]):
        x =  tf.identity(w)

    return i+1,x

i_final,x_final = tf.while_loop(lambda i,x: i < 20, body, [0,0])
s = tf.Session()
s.run(tf.global_variables_initializer())


for i in range(5):
    s.run(x_final)
    print(s.run(w))



# w = tf.get_variable("w", shape=(), dtype=tf.int32,
#                       initializer=tf.constant_initializer(0))
#
#
# reset=tf.assign(w,0)
# update = tf.assign(w,w+3)
#
# with tf.control_dependencies([update]):
#     t = tf.identity(w)
#
# with tf.control_dependencies([reset]):
#     x = t+1
#
#
# s = tf.Session()
# s.run(tf.global_variables_initializer())
#
# for i in range(50):
#     print(s.run([x]))
# print(s.run(w))
#
