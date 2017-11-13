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


# w = tf.get_variable("w", shape=(), dtype=tf.int32,
#                     initializer=tf.constant_initializer(0))
#
# update_1 = tf.assign(w, w + 1)
# update_2 = tf.assign(w, w + 1)
# update_3 = tf.assign(w, w + 1)
#
#
# def body(i,op_in):
#
#     # with tf.control_dependencies([update_3]):
#     #     op1 =  tf.identity(op_in)
#     #
#     # with tf.control_dependencies([update_2]):
#     #     op2 =  tf.identity(op1)
#
#     with tf.control_dependencies([update_1]):
#         op_out =  tf.identity(w)
#
#     return tf.add(i,1),op_out
#
# i_final,x_final = tf.while_loop(lambda i,x: i < 5, body, [0,0])
# s = tf.Session()
# s.run(tf.global_variables_initializer())
#
# s.run(x_final)
# print(s.run(w))

# for i in range(5):
#     xv=s.run(x_final)
#     print(xv)
#     print(s.run(w))



w = tf.get_variable("w", shape=(), dtype=tf.int32,
                      initializer=tf.zeros_initializer)

reset=tf.assign(w,0)
update1 = tf.assign(w,w+2)
update = tf.assign(w,w+1)

# with tf.control_dependencies([update1]):
#     t = tf.identity(w)
with tf.control_dependencies([update,update1]):
    t = tf.identity(w)
    x = tf.identity(t)


s = tf.Session()
s.run(tf.global_variables_initializer())

for i in range(50):
    s.run(x)
print('w= ', s.run(w))

