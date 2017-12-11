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

#在construct过程会根据关系展开成图，如果有首尾连接就会展开成串行图
#没有首尾连接关系，则展开成并行计算的图
def body(i,x):

    # reset = tf.assign(w, 0)
    # reset = tf.Print(reset, [reset], 'reset')

    #因为tf.shape(x)是确定的，因此会在构造阶段赋值成常数，运行是并没有首尾连接
    update_0 = tf.assign(w, x-x)
    update_0 = tf.Print(update_0, [update_0], 'ud_0')
    # 这是一条依赖链
    with tf.control_dependencies([update_0]):
        update_1 = tf.assign(w, w + 1)
        update_1 = tf.Print(update_1, [update_1], 'ud_1')

        with tf.control_dependencies([update_1]):
            update_2 = tf.assign(w, w + 1)
            update_2 = tf.Print(update_2, [update_2], 'ud_2')
            with tf.control_dependencies([update_2]):
                x = w.read_value()

    return tf.add(i,1),x

i_final,x_final = tf.while_loop(lambda i,x: i <10, body, [0,0])
s = tf.Session()
s.run(tf.global_variables_initializer())
s.run(x_final)
print(s.run(w))




# w = tf.get_variable("w", dtype=tf.int32,
#                       initializer=0)
#
# reset = tf.assign(w,0)
# reset = tf.Print(reset, [reset], 'reset')
#
# update_0 = tf.assign(w, w + 1)
# update_0 = tf.Print(update_0, [update_0], 'ud_0')
#
# with tf.control_dependencies([update_0]):
#     update_1 = tf.assign(w, w + 1)
#     update_1 = tf.Print(update_1, [update_1],'ud_1')
#
# with tf.control_dependencies([update_1]):
#     update_2 = tf.assign(w, w + 1)
#     update_2 = tf.Print(update_2, [update_2],'ud_2')
#
# with tf.control_dependencies([update_2]):
#     # x=w.read_value()
#     x=tf.identity(w)
#
#
#
# s = tf.Session()
# s.run(tf.global_variables_initializer())
#
# for i in range(50):
#     s.run(x)
#     print('w= ', s.run(w))

