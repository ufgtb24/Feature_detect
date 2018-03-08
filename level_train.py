import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import commen_structure as commen
import os
from config import MODEL_PATH, SHAPE_BOX, TrainDataConfig, ValiDataConfig
from dataRelated import BatchGenerator
import inception_v3 as icp


class DetectNet(object):
    def __init__(self, is_training,scope, input_box, targets):
        '''

        :param Param:
        :param is_training: place_holder used in running
        :param scope:
        :param input_box: shape=[None] + SHAPE_BOX   placeholder
        :param keep_prob:
        :param need_target:
        '''


        with tf.variable_scope(scope):
            # cnn = CNN(param=Param, phase=self.phase, keep_prob=self.keep_prob, box=self.box)
            with slim.arg_scope(icp.inception_v3_arg_scope()):
                self.pred = icp.inception_v3(input_box, num_features=6, is_training=is_training,dropout_keep_prob=0.5)

                with tf.variable_scope('error'):
                    self.error=tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred - targets), axis=1) /2, axis=0)

                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):
                with tf.variable_scope('optimizer'):
                    # Ensures that we execute the update_ops before performing the train_step
                    self.optimizer = tf.train.AdamOptimizer(0.01).minimize( self.error)



if __name__ == '__main__':

    final_error=0


    is_training = tf.placeholder(tf.bool, name='is_training')
    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    targets = tf.placeholder(tf.float32, shape=[None, 6],
                                  name="targets")

    box = tf.to_float(input_box)

    detector = DetectNet(is_training=is_training, scope='detector', input_box=box, targets=targets)



    # saver = tf.train.Saver(max_to_keep=1)
    ################
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

    # get_var_list = tf.get_collection(key='moving_vars', scope='detector/InceptionV3/Conv2d_1')
    e1_params = [t for t in tf.get_collection(key='moving_vars') if
                 t.name.startswith('detector/InceptionV3/Conv2d_1') and
                  t.name.endswith('moving_variance:0')]
    # e1_params = tf.get_collection(key='moving_vars')
    # get_c = tf.get_default_graph().get_all_collection_keys()


    def print_moving_vars(sess):

        print(sess.run(e1_params))


    ################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)

        NEED_RESTORE = False
        NEED_SAVE = True
        test_step = 10
        average = 0
        remember = 0.9
        less_100_case = 0
        longest_term = 0
        start = False
        need_early_stop = True
        EARLY_STOP_STEP=2000

        winner_loss=10**10
        step_from_last_mininum = 0

        train_batch_gen=BatchGenerator(TrainDataConfig, name='_train')
        test_batch_gen= BatchGenerator(ValiDataConfig, name='_test')

        sess.run(tf.global_variables_initializer())

        if NEED_RESTORE:
            assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
            saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量

        # loss_last=2>>31
        case_name=' '
        for iter in range(100000):
            box_batch, y_batch = train_batch_gen.get_batch()
            feed_dict = {input_box: box_batch, targets: y_batch, is_training: True}

            _, loss_train = sess.run([detector.optimizer, detector.error], feed_dict=feed_dict)

            if iter % test_step == 0:
                if start == False:
                    save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')
                    start = True
                if  need_early_stop and step_from_last_mininum>EARLY_STOP_STEP:
                    final_error=winner_loss
                    break
                step_from_last_mininum += 1
                box_batch, y_batch = test_batch_gen.get_batch()

                feed_dict = {input_box: box_batch, targets: y_batch,is_training: False}
                loss_test = sess.run(detector.error, feed_dict=feed_dict)
                if loss_test < winner_loss:
                    winner_loss = loss_test
                    step_from_last_mininum = 0
                    if NEED_SAVE and loss_test < 50:
                        save_path = saver.save(sess, MODEL_PATH + '\\model.ckpt')

                # print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d\n"
                #       % (iter, loss_train, loss_test, winner_loss, step_from_last_mininum))
                print_moving_vars(sess)






