import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import commen_structure as commen
import os
from config import MODEL_PATH, SHAPE_BOX, TrainDataConfig, ValiDataConfig
from dataRelated import BatchGenerator
import inception_v3 as icp


class DetectNet(object):
    def __init__(self, is_training, phase,scope, input_box, keep_prob, targets):
        '''

        :param Param:
        :param is_training: hard code used in construction
        :param phase: place_holder used in running
        :param scope:
        :param input_box: shape=[None] + SHAPE_BOX   placeholder
        :param keep_prob:
        :param need_target:
        '''


        with tf.variable_scope(scope):
            # cnn = CNN(param=Param, phase=self.phase, keep_prob=self.keep_prob, box=self.box)
            with slim.arg_scope(icp.inception_v3_arg_scope()):
                self.pred = icp.inception_v3(input_box, num_features=6, is_training=phase,dropout_keep_prob=keep_prob)


            if is_training == True:
                with tf.variable_scope('error'):
                    self.error=tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred - targets), axis=1) /2, axis=0)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Ensures that we execute the update_ops before performing the train_step
                    self.optimizer = commen.Optimizer(self.error,initial_learning_rate=0.01,
                                                                 max_global_norm=1.0).optimize_op





if __name__ == '__main__':

    final_error=0


    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
    phase = tf.placeholder(tf.bool, name='phase_input')
    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    targets = tf.placeholder(tf.float32, shape=[None, 6],
                                  name="targets")

    box = tf.to_float(input_box)

    detector = DetectNet(is_training=True, scope='level_1', input_box=box,
                         keep_prob=keep_prob, phase=phase, targets=targets)


    # saver = tf.train.Saver(max_to_keep=1)
    ################
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
    ################

    with tf.Session() as sess:
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
            box_batch=np.expand_dims(box_batch,4)
            feed_dict = {input_box: box_batch, targets: y_batch,
                         phase: True, keep_prob: 0.5}

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
                box_batch = np.expand_dims(box_batch, 4)

                feed_dict = {input_box: box_batch, targets: y_batch,
                             phase: False, keep_prob: 1}
                loss_test = sess.run(detector.error, feed_dict=feed_dict)
                if loss_test < winner_loss:
                    winner_loss = loss_test
                    step_from_last_mininum = 0
                    if NEED_SAVE and loss_test < 50:
                        save_path = saver.save(sess, MODEL_PATH + '\\model.ckpt')

                print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d\n"
                      % (iter, loss_train, loss_test, winner_loss, step_from_last_mininum))






