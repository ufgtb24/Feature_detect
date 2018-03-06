import tensorflow as tf
from tensorflow.contrib import slim

import commen_structure as commen
import os
import numpy as np
from CNN import CNN
from config import MODEL_PATH, SHAPE_BOX, TASK_DICT, NetConfig, \
    TrainDataConfig, ValiDataConfig,TRAIL_DETAIL
from dataRelated import BatchGenerator
import inception_v3 as icp


class Level(object):
    def __init__(self, Param, is_training, scope, input_box, keep_prob, phase, need_target=True):
        '''

        :param Param:
        :param is_training:
        :param scope:
        :param input_box: shape=[len(TASK_DICT),None] + SHAPE_BOX
        :param keep_prob:
        :param phase:
        :param need_target:
        '''
        self.box = input_box

        self.keep_prob = keep_prob
        self.phase = phase

        with tf.variable_scope(scope):
            # cnn = CNN(param=Param, phase=self.phase, keep_prob=self.keep_prob, box=self.box)
            with slim.arg_scope(icp.inception_v3_arg_scope()):
                self.pred = icp.inception_v3(input_box, num_features=6, is_training=phase)
                self.optimizers = {}

            if need_target:
                self.targets= tf.placeholder(tf.float32, shape=[None, Param.output_size],
                                                    name="multi_task_target")
                with tf.variable_scope('error'):
                    self.error=tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred - self.targets), axis=1) /2, axis=0)

            if is_training == True:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Ensures that we execute the update_ops before performing the train_step
                    self.optimizer = commen.Optimizer(self.losses,initial_learning_rate=0.01,
                                                                 max_global_norm=1.0).optimize_op



def getRandomTask():
    task_num = len(TASK_DICT)
    x = np.random.rand() * task_num
    key_list = list(TASK_DICT.keys())
    for i in range(task_num):
        if x < i + 1:
            return key_list[i]


if __name__ == '__main__':

    for i in range(len(TRAIL_DETAIL)):
        for content in TASK_DICT.values():
            content['fc_size']=TRAIL_DETAIL[i]['FC_SIZE']

        NetConfig.task_dict=TASK_DICT
        NetConfig.task_layer_num=TRAIL_DETAIL[i]['task_layer_num']
        NetConfig.regularization_term=TRAIL_DETAIL[i]['regularization_term']
        final_error=0


        keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
        phase = tf.placeholder(tf.bool, name='phase_input')
        input_box = tf.placeholder(tf.uint8, shape=[len(TASK_DICT),None] + SHAPE_BOX, name='input_box')
        box = tf.to_float(input_box)
        inception_v3()

        level = Level(Param=NetConfig, is_training=True, scope='level_1', input_box=box,
                      keep_prob=keep_prob, phase=phase)


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
            train_batch_gen = {}
            test_batch_gen = {}
            for task, task_content in NetConfig.task_dict.items():
                TrainDataConfig.data_list = task_content['input_tooth']
                ValiDataConfig.data_list = TrainDataConfig.data_list
                train_batch_gen[task] = BatchGenerator(TrainDataConfig, name='_train')
                test_batch_gen[task] = BatchGenerator(ValiDataConfig, name='_test')

            sess.run(tf.global_variables_initializer())

            if NEED_RESTORE:
                assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
                saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量

            # loss_last=2>>31
            case_name=' '
            for iter in range(100000):
                #[task:[box_batch,y_batch]]
                task_box_and_y_batch_list=[train_batch_gen[task].get_batch() for task in NetConfig.task_dict.keys()]
                # [box_batch_shape]*task_num, [y_batch_shape]*task_num
                box_task_list,y_task_list=zip(*task_box_and_y_batch_list)
                box_task_batch=np.stack(box_task_list)
                y_batch=np.concatenate(y_task_list,axis=1)
                feed_dict = {input_box: box_task_batch, level.targets: y_batch,
                             phase: True, keep_prob: 1}
                _, loss_train = sess.run([level.optimizer, level.error], feed_dict=feed_dict)
                # if loss_train-8000>loss_last:
                #     with open('log/log.txt', 'a') as f:
                #         f.write('error_case= %s \n ' % (case_name))
                # loss_last=loss_train

                if iter % test_step == 0:
                    if start == False:
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')
                        start = True
                    if  need_early_stop and step_from_last_mininum>EARLY_STOP_STEP:
                        final_error=winner_loss
                        break
                    step_from_last_mininum += 1
                    # [task:[box_batch,y_batch]]
                    task_box_and_y_batch_list = [test_batch_gen[task].get_batch() for task in NetConfig.task_dict.keys()]
                    # [box_batch_shape]*task_num, [y_batch_shape]*task_num
                    box_task_list, y_task_list = zip(*task_box_and_y_batch_list)
                    box_task_batch = np.stack(box_task_list)
                    y_batch = np.concatenate(y_task_list, axis=1)

                    feed_dict = {input_box: box_task_batch, level.targets: y_batch,
                                 phase: False, keep_prob: 1}
                    loss_test,reg_loss = sess.run([level.error,level.reg_term], feed_dict=feed_dict)
                    if loss_test < winner_loss:
                        winner_loss = loss_test
                        step_from_last_mininum = 0
                        if NEED_SAVE and loss_test < 100:
                            save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')

                    print("%d  trainCost=%f   testCost=%f   winnerCost=%f   reg_loss=%f   test_step=%d\n "
                          % (iter, loss_train, loss_test, winner_loss, reg_loss,step_from_last_mininum))


        tf.reset_default_graph()






