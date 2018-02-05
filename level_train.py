import tensorflow as tf
import commen_structure as commen
import os
import numpy as np
from CNN import CNN
from config import MODEL_PATH, SHAPE_BOX, DATA_LIST
from dataRelated import BatchGenerator


class Level(object):
    def __init__(self, Param, is_training, scope, input_box, keep_prob, phase, need_target=True):
        self.box = input_box

        self.keep_prob = keep_prob
        self.phase = phase

        with tf.variable_scope(scope):
            self.pred = CNN(param=Param, phase=self.phase, keep_prob=self.keep_prob, box=self.box).output
            self.optimizers = {}

            if need_target:
                self.targets = {}
                self.losses = {}
                for task, task_content in Param.task_dict.items():
                    self.targets[task] = tf.placeholder(tf.float32, shape=[None, task_content['fc_size'][-1]],
                                                        name="Place_holder" + task)
                    self.losses[task] = tf.reduce_mean(
                        tf.reduce_sum(tf.square(self.pred[task] - self.targets[task]), axis=1) / 2., axis=0)
            if is_training == True:
                self.optimizers = {}
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                for task in Param.task_dict.keys():
                    with tf.control_dependencies(update_ops):
                        # Ensures that we execute the update_ops before performing the train_step
                        self.optimizers[task] = commen.Optimizer(self.losses[task], name=task,initial_learning_rate=0.01,
                                                                 max_global_norm=1.0).optimize_op


class NetConfig(object):
    shape_box = SHAPE_BOX
    channels = [32, 32, 64, 64, 128, 128, 256]  # 决定左侧的参数多少和左侧的memory
    task_dict = {'left':
                     {
                      'input_tooth':['tooth2', 'tooth3', 'tooth4', 'tooth5'],
                      'fc_size':[128, 6]
                      },
                 'right':
                     {
                      'input_tooth':['tooth12', 'tooth13', 'tooth14', 'tooth15'],
                      'fc_size':[128, 6]
                      }
                 }

    pooling = [True, True, True, True, True, True, True]
    filter_size = [5, 3, 3, 3, 3, 3, 3]  # 决定左侧的参数多少
    stride = [1, 1, 1, 1, 1, 1, 1]  # 决定右侧的memory
    layer_num = len(channels) - 1
    task_layer_num = 2


class TrainDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 4
    # total_case_dir='F:/ProjectData/Feature/Tooth'
    total_case_dir = 'F:/ProjectData/Feature2/Tooth_test/Tooth'
    data_list=None
    load_case_once = 10  # 每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'


class TestDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 4
    total_case_dir = 'F:/ProjectData/Feature2/test_mul'
    data_list=None
    load_case_once = 0  # 每次读的病例数
    switch_after_shuffles = 10 ** 10  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'


if __name__ == '__main__':
    NEED_RESTORE = False
    NEED_SAVE = True
    MODEL_PATH = MODEL_PATH + 'level_1/'

    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
    phase = tf.placeholder(tf.bool, name='phase_input')
    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    box = tf.to_float(input_box)

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

        writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())

        if NEED_RESTORE:
            assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
            saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量

        winner_loss = {'left':10 ** 10,'right':10 ** 10}
        step_from_last_mininum = {'left':0,'right':0}
        test_step = 5
        average = 0
        remember = 0.9
        less_100_case = 0
        longest_term = 0
        start = False

        train_batch_gen = {}
        test_batch_gen = {}
        for task, task_content in NetConfig.task_dict.items():
            TrainDataConfig.data_list = task_content['input_tooth']
            TestDataConfig.data_list = TrainDataConfig.data_list
            train_batch_gen[task] = BatchGenerator(TrainDataConfig)
            test_batch_gen[task] = BatchGenerator(TestDataConfig)

        for iter in range(20000):

            if np.random.rand() < 0.5:
                task='left'
            else:
                task='right'

            box_batch, y_batch = train_batch_gen[task].get_batch()
            feed_dict = {level.box: box_batch, level.targets[task]: y_batch,
                         phase: True, keep_prob: 0.5}
            _, loss_train = sess.run([level.optimizers[task], level.losses[task]], feed_dict=feed_dict)
            if iter % test_step == 0:
                if start == False:
                    save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')
                    start = True
                step_from_last_mininum[task] += 1
                box_batch, y_batch = test_batch_gen[task].get_batch()
                feed_dict = {level.box: box_batch, level.targets[task]: y_batch,
                             phase: False, keep_prob: 1}
                loss_test = sess.run(level.losses[task], feed_dict=feed_dict)
                if loss_test < winner_loss[task]:
                    winner_loss[task] = loss_test
                    step_from_last_mininum[task] = 0
                    if NEED_SAVE and loss_test < 200:
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')

                print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d  task=%s\n "
                      % (iter, loss_train, loss_test, winner_loss[task], step_from_last_mininum[task],task))





