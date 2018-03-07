import tensorflow as tf
import os

from config import MODEL_PATH, SHAPE_BOX
from dataRelated import  BatchGenerator
from level_train import Level


class NetConfig(object):
    shape_box=SHAPE_BOX
    channels = [32, 32,  64, 64, 128, 128, 256]  # 决定左侧的参数多少和左侧的memory
    fc_size = [128, 6]
    pooling = [True, True,True, True, True, True, True]
    filter_size=[5,3,3,3,3,3,3] #决定左侧的参数多少
    stride=[1,1,1,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1

class TrainDataConfig(object):
    world_to_cubic=128/12.
    batch_size=4
    # total_case_dir='F:/ProjectData/Feature/Tooth'
    total_case_dir='F:/ProjectData/Feature2/Tooth'
    data_list=None
    load_case_once=10  #每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles=1 #当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'

class TestDataConfig(object):
    world_to_cubic=128/12.
    batch_size=4
    total_case_dir='F:/ProjectData/Feature2/test_mul'
    data_list=None
    load_case_once=0  #每次读的病例数
    switch_after_shuffles=10**10 #当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'

if __name__ == '__main__':
    NEED_RESTORE=True
    NEED_SAVE=True

    keep_prob = tf.placeholder(tf.float32,name='keep_prob_input')
    phase = tf.placeholder(tf.bool,name='phase_input')
    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    box=tf.to_float(input_box)

    level=Level(Param=NetConfig,is_training=True,scope='level_1',input_box=box,
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
        sess.run(tf.global_variables_initializer())



        winner_loss = 10**10
        step_from_last_mininum = 0
        test_step = 5
        average=0
        remember=0.9
        less_100_case=0
        longest_term=0
        start=False
        for toothID in [2,3,4,5]:
            TrainDataConfig.data_list=['tooth'+str(toothID)]
            TestDataConfig.data_list=['tooth'+str(toothID)]
            train_batch_gen = BatchGenerator(TrainDataConfig)
            test_batch_gen = BatchGenerator(TestDataConfig)
            MODEL_PATH = 'F:/ProjectData/Feature2/model_tooth'+str(toothID) + '/level_1/'
            MODEL_BASIC_PATH = 'F:/ProjectData/Feature2/model/level_1/'
            if NEED_RESTORE:
                assert os.path.exists(MODEL_BASIC_PATH + 'checkpoint')  # 判断模型是否存在
                saver.restore(sess, MODEL_BASIC_PATH + 'model.ckpt')  # 存在就从模型中恢复变量

            for iter in range(20000):
                box_batch, y_batch=train_batch_gen.get_batch()
                feed_dict={level.box:box_batch, level.targets:y_batch,
                           phase:True,keep_prob:0.5}
                _,loss_train=sess.run([level.optimizer,level.loss],feed_dict=feed_dict)

                if iter % test_step==0:
                    if start==False:
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')
                        start=True
                    step_from_last_mininum += 1
                    box_batch, y_batch = test_batch_gen.get_batch()
                    feed_dict = {level.box: box_batch, level.targets: y_batch,
                                 phase: False,keep_prob:1}
                    loss_test = sess.run(level.loss, feed_dict=feed_dict)
                    if loss_test<winner_loss:
                        winner_loss=loss_test
                        step_from_last_mininum=0
                        if NEED_SAVE and loss_test<200:
                            save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')

                    print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d\n"
                          % (iter, loss_train, loss_test, winner_loss, step_from_last_mininum))





