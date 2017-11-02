import tensorflow as tf
import commen_structure as commen
import os

from CNN import CNN
from crop_data import CropedBatchGenerator
from dataRelated import  BatchGenerator
from level_train import Level


class NetConfig(object):
    shape_box=[32,32,32]
    channels = [32,  32,   64,  64, 128]#决定左侧的参数多少和左侧的memory
    fc_size = [64,3]
    pooling=[True,False,False,True,True,True,True]
    filter_size=[3,3,3,3,3,3,3] #决定左侧的参数多少
    stride=[1,1,1,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1



class TrainDataConfig(object):
    world_to_cubic=128/12.
    batch_size=4
    total_case_dir='F:/ProjectData/Feature/croped/feature_1/'
    load_case_once=10  #每次读的病例数
    switch_after_shuffles=1 #当前数据洗牌n次读取新数据,仅当load_case_once>0时有效

class TestDataConfig(object):
    world_to_cubic=128/12.
    batch_size=1
    total_case_dir='F:/ProjectData/Feature/test_crop/feature_1/'
    load_case_once=0  #每次读的病例数
    switch_after_shuffles=10**10 #当前数据洗牌n次读取新数据,仅当load_case_once>0时有效


if __name__ == '__main__':
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_22'
    NEED_RESTORE=False
    NEED_SAVE=True


    level=Level(Param=NetConfig,is_training=True,scope='level_22')

    # saver = tf.train.Saver(max_to_keep=1)
################
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
################
    train_batch_gen=CropedBatchGenerator(TrainDataConfig)
    test_batch_gen=CropedBatchGenerator(TestDataConfig)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)

        if NEED_RESTORE:
            assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
            saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量
        else:
            #会初始化所有已经声明的变量
            sess.run(tf.global_variables_initializer())

        winner_loss = 10**10
        step_from_last_mininum = 0
        test_step = 50
        average=0
        remember=0.9
        less_100_case=0
        longest_term=0
        start=False

        for iter in range(10**8):
            box_batch, y_batch=train_batch_gen.get_batch()
            feed_dict={level.box:box_batch, level.targets:y_batch,
                       level.phase:True,level.keep_prob:0.5}
            _,loss_train=sess.run([level.optimizer,level.loss],feed_dict=feed_dict)
            if iter % test_step==0:
                if start==False:
                    save_path = saver.save(sess, MODEL_PATH + '\\model.ckpt')
                    start=True
                step_from_last_mininum += 1
                box_batch, y_batch = test_batch_gen.get_batch()
                feed_dict = {level.box: box_batch, level.targets: y_batch,
                             level.phase: False,level.keep_prob:1}
                loss_test = sess.run(level.loss, feed_dict=feed_dict)
                if loss_test<winner_loss:
                    winner_loss=loss_test
                    step_from_last_mininum=0
                    if NEED_SAVE and loss_test<1:
                        save_path = saver.save(sess, MODEL_PATH + '\\model.ckpt')

                print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d\n"
                      % (iter, loss_train, loss_test, winner_loss, step_from_last_mininum))

                # if loss_train<100:
                #     less_100_case+=1
                #     if less_100_case>longest_term:
                #         longest_term=less_100_case
                # else:
                #     less_100_case=0
                #
                # average=average*remember+loss_train*(1-remember)
                #
                #
                #
                #
                # print("%d  trainCost=%f   averageCost=%f   less_100=%d  longest_term=%d\n"
                #       % (iter, loss_train, average,less_100_case,longest_term))




