import tensorflow as tf
import commen_structure as commen
import os

from CNN import CNN
from dataRelated import  BatchGenerator


class Level(object):
    def __init__(self,Param,is_training,scope):
        self.box = tf.placeholder(tf.float32, shape=[None]+ Param.shape_box+[1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32, [])

        self.targets = tf.placeholder(tf.float32, shape=[None, 6])
        self.phase = tf.placeholder(tf.bool, name='phase')
        with tf.variable_scope(scope):
            self.pred = CNN(param=Param, phase=self.phase,keep_prob=self.keep_prob, box=self.box).output
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.pred - self.targets),axis=1)/2.)

        if is_training==True:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.optimizer = commen.Optimizer(self.loss, initial_learning_rate=0.01,
                                             max_global_norm=1.0).optimize_op

class NetConfig(object):
    shape_box=[128,128,128]
    channels = [64,  128,   128,  256,]#决定左侧的参数多少和左侧的memory
    pooling=[False,True,True,True]
    filter_size=[5,3,3,3,3] #决定左侧的参数多少
    stride=[2,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1
    fc_size = [512, 128, 6]



class TrainDataConfig(object):
    shape_box=[128,128,128]
    shape_crop=[64,64,64]
    world_to_cubic=128/12.
    batch_size=4
    total_case_dir='F:/ProjectData/Feature/Tooth'
    load_case_once=10  #每次读的病例数
    switch_after_shuffles=1 #当前数据洗牌n次读取新数据
    format = 'mhd'

class TestDataConfig(object):
    shape_box=[128,128,128]
    shape_crop=[64,64,64]
    world_to_cubic=128/12.
    batch_size=1
    total_case_dir='F:/ProjectData/Feature/test'
    load_case_once=1  #每次读的病例数
    switch_after_shuffles=1000000 #当前数据洗牌n次读取新数据
    format = 'mhd'


if __name__ == '__main__':
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1'
    NEED_RESTORE=False
    NEED_SAVE=False



    with tf.variable_scope('Level_1'):
        level=Level(Param=NetConfig,is_training=True,scope='level_1')

    saver = tf.train.Saver()

    train_batch_gen=BatchGenerator(TrainDataConfig)
    test_batch_gen=BatchGenerator(TestDataConfig)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)

        if NEED_RESTORE:
            assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
            saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量
        else:
            #会初始化所有已经声明的变量
            sess.run(tf.global_variables_initializer())

        winner_loss = 10 ** 8
        step_from_last_mininum = 0
        test_step = 10
        average=0
        remember=0.9
        less_100_case=0
        longest_term=0

        for iter in range(10**8):
            box_batch, y_batch=train_batch_gen.get_batch()
            feed_dict={level.box:box_batch, level.targets:y_batch,
                       level.phase:True,level.keep_prob:0.5}
            _,loss_train=sess.run([level.optimizer,level.loss],feed_dict=feed_dict)
            if iter % test_step==0:
                step_from_last_mininum += 1
                box_batch, y_batch = test_batch_gen.get_batch()
                feed_dict = {level.box: box_batch, level.targets: y_batch,
                             level.phase: False,level.keep_prob:1}
                loss_test = sess.run(level.loss, feed_dict=feed_dict)
                if loss_test<winner_loss:
                    winner_loss=loss_test
                    step_from_last_mininum=0
                    if NEED_SAVE:
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')

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




