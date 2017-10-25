import tensorflow as tf
import commen_structure as commen
import os

from CNN import CNN
from dataRelated import DataManager


class Level(object):
    def __init__(self,Param,is_training,scope):
        self.box = tf.placeholder(tf.float32, shape=[None]+ Param.shape_box+[1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32, [])

        self.targets = tf.placeholder(tf.float32, shape=[None, 8])
        self.phase = tf.placeholder(tf.bool, name='phase')
        with tf.variable_scope(scope):
            self.pred = CNN(param=Param, phase=self.phase,keep_prob=self.keep_prob, box=self.box).output
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets))

        if is_training==True:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.optimizer = commen.Optimizer(self.loss, initial_learning_rate=0.02,
                                             max_global_norm=1.0).optimize_op

class NetConfig(object):
    shape_box=[128,128,128]
    channels = [1, 40, 60, 80, 100]
    layer_num = len(channels) - 1
    fc_size = [512, 8]


class DataConfig(object):
    shape_box=[128,128,128]
    shape_crop=[64,64,64]
    world_to_cubic=128/20.
    batch_size_train = 2
    batch_size_test = 1
    need_Save = False
    need_Restore = False
    format = 'mhd'


if __name__ == '__main__':
    ROOT_PATH = 'F:/ProjectData/Feature'
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1'



    with tf.variable_scope('Level_1'):
        level=Level(Param=NetConfig,is_training=True,scope='level_1')

    saver = tf.train.Saver()

    dataManager =DataManager(DataConfig(),
                train_box_path=os.path.join(ROOT_PATH,'train'),
                train_info_file=os.path.join(ROOT_PATH,'train','info.txt'),
                test_box_path=os.path.join(ROOT_PATH,'test'),
                test_info_file=os.path.join(ROOT_PATH,'test','info.txt')
                )

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('log/', sess.graph)

        if DataConfig.need_Restore:
            assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
            saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量
        else:
            #会初始化所有已经声明的变量
            sess.run(tf.global_variables_initializer())

        winner_loss = 10 ** 8
        step_from_last_mininum = 0
        test_step = 300

        for iter in range(10**8):
            box_batch, pos_batch, y_batch=dataManager.getTrainBatch()
            feed_dict={level.box:box_batch, level.targets:y_batch,
                       level.phase:True,level.keep_prob:0.5}
            _,loss_train=sess.run([level.optimizer,level.loss],feed_dict=feed_dict)
            if iter % test_step==0:
                step_from_last_mininum += 1
                box_batch, pos_batch, y_batch = dataManager.getTestBatch()
                feed_dict = {level.box: box_batch, level.targets: y_batch,
                             level.phase: False,level.keep_prob:1}
                loss_test = sess.run(level.loss, feed_dict=feed_dict)
                if loss_test<winner_loss:
                    winner_loss=loss_test
                    step_from_last_mininum=0
                    if DataConfig.need_Save:
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')

                print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d\n"
                      % (iter, loss_train, loss_test, winner_loss, step_from_last_mininum))




