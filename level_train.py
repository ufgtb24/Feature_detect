import tensorflow as tf
from tensorflow.contrib import slim
import os
from config import MODEL_PATH, SHAPE_BOX, TrainDataConfig, ValiDataConfig, DataConfig
from dataRelated import BatchGenerator
import inception_v3 as icp

train_batch_gen = BatchGenerator(TrainDataConfig)
test_batch_gen = BatchGenerator(ValiDataConfig)


class DetectNet(object):
    def __init__(self, is_training,input_box, targets,scope='detector',
                 need_optim=True,clip_grd=True):
        '''
        :param Param:
        :param is_training: place_holder used in running
        :param scope:
        :param input_box: shape=[None] + SHAPE_BOX   placeholder
        :param keep_prob:

        '''

        with tf.variable_scope(scope):
            # cnn = CNN(param=Param, phase=self.phase, keep_prob=self.keep_prob, box=self.box)
            with slim.arg_scope(icp.inception_v3_arg_scope()):
                self.pred = icp.inception_v3(input_box, output_dim=DataConfig.output_dim,
                                             is_training=is_training,scope='InceptionV3',
                                             depth_multiplier=1.)

                with tf.variable_scope('error'):
                    self.error_f=tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred[:15] - targets[:15]), axis=1) /5, axis=0)
                    self.error_g=tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred[15:] - targets[15:]), axis=1) /2, axis=0)
                    
                    self.error=tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred - targets), axis=1) /DataConfig.num_feature_need, axis=0)

            if need_optim:
                with tf.variable_scope('optimizer'):
                    # Ensures that we execute the update_ops before performing the train_step
                    # optimizer = tf.train.AdamOptimizer(0.001,epsilon=1.0)
                    optimizer = tf.train.AdamOptimizer()
                    if clip_grd:
                        gvs = optimizer.compute_gradients(self.error)
                        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                        self.train_op = optimizer.apply_gradients(capped_gvs)
                    else:
                        self.train_op = optimizer.minimize(self.error)

if __name__ == '__main__':

    final_error=0


    is_training = tf.placeholder(tf.bool, name='is_training')
    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    targets = tf.placeholder(tf.float32, shape=[None, DataConfig.output_dim],
                                  name="targets")

    box = tf.to_float(input_box)

    detector = DetectNet(is_training=is_training, input_box=box, targets=targets)



    # saver = tf.train.Saver(max_to_keep=1)
    ################
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    ########## 确认BN 模块中的名字是否如下，如不一致，将不会保存！！！！
    bn_moving_vars = [g for g in g_list if 'moving_avg' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_var' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

    ################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)

        NEED_RESTORE = False
        NEED_SAVE = False
        
        TOTAL_EPHOC=10000
        test_step = 3
        need_early_stop = True
        EARLY_STOP_STEP=2000

        winner_loss_f=winner_loss_g=10**10
        step_from_last_mininum_f=step_from_last_mininum_g = 0
        start = False

        # train_batch_gen=BatchGenerator(TrainDataConfig)
        # test_batch_gen= BatchGenerator(ValiDataConfig)

        sess.run(tf.global_variables_initializer())

        if NEED_RESTORE:
            assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
            #文件内容必须大于等于模型内容
            saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量

        for iter in range(TOTAL_EPHOC):
            box_batch, y_batch = train_batch_gen.get_batch()
            feed_dict = {input_box: box_batch, targets: y_batch, is_training: True}

            _, loss_train_f,loss_train_g = sess.run([detector.train_op, detector.error_f,detector.error_g], feed_dict=feed_dict)

            if iter % test_step == 0:
                if start == False:
                    save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')
                    start = True
                step_from_last_mininum_f += 1
                step_from_last_mininum_g += 1
                box_batch, y_batch = test_batch_gen.get_batch()

                feed_dict = {input_box: box_batch, targets: y_batch,is_training: False}
                loss_test_f,loss_test_g = sess.run([detector.error_f,detector.error_g], feed_dict=feed_dict)
                if loss_test_f < winner_loss_f:
                    winner_loss_f = loss_test_f
                    step_from_last_mininum_f = 0
                if loss_test_g < winner_loss_g:
                    winner_loss_g = loss_test_g
                    step_from_last_mininum_g = 0
                # print('\n\n\n')
                # print(y_batch)
                # print('#################')
                # print(pred_get)
                print("%d train_f=%f  test_f=%f   winner_f=%f   step_f=%d ||train_g=%f  test_g=%f   winner_g=%f   step_g=%d\n"
                      % (iter, loss_train_f,loss_test_f, winner_loss_f, step_from_last_mininum_f,
                         loss_train_g,loss_test_g, winner_loss_g, step_from_last_mininum_g))






