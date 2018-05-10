import tensorflow as tf
from tensorflow.contrib import slim
import os
from config_process import SHAPE_BOX, TrainDataConfig, ValiDataConfig, DataConfig, DATA_DICT, get_feature_num, \
    TASK_DICT_F, TASK_DICT
from dataRelated import BatchGenerator
import inception_v3 as icp


class DetectNet(object):
    def __init__(self, is_training, input_box, targets, scope='detector',
                 need_optim=True, clip_grd=True):
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
                                             is_training=is_training, scope='InceptionV3',
                                             depth_multiplier=1.)
                
                with tf.variable_scope('error'):
                    self.error = tf.reduce_mean(tf.reduce_sum(
                        tf.square(self.pred - targets), axis=1) / DataConfig.num_feature_need, axis=0)
            
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
    

    for data_set,config_content in DATA_DICT.items():
        
        
        if config_content['task']=='facc':
            DataConfig.task_dict=TASK_DICT_F
            DataConfig.num_feature_need=get_feature_num(TASK_DICT_F)
        else:
            DataConfig.task_dict=TASK_DICT
            DataConfig.num_feature_need=get_feature_num(TASK_DICT)
            
        DataConfig.output_dim=3*DataConfig.num_feature_need
        DataConfig.data_list=config_content['data_set']
        MODEL_PATH=config_content['model_path']
        
        train_batch_gen = BatchGenerator(TrainDataConfig)
        test_batch_gen = BatchGenerator(ValiDataConfig)
    
    ###############################
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

    ##################################

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # writer = tf.summary.FileWriter('log/', sess.graph)
            
            NEED_RESTORE = False
            NEED_SAVE = True
            
            TOTAL_EPHOC = 10000
            test_step = 3
            need_early_stop = True
            EARLY_STOP_STEP = 600
            total_step=0
            winner_loss = 10 ** 10
            step_from_last_mininum = 0
            start = False
            
            # train_batch_gen=BatchGenerator(TrainDataConfig)
            # test_batch_gen= BatchGenerator(ValiDataConfig)
            
            sess.run(tf.global_variables_initializer())
            
            if NEED_RESTORE:
                assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
                # 文件内容必须大于等于模型内容
                saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量
            
            for iter in range(TOTAL_EPHOC):
                box_batch, y_batch = train_batch_gen.get_batch()
                feed_dict = {input_box: box_batch, targets: y_batch, is_training: True}
                
                _, loss_train = sess.run([detector.train_op, detector.error], feed_dict=feed_dict)
                
                if iter % test_step == 0:
                    if start == False:
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt')
                        start = True
                    if need_early_stop and step_from_last_mininum > EARLY_STOP_STEP:
                        total_step=iter
                        break
                    step_from_last_mininum += 1
                    box_batch, y_batch = test_batch_gen.get_batch()
                    
                    feed_dict = {input_box: box_batch, targets: y_batch, is_training: False}
                    loss_test, pred_get = sess.run([detector.error, detector.pred], feed_dict=feed_dict)
                    if loss_test < winner_loss:
                        winner_loss = loss_test
                        step_from_last_mininum = 0
                        if NEED_SAVE and loss_test < 10000:
                            save_path = saver.save(sess, MODEL_PATH + '\\model.ckpt')
                    # print('\n\n\n')
                    # print(y_batch)
                    # print('#################')
                    # print(pred_get)
                    print("%d  trainCost=%f   testCost=%f   winnerCost=%f   test_step=%d\n"
                          % (iter, loss_train, loss_test, winner_loss, step_from_last_mininum))
                    
        tf.reset_default_graph()
        with open('log/log.txt', 'a') as f:
            f.write('data_set  =%s, error=%f  total_sep=%d\n '%(data_set,winner_loss,total_step))







