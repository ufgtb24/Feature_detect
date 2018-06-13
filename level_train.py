import tensorflow as tf
from tensorflow.contrib import slim
import os
from config import MODEL_PATH, SHAPE_BOX, TrainDataConfig, ValiDataConfig, DataConfig, LOG_PATH
from dataRelated import BatchGenerator
# import inception_v3 as icp
import inception_resnet_v2 as icp

class DetectNet(object):
    def __init__(self, need_targets=True,is_training_sti=True,clip_grd=True,scope='detector'):
        '''
        :param Param:
        :param is_training: place_holder used in running
        :param scope:
        :param input_box: shape=[None] + SHAPE_BOX   placeholder
        :param keep_prob:

        '''

        with tf.variable_scope(scope):
    
            self.input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
            box = tf.to_float(self.input_box)
            
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # cnn = CNN(param=Param, phase=self.phase, keep_prob=self.keep_prob, box=self.box)
            with slim.arg_scope(icp.inception_resnet_v2_arg_scope()):
                # [batch_size,768]
                net = icp.inception_resnet_v2(box, is_training=self.is_training, scope='InceptionRes2')
                

                pred = slim.fully_connected(net, DataConfig.feature_dim+4, activation_fn=None,
                                              scope='Logits')


                self.class_output=tf.to_float(tf.argmax(pred[:,:4], axis=1))
                self.features=pred[:,4:]
                self.total_output=tf.concat([tf.reshape(self.class_output,[-1,1]),self.features],axis=1,name='output_node')
                
                if need_targets:
                    with tf.variable_scope('error'):
                        self.targets = tf.placeholder(tf.float32, shape=[None, DataConfig.feature_dim],
                                                 name="targets")
                        self.f_mask = tf.placeholder(tf.bool, shape=(None, DataConfig.feature_dim))
                        self.labels = tf.placeholder(tf.int32, shape=(None,))

                        Y = tf.one_hot(self.labels, depth=4, axis=1, dtype=tf.float32)
                        self.classify_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred[:, :4]))
                        
                        self.f_output_masked = tf.boolean_mask(self.features, self.f_mask)
                        target_masked = tf.boolean_mask(self.targets, self.f_mask)
                        self.feature_loss = 3 * tf.reduce_mean(tf.square(self.f_output_masked - target_masked))
                        
                        correct_prediction = tf.equal(tf.to_int32(self.labels), tf.to_int32(self.class_output))
                        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        

                    ####################################
                        if is_training_sti:
                            self.total_loss = self.feature_loss + 10*self.classify_loss
                            train_summary = []
                            train_summary.append(tf.summary.scalar('feature_error', self.feature_loss))
                            train_summary.append(tf.summary.scalar('accuracy', self.accuracy))
                            self.train_summary = tf.summary.merge(train_summary)
    
                            with tf.variable_scope('optimizer'):
                                # Ensures that we execute the update_ops before performing the train_step
                                # optimizer = tf.train.AdamOptimizer(0.001,epsilon=1.0)
                                optimizer = tf.train.AdamOptimizer()
                                if clip_grd:
                                    gvs = optimizer.compute_gradients(self.total_loss)
                                    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                                    self.train_op = optimizer.apply_gradients(capped_gvs)
                                else:
                                    self.train_op = optimizer.minimize(self.total_loss)

if __name__ == '__main__':

    final_error=0

    detector = DetectNet()

    train_batch_gen = BatchGenerator(TrainDataConfig)
    test_batch_gen = BatchGenerator(ValiDataConfig)

    # saver = tf.train.Saver(max_to_keep=1)
    ################
    ########## 确认BN 模块中的名字是否如下，如不一致，将不会保存！！！！
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_avg' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_var' in g.name]
    ################# important  !!!!!!!!!!!!!!!  dont delete
    #if some structure changed compared to the saved model, need to load different vars
    # load_list = [t for t in tf.trainable_variables() if not
    #              t.name.endswith('pred_output/biases:0')
    #              and not t.name.endswith('pred_output/weights:0')]
    #
    # var_list=load_list+bn_moving_vars
    # loader = tf.train.Saver(var_list=var_list, max_to_keep=1)

    ##################
    var_list = tf.trainable_variables()+bn_moving_vars

    ################

    NEED_RESTORE = False
    NEED_SAVE = True
    NEED_INIT_SAVE = False

    TOTAL_EPHOC = 100000
    test_step = 5
    save_step = 200
    need_early_stop = True
    EARLY_STOP_STEP = 100

    winner_loss = 10 ** 10
    step_from_last_mininum = 0
    start = False

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        
        writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=EARLY_STOP_STEP)
        sess.run(tf.global_variables_initializer())

        if NEED_RESTORE:
            assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
            #文件内容必须大于等于模型内容

            model_file = tf.train.latest_checkpoint(MODEL_PATH)
            saver.restore(sess, model_file)  # 从模型中恢复最新变量
            # saver.restore(sess, MODEL_PATH+'-6')  # 从模型中恢复指定变量

        for iter in range(TOTAL_EPHOC):
            box_batch ,y_batch, mask_batch,class_batch = train_batch_gen.get_batch()
            
            feed_dict = {detector.input_box: box_batch,
                         detector.targets: y_batch,
                         detector.f_mask:mask_batch,
                         detector.labels:class_batch,
                         detector.is_training: True}

            _, train_feature_loss = sess.run([detector.train_op, detector.feature_loss], feed_dict=feed_dict)


            
            
            if iter % test_step == 0:
                if NEED_INIT_SAVE and start == False:
                    save_path = saver.save(sess, MODEL_PATH+'model.ckpt',iter)
                    start = True
                if  need_early_stop and step_from_last_mininum>EARLY_STOP_STEP:
                    final_error=winner_loss
                    break
                step_from_last_mininum += 1
                
                box_batch, y_batch, mask_batch, class_batch = test_batch_gen.get_batch()


                feed_dict = {detector.input_box: box_batch,
                             detector.targets: y_batch,
                             detector.f_mask: mask_batch,
                             detector.labels: class_batch,
                             detector.is_training: False}

                feature_loss,accuracy,summary = sess.run([detector.feature_loss,
                                                          detector.accuracy,
                                                          detector.train_summary], feed_dict=feed_dict)
                
                if feature_loss < winner_loss:
                    winner_loss = feature_loss
                    step_from_last_mininum = 0
                    
                print("%d  trainCost=%f   feature_loss =%f   winnerCost=%f   test_step=%d          accuracy=%f\n"
                      % (iter, train_feature_loss, feature_loss, winner_loss, step_from_last_mininum,accuracy))

                if iter%save_step==0:
                    print('sample to summary:  test_loss =', feature_loss)
                    writer.add_summary(summary, int(iter / save_step))
                    if NEED_SAVE and feature_loss < 20  :
                        save_path = saver.save(sess, MODEL_PATH + 'model.ckpt', int(iter / save_step))

                    





