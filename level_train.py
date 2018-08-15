import tensorflow as tf
from tensorflow.contrib import slim
from config import MODEL_PATH, SHAPE_BOX, TrainDataConfig, ValiDataConfig, DataConfig, LOG_PATH, MODEL_NAME, LOSS_WEIGHT
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
                

                pred = slim.fully_connected(net, DataConfig.feature_dim, activation_fn=None,
                                              scope='Logits')

                self.output=tf.identity(pred, name='output_node')
                
                if need_targets:
                    with tf.variable_scope('error'):
                        self.targets = tf.placeholder(tf.float32, shape=[None, DataConfig.feature_dim],
                                                 name="targets")
                        self.f_mask = tf.placeholder(tf.bool, shape=(None, DataConfig.feature_dim))
                        
                        
                        self.loss_matrix= tf.square(self.output - self.targets)
                        self.weight_loss_matrix=self.loss_matrix * LOSS_WEIGHT
                        self.weight_loss = 3 * tf.reduce_mean(tf.boolean_mask(self.weight_loss_matrix, self.f_mask))
                        self.equal_loss=3 * tf.reduce_mean(tf.boolean_mask(self.loss_matrix, self.f_mask))
                    ####################################
                        if is_training_sti:
                            train_summary = []
                            train_summary.append(tf.summary.scalar('feature_error', self.weight_loss))
                            self.train_summary = tf.summary.merge(train_summary)
    
                            with tf.variable_scope('optimizer'):
                                # Ensures that we execute the update_ops before performing the train_step
                                # optimizer = tf.train.AdamOptimizer(0.001,epsilon=1.0)
                                optimizer = tf.train.AdamOptimizer()
                                if clip_grd:
                                    gvs = optimizer.compute_gradients(self.weight_loss)
                                    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                                    self.train_op = optimizer.apply_gradients(capped_gvs)
                                else:
                                    self.train_op = optimizer.minimize(self.weight_loss)

if __name__ == '__main__':

    final_error=0
    train_batch_gen = BatchGenerator(TrainDataConfig)
    test_batch_gen = BatchGenerator(ValiDataConfig)

    detector = DetectNet()


    # saver = tf.train.Saver(max_to_keep=1)
    ################
    ########## 确认BN 模块中的名字是否如下，如不一致，将不会保存！！！！
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_avg' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_var' in g.name]
    
    ################# important  !!!!!!!!!!!!!!!  dont delete
    # if some structure changed compared to the saved model, need to load different vars
    # 不能让 Saver retore .data 中不存在的 变量， 所以要缩减任务
    load_list = [t for t in tf.trainable_variables() if not
                 t.name.startswith('detector/Logits')]
                 # and not t.name.endswith('pred_output/weights:0')]

    var_list=load_list+bn_moving_vars
    loader = tf.train.Saver(var_list=var_list, max_to_keep=1)

    ##################
    
    var_list = tf.trainable_variables()+bn_moving_vars

    NEED_RESTORE = False
    NEED_SAVE = True
    NEED_INIT_SAVE = False

    TOTAL_EPHOC = 100000
    test_step = 10
    save_step=1000
    need_early_stop = False
    EARLY_STOP_STEP = 100

    winner_loss = 10 ** 10
    step_from_last_mininum = 0
    start = False

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        
        writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        if NEED_RESTORE:
            # assert os.path.exists(MODEL_PATH + 'checkpoint')  # 判断模型是否存在
            #文件内容必须大于等于模型内容

            # model_file = tf.train.latest_checkpoint(MODEL_PATH)
            # saver.restore(sess, model_file)  # 从模型中恢复最新变量
            loader.restore(sess, MODEL_PATH+MODEL_NAME)  # 从模型中恢复指定变量

        for iter in range(TOTAL_EPHOC):
            box_batch ,y_batch, mask_batch = train_batch_gen.get_batch()
            feed_dict = {detector.input_box: box_batch,
                         detector.targets: y_batch,
                         detector.f_mask:mask_batch,
                         detector.is_training: True}

            _, train_eloss,train_wloss = sess.run([detector.train_op, detector.equal_loss,detector.weight_loss], feed_dict=feed_dict)
            
            # _,outputs,targets,f_mask,loss_matrix,weight_loss_matrix,wloss,eloss \
            #     = sess.run([
            #     detector.train_op,
            #                 detector.output,
            #                detector.targets,
            #                detector.f_mask,
            #                detector.loss_matrix,
            #                detector.weight_loss_matrix,
            #                detector.weight_loss,
            #                detector.equal_loss
            #                ], feed_dict=feed_dict)

            
            
            if iter % test_step == 0:
                if NEED_INIT_SAVE and start == False:
                    save_path = saver.save(sess, MODEL_PATH+'model.ckpt',iter)
                    start = True
                if  need_early_stop and step_from_last_mininum>EARLY_STOP_STEP:
                    final_error=winner_loss
                    break
                step_from_last_mininum += 1
                box_batch, y_batch, mask_batch = test_batch_gen.get_batch()


                feed_dict = {detector.input_box: box_batch,
                             detector.targets: y_batch,
                             detector.f_mask: mask_batch,
                             detector.is_training: False}

                test_eloss, test_wloss,summary = sess.run([detector.equal_loss,
                                                           detector.weight_loss,
                                                detector.train_summary], feed_dict=feed_dict)
                
                if test_eloss < winner_loss:
                    winner_loss = test_eloss
                    step_from_last_mininum = 0
                    
                writer.add_summary(summary, int(iter / test_step))
                if NEED_SAVE and iter%save_step==0  :
                    save_path = saver.save(sess, MODEL_PATH + 'model.ckpt', int(iter / save_step))

                print("%d  trainCost=%f   test_loss =%f   winnerCost=%f   test_step=%d    train_wloss=%f    test_wloss=%f\n"
                      % (iter, train_eloss, test_eloss, winner_loss, step_from_last_mininum,train_wloss,test_wloss))
                
                prop_dict = train_batch_gen.get_data_static()
                for k,v in prop_dict.items():
                    print("%s: %f  "%(k,v))
                print('\n')




                    





