import os
import tensorflow as tf
from tensorflow.contrib import slim
from config import MODEL_PATH, SHAPE_BOX, TrainDataConfig, ValiDataConfig, DataConfig, LOSS_WEIGHT
from dataRelated import BatchGenerator
# import inception_v3 as icp
import inception_resnet_v2 as icp
from datetime import datetime
import numpy as np
import math
co_path=os.path.join
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
                print('DataConfig.feature_dim: ',DataConfig.feature_dim)
                self.output=tf.identity(pred, name='output_node')
                
                if need_targets:
                    with tf.variable_scope('error'):
                        self.targets = tf.placeholder(tf.float32, shape=[None, DataConfig.feature_dim],
                                                 name="targets")
                        self.f_mask = tf.placeholder(tf.bool, shape=(None, DataConfig.feature_dim))
                        
                        
                        self.loss_matrix= tf.square(self.output - self.targets)

                        self.eloss_m=self.loss_matrix[:,:6]
                        self.floss_m=self.loss_matrix[:,6:21]
                        self.gloss_m=self.loss_matrix[:,21:]
                        
                        self.emask=self.f_mask[:,:6]
                        self.fmask=self.f_mask[:,6:21]
                        self.gmask=self.f_mask[:,21:]
                        
                        self.eloss=3 * tf.reduce_mean(tf.boolean_mask(self.eloss_m, self.emask))
                        self.floss=3 * tf.reduce_mean(tf.boolean_mask(self.floss_m, self.fmask))
                        self.gloss=3 * tf.reduce_mean(tf.boolean_mask(self.gloss_m, self.gmask))
                        
                        self.equal_loss=3 * tf.reduce_mean(tf.boolean_mask(self.loss_matrix, self.f_mask))

                        self.weight_loss_matrix=self.loss_matrix * LOSS_WEIGHT

           
                        self.weight_loss = 3 * tf.reduce_mean(tf.boolean_mask(self.weight_loss_matrix, self.f_mask))
                    ####################################
                        if is_training_sti:
                            # train_summary = []
                            # train_summary.append(tf.summary.scalar('feature_error', self.equal_loss))
                            # train_summary.append(tf.summary.scalar('edge_error', self.eloss))
                            # train_summary.append(tf.summary.scalar('facc_error', self.floss))
                            # train_summary.append(tf.summary.scalar('groove_error', self.gloss))
                            #
                            # self.train_summary = tf.summary.merge(train_summary)
                            
    
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

def summary(avg_loss):
    train_summary = []
    train_summary.append(tf.summary.scalar('feature_error', avg_loss[0]))
    train_summary.append(tf.summary.scalar('edge_error', avg_loss[1]))
    train_summary.append(tf.summary.scalar('facc_error', avg_loss[2]))
    train_summary.append(tf.summary.scalar('groove_error', avg_loss[3]))
    return tf.summary.merge(train_summary)

if __name__ == '__main__':

    final_error=0

    detector = DetectNet()
    avg_loss_node=tf.placeholder(tf.float32,[4])
    sum_node=summary(avg_loss_node)

    
    train_batch_gen = BatchGenerator(TrainDataConfig)
    test_batch_gen = BatchGenerator(ValiDataConfig)


    # saver = tf.train.Saver(max_to_keep=1)
    ################
    ########## 确认BN 模块中的名字是否如下，如不一致，将不会保存！！！！
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_avg' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_var' in g.name]
    
    ################# important  !!!!!!!!!!!!!!!  dont delete
    # # if some structure changed compared to the saved model, need to load different vars
    # # 不能让 Saver retore .data 中不存在的 变量， 所以要缩减任务
    # load_list = [t for t in tf.trainable_variables() if not
    #              t.name.startswith('detector/Logits')]
    #              # and not t.name.endswith('pred_output/weights:0')]
    #
    # var_list=load_list+bn_moving_vars
    # loader = tf.train.Saver(var_list=var_list, max_to_keep=1)

    ##################
    
    var_list = tf.trainable_variables()+bn_moving_vars
    load_var_list=var_list

    NEED_INIT_SAVE = False
    


    TOTAL_EPHOC = 100000
    test_step = 1
    test_batch_num=10
    save_step=500000
    need_early_stop = False
    EARLY_STOP_STEP = 20
    

    winner_loss = 10 ** 10
    step_from_last_mininum = 0
    start = False

    dir_load= '20180902-1136/'  # where to restore the model
    dir_save= None  # where to save the model

    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        
        loader = tf.train.Saver(var_list=load_var_list)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
        sess.run(tf.global_variables_initializer())
        
        load_checkpoints_dir=None
        if dir_load is not None:
            load_checkpoints_dir= MODEL_PATH + dir_load
        elif dir_save is not  None:
            load_checkpoints_dir= MODEL_PATH + dir_save
            
        if load_checkpoints_dir is not None:
            # var_file = tf.train.latest_checkpoint(load_checkpoints_dir)
            var_file= os.path.join(load_checkpoints_dir,'model.ckpt-7')

            loader.restore(sess, var_file)  # 从模型中恢复最新变量

        if dir_save is None:
            dir_save = datetime.now().strftime("%Y%m%d-%H%M")
            save_checkpoints_dir = MODEL_PATH + dir_save + '/'

            try:
                os.makedirs(save_checkpoints_dir)
            except os.error:
                pass
        else:
            save_checkpoints_dir = MODEL_PATH + dir_save + '/'

        

        writer = tf.summary.FileWriter(save_checkpoints_dir, sess.graph)

        for iter in range(1,TOTAL_EPHOC):
            return_dict = train_batch_gen.get_batch()
            
            # display_batch(box_batch, y_batch, mask_batch)
            feed_dict = {detector.input_box: return_dict['box'],
                         detector.targets: return_dict['y'],
                         detector.f_mask:return_dict['mask'],
                         detector.is_training: True}
            
            # _, train_eloss,train_wloss = sess.run([detector.train_op, detector.equal_loss,detector.weight_loss], feed_dict=feed_dict)
            
            _,outputs,loss_matrix,train_eloss\
                = sess.run([
                            detector.train_op,
                            detector.output,
                           detector.loss_matrix,
                           detector.equal_loss,
                           ], feed_dict=feed_dict)
            # print('edge_loss =%f   facc_loss=%f    gro_loss=%f'%(edge_loss,facc_loss,gro_loss))


            if iter % test_step == 0:
                if NEED_INIT_SAVE and start == False:
                    save_path = saver.save(sess, MODEL_PATH+'model.ckpt',iter)
                    start = True
                if  need_early_stop and step_from_last_mininum>EARLY_STOP_STEP:
                    final_error=winner_loss
                    break
                step_from_last_mininum += 1
                epoch_restart=False

                f_loss_epoch={'edge':0,'facc':0, 'groove':0}
                f_num_epoch={'edge':0,'facc':0, 'groove':0}


                def count_f_num(mask):
                    count_dict = {}
                    count_dict['edge'] = np.sum(mask[:, :6]) / 3
                    count_dict['facc'] = np.sum(mask[:, 6:21]) / 3
                    count_dict['groove'] = np.sum(mask[:, 21:]) / 3
                    return count_dict

                test_batch_iter=0
                while(test_batch_iter<test_batch_num):
                    print('test_batch_iter =  %d '%(test_batch_iter))
                    test_batch_iter+=1
                    return_dict= test_batch_gen.get_batch()
                    epoch_restart=return_dict['epoch_restart']
                    
                    
                    
    
                    feed_dict = {detector.input_box: return_dict['box'],
                                 detector.targets: return_dict['y'],
                                 detector.f_mask: return_dict['mask'],
                                 detector.is_training: False}
                    
                    f_loss_batch = {}
                    f_loss_batch['edge'], f_loss_batch['facc'], f_loss_batch['groove'] \
                        = sess.run(
                                    [
                                    detector.eloss,
                                    detector.floss,
                                    detector.gloss
                                     ],
                        feed_dict=feed_dict)
                    
                    data_count_batch=count_f_num(return_dict['mask'])
                    for k in data_count_batch:
                        if data_count_batch[k]!=0:
                            f_loss_epoch[k]+= f_loss_batch[k] * data_count_batch[k]
                            f_num_epoch[k]+=data_count_batch[k]
                        
                total=sum(f_num_epoch.values())
                # integ_loss=0
                for k in f_loss_epoch:
                    f_loss_epoch[k]=f_loss_epoch[k]/f_num_epoch[k]
                    # integ_loss+=f_loss_epoch[k]*f_num_epoch[k]/total
                integ_loss=(f_loss_epoch['edge']+f_loss_epoch['facc']+f_loss_epoch['groove'])/3.

                avg_loss=[integ_loss]+[v for v in f_loss_epoch.values()]
                feed_dict = {avg_loss_node:np.array(avg_loss)}

                summary=sess.run(sum_node,feed_dict=feed_dict)

                
                writer.add_summary(summary, int(iter / test_step))
                
                if integ_loss < winner_loss:
                    winner_loss = integ_loss
                    step_from_last_mininum = 0
                    
                if iter%save_step==0  :
                    save_path = saver.save(sess, save_checkpoints_dir + 'model.ckpt', int(iter / save_step))

                print("%d  trainCost=%f   integ_loss =%f   winnerCost=%f   test_step=%d  edge_loss =%f   facc_loss=%f    gro_loss=%f\n"
                      % (iter, train_eloss, integ_loss, winner_loss, step_from_last_mininum,
                         f_loss_epoch['edge'],f_loss_epoch['facc'],f_loss_epoch['groove']))
                
                prop_dict = train_batch_gen.get_data_static()
                print('\n##################  train prop')
                for k,v in prop_dict.items():
                    print("%s: %f  "%(k,v))
                
                print('##################  test prop')
                prop_dict = test_batch_gen.get_data_static()
                for k, v in prop_dict.items():
                    print("%s: %f  " % (k, v))
                print('\n')





                    





