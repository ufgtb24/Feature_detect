import tensorflow as tf
import numpy as np
# from crop_data import crop_batch
from combine import PB_PATH, gen_frozen_graph, load_graph
from config import MODEL_PATH, TestDataConfig, ValiDataConfig
from dataRelated import BatchGenerator
from display import  display_batch
from level_train import DetectNet
import os
if __name__ == '__main__':

    NEED_INFERENCE=True
    NEED_DISPLAY=False
    NEED_WRITE_GRAPH=False
    NEED_TARGET=True # no need to change
    NEED_PB=False

    if not NEED_PB:
        if NEED_WRITE_GRAPH:
            NEED_TARGET=False
        detector = DetectNet(need_targets=NEED_TARGET,is_training_sti=False)
        # pred_end = tf.to_int32(tf.identity(detector.pred,name="output_node"))
        
        #############
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
    
        task_spec_vars = [g for g in var_list if 'task_spec_conv' in g.name]
        ########## 确认BN 模块中的名字是否如下，如不一致，将不会保存！！！！
        bn_moving_vars = [g for g in g_list if 'moving_avg' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_var' in g.name]
        var_list += bn_moving_vars
    
        #############
        saver = tf.train.Saver(var_list)
        
        
        
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)
        if NEED_PB:
            g=load_graph(sess,"E://TensorFlowCplusplus//feature_detect//x64//Release//up_graph.pb")
            node_list=[n.name for n in tf.get_default_graph().as_graph_def().node]
            pass

        else:

            sess.run(tf.global_variables_initializer())
            
            dir_load = '20180910-1919'  # where to restore the model
            load_checkpoints_dir= MODEL_PATH + dir_load
            # var_file = tf.train.latest_checkpoint(load_checkpoints_dir)
            var_file= os.path.join(load_checkpoints_dir,'model.ckpt-100')
            saver.restore(sess, var_file)  # 从模型中恢复最新变量



            if NEED_WRITE_GRAPH:
                gd = sess.graph.as_graph_def()
    
                for node in gd.node:
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in range(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] = node.input[index] + '/read'
    
                    elif node.op == 'AssignSub':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr: del node.attr['use_locking']
                    elif node.op == 'AssignAdd':
                        node.op = 'Add'
                        if 'use_locking' in node.attr: del node.attr['use_locking']
                #
                # g_list = tf.global_variables()
                # bn_moving_vars = [g.name[:-2] for g in g_list if 'moving_var' in g.name or 'moving_avg' in g.name]
                #
                # output_graph_def = tf.graph_util.convert_variables_to_constants(
                #     sess, gd, ['output_node'],
                #     variable_names_blacklist=bn_moving_vars)
    
    
                # black_list=','.join(bn_moving_vars)
                # serialize the graph_def to a dist file
                tf.train.write_graph(gd, load_checkpoints_dir, PB_PATH, as_text=True)
                # load the serialized file, convert the current graph variables to constants, embed the converted
                # constants into the loaded structure
                gen_frozen_graph(var_file, load_checkpoints_dir)

        if NEED_INFERENCE:
            if NEED_TARGET:
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=True,need_name=True)
                
                # avg=[0]*5
                # def get_avg(i,loss_list):
                #     for i,loss in enumerate(loss_list):
                #         avg[i]=i/(i+1)*avg[i]+1/(i+1)*loss
                
                load_all=False
                r_list=[]
                while (not load_all):
                    target = test_batch_gen.get_batch()
                    load_all=target['epoch_restart']
                    feed_dict = {detector.input_box: target['box'],
                                 detector.targets: target['y'],
                                 detector.f_mask: target['mask'],
                                 detector.is_training: False}
                    
                    f, edge_loss ,facc_loss ,groove_loss , output_mask = \
                        sess.run([detector.output,
                                         detector.eloss,
                                         detector.floss,
                                         detector.gloss,
                                         detector.f_mask,
                                         
                                         ], feed_dict=feed_dict)
                    
                    # get_avg(iter,[edge_loss,facc_loss ,groove_loss ])
                    
                    # print('edge_loss= ', edge_loss,'    facc_loss= ', facc_loss,'   groove_loss= ', groove_loss,
                    #       '    name: ', target['name'][0])

                    if edge_loss>300 or facc_loss>300 or groove_loss>300:
                        # print(target['name'][0],'\n')
                        print('edge_loss = %f,    facc_loss = %f,      groove_loss = %f              %s '
                              %(edge_loss,facc_loss,groove_loss,target['name'][0]))
                        if target['name'] not in r_list:
                            r_list.append(target['name'][0])
                            

                        display_batch(target['box'], f, target['mask'])
                        display_batch(target['box'], target['y'], target['mask'])
                    

    
                    if NEED_DISPLAY:
                        display_batch(target['box'], f, target['mask'])
                
                with open('bad_data.txt',mode='w') as record:
                    for line in r_list:
                        record.write(line+'\n')

                
                # print('avg  ', avg)
            else:
                
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=True, need_name=True)
                while True:
                    target = test_batch_gen.get_batch()

                    if NEED_PB:
                        pred_end = sess.graph.get_tensor_by_name('import/detector/output_node:0')
                        feed_dict = {'import/detector/input_box:0': target['box'], 'import/detector/is_training:0': False}
                    else:
                        feed_dict = {detector.input_box: target['box'], detector.is_training: False}
                        pred_end=detector.output
        
                    f = sess.run(pred_end, feed_dict=feed_dict)
                    # print('feature\n  ',f,'    name: ',name_batch[0])

                    # f=f*12./128.
                    f = np.int32(f)
                    print(f)
                    # result['mask'][:,6:21]=True
                    if NEED_DISPLAY:
                        target['mask']=np.ones([target['box'].shape[0], 27]).astype(bool)
                        display_batch(target['box'], f, target['mask'])

    
        

