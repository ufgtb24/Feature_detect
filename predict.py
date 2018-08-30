import tensorflow as tf
import numpy as np
# from crop_data import crop_batch
from combine import PB_PATH, gen_frozen_graph, load_graph
from config import MODEL_PATH, SHAPE_BOX, TestDataConfig, DataConfig
from dataRelated import BatchGenerator
from display import  display_batch
from level_train import DetectNet


if __name__ == '__main__':

    NEED_INFERENCE=False
    NEED_DISPLAY=False
    NEED_WRITE_GRAPH=True
    NEED_TARGET=False # no need to change
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
            model_file = tf.train.latest_checkpoint(MODEL_PATH)
            saver.restore(sess, model_file)  # 从模型中恢复最新变量
            # saver.restore(sess, MODEL_PATH + MODEL_NAME)

            # saver.restore(sess, MODEL_PATH+MODEL_NAME)  # 存在就从模型中恢复变量

            
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
                tf.train.write_graph(gd, MODEL_PATH, PB_PATH, as_text=True)
                # load the serialized file, convert the current graph variables to constants, embed the converted
                # constants into the loaded structure
                gen_frozen_graph(model_file)

        if NEED_INFERENCE:
            if NEED_TARGET:
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=True,need_name=True)
                
                avg=[0]*5
                def get_avg(i,loss_list):
                    for i,loss in enumerate(loss_list):
                        avg[i]=i/(i+1)*avg[i]+1/(i+1)*loss
                
                for iter in range(500):
                    box_batch, y_batch, mask_batch, name_batch = test_batch_gen.get_batch()
                    
                    feed_dict = {detector.input_box: box_batch,
                                 detector.targets: y_batch,
                                 detector.f_mask: mask_batch,
                                 detector.is_training: False}
                    
                    f, edge_loss ,facc_loss ,groove_loss , output_mask = \
                        sess.run([detector.output,
                                         detector.eloss,
                                         detector.floss,
                                         detector.gloss,
                                         detector.f_mask,
                                         
                                         ], feed_dict=feed_dict)
                    
                    get_avg(iter,[edge_loss,facc_loss ,groove_loss ])
                    
                    # #将target 和 result 同框显示
                    # f=np.concatenate([y_batch,f],axis=1)
                    # mask_batch=np.concatenate([mask_batch,mask_batch],axis=1)
                    
                    print('edge_loss= ',edge_loss,'    facc_loss= ',facc_loss,'   groove_loss= ',groove_loss,
                          '    name: ',name_batch[0])
                    if edge_loss>100:
                        display_batch(box_batch, f, mask_batch)
                        display_batch(box_batch, y_batch, mask_batch)

    
                    if NEED_DISPLAY:
                        display_batch(box_batch, f, mask_batch)
                        
                print('avg  ', avg)
            else:
                
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=False, need_name=True)
                while True:
                    box_batch, name_batch = test_batch_gen.get_batch()

                    if NEED_PB:
                        pred_end = sess.graph.get_tensor_by_name('import/detector/output_node:0')
                        feed_dict = {'import/detector/input_box:0': box_batch, 'import/detector/is_training:0': False}
                    else:
                        feed_dict = {detector.input_box: box_batch,detector.is_training: False}
        
                    f = sess.run(detector.output, feed_dict=feed_dict)
                    # print('feature\n  ',f,'    name: ',name_batch[0])

                    # f=f*12./128.
                    f = np.int32(f)
                    print(f)
                    mask_batch=np.ones([box_batch.shape[0],27]).astype(bool)
                    # mask_batch[:,6:21]=True
                    if NEED_DISPLAY:
                        display_batch(box_batch, f, mask_batch)

    
        

