import tensorflow as tf
import numpy as np
# from crop_data import crop_batch
from combine import  PB_PATH, gen_frozen_graph
from config import MODEL_PATH, SHAPE_BOX, TestDataConfig, DataConfig
from dataRelated import BatchGenerator
from display import  display_batch
from level_train import DetectNet


if __name__ == '__main__':

    
    NEED_INFERENCE=False
    NEED_TARGET=False
    NEED_DISPLAY=False
    NEED_WRITE_GRAPH=True
    
    
    
    
    detector = DetectNet(need_targets=NEED_TARGET,is_training_sti=False)
    # pred_end = tf.to_int32(tf.identity(detector.pred,name="output_node"))
    pred_end = detector.total_output
    
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

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)
        
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_PATH+'model.ckpt-74')  # 存在就从模型中恢复变量

        # if NEED_SPLIT:
        #     saver_commen.save(sess, os.path.join(MODEL_PATH,'commen/model.ckpt'))
        #     saver_spec.save(sess, os.path.join(MODEL_PATH,'spec/model.ckpt'))
            
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
            gen_frozen_graph()
            # load_graph(output_graph)

        if NEED_INFERENCE:
            if NEED_TARGET:
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=True,need_name=True)
                for iter in range(100):
                    box_batch, y_batch, mask_batch, class_batch, name_batch = test_batch_gen.get_batch()
                    
                    feed_dict = {detector.input_box: box_batch,
                                 detector.targets: y_batch,
                                 detector.f_mask: mask_batch,
                                 detector.is_training: False}
                    f, error = sess.run([pred_end, detector.feature_loss], feed_dict=feed_dict)

                    f=np.int32(f)
                    print('feature_loss= ',error)
    
                    if NEED_DISPLAY:
                        box_batch = np.squeeze(box_batch, 4)
                        display_batch(box_batch, f, mask_batch, TestDataConfig.num_feature_need)
            else:
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=False, need_name=True)
                while True:
                    box_batch,  name_batch = test_batch_gen.get_batch()
        
                    feed_dict = {detector.input_box: box_batch,detector.is_training: False}
        
                    f = sess.run(pred_end, feed_dict=feed_dict)
                    # f=f*12./128.
                    f = np.int32(f)
                    print(f)
        
                    if NEED_DISPLAY:
                        box_batch = np.squeeze(box_batch, 4)
                        display_batch(box_batch, f, TestDataConfig.num_feature_need)

    
        

