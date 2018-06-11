import time
t0=time.time()
import os
import tensorflow as tf
import numpy as np
# from crop_data import crop_batch
from combine import  PB_PATH, gen_frozen_graph
from config import MODEL_PATH, SHAPE_BOX, TestDataConfig, DataConfig
from dataRelated import BatchGenerator
from display import  display_batch
from level_train import DetectNet


if __name__ == '__main__':

    
    NEED_INFERENCE=True
    NEED_TARGET=True
    NEED_DISPLAY=True
    NEED_WRITE_GRAPH=False
    
    
    
    t1=time.time()
    print('before build time= ',t1-t0)
    is_training = tf.placeholder(tf.bool,name='is_training')

    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    box = tf.to_float(input_box)
    targets = tf.placeholder(tf.float32, shape=[None, DataConfig.output_dim],
                                  name="targets")
    f_mask = tf.placeholder(tf.bool, shape=(None, DataConfig.output_dim))

    detector = DetectNet(is_training=is_training, f_mask=f_mask, need_optim=False,scope='detector', input_box=box, targets=targets)
    # pred_end = tf.to_int32(tf.identity(detector.pred,name="output_node"))
    pred_end = tf.identity(detector.pred,name='output_node')
    
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
    t2=time.time()
    print('build graph time = ',t2-t1)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)
        
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(MODEL_PATH,'model.ckpt'))  # 存在就从模型中恢复变量
        t3 = time.time()
        print('init variable time = ', t3 - t2)

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
                for iter in range(5):
                    box_batch,class_batch, y_batch, name_batch = test_batch_gen.get_batch()

                    mask = np.ones_like(y_batch, dtype=bool)
                    for i, class_num in enumerate(class_batch.tolist()):
                        if class_num > 1:
                            mask[i, 15:] = False

                    feed_dict = {input_box: box_batch, targets: y_batch,
                                 f_mask: mask,
                                 is_training: False}
                    t1 = time.time()
                    f, error = sess.run([pred_end, detector.error], feed_dict=feed_dict)
                    t2 = time.time()
                    print('run sess time = ', t2 - t1)

                    f=np.int32(f)
                    print(error)
    
                    if NEED_DISPLAY:
                        box_batch = np.squeeze(box_batch, 4)
                        display_batch(box_batch, f, TestDataConfig.num_feature_need)
            else:
                test_batch_gen = BatchGenerator(TestDataConfig, need_target=False, need_name=True)
                while True:
                    box_batch,  name_batch = test_batch_gen.get_batch()
        
                    feed_dict = {input_box: box_batch,is_training: False}
        
                    f = sess.run(pred_end, feed_dict=feed_dict)
                    # f=f*12./128.
                    f = np.int32(f)
                    print(f)
        
                    if NEED_DISPLAY:
                        box_batch = np.squeeze(box_batch, 4)
                        display_batch(box_batch, f, TestDataConfig.num_feature_need)

    
        

