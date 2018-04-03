import argparse

import tensorflow as tf
import os
import numpy as np
from mayavi import mlab
# from crop_data import crop_batch
from combine import load_graph, output_graph, PB_PATH, gen_frozen_graph
from config import MODEL_PATH, ValiDataConfig, SHAPE_BOX, TestDataConfig, DataConfig
from dataRelated import BatchGenerator
from display import edges
from level_train import DetectNet



def recover_coord(fp_1,fp_2,shape_crop):
    with tf.name_scope('recover_coord'):
        shape_crop=np.array(shape_crop)
        cubic_pos=fp_1-(shape_crop / 2).astype(np.int32)+fp_2
        cubic_pos=tf.to_int32(cubic_pos)
        return cubic_pos


if __name__ == '__main__':
    NEED_INFERENCE=True
    NEED_DISPLAY=False
    NEED_SPLIT=True
    NEED_WRITE_GRAPH=False
    
    is_training = tf.placeholder(tf.bool,name='is_training')

    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    box = tf.to_float(input_box)
    targets = tf.placeholder(tf.float32, shape=[None, DataConfig.output_dim],
                                  name="targets")

    detector = DetectNet(is_training=is_training, need_optim=False,scope='detector', input_box=box, targets=targets)
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

    
    
    # var_list = tf.trainable_variables()
    # task_spec_vars = [g for g in var_list if 'task_spec_conv' in g.name]

    #############
    saver = tf.train.Saver(var_list)
    saver_spec = tf.train.Saver(task_spec_vars)
    saver_commen = tf.train.Saver(var_list-task_spec_vars)


    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(MODEL_PATH,'model.ckpt'))  # 存在就从模型中恢复变量
        
        if NEED_SPLIT:
            saver_commen.save(sess, os.path.join(MODEL_PATH,'commen/model.ckpt'))
            saver_spec.save(sess, os.path.join(MODEL_PATH,'spec/model.ckpt'))
            
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
            test_batch_gen = BatchGenerator(ValiDataConfig, need_target=True,need_name=True)
            while True:
                box_batch, y_batch, name_batch = test_batch_gen.get_batch()

                feed_dict = {input_box: box_batch, targets: y_batch,
                             is_training: False}

                f, error = sess.run([pred_end, detector.error], feed_dict=feed_dict)
                print(error)

        if NEED_DISPLAY:
            test_batch_gen = BatchGenerator(TestDataConfig,need_target=True,need_name=True)
            while True:
                # box_batch,y_batch,name_batch = test_batch_gen.get_batch()
                box_batch,y_batch,name_batch = test_batch_gen.get_batch()

                feed_dict = {input_box: box_batch,targets: y_batch,
                             is_training: False}

                f,error = sess.run([pred_end,detector.error], feed_dict=feed_dict)
                f=np.int32(f[0])
                print(name_batch[0],"  ",f,"   ",error)
                f_1=f[:3]
                f_2=f[3:]

                box=np.squeeze(box_batch)

                # box[target_1[i,0], target_1[i,1], target_1[i,2]] = 2
                # box[target_2[i,0], target_2[i,1], target_2[i,2]] = 2
                box[f_1[0], f_1[1], f_1[2]] = 3
                box[f_2[0], f_2[1], f_2[2]] = 3

                x, y, z = np.where(box == 1)
                ex, ey, ez = edges(128)
                # fx, fy, fz = np.where(box == 2)
                fxp, fyp, fzp = np.where(box == 3)

                mlab.points3d(ex, ey, ez,
                              mode="cube",
                              color=(0, 0, 1),
                              scale_factor=1)

                mlab.points3d(x, y, z,
                              mode="cube",
                              color=(0, 1, 0),
                              scale_factor=1)
                              # transparent=True)

                # mlab.points3d(fx, fy, fz,
                #             mode="cube",
                #             color=(1, 0, 0),
                #             scale_factor=1,
                #               transparent=True)

                mlab.points3d(fxp, fyp, fzp,
                            mode="cube",
                            color=(0, 0, 1),
                            scale_factor=1,
                              transparent=True)

                mlab.show()

