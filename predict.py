import tensorflow as tf
import os
import numpy as np
from mayavi import mlab
# from crop_data import crop_batch
from combine import generate_pb
from config import MODEL_PATH, ValiDataConfig, SHAPE_BOX
from dataRelated import BatchGenerator
from display import edges
from level_train import DetectNet



def recover_coord(fp_1,fp_2,shape_crop):
    with tf.name_scope('recover_coord'):
        shape_crop=np.array(shape_crop)
        cubic_pos=fp_1-(shape_crop / 2).astype(np.int32)+fp_2
        cubic_pos=tf.to_int32(cubic_pos)
        return cubic_pos

class TestDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 4
    total_case_dir = 'F:/ProjectData/Feature2/DataSet/Validate'
    data_list=['tooth30']
    load_case_once =0 # 每次读的病例数
    switch_after_shuffles =10**10  # 当前读取的数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'

if __name__ == '__main__':
    NEED_WRITE_GRAPH=False
    NEED_DISPLAY=True
    is_training = tf.placeholder(tf.bool,name='is_training')

    input_box = tf.placeholder(tf.uint8, shape=[None] + SHAPE_BOX, name='input_box')
    box = tf.to_float(input_box)
    targets = tf.placeholder(tf.float32, shape=[None, 6],
                                  name="targets")

    detector = DetectNet(is_training=is_training, need_optim=False,scope='detector', input_box=box, targets=targets)
    pred_end = tf.to_int32(tf.identity(detector.pred,name="output_node"))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        # assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
        saver.restore(sess, os.path.join(MODEL_PATH,'model.ckpt'))  # 存在就从模型中恢复变量
        # saver.save(sess, os.path.join(MODEL_PATH,'whole/model.ckpt'))

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
            generate_pb(gd)

        if NEED_DISPLAY:
            test_batch_gen = BatchGenerator(ValiDataConfig, name='test',need_target=True,need_name=True)
            while True:
                # box_batch,y_batch,name_batch = test_batch_gen.get_batch()
                box_batch,y_batch,name_batch = test_batch_gen.get_batch()

                feed_dict = {input_box: box_batch,targets: y_batch,
                             is_training: False}

                f,error = sess.run([pred_end,detector.error], feed_dict=feed_dict)

                for i in range(box_batch.shape[0]):
                    f=f[i]
                    print(name_batch[i],"  ",f,"   ",error)
                    f_1=f[:3]
                    f_2=f[3:]

                    box=box_batch[0][i]

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
                                  scale_factor=1,
                                  transparent=True)

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

