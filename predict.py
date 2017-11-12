import tensorflow as tf
import os
import numpy as np
from mayavi import mlab

# from crop_data import crop_batch
from crop_data_tf import crop_batch
from dataRelated import BatchGenerator
from display import edges
from level_train import Level

SHAPE_BOX = [128, 128, 128]
SHAPE_CROP = [32, 32, 32]


def recover_coord(fp_1,fp_2,shape_crop):
    with tf.name_scope('recover_coord'):
        shape_crop=np.array(shape_crop)
        cubic_pos=fp_1-(shape_crop / 2).astype(np.int32)+fp_2
        cubic_pos=tf.to_int32(cubic_pos)
        return cubic_pos

class NetConfig_1(object):
    shape_box=SHAPE_BOX
    channels = [32,  32,   32,  32, 64,128,256]#决定左侧的参数多少和左侧的memory
    fc_size = [128,6]
    pooling=[True,False,False,True,True,True,True]
    filter_size=[5,3,3,3,3,3,3] #决定左侧的参数多少
    stride=[1,1,1,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1

class NetConfig_2(object):
    shape_box=SHAPE_CROP
    channels = [32,  32,   64,  64, 128]#决定左侧的参数多少和左侧的memory
    fc_size = [64,3]
    pooling=[True,False,False,True,True,True,True]
    filter_size=[3,3,3,3,3,3,3] #决定左侧的参数多少
    stride=[1,1,1,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1



class DataConfig(object):
    world_to_cubic=128/12.
    batch_size=4
    total_case_dir='F:/ProjectData/Feature/predict/Tooth/'
    load_case_once=20  #每次读的病例数
    switch_after_shuffles=1000 #当前数据洗牌n次读取新数据


if __name__ == '__main__':
    MODEL_PATH= 'F:/ProjectData/Feature/model/'
    NEED_RESTORE=False
    NEED_SAVE=True
    keep_prob = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool)

    level_1=Level(Param=NetConfig_1, is_training=False, scope='level_1',
                  keep_prob=keep_prob,phase=phase)

    saver_1 = tf.train.Saver(var_list=tf.global_variables())

    box_21 = crop_batch(level_1.pred[:, :3], level_1.box, SHAPE_BOX, SHAPE_CROP,'crop_batch_1')
    box_22 = crop_batch(level_1.pred[:, 3:], level_1.box, SHAPE_BOX, SHAPE_CROP,'crop_batch_2')

    temp_var_g = set(tf.global_variables())
    level_21=Level(Param=NetConfig_2, is_training=False,need_target=False,
                   scope='level_21',input_box=box_21,keep_prob=keep_prob,phase=phase)

    g_list = list(set(tf.global_variables()) - temp_var_g)
    saver_21 = tf.train.Saver(var_list=g_list)

    temp_var_g = set(tf.global_variables())
    level_22=Level(Param=NetConfig_2, is_training=False, need_target=False,
                   scope='level_22',input_box=box_22,keep_prob=keep_prob,phase=phase)

    g_list = list(set(tf.global_variables()) - temp_var_g)
    saver_22 = tf.train.Saver(var_list=g_list)

    pred_end_1 = recover_coord(level_1.pred[:, :3], level_21.pred, SHAPE_CROP)
    pred_end_2 = recover_coord(level_1.pred[:, 3:], level_22.pred, SHAPE_CROP)
    pred_end_1 = tf.identity(pred_end_1, name="output_1")
    pred_end_2 = tf.identity(pred_end_2, name="output_2")

    saver = tf.train.Saver()

    test_batch_gen=BatchGenerator(DataConfig,need_target=False)
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        # assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
        saver_1.restore(sess, os.path.join(MODEL_PATH,'level_1/model.ckpt'))  # 存在就从模型中恢复变量
        saver_21.restore(sess, os.path.join(MODEL_PATH,'level_21/model.ckpt'))  # 存在就从模型中恢复变量
        saver_22.restore(sess, os.path.join(MODEL_PATH,'level_22/model.ckpt'))  # 存在就从模型中恢复变量
        saver.save(sess, os.path.join(MODEL_PATH,'whole/model.ckpt'))
        tf.train.write_graph(sess.graph_def, MODEL_PATH, 'graph.pb')
        writer = tf.summary.FileWriter(os.path.join(MODEL_PATH,'../logs/'), sess.graph)

        while True:

            box_batch = test_batch_gen.get_batch()
            # box_batch, target = test_batch_gen.get_batch()
            # target_1=target[:,:3]
            # target_2=target[:,3:]

            feed_dict = {level_1.box: box_batch,
                         phase: False, keep_prob: 1}

            f_1, f_2 = sess.run([pred_end_1,pred_end_2], feed_dict=feed_dict)

            for i in range(DataConfig.batch_size):

                pred_1=f_1[i]
                pred_2=f_2[i]
                box=box_batch[i]

                # box[target_1[i,0], target_1[i,1], target_1[i,2]] = 2
                # box[target_2[i,0], target_2[i,1], target_2[i,2]] = 2
                box[pred_1[0], pred_1[1], pred_1[2]] = 3
                box[pred_2[0], pred_2[1], pred_2[2]] = 3

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
                              scale_factor=1,)
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

