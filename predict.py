import tensorflow as tf
import os
import SimpleITK as sitk
import numpy as np
from mayavi import mlab

from crop_data import crop_batch
from dataRelated import DataManager, BatchGenerator
from level_train import Level


def recover_coord(fp_1,fp_2,shape_crop):
    shape_crop=np.array(shape_crop)
    cubic_pos=fp_1-(shape_crop / 2).astype(np.int32)+fp_2
    return cubic_pos

if __name__ == '__main__':
    MODEL_PATH = 'F:/ProjectData/Feature/model/'
    DATA_PATH=''
    SHOW=True


    class NetConfig_1(object):
        shape_box = [128, 128, 128]
        channels = [1, 40, 60, 80, 100]
        layer_num = len(channels) - 1
        fc_size = [512, 8]


    class NetConfig_2(object):
        shape_box = [32, 32, 32]
        channels = [1, 40, 60, 80, 100]
        layer_num = len(channels) - 1
        fc_size = [512, 8]


    class DataConfig(object):
        shape_box = [128, 128, 128]
        shape_crop = [32, 32, 32]
        world_to_cubic = 128 / 20.
        batch_size_train = 2
        batch_size_test = 1
        need_Save = False
        need_Restore = False
        format = 'mhd'

    with tf.variable_scope('Level_1'):
        level_2=[]
        level=Level(Param=NetConfig_1,is_training=False,scope='level_1')
        level_2.append(Level(Param=NetConfig_2,is_training=False,scope='level_21'))
        level_2.append(Level(Param=NetConfig_2,is_training=False,scope='level_22'))

    saver = tf.train.Saver()

    dataManager =DataManager(DataConfig(),
                test_box_path=os.path.join(DATA_PATH,'test'),
                test_info_file=os.path.join(DATA_PATH,'test','info.txt')
                )

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('log/', sess.graph)

        saver.restore(sess, os.path.join(MODEL_PATH ,'level_1.ckpt'))  # 存在就从模型中恢复变量
        saver.restore(sess, os.path.join(MODEL_PATH ,'level_21.ckpt'))  # 存在就从模型中恢复变量
        saver.restore(sess, os.path.join(MODEL_PATH ,'level_22.ckpt'))  # 存在就从模型中恢复变量


        box_batch, pos_batch, y_batch=dataManager.getTestBatch()
        feed_dict={level.box:box_batch, level.targets:y_batch,
                   level.phase:True,level.keep_prob:0.5}
        point_batch,loss_train=sess.run([level.pred,level.loss],feed_dict=feed_dict)
        point_batch_list=np.split(point_batch,2,axis=1)

        y_list=np.split(y_batch,2,axis=1)
        for feature_point,y_class,i in zip(point_batch_list,y_list,range(2)):
            # [b]+shape_crop
            croped_batch = crop_batch(feature_point, box_batch, DataConfig.shape_box, DataConfig.shape_crop)

            feed_dict = {level_2[i].box: croped_batch,
                         level_2[i].phase: False,level_2[i].keep_prob:1}
            feature_point_2 = sess.run([level_2[i].pred,level_2[i].loss], feed_dict=feed_dict)

            cubic_pos=recover_coord(feature_point,feature_point_2,DataConfig.shape_crop)

            if SHOW:
                cubic_pos=np.squeeze(cubic_pos,axis=0)
                box_batch=np.squeeze(box_batch, axis=0)
                box_batch[cubic_pos[0],cubic_pos[1],cubic_pos[2]]=2

        if SHOW:
            box_batch = np.squeeze(box_batch, axis=0)
            x, y, z = np.where(box_batch == 255)
            fz, fy, fx = np.where(box_batch == 2)
            mlab.points3d(x, y, z,
                          mode="cube",
                          color=(0, 1, 0),
                          scale_factor=1)
            mlab.points3d(fx, fy, fz,
                          mode="cube",
                          color=(1, 0, 0),
                          scale_factor=1)
            mlab.show()






