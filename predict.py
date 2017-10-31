import tensorflow as tf
import os
import numpy as np
from mayavi import mlab

# from crop_data import crop_batch
from dataRelated import BatchGenerator
from display import edges
from level_train import Level

def recover_coord(fp_1,fp_2,shape_crop):
    shape_crop=np.array(shape_crop)
    cubic_pos=fp_1-(shape_crop / 2).astype(np.int32)+fp_2
    return cubic_pos

class NetConfig(object):
    shape_box=[128,128,128]
    channels = [32,  32,   64,  128]#决定左侧的参数多少和左侧的memory
    fc_size = [512,128,6]
    pooling=[True,True,True,True,True]
    filter_size=[5,3,3,3,3] #决定左侧的参数多少
    stride=[2,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1

class TestDataConfig(object):
    shape_box=[128,128,128]
    shape_crop=[64,64,64]
    world_to_cubic=128/12.
    batch_size=28
    total_case_dir='F:/ProjectData/Feature/test_mul'
    load_case_once=1  #每次读的病例数
    switch_after_shuffles=1000000 #当前数据洗牌n次读取新数据
    format = 'mhd'


if __name__ == '__main__':
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1/'
    NEED_RESTORE=False
    NEED_SAVE=True


    with tf.variable_scope('Level_1'):
        level=Level(Param=NetConfig,is_training=True,scope='level_1')

    # saver = tf.train.Saver(max_to_keep=1)
################
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
################
    test_batch_gen=BatchGenerator(TestDataConfig)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)

        assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
        saver.restore(sess, MODEL_PATH + 'model.ckpt')  # 存在就从模型中恢复变量

        box_batch, target = test_batch_gen.get_batch()

        num = target.shape[0]
        for i in range(num):


            feed_dict={level.box:box_batch[i][np.newaxis,:], level.targets:target[i][np.newaxis,:],
                           level.phase:False,level.keep_prob:1}

            pred, loss_test=sess.run([level.pred, level.loss], feed_dict=feed_dict)

            pred=np.squeeze(pred, axis=0).astype(np.int32)
            box=np.squeeze(box_batch[i],axis=3)
            box[target[i,0], target[i,1], target[i,2]] = 2
            box[target[i,3], target[i,4], target[i,5]] = 2
            box[pred[0], pred[1], pred[2]] = 3
            box[pred[3], pred[4], pred[5]] = 3

            x, y, z = np.where(box == 1)
            ex, ey, ez = edges(128)
            fz, fy, fx = np.where(box == 2)
            fzp, fyp, fxp = np.where(box == 3)

            mlab.points3d(ex, ey, ez,
                          mode="cube",
                          color=(0, 0, 1),
                          scale_factor=1)

            mlab.points3d(x, y, z,
                          mode="cube",
                          color=(0, 1, 0),
                          scale_factor=1,
                          transparent=True)

            mlab.points3d(fx, fy, fz,
                        mode="cube",
                        color=(1, 0, 0),
                        scale_factor=1,
                          transparent=True)

            mlab.points3d(fxp, fyp, fzp,
                        mode="cube",
                        color=(0, 0, 1),
                        scale_factor=1,
                          transparent=True)

            mlab.show()




            #级联双精度版本
    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter('log/', sess.graph)
    #
    #     saver.restore(sess, os.path.join(MODEL_PATH ,'model.ckpt'))  # 存在就从模型中恢复变量
    #
    #     box_batch, pos_batch, y_batch=dataManager.getTestBatch()
    #     feed_dict={level.box:box_batch, level.targets:y_batch,
    #                level.phase:True,level.keep_prob:0.5}
    #     point_batch,loss_train=sess.run([level.pred,level.loss],feed_dict=feed_dict)
    #     point_batch_list=np.split(point_batch,2,axis=1)
    #
    #     y_list=np.split(y_batch,2,axis=1)
    #     for feature_point,y_class,i in zip(point_batch_list,y_list,range(2)):
    #         # [b]+shape_crop
    #         croped_batch = crop_batch(feature_point, box_batch, DataConfig.shape_box, DataConfig.shape_crop)
    #
    #         feed_dict = {level_2[i].box: croped_batch,
    #                      level_2[i].phase: False,level_2[i].keep_prob:1}
    #         feature_point_2 = sess.run([level_2[i].pred,level_2[i].loss], feed_dict=feed_dict)
    #
    #         cubic_pos=recover_coord(feature_point,feature_point_2,DataConfig.shape_crop)
    #
    #         if SHOW:
    #             cubic_pos=np.squeeze(cubic_pos,axis=0)
    #             box_batch=np.squeeze(box_batch, axis=0)
    #             box_batch[cubic_pos[0],cubic_pos[1],cubic_pos[2]]=2
    #
    #     if SHOW:
    #         box_batch = np.squeeze(box_batch, axis=0)
    #         x, y, z = np.where(box_batch == 255)
    #         fz, fy, fx = np.where(box_batch == 2)
    #         mlab.points3d(x, y, z,
    #                       mode="cube",
    #                       color=(0, 1, 0),
    #                       scale_factor=1)
    #         mlab.points3d(fx, fy, fz,
    #                       mode="cube",
    #                       color=(1, 0, 0),
    #                       scale_factor=1)
    #         mlab.show()






