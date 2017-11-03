import tensorflow as tf
import os
import numpy as np
from mayavi import mlab

# from crop_data import crop_batch
from crop_data import crop_batch
from dataRelated import BatchGenerator
from display import edges
from level_train import Level

def recover_coord(fp_1,fp_2,shape_crop):
    shape_crop=np.array(shape_crop)
    cubic_pos=fp_1-(shape_crop / 2).astype(np.int32)+fp_2
    return cubic_pos

class NetConfig_1(object):
    shape_box=[128,128,128]
    channels = [32,  32,   32,  32, 64,128,256]#决定左侧的参数多少和左侧的memory
    fc_size = [128,6]
    pooling=[True,False,False,True,True,True,True]
    filter_size=[5,3,3,3,3,3,3] #决定左侧的参数多少
    stride=[1,1,1,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1

class NetConfig_2(object):
    shape_box=[32,32,32]
    channels = [32,  32,   64,  64, 128]#决定左侧的参数多少和左侧的memory
    fc_size = [64,3]
    pooling=[True,False,False,True,True,True,True]
    filter_size=[3,3,3,3,3,3,3] #决定左侧的参数多少
    stride=[1,1,1,1,1,1,1] #决定右侧的memory
    layer_num = len(channels) - 1



class DataConfig(object):
    world_to_cubic=128/12.
    batch_size=2
    total_case_dir='F:/ProjectData/Feature/test_mul'
    load_case_once=1  #每次读的病例数
    switch_after_shuffles=1000000 #当前数据洗牌n次读取新数据


if __name__ == '__main__':
    MODEL_PATH= 'F:/ProjectData/Feature/model/'
    NEED_RESTORE=False
    NEED_SAVE=True
    shape_box=[128,128,128]
    shape_crop=[32,32,32]


    level_1=Level(Param=NetConfig_1, is_training=False, scope='level_1')

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name ]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name ]
    var_list += bn_moving_vars
    saver_1 = tf.train.Saver(var_list=var_list)




    temp_var_t = set(tf.trainable_variables())
    temp_var_g = set(tf.global_variables())

    level_21=Level(Param=NetConfig_2, is_training=False, scope='level_21')

    var_list = list(set(tf.trainable_variables()) - temp_var_t)
    g_list = list(set(tf.global_variables()) - temp_var_g)
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name ]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name ]
    var_list += bn_moving_vars
    saver_21 = tf.train.Saver(var_list=var_list)

    temp_var_t = set(tf.trainable_variables())
    temp_var_g = set(tf.global_variables())

    level_22=Level(Param=NetConfig_2, is_training=False, scope='level_22')

    var_list = list(set(tf.trainable_variables()) - temp_var_t)
    g_list = list(set(tf.global_variables()) - temp_var_g)
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name ]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name ]
    var_list += bn_moving_vars
    saver_22 = tf.train.Saver(var_list=var_list)


################
    test_batch_gen=BatchGenerator(DataConfig)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('log/', sess.graph)

        # assert os.path.exists(MODEL_PATH+ 'checkpoint')  # 判断模型是否存在
        saver_1.restore(sess, os.path.join(MODEL_PATH,'level_1/model.ckpt'))  # 存在就从模型中恢复变量
        saver_21.restore(sess, os.path.join(MODEL_PATH,'level_21/model.ckpt'))  # 存在就从模型中恢复变量
        saver_22.restore(sess, os.path.join(MODEL_PATH,'level_22/model.ckpt'))  # 存在就从模型中恢复变量

        while True:

            box_batch, target = test_batch_gen.get_batch()
            target_1=target[:,:3]
            target_2=target[:,3:]

            feed_dict = {level_1.box: box_batch,
                         level_1.phase: False, level_1.keep_prob: 1}

            pred = sess.run(level_1.pred, feed_dict=feed_dict)

            box_batch=np.squeeze(box_batch,axis=4)

            croped_batch = crop_batch(pred[:,:3].astype(np.int32), box_batch, shape_box, shape_crop)
            croped_batch=np.expand_dims(croped_batch,axis=4)
            feed_dict = {level_21.box: croped_batch,level_21.targets:target_1,
                         level_21.phase: False, level_21.keep_prob: 1}
            pred_21,loss = sess.run([level_21.pred, level_21.loss], feed_dict=feed_dict)

            pred_end_1 = recover_coord(pred[:, :3], pred_21, shape_crop).astype(np.int32)


            croped_batch = crop_batch(pred[:,3:].astype(np.int32), box_batch, shape_box, shape_crop)
            croped_batch=np.expand_dims(croped_batch,axis=4)
            feed_dict = {level_22.box: croped_batch,level_22.targets:target_2,
                         level_22.phase: False, level_22.keep_prob: 1}
            pred_22,loss = sess.run([level_22.pred, level_22.loss], feed_dict=feed_dict)

            pred_end_2 = recover_coord(pred[:, 3:], pred_22, shape_crop).astype(np.int32)
            for i in range(DataConfig.batch_size):

                pred_1=pred_end_1[i]
                pred_2=pred_end_2[i]
                box=box_batch[i]
                # box=np.squeeze(box_batch[i],axis=3)

                box[target_1[i,0], target_1[i,1], target_1[i,2]] = 2
                box[target_2[i,0], target_2[i,1], target_2[i,2]] = 2
                box[pred_1[0], pred_1[1], pred_1[2]] = 3
                box[pred_2[0], pred_2[1], pred_2[2]] = 3

                x, y, z = np.where(box == 1)
                ex, ey, ez = edges(128)
                fx, fy, fz = np.where(box == 2)
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





    #
    #         #级联双精度版本
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
    #
    #
    #
    #
    #
    #
