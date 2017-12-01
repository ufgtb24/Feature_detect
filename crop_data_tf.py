import os
import numpy as np
from dataRelated import BatchGenerator
import pickle
import tensorflow as tf


class CropedBatchGenerator(BatchGenerator):
    def __init__(self,data_config):
        super(CropedBatchGenerator,self).__init__(data_config)


    def get_total_case_list(self):
        case_name_list = []
        for fileName in os.listdir(self.total_case_dir):
            if os.path.splitext(fileName)[1] == '.pkl':
                case_name_list.append(os.path.splitext(fileName)[0])
        return case_name_list

    def loadpickle(self,path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_case_list(self,case_load):
        print('load crop data')
        box_list=[]
        y_list=[]
        for case_name in case_load:
            full_pkl_path=self.total_case_dir+'\\'+case_name+'.pkl'
            full_txt_path=self.total_case_dir+'\\'+case_name+'.txt'
            box_list.append(self.loadpickle(full_pkl_path))
            y_list.append(np.loadtxt(full_txt_path))

        self.box=np.expand_dims(np.concatenate(box_list,axis=0),axis=4)
        self.y=np.concatenate(y_list,axis=0)
        self.sample_num=self.y.shape[0]
        assert self.batch_size <= self.sample_num, 'batch_size should be smaller than sample_num'


def crop_case(crop_center, box, box_crop):

    # 不能序列操作，因为涉及索引
    shape_box = tf.shape(box)
    shape_crop = tf.shape(box_crop)

    # [3]
    crop_center=tf.to_int32(crop_center)
    c0 = crop_center - tf.to_int32(shape_crop / 2)
    c1 = crop_center + tf.to_int32(shape_crop / 2)

    a0 = tf.maximum(-c0, tf.zeros((3), dtype=tf.int32))
    a1 = tf.minimum(shape_box - c0, shape_crop)

    s0 = tf.maximum(c0, tf.zeros((3), dtype=tf.int32))
    s1 = tf.minimum(c1, shape_box)


    reset=tf.assign(box_crop,tf.zeros(tf.shape(box_crop)))
    reset = tf.Print(reset, [box_crop.name], 'reset')

    with tf.control_dependencies([reset]):
        update = tf.assign(box_crop[a0[0]:a1[0],
                           a0[1]: a1[1],
                           a0[2]: a1[2]
                           ],
                           box[s0[0]: s1[0],
                           s0[1]: s1[1],
                           s0[2]: s1[2]])
        update = tf.Print(update, [box_crop.name], 'update')


    with tf.control_dependencies([update]):
        new_box_crop = box_crop.read_value()
    return new_box_crop



def crop_batch(crop_center, box_batch, shape_box,shape_crop,scope):
    '''
    按批在box中截取crop_center为中心的小box
    :param crop_center: [b,3]
    :param box_batch: [b,w,h,d]
    :param shape_box: [w,h,d]
    :param shape_crop: [wc,hc,dc]
    :return:
    '''

    def condition(index, output_array):
        with tf.name_scope('condition'):
            return tf.less(index, tf.shape(crop_center)[0])

    # The loop body, this will return a result tuple in the same form (index, summation)


    def body(index, output_array):
        with tf.name_scope('body'):
            center_i = crop_center[index]
            box_i = box_batch[index]
            # box_crop_var = tf.Variable(tf.zeros(shape=shape_crop),name='box_crop_var')
            # box_crop_var = tf.get_variable(name='box_crop_var')
            box_crop_var = tf.get_variable('box_crop_var', shape=shape_crop, dtype=tf.float32,
                                           initializer=tf.zeros_initializer)
            box_crop = crop_case(center_i, box_i, shape_box, shape_crop, box_crop_var)
            output_array = output_array.write(index, box_crop)
            return tf.add(index, 1), output_array

    # with tf.variable_scope(scope):
    #
    #     box_crop_var=tf.get_variable('box_crop_var',shape=shape_crop,dtype=tf.float32,initializer=tf.zeros_initializer)

    with tf.variable_scope(scope,reuse=False):
        crop_center = tf.to_int32(crop_center)
        output_array = tf.TensorArray(size=tf.shape(crop_center)[0], dtype=tf.float32)
        index =tf.constant(0)
        index_final, output_array_final=tf.while_loop(condition, body, [index,output_array])
        box_crop_batch=output_array_final.stack()
        return box_crop_batch

if __name__ == '__main__':
    CROP_AUG_SAVE_PATH = 'F:/ProjectData/Feature/croped'
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1'
    shape_box=[128,128,128]
    shape_crop=[32,32,32]
    # crop
    class CropDataConfig(object):
        world_to_cubic = 128 / 12.
        batch_size = 27
        total_case_dir = 'F:/ProjectData/Feature/Tooth'
        load_case_once = 1  # 每次读的病例数
        switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效


    test_batch_gen=BatchGenerator(CropDataConfig)
    total_case_list = os.listdir(CropDataConfig.total_case_dir)












