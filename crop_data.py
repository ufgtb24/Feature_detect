import tensorflow as tf
import os
import numpy as np
from dataRelated import DataManager
from level_train import Level, NetConfig, DataConfig
import pickle



# def crop(crop_center, box, shape_box,shape_crop, point_class, save_path):
#     #不能序列操作，因为涉及索引
#     batch_num=box.shape[0]
#     shape_box=np.array(shape_box)
#     shape_crop=np.array(shape_crop)
#
#     #[b,3]
#     margin_left= crop_center - shape_crop / 2
#     margin_right= crop_center + shape_crop / 2
#
#
#     sample_margin_left=np.maximum(margin_left,np.zeros((3))[np.newaxis,:])
#     sample_margin_right=np.minimum(margin_right,shape_box[np.newaxis,:])
#
#     assign_margin_left=np.abs(margin_left)
#     assign_margin_right=shape_crop+sample_margin_right-margin_right
#     assign_index=np.arange()
#     box_crop=np.zeros([batch_num]+shape_crop)
#     box_crop[]



def crop_case(crop_center, box, shape_box,shape_crop):
    #不能序列操作，因为涉及索引
    shape_box=np.array(shape_box)
    shape_crop=np.array(shape_crop)

    #[3]
    margin_left= crop_center - shape_crop / 2
    margin_right= crop_center + shape_crop / 2


    sample_margin_left=np.maximum(margin_left,np.zeros((3)))
    sample_margin_right=np.minimum(margin_right,shape_box)

    assign_margin_left=np.abs(margin_left)
    assign_margin_right=shape_crop+sample_margin_right-margin_right


    mask_c_x=np.arange(assign_margin_left[0],assign_margin_right[0])
    mask_c_y=np.arange(assign_margin_left[1],assign_margin_right[1])
    mask_c_z=np.arange(assign_margin_left[2],assign_margin_right[2])

    mask_x=np.arange(sample_margin_left[0],sample_margin_right[0])
    mask_y=np.arange(sample_margin_left[1],sample_margin_right[1])
    mask_z=np.arange(sample_margin_left[2],sample_margin_right[2])

    box_crop=np.zeros(shape_crop)
    box_crop[mask_c_x,mask_c_y,mask_c_z]=box[mask_x,mask_y,mask_z]
    return box_crop

def crop_batch(crop_center, box, shape_box,shape_crop):
    batch_num=box.shape[0]
    box_crop_list=[]
    for i in range(batch_num):
        box_crop=crop_case(crop_center[i], box[i], shape_box,shape_crop)
        box_crop_list.append(box_crop)
    box_crop_batch=np.stack(box_crop_list)
    return box_crop_batch


def save_croped_batch(obj,path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_croped_batch(path):
    with open(path,'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == '__main__':
    ROOT_PATH = 'F:/ProjectData/Feature'
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1'

    DataConfig.batch_size_train=1
    DataConfig.batch_size_test=1

    dataManager =DataManager(DataConfig(),
                train_box_path=os.path.join(ROOT_PATH,'train'),
                train_info_file=os.path.join(ROOT_PATH,'train','info.txt'),
                test_box_path=os.path.join(ROOT_PATH,'test'),
                test_info_file=os.path.join(ROOT_PATH,'test','info.txt')
                )

    box_batch, point_batch = dataManager.getTrainBatch()
    point_batch_list=np.split(point_batch,4,axis=1)
    i=0
    for point_class_batch in point_batch_list:
        croped_batch=crop_batch(point_class_batch, box_batch, DataConfig.shape_box, DataConfig.shape_crop)
        save_croped_batch(croped_batch,path=os.path.join(ROOT_PATH,'crop_{0}'.format(i)))
        i+=1







