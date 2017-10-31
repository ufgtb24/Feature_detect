import os
import numpy as np
from dataRelated import BatchGenerator
import pickle
from mayavi import mlab



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


    # noise = np.random.normal(0, 0.01, data.shape).astype(np.float32)
    # data_added = data + noise
def generate_edge(a1, a2, a3, len_):
    def axis(a):
        if a==0:
            return np.zeros([len_], dtype=np.int32)
        elif a==1:
            return np.ones([len_], dtype=np.int32) * (len_ - 1)
        else:
            return np.arange(len_)
    return np.stack([axis(a1),axis(a2),axis(a3)])

def edges():
    e_list=[]
    e_list.append(generate_edge(0,0,2,32))
    e_list.append(generate_edge(0,1,2,32))
    e_list.append(generate_edge(1,1,2,32))
    e_list.append(generate_edge(1,0,2,32))

    e_list.append(generate_edge(0,2,0,32))
    e_list.append(generate_edge(0,2,1,32))
    e_list.append(generate_edge(1,2,1,32))
    e_list.append(generate_edge(1,2,0,32))

    e_list.append(generate_edge(2,0,0,32))
    e_list.append(generate_edge(2,0,1,32))
    e_list.append(generate_edge(2,1,1,32))
    e_list.append(generate_edge(2,1,0,32))

    e=np.concatenate(e_list,axis=1) #3,32*12
    ex,ey,ez=np.split(e,3,axis=0)
    return ex,ey,ez

def crop_case(crop_center, box, shape_box,shape_crop):
    #不能序列操作，因为涉及索引
    shape_box=np.array(shape_box)
    shape_crop=np.array(shape_crop)

    #[3]
    margin_left= crop_center - (shape_crop / 2).astype(np.int32)
    margin_right= crop_center + (shape_crop / 2).astype(np.int32)


    sample_margin_left=np.maximum(margin_left,np.zeros((3),dtype=np.int32))
    sample_margin_right=np.minimum(margin_right,shape_box)

    assign_margin_left=np.maximum(-margin_left,np.zeros((3),dtype=np.int32))
    assign_margin_right=np.minimum(shape_box-margin_left,shape_crop)

    box_crop=np.zeros(shape_crop,dtype=np.int32)
    box_crop[ assign_margin_left[0]:assign_margin_right[0],
    assign_margin_left[1]: assign_margin_right[1],
    assign_margin_left[2]: assign_margin_right[2]
    ]=box[sample_margin_left[0]: sample_margin_right[0],
          sample_margin_left[1]: sample_margin_right[1],
          sample_margin_left[2]: sample_margin_right[2]]

    return box_crop

def augment_crop(feature_id,case_name, crop_center, box_batch, shape_box, shape_crop):

    sampling_grid=np.random.uniform(-5,5,(30,3)).astype(np.int32)
    feature_aug_list=[]
    feature_txt_list=[]
    for aug in range(30):
        bias=sampling_grid[aug]#[3,]

        croped_batch=crop_batch(crop_center + bias, box_batch, shape_box, shape_crop)
        feature_aug_list.append(croped_batch)

        shape_crop = np.array(shape_crop)#[3,]
        new_target=(shape_crop / 2).astype(np.int32)-bias
        feature_txt_list.append(new_target)

    feature_aug=np.concatenate(feature_aug_list)
    feature_txt=np.stack(feature_txt_list)
    os.makedirs(os.path.join(ROOT_PATH, 'croped',case_name))
    save_croped_batch(feature_aug,
                      path=os.path.join(ROOT_PATH, 'croped', 'feature_{0}'.format(feature_id),
                                        case_name,'.pkl'))

    with open(os.path.join(ROOT_PATH,'croped', 'aug_target{0}.txt'.format(feature_id)), 'wb') as file:
        np.savetxt(file, feature_txt, fmt='%d')


def crop_batch(crop_center, box_batch, shape_box, shape_crop):
    batch_num=box_batch.shape[0]
    box_crop_list=[]
    for i in range(batch_num):
        box_crop=crop_case(crop_center[i], box_batch[i], shape_box, shape_crop)
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

class CropBatchGenerator(BatchGenerator):
    def __init__(self,data_config):
        super(CropBatchGenerator,self).__init__(data_config)
        self.current_case=0



    def get_batch(self):
        print('reload data')
        full_case_dir=super(CropBatchGenerator,self).total_case_dir+'\\'+\
            super(CropBatchGenerator,self).case_dir_list[self.current_case]
        box =np.expand_dims(super(CropBatchGenerator,self).loadmhds(full_case_dir),axis=4)
        y=super(CropBatchGenerator, self).load_y(full_case_dir + '\\info.txt')
        return box,y



if __name__ == '__main__':
    ROOT_PATH = 'F:/ProjectData/Feature'
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1'


    class CropDataConfig(object):
        shape_box = [128, 128, 128]
        shape_crop = [32, 32, 32]
        world_to_cubic = 128 / 12.
        batch_size = 28
        total_case_dir = 'F:/ProjectData/Feature/test_mul'
        format = 'mhd'


    test_batch_gen=BatchGenerator(CropDataConfig)

    box_batch, point_batch = test_batch_gen.get_batch()
    point_batch_list=np.split(point_batch,2,axis=1)


    for feature_point,feature_id in zip(point_batch_list,range(len(point_batch_list))):
        augment_crop(feature_id,case_name,feature_point, box_batch, CropDataConfig.shape_box, CropDataConfig.shape_crop)








