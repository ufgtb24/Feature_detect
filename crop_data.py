import os
import numpy as np
from dataRelated import BatchGenerator
import pickle


class CropBatchGenerator(BatchGenerator):
    def __init__(self,data_config):
        super(CropBatchGenerator,self).__init__(data_config)
        self.total_case_list=self.get_case_name_list()
        self.load_case_list(self.get_case_list())


    def get_case_name_list(self):
        case_name_list = []
        for fileName in os.listdir(self.total_case_dir):
            if os.path.splitext(fileName)[1] == '.pkl':
                case_name_list.append(fileName[0])
        return case_name_list

    def loadpickle(self,path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_case_list(self,case_load):
        print('load crop data')
        box_list=[]
        y_list=[]
        for case_dir in case_load:
            full_pkl_path=self.total_case_dir+'\\'+case_dir+'.pkl'
            full_txt_path=self.total_case_dir+'\\'+case_dir+'.txt'
            box_list.append(self.loadpickle(full_pkl_path))
            y_list.append(np.loadtxt(full_txt_path))

        self.box=np.expand_dims(np.concatenate(box_list,axis=0),axis=4)
        self.y=np.concatenate(y_list,axis=0)
        self.sample_num=self.y.shape[0]
        assert self.batch_size <= self.sample_num, 'batch_size should be smaller than sample_num'





def crop_case(crop_center, box, shape_box,shape_crop):
    #不能序列操作，因为涉及索引
    shape_box=np.array(shape_box)
    shape_crop=np.array(shape_crop)

    #[3]
    c0= crop_center - (shape_crop / 2).astype(np.int32)
    c1= crop_center + (shape_crop / 2).astype(np.int32)

    a0=np.maximum(-c0,np.zeros((3),dtype=np.int32))
    a1=np.minimum(shape_box-c0,shape_crop)

    s0=np.maximum(c0,np.zeros((3),dtype=np.int32))
    s1=np.minimum(c1,shape_box)

    box_crop=np.zeros(shape_crop,dtype=np.int32)
    box_crop[ a0[0]:a1[0],
    a0[1]: a1[1],
    a0[2]: a1[2]
    ]=box[s0[0]: s1[0],
          s0[1]: s1[1],
          s0[2]: s1[2]]

    return box_crop

def augment_crop(feature_id,case_dir, crop_center, box_batch, shape_box, shape_crop,
                 bias_range,aug_num):

    sampling_grid=np.random.uniform(-bias_range,bias_range,(30,3)).astype(np.int32)
    feature_aug_list=[]
    feature_txt_list=[]
    for aug in range(aug_num):
        bias=sampling_grid[aug]#[3,]
        #[b,crop_shape]
        croped_batch=crop_batch(crop_center + bias, box_batch, shape_box, shape_crop)
        feature_aug_list.append(croped_batch)
        batch_size=box_batch.shape[0]
        shape_crop = np.array(shape_crop)#[3]
        new_target=np.tile((shape_crop / 2).astype(np.int32),(batch_size,1))-bias
        feature_txt_list.append(new_target)

    #[b*30,shape_crop] b=27
    feature_aug=np.concatenate(feature_aug_list)
    #[b*30,3] b=27
    feature_txt=np.concatenate(feature_txt_list)

    feature_path=os.path.join(CROP_PATH, 'feature_{0}'.format(feature_id))
    if os.path.exists(feature_path)==False:
        os.makedirs(feature_path)
    save_croped_batch(feature_aug,
                      path=os.path.join(feature_path,case_dir+'.pkl'))

    with open(os.path.join(feature_path,case_dir+'.txt'), 'wb') as file:
        np.savetxt(file, feature_txt, fmt='%d')


def crop_batch(crop_center, box_batch, shape_box, shape_crop):
    batch_num=box_batch.shape[0]
    box_crop_list=[]
    for i in range(batch_num):
        box_crop=crop_case(crop_center[i], box_batch[i], shape_box, shape_crop)
        box_crop_list.append(box_crop)
    box_crop_batch=np.stack(box_crop_list)
    #[b,shape_crop]
    return box_crop_batch


def save_croped_batch(obj,path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


if __name__ == '__main__':
    CROP_PATH = 'F:/ProjectData/Feature/croped'
    MODEL_PATH= 'F:/ProjectData/Feature/model/level_1'


    class CropDataConfig(object):
        shape_box = [128, 128, 128]
        shape_crop = [32, 32, 32]
        world_to_cubic = 128 / 12.
        batch_size = 27
        total_case_dir = 'F:/ProjectData/Feature/Tooth'
        load_case_once = 1  # 每次读的病例数
        switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
        format = 'mhd'

    test_batch_gen=BatchGenerator(CropDataConfig)
    total_case_list = os.listdir(CropDataConfig.total_case_dir)

    for case_dir in total_case_list:
        box_batch, point_batch = test_batch_gen.get_batch()
        box_batch=np.squeeze(box_batch,axis=4)
        # if test_batch_gen.index_dir>test_batch_gen.total_case_num:
        if test_batch_gen.index_dir>2:
            break
        point_batch_list=np.split(point_batch,2,axis=1)

        #每个case对应一个pkl
        for feature_point,feature_id in zip(point_batch_list,range(len(point_batch_list))):
            augment_crop(
                feature_id,case_dir,feature_point, box_batch,
                CropDataConfig.shape_box, CropDataConfig.shape_crop,
            bias_range=5,aug_num=5)











