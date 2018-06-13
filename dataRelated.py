import numpy as np
import SimpleITK as sitk
import os

from config import DataConfig, ValiDataConfig, TrainDataConfig
from collections import OrderedDict


class BatchGenerator(object):
    def __init__(self, data_config,
                 need_target=True,
                 need_name=False):
        self.usage = data_config.usage
        self.class_define=data_config.class_define
        if need_target:
            final_task_dict = OrderedDict([])
            for task, task_content in data_config.task_dict.items():
                
                index = []
                for f in task_content['feature_need']:
                    index += (3 * f + np.array([0, 1, 2])).tolist()
                
                task_content['index'] = index
                
                final_task_dict[task] = task_content
            
        self.class_ftr_dict = data_config.class_ftr_dict
        self.down_rate = data_config.down_rate
        self.box = None
        self.need_target = need_target
        self.need_name = need_name
        self.total_case_dir = data_config.total_case_dir
        self.total_case_list = os.listdir(self.total_case_dir)
        self.total_case_num = len(self.total_case_list)
        self.load_case_once = data_config.load_case_once
        assert self.load_case_once <= self.total_case_num, \
            'load_case_once should be smaller than total_case_num'
        
        self.switch_after_shuffles = data_config.switch_after_shuffles
        self.world_to_cubic = data_config.world_to_cubic
        self.batch_size = data_config.batch_size
        self.tooth_list = data_config.data_list
        self.index = 0
        self.index_dir = 0
        self.shuffle_times = 0

        self.box_class_list = [[], [], [], []]
        self.y_class_list = [[], [], [], []]
        self.class_box_y_dict={0:None, 1:None, 2:None, 3:None}

        if self.load_case_once > 0:
            self.load_case_list(self.get_case_list())
        else:
            self.load_case_list(self.total_case_list)
        self.suffle()
    
    def get_case_list(self):
        if self.index_dir + self.load_case_once > self.total_case_num:
            self.index_dir = 0
            perm = np.arange(self.total_case_num)
            np.random.shuffle(perm)  # 打乱
            array = np.array(self.total_case_list)
            array = array[perm]
            self.total_case_list = list(array)
        
        case_load = self.total_case_list[self.index_dir:self.index_dir + self.load_case_once]
        self.index_dir += self.load_case_once
        return case_load
    
    def load_useful_tooth(self, full_case_dir, target_tooth_list,box_class_set,y_class_set):
        # 读取一个病例中的多颗牙齿
        actual_tooth_list = os.listdir(full_case_dir)
        
        
        for tooth in target_tooth_list:
            if tooth not in actual_tooth_list:
                continue
            tooth_dir = full_case_dir + tooth + '/'
            # append [aug_num,box_size]
            box_class_set[self.class_define[tooth]].append(self.loadmhds(tooth_dir))
            feature_list=self.class_ftr_dict[self.class_define[tooth]]
            if self.need_target:
                augment_list = []
                for feature in feature_list:
                    # feature is a dict define the feature
                    f_array = self.load_y(
                        tooth_dir + feature['label_file'],
                        feature['num_feature'],
                        len(feature['feature_need'])
                    )
                    augment_list.append(f_array)
                # [aug_num,feature_dim]
                tooth_array = np.concatenate(augment_list, axis=1)
                y_class_set[self.class_define[tooth]].append(tooth_array)
                
    def load_case_list(self, case_load):
        # 读取多个病例
        self.box = None
        
        if self.need_name:
            self.case_load = np.array(case_load)
        for i, case_name in enumerate(case_load):
            full_case_dir = self.total_case_dir + case_name + '/'
            
            
            #[4,tooth_num_class,array[aug_num_tooth,data_size]]
            # box_class_set=[   [array1(aug_num_tooth,data_size)] x tooth_num,   [],[],[]]
            box_class_set=[[],[],[],[]]
            y_class_set=[[],[],[],[]]
            
            self.load_useful_tooth(full_case_dir, self.tooth_list, box_class_set,y_class_set)
            for i,class_box in enumerate(box_class_set):
                # [tooth_num_class*aug_num_tooth,data_size]
                box_class_case=np.concatenate(class_box)
                #[   [array1(tooth_num_class*aug_num_tooth,data_size)]    x case_num,  [][][]]
                self.box_class_list[i].append(box_class_case)
            
            for i,class_y in enumerate(y_class_set):
                # [tooth_num_class*aug_num_tooth,data_size]
                y_class_case=np.concatenate(class_y)
                #[   [array1(tooth_num_class*aug_num_tooth,data_size)]    x case_num,  [][][]]
                self.y_class_list[i].append(y_class_case)
            
    
    def load_y(self, info_file, num_feature, num_feature_point):
        # print('file_dir = ',info_file,'\n')
        info = np.reshape(np.loadtxt(info_file), [-1, 3 * (num_feature + 1)])
        origin = np.reshape(info[:, :3], [-1, 3])
        # [batch_size, 3*num_feature_point]
        origin = np.tile(origin, [1,num_feature_point])
        
        # [batch_size, 3*num_feature_point]
        feature = info[:, 3:]
        feature = ((feature - origin) * self.world_to_cubic).astype(np.int32)
        # [batch_size, 3*num_feature_point]
        return feature
    
    def load_mhd(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)
        if self.down_rate>1:
            ct_scan=ct_scan[::self.down_rate, ::self.down_rate,::self.down_rate]

        return ct_scan
    
    def loadmhds(self, collection_path):
        '''
        :param collection_path: train或test路径，路径下包含多个病例
        :return: [b,32,32,32]
        '''
        
        box_list = []
        if not os.path.exists(collection_path):
            return None
        for fileName in os.listdir(collection_path):
            if os.path.splitext(fileName)[1] == '.mhd':
                toothPath = os.path.join(collection_path, fileName)
                box_list.append(self.load_mhd(toothPath))
        box = np.stack(box_list)
        box.shape = [-1] + list(box.shape[-3:])
        return box
    
    def suffle(self):
        perm = np.arange(self.sample_num)
        np.random.shuffle(perm)  # 打乱
        self.box = self.box[perm]
        self.class_ = self.class_[perm]
        if self.need_name:
            self.name_index = self.name_index[perm]
        if self.need_target:
            self.y = self.y[perm]
            
    def reshape_class_data(self):
        # [   [array1(tooth_num_class*aug_num_tooth,data_size)]    x case_num,  [][][]]
        for i,box_class,y_class in enumerate(zip(self.box_class_list, self.y_class_list)):
            # i is in range(num of case_load_once)
            # [case_num * tooth_num_class*aug_num_tooth, data_size]
            concat_box_class=np.concatenate(box_class)
            # [case_num * tooth_num_class*aug_num_tooth, data_size]
            concat_y_class =np.concatenate(y_class)
            class_dict={'box':concat_box_class,'y':concat_y_class}
            self.class_box_y_dict[i]=class_dict
        self.box_class_list=None
        self.y_class_list=None
        
        
    def get_batch(self,class_):
        box_y_dict=self.class_box_y_dict[class_]
        
        
    def get_batch(self):
        if self.index + self.batch_size > self.sample_num:
            self.index = 0
            self.shuffle_times += 1
            if self.load_case_once > 0 and self.shuffle_times >= self.switch_after_shuffles:
                print('load data for ' + self.usage)
                self.load_case_list(self.get_case_list())
                self.shuffle_times = 0
            self.suffle()
        
        box_batch = np.expand_dims(self.box[self.index:   self.index + self.batch_size].copy(), 4)
        return_list = [box_batch]
        
        if self.need_target:
            y_batch = self.y[self.index:   self.index + self.batch_size]
            mask_batch = np.ones_like(y_batch, dtype=bool)
            class_batch = self.class_[self.index:   self.index + self.batch_size]
            for i, class_num in enumerate(class_batch.tolist()):
                if class_num > 1:
                    mask_batch[i, 15:] = False

            target_list=[y_batch,mask_batch,class_batch]
            return_list+=target_list
            
        if self.need_name:
            name_index_batch = self.name_index[self.index:   self.index + self.batch_size]
            return_list+=[self.case_load[name_index_batch]]

        self.index += self.batch_size

        return return_list


if __name__ == '__main__':
    gen = BatchGenerator(TrainDataConfig)
    pass
    # while True:
    #     box, class_, y = gen.get_batch()
    #     mask = np.ones_like(y, dtype=bool)
    #     for i, class_num in enumerate(class_.tolist()):
    #         if class_num > 1:
    #             mask[i, 15:] = False
    #
    #     pass

