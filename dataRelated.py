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
            
            self.task_dict = final_task_dict
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
        self.data_list = data_config.data_list
        self.index = 0
        self.index_dir = 0
        self.shuffle_times = 0
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
    
    def load_useful_tooth(self, full_case_dir, target_tooth_list):
        # 读取一个病例中的多颗牙齿
        box_list = []
        case_list = []
        mask_list=[]
        class_list = []
        case_y = None
        case_mask=None
        actual_tooth_list = os.listdir(full_case_dir)
        
        
        for tooth in target_tooth_list:
            if tooth not in actual_tooth_list:
                continue
            tooth_dir = full_case_dir + tooth + '/'
            box_case_tooth=self.loadmhds(tooth_dir)
            augment_num=box_case_tooth.shape[0]
            box_list.append(box_case_tooth)
            if self.need_target:
                augment_list = []
                augment_mask_list = []
                
                for task_content in self.task_dict.values():
                    if not os.path.exists(tooth_dir + task_content['label_file']):
                        f_array = np.zeros([augment_num, task_content['num_feature'] * 3])
                        mask_array=np.zeros_like(f_array).astype(np.bool)
                    else:
                        f_array = self.load_y(
                            tooth_dir + task_content['label_file'],
                            task_content['num_feature'],
                            len(task_content['feature_need']),
                            task_content['index']
                        )
                        mask_array=np.ones_like(f_array).astype(np.bool)

                    augment_list.append(f_array)
                    augment_mask_list.append(mask_array)
                
                tooth_array = np.concatenate(augment_list, axis=1)
                tooth_mask_array = np.concatenate(augment_mask_list, axis=1)
                case_list.append(tooth_array)
                mask_list.append(tooth_mask_array)
                
            class_array = self.class_define[tooth] * np.ones([augment_num])
            class_list.append(class_array)
        if box_list != []:
            case_box = np.concatenate(box_list, axis=0)
            case_class = np.concatenate(class_list, axis=0)
            if self.need_target:
                case_y = np.concatenate(case_list, axis=0)
                case_mask = np.concatenate(mask_list, axis=0)
                
            return case_box, case_y, case_mask,case_class
        else:
            return None
    
    def load_case_list(self, case_load):
        # 读取多个病例
        self.box = None
        
        box_list = []
        y_list = []
        mask_list = []
        class_list = []
        name_index_list = []
        if self.need_name:
            self.case_load = np.array(case_load)
        for i, case_name in enumerate(case_load):
            full_case_dir = self.total_case_dir + case_name + '/'
            load_result = self.load_useful_tooth(full_case_dir, self.data_list)
            if load_result is not None:
                box, y,mask, class_ = load_result
            else:
                continue
            box_list.append(box)
            class_list.append(class_)
            if self.need_name:
                name_index = np.ones((box.shape[0]), dtype=np.int32) * i
                name_index_list.append(name_index)
            if self.need_target:
                y_list.append(y)
                mask_list.append(mask)
        
        if box_list != None:
            self.box = np.concatenate(box_list, axis=0)
            self.class_ = np.concatenate(class_list, axis=0)
            if self.need_name:
                self.name_index = np.concatenate(name_index_list, axis=0)
            if self.need_target:
                self.y = np.concatenate(y_list, axis=0)
                self.mask = np.concatenate(mask_list, axis=0)
            self.sample_num = self.box.shape[0]
            assert self.batch_size <= self.sample_num, 'batch_size should be smaller than sample_num'
    
    def load_y(self, info_file, num_feature, num_feature_need, info_index):
        # print('file_dir = ',info_file,'\n')
        info = np.reshape(np.loadtxt(info_file), [-1, 3 * (num_feature + 1)])
        origin = np.reshape(info[:, :3], [-1, 3])
        origin = np.reshape(np.tile(origin, num_feature_need), [-1, 3])
        
        info_need = info[:, info_index]
        feature = np.reshape(info_need, [-1, 3])
        feature = np.reshape((feature - origin) * self.world_to_cubic, [-1, 3 * num_feature_need]).astype(np.int32)
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
            mask_batch = self.mask[self.index:   self.index + self.batch_size]
            class_batch = self.class_[self.index:   self.index + self.batch_size]
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

