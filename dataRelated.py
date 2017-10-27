import numpy as np
import SimpleITK as sitk
import os
import pickle
import random


class BatchGenerator(object):
    def __init__(self,data_config):

        self.shape_box = data_config.shape_box
        self.box_train=None
        self.total_case_dir=data_config.total_case_dir
        self.case_dir_list=os.listdir(self.total_case_dir)
        self.total_case_num=len(self.case_dir_list)
        self.load_case_once=data_config.load_case_once
        self.switch_after_shuffles=data_config.switch_after_shuffles
        self.world_to_cubic=data_config.world_to_cubic
        self.batch_size=data_config.batch_size
        self.index=0
        self.shuffle_times=0
        self.load_case()

    def load_case(self):
        print('reload data')
        sample_dir_list = random.sample(self.case_dir_list, self.load_case_once)
        box_list=[]
        y_list=[]
        for case_dir in sample_dir_list:
            full_case_dir=self.total_case_dir+'\\'+case_dir
            box_list.append(self.loadmhds(full_case_dir))
            y_list.append(self.load_y(full_case_dir+'\\info.txt'))

        self.box=np.expand_dims(np.concatenate(box_list,axis=0),axis=4)
        self.y=np.concatenate(y_list,axis=0)
        self.sample_num=self.y.shape[0]
        assert self.batch_size <= self.sample_num, 'batch_size should be smaller than sample_num'
        self.suffle()

    def load_y(self, info_file):
        info = np.reshape(np.loadtxt(info_file), [-1, 9])

        origin = np.reshape(info[:, :3], [-1, 3])
        origin = np.reshape(np.tile(origin, np.array([2])), [-1, 3])
        target = np.reshape(info[:, 3:], [-1, 3])
        target = np.reshape((target - origin) * self.world_to_cubic, [-1, 6]).astype(np.int32)
        return target

    def load_itk(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        return ct_scan

    def loadpickles(self, collection_path):
        box_list = []
        for fileName in os.listdir(collection_path):
            if os.path.splitext(fileName)[1] == '.pickle':
                toothPath = os.path.join(collection_path, fileName)
                box_list.append(self.loadpickle(toothPath))
        box = np.stack(box_list)
        box.shape = [-1] + self.shape_box
        return box

    def loadpickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def savePickle(self, data, path):
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def loadmhds(self, collection_path):
        '''
        :param collection_path: train或test路径，路径下包含多个病例
        :return: [b,32,32,32]
        '''

        box_list = []
        for fileName in os.listdir(collection_path):
            if os.path.splitext(fileName)[1] == '.mhd':
                toothPath = os.path.join(collection_path, fileName)
                box_list.append(self.load_itk(toothPath))

        box = np.stack(box_list)
        box.shape = [-1] + self.shape_box
        return box

    def suffle(self):
        perm = np.arange(self.sample_num)
        np.random.shuffle(perm)  # 打乱
        self.box = self.box[perm]
        self.y = self.y[perm]

    def get_batch(self):
        if self.index+ self.batch_size>self.sample_num:
            if self.shuffle_times > self.switch_after_shuffles:
                self.load_case()
                self.shuffle_times = 0
            self.index=0
            self.shuffle_times+=1
            self.suffle()
            # Start next epoch
            assert self.batch_size <= self.sample_num,'batch_size should be smaller than sample_num'


        box_batch= self.box[self.index:   self.index + self.batch_size]
        y_batch= self.y[self.index:   self.index + self.batch_size]
        self.index+=self.batch_size

        return box_batch,y_batch




if __name__ == '__main__':

    class DataConfig(object):
        shape_box = [128, 128, 128]
        shape_crop = [64, 64, 64]
        world_to_cubic = 128 / 12.
        batch_size_train = 2
        batch_size_test = 1
        total_case_dir = 'F:\\ProjectData\\Feature\\Tooth'
        each_load_num = 10  # 每次读的病例数
        switch_data_internal = 10  # 当前数据洗牌n次读取新数据
        need_Save = False
        need_Restore = False
        format = 'mhd'


