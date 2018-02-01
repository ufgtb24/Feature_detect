import numpy as np
import SimpleITK as sitk
import os



class BatchGenerator(object):
    def __init__(self,data_config,need_target=True,need_name=False):

        self.box_train=None
        self.need_target=need_target
        self.need_name=need_name
        self.total_case_dir=data_config.total_case_dir
        self.total_case_list=self.get_total_case_list()
        self.total_case_num=len(self.total_case_list)
        self.load_case_once=data_config.load_case_once
        assert self.load_case_once <= self.total_case_num,\
            'load_case_once should be smaller than total_case_num'

        self.switch_after_shuffles=data_config.switch_after_shuffles
        self.world_to_cubic=data_config.world_to_cubic
        self.batch_size=data_config.batch_size
        self.data_list=data_config.data_list
        self.index=0
        self.index_dir=0
        self.shuffle_times=0
        if self.load_case_once>0:
            self.load_case_list(self.get_case_list())
        else:
            self.load_case_list(self.total_case_list)
        self.suffle()

    def get_total_case_list(self):
        return os.listdir(self.total_case_dir)


    def get_case_list(self):
        if self.index_dir + self.load_case_once > self.total_case_num:
            self.index_dir = 0
            perm = np.arange(self.total_case_num)
            np.random.shuffle(perm)  # 打乱
            array=np.array(self.total_case_list)
            array = array[perm]
            self.total_case_list = list(array)

        case_load = self.total_case_list[self.index_dir:self.index_dir + self.load_case_once]
        self.index_dir += self.load_case_once
        return case_load

    def load_single_side(self,full_case_dir,tooth_list):
        # 读取一个病例中的多颗牙齿
        box_list=[]
        y_list=[]
        y=None
        for tooth in tooth_list:
            tooth_dir=full_case_dir+'\\'+tooth
            box_list.append(self.loadmhds(tooth_dir))
            if self.need_target:
                y_list.append(self.load_y(tooth_dir+'\\info.txt'))
        box=np.concatenate(box_list,axis=0)
        if self.need_target:
            y=np.concatenate(y_list,axis=0)
        return box,y

    def load_case_list(self,case_load):
        # 读取多个病例
        print('load data')
        box_list=[]
        y_list=[]
        name_index_list=[]
        if self.need_name:
            self.case_load=np.array(case_load)
        for case_name,i in zip(case_load,range(len(case_load))):
            full_case_dir=self.total_case_dir+'\\'+case_name
            box,y=self.load_single_side(full_case_dir,self.data_list)
            box_list.append(box)
            if self.need_name:
                name_index=np.ones((box.shape[0]),dtype=np.int32)*i
                name_index_list.append(name_index)
            if self.need_target:
                y_list.append(y)

        self.box=np.concatenate(box_list,axis=0)
        if self.need_name:
            self.name_index=np.concatenate(name_index_list,axis=0)
        if self.need_target:
            self.y=np.concatenate(y_list,axis=0)
        self.sample_num=self.box.shape[0]
        assert self.batch_size <= self.sample_num, 'batch_size should be smaller than sample_num'

    # def load_case_list(self,case_load):
    # # 一个病例只包含一颗牙
    #     print('load data')
    #     box_list=[]
    #     y_list=[]
    #     for case_dir in case_load:
    #         full_case_dir=self.total_case_dir+'\\'+case_dir
    #         box_list.append(self.loadmhds(full_case_dir))
    #         if self.need_target:
    #             y_list.append(self.load_y(full_case_dir+'\\info.txt'))
    #
    #     self.box=np.concatenate(box_list,axis=0)
    #     if self.need_target:
    #         self.y=np.concatenate(y_list,axis=0)
    #     self.sample_num=self.box.shape[0]
    #     assert self.batch_size <= self.sample_num, 'batch_size should be smaller than sample_num'

    def load_y(self, info_file):
        info = np.reshape(np.loadtxt(info_file), [-1, 9])
        origin = np.reshape(info[:, :3], [-1, 3])
        origin = np.reshape(np.tile(origin, np.array([2])), [-1, 3])
        target = np.reshape(info[:, 3:], [-1, 3])
        target = np.reshape((target - origin) * self.world_to_cubic, [-1, 6]).astype(np.int32)
        return target

    def load_mhd(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        return ct_scan

    def loadmhds(self, collection_path):
        '''
        :param collection_path: train或test路径，路径下包含多个病例
        :return: [b,32,32,32]
        '''

        box_list = []
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
        if self.need_name:
            self.name_index=self.name_index[perm]
        if self.need_target:
            self.y = self.y[perm]

    def get_batch(self):
        if self.index+ self.batch_size>self.sample_num:
            self.index=0
            self.shuffle_times+=1
            if self.load_case_once>0 and self.shuffle_times >= self.switch_after_shuffles:
                self.load_case_list(self.get_case_list())
                self.shuffle_times = 0
            self.suffle()


        box_batch= self.box[self.index:   self.index + self.batch_size].copy()
        if self.need_name:
            global name_index_batch
            name_index_batch= self.name_index[self.index:   self.index + self.batch_size]
        if self.need_target:
            global y_batch
            y_batch= self.y[self.index:   self.index + self.batch_size]
        self.index+=self.batch_size
        return_list=[box_batch]
        if self.need_target:
            return_list.append(y_batch)
        if self.need_name:
            return_list.append(self.case_load[name_index_batch])

        return return_list





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


