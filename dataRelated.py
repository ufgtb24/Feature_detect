import numpy as np
import SimpleITK as sitk
import os

class BatchGenerator(object):
    def __init__(self,box,info,batch_size):
        self.index=0
        self.box=box
        self.info=info
        self.batch_size=batch_size
        self.sample_num=box.shape[0]




    def get_batch(self):
        if self.index+ self.batch_size>self.sample_num:
            self.index=0
            perm = np.arange(self.sample_num)
            np.random.shuffle(perm)  # 打乱
            self.box = self.box[perm]
            self.info = self.info[perm]
            # Start next epoch
            assert self.batch_size <= self.sample_num,'batch_size should be smaller than sample_num'


        box_batch= self.box[self.index:   self.index + self.batch_size]
        info_batch= self.info[self.index:   self.index + self.batch_size]
        self.index+=self.batch_size
        return box_batch,info_batch


BOX_LEN=32

class DataManager(object):
    def __init__(self, data_config,
                 train_box_path, train_info_file,
                 test_box_path, test_info_file,
                 ):
        self.box_w=data_config.w
        self.box_h=data_config.h
        self.box_d=data_config.d

        self.box_train=self.loadbox(train_box_path)
        self.box_test=self.loadbox(test_box_path)

        self.y_train=self.load_local_coord(train_info_file, data_config.world_to_cubic)
        self.y_test=self.load_local_coord(test_info_file, data_config.world_to_cubic)

        self.__trainGenerator = BatchGenerator(self.box_train, self.y_train, data_config.batch_size_train)
        self.__testGenerator = BatchGenerator(self.box_test, self.y_test, data_config.batch_size_test)

    def load_local_coord(self, info_file, world_to_cubic):
        info=np.loadtxt(info_file)
        origin=info[:,:3]
        target=info[:,3:]
        x_index=[0,3,6,9]
        y_index=[1,4,7,10]
        z_index=[2,5,8,11]
        target[:, x_index]=(target[:,x_index]-origin[:,0][:,np.newaxis])*world_to_cubic
        target[:, y_index]=(target[:,y_index]-origin[:,1][:,np.newaxis])*world_to_cubic
        target[:, z_index]=(target[:,z_index]-origin[:,2][:,np.newaxis])*world_to_cubic
        return target


    def load_itk(self,filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        return ct_scan


    def loadbox(self, collection_path):
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
        box.shape=[-1,self.box_w,self.box_h,self.box_d,1]
        return box





    def getTrainBatch(self):
        return self.__trainGenerator.get_batch()

    def getTestBatch(self):
        return self.__testGenerator.get_batch()


if __name__ == '__main__':
    class DataConfig(object):
        w = 128
        h = 128
        d = 128
        y_size = 8
        world_to_cubic=128/20
        batch_size_train = 2
        batch_size_test = 1
        need_Save = False
        need_Restore = False



    root='F:\ProjectData\Retouch\\root'
    DataManager(DataConfig(),
                train_box_path=os.path.join(root,'train'),
                train_info_file=os.path.join(root,'train','info.txt'),
                test_box_path=os.path.join(root,'test'),
                test_info_file=os.path.join(root,'test','info.txt')
                )
