import numpy as np
import SimpleITK as sitk
import os
import pickle



class BatchGenerator(object):
    def __init__(self,batch_size,box,y):
        self.index=0
        self.box=box
        self.y=y

        self.batch_size=batch_size
        self.sample_num=box.shape[0]




    def get_batch(self):
        if self.index+ self.batch_size>self.sample_num:
            self.index=0
            perm = np.arange(self.sample_num)
            np.random.shuffle(perm)  # 打乱
            self.box = self.box[perm]
            self.y = self.y[perm]
            # Start next epoch
            assert self.batch_size <= self.sample_num,'batch_size should be smaller than sample_num'


        box_batch= self.box[self.index:   self.index + self.batch_size]
        y_batch= self.y[self.index:   self.index + self.batch_size]
        self.index+=self.batch_size
        return box_batch,y_batch

BOX_LEN=32

class DataManager(object):
    def __init__(self, data_config,
                 test_box_path=None, test_info_file=None,
                 train_box_path=None, train_info_file=None
                 ):
        self.shape_box = data_config.shape_box
        self.box_train=None
        need_train=False

        if train_box_path!=None:
            need_train=True
            if data_config.format=='pickle':
                self.box_train=self.loadpickles(train_box_path)
            elif data_config.format=='mhd':
                self.box_train=self.loadmhds(train_box_path)
            self.y_train=self.load_y(train_info_file, data_config.world_to_cubic)


        if data_config.format == 'pickle':
            self.box_test = self.loadpickles(test_box_path)
        elif data_config.format == 'mhd':
            self.box_test = self.loadmhds(test_box_path)
        self.y_test = self.load_y(test_info_file, data_config.world_to_cubic)

        if need_train:
            self.__trainGenerator = BatchGenerator(batch_size=data_config.batch_size_train,box=self.box_train,y= self.y_train)
        self.__testGenerator = BatchGenerator(batch_size=data_config.batch_size_test,box=self.box_test, y=self.y_test )




    def load_y(self, info_file, world_to_cubic):
        info=np.loadtxt(info_file)
        info.shape=[-1,9]
        origin=info[:,:3]
        target=info[:,3:]
        x_index=[0,3]
        y_index=[1,4]
        z_index=[2,5]
        target[:, x_index]=(target[:,x_index]-origin[:,0][:,np.newaxis])*world_to_cubic
        target[:, y_index]=(target[:,y_index]-origin[:,1][:,np.newaxis])*world_to_cubic
        target[:, z_index]=(target[:,z_index]-origin[:,2][:,np.newaxis])*world_to_cubic
        target=target.astype(np.int32)
        return target


    def load_itk(self,filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        return ct_scan


    def loadpickles(self,collection_path):
        box_list = []
        for fileName in os.listdir(collection_path):
            if os.path.splitext(fileName)[1] == '.pickle':
                toothPath = os.path.join(collection_path, fileName)
                box_list.append(self.loadpickle(toothPath))
        box = np.stack(box_list)
        box.shape=[-1]+self.shape_box
        return box

    def loadpickle(self,path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data


    def savePickle(self, data,path):
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
        box.shape=[-1]+self.shape_box
        return box


    def getTrainBatch(self):
        return self.__trainGenerator.get_batch()

    def getTestBatch(self):
        return self.__testGenerator.get_batch()


if __name__ == '__main__':

    class DataConfig(object):
        shape_box = [128, 128, 128]
        shape_crop = [64, 64, 64]
        world_to_cubic = 0.1
        batch_size_train = 2
        batch_size_test = 1
        need_Save = False
        need_Restore = False
        format = 'mhd'


    root='F:\ProjectData\Retouch\\root'
    DataManager(DataConfig,
                train_box_path=os.path.join(root,'train'),
                train_info_file=os.path.join(root,'train','info.txt'),
                test_box_path=os.path.join(root,'test'),
                test_info_file=os.path.join(root,'test','info.txt')
                )
