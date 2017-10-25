import tensorflow as tf
import os
import pickle

from math import pi, sin, cos
import numpy as np
import SimpleITK as sitk
import Transformer as tr
PATH=''

class Augment(object):
    def __init__(self, data_config,
                 box_dir, info_file, save_dir
                 ):
        self.shape_box=data_config.shape_box
        self.box=None
        self.save_dir=save_dir
        self.feature_size=data_config.feature_size
        if data_config.format=='pickle':
            self.box=self.loadpickle(box_dir).astype(np.float32)

        elif data_config.format=='mhd':
            self.box=self.loadmhd(box_dir).astype(np.float32)


        self.y=self.load_y(info_file, data_config.world_to_cubic)

        self.save(box=self.box, y=self.y,box_name='origin',pos_file_name='pos')



    def save(self, box, y, box_name,pos_file_name):
        self.savePickle(box, os.path.join(self.save_dir,box_name))

        with open(os.path.join(self.save_dir,pos_file_name), 'ab') as file:
            np.savetxt(file, y, fmt='%.6f')

    def do_augment(self,sess):
        #[b,6]
        # y=np.reshape(self.y,[-1,3])

        y_list=np.split(self.y,self.feature_size,axis=1)

        batch_num=self.y.shape[0]
        # point_num=y.shape[0] #2b
        t = np.zeros([batch_num, 3], dtype=np.float32)

        ones=np.ones((batch_num,1))
        #[point_num,4]
        y_list_split=[]
        for y in y_list:
            y=np.concatenate((y,ones),axis=1).astype(np.float32)
            # [point_num,4,1]
            y = y[:, :, np.newaxis]
            y_list_split.append(y)


        # #[point_num,4,1]
        # y=y[:,:,np.newaxis]



        i=0
        for theta in [-5.0,5.0]:
            for u_axis in [0,1,2]:
                i+=1
                u=np.zeros([batch_num,3],dtype=np.float32)
                u[:,u_axis]=1
                #[b,3,4]
                trans_matrix = tr.get_transformer_matrix(theta, u, t)
                #box.shape=[b,w,h,d]
                trans_box = tr.spatial_transformer_network(self.box, trans_matrix)

                y_list_trans=[]
                for y in y_list_split:
                    # [b,3,1]
                    trans_y = tf.matmul(trans_matrix, y)
                    # [b,3]
                    trans_y = tf.squeeze(trans_y, axis=[2])
                    y_list_trans.append(trans_y)

                trans_y=tf.concat(y_list_trans,axis=1)

                trans_box,trans_y=sess.run([trans_box,trans_y])
                blank=np.zeros(trans_box.shape)
                full=np.ones(trans_box.shape)
                box=np.where(trans_box.astype(np.bool),full,blank)


                self.save(
                    box,
                    trans_y,
                    'aug{0}'.format(i),
                    'pos'
                )






    def load_y(self, info_file, world_to_cubic):
        info=np.loadtxt(info_file).astype(np.float32)
        info.shape=[-1,9]
        origin=info[:,:3]
        target=info[:,3:]
        x_index=[0,3]
        y_index=[1,4]
        z_index=[2,5]
        target[:, x_index]=(target[:,x_index]-origin[:,0][:,np.newaxis])*world_to_cubic
        target[:, y_index]=(target[:,y_index]-origin[:,1][:,np.newaxis])*world_to_cubic
        target[:, z_index]=(target[:,z_index]-origin[:,2][:,np.newaxis])*world_to_cubic
        return target


    def savePickle(self, data,path):
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    def loadpickle(self,path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_itk(self,filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        return ct_scan

    def loadmhd(self, collection_path):
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

class DataConfig(object):
    shape_box=[128,128,128]
    format='mhd'
    world_to_cubic=128/20.
    feature_size=2

if __name__ == '__main__':

    root='F:\ProjectData\Feature'
    ag=Augment(DataConfig(),
            box_dir=os.path.join(root,'origin'),
            info_file=os.path.join(root,'origin','info.txt'),
            save_dir=os.path.join(root,'augment'),
                )
    with tf.Session() as sess:
        ag.do_augment(sess)