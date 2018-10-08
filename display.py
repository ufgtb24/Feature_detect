import SimpleITK as sitk
import numpy as np
from mayavi import mlab
import pickle
import os

from config import DataConfig, TrainDataConfig, TestDataConfig, BOX_LEN, ValiDataConfig
from dataRelated import BatchGenerator

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''


def generate_edge(a1, a2, a3, len_):
    def axis(a):
        if a == 0:
            return np.zeros([len_], dtype=np.int32)
        elif a == 1:
            return np.ones([len_], dtype=np.int32) * (len_ - 1)
        else:
            return np.arange(len_)

    return np.stack([axis(a1), axis(a2), axis(a3)])


def edges(len_):
    e_list = []
    e_list.append(generate_edge(0, 0, 2, len_))
    e_list.append(generate_edge(0, 1, 2, len_))
    e_list.append(generate_edge(1, 1, 2, len_))
    e_list.append(generate_edge(1, 0, 2, len_))

    e_list.append(generate_edge(0, 2, 0, len_))
    e_list.append(generate_edge(0, 2, 1, len_))
    e_list.append(generate_edge(1, 2, 1, len_))
    e_list.append(generate_edge(1, 2, 0, len_))

    e_list.append(generate_edge(2, 0, 0, len_))
    e_list.append(generate_edge(2, 0, 1, len_))
    e_list.append(generate_edge(2, 1, 1, len_))
    e_list.append(generate_edge(2, 1, 0, len_))

    e = np.concatenate(e_list, axis=1)  # 3,32*12
    ex, ey, ez = np.split(e, 3, axis=0)
    return ex, ey, ez


def loadpickles(collection_path):
    box_list = []
    for fileName in os.listdir(collection_path):
        if os.path.splitext(fileName)[1] == '.pkl':
            toothPath = os.path.join(collection_path, fileName)
            box_list.append(loadpickle(toothPath))
    box = np.concatenate(box_list)
    return box


def loadtxts(collection_path):
    txt_list = []
    for fileName in os.listdir(collection_path):
        if os.path.splitext(fileName)[1] == '.txt':
            toothPath = os.path.join(collection_path, fileName)
            txt_list.append(np.loadtxt(toothPath))
    txt = np.concatenate(txt_list)
    return txt


def loadpickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_y(info_file,feature_num):
    info = np.reshape(np.loadtxt(info_file), [-1, 3*(feature_num+1)])
    origin = info[:, :3]

    origin = np.tile(origin, np.array([feature_num]))
    target = info[:, 3:]
    valide=True
    if 0 in target:
        valide=False
    target = ((target - origin) * 128/12.).astype(int)
    return target,valide


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    # origin = np.array(list(reversed(itkimage.GetOrigin())))
    #
    # # Read the spacing along each dimension
    # spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan


def loadmhd(collection_path):
    '''
    :param collection_path: train或test路径，路径下包含多个病例
    :return: [b,32,32,32]
    '''

    box_list = []
    for fileName in os.listdir(collection_path):
        if os.path.splitext(fileName)[1] == '.mhd':
            toothPath = os.path.join(collection_path, fileName)
            box_list.append(load_itk(toothPath))

    box = np.stack(box_list)
    box.shape = [-1, BOX_LEN, BOX_LEN, BOX_LEN]
    return box


def show_single(dir,req_index):
    cur=0
    show_name=None
    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.mhd':
            if req_index==cur:
                show_name=fileName
            cur+=1
            
    ct = load_itk(os.path.join(dir, show_name))
    info_file = os.path.join(dir, 'edge.txt')

    feature,non_zero = load_y(info_file,2)
    
    x1, x2, x3 = np.where(ct == 1)
    mlab.points3d(x1, x2, x3,
                  mode="cube",
                  color=(0, 1, 0),
                  scale_factor=1,
                  transparent=True)
    
    ex, ey, ez = edges(BOX_LEN)

    mlab.points3d(ex+100, ey, ez,
                  mode="cube",
                  color=(0, 0, 1),
                  scale_factor=1)

    
    colors = [(1, 0, 0), (0, 0, 1), (0, 0, 0), (0.5, 0.5, 0.5)]

    mlab.points3d(feature[req_index,0], feature[req_index,1], feature[req_index,2],
                  mode="cube",
                  color=colors[0],
                  scale_factor=1,
                  transparent=False)
    mlab.points3d(feature[req_index,3], feature[req_index,4], feature[req_index,5],
                  mode="cube",
                  color=colors[1],
                  scale_factor=1,
                  transparent=False)

    mlab.show()

def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir + '\\' + case_name

        valide_case=True
        for tooth in os.listdir(full_case_dir):
            case_tooth_dir = full_case_dir + '\\' + tooth
            info_file=case_tooth_dir+"\\info.txt"

            feature,valide_tooth = load_y(info_file, BOX_LEN / WORLD_SIZE)
            if valide_tooth==False:
                valide_case=False
                break

        if valide_case==False:
            print(full_case_dir)
            # for i in range(num):
            #     if feature[i, 0]<feature[i, 3]:
            #         error_num+=1
            #         print('data_error in %s'%(case_tooth_dir))

    return error_num

def display_batch(box, y, mask,name=None):
    box = np.squeeze(box, 4)
    num=box.shape[0]
    ex, ey, ez = edges(BOX_LEN)

    y=y.astype(int)
    for i in range(num):
        # mlab.points3d(ex , ey, ez,
        #               mode="cube",
        #               color=(0, 0, 1),
        #               scale_factor=1)
    
        if name is not None:
            print(name[i])
        ct = box[i]
        single_y=y[i,mask[i]]
        single_mask=np.where(mask[i])[0]
        feature_need = int(single_mask.shape[0]/3)
        # print(class_batch[i])
        x1, x2, x3 = np.where(ct == 1)
        mlab.points3d(x1, x2, x3,
                      mode="cube",
                      color=(0, 1, 0),
                      scale_factor=1,
                      transparent=False)
        colors=[(1,0,0),(0,0,1)]+[(0,0,0)]*10
        # 上牙是左蓝右红，下牙是左红右蓝
        for j in range(feature_need):
            # try:
            #     ct[single_y[3*j], single_y[3*j+1], single_y[3*j+2]]=j+2
            # except:
            #     print('out bound')
            #     continue
            # feature_index.append(np.where(ct == j+2))
            
            # print('%d  %d  %d  '%(single_y[3*j], single_y[3*j+1], single_y[3*j+2]))
            mlab.points3d(single_y[3*j], single_y[3*j+1], single_y[3*j+2],
                          mode="cube",
                          color=colors[j],
                          scale_factor=1,
                          transparent=False)
        mlab.show()



def traverse_dir(dir):
    # 读取世界坐标
    box = loadmhd(dir)
    info_file = os.path.join(dir, 'FaccControlPts.txt')
    feature_need=5
    # feature ,_= load_y(info_file, GRID_SIZE / WORLD_SIZE)
    y ,_= load_y(info_file, feature_need)
    num = y.shape[0]
    
    ex, ey, ez = edges(BOX_LEN)

    for i in range(num):
        mlab.points3d(ex , ey, ez,
                      mode="cube",
                      color=(0, 0, 1),
                      scale_factor=1)
        ct = box[i]
        single_y = y[i]
        # print(class_batch[i])
        x1, x2, x3 = np.where(ct == 1)
        mlab.points3d(x1, x2, x3,
                      mode="cube",
                      color=(0, 1, 0),
                      scale_factor=1,
                      transparent=False)
        colors = [(1, 0, 0), (0, 0, 1)] + [(0, 0, 0)] * 10
        # 上牙是左蓝右红，下牙是左红右蓝
        for j in range(feature_need):
            mlab.points3d(single_y[3 * j], single_y[3 * j + 1], single_y[3 * j + 2],
                          mode="cube",
                          color=colors[j],
                          scale_factor=1,
                          transparent=False)
        mlab.show()


def traverse_croped(dir):
    # 读取网格坐标
    box = loadpickles(dir)
    center_points = loadtxts(dir)
    i = 0
    while True:
        fx = np.array([center_points[i, 0]])
        fy = np.array([center_points[i, 1]])
        fz = np.array([center_points[i, 2]])
        ct = box[i]
        ex, ey, ez = edges(BOX_LEN)
        x, y, z = np.where(ct == 1)

        mlab.points3d(x, y, z,
                      mode="cube",
                      color=(0, 1, 0),
                      scale_factor=1,
                      transparent=True)

        mlab.points3d(ex, ey, ez,
                      mode="cube",
                      color=(0, 0, 1),
                      scale_factor=1)
        mlab.points3d(fx, fy, fz,
                      mode="cube",
                      color=(1, 0, 0),
                      scale_factor=1,
                      transparent=True)

        mlab.show()
        i += 1


WORLD_SIZE = 12.0

if __name__ == '__main__':
    traverse_dir('F:\\ProjectData\\tmp\\Try\\Validate\\0828 AlbertDiaz-Conti\\tooth4\\')
    
    # train_batch_gen = BatchGenerator(TestDataConfig,need_name=True)
    # for i in range(1000):
    #     return_dict=train_batch_gen.get_batch()
    #     display_batch(return_dict['box'], return_dict['y'], return_dict['mask'],return_dict['name'])
    #
    # show_single('F:/ProjectData/tmp/Train/1213 11294836_mirror\\tooth8',6)
