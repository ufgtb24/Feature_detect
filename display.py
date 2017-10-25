import SimpleITK as sitk
import numpy as np
from mayavi import mlab
import pickle
import os
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def generate_edge(a1, a2, a3, len_):
    def axis(a):
        if a==0:
            return np.zeros([len_], dtype=np.int32)
        elif a==1:
            return np.ones([len_], dtype=np.int32) * (len_ - 1)
        else:
            return np.arange(len_)
    return np.stack([axis(a1),axis(a2),axis(a3)])

def edges(len_):
    e_list=[]
    e_list.append(generate_edge(0,0,2,len_))
    e_list.append(generate_edge(0,1,2,len_))
    e_list.append(generate_edge(1,1,2,len_))
    e_list.append(generate_edge(1,0,2,len_))

    e_list.append(generate_edge(0,2,0,len_))
    e_list.append(generate_edge(0,2,1,len_))
    e_list.append(generate_edge(1,2,1,len_))
    e_list.append(generate_edge(1,2,0,len_))

    e_list.append(generate_edge(2,0,0,len_))
    e_list.append(generate_edge(2,0,1,len_))
    e_list.append(generate_edge(2,1,1,len_))
    e_list.append(generate_edge(2,1,0,len_))

    e=np.concatenate(e_list,axis=1) #3,32*12
    ex,ey,ez=np.split(e,3,axis=0)
    return ex,ey,ez

def loadpickles(collection_path):
    box_list = []
    for fileName in os.listdir(collection_path):
        if os.path.splitext(fileName)[1] == '.pkl':
            toothPath = os.path.join(collection_path, fileName)
            box_list.append(loadpickle(toothPath))
    box=np.concatenate(box_list)
    return box

def loadtxts(collection_path):
    txt_list = []
    for fileName in os.listdir(collection_path):
        if os.path.splitext(fileName)[1] == '.txt':
            toothPath = os.path.join(collection_path, fileName)
            txt_list.append(np.loadtxt(toothPath))
    txt=np.concatenate(txt_list)
    return txt



def loadpickle( path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_y(info_file, world_to_cubic):
    info = np.loadtxt(info_file)
    origin=np.reshape(info[:3],[-1,3])
    target = np.reshape(info[3:],[-1,3])
    target=(target-origin)*world_to_cubic
    target =target.astype(np.int32)
    return target


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def show_single(dir):
    ct, ori, sp = load_itk(os.path.join(dir,'toothlabel4.mhd'))
    info_file=os.path.join(dir,'info.txt')
    
    feature = load_y(info_file, 128 / 20.0)

    ct[feature[0,0],feature[0,1],feature[0,2]]=2
    ct[feature[1,0],feature[1,1],feature[1,2]]=2
    fz, fy, fx = np.where(ct == 2)
    x, y, z = np.where(ct ==1)
    ex,ey,ez=edges(128)

    mlab.points3d(x, y, z,
                         mode="cube",
                         color=(0, 1, 0),
                         scale_factor=1,)
                  # transparent=True)
    mlab.points3d(ex, ey, ez,
                         mode="cube",
                         color=(0, 0, 1),
                         scale_factor=1)
    mlab.points3d(fx, fy, fz,
                         mode="cube",
                         color=(1, 0, 0),
                         scale_factor=1)
    
    mlab.show()



def traverse(dir):
    box=loadpickles(dir)
    center_points=loadtxts(dir)
    i=0
    while True:

        fx = np.array([center_points[i,0]])
        fy = np.array([center_points[i,1]])
        fz = np.array([center_points[i,2]])
        ct=box[i]
        ex, ey, ez = edges()
        x, y, z = np.where(ct == 255)

        mlab.points3d(x, y, z,
                      mode="cube",
                      color=(0, 1, 0),
                      scale_factor=1)
        mlab.points3d(ex, ey, ez,
                      mode="cube",
                      color=(0, 0, 1),
                      scale_factor=1)
        mlab.points3d(fx, fy, fz,
                             mode="cube",
                             color=(1, 0, 0),
                             scale_factor=1)
        mlab.show()
        i+=1

if __name__ == '__main__':
    # traverse('F:\\ProjectData\\Feature\\croped\\')
    show_single('F:\\ProjectData\\Feature\\new3\\')











