import numpy as np
import os

def switch_line(dir):
    a=np.loadtxt(dir)
    a[:,[0,2,3,5,6,8]]=a[:,[2,0,5,3,8,6]]
    np.savetxt(dir,a,fmt="%.5f")


def switch_line_folder(case_dir):
    for case_name in os.listdir(case_dir):
        case_name = case_dir + '\\' + case_name
        for tooth_name in os.listdir(case_name):
            tooth_name = case_name + '\\' + tooth_name
            if os.path.isdir(tooth_name):
                switch_line(os.path.join(tooth_name,'info.txt'))




if __name__=='__main__':
    switch_line_folder('F:\\ProjectData\\Feature2\\predict')
    # switch_line('F:\\ProjectData\\Feature2\\crop_test\\tooth2/info.txt')