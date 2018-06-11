import os
import numpy as np

# label file column number check
'''
def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        
        for tooth in tooth_list:
            facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
            groove_dir=full_case_dir+'/'+tooth+'/info.txt'
            if os.path.exists(facc_dir):
                with open(facc_dir, 'rb') as f:
                    line =f.readline()  # read 1 line
                    n = len(line.split())
                    if n!=18:
                        print(facc_dir)
                    
            if os.path.exists(groove_dir):
                with open(groove_dir, 'rb') as f:
                    line =f.readline()  # read 1 line
                    n = len(line.split())
                    if n!=9:
                        print(groove_dir)
    return error_num
'''

# label file line number check
'''
def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        for tooth in tooth_list:
            facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
            groove_dir=full_case_dir+'/'+tooth+'/info.txt'
            if os.path.exists(facc_dir):
                if len(open(facc_dir, 'rU').readlines())!=27:
                    print(case_name)
            if os.path.exists(groove_dir):
                if len(open(groove_dir, 'rU').readlines())!=27:
                    print(case_name)


            # if 'tooth'+str(i) not in data_list:
            #     print(case_name)

    return error_num
'''

# file name check
'''
def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        for tooth in tooth_list:
            full_tooth_dir=full_case_dir+'/'+tooth+'/'
            aug_list = os.listdir(full_tooth_dir)
            for file in aug_list:
                if os.path.splitext(file)[1] == '.mhd':
                    if os.path.splitext(file)[0][-1]=='9' or os.path.splitext(file)[0][-5]=='9':
                        print(case_name+'/'+tooth+'\n')

    return error_num
'''

# label file line number check
def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        for tooth in tooth_list:
            facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
            groove_dir=full_case_dir+'/'+tooth+'/info.txt'
            line_num=0
            if os.path.exists(facc_dir):
                line_num=len(open(facc_dir, 'rU').readlines())
            if os.path.exists(groove_dir):
                if len(open(groove_dir, 'rU').readlines())!=line_num:
                    print(case_name)


            # if 'tooth'+str(i) not in data_list:
            #     print(case_name)

    return error_num

if __name__ == '__main__':
    check_availability('F:/ProjectData/tmp/Train/')