import os

# label file column number check
import shutil

# from display import load_y

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
pass
# def check_availability(dir):
#     Tooth_dir=os.listdir(dir)
#     error_num=0
#
#     for case_name in Tooth_dir:
#         full_case_dir = dir  + case_name
#         tooth_list=os.listdir(full_case_dir)
#         for tooth in tooth_list:
#             facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
#             groove_dir=full_case_dir+'/'+tooth+'/info.txt'
#             edge_dir=full_case_dir+'/'+tooth+'/edge.txt'
#             if os.path.exists(facc_dir):
#                 if len(open(facc_dir, 'rU').readlines())!=81:
#                     print(case_name,'  lack  facc ')
#
#             if os.path.exists(groove_dir):
#                 if len(open(groove_dir, 'rU').readlines())!=81:
#                     print(case_name,'  lack  groove ')
#
#             if os.path.exists(edge_dir):
#                 if len(open(edge_dir, 'rU').readlines())!=81:
#                     print(case_name,'  lack  edge ')
#
#
#             # if 'tooth'+str(i) not in data_list:
#             #     print(case_name)
#
#     return error_num

# label file content relation check
'''
def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        # tooth_list=os.listdir(full_case_dir)
        tooth_list=['tooth6','tooth7','tooth8']
        valide=True
        for tooth in tooth_list:
            # global valide
            valide=True
            edge_dir=full_case_dir+'/'+tooth+'/edge.txt'
            if os.path.exists(edge_dir):
                y,non_zero= load_y(edge_dir, 2)
                valide=non_zero
                for i in range(y.shape[0]):
                    if y[i][0]<y[i,3] :
                        valide = False
            if not valide:
                print(case_name,'   ',tooth)
                
        if not valide:
            error_num += 1

    print(error_num)
    return error_num

'''


# # label file line number identical and validate check
pass
def check_availability(dir):
    Tooth_dir=os.listdir(dir)

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        for tooth in tooth_list:
            valide=True
            facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
            groove_dir=full_case_dir+'/'+tooth+'/info.txt'
            edge_dir=full_case_dir+'/'+tooth+'/edge.txt'
            first_check=None
            if os.path.exists(facc_dir):
                f_num=len(open(facc_dir).readlines())
                first_check=f_num
                if f_num>81:
                    valide = False

            if os.path.exists(groove_dir):
                g_num =len(open(groove_dir).readlines())
                if first_check is None:
                    first_check=g_num
                elif g_num!=first_check:
                    valide = False

                if g_num>81:
                    valide = False


            if os.path.exists(edge_dir):
                e_num =len(open(edge_dir).readlines())

                if first_check is not None and e_num!=first_check:
                    valide = False

                if e_num>81:
                    valide = False


            if valide==False:
                print(case_name+'  '+tooth)
                
# # label text identical line check
pass
def check_identical_line(dir):
    Tooth_dir=os.listdir(dir)
    
    def check_unique(file_path):
        res_line=[]
        index=[]
        unique=True
        with open(file_path) as f:
            for i,line in enumerate(f.readlines()):
                if line in res_line:
                    unique=False
                    print('line %d and line %d is duplicated'%(index[res_line.index(line)]+1,i+1))
                else:
                    res_line.append(line)
                    index.append(i)
        return unique
    dup_num=0
    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        valid=True
        for tooth in tooth_list:
            facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
            groove_dir=full_case_dir+'/'+tooth+'/info.txt'
            edge_dir=full_case_dir+'/'+tooth+'/edge.txt'
            if os.path.exists(facc_dir):
                if not check_unique(facc_dir):
                    print('%s %s has duplicated facc\n'%(case_name,tooth))
                    valid=False
            if os.path.exists(groove_dir):
                if not check_unique(groove_dir):
                    print('%s %s has duplicated groove\n' % (case_name, tooth))
                    valid=False

            if os.path.exists(edge_dir):
                if not check_unique(edge_dir):
                    print('%s %s has duplicated edge\n'%(case_name,tooth))
                    valid=False
        if not valid:
            dup_num+=1
    print('dup_num = ',dup_num)
    
                

# # delete never used data
pass
def delete(dir):
    Tooth_dir=os.listdir(dir)

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        for tooth in tooth_list:
            if int(tooth[5:])>8 and int(tooth[5:])<25:
                shutil.rmtree(full_case_dir+'/'+tooth)
                
# # two dataset has identical case
pass
def check_same_case(validate,train):
    vali_list=os.listdir(validate)
    train_list=os.listdir(train)

    for case_name in vali_list:
        if case_name in train_list:
            print(case_name)



# file name check
pass
# def check_availability(dir):
#     Tooth_dir=os.listdir(dir)
#     error_num=0
#
#     for case_name in Tooth_dir:
#         full_case_dir = dir  + case_name
#         tooth_list=os.listdir(full_case_dir)
#         for tooth in tooth_list:
#             full_tooth_dir=full_case_dir+'/'+tooth+'/'
#             aug_list = os.listdir(full_tooth_dir)
#             for file in aug_list:
#                 if os.path.splitext(file)[1] == '.mhd':
#                     if os.path.splitext(file)[0][-1]=='9' or os.path.splitext(file)[0][-5]=='9':
#                         print(case_name+'/'+tooth+'\n')
#
#     return error_num

def check_unique(file_path):
    res_line = []
    index = []
    unique = True
    dup_num=0
    with open(file_path) as f:
        for i, line in enumerate(f.readlines()):
            if line in res_line:
                dup_num+=1
                unique = False
                print('line %d and line %d is duplicated' % (index[res_line.index(line)] + 1, i + 1))
            else:
                res_line.append(line)
                index.append(i)
    print(len(res_line))
    return unique

def record_case(dir):
    case_list=os.listdir(dir)
    with open('case_list.txt','w')as f:
        for line in case_list:
            f.write(line+'\n')

def filter_axis_rotate(tooth_dir,axis,oth_rot1,oth_rot2):
    
    facc_dir = tooth_dir + 'FaccControlPts.txt'
    groove_dir = tooth_dir + 'info.txt'
    edge_dir = tooth_dir + 'edge.txt'
    index_list=[]
    pos=[-5,-3,-1]
    pos.remove(pos[axis])
    mhd_list=[name for name in  os.listdir(tooth_dir) if os.path.splitext(name)[1] == '.mhd']
    for i,fileName in enumerate(mhd_list):
        if int(os.path.splitext(fileName)[0][pos[0]])==oth_rot1 and \
            int(os.path.splitext(fileName)[0][pos[1]]) == oth_rot2:
            index_list.append(i)
        else:
            os.remove(os.path.join(tooth_dir,fileName))
            os.remove(os.path.join(tooth_dir,os.path.splitext(fileName)[0]+'.zraw'))
    def delete_lines(file_path,need_list):
        line_list=[]
        with open(file_path) as f:
            for i, line in enumerate(f.readlines()):
                if i in need_list:
                    line_list.append(line)
        os.remove(file_path)
        with open(file_path,'w')as new_f:
            for line in line_list:
                new_f.write(line)

    if os.path.exists(facc_dir):
        delete_lines(facc_dir,index_list)
    if os.path.exists(groove_dir):
        delete_lines(groove_dir,index_list)
    if os.path.exists(edge_dir):
        delete_lines(edge_dir,index_list)


if __name__ == '__main__':
    
    
    # check_identical_line('F:\\ProjectData\\tmp\\Try\\Tooth_m\\')
    check_unique('F:\\ProjectData\\tmp\\Try\\Validate\\0828 AlbertDiaz-Conti\\tooth7\\edge.txt')
    # filter_axis_rotate('F:\\ProjectData\\tmp\\Try\\Validate\\0828 AlbertDiaz-Conti\\tooth4\\',
    #                    axis=1,oth_rot1=1,oth_rot2=1)
    # record_case('F:\\ProjectData\\tmp\\Train')
    # delete('F:\\ProjectData\\tmp\\Validate\\')
    # duplicated('F:\\ProjectData\\tmp\\Validate\\','F:\\ProjectData\\tmp\\Train\\')
