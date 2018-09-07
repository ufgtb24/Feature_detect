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
                

# # label file line number identical and validate check
pass
def delete(dir):
    Tooth_dir=os.listdir(dir)

    for case_name in Tooth_dir:
        full_case_dir = dir  + case_name
        tooth_list=os.listdir(full_case_dir)
        for tooth in tooth_list:
            if int(tooth[5:])>8 and int(tooth[5:])<25:
                shutil.rmtree(full_case_dir+'/'+tooth)
                

def duplicated(validate,train):
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


if __name__ == '__main__':
    # check_availability('F:/ProjectData/tmp/Tooth0904/Tooth/')
    delete('F:\\ProjectData\\tmp\\Validate\\')
    # duplicated('F:\\ProjectData\\tmp\\Validate\\','F:\\ProjectData\\tmp\\Train\\')
