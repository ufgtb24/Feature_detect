import os

# label file column number check
from display import load_y

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
            edge_dir=full_case_dir+'/'+tooth+'/edge.txt'
            if os.path.exists(facc_dir):
                if len(open(facc_dir, 'rU').readlines())!=81:
                    print(case_name,'  lack  facc ')
                    
            if os.path.exists(groove_dir):
                if len(open(groove_dir, 'rU').readlines())!=81:
                    print(case_name,'  lack  groove ')

            if os.path.exists(edge_dir):
                if len(open(edge_dir, 'rU').readlines())!=81:
                    print(case_name,'  lack  edge ')


            # if 'tooth'+str(i) not in data_list:
            #     print(case_name)

    return error_num
'''
# label file line number identical check
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
            
    
                
            


            # if 'tooth'+str(i) not in data_list:
            #     print(case_name)

    return error_num



# # label file line number identical check
# def check_availability(dir):
#     Tooth_dir=os.listdir(dir)
#     error_num=0
#
#     for case_name in Tooth_dir:
#         full_case_dir = dir  + case_name
#         tooth_list=os.listdir(full_case_dir)
#         for tooth in tooth_list:
#             valide=True
#             facc_dir=full_case_dir+'/'+tooth+'/FaccControlPts.txt'
#             groove_dir=full_case_dir+'/'+tooth+'/info.txt'
#             edge_dir=full_case_dir+'/'+tooth+'/edge.txt'
#             f_num=g_num=e_num=0
#             if os.path.exists(facc_dir):
#                 f_num=len(open(facc_dir, 'rU').readlines())
#             if os.path.exists(groove_dir):
#                 g_num =len(open(groove_dir, 'rU').readlines())
#             if os.path.exists(edge_dir):
#                 e_num =len(open(edge_dir, 'rU').readlines())
#             files_num=[f_num,g_num,e_num]
#             max_num=max(files_num)
#             for file_num in files_num:
#                 if file_num !=max_num and file_num!=0:
#                     valide=False
#             if valide==False:
#                 print(case_name+'  '+tooth)
#
#
#
#             # if 'tooth'+str(i) not in data_list:
#             #     print(case_name)
#
#     return error_num


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


if __name__ == '__main__':
    check_availability('F:/ProjectData/tmp/Train/')
