import os



def check_availability(dir):
    Tooth_dir=os.listdir(dir)
    error_num=0

    for case_name in Tooth_dir:
        full_case_dir = dir + '\\' + case_name
        data_list=os.listdir(full_case_dir)
        for i in range(2,8):
            if 'tooth'+str(i) not in data_list:
                print(case_name)

    return error_num
if __name__ == '__main__':
    check_availability('F:/ProjectData/tmp/Train')