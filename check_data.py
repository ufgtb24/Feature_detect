import os



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



if __name__ == '__main__':
    check_availability('F:/ProjectData/tmp/Train/')