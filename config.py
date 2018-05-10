
MODEL_PATH = 'F:/ProjectData/tmp/model/up_front/'
SHAPE_BOX=[128,128,128,1]
total_set=['tooth2','tooth3','tooth4','tooth5','tooth6','tooth7','tooth8']
# Lowwer_set=['tooth18','tooth19','tooth20','tooth21','tooth28','tooth29','tooth30','tooth31']

def get_feature_num():
    num_feature_need = 0
    for content in TASK_DICT.values():
        num_feature_need += len(content['feature_need'])
    return num_feature_need

TASK_DICT={
    'facc':{
        'num_feature':5,
        'feature_need':[1,2,3,4,5],
        'label_file':'FaccControlPts.txt'
    },
    # 'groove':{
    #     'num_feature':2,
    #     'feature_need':[1,2],
    #     'label_file':'info.txt'
    # }
}

up_back=['tooth2','tooth3']
up_middle=['tooth4','tooth5']
up_canine=['tooth6']
up_front=['tooth7','tooth8']
low_back=['tooth30','tooth31']
low_middle=['tooth28','tooth29']
low_canine=['tooth27']
low_front=['tooth25','tooth26']

class DataConfig(object):
    data_list = up_canine
    world_to_cubic = 128 / 12.
    # base_case_dir='F:/ProjectData/Feature2/DataSet/'
    base_case_dir='F:/ProjectData/tmp/Try/'
    # output_dim=3*len(feature_need)
    # label_file_name='info.txt'
    task_dict=TASK_DICT
    num_feature_need=get_feature_num()
    output_dim = 3 * num_feature_need


class TrainDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir+'Train/'
    load_case_once = 20 # 每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    usage='_Train'

class ValiDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir+'Validate/'
    load_case_once =0 # 每次读的病例数
    switch_after_shuffles =10**10  # 当前读取的数据洗牌n次读取新数据,仅当load_case_once>0时有效
    usage='_Validate'


class TestDataConfig(DataConfig):
    batch_size=1
    total_case_dir=DataConfig.base_case_dir+'Validate/'
    load_case_once=1  #每次读的病例数
    switch_after_shuffles=1 #当前数据洗牌n次读取新数据
    usage='_Test'


if __name__ == '__main__':
    print(TestDataConfig.world_to_cubic)