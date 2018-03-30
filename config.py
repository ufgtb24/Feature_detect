
# MODEL_PATH = 'F:/ProjectData/Feature2/models/model_low/'
MODEL_PATH = 'F:/ProjectData/Feature2/models/model_up/'
# MODEL_PATH = 'F:/ProjectData/Feature2/model_bk/models_whole_latest/models_3_10_17_32/model_up'
SHAPE_BOX=[128,128,128,1]
Upper_set=['tooth2','tooth3','tooth4','tooth5','tooth12','tooth13','tooth14','tooth15']
# Lowwer_set=['tooth18','tooth19','tooth20','tooth21','tooth28','tooth29','tooth30','tooth31']
back_set=['tooth2','tooth3']
middle_set=['tooth4','tooth5','tooth6']
front_set=['tooth7','tooth8']

class DataConfig(object):
    world_to_cubic = 128 / 12.
    data_list = back_set
    # base_case_dir='F:/ProjectData/Feature2/DataSet/'
    base_case_dir='F:/ProjectData/tmp/'
    num_feature=5
    feature_need=[1]
    output_dim=3*len(feature_need)
    # label_file_name='info.txt'
    label_file_name='FaccControlPts.txt'

class TrainDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir+'Train/'
    load_case_once = 10 # 每次读的病例数 若果=0,则只load一次，读入全部
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