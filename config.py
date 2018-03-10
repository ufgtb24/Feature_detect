
MODEL_PATH = 'F:/ProjectData/Feature2/models/model_up/'
# MODEL_PATH = 'F:/ProjectData/Feature2/output_pb/'
Feature_Target=1
SHAPE_BOX=[128,128,128,1]
FC_SIZE=[128,6]



class TrainDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 16
    # total_case_dir='F:/ProjectData/Feature/Tooth'
    total_case_dir = 'F:/ProjectData/Feature2/DataSet/Train'
    data_list=['tooth2','tooth3','tooth4','tooth5','tooth12','tooth13','tooth14','tooth15']
    load_case_once = 10 # 每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'


class ValiDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 16
    total_case_dir = 'F:/ProjectData/Feature2/DataSet/Validate'
    data_list=TrainDataConfig.data_list
    load_case_once =0 # 每次读的病例数
    switch_after_shuffles =10**10  # 当前读取的数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'

class TestDataConfig(object):
    world_to_cubic=128/12.
    batch_size=1
    total_case_dir=ValiDataConfig.total_case_dir
    data_list=TrainDataConfig.data_list
    load_case_once=1  #每次读的病例数
    switch_after_shuffles=1 #当前数据洗牌n次读取新数据

