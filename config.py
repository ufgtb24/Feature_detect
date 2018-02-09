
MODEL_PATH = 'F:/ProjectData/Feature2/models/model_up/'
# MODEL_PATH = 'F:/ProjectData/Feature2/output_pb/'
Feature_Target=1
SHAPE_BOX=[128,128,128]
FC_SIZE=[512,512,6]
TASK_DICT = {
    'LB':
        {
            'input_tooth': ['tooth2','tooth3'],
            'fc_size': FC_SIZE
        },

    'LF':
        {
            'input_tooth': ['tooth4','tooth5'],
            'fc_size': FC_SIZE
        },

    'RB':
        {
            'input_tooth': ['tooth14','tooth15'],
            'fc_size': FC_SIZE
        },

    'RF':
        {
            'input_tooth': ['tooth12','tooth13'],
            'fc_size': FC_SIZE
        }
}

# TASK_DICT = {
#     'LB':
#         {
#             'input_tooth': ['tooth30','tooth31'],
#             'fc_size': FC_SIZE
#         },
#
#     'LF':
#         {
#             'input_tooth': ['tooth28','tooth29'],
#             'fc_size': FC_SIZE
#         },
#
#     'RB':
#         {
#             'input_tooth': ['tooth18','tooth19'],
#             'fc_size': FC_SIZE
#         },
#
#     'RF':
#         {
#             'input_tooth': ['tooth20','tooth21'],
#             'fc_size': FC_SIZE
#         }
# }

class NetConfig(object):
    shape_box = SHAPE_BOX
    channels = [32, 32, 64, 64, 128, 256, 512]  # 决定左侧的参数多少和左侧的memory
    # channels = [32, 32, 64, 64, 128]  # 决定左侧的参数多少和左侧的memory
    task_dict =TASK_DICT
    pooling = [True, True, True, True, True, True, True,False]
    filter_size = [5, 3, 3, 3, 3, 3, 3,1,1]  # 决定左侧的参数多少
    stride = [1]*9  # 决定右侧的memory
    layer_num = len(channels)
    task_layer_num = 0


class TrainDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 4
    # total_case_dir='F:/ProjectData/Feature/Tooth'
    total_case_dir = 'F:/ProjectData/Feature2/DataSet/Train'
    data_list=None
    load_case_once = 10  # 每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'


class ValiDataConfig(object):
    world_to_cubic = 128 / 12.
    batch_size = 4
    total_case_dir = 'F:/ProjectData/Feature2/DataSet/Validate'
    data_list=None
    load_case_once = 0  # 每次读的病例数
    switch_after_shuffles = 10 ** 10  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    format = 'mhd'

class TestDataConfig(object):
    world_to_cubic=128/12.
    batch_size=1
    total_case_dir='F:/ProjectData/Feature2/DataSet/Validate'
    data_list=None
    load_case_once=1  #每次读的病例数
    switch_after_shuffles=1 #当前数据洗牌n次读取新数据
