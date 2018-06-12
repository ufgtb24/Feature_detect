from collections import OrderedDict
BOX_LEN=128
MODEL_PATH = 'F:/ProjectData/tmp/model/lower/'
LOG_PATH='log/lower/'
SHAPE_BOX = [BOX_LEN]*3+[ 1]
up_set = ['tooth2', 'tooth3', 'tooth4', 'tooth5', 'tooth6', 'tooth7', 'tooth8']
low_set = ['tooth30', 'tooth31', 'tooth28', 'tooth29', 'tooth27', 'tooth25', 'tooth26']


def get_feature_num():
    num_feature_need = 0
    for content in TASK_DICT.values():
        num_feature_need += len(content['feature_need'])
    return num_feature_need


CLASS_DICT_U = {
    'tooth2': 0,
    'tooth3': 0,
    'tooth4': 1,
    'tooth5': 1,
    'tooth6': 2,
    'tooth7': 3,
    'tooth8': 3,
}
CLASS_DICT_L = {
    'tooth30': 0,
    'tooth31': 0,
    'tooth28': 1,
    'tooth29': 1,
    'tooth27': 2,
    'tooth26': 3,
    'tooth25': 3,
}
facc_prop= {
             'num_feature': 5,
             'feature_need': [1, 2, 3, 4, 5],
             'label_file': 'FaccControlPts.txt'
         }
groove_prop= {
             'num_feature': 2,
             'feature_need': [1, 2],
             'label_file': 'info.txt'
         }

CLASS_FTR_DICT={
    0:[facc_prop,groove_prop],
    1:[facc_prop,groove_prop],
    2:[facc_prop],
    3:[facc_prop]
}



# DATA_DICT = {'up_back':{
#         'data_set': up_back,
#         'model_path': MODEL_PATH + 'up_back/',
#         'task': 'all'
#     },
#     'up_middle':{
#         'data_set': up_middle,
#         'model_path': MODEL_PATH + 'up_middle/',
#         'task': 'all'
#
#     },
#     'up_canine':{
#         'data_set': up_canine,
#         'model_path': MODEL_PATH + 'up_canine/',
#         'task': 'facc'
#     },
#     'up_front': {
#         'data_set': up_front,
#         'model_path': MODEL_PATH + 'up_front/',
#         'task': 'facc'
#     }
# }
#
#
# CLASS_TASK_DICT = OrderedDict(
#     [
#         ('facc',
#          {
#              'num_feature': 5,
#              'feature_need': [1, 2, 3, 4, 5],
#              'label_file': 'FaccControlPts.txt'
#          }
#          ),
#
#         ('groove',
#          {
#              'num_feature': 2,
#              'feature_need': [1, 2],
#              'label_file': 'info.txt'
#          }
#          )
#     ]
# )



class DataConfig(object):
    data_list = low_set
    world_to_cubic = BOX_LEN / 12.
    # base_case_dir='F:/ProjectData/Feature2/DataSet/'
    base_case_dir = 'F:/ProjectData/tmp/'
    # output_dim=3*len(feature_need)
    # label_file_name='info.txt'
    class_ftr_dict = CLASS_FTR_DICT
    num_feature_need = get_feature_num()
    feature_dim = 3 * num_feature_need
    down_rate=int(128/BOX_LEN)
    class_define=CLASS_DICT_L

class TrainDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir + 'Train/'
    load_case_once = 4  # 每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    usage = '_Train'


class ValiDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir + 'Validate/'
    load_case_once = 4  # 每次读的病例数
    switch_after_shuffles = 10 ** 10  # 当前读取的数据洗牌n次读取新数据,仅当load_case_once>0时有效
    usage = '_Validate'


class TestDataConfig(DataConfig):
    batch_size = 1
    total_case_dir = DataConfig.base_case_dir + 'Validate/'
    load_case_once = 1  # 每次读的病例数
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据
    usage = '_Test'


if __name__ == '__main__':
    print(TestDataConfig.world_to_cubic)