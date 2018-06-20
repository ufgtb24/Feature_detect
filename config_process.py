from collections import OrderedDict

from config import BOX_LEN

MODEL_PATH = 'F:/ProjectData/tmp/model/'
SHAPE_BOX = [BOX_LEN, BOX_LEN, BOX_LEN, 1]
total_set = ['tooth2', 'tooth3', 'tooth4', 'tooth5', 'tooth6', 'tooth7', 'tooth8']
# Lowwer_set=['tooth18','tooth19','tooth20','tooth21','tooth28','tooth29','tooth30','tooth31']

up_back = ['tooth2', 'tooth3']
up_middle = ['tooth4', 'tooth5']
up_canine = ['tooth6']
up_front = ['tooth7', 'tooth8']
low_back = ['tooth30', 'tooth31']
low_middle = ['tooth28', 'tooth29']
low_canine = ['tooth27']
low_front = ['tooth25', 'tooth26']

TASK_DICT = {
    'facc': {
        'num_feature': 5,
        'feature_need': [1, 2, 3, 4, 5],
        'label_file': 'FaccControlPts.txt'
    },
    'groove': {
        'num_feature': 2,
        'feature_need': [1, 2],
        'label_file': 'info.txt'
    }
}

TASK_DICT_F = {
    'facc': {
        'num_feature': 5,
        'feature_need': [1, 2, 3, 4, 5],
        'label_file': 'FaccControlPts.txt'
    },
}

DATA_DICT = OrderedDict([
    ('up_back',{
        'data_set': up_back,
        'model_path': MODEL_PATH + 'up_back/',
        'task': 'all'
    }),
    ('up_middle', {
        'data_set': up_middle,
        'model_path': MODEL_PATH + 'up_middle/',
        'task': 'all'

    }),
    ('up_canine',{
        'data_set': up_canine,
        'model_path': MODEL_PATH + 'up_canine/',
        'task': 'facc'
    }),
    ('up_front', {
        'data_set': up_front,
        'model_path': MODEL_PATH + 'up_front/',
        'task': 'facc'
    }),
    ('low_back', {
        'data_set': low_back,
        'model_path': MODEL_PATH + 'low_back/',
        'task': 'all'

    }),
    ('low_middle', {
        'data_set': low_middle,
        'model_path': MODEL_PATH + 'low_middle/',
        'task': 'all',

    }),
    ('low_canine', {
        'data_set': low_canine,
        'model_path': MODEL_PATH + 'low_canine/',
        'task': 'facc'

    }),
    ('low_front', {
        'data_set': low_front,
        'model_path': MODEL_PATH + 'low_front/',
        'task': 'facc'

    }),
])


def get_feature_num(task_dict):
    num_feature_need = 0
    for content in task_dict.values():
        num_feature_need += len(content['feature_need'])
    return num_feature_need


class DataConfig(object):
    data_list = low_middle
    world_to_cubic = BOX_LEN / 12.
    # base_case_dir='F:/ProjectData/Feature2/DataSet/'
    base_case_dir = 'F:/ProjectData/tmp/'
    # output_dim=3*len(feature_need)
    # label_file_name='info.txt'
    task_dict = None
    num_feature_need = None
    output_dim = None


class TrainDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir + 'Train/'
    load_case_once = 20  # 每次读的病例数 若果=0,则只load一次，读入全部
    switch_after_shuffles = 1  # 当前数据洗牌n次读取新数据,仅当load_case_once>0时有效
    usage = '_Train'


class ValiDataConfig(DataConfig):
    batch_size = 16
    total_case_dir = DataConfig.base_case_dir + 'Validate/'
    load_case_once = 0  # 每次读的病例数
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