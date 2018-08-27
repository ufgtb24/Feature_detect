from collections import OrderedDict
BOX_LEN=128
MODEL_PATH = 'F:/ProjectData/tmp/model/up5104_3/'
MODEL_NAME='model.ckpt-166'
SHAPE_BOX = [BOX_LEN]*3+[ 1]
# up_front=['tooth6', 'tooth7', 'tooth8']

def get_feature_num():
    num_feature_need = 0
    for content in TASK_DICT.values():
        num_feature_need += len(content['feature_need'])
    return num_feature_need



# LOSS_WEIGHT=np.array([1]*6+[0.3]*15+[0.5]*6)

# python 3.5 之前的 以字符串为索引的 dict 是无序的，以数字为索引的有序
TASK_DICT = OrderedDict(
    [
        ('edge',
         {
             'num_feature': 2,
             'feature_need': [1, 2],
             'label_file': 'edge.txt',
             'loss_weight':1,
             'sample_propotion':1
         }
         ),
        
        ('facc',
         {
             'num_feature': 5,
             'feature_need': [1, 2, 3, 4, 5],
             'label_file': 'FaccControlPts.txt',
             'loss_weight': 1,
             'sample_propotion': 1

         }
         ),

        ('groove',
         {
             'num_feature': 2,
             'feature_need': [1, 2],
             'label_file': 'info.txt',
             'loss_weight': 1,
             'sample_propotion': 1
         }
         )
    ]
)
LOSS_WEIGHT=[]
SAMPLE_PROP={}
for key,content in TASK_DICT.items():
    LOSS_WEIGHT.extend([content['loss_weight']]*(content['num_feature']*3))
    SAMPLE_PROP[key]=content['sample_propotion']
    
    
up_back=['tooth2','tooth3']
up_middle=['tooth4','tooth5']
up_canine=['tooth6']
up_front=['tooth7', 'tooth8']
low_back=['tooth30','tooth31']
low_middle=['tooth28','tooth29']
low_canine=['tooth27']
low_front=['tooth25','tooth26']

up_edge=['tooth6', 'tooth7', 'tooth8']
low_edge=['tooth25','tooth26','tooth27']

up_set = ['tooth2', 'tooth3', 'tooth4', 'tooth5', 'tooth6', 'tooth7', 'tooth8']
low_set = ['tooth30', 'tooth31', 'tooth28', 'tooth29', 'tooth27', 'tooth25', 'tooth26']

class DataConfig(object):
    data_list = up_edge
    world_to_cubic = BOX_LEN / 12.
    # base_case_dir='F:/ProjectData/Feature2/DataSet/'
    base_case_dir = 'F:/ProjectData/tmp/'
    # output_dim=3*len(feature_need)
    # label_file_name='info.txt'
    task_dict = TASK_DICT
    num_feature_need = get_feature_num()
    feature_dim = 3 * num_feature_need
    down_rate=int(128/BOX_LEN)
    sample_prob=SAMPLE_PROP

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