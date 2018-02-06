
MODEL_PATH = 'F:/ProjectData/Feature2/model/'
# MODEL_PATH = 'F:/ProjectData/Feature2/output_pb/'
Feature_Target=1
SHAPE_BOX=[128,128,128]
FC_SIZE=[256, 6]
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
