import tensorflow as tf
import commen_structure as commen

class CNN(object):
    def __init__(self,param,phase,keep_prob,box):
        '''
        conv and fc
        :param param:
        :param phase:
        :param box: [b,24,h,w,d]
        :param pos: [b,24,3]
        '''
        self.param=param
        self.phase=phase
        self.keep_prob=keep_prob
        box=tf.expand_dims(box,axis=4)
        if len(param.task_dict)==1:
            self.param.task_layer_num=0
        commen_layer_index = range(self.param.layer_num - self.param.task_layer_num)
        task_layer_index = range(self.param.layer_num - self.param.task_layer_num, self.param.layer_num)
        with tf.variable_scope('commen'):
            commen = self.build_CNN(box, commen_layer_index)
        self.output = {}
        for task,task_content in param.task_dict.items():
            with tf.variable_scope(task):
                conv = self.build_CNN(commen,task_layer_index)
                fc_size=task_content['fc_size']
                output=self.build_FC(conv,fc_size)
                self.output[task]=output


    def build_CNN(self,input_box,layer_index):
        with tf.variable_scope('CNN'):
            # # self.box.shape=[None, self.param.h, self.param.w, self.param.d, 1]
            # box = tf.reshape(self.box, shape=[-1, self.param.h, self.param.w, self.param.d, 1])
            # [B,H,W,D,1]
            conv = input_box
            for c in layer_index:
                conv = commen.conv3d(conv, self.param.channels[c ], filter_size=self.param.filter_size[c],
                                     stride=self.param.stride[c ],phase=self.phase,
                                      pooling=self.param.pooling[c ],scope="conv_"+ str(c))

        return conv

    def build_FC(self,conv,fc_size):
        with tf.variable_scope('FC'):
            fc = commen.flatten(conv)
            fc_layer=1
            for size in fc_size[:-1]:
                fc = commen.dense_relu_batch_dropout(fc, output_size=size,
                                                      phase=self.phase, keep_prob=self.keep_prob,scope='fc_'+str(fc_layer))
                fc_layer+=1
        output = commen.dense(fc, output_size=fc_size[-1], scope='output')
        return output


