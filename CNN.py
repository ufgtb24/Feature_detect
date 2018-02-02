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
        if len(param.tasks)==1:
            self.param.task_layer_num=0
        commen_layer_index = range(self.param.layer_num - self.param.task_layer_num)
        task_layer_index = range(self.param.layer_num - self.param.task_layer_num, self.param.layer_num)
        commen = self.build_CNN(box, commen_layer_index, 'commen')
        self.output = {}

        for task in param.tasks:
            conv = self.build_CNN(commen,task_layer_index, task)
            fc_output=self.build_FC(conv,task)
            self.output[task]=fc_output


    def build_CNN(self,input_box,layer_index,name):
        with tf.variable_scope('CNN_'+name):
            # # self.box.shape=[None, self.param.h, self.param.w, self.param.d, 1]
            # box = tf.reshape(self.box, shape=[-1, self.param.h, self.param.w, self.param.d, 1])
            # [B,H,W,D,1]
            conv = input_box
            for c in layer_index:
                conv = commen.conv3d(conv, self.param.channels[c ], filter_size=self.param.filter_size[c],
                                     stride=self.param.stride[c ],phase=self.phase,
                                      pooling=self.param.pooling[c ],scope="conv_"+ str(c))

        return conv

    def build_FC(self,conv,name):
        with tf.variable_scope('FC_'+name):
            fc = commen.flatten(conv)
            fc = commen.dense_relu_batch_dropout(fc, output_size=self.param.fc_size[0],
                                                  phase=self.phase, keep_prob=self.keep_prob,scope='fc_1')
            # fc = commen.dense_relu_batch_dropout(fc, output_size=self.param.fc_size[1],
            #                                       phase=self.phase, keep_prob=self.keep_prob,scope='fc_2')
            fc_output = commen.dense(fc, output_size=self.param.fc_size[1],scope='fc_output')
        return fc_output


