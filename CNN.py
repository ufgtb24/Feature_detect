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
        self.box=box
        conv=self.build_CNN()
        self.output=self.build_FC(conv)


    def build_CNN(self):
        with tf.variable_scope('CNN'):
            # # self.box.shape=[None, self.param.h, self.param.w, self.param.d, 1]
            # box = tf.reshape(self.box, shape=[-1, self.param.h, self.param.w, self.param.d, 1])
            # [B,H,W,D,1]
            conv = self.box
            for c in range(self.param.layer_num):
                conv = commen.conv3d(conv, self.param.channels[c ], filter_size=self.param.filter_size[c],
                                     stride=self.param.stride[c ],phase=self.phase,
                                      pooling=self.param.pooling[c ],name="conv_"+ str(c))

        return conv

    def build_FC(self,conv):
        with tf.variable_scope('FC_for_CNN'):
            fc = commen.flatten(conv)
            fc = commen.dense_relu_batch_dropout(fc, output_size=self.param.fc_size[0],
                                                  phase=self.phase, keep_prob=self.keep_prob,scope='fc_1')
            fc_output = commen.dense(fc, output_size=self.param.fc_size[1],scope='fc_output')
        return fc_output


