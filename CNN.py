import tensorflow as tf
import commen_structure as commen

class CNN(object):
    def __init__(self,param,phase,keep_prob,box):
        '''
        conv and fc
        :param param:
        :param phase:
        :param box: [task_num,b,h,w,d]
        '''
        self.param=param
        self.phase=phase
        self.keep_prob=keep_prob
        box=tf.expand_dims(box,axis=5)
        if len(param.task_dict)==1:
            self.param.task_layer_num=0
        commen_layer_index = range(self.param.layer_num - self.param.task_layer_num)
        task_layer_index = range(self.param.layer_num - self.param.task_layer_num, self.param.layer_num)
        commen_task_list=[]
        output_task_list=[]
        index=0
        for task,task_content in param.task_dict.items():
            with tf.variable_scope(task):
                with tf.variable_scope('commen'):
                    commen,filter_layer_list = self.build_CNN(box[index], commen_layer_index)
                    commen_task_list.append(filter_layer_list)
                    index+=1

                with tf.variable_scope('specialized'):
                    conv,_ = self.build_CNN(commen,task_layer_index)
                    fc_size=task_content['fc_size']
                    output=self.build_FC(conv,fc_size)
                    output_task_list.append(output)

        #[b,multi_output_size]
        self.output_multi_task=tf.concat(output_task_list,axis=1,name='build_predict')

        def regular_layer(task_var_list):
            assert len(task_var_list) > 1
            layer_term=0
            with tf.variable_scope('add_var_in_layer'):
                for i in range(len(task_var_list)-1):
                    layer_term+=tf.reduce_sum(tf.square(task_var_list[i] - task_var_list[i+1]))
                layer_term += tf.reduce_sum(tf.square(task_var_list[-1] - task_var_list[0]))
                return layer_term

        if len(param.task_dict) > 1:
            with tf.variable_scope('calculate_reg_term'):
                commen_layer_list = zip(*commen_task_list)
                self.regularization_term=tf.reduce_sum(tf.stack([regular_layer(task_list) for task_list in commen_layer_list]))


    def build_CNN(self,input_box,layer_index):
        with tf.variable_scope('CNN'):
            # # self.box.shape=[None, self.param.h, self.param.w, self.param.d, 1]
            # box = tf.reshape(self.box, shape=[-1, self.param.h, self.param.w, self.param.d, 1])
            # [B,H,W,D,1]
            conv = input_box
            filter_vars=[]
            for c in layer_index:
                conv,filter_var = commen.conv3d(conv, self.param.channels[c ], filter_size=self.param.filter_size[c],
                                     stride=self.param.stride[c ],phase=self.phase,
                                      pooling=self.param.pooling[c ],scope="conv_"+ str(c))
                filter_vars.append(filter_var)

        return conv,filter_vars

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


