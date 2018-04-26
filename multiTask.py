import tensorflow as tf
import numpy as np

BALANCED = 'balanced'
IMBALANCED = 'imbalanced'


class CrossStitchLayer:
    """Cross-stitch layer class."""
    
    def __init__(self, num_tasks, num_subspaces=1,
                 init_scheme=BALANCED):
        """
        Initializes a CrossStitchLayer.
        :param model: the DyNet Model
        :param num_tasks: the number of tasks
        :param hidden_dim: the # of hidden dimensions of the previous LSTM layer
        :param num_subspaces: the number of subspaces
        :param init_scheme: the initialization scheme; balanced or imbalanced
        """
        print('Using %d subspaces...' % num_subspaces, flush=True)
        
        alpha_params = np.full((num_tasks * num_subspaces,
                                num_tasks * num_subspaces),
                               1. / (num_tasks * num_subspaces))
        # alpha_params = tf.get_variable('alpha',
        #                        (num_tasks * num_subspaces,
        #                         num_tasks * num_subspaces),
        #                         initializer=tf.constant_initializer(1. / (num_tasks * num_subspaces))
        #                        )
        if init_scheme == IMBALANCED:
            if num_subspaces == 1:
                alpha_params = np.full((num_tasks, num_tasks),
                                       0.1 / (num_tasks - 1))
                for i in range(num_tasks):
                    alpha_params[i, i] = 0.9
            else:
                # 0 1 0 1
                # 0 1 0 1
                # 1 0 1 0
                # 1 0 1 0
                for (x, y), value in np.ndenumerate(alpha_params):
                    if (y + 1) % num_subspaces == 0 and not \
                            (x in range(num_tasks, num_tasks + num_subspaces)):
                        alpha_params[x, y] = 0.95
                    elif (y + num_subspaces) % num_subspaces == 0 and x \
                            in range(num_tasks, num_tasks + num_subspaces):
                        alpha_params[x, y] = 0.95
                    else:
                        alpha_params[x, y] = 0.05
        
        self.alphas = tf.get_variable('alphas',
                                      initializer=tf.constant(alpha_params,dtype=tf.float32)
                                      )
        
        self.num_tasks = num_tasks
        self.num_subspaces = num_subspaces
        # self.shape=shape
    
    def stitch(self, predictions,task_shape):
        """
        Takes as inputs a list of the predicted states of the previous layers of
        each task, e.g. for two tasks a list containing two lists of
        n-dimensional output states. For every time step, the predictions of
        each previous task layer are then multiplied with the cross-stitch
        units to obtain a linear combination. In the end, we obtain a list of
        lists of linear combinations of states for every subsequent task layer.
        :param predictions: a list of length num_tasks with shape [HWDC]
        :return: a list of length num_tasks containing the linear combination of
                 predictions for each task
        """
        assert self.num_tasks == len(predictions)
        # iterate over tuples of predictions of each task at every time step
        
        channel = task_shape[-1]
        # shape_NWHD=self.shape[:,-1]
        # channel=self.shape[-1]
        # [NWHD]+[t*c]
        predictions = tf.concat(predictions, axis=-1)
        # [NWHD]+[t*s,c/s]
        aug_shape =np.array([self.num_tasks * self.num_subspaces,int(channel / self.num_subspaces)])
        shape_NWHD = np.array(task_shape[1:-1])
        # combine_shape =tf.concat([[-1],shape_NWHD,aug_shape],axis=0)
        combine_shape=[-1]+task_shape[1:-1]+[self.num_tasks * self.num_subspaces, int(channel / self.num_subspaces)]
        predictions = tf.reshape(predictions, combine_shape)
        product = tf.tensordot(self.alphas, predictions,axes=[[1],[4]])
        np.matmul()
        tf.batch_matmul()
        product = tf.matmul(self.alphas, predictions)
        if self.num_subspaces != 1:
            # integrade all the subspace into their task
            product = tf.reshape(product,
                                 shape_NWHD + [self.num_tasks, channel])
        # split the concated tensor by task into a list
        # element shape [NWHDC]
        stitched = [tf.squeeze(task_space, [-2])
                    for task_space in tf.split(product, self.num_tasks, axis=-2)]
        
        return stitched

# class LayerStitchLayer:
#     """Layer-stitch layer class."""
#     def __init__(self, num_layers, hidden_dim, init_scheme=IMBALANCED):
#         """
#         Initializes a LayerStitchLayer.
#         :param model: the DyNet model
#         :param num_layers: the number of layers
#         :param hidden_dim: the hidden dimensions of the LSTM layers
#         :param init_scheme: the initialisation scheme; balanced or imbalanced
#         """
#         if init_scheme == IMBALANCED:
#             beta_params = np.full((num_layers), 0.1 / (num_layers - 1))
#             beta_params[-1] = 0.9
#         elif init_scheme == BALANCED:
#             beta_params = np.full((num_layers), 1. / num_layers)
#         else:
#             raise ValueError('Invalid initialization scheme for layer-stitch '
#                              'units: %s.' % init_scheme)
#         self.betas = tf.get_variable('betas',
#                                 initializer=tf.constant_initializer(tf.constant(beta_params))
#                                )
#
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#
#     def stitch(self, layer_predictions):
#         """
#         Takes as input the predicted states of all the layers of a task-specific
#         network and produces a linear combination of them.
#         :param layer_predictions: a list of length num_layers containing lists
#                                   of length seq_len of predicted states for
#                                   each layer
#         :return: a list of linear combinations of the predicted states at every
#                 time step for each layer
#         """
#         assert len(layer_predictions) == self.num_layers
#         linear_combinations = []
#         # iterate over tuples of predictions of each layer at every time step
#         for layer_states in zip(*layer_predictions):
#             # concatenate the predicted state for all layers to a matrix of
#             # shape (num_layers, hidden_dim)
#             concatenated_layer_states = dynet.reshape(dynet.concatenate_cols(
#                 list(layer_states)), (self.num_layers, self.hidden_dim))
#
#             # multiply with (1, num_layers) betas to produce (1, hidden_dim)
#             product = dynet.transpose(dynet.parameter(
#                 self.betas)) * concatenated_layer_states
#
#             # reshape to (hidden_dim)
#             reshaped = dynet.reshape(product, (self.hidden_dim,))
#             linear_combinations.append(reshaped)
#         return linear_combinations