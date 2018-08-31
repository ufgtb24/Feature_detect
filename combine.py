import os

from tensorflow.python.tools import freeze_graph
import tensorflow as tf

PB_PATH = 'input_graph.pb'

input_saver = ""
input_binary = False
# input_checkpoint = os.path.join(MODEL_PATH,MODEL_NAME)

# Note that we this normally should be only "output_node"!!!
output_node_names = "detector/output_node"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
clear_devices = False
initializer_nodes=[]
variable_names_whitelist=""
variable_names_blacklist=""

def gen_frozen_graph(var_file, ckpt_dir):
    freeze_graph.freeze_graph(os.path.join(ckpt_dir, PB_PATH),
                              input_saver,
                              input_binary,
                              var_file,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              os.path.join(ckpt_dir, 'output_graph.pb'),
                              clear_devices,
                              initializer_nodes,
                              variable_names_whitelist,
                              variable_names_blacklist
                              )
    os.remove(os.path.join(ckpt_dir, PB_PATH))


# def load_graph(frozen_graph_filename):
#     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(
#             graph_def,
#             input_map=None,
#             return_elements=None,
#             op_dict=None,
#             producer_op_list=None
#         )
#
#     return graph
def load_graph(sess,frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with sess.graph.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return sess.graph