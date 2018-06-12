import os

from tensorflow.python.tools import freeze_graph
import tensorflow as tf
from config import MODEL_PATH
PB_PATH = 'input_graph.pb'

checkpoint_state_name = "checkpoint_state"
input_graph = os.path.join(MODEL_PATH,PB_PATH)
input_saver = ""
input_binary = False
input_checkpoint = os.path.join(MODEL_PATH,'model.ckpt-74')

# Note that we this normally should be only "output_node"!!!
output_node_names = "detector/output_node"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph = os.path.join(MODEL_PATH,'output_graph.pb')
clear_devices = False
initializer_nodes=[]
variable_names_whitelist=""
variable_names_blacklist=""

def gen_frozen_graph():
    freeze_graph.freeze_graph(input_graph,
                              input_saver,
                              input_binary,
                              input_checkpoint,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph,
                              clear_devices,
                              initializer_nodes,
                              variable_names_whitelist,
                              variable_names_blacklist
                              )


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
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    g=tf.Graph()
    # Then, we import the graph_def into a new Graph and returns it
    with g.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return g