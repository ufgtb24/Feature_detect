import os

from tensorflow.python.tools import freeze_graph

MODEL_PATH = 'F:/ProjectData/Feature/model/'

checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

input_graph = os.path.join(MODEL_PATH,'whole/input_graph.pb')
input_saver = ""
input_binary = False
input_checkpoint = os.path.join(MODEL_PATH,'whole/model.ckpt')

# Note that we this normally should be only "output_node"!!!
output_node_names = "output_node"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph = os.path.join(MODEL_PATH,'whole/output_graph.pb')
clear_devices = False
initializer_nodes=[]
variable_names_whitelist=""
variable_names_blacklist="box_crop_var1,box_crop_var2"
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