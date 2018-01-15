import argparse
import tensorflow as tf
import os

from config import MODEL_PATH
from display import edges
from predict import DataConfig
from dataRelated import BatchGenerator
from mayavi import mlab
import numpy as np
def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    # MODEL_PATH = 'F:/ProjectData/Feature2/model/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename",
                        default=os.path.join(MODEL_PATH,'whole//output_graph.pb'),
                        type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    # 加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name, op.values())
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    # 注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    input_box = graph.get_tensor_by_name('prefix/input_box:0')
    phase = graph.get_tensor_by_name('prefix/phase_input:0')
    keep_prob = graph.get_tensor_by_name('prefix/keep_prob_input:0')
    pred_end = graph.get_tensor_by_name('prefix/output_node:0')

    test_batch_gen=BatchGenerator(DataConfig,need_target=False,need_name=False)

    with tf.Session(graph=graph) as sess:
        while True:
            box_batch = test_batch_gen.get_batch()[0]
            # box_batch, target = test_batch_gen.get_batch()
            # target_1=target[:,:3]
            # target_2=target[:,3:]

            feed_dict = {input_box: box_batch,
                         phase: False, keep_prob: 1}

            f = sess.run(pred_end, feed_dict=feed_dict)
            f_1 = f[:3]
            f_2 = f[3:]
            print(f)
            box = box_batch[0]

            # box[target_1[i,0], target_1[i,1], target_1[i,2]] = 2
            # box[target_2[i,0], target_2[i,1], target_2[i,2]] = 2
            box[f_1[0], f_1[1], f_1[2]] = 3
            box[f_2[0], f_2[1], f_2[2]] = 3

            x, y, z = np.where(box == 1)
            ex, ey, ez = edges(128)
            # fx, fy, fz = np.where(box == 2)
            fxp, fyp, fzp = np.where(box == 3)

            mlab.points3d(ex, ey, ez,
                          mode="cube",
                          color=(0, 0, 1),
                          scale_factor=1)

            mlab.points3d(x, y, z,
                          mode="cube",
                          color=(0, 1, 0),
                          scale_factor=1,
                          transparent=True)

            # mlab.points3d(fx, fy, fz,
            #             mode="cube",
            #             color=(1, 0, 0),
            #             scale_factor=1,
            #               transparent=True)

            mlab.points3d(fxp, fyp, fzp,
                          mode="cube",
                          color=(0, 0, 1),
                          scale_factor=1,
                          transparent=True)

            mlab.show()
