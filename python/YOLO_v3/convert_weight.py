import os
import sys
import wget
import time
import argparse
import tensorflow as tf
from core import yolov3, utils

class parser(argparse.ArgumentParser):
    def __init__(self, description):
        super(parser, self).__init__(description)

        # 权重文件路径
        self.add_argument(
            "--ckpt_file", "-cf", 
            default='./checkpoint/yolov3.ckpt-40000', type=str,
        )

        # 能识别的类别数量
        self.add_argument(
            "--num_classes", "-nc", default=80, type=int,
            help="[default: %(default)s] The number of classes ...",
            metavar="<NC>",
        )

        # anchor 文件地址
        self.add_argument(
            "--anchors_path", "-ap", default="./data/coco_anchors.txt", type=str,
            help="[default: %(default)s] The path of anchors ...",
            metavar="<AP>",
        )

        # 
        self.add_argument(
            "--weights_path", "-wp", default='./checkpoint/yolov3.weights', type=str,
            help="[default: %(default)s] Download binary file with desired weights",
            metavar="<WP>",
        )

        self.add_argument(
            "--convert", "-cv", action='store_true',
            help="[default: %(default)s] Downloading yolov3 weights and convert them",
        )

        self.add_argument(
            "--freeze", "-fz", action='store_true', default=True,
            help="[default: %(default)s] freeze the yolov3 graph to pb ...",
        )

        self.add_argument(
            "--image_h", "-ih", default=416, type=int,
            help="[default: %(default)s] The height of image, 416 or 608",
            metavar="<IH>",
        )

        self.add_argument(
            "--image_w", "-iw", default=416, type=int,
            help="[default: %(default)s] The width of image, 416 or 608",
            metavar="<IW>",
        )

        self.add_argument(
            "--iou_threshold", "-it", default=0.5, type=float,
            help="[default: %(default)s] The iou_threshold for gpu nms",
            metavar="<IT>",
        )

        self.add_argument(
            "--score_threshold", "-st", default=0.5, type=float,
            help="[default: %(default)s] The score_threshold for gpu nms",
            metavar="<ST>",
        )


def main(argv):
    flags = parser(description="freeze yolov3 graph from checkpoint file").parse_args()
    print("=> the input image size is [%d, %d]" %(flags.image_h, flags.image_w))
    anchors = utils.get_anchors(flags.anchors_path, flags.image_h, flags.image_w)
    # 生成初始模型
    model = yolov3.yolov3(flags.num_classes, anchors)

    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        # name 是 Placeholder:0
        inputs = tf.placeholder(tf.float32, [1, flags.image_h, flags.image_w, 3])
        print("=>", inputs)

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs, is_training=False)

        boxes, confs, probs = model.predict(feature_map)
        scores = confs * probs
        # concat_9 mul_6 我也不知道名字为什么是这样
        print("=>", boxes.name[:-2], scores.name[:-2])
        # 定义 cup_nms 网络输出的节点
        cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
        boxes, scores, labels = utils.gpu_nms(
            boxes,
            scores,
            flags.num_classes,
            score_thresh=flags.score_threshold,
            iou_thresh=flags.iou_threshold
        )
        # concat_10 concat_11 concat_12
        print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
        # 定义 gpu_nums 网络输出的节点
        gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

        if flags.convert:
            pass
        
        if flags.freeze:
            saver.restore(sess, flags.ckpt_file)
            print('=> checkpoint file restored from ', flags.ckpt_file)
            # 将权重和网络都保存下来，如果不明白的话可以看下面的链接
            # https://blog.csdn.net/encodets/article/details/54428456
            utils.freeze_graph(sess, './checkpoint/yolov3_cpu_nms.pb', cpu_out_node_names)
            utils.freeze_graph(sess, './checkpoint/yolov3_gpu_nms.pb', gpu_out_node_names)


if __name__ == "__main__": 
    main(sys.argv)


