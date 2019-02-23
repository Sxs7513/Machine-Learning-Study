from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


def demo(sess, net, image_name):
    im_file = os.path.join(cfg.FLAGS2["data_dir"], demo, image_name)
    im = cv2.imread(im_file)

    # 获得所有推荐框以及它们的得分
    time = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # 非极大值抑制之前，选择要保留下来的框
    CONF_THRESH = 0.1
    # 非极大值抑制的阀值
    NMS_THRESH = 0.1
    # 针对每一类做nms
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        # 
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        # 
        cls_scores = scores[:, cls_ind]



if __name__ == "__main__":
    args = parse_args()

    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', NETS[demonet][0])
    
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)
    
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    # net.create_architecture(sess, "Train", 21, tag='default', anchor_scales=[8, 16, 32])
    # saver = tf.train.Saver()
    # saver.restore(sess, tfmodel)

    # print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['000456.jpg', '000457.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    # for im_name in im_names:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for data/demo/{}'.format(im_name))
    #     demo(sess, net, im_name)