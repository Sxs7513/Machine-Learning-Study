from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from lib.config import config as cfg

class Network(object):
    def __init__(self, batch_size=1):
        self._batch_size = batch_size
        self._layers = {}
        self._act_summaries = []
        self._variables_to_fix = {}

    # https://github.com/endernewton/tf-faster-rcnn/issues/230 提出了同样的问题
    # 为什么要这么费劲的进行维度变换，暂时没搞清楚
    # 似乎看起来只是为了代码更清晰明了，经过简单测试，似乎没有发现直接 reshape 的区别
    def _reshape_layer(self, bottom, num_dims, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            # 首先把 channel 拉到第二维
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # 然后把第二维置为 2，并伸展第三维
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num_dims, -1], [input_shape[2]]]))
            # 最后再将第二维放到最后
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])

            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            # 打平，只留下最后一个维度（即 2）
            # 即等于计算每一个点提取的所有 anchor 上面所有像素的 front or back 概率
            bottom_reshaped = tf.reshape(input_shape, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name)

    def create_architecture(self, sess, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        # 每个像素 anchors 数量（3 * 3）
        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizer here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.FLAGS.weight_decay)
        if cfg.FLAGS.bias_decay:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope(
            [slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
            weights_regularizer=weights_regularizer,
            biases_regularizer=biases_regularizer,
            biases_initializer=tf.constant_initializer(0.0)
        ):
            net = self.build_network(sess, training)

        return net

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + 'default'):
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))

    def build_network(self, sess, training):
        raise NotImplementedError