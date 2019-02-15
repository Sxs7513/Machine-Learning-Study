from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from lib.config import config as cfg
from lib.layer_utils.snippets import generate_anchors_pre
from lib.layer_utils.proposal_layer import proposal_layer
from lib.layer_utils.anchor_target_layer import anchor_target_layer

class Network(object):
    def __init__(self, batch_size=1):
        # vgg16 模型总共有4个池化层，到 rpn 的时候相比原图缩小16倍
        self._feat_stride = [16, ]
        self._batch_size = batch_size
        self._predictions = {}
        self._anchor_targets = {}
        self._layers = {}
        self._losses = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._event_summaries = {}
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
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name)

    # 选取推荐的 anchor (128 个正面样本, 128个负面), 用于下一层 rcnn 的分类
    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(
                proposal_layer,
                [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32]
            )

            # 5 的原因具体看 proposal_layer 最后
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
        # 计算所有 anchor 对应的类型, 以及它们的 bounding-regression, 以及帮助计算 bbox-loss 的矩阵
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                    anchor_target_layer,
                    [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                    [tf.float32, tf.float32, tf.float32, tf.float32]
            )

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def create_architecture(self, sess, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

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

        self._add_losses()

        layers_to_output = {}

        if mode == 'TEST':
            pass
        else:
            layers_to_output.update(self._losses)

        return layers_to_output

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + 'default'):
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
            anchors, anchors_length = tf.py_func(
                generate_anchors_pre,
                [height, width, self._feat_stride, self._anchor_scales, self._anchor_ratios],
                [tf.float32, tf.int32], name="generate_anchors"
            )

            anchors.set_shape([None, 4])
            anchors_length.set_shape([])
            self._anchors = anchors
            self.anchors_length = anchors_length

    def build_network(self, sess, training):
        raise NotImplementedError
    
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_targets - bbox_pred
        # 只是为了将非前景的 bounding-regression 全部置为 0, 所以直接乘
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        # smmoth 是分段的, 所以要打成两段, 这一段是大于 1 的位置, 非大于1 的都被置为 0, 用 1 减去这个即得到另一段在哪
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1 / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        # Nreg
        out_loss_box = bbox_outside_weights * in_loss_box
        # 多维度求和(公式要求)
        loss_box = tf.reduce_sum(
            out_loss_box, 
            axis=dim
        )

        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag):
            # RPN, class loss
            # rpn_labels 是每个 anchor 的标签 ([N W H 9]) rpn_cls_score_reshape 是 [N, H * 9, W, 2]
            # softmax 是对最后一维进行处理, 所以是对每一个 anchor 都有一个 [前景概率, 背景概率]
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))

            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            # 使用 sparse_softmax_cross_entropy_with_logits, 可以直接传 [batch_size, 1] 形状的 label
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
            )

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            # dims 为 [1,2,3] 代表除了第一维度(样本数量)之外, 其他维度均加起来
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])


            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = rpn_cross_entropy + rpn_loss_box
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

    def train_step(self, sess, blobs, train_op):
        feed_dict = {
            self._image: blobs['data'], 
            self._im_info: blobs['im_info'],
            self._gt_boxes: blobs['gt_boxes']
        }

        rpn_loss_cls, rpn_loss_box, _ = sess.run([
            self._losses['rpn_cross_entropy'],
            self._losses['rpn_loss_box'],
            train_op,
        ], feed_dict=feed_dict)

        return rpn_loss_cls, rpn_loss_box