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
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.layer_utils.proposal_top_layer import proposal_top_layer
from lib.layer_utils.anchor_target_layer import anchor_target_layer

class Network(object):
    def __init__(self, batch_size=1):
        # vgg16 模型总共有4个池化层，到 rpn 的时候相比原图缩小16倍
        self._feat_stride = [16, ]
        self._feat_compress = [1. / 16., ]
        self._batch_size = batch_size
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
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
        return tf.nn.softmax(bottom, name=name)

    # 使用经过 rpn 网络层后生成的 rpn_bbox_prob 把 anchor 位置进行第一次修正
    # 按照得分排序，取前 12000 个anchor，再 nms,取前面 2000 个
    # 但是这个数字在test的时候就变成了 6000 和 300，这就是最后结果 300 个框的来源
    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(
                proposal_layer,
                [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32]
            )

            # shape 5 的原因具体看 proposal_layer 最后
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    # 为 rpn 网络的训练准备数据，包括最终进入 rpn 训练的 anchors（256个），它们的 bounding-Regression
    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
            # 计算所有 anchor 对应的类型, 以及它们的 bounding-regression, 以及帮助计算 bbox-loss 的矩阵
            # 注意只有 256 个会进入 rnp-loss 计算
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
            # 训练所需数据缓存起来
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    # 获得属于最后的分类网络的 label，使用的是 _proposal_layer 提供的数据，进行进一步的筛选
    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            )

            rois.set_shape([cfg.FLAGS.batch_size, 5])
            roi_scores.set_shape([cfg.FLAGS.batch_size])
            labels.set_shape([cfg.FLAGS.batch_size, 1])
            bbox_targets.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])

            # 缓存起来，供 fast-rcnn 网络使用
            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores
    
    # 用于 test 时候的推荐 anchor 选取，直接选取高得分的即可
    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.FLAGS.rpn_top_n, 5])
            rpn_scores.set_shape([cfg.FLAGS.rpn_top_n, 1])

        return rois, rpn_scores

    # 没有用论文里的 roi-polling，而是直接 resize-crop 了
    # bottom 就是共享的 conv_5 网络的输出
    # 从网路的输出中，从 rois 坐标中截取并 resize 成相同的大小
    # 输出的 shape 为 (256, 7, 7, 512)， 256 为 rois 的个数，512 为 conv_5 的输出特征图个数
    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            # tf.image.crop_and_resize 函数需要传入的 boxes 归一化
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

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
        # conv_5 中每个像素 anchors 数量（3 * 3）
        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizer here
        # 添加 L2 正则项
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.FLAGS.weight_decay)
        if cfg.FLAGS.bias_decay:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        # 使用 slim 提供的 arg_scope 来简化网络层参数的写法, 赋予一些默认参数
        # 创建所有网络，包括 rpn 网络， fast-rcnn 网络
        with arg_scope(
            [slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
            weights_regularizer=weights_regularizer,
            biases_regularizer=biases_regularizer,
            biases_initializer=tf.constant_initializer(0.0)
        ):
            rois, cls_prob, bbox_pred = self.build_network(sess, training)

        # rois 是最终进入 fast-rcnn 网络的 anchor
        # 将 _predictions 缓存起来，供 train.py 调用
        layers_to_output = {'rois': rois}
        layers_to_output.update(self._predictions)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds"]), (self._num_classes))
            means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means"]), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            # 训练模式下生成损失函数
            self._add_losses()
            layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_image_summary(self._image, self._gt_boxes))
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    # 在 conv5 给予的特征图上提取 anchors，注意 anchors 的位置是原始图的上面的哦
    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + 'default'):
            # just to get the shape right
            # 精确获得进入 vgg16 网络前图片的大小（不是原图哦）
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
            # 生成 conv_5 视野下的所有点对应的 anchor，anchors 是二维，shape 为 [height * width * 9, 4]
            anchors, anchors_length = tf.py_func(
                generate_anchors_pre,
                [height, width, self._feat_stride, self._anchor_scales, self._anchor_ratios],
                [tf.float32, tf.int32], name="generate_anchors"
            )
            # 必须的操作，经测试发现不重新 set_shape 一遍训练会出错
            anchors.set_shape([None, 4])
            anchors_length.set_shape([])
            # 挂载生成的 anchors
            self._anchors = anchors
            self.anchors_length = anchors_length

    def build_network(self, sess, training):
        raise NotImplementedError
    
    # bbox-regression 的损失函数
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    # 生成损失函数
    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag):
            # RPN, class loss
            # rpn_labels 是每个 anchor 的标签 ([N W H 9]) rpn_cls_score_reshape 是 [N, H * 9, W, 2]
            # softmax 是对最后一维进行处理, 所以是对每一个 anchor 都有一个 [前景概率, 背景概率]
            # 注意真正进行了loss计算的只有那 256 个 anchor
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            # 非背景和非前景均忽略
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))

            # 利用 gather 选取出来要训练的数据
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

            # RCNN, class loss
            # 同理
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = tf.reshape(cls_score, [-1, self._num_classes]),
                    labels=label
                )
            )

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            # 总损失函数
            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    # 训练函数，计算四个损失函数的过程中，等于前向传播
    # train_op 即为自动求导，反向传播过程
    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run(
            [self._losses["rpn_cross_entropy"],
            self._losses['rpn_loss_box'],
            self._losses['cross_entropy'],
            self._losses['loss_box'],
            self._losses['total_loss'],
            train_op],
            feed_dict=feed_dict
        )

        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image, self._im_info: im_info}

        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    # Summaries #
    def _add_image_summary(self, image, boxes):
        # add back mean
        image += cfg.FLAGS2["pixel_means"]
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        #assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image('ground_truth', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)