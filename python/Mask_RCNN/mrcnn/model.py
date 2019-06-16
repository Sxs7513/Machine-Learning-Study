import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from mrcnn import utils

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess) 

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


# 为什么这么写的原因可以看下面的链接
# https://zhuanlan.zhihu.com/p/56225304
# 大体就是 keras 中的 BN 层会有一些 bug，需要 fix 它
class BatchNorm(KL.BatchNormalization):
    # 模拟 keras 的调用方式
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)


# 计算 image 在经过 resnet 后，各个 feature_map 的大小
# 总共要计算 5 个 feature_map
def compute_backbone_shapes(config, image_shape):
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array([
        [int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
        for stride in config.BACKBONE_STRIDES
    ])    



############################################################
#  Resnet Graph
############################################################

# identity_block 与 conv_block 的区别在于 identity_block 的旁路是直接一条线，conv_block 的旁路有一个卷积层
# 有这样的区别是为了保证旁路出来的 featuremap 和主路的 featuremap 尺寸一致，这样它们才能相加
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    # x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


# block，在 yolo_v3 中也有应用
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    # x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 
    x = KL.Conv2D(filters=nb_filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # 
    x = KL.Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # strides=(1, 1) 是默认的
    x = KL.Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    # shortcut 也不用多说, 残差网络核心
    shortcut = KL.Conv2D(nb_filter3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    # 直接相加，注意不是 concat，yolo_v3 中是 concat
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    # [N, 3, 1030, 1030] 注意 padding 会在上下左右都加，所以乘以 2
    x = KL.ZeroPadding2D(padding=(3, 3))(input_image)
    # 默认 valid 结合 2，2 的卷积滑动步长可以达到池化层的效果，分辨率除以 2
    # 卷积输出大小的计算在有填充的时候是（W-ketnalSize+2P）/S+1，没有零填充的时候，其计算为（W-ketnalSize+1）/S
    # [N, 512, 512, 64]  
    x = KL.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name="bn_conv1")(x, training=train_bn)
    x = KL.Activation("relu")(x)
    # [N, 256，256，64] 输出大小计算公式与卷积层一样
    C1 = x = KL.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    # (x => [N, 256，256，64] => [N, 256，256，64] => [N, 256，256，256]) + (x => [N, 256，256，256] shortcut) => [N, 256，256，256]
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    # (x => [N, 256，256，64] => [N, 256，256，64] => [N, 256，256，256]) + (x) => [N, 256，256，256]
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # [N, 256，256，256]
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    # (x => [N, 128，128，128] => [N, 128，128，128] => [N, 128，128，512]) + (x => [N, 128，128，512] shortcut) => [N, 128，128，512]
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    # (x => [N, 128，128，128] => [N, 128，128，128] => [N, 128，128，512]) + (x) => [N, 128，128，512]
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    # [N, 128，128，512]
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # [N, 128，128，512]
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    # [N, 64, 64, 1024]
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    # [N, 64, 64, 1024]
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        # [N, 32, 32, 2048]
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]



############################################################
#  Proposal Layer
############################################################

# 作用是将预测的边框回归应用在 anchor 上面, 获得 anchor 经回归后的坐标
# boxes => 在图片大小为 1024 * 1024 的情况下为 [261888, 4] (y1, x1, y2, x2)
# deltas => [261888, 4] (dy, dx, log(dh), log(dw)) 
def apply_box_deltas_graph(boxes, deltas):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # apply deltas
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=-1, name="apply_box_deltas_out")
    return result


# 作用是保证 boxes 都在原图内，不会出现负数
# boxes => [261888, 4]
# window => np.array([0, 0, 1, 1])
def clip_boxes_graph(boxes, window):
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)

    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


# ProposalLayer 网络层，属于自定义层，起到初步选择推荐框的作用
class ProposalLayer(KE.Layer):
    # proposal_count => 经过 nms 之后保留多少 ROIs
    # nms_threshold => nms 阈值, 不必多说
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    
    # inputs => [rpn_class, rpn_bbox, anchors]
    # rpn_class => [N, AV, 2]  AV 是五个 feature_map 所有 anchors 的数量
    # rpn_bbox => [N, AV, 4]
    # anchors => 在图片大小为 1024 * 1024 的情况下为 [N, 261888, 4]
    def call(self, inputs):
        # 取出前景得分 [N, AV]
        scores = inputs[0][:, :, 1]
        # 取出 bbox 预测, 并
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # 取出 anchors
        anchors = inputs[2]

        # 保险起见, 一般不会发生, 总不至于 6000 个 anhcor 都提取不出来吧?
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # 找到分最高的 anchor 的 index
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        # tf.gather 不支持 batch, 因为它只支持单维度的切片, 所以 hack 下, 选择出来对应的得分
        # 不过这里为社么不用 tf.gather_nd ? 之后可以来实验一下
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        # 同理, 对应的 bbox
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        # 对应的 anchor
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])
        # 到此, 已选出来 N * PRE_NMS_LIMIT 个 anchor 及其对应的前景得分与 bbox

        # 将预测的边框回归应用到 boxes 上面
        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas],
            lambda x, y:apply_box_deltas_graph(x, y),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors"]
        )

        # 保证 boxes 区域都在正常范围内
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors_clipped"]
        )

        # 对每个 batch 进行非极大值抑制
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression"
            )
            proposals = tf.gather(boxes, indices)
            # 如果数量不够的话，计算填充几个空的推荐框
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            # 表示增加多少个，可以用 numpy 试一下就知道怎么回事了，tf 和 numpy 这个方法实现一样
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)

        return proposals

    
    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

# log2, 即 x = 2 的时候输出为 1
def log2_graph(x):
    return tf.log(x) / tf.log(2.0)


# 自定义层，RoIAlign, 它的优势网上有很多文章
# http://blog.lkj666.top/2018/09/20/Mask%20R-CNN%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/
# https://zhuanlan.zhihu.com/p/37998710
class PyramidROIAlign(KE.Layer):
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    
    # inputs => [[N, 200, 4], [N, 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES], [N, 256，256，256], [N, 128，128，256], [N, 64，64，256], [N, 32，32，256]]
    def call(self, inputs):
        boxes = inputs[0]
        image_meta = inputs[1]
        feature_maps = inputs[2:]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        # [N, 200, 1]，因为 tf.split 不会降维
        h = y2 - y1
        w = x2 - x1

        # [1024, 1024]
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # [N]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # 用于根据 ROI 大小来分配不同的 feature_map, 因为在提取 anchor 的时候
        # 大的 feature_map 提取的 anchor 都小, 反之...
        # https://blog.csdn.net/pangsmao/article/details/81952495
        # 首先看和 224 比它是大还是小, 注意 h w 已经归一化，所以会像下面这样除
        # 引入 log 是为了引入负数，方面下面直接加上 k0 即基数
        # [N, 200, 1]
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # 然后首先确保值在 2-5 之间(inputs从 2-5 为 feature_map)
        # 计算应该输入哪个 feature_map, 如果莫个 roi 大小是 112 * 112
        # 那么它应该被归给 64 * 64 这个
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # 去掉第三维度 [N, 200]
        roi_level = tf.squeeze(roi_level, 2)

        pooled = []
        box_to_level = []
        # 遍历每个 feature_map
        for i, level in enumerate(range(2, 6)):
            # 找到每个 feature_map 都与哪些 roi 匹配的坐标
            # tf.where 返回值格式 [坐标1(第几个N, 第几个box), 坐标2,……]
            # np.where 返回值格式 [[坐标1.x, 坐标2.x……], [坐标1.y, 坐标2.y……]]
            # [num, 2]
            ix = tf.where(tf.equal(roi_level, level))
            # 获取到这些 rois, 注意带有 batch 维度
            # [N, featurex_boxes_num, 4]
            level_boxes = tf.gather_nd(boxes, ix)

            box_indices = tf.cast(ix[:, 0], tf.int32)
            # 将对应 rois 的位置坐标缓存起来，ix 是个数组
            box_to_level.append(ix)

            # level_boxes 和 box_indices 本身属于 RPN 计算出来结果
            # 但是两者作用于 feature 后的输出 Tensor 却是 RCNN 部分的输入
            # 但是两部分的梯度不能相互流通的，所以需要 tf.stop_gradient() 截断梯度传播
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # 这里并没有采用论文中提到的 4 个采样点的方法，而是采用了论文中提到的也非常有效的 1 个点采样的方法
            # 由于 crop_and_resize 默认使用双线性插值，即将格子中心的插值结果做为输出，而不是取四个点再做池化
            # 即首先剪切出来 roi，然后使用双线性差值到固定大小比如 7 * 7。原来在 4 个采样点的时候，先双线性差值
            # 出来四个点，然后 max-pool 来获得该区域的最大值，在一个采样点的时候，就等于先在区域中心双线性差值
            # 然后直接取该点即可！
            # http://blog.lkj666.top/2018/09/20/Mask%20R-CNN%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/
            # 关于双线性差值法，可以看下面的链接
            # https://blog.csdn.net/u013883974/article/details/76812735
            # https://blog.csdn.net/u012193416/article/details/86525411
            # [featurex_boxes_num * N, 7, 7, 256]
            pooled.append(
                tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear")
            )
        # 遍历完毕后，pooled => [[feature1_boxes_num * N, 7, 7, 256], [N, feature2_boxes_num * N, 7, 7, 256], ...]
        # box_to_level => [[num1, 2], [num2, 2], [num3, 2], [num4, 2]]

        # 将所有结果合并起来，没有 batch 区分
        # [N * 200, 7, 7, 256]
        pooled = tf.concat(pooled, axis=0)
        # 同上, [N * 200, 2]
        box_to_level = tf.concat(box_to_level, axis=0)
        # [N * 200, 1]
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        # [N * 200, 3]
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # 挺好玩的，为了再把 batch 分出来以及不择手段了，top_k 具有 sort 的功能
        # 通过取它的 indices 结合下面的 reshape 即可重新分出来 batch
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        # [N * 200, 7, 7, 256]
        pooled = tf.gather(pooled, ix)

        # 值为 [N, 200, 7, 7, 256]
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        # [N, 200, 7, 7, 256]
        pooled = tf.reshape(pooled, shape)
        return pooled


    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################

# 输出 tensor 的计算 iou 的方法
def overlaps_graph(boxes1, boxes2):
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


# proposals => [proposal_count, 4]
# gt_class_ids => [100]
# gt_boxes => [100, 4]
# gt_masks => [1024(28), 1024(28), 100]
def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion")]
    
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # 去掉坐标为 0 的 proposals
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    # gt_boxes 同样, 这个是必须的, 因为 data_generator 的时候, 会填充很多空的
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    # 同样
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    # 同样, 注意 tf.where 与 np.where 不一样, 具体看 http://www.studyai.com/article/3c11b2eb
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")

    # 把非重叠的 truth-box 筛选出来
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # 计算每个推荐框与 truth-boxes 的 iou
    # [num_proposals, num_gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # 计算每个推荐框与重叠框的 iou, 标记小于 0.001 的为正常框
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # 找到每个推荐框的最大重叠率
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 标记大于 0.5 的为正预测
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 标记小于 0.5 并且与重叠框不交集的为负预测
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # 正预测的数量
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    # 正预测的序号
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # 正样本真实的数量, 可能存在小于 200 * 0.33
    positive_count = tf.shape(positive_indices)[0]

    # 负预测的数量是正预测的 1.0 / config.ROI_POSITIVE_RATIO - 1 倍
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    # 取得需要得正负预测
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    positive_overlaps = tf.gather(overlaps, positive_indices)
    # 找到正预测里每个和哪个 truth-box 最接近
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]), tf.int64)
    )
    # 同上
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # [num, 4]
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # 主要是用来在下面的crop_and_resize，需要伪造一个通道 [num, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # 取出来对应的 masks
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    boxes = positive_rois
    if config.USE_MINI_MASK:
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    # 给这些正预测做一个新的编号
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    # 这一步针对没有开启 USE_MINI_MASK 的是有用的
    # 但是 USE_MINI_MASK 是默认开启的，所以 mask 当前已经
    # 是最精简模式
    masks = tf.image.crop_and_resize(
        tf.cast(roi_masks, tf.float32), 
        boxes,
        box_ids,
        config.MASK_SHAPE
    )

    # 把上面的那个 expand_dims 的去掉，已经没有利用价值啦
    masks = tf.squeeze(masks, axis=3)
    # 四舍五入，因为上面的 crop_and_resize 是双线性差值, 那么 mask 中肯定会产生浮点数
    # 但是 mask 是需要只有 0 1 的, 所以处理一下
    masks = tf.round(masks)

    # [200(可能小于这个数), 4]
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    # 不足的用 0 来补, [200, 4]
    rois = tf.pad(rois, [(0, P), (0, 0)])
    # [200, 4]
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    # [200, 1]
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    # [200, 4]
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    # [200, height, width]
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


# 自定义层, 用于筛选出来 rois, 用于损失计算
class DetectionTargetLayer(KE.Layer):
    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    
    def call(self, inputs):
        # [N, proposal_count, 4]
        proposals = inputs[0]
        # [N, 100]
        gt_class_ids = inputs[1]
        # [N, 100, 4]
        gt_boxes = inputs[2]
        # [N, 1024(28), 1024(28), 100]
        gt_masks = inputs[3]

        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names
        )
        # outputs => [[N, 200, 4], [N, 200], [N, 200, 4], [N, 200, height, width]]
        return outputs


    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_ROIS_PER_IMAGE),
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], self.config.MASK_SHAPE[1])
        ]


    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]



############################################################
#  Detection Layer
############################################################

# rois => [roi_num, 4]
# probs => [roi_num, num_classes]
# deltas => [roi_num, 4]
# window => [4] window 具体含义看 utils 模块的 resize_image
# 但是注意在这里它是相对于缩放后的图像大小的，所以是小数
def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # [roi_num]
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # 自己写个用例试一下就知道了，为了配合 gather_nd 来获取到对应的 box
    # [roi_num, 2]
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    # [roi_num]
    class_scores = tf.gather_nd(probs, indices)
    # [roi_num]
    deltas_specific = tf.gather_nd(deltas, indices)
    # bbox 应用到 rois 上面
    # [roi_num, 4]
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV
    )
    # test 阶段要严格一些，需要保证 box 都在 window 内
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(
            tf.expand_dims(keep, 0),
            tf.expand_dims(conf_keep, 0)
        )
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD
        )
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(
            class_keep, [(0, gap)],
            mode='CONSTANT', constant_values=-1
        )
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(
        nms_keep_map, unique_pre_nms_class_ids,
        dtype=tf.int64
    )
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(
        tf.expand_dims(keep, 0),
        tf.expand_dims(nms_keep, 0)
    )
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat(
        [
            tf.gather(refined_rois, keep),
            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], 
        axis=1
    )

    # 如果选出的框不够，用 0 来填充
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        # [N, proposal_count, 4]
        rois = inputs[0]
        # [N, proposal_count, num_classes]
        mrcnn_class = inputs[1]
        # [N, proposal_count, num_classes, 4]
        mrcnn_bbox = inputs[2]
        # 图片缩放的信息，与数据集所有的类别 [N, 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES]
        image_meta = inputs[3]

        
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        # [N, num_detections, (y1, x1, y2, x2, class_id, class_score)]
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6]
        )

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)



############################################################
#  Region Proposal Network (RPN)
############################################################

# 与 Faster-RCNN 原理一致, 里面不做多解释
# feature_map => [N, None, None, 256]
# anchors_per_location => 默认为 3
# anchor_stride => 默认为 1
def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    # 与 Faster-RCNN 一致, 先经过一个卷积层, 原因未知
    shared = KL.Conv2D(512, (3, 3), padding="same", activation='relu', strides=anchor_stride, name='rpn_conv_shared')(feature_map)
    # 与 Faster-RCNN 一致, 前景背景得分, 每个 anchor 预测两个分
    # [N, height, width, 2 * anchors_per_location]
    x = KL.Conv2D(2 * anchors_per_location, kernel_size=(1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

    # [N, V, 2]  V 是该特征图里 anchor 的数量
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    # [N, V, 2] 
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # [N, height, width, 4 * anchors_per_location]
    x = KL.Conv2D(anchors_per_location * 4, kernel_size=(1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)
    # [N, V, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


# 调用 rpn_graph 生成 rpn 网络, 然后封装成模型, 用于为 ProposalLayer 层
# 提供寻找推荐框的数据, 与 Faster-RCNN 基本一致
# anchor_stride => 每隔几个 cell 创建 anchors，默认为 1
# anchors_per_location =>  每个 cell 提取的几个 anchor, 默认为 3
# depth => 特征图的层数, 均为 256 层
def build_rpn_model(anchor_stride, anchors_per_location, depth):
    input_feature_map = KL.Input(
        shape=[None, None, depth],
        name="input_rpn_feature_map"
    )
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

# 用于输出 rois 创建分类与边框回归的预测值
# rois => [N, 200(或者2000), 4]
# feature_maps => [[N, 256，256，256], [N, 128，128，256], [N, 64，64，256], [N, 32，32，256]]
# image_meta => [N, 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES]
# pool_size => [7, 7]
# num_classes => 81
def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    # ROIAlign 层，将所有 rois 统一为 7 * 7 大小
    # x => [N, 200, 7, 7, 256]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # [N, 200, 1, 1, 1024]
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation("relu")(x)
    # [N, 200, 1, 1, 1024]
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # 起到打平的效果 [N, 200, 1024] 
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # 用于分类 [N, 200, num_classes]
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # 用于边框回归, 注意这里对每一类都预测了回归, 这样可以达到避免同类竞争的目的
    # [N, 200, num_classes * 4]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    s = K.int_shape(x)
    # [N, 200, num_classes, 4], 注意 Reshape 不包含 N 维
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


# 用于输出 rois 的 mask 的预测值
# rois => [N, 200, 4]
# feature_maps => [[N, 256，256，256], [N, 128，128，256], [N, 64，64，256], [N, 32，32，256]]
# image_meta => [N, 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES]
# pool_size => [7, 7]
# num_classes => 81
def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    # ROIAlign 层，将所有 rois 统一为 14 * 14 大小
    # x => [N, 200, 14, 14, 256]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

    # 一系列的卷积层, 保持滑动窗口大小均为 (3, 3), 不会改变大小
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # 反卷积, 扩大大小, [N, 200, 28, 28, 256], 公式未 new = s(i−1) + k − 2p, i 为图大小, k 为卷积核大小, p 为 padding
    # https://www.cnblogs.com/cvtoEyes/p/8513958.html
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    # 对每一类都输出一个 mask 预测, 这样可以避免不同实例之间的类别竞争
    # 并且注意 mask 需要用 sigmoid 来激活
    # [N, 200, 28, 28, num_classes]
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)

    return x



############################################################
#  Loss Functions
############################################################

# 计算边框回归的坐标损失, 下面的链接说明了为社么用 L1 损失
# https://zhuanlan.zhihu.com/p/48426076
# y_true => [num, 4]
# y_pred => [num, 4]
def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    
    return loss


# 计算 rpn 的前景背景分类的损失
# rpn_match => 前景背景分类真实值 [N, 216888, 1]
# rpn_class_logits => 前景背景分类预测值 [N, 216888, 2]
def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    # [N, 216888]
    rpn_match = tf.squeeze(rpn_match, -1)
    # 找到前景 anchor 位置, 并将它们标记为 1, 其他的为 0
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # 找到所有前景背景 anchor 的位置
    indices = tf.where(K.not_equal(rpn_match, 0))
    # 找到对应的预测值 [N * 256, 2], tf.gather_nd 与 tf.gather 区别如下
    # https://zhuanlan.zhihu.com/p/51446095
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    # [N * 256], 前景为 1, 背景为 0
    anchor_class = tf.gather_nd(anchor_class, indices)

    # softmax 激活, 交叉熵计算损失函数
    loss = K.sparse_categorical_crossentropy(
        target=anchor_class,
        output=rpn_class_logits,
        from_logits=True
    )
    # 应该是为了减少计算吧, 在没有正负样本的时候, 直接忽略....
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

    return loss


# 计算 rpn 的边框回归的损失, 只有 正anchor 会进入计算
# target_bbox => [N, 256, 4]
# rpn_match => [N, 261888, 1]
# rpn_bbox => [N, 261888, 4]
def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))
    # 注意只有 正anchors 会进入损失计算, [N * 正面数量, 4]
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # 找到每个 batch 里面 正anchor 的个数, [N, 1]
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    # 将所有 正achor 抽取出来, [N * 128(正的个数为一半), 4]
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    # 计算损失
    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


# 计算 rois 分类损失
# target_class_ids => [N, 200]
# pred_class_logits => [N, 200, num_classes]
# active_class_ids => [N, num_classes]
def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    target_class_ids = tf.cast(target_class_ids, "int64")
    # 找到预测的 rois 的类别, [N, 200]
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # 如果预测的类别在数据集中, 那么为 1, 否则为 0
    # [N, 200], 为社么是这个 shape 是由于 gather 的特性导致的, 实在不理解可以自己在 test 里面试试
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # softmax交叉熵计算损失, [N, 200]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)

    # 保证这个框最大的得分 class 如果不属于其数据集，则不进入本框 Loss
    loss = loss * pred_active

    # 计算平均, 只除以在本数据集的类别数 [N * 200 * loss] / [N * 200 * (0或者1)] = [1] 
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)

    return loss


# 计算 rois 的边框回归
# target_bbox => [N, 200, 4]
# target_class_ids => [N, 200]
# pred_bbox => [N, 200, num_classes, 4]
def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    target_class_ids = K.reshape(target_class_ids, (-1, ))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    # [N * 200, num_classes, 4]
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # 找到正样本的序号
    # tf.where(target_class_ids > 0) => [N * 正个数, 1]
    # [N * 正个数]
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # 找到对应的类别 [N * 正个数]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    # [N * 正个数, N * 正个数]
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # [N * 正个数, 4]
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    # 取出来预测的对应类别的 bbox, 因为每个类别都会预测一个 bbox
    # [N * 正个数, 4]
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # [N * 正个数, 4]
    loss = K.switch(
        tf.size(target_bbox) > 0,
        smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
        tf.constant(0.0))

    # K.mean 是将所有数字加起来, 然后除以它们的个数
    loss = K.mean(loss)

    return loss


# 计算 mask 的损失
# target_masks => [N, 200, 28, 28]
# target_class_ids => [N, 200]
# pred_masks => [N, 200, 28, 28, 81]
def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    # [N * 200]
    target_class_ids = K.reshape(target_class_ids, (-1, ))
    mask_shape = tf.shape(target_masks)
    # [N * 200, 28, 28]
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    # [N * 200, 28, 28, 81]
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # [N * 200, 81, 28, 28]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # 下面与 mrcnn_bbox_loss_graph 逻辑基本一致
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # [N * 正个数, 28, 28]
    y_true = tf.gather(target_masks, positive_ix)
    # [N * 正个数, 28, 28]
    y_pred = tf.gather_nd(pred_masks, indices)

    # 使用 sigmiod 来计算损失, 因为 mask 真值都是由 0 1 组成的 
    loss = K.switch(
        tf.size(y_true) > 0,
        K.binary_crossentropy(target=y_true, output=y_pred),
        tf.constant(0.0))

    loss = K.mean(loss)

    return loss



############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
    # 加载图片
    image = dataset.load_image(image_id)
    # 获取图片中的掩膜位置信息，掩膜类别
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    # 缩放图像同时保持宽高比不变
    # image 是 resize 后的图片
    # window 如果给出了max_dim, 可能会对返回图像进行填充, window 代表经过缩放后
    # 的图片在 max_dim 中的实际位置，左上角与右下角坐标
    # scale 是图像缩放因子
    # padding: 图像填充部分[(top, bottom), (left, right), (0, 0)]
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE
    )
    # 掩膜也是图片, 它也要 resize
    mask = utils.resize_mask(mask, scale, padding, crop)

    # 如果要图片增强, 那么进入下面逻辑
    if augmentation:
        import imgaug

        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        image_shape = image.shape
        mask_shape = mask.shape
        # 固定变换序列,之后就可以先变换图像然后变换关键点,这样可以保证两次的变换完全相同
        # 如果调用次函数,需要在每次 batch 的时候都调用一次,否则不同的 batch 执行相同的变换
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # 有些掩膜是空的, 排除它们
    _idx = np.sum(mask, axis=(0, 1)) > 0
    # mask 的格式很特殊，看 coco.py 里面的 load_mask 便知道了
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    # 从掩膜数据中, 直接提取出来该图片的 truth-box 的位置大小
    # [该图中 truth-box 的数量, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # 该图片隶属数据集中所有的 class 标记为 1，不隶属本数据集合的 class 标记为0
    # 这个是为了在 mrcnn_class_loss_graph 计算分类损失的时候, 保证这个框最大的得分 class 如果不属于其数据集，则不进入本框 Loss
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # 有需要的话，把 mask 缩小，来节省内存，之后可以复原的
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # 把图片的一些信息缓存下来，具体看该方法
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


# 
# image => 经过缩放后的图像，anchors => 该图像提取的所有 anchor
# gt_class_ids => 图片中所有的 truth-boxes 类别
# gt_boxes => 图片中所有的 truth-boxes 坐标
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    # 用于标记每个 anchor 是否是正负样本, [num_anchors]
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # 正样本 anchors 的 bbox 回归值, [256, 4]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # coco中重叠框的问题，要剔除掉
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # 计算重叠boxes 和 anchors 的 iou
        # [num_anchors, num_crowd_boxes]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        # 找到第二维度的最大值，即 anchors 与所有重叠框的最大 iou
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        # 然后将重叠小于 0.001 的标记为 true，即代表正常的 anchors，后面会用到
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # 如果没有重叠框，那么标记所有 truth-boxes 都是 true，后面会用到
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # 计算 anchors 与所有 truth-boxes 的 iou
    # [num_anchors, num_truth-boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # 找到每个 anchor 与哪个 truth-boxes 的 iou 最大 
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    # [num_anchors, 1], 不多解释
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    # 对于最大 iou 小于 0.3 的 anchor，和与重叠框有较大重叠的 anchor，都标记为背景
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 扎到每个 truth-box 与哪个 anchor 最接近。这里可能存在重复
    # 比如第一个和第二个 truth-box 与 第三个 anchor 都最接近
    # 这里计算比较难懂，可以自己随便弄个矩阵看看
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    # 与 Faster-RCNN 一致, 这个不管重复率如何, 都要标记为正样本
    rpn_match[gt_iou_argmax] = 1
    # 再标记大于 0.7 的为正样本
    rpn_match[anchor_iou_max >= 0.7] = 1
    # 到此, rpn_match 已经生成完毕

    # 定位到所有正样本位置
    ids = np.where(rpn_match == 1)[0]
    # 为了保证正样本数量不大于 256 的一半, 如果大于, 那么将 rpn_match 中多余的标记为啥都不是
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # 负样本一样
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == 1)[0]
    ix = 1
    # 填充 rpn_bbox, 只会填正样本
    for i, a in zip(ids, anchors[ids]):
        # 找到每个正样本与哪个 truth-box 最接近
        # [y1, x1, y2, x2]
        gt = gt_boxes[anchor_iou_argmax[i]]

        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    
    return rpn_match, rpn_bbox


def data_generator(
    dataset, config, shuffle=True, augment=False, augmentation=None,
    random_rois=0, batch_size=1, detection_targets=False, no_augmentation_sources=None
):
    b = 0
    image_index = -1
    # 图片的id, 注意不是数据集原有的 id, 是重新定义的新的 id, 从 0 到 len(图片)
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # 经过 resnet 后提取的五个特征图的大小, 以 resnet-101 举例
    # [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)

    # 生成五个特征图的所有 anchors，大小位置相对于1024 * 1024, a => [261888 , 4]
    anchors = utils.generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES,
        config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE
    )

    while True:
        try:
            # image_index 逐渐递增, 当到极限的时候, 又变回 0
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)
            
            image_id = image_ids[image_index]

            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
                    dataset, config, image_id, augment=augment,
                    augmentation=None,
                    use_mini_mask=config.USE_MINI_MASK
                )
            else:
                # image => 缩放后的图片 [1024, 1024, 3]
                # image_meta => 图片缩放的信息，与数据集所有的类别, [1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES]
                # gt_class_ids => 图片中 truth-boxes 的类别 [num_truth_boxes, 1]
                # gt_boxes => 图片中的 truth-boxes 坐标 [num_truth_boxes, 4]
                # gt_masks => 掩膜 [height, width, num_mask]
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
                    dataset, config, image_id, augment=augment,
                    augmentation=augmentation,
                    use_mini_mask=config.USE_MINI_MASK
                )

            if not np.any(gt_class_ids > 0):
                continue

            # rpn_match => 用于标记每个 anchor 是否是正负样本, 其中正负样本数总和为 256, [num_anchors(261888)], 非正负样本的值均为 0
            # rpn_bbox => 正样本 anchors 的 bbox 回归值, [256, 4]
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)

            # 
            if random_rois:
                pass

            # 初始化的时候, 生成 batch 矩阵, 不必多说
            if b == 0:
                # [N, 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES]
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                # [N, 261888, 1]
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                # [N, 256, 4]
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                # [N, height, width, 3]
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                # [N, 100], 但是这个 100 是最大值, 基本填不满的
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                # [N, 100, 4] 与上面一样
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                # [N, height, width, 100] 同上
                batch_gt_masks = np.zeros(
                    (
                        batch_size, gt_masks.shape[0], gt_masks.shape[1],
                        config.MAX_GT_INSTANCES
                    ), 
                    dtype=gt_masks.dtype
                )
                if random_rois:
                    pass
            
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            
            if random_rois:
                pass
            b += 1

            # 当填满一个 batch, 那么把上面生成的数据拼起来返回
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    pass
                # outputs 为空即可, loss 函数不依赖于它
                yield inputs, outputs

                # 很关键, 每填满一个batch必须要记得重置为0
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    # model_dir => 存储模型的位置
    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    
    def build(self, mode, config):
        assert mode in ['training', 'inference']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )
        
        # 本次处理的图片 [N, height, width, 3]
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        # 图片缩放的信息，与数据集所有的类别 [N, 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES]
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        if mode == "training":
            # 下面凡是 Input 的都是真实值, 需要 data_generator 构建并输入的

            # 用于标记每个 anchor 是否是正负样本, 其中正负样本数总和为 256, [N, num_anchors(261888), 1], 非正负样本的值均为 0
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            # rois 的 bbox 回归值, [N, 256, 4]
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            
            # [N, 100] 但是这个 100 是最大值, 基本填不满的
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # [N, 100, 4] 但是这个 100 是最大值, 基本填不满的
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # [N, 100, 4] 将 truth-boxes 归一化
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
            
            # 针对是否开启缩小 mask
            if config.USE_MINI_MASK:
                # [N, 28, 28, 100]
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                # [N, 1024, 1024, 100]
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool
                )

        elif mode == "inference":
            # [N, 261888, 4]
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # 构建基础网络，这里默认使用 resnet101
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, 
                stage5=True,
                train_bn=config.TRAIN_BN
            )
        else:
            # C2 => [N, 256，256，256]
            # C3 => [N, 128，128，512]
            # C4 => [N, 64, 64, 1024]
            # C5 => [N, 32, 32, 2048]
            _, C2, C3, C4, C5 = resnet_graph(
                input_image, 
                config.BACKBONE,
                stage5=True, 
                train_bn=config.TRAIN_BN
            )

        # 获得最终的多层特征图，把高层的特征传下来，补充低层的语义，这样就可以获得高分辨率、强语义的特征，有利于小目标的检测
        # [N, 32, 32, 256]
        P5 = KL.Conv2D(filters=config.TOP_DOWN_PYRAMID_SIZE, kernel_size=(1, 1), name='fpn_c5p5')(C5)
        # [N, 64, 64, 256] + [N, 64, 64, 256] = [N, 64, 64, 256]    
        P4 = KL.Add(name="fpn_p4add")([
            # 上采样，注意 keras 在这里只会简单重复一下而已，没有复杂的算法
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)
        ])
        # [N, 128, 128, 256] + [N, 128，128，256] = [N, 128, 128, 256]
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)
        ])
        # [N, 256，256，256] + [N, 256，256，256] = [N, 256，256，256]
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)
        ])

        # 上面生成的所有特征图全部走一遍 3 * 3 卷积，全部变成 256 层 (channel 维度变成 256)
        # 但是保持原大小，此时特征图生成完毕
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 的作用是 xx
        # [N, 16, 16, 256]
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # P6 只用于 rpn 推荐 anchor，并不用于之后的分类
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == 'training':
            # 生成位置大小相对于原图的 anchors
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # 上面生成的 anchors 是针对一张原图的，但是可能是多张图同时训练
            # 所以需要为每张图都生成 anchors, 在 batchSize 为 2 的时候, anchors => [2, 261888, 4]
            # 为什么这么做是因为 keras 需要这样
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # 
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model, 用于为 ProposalLayer 层提供寻找推荐框的数据
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = []
        # 五个 feature_map 全部扔进 rnp 网络中, 得到模型输出
        # 每个 feature_map 会输出三个矩阵, 分别是 "rpn_class_logits", "rpn_class", "rpn_bbox"
        # rpn_class_logits => [N, V, 2]  V 是某个 feature_map 中所有 anchors 的数量
        # rpn_class => [N, V, 2]    rpn_bbox => [N, V, 4]
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        # 每个 feature_map 经过 rpn 网络后, 都会输出 "rpn_class_logits", "rpn_class", "rpn_bbox"
        # 但是我们希望将比如五个 rpn_class_logits 放到一起, 就像 [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [
            # 此时 o 是一个列表, 以 rpn_class 为例, 该列表包含了五个 feature_map 的 rpn_class
            # 那么将它们在 V 这个维度拼接起来, 合并为一个, 其他的同理
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs, output_names)
        ]

        # rpn_class, rpn_bbox 是要进入 ProposalLayer 层的数据并且会进入损失函数, rpn_class_logits 只用于损失函数
        # rpn_class_logits => [N, AV, 2]  AV 是五个 feature_map 所有 anchors 的数量(216888)
        # rpn_class => [N, AV, 2]
        # rpn_bbox => [N, AV, 4]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # 找推荐框时 non-maximum suppression之后保留多少 ROIs
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        # 获得初步的推荐框
        # rpn_rois => [N, proposal_count, 4]
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config
        )([rpn_class, rpn_bbox, anchors])

        if mode == 'training':
            # 从 data-generator 生成的 meta 里提取需要的信息
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # 这个会有人用吗, 想卡死?
                pass
            else:
                target_rois = rpn_rois

            # 用于提供 rois 即进入 FPN 网络的训练数据, 包括预测的正负样本
            # 预测的正样本与最接近的 truth-box 的边框回归值 (可能会奇怪为社么是200个
            # 因为补 0 了, 负样本对应的全是 0). 还有正样本对应的类别以及正样本对应的 mask
            # rois => [N, 200, 4]
            # target_class_ids => [N, 200]
            # target_bbox => [N, 200, 4]
            # target_mask => [N, 200, 28, 28]
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                config, name="proposal_targets"
            )([
                target_rois, 
                input_gt_class_ids, 
                gt_boxes, 
                input_gt_masks
            ])

            # predict

            # 获得针对 rois 的分类与边框回归的预测值
            # mrcnn_class_logits => [N, 200, num_classes]
            # mrcnn_class => [N, 200, num_classes]
            # mrcnn_bbox => [N, 200, num_classes, 4]
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rois, mrcnn_feature_maps, input_image_meta,
                config.POOL_SIZE, config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE
            ) 

            # 获得针对 rois 的 mask 预测值
            # [N, 200, 28, 28, 81(类别数)]
            mrcnn_mask = build_fpn_mask_graph(
                rois, mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN
            )

            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # 构建损失函数

            # 计算 rpn 的前景背景分类损失
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            # 计算 rpn 的边框回归损失
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])
            # 计算 rois 分类损失
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")([target_class_ids, mrcnn_class_logits, active_class_ids])
            # 计算 rois 的边框回归损失
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")([target_bbox, target_class_ids, mrcnn_bbox])
            # 计算 mask 的损失
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")([target_mask, target_class_ids, mrcnn_mask])

            # 这些都是真值, 需要外界输入的
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]

            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)

            outputs = [
                rpn_class_logits, rpn_class, rpn_bbox,
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                rpn_rois, output_rois,
                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss
            ]
            # 生成网络
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # 在 test 的情况下, 生成网络

            # 在 test 情况下, 先生成 ProposalLayer 初步生成的 rois 的分类与边框回归的预测值
            # mrcnn_class_logits => [N, proposal_count, num_classes]
            # mrcnn_class => [N, proposal_count, num_classes]
            # mrcnn_bbox => [N, proposal_count, num_classes, 4]
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rpn_rois, mrcnn_feature_maps, input_image_meta,
                config.POOL_SIZE, config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE
            )
            
            # [N, num_detections, (y1, x1, y2, x2, class_id, class_score)]
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
            )

            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(
                detection_boxes, mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN
            )

            model = KM.Model(
                [input_image, input_image_meta, input_anchors],
                [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                name='mask_rcnn'
            )
        
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    
    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        # 关于 os.walk 的返回, 看下面的连接
        # https://blog.csdn.net/qq_33733970/article/details/77585297
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir)
            )
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

            

    def load_weights(self, filepath, by_name=False, exclude=None):
        import h5py
        
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM
        )
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.)
            )
            # 使用 add_loss 的原因是它更灵活, 而不必被限制于 model.fit 中传入的 Y
            # https://stackoverflow.com/questions/50063613/add-loss-function-in-keras
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs)
        )

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.)
            )
            self.keras_model.metrics_tensors.append(loss)

    
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4
                )
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))


    # 初始化保存模型的路径，并且如果指定了 model_path，那么尝试从文件名中还原 epoch 步数
    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()
        
        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(
            self.model_dir, 
            "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now)
        )

        self.checkpoint_path = os.path.join(
            self.log_dir, 
            "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower())
        )
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    
    def train(
        self, train_dataset, val_dataset, learning_rate, epochs, layers, 
        augmentation=None, custom_callbacks=None, no_augmentation_sources=None
    ):
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }

        # 从传入的配置找到可训练的层
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # 生成输出训练数据的生成器
        train_generator = data_generator(
            train_dataset, self.config, shuffle=True,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources
        )
        val_generator = data_generator(
            val_dataset, self.config, shuffle=True,
            batch_size=self.config.BATCH_SIZE
        )

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=0, write_graph=True, write_images=False
            ),
            keras.callbacks.ModelCheckpoint(
                self.checkpoint_path,
                verbose=0, save_weights_only=True
            ),
        ]

        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))

        # 设置可训练的层
        self.set_trainable(layers)
        # 编译模型
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    
    def mold_inputs(self, images):
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE
            )
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    
    def unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        zero_ix = np.where(detections[:, 4] == 0)[0]
        # 非被填充框是从哪开始
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        # 找到预测的对应类别的掩膜
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # 作者注释写的很明白
        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing 
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy2, wx2])
        # 窗口的大小，即图像经过缩放后的大小
        wh = wy2 - wy1
        ww = wx2 - wx1
        scale = np.array([wh, ww, wh, ww])
        # box 相对于 window 的位置大小
        boxes = np.divide(boxes - shift, scale)
        # 将 box 还原为相对于原图的真实大小位置，因为 window 是只经过缩放的
        # 所以直接按照原图 shape 来还原就可以了
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])
        
        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0
        )[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        full_masks = []
        for i in range(N):
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks   


    def detect(self, images, verbose=0):
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # 查一查相关函数就知道这三个参数的用处了
        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # 生成 anchor 操作与训练阶段相同
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([ mold_image, image_metas, anchors ])

        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(
                    detections[i], mrcnn_mask[i],
                    image.shape, molded_images[i].shape,
                    windows[i]
                )
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results



    
    # 作用是生成 5 个特征图的所有坐标归一化后的 anchors
    def get_anchors(self, image_shape):
        # 计算 6 个 feature_map 的大小
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            # 存储着每种原图大小(1024, 1024)对应的所有 anchors
            self._anchor_cache = {}
        # 生成一次即可，会缓存起来
        if not tuple(image_shape) in self._anchor_cache:
            # 生成五个特征图的所有 anchors，大小位置相对于原图
            # 对于 1024 * 1024 的原图, a => [261888 , 4]
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE
            )
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        
        return self._anchor_cache[tuple(image_shape)]



############################################################
#  Data Formatting
############################################################

# 将某个图片的现在的 id，原始的大小[height, width, 3]，进入训练时的大小(比如 1024 * 1024)
# 窗口信息(看 utils 模块的 resize_image 方法)，scale 即图像缩放因子
# active_class_ids 即图片隶属数据集中所有的 class，是一个数组，数量为 class 数量, 里面所有的数为 1
def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    meta = np.array(
        [image_id] +
        list(original_image_shape) +
        list(image_shape) +
        list(window) +
        [scale] +
        list(active_class_ids)
    )
    
    return meta


# 从 data-generator 生成的 meta 里提取需要的信息
def parse_image_meta_graph(meta):
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    # (y1, x1, y2, x2) window of image in in pixels
    window = meta[:, 7:11]
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]

    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


# 图片像素减去了个平均值
def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL


############################################################
#  Miscellenous Graph Functions
############################################################

# 作用是选出 boxes 中非零的 box, 并顺便也返回非零 box 的序号
# boxes => [num, 4]
def trim_zeros_graph(boxes, name='trim_zeros'):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


# 作用是将 box 的坐标归一化, 虽然与 util 模块重复了
# 但是这里的是用 tf 的方法, 所以输出的是 tensor, 它是在网络中的
# boxes => [N, num, 4]
# shape => 原图大小 [1024, 1024]
def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)