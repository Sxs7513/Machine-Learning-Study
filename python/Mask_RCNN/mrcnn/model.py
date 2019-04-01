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

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    return


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
# 有这样的区别是为了保证旁路出来的featuremap和主路的featuremap尺寸一致，这样它们才能相加
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

    
    # inputs => rpn_class, rpn_bbox, anchors
    # rpn_class => [N, AV, 2]  AV 是五个 feature_map 所有 anchors 的数量
    # rpn_bbox => [N, AV, 4]
    # anchors => 在图片大小为 1024 * 1024 的情况下为 [N, 261888, 4]
    def call(self, inputs):
        # 取出前景得分 [N, AV, 1]
        scores = inputs[0][:, :, 1]
        # 取出 bbox 预测, 并
        deltas = inputs[1]
        deltas = deltas * np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # 取出 anchors
        anchors = inputs[2]

        # 保险起见, 一般不会发生, 总不至于 6000 个 anhcor 都提取不出来吧?
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # 找到分最高的 anchor 的 index
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        # tf.gather 不支持 batch, 所以 hack 下, 选择出来对应的得分
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

        # 保证 boxes 区域都在原图内
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            [boxes, window],
            lambda x, y: clip_boxes_graph(x, y),
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
#  Detection Target Layer
############################################################

# 
class DetectionTargetLayer(KE.Layer):
    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    
    def call(self, input):
        # [N, proposal_count, 4]
        proposals = input[0]
        # [N]
        gt_class_ids = inputs[1]
        # [N, 4]
        gt_boxes = inputs[2]
        # 
        gt_masks = inputs[3]


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
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [t.shape[0], -1, 2]))(x)
    # [N, V, 2] 
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # [N, height, width, 4 * anchors_per_location]
    x = KL.Conv2D(anchors_per_location * 4, kernel_size=(1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')
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
    # window 如果给出了max_dim, 可能会对返回图像进行填充如果是这样的，则窗口是全图的部分图像坐标 (不包括填充的部分)
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
    # 这一步真的是没看懂...有什么必要吗???
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # 把图片的一些信息缓存下来，具体看该方法
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


# 
# image => 经过缩放后的图像，anchors => 该图像提取的所有 anchor
# gt_class_ids => 图片中所有的 truth-boxes 类别
# gt_boxes => 图片中所有的 truth-boxes 坐标
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
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

    # 生成五个特征图的所有 anchors，大小位置相对于原图
    # 对于 1024 * 1024 的原图, a => [261888 , 4]
    anchors = utils.generate_pyramid_anchors(
        self.config.RPN_ANCHOR_SCALES,
        self.config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        self.config.BACKBONE_STRIDES,
        self.config.RPN_ANCHOR_STRIDE
    )

    while True:
        try:
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
                # image 为缩放后的图片
                # image_meta 为图片缩放的信息，与数据集所有的类别
                # gt_class_ids 为图片中 truth-boxes 的类别
                # gt_boxes 为图片中的 truth-boxes 坐标
                # gt_masks 为掩膜
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
                    dataset, config, image_id, augment=augment,
                    augmentation=augmentation,
                    use_mini_mask=config.USE_MINI_MASK
                )

            if not np.any(gt_class_ids > 0):
                continue

            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)

        except:
            return


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

    
    def build(self, mode, config):
        assert mode in ['training', 'inference']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )
        
        # Inputs
        input_image = KL.input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        if mode == "training":
            # RPN GT
            
            # 
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # 
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)

            # 
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)

        elif mode == "inference":
            pass

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

        # 要进入 ProposalLayer 层的数据 
        # rpn_class_logits => [N, AV, 2]  AV 是五个 feature_map 所有 anchors 的数量 
        # rpn_class => [N, AV, 2]
        # rpn_bbox => [N, AV, 4]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # 找推荐框时 non-maximum suppression之后保留多少 ROIs
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        # 获得初步的推荐框
        # rpn_rois => [N, proposal_count, 4]
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors]
        )([rpn_class, rpn_bbox, anchors])

        if mode == 'training':
            # 
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # 这个会有人用吗, 想卡死?
                pass
            else:
                target_rois = rpn_rois

        # 
        rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
            config, name="proposal_targets"
        )([
            target_rois, 
            input_gt_class_ids, 
            gt_boxes, 
            input_gt_masks
        ])



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

        # 
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

    
    # 作用是生成 5 个特征图的所有坐标归一化后的 anchors
    def get_anchors(self, image_shape):
        # 计算 6 个 feature_map 的大小
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            # 存储着每种原图大小(1024, 1024)对应的所有 anchors
            self._anchor_cache = {}
        # 生成一次即可，会缓存起来
        if not tuple(config.IMAGE_SHAPE) in self._anchor_cache:
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

# 将某个图片的现在的 id，原始的大小，进入训练时的大小(比如 1024 * 1024)
# 窗口信息(看 utils 模块的 resize_image 方法)，scale 即图像缩放因子
# active_class_ids 即图片隶属数据集中所有的 class，是一个数组，数量为 class 数量
# 里面所有的数为 1
def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    meta = np.array(
        [image_id],
        list(original_image_shape),
        list(image_shape),
        list(window),
        [scale],
        list(active_class_ids)
    )
    
    return meta


# 从 meta 中提取每一张图片的原始信息, 具体都啥意思回头来写
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



############################################################
#  Miscellenous Graph Functions
############################################################

# 作用是将 anchors 的坐标归一化, 虽然与 util 模块重复了
# 但是这里的是用 tf 的方法, 所以输出的是 tensor, 它是在网络中的
# boxes => anchors
# shape => 原图大小 [1024, 1024]
def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), axis=-1)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)