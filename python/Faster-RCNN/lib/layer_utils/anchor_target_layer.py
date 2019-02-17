from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    A = num_anchors
    # 所有 anchors 数量
    total_anchors = all_anchors.shape[0]
    # 统计一下在多少个中心提取出来这么多 anchors
    K = total_anchors / num_anchors
    # 每次只处理一张图片
    im_info = im_info[0]

    # 允许框紧贴图像边缘
    _allowed_border = 0

    # [H, W]
    height, width = rpn_cls_score.shape[1:3]

    # 过滤掉不在图像范围内的Boxes, 首先用where函数加条件筛选出索引
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    anchors = all_anchors[inds_inside, :]

    # label加一个维度，进行正、负、非正非负样本标注
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # 计算 anchors 和 gt boxes 的重合率
    # bbox_overlaps 是一个现成函数：from utils.cython_bbox import bbox_overlaps（）
    # np.ascontiguousarray返回一个指定数据类型的连续数组，转存为顺序结构的数据
    # 可以看 util 中的 bbox.pyx
    # overlaps 为 len(anchors) * inds_inside 的矩阵，即一行为某个 anchor 与所有 gt-box 的重叠率
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    # 每个 anchor 最接近哪个 gt-box
    argmax_overlaps = overlaps.argmax(axis=1)
    # 提取每一行最大重叠率
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # 获得与某个 gtbox 重合率最大的 anchor 的列索引，即对应 overlaps 的行索引
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    # 提取每一列最大重叠率
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    # 获得与每个 gt-box 重叠率最大的 anchor 在 overlaps 中第一维的坐标 
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # 打上前景标签：对于每一个gt框，重叠率最大的检测框不论阈值多少都算foreground.
    labels[gt_argmax_overlaps] = 1

    # 打上前景标签2：满足重叠率的检测结果打上foreground标签
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1

    # 打上背景标签
    if cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # 过滤机制：如果正样本过多，再采样一次，就是正负样本平衡
    # cfg.TRAIN.RPN_FG_FRACTION=正样本在每次训练rpn的比例，默认是0.5X256
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # 对负样本也进行同样操作，但在数量限制上，首先保证TRAIN.RPN_BATCHSIZE，然后减去正样本数量。
    # 这样其实不能平衡正负样本。
    # 同时指出 rpn_batchsize 这个参数，指的是训练时每次计算loss的batch数量，不能设置太小，不然随机掉很多好样本没得训练
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 计算 bounding-regression, 注意是 anchor 与其最接近的 gt-box 才进行计算
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS2["bbox_inside_weights"])

    # rpn-bbox-loss 里面的 Nreg
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.FLAGS.rpn_positive_weight < 0:
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # unmap 的作用就是映射回原来的 total_anchor 样子
    # 因为 all_anchors 裁减掉了2/3左右，仅仅保留在图像内的anchor。这里就是将其复原作为下一层的输入了，并reshape成相应的格式
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # 将 labels reshape 为类似于 rpn_cls_score_reshape 的形式，可以发现 labels 的 C 维在
    # 第二维，这个没有关系，在训练的时候会把它打平成一维，一样的效果
    # 注意是 A * height，因为这个是前景分类标签
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets 要恢复成原始的 shape
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
    rpn_bbox_targets = bbox_targets

    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _compute_targets(ex_rois, gt_rois):

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def _unmap(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret