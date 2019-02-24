from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
# from lib.utils.cython_bbox import bbox_overlaps
from cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform

# 因为之前的 anchor 位置已经修正过了，所以这里又计算了一次经过 proposal_layer 修正后的的 box 与 GT 的 IOU 来得到 label
# 但是阈值不一样了，变成了大于等于0.5为1，小于为0，并且这里得到的正样本很少，通常只有2-20个，甚至有0个
# 并且正样本最多为64个，负样本则有比较多个，相应的也重新计算了一次bbox_targets

# 产生筛选后的 roi，对应labels，三个(len(rois), 4*21)大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh）
# 另两个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
    all_rois = rpn_rois
    all_scores = rpn_scores

    if cfg.FLAGS.proposal_use_gt:
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = np.vstack((all_scores, zeros))
    
    num_images = 1
    rois_per_image = cfg.FLAGS.batch_size / num_images
    # 正面样本要求的个数， 256 * 0.25
    fg_rois_per_image = np.round(cfg.FLAGS.proposal_fg_fraction * rois_per_image)

    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes
    )

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    # bbox_targets的列是类别数 * 4哦
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

# 从 all_rois 再次选取，与 anchor-target_layer 一样
def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float)
    )
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # roi 对应的真正分类，并不是前景背景那个分类
    labels = gt_boxes[gt_assignment, 4]

    fg_inds = np.where(max_overlaps >= cfg.FLAGS.roi_fg_threshold)[0]
    bg_inds = np.where((max_overlaps < cfg.FLAGS.roi_bg_threshold_high) &
                       (max_overlaps >= cfg.FLAGS.roi_bg_threshold_low))[0]

    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        # import pdb
        # pdb.set_trace()
        print("pdb.set_trace()")

    keep_inds = np.append(fg_inds, bg_inds)
    labels = labels[keep_inds]
    labels[int(fg_rois_per_image):] = 0

    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]

    # 计算回归值，构建一个 label => regression 的矩阵
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels
    )

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    clss = bbox_target_data[:, 0]
    # (len(rois), 4*21)
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        # 对 fg-roi 对应引索行的对应类别的4个位置填上（dx,dy,dw,dh）
        # 注意，允许有 0 存在哦，即 rois 对该类别是 0 regession
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.FLAGS2["bbox_inside_weights"]
    
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.FLAGS.bbox_normalize_targets_precomputed:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.FLAGS2["bbox_normalize_means"]))
                   / np.array(cfg.FLAGS2["bbox_normalize_stds"]))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)