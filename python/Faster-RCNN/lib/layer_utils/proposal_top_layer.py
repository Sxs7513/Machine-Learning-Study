from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes

# 在测试模式下，选取进入 fast-RCNN 的 anchors
def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
    rpn_top_n = cfg.FLAGS.rpn_top_n
    im_info = im_info[0]

    scores = rpn_cls_prob[:, :, :, num_anchors:]
    
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))

    length = scores.shape[0]
    if length < rpn_top_n:
        top_inds = npr.choice(length, size=rpn_top_n, replace=True)
    else:
        top_inds = scores.argsort()[::-1]
        top_inds = top_inds[:rpn_top_n]
        top_inds = top_inds.reshape(rpn_top_n, )

    anchors = anchors[top_inds, :]
    rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
    scores = scores[top_inds]

    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

    # Clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob, scores
