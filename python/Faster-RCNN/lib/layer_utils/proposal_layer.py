from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sys 
sys.path.append('../..')
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):

    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    # 每次只会处理一张图片，取出原始图的 shape 信息，第一维是图片数量
    im_info = im_info[0]
    # 取出RPN预测的框属于前景的分数，在18个channel中，前9个是框属于背景的概率，后9个才是属于前景的概率
    # 注意这个是人为定的哦，并没有绝对的哪个是哪个
    scores = rpn_cls_prob[:,:,:, num_anchors:]
    # 回归预测打平，为了之后与 前景score 一一对应
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))

    # 根据 rpn_bbox_pred 的回归修正 anchor
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    # 修剪 anchor
    proposals = clip_boxes(proposals, im_info[:2])

    # 将 scores 打成一维然后获得从小到大的索引值，最后用 [::-1] 颠倒成从大到小
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        # 对所有的框按照前景分数进行排序，选择排序后的前pre_nms_topN
        order = order[:pre_nms_topN]
    # 注意 scores 和 rpn_bbox_pred 是一一对应的，即一个 anchor 对应一个分
    # 这里选取了所有高分对应的 anchor
    proposals = proposals[order, :]
    scores = scores[order]

    # 非极大值抑制
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    # 因为要进行roi_pooling，在保留框的坐标信息前面插入batch中图片的编号信息。此时，由于batch_size为1，因此都插入0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores