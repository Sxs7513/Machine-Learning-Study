from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from lib.utils.timer import Timer
# from utils.cython_nms import nms, nms_new
from lib.utils.py_cpu_nms import py_cpu_nms as nms
from lib.utils.blob import im_list_to_blob

# from model.config import cfg, get_output_dir
from lib.config.config import get_output_dir
from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform_inv

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.FLAGS2["pixel_means"]

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.FLAGS2["test_scales"]:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.FLAGS.test_max_size:
            im_scale = float(cfg.FLAGS.test_max_size) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs["data"]
    blobs["im_info"] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    # rois 为全部进入 fast-rcnn 层的 anchor， scores 即为分类预测
    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    # 还原回原始的体积
    boxes = rois[:, 1:5] / im_scales[0]
    # 打平，注意哦，这里是把 bachsize 打平了哈哈
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

    if cfg.FLAGS.test_bbox_reg:
        # 修正 anchor
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes