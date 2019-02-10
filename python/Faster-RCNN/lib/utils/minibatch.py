"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import numpy.random as npr

import sys
sys.path.append("../..")
from lib.config import config as cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob

# Given a roidb, construct a minibatch sampled from it
def get_minibatch(roidb, num_classes):
    num_images = len(roidb)
    random_scale_inds = npr.randint(0, high=len(cfg.FLAGS2["scales"]), size=num_images)
    assert (cfg.FLAGS.batch_size % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.FLAGS.batch_size)
    
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {"data": im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.FLAGS.use_all_gt:
        gt_inds = np.where(roidb[0]["gt_classes"] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]["gt_classes"][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32
    )

    return blobs

    return 

# Builds an input blob from the images in the roidb at the specified scales
def _get_image_blob(roidb, scale_inds):
    num_images = len(roidb)
    processed_ims = []
    im_scales = []

    for i in range(num_images):
        im = cv2.imread(roidb[i]["image"])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.FLAGS2["scales"][scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.FLAGS2["pixel_means"], target_size, cfg.FLAGS.max_size)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales