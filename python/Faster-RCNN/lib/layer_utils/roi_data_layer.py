from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

import sys
sys.path.append("../..")
from lib.config import config as cfg
from lib.utils.minibatch import get_minibatch

# Fast R-CNN data layer used for training.
class RoIDataLayer(object):
    def __init__(self, roidb, num_classes, random=False):
        self._roidb = roidb
        self._num_classes = num_classes
        # Also set a random flag
        self._random = random
        self._shuffle_roidb_inds()

    # Randomly permute the training roidb.
    def _shuffle_roidb_inds(self):
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        if self._cur + cfg.FLAGS.ims_per_batch >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur+cfg.FLAGS.ims_per_batch]
        self._cur += cfg.FLAGS.ims_per_batch

        return db_inds

    def _get_next_minibatch(self):
        # 获取这次要处理的图片的 index
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        # 获得 gt_boxes 即图片里的目标信息，它 shape 为 (x1, y1, x2, y2, cls)， 其中坐标为经过缩放后的位置
        # im_info 为缩放后的图片 shape 与缩放信
        return get_minibatch(minibatch_db, self._num_classes)

    # Get blobs and copy them into this layer's top blob vector
    def forward(self):
        blobs = self._get_next_minibatch()
        return blobs