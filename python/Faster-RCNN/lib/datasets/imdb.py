from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import PIL
import numpy as np
import scipy.sparse
from lib.config import config as cfg

class imdb(object):
    # 定义通用的图像数据库类

    def __init__(self, name, classes=None):
        self.name = name
        self._num_classes = 0
        if not classes: 
            self._classes = []
        else:
            self._classes = classes
        self._image_index = []
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def image_index(self):
        return self._image_index

    @property
    def num_images(self):
        return len(self.image_index)

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    # 与 set_proposal_method 配合，获取 roidb 数据集
    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.FLAGS2["data_dir"], "cache"))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path
    
    def default_roidb(self):
        raise NotImplementedError

    # 获得图片宽度
    def _get_widths(self):
        return [
            PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)
        ]

    # 将原图翻转来增强数据
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]["boxes"].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            entry = {
                'boxes': boxes,
                'gt_overlaps': self.roidb[i]['gt_overlaps'],
                'gt_classes': self.roidb[i]['gt_classes'],
                'flipped': True
            }
            self.roidb.append(entry)
        self._image_index = self._image_index * 2