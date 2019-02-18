from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

import sys
sys.path.append("../..")

from lib.config import config as cfg
from lib.datasets.imdb import imdb

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        # 要加载的 imageSet，即该图片对应的编号
        self._image_index = self._load_image_set_index()

        # PASCAL specific config options
        self.config = {
            'cleanup': True,
            'use_salt': True,
            'use_diff': False,
            'matlab_eval': False,
            'rpn_file': None
        }

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    # Return the absolute path to image i in the image sequence.
    # 获得对应编号图片的路径
    def image_path_at(self, i):
        return self.image_path_from_index(self.image_index[i])

    # Construct an image path from the image's "index" identifier.
    def image_path_from_index(self, index):
        image_path = os.path.join(
            self._data_path,
            "JPEGImages",
            index + self._image_ext
        )
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    # Return the default path where PASCAL VOC is expected to be installed.
    def _get_default_path(self):
        return os.path.join(cfg.FLAGS2["data_dir"], "VOCdevkit" + self._year)

    # Load the indexes listed in this dataset's image set file.
    def _load_image_set_index(self):
        image_set_file = os.path.join(
            self._data_path,
            "ImageSets",
            "Main",
            self._image_set + ".txt"
        )
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    # Return the database of ground-truth regions of interest.
    # This function loads/saves from/to a cache file to speed up future calls.
    # 加载所有图片的 xml 信息，每个图片对应一个对象
    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding="bytes")
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        
        gt_roidb = [
            self._load_pascal_annotation(index)
            for index in self._image_index
        ]
        with open(cache_file, "wb") as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

        # self._load_pascal_annotation(self._image_index[0])

    # Load image and bounding boxes info from XML file in the PASCAL VOC format.
    # 加载某一个编号的图片 xml 信息，包括图片中目标位置信息，目标类别，目标长宽
    def _load_pascal_annotation(self, index):
        filename = os.path.join(
            self._data_path,
            "Annotations",
            index + '.xml'
        )
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config["use_diff"]:
            # Exclude the samples labeled as difficult
            # 去除面积过小样本
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0
            ]
            objs = non_diff_objs
        num_objs = len(objs)

        # 已标出的检测目标的位置与类别
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # 供 softmax 使用的 one-hot 向量
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        # box的长宽
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find("bndbox")
            # Make pixel indexes 0-based
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find("name").text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # 稀疏矩阵压缩，缩小保存体积
        overlaps = scipy.sparse.csr_matrix(overlaps)
        
        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas
        }
            

if __name__ == "__main__":
    d = pascal_voc('trainval', '2007')
    d.gt_roidb()