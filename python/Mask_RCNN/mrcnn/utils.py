import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
# from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Dataset
############################################################
# dataset 的基类
class Dataset(object):
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # 背景永远是第一个类别
        self.class_info = [{ "source": "", "id": 0, "name": "BG" }]
        self.source_class_ids = {}

    
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "source name cannot contain a dot"

        for info in self.class_info:
            # 不重复添加, 类别具有唯一性
            if info["source"] == source and info["id"] == class_id:
                return

        # 如果没有添加过该类别, 那么添加类别 id 与 名称
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })
    

    # 将某张图片的相关信息存储起来
    def add_image(self, source, image_id, path, **kwargs):
        # 基本信息
        image_info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        # 图片宽高大小, annotations 等
        image_info.update(kwargs)
        self.image_info.append(image_info)

    # 洗数据
    def prepare(self, class_map=None):
        """
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        # pycocotools 给的类别 id 不是连续的, 所以有 90 个
        # 真实其实只有 80 个，这里给重新赋予 id
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        # image 也重新赋予 id
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # key => source.origin_class_id 
        # value => new_class_id
        self.class_from_source_map = {
            "{}.{}".format(info['source'], info['id']): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        # key => source.origin_image_id
        # value => new_image_id
        self.image_from_source_map = {
            "{}.{}".format(info['source'], info['id']): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        self.sources = list(set(i["source"] for i in self.class_info))
        # key => source
        # value => new_class_id
        self.source_class_ids = {}

        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                # 包括背景
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

        