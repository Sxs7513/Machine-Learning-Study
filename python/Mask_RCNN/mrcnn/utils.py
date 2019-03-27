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

    
    def prepare(self, class_map=None):
        return