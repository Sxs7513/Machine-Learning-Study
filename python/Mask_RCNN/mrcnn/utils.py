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



############################################################
#  Anchors
############################################################

# 具体的为每一特征层生成 anchors, 与 Faster-RCNN 有所不同，Faster-RCNN 生成的 anchor 是相对于特征图的(需要指出其实 Faster-RCNN 没有特征图)
# 而该函数提取的 anchors 位置与大小直接是相对于原图的
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    # 这里不是生成网格，只是为了计算方便
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # 计算每个 ratio 对应的宽高
    # 比如 [[45.254834], [32], [22.627417]]
    heights = scales / np.sqrt(ratios) 
    # 比如 [[22.627417], [32], [45.254834]]
    widths = scales * np.sqrt(ratios)

    # anchor 中心点映射回原图的位置，生成网格
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # 生成所有 anchors 对应的宽高，中心位置，注意都是相对于原图的
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack([box_centers_x, box_centers_y], axis=-1).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=-1).reshape([-1, 2])

    # 变为 [y1, x1, y2, x2]
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=-1)
    return boxes



# scales => 提取的 anchor 的边长 (32, 64, 128, 256, 512)
# ratios => 每个 cell 提取的三个 anchor 的宽高比 [0.5, 1, 2]
# feature_shapes => 6 个特征图的大小 [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16]] 
# feature_strides => 经过 reset 网络后提取的 5 个 feature_map 相比原图缩小的比例 [4, 8, 16, 32, 64]
# anchor_stride => 每隔几个 cell 创建 anchors，默认为 1
def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    anchors = []
    for i in range(len(scales)):
        anchors.append(
            generate_anchors(
                scales[i], ratios, 
                feature_shapes[i],
                feature_strides[i], 
                anchor_stride
            )
        )

    # 5 个特征图的所有 anchor 合并到一起
    # [(256*256 + 128*128 + 64*64 + 32*32 + 16*16)*3, 4] = [261888, 4]
    return np.concatenate(anchors, axis=0)



############################################################
#  Miscellaneous
############################################################

# 作用是将 acnhor 的坐标归一化
# boxes => anchors
# shape => 原图大小 [1024, 1024]
def norm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


# 有些运算仅支持单 batch, 比如 tf.gather 及作者自己写的 apply_box_deltas_graph，clip_boxes_graph 等
# 所以作者用了一个 hack 来克服这个问题
def batch_slice(inputs, graph_fn, batch_size, names=None):
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        # 取出每个 inputs 对应的 batch 的数据
        inputs_slice = [x[i] for x in inputs]
        # 应用传入的方法来进行切片
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    # 与 model.py 中 rpn 类似, 首先把英文原文解释 copy 过来
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    # 用中文说呢, 假设 inputs 为 [a, b], 那么 outputs 为 [[a第一个batch经过切片, a第二个batch经过切片], ....]
    # 用下面方法还原到 [a经过切片, b经过切片]
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    # 这个主要是为了输出 tensor, 因为上面的 outputs 是一个普通的 list
    # 对数据格式不会做任何修改
    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    # 如果 input 只有一个, 那么不用把它框起来了
    if len(result) == 1:
        result = result[0]

    return result


if __name__ == "__main__":
    # print(generate_anchors(
    #     32, 
    #     [0.5, 1, 2],
    #     [256, 256],
    #     4,
    #     1
    # ))
    boxes = generate_pyramid_anchors(
        (32, 64, 128, 256, 512),
        [0.5, 1, 2],
        [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16]],
        [4, 8, 16, 32, 64],
        1
    )
    boxes = norm_boxes(boxes, [1024, 1024])
    print(
        np.broadcast_to(boxes, (2,) + boxes.shape).shape
    )