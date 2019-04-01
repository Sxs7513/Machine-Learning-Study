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
#  Bounding Boxes
############################################################

# 从掩膜数据中, 直接提取出来 truth-box 的位置大小
# mask => [height, width, 该图片中 num_mask]
def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # 这个是找出 box 的边界, 因为掩膜是以像素为单位的, 所以用 any 来找
        # 只要水平方向的有任何一个点，该行就是 true, 然后用 where 来定位所有有
        # 掩膜的行, 注意 [0] 不是为了取出上边界，而只是 where 会用 [] 套一层而已
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        # 垂直的同理
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            # 找到边界, [[0, -1]] 代表找到 0 -1 然后套一层 []
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


# 用于计算 boxes-iou 的，这个不多做解释
def compute_overlaps(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


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
        # image 也重新赋予 id, 不用 coco 原有的那个长长的
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


    def map_source_class_id(self, source_class_id):
        return self.class_from_source_map[source_class_id]

    
    @property
    def image_ids(self):
        return self._image_ids


    # 加载图片
    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]["path"])
        # 保证是原始图像是三维的
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # 保证原图图像通道数为 3
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    
    # 该方法需要被重写, 否则会返回空的
    def load_mask(self, image_id):
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


# 缩放图像同时保持宽高比不变，并且会返回 resize 的一些信息
# min_dim: 如果给出了该值，缩放图像时要保持短边 == min_dim
# max_dim: 如果给出了该值，缩放图像时要保持长边不超过它.
# min_scale: 如果给出了该值，则使用它来缩放图像，而不管是否满足min_dim.
# mode: 缩放模式.
#     none:   无缩放或填充. 返回原图.
#     square: 缩放或填充0，返回[max_dim, max_dim]大小的图像.
#     pad64:  宽和高填充0，使他们成为64的倍数.
#             如果IMAGE_MIN_DIM 或 IMAGE_MIN_SCALE不为None, 则在填充之前先
#             缩放. IMAGE_MAX_DIM在该模式中被忽略.
#             要求为64的倍数是因为在对FPN金字塔的6个levels进行上/下采样时保证平滑(2**6=64).
#     crop:   对图像进行随机裁剪. 首先, 基于IMAGE_MIN_DIM和IMAGE_MIN_SCALE
#             对图像进行缩放, 然后随机裁剪IMAGE_MIN_DIM x IMAGE_MIN_DIM大小. 
#             仅在训练时使用.
#             IMAGE_MAX_DIM在该模式中未使用.

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # 下面这两段代码是设置
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # 如果要求图片是正方形，那么采用填充的方式，而不是直接 resize
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        # 代表的意思是在整个 (max_dim, max_dim) 的尺寸下，图像从什么位置开始真正存在，因为其他部分都是 0 填充
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    # 剪切的情况
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        # 代表的意思是图像的大小，这里并不像 square 还能有位置信息
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))

    # image: 缩放后的图像
    # window: (y1, x1, y2, x2). 如果给出了max_dim, 可能会对返回图像进行填充
    # 如果是这样的，则窗口是全图的部分图像坐标 (不包括填充的部分)
    # scale: 图像缩放因子
    # padding: 图像填充部分[(top, bottom), (left, right), (0, 0)]
    return image.astype(image_dtype), window, scale, padding, crop


# 因为掩膜也是一张图片, 所以在 resize 原图的时候, 也要顺带着处理下掩膜
# 用指定的 scale 和 padding 缩放 mask。一般来说, 为了保持图像和 mask 的一致性，scale 和 padding 是通过 resize_image() 获取的
# scale: mask缩放因子
# padding: 填充[(top, bottom), (left, right), (0, 0)]
def resize_mask(mask, scale, padding, crop=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        # 填充
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


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


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


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